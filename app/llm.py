"""Thin async LLM client.

Design goals:
  * Free-tier friendly: a global rate limiter keeps us under the provider's RPM.
  * Deterministic: temperature 0 + an in-process cache keyed by the exact prompt,
    so the same input always yields the same output within a run.
  * Never blocks the endpoint past its budget: callers pass a timeout and get
    None back on any failure, then fall back to the rule-based composer.

Providers: "gemini" (default, free tier), "groq" (free tier, OpenAI-compatible), "openai" (any OpenAI-compatible URL, e.g.
Groq/OpenRouter), "anthropic", or "none" (pure rule-based mode).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time

import httpx

log = logging.getLogger("vera.llm")


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from a .env file next to the app (no extra dependency).
    Real environment variables always win."""
    from pathlib import Path
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


_load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
API_KEY = (os.getenv("LLM_API_KEY", "") or os.getenv("GROQ_API_KEY", "") or os.getenv("GEMINI_API_KEY", ""))
MODEL = os.getenv("LLM_MODEL", "") or {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}.get(PROVIDER, "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or (
    "https://api.groq.com/openai/v1" if PROVIDER == "groq" else "https://api.openai.com/v1")
# Groq's free tier is capped at ~8K tokens/minute; one compose is ~2.5K tokens, so ~3 calls/min.
RPM = max(1, int(os.getenv("LLM_RPM", "3" if PROVIDER == "groq" else "9")))
# Groq's free tier also caps tokens/minute (8K); budget below it so we never trigger a 429 cascade.
TPM = int(os.getenv("LLM_TPM", "7000" if PROVIDER == "groq" else "100000000"))
OUTPUT_TOKENS_EST = 700            # JSON body + low-effort reasoning, typical
_cooldown_until = 0.0              # set from the provider's "try again in Xs" on a 429

_cache: dict[str, dict] = {}
_inflight: dict[str, asyncio.Future] = {}
_waiting: list[int] = []          # priorities of callers waiting for a slot
_last_calls: list[list[float]] = []   # [start_time, tokens] per call in the last 60s
_client: httpx.AsyncClient | None = None
_gemini_thinking_ok = True
LAST_ERROR = ""
STATS = {"calls": 0, "ok": 0, "errors": 0, "cache_hits": 0, "rate_limited": 0}


def enabled() -> bool:
    return PROVIDER != "none" and bool(API_KEY)


def model_label() -> str:
    return f"{PROVIDER}:{MODEL}" if enabled() else "rule-based (no LLM key set)"


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=5.0))
    return _client


async def _acquire_slot(deadline: float, priority: int = 0, tokens: int = 0) -> list[float] | None:
    """Sliding-window limiter on requests AND tokens per minute, with priorities: when
    capacity is scarce, the most valuable waiting message gets the next slot. Returns the
    window entry (so real usage can be recorded) or None if we'd wait past the deadline."""
    _waiting.append(priority)
    try:
        while True:
            now = time.monotonic()
            while _last_calls and now - _last_calls[0][0] > 60:
                _last_calls.pop(0)
            used = sum(c[1] for c in _last_calls)
            if (now >= _cooldown_until and len(_last_calls) < RPM and used + tokens <= TPM
                    and priority >= max(_waiting)):
                entry = [now, float(tokens)]
                _last_calls.append(entry)
                return entry
            if now >= deadline:
                STATS["rate_limited"] += 1
                return None
            await asyncio.sleep(0.25)
    finally:
        _waiting.remove(priority)


def _retry_after(text: str) -> float:
    """'Please try again in 1m2.5s' / '12.3s' / '450ms' -> seconds."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", text or "")
    if not m:
        return 20.0
    secs = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    return secs + 60 * int(m.group(1) or 0) + 0.5


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None


async def _call_gemini(system: str, user: str, timeout: float, meta: dict | None = None) -> str:
    global _gemini_thinking_ok
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    gen = {"temperature": 0, "topP": 1, "topK": 1, "maxOutputTokens": 1024,
           "responseMimeType": "application/json", "seed": 7}
    if _gemini_thinking_ok and "2.5" in MODEL:
        gen["thinkingConfig"] = {"thinkingBudget": 0}     # faster, and plenty for this task
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen}
    r = await _http().post(url, params={"key": API_KEY}, json=body, timeout=timeout)
    if r.status_code == 400 and "thinking" in r.text.lower() and "thinkingConfig" in gen:
        _gemini_thinking_ok = False
        gen.pop("thinkingConfig")
        r = await _http().post(url, params={"key": API_KEY}, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


_openai_optional_ok = True


async def _call_openai(system: str, user: str, timeout: float, meta: dict | None = None) -> str:
    """OpenAI-compatible chat completions (OpenAI, Groq, OpenRouter...)."""
    global _openai_optional_ok
    base = {"model": MODEL, "temperature": 0, "seed": 7, "max_tokens": 1000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    optional = {"response_format": {"type": "json_object"}}
    if "gpt-oss" in MODEL:
        optional["reasoning_effort"] = "low"          # reasoning tokens count against Groq's TPM cap
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    body = {**base, **optional} if _openai_optional_ok else base
    r = await _http().post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code == 400 and _openai_optional_ok:   # model rejected an optional param: retry plain
        _openai_optional_ok = False
        r = await _http().post(url, headers=headers, json=base, timeout=timeout)
    if r.status_code == 429:
        global _cooldown_until
        _cooldown_until = max(_cooldown_until, time.monotonic() + _retry_after(r.text))
        raise RuntimeError(f"rate limited: {re.sub(r'org_[A-Za-z0-9]+', 'org', r.text)[:300]}")
    r.raise_for_status()
    data = r.json()
    if meta is not None:
        meta["tokens"] = (data.get("usage") or {}).get("total_tokens")
    return data["choices"][0]["message"].get("content") or ""


async def _call_anthropic(system: str, user: str, timeout: float, meta: dict | None = None) -> str:
    r = await _http().post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
        json={"model": MODEL, "max_tokens": 1024, "temperature": 0, "system": system,
              "messages": [{"role": "user", "content": user}]},
        timeout=timeout)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json().get("content", []))


async def complete_json(system: str, user: str, timeout: float = 12.0, priority: int = 0) -> dict | None:
    """Return parsed JSON from the model, or None on any failure / timeout / rate limit."""
    if not enabled():
        return None
    key = hashlib.sha256(f"{MODEL}\n{system}\n{user}".encode()).hexdigest()
    if key in _cache:
        STATS["cache_hits"] += 1
        return _cache[key]
    if key in _inflight:                                   # identical request already running
        try:
            return await asyncio.wait_for(asyncio.shield(_inflight[key]), timeout)
        except Exception:
            return None

    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    _inflight[key] = fut
    result: dict | None = None
    deadline = time.monotonic() + timeout
    try:
        est = int((len(system) + len(user)) / 3.6) + OUTPUT_TOKENS_EST
        entry = await _acquire_slot(deadline, priority, est)
        if entry is not None:
            remaining = max(2.0, deadline - time.monotonic())
            meta: dict = {}
            STATS["calls"] += 1
            fn = {"gemini": _call_gemini, "groq": _call_openai, "openai": _call_openai,
                  "anthropic": _call_anthropic}.get(PROVIDER)
            if fn:
                text = await asyncio.wait_for(fn(system, user, remaining, meta), remaining)
                if meta.get("tokens"):
                    entry[1] = float(meta["tokens"])      # replace the estimate with real usage
                result = _extract_json(text)
                if result is not None:
                    STATS["ok"] += 1
                    _cache[key] = result
    except Exception as e:  # network, 429, parse, timeout: all degrade to fallback
        global LAST_ERROR
        STATS["errors"] += 1
        detail = str(e)
        if isinstance(e, httpx.HTTPStatusError):
            detail = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        LAST_ERROR = (type(e).__name__ + ": " + detail)[:300]
        log.warning("LLM call failed: %s", LAST_ERROR)
    finally:
        if not fut.done():
            fut.set_result(result)
        _inflight.pop(key, None)
    return result
