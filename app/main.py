"""Vera bot - HTTP surface for the magicpin AI Challenge judge harness.

Endpoints: POST /v1/context, POST /v1/tick, POST /v1/reply, GET /v1/healthz,
GET /v1/metadata (+ optional POST /v1/teardown).

Latency strategy: the moment a trigger (or an updated merchant) is pushed, a
composition job starts in the background. /v1/tick then only has to collect
finished jobs, so it answers well inside the harness's 10-15s budget even when the
free-tier LLM is slow. Anything not ready in time uses the rule-based composer.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import llm
from .composer import PROMPT_VERSION, compose, conversation_id_for
from .facts import parse_iso
from .replies import handle_reply
from .store import SCOPES, STORE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vera")

TICK_BUDGET = float(os.getenv("TICK_BUDGET_SECONDS", "8"))
BG_COMPOSE_SECONDS = float(os.getenv("BG_COMPOSE_SECONDS", "150"))
MAX_ACTIONS = 20
VERSION = "1.0.0"
SUBMITTED_AT = os.getenv("SUBMITTED_AT", "2026-09-27T00:00:00Z")

app = FastAPI(title="Vera bot", version=VERSION)
_jobs: dict[tuple, asyncio.Task] = {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- composition jobs
def _inputs_for(trigger_id: str):
    trg = STORE.get("trigger", trigger_id)
    if not trg:
        return None
    mid = trg.get("merchant_id") or (trg.get("payload") or {}).get("merchant_id")
    merchant = STORE.get("merchant", mid)
    if not merchant:
        return None
    category = STORE.get("category", merchant.get("category_slug"))
    if not category:
        return None
    cid = trg.get("customer_id")
    customer = STORE.get("customer", cid) if cid else None
    key = (trigger_id, STORE.version("trigger", trigger_id), mid, STORE.version("merchant", mid),
           merchant.get("category_slug"), STORE.version("category", merchant.get("category_slug")),
           cid, STORE.version("customer", cid))
    return key, category, merchant, trg, customer


def _schedule(trigger_id: str, now: datetime | None = None) -> asyncio.Task | None:
    inp = _inputs_for(trigger_id)
    if not inp:
        return None
    key, category, merchant, trg, customer = inp
    task = _jobs.get(key)
    if task is None:
        # background jobs may queue for an LLM slot for a while; tick never waits longer than its budget
        task = asyncio.create_task(compose(category, merchant, trg, customer, now=now or _utcnow(), timeout=BG_COMPOSE_SECONDS))
        _jobs[key] = task
    return task


# ---------------------------------------------------------------- endpoints
@app.get("/")
async def root():
    return {"service": "vera-bot", "status": "ok", "docs": "/docs"}


@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - STORE.started),
            "contexts_loaded": STORE.counts()}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.getenv("TEAM_NAME", "Aarav Jhamb"),
        "team_members": [os.getenv("TEAM_MEMBER", "Aarav Jhamb")],
        "model": llm.model_label(),
        "approach": ("Vera does the work before it asks: why now -> one real number -> a finished draft -> reply YES. "
                     "Grounded fact sheet + per-trigger playbooks -> LLM (temp 0) picks the angle and what to attach; "
                     "the bot builds the quoted draft from verified merchant fields. Strict validator (numbers, prices, "
                     "dates, salutation, single final CTA) -> repair retry -> rule-based fallback. Token-aware limiter "
                     "with AI triage by value; deterministic reply router that answers with the actual deliverable."),
        "contact_email": os.getenv("CONTACT_EMAIL", ""),
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "submitted_at": SUBMITTED_AT,
    }


@app.get("/v1/debug/llm")
async def llm_debug():
    """Local diagnostics for the harness: model, call counters, last provider error (never the key)."""
    return {"model": llm.model_label(), "stats": llm.STATS, "last_error": llm.LAST_ERROR}


@app.post("/v1/context")
async def push_context(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_json", "details": "body is not JSON"})
    scope, cid, version, payload = body.get("scope"), body.get("context_id"), body.get("version"), body.get("payload")
    if scope not in SCOPES:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {SCOPES}"})
    if not cid or not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_payload", "details": "context_id and object payload required"})
    try:
        version = int(version)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_version", "details": "version must be an integer"})

    accepted, current = STORE.put_context(scope, str(cid), version, payload)
    if not accepted:
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": current})

    # warm the composer for anything this push affects
    if scope == "trigger":
        _schedule(str(cid))
    elif scope in ("merchant", "customer"):
        mid = payload.get("merchant_id") if scope == "customer" else str(cid)
        for (s, tid), c in list(STORE.contexts.items()):
            if s == "trigger" and (c["payload"].get("merchant_id") == mid) and \
                    c["payload"].get("suppression_key") not in STORE.sent_suppression_keys:
                _schedule(tid)
    return {"accepted": True, "ack_id": f"ack_{cid}_v{version}", "stored_at": _iso(_utcnow())}


@app.post("/v1/tick")
async def tick(request: Request):
    started = time.monotonic()
    try:
        body = await request.json()
    except Exception:
        body = {}
    now = parse_iso(body.get("now")) or _utcnow()
    hinted: list[str] = [t for t in (body.get("available_triggers") or []) if isinstance(t, str)]

    # The judge's hint list is the source of truth for "active now". With no hint we
    # consider our own stored, unexpired, unsent triggers.
    if hinted:
        candidates = hinted
    else:
        candidates = []
        for (s, tid), c in list(STORE.contexts.items()):
            exp = parse_iso(c["payload"].get("expires_at"))
            if s == "trigger" and (exp is None or exp > now):
                candidates.append(tid)

    chosen: list[tuple[str, dict]] = []
    merchants_this_tick: set[tuple] = set()
    skipped: dict[str, str] = {}
    trig_objs = [(tid, STORE.get("trigger", tid)) for tid in dict.fromkeys(candidates)]
    trig_objs = [(tid, t) for tid, t in trig_objs if t]
    trig_objs.sort(key=lambda x: (-int(x[1].get("urgency") or 0), x[0]))
    for tid, trg in trig_objs:
        mid = trg.get("merchant_id")
        sk = trg.get("suppression_key") or tid
        if sk in STORE.sent_suppression_keys:
            skipped[tid] = "already_sent"; continue
        if STORE.is_opted_out(mid):
            skipped[tid] = "merchant_opted_out"; continue
        if _inputs_for(tid) is None:
            skipped[tid] = "missing_merchant_or_category_context"; continue
        cid = trg.get("customer_id")
        if trg.get("scope") == "customer" or cid:
            cust = STORE.get("customer", cid)
            if not cust:
                skipped[tid] = "customer_context_missing"; continue
            if (cust.get("preferences") or {}).get("reminder_opt_in") is False:
                skipped[tid] = "customer_not_opted_in"; continue
        # brief: at most one action per (merchant_id, conversation_id) per tick; each trigger is its own conversation
        key_who = (mid, conversation_id_for(trg, mid, cid))
        if key_who in merchants_this_tick:
            skipped[tid] = "one_message_per_conversation_per_tick"; continue
        merchants_this_tick.add(key_who)
        chosen.append((tid, trg))
        if len(chosen) >= MAX_ACTIONS:
            break

    tasks = {tid: _schedule(tid, now) for tid, _ in chosen}
    tasks = {k: v for k, v in tasks.items() if v is not None}
    remaining = max(0.5, TICK_BUDGET - (time.monotonic() - started))
    if tasks:
        await asyncio.wait(list(tasks.values()), timeout=remaining)

    actions = []
    for tid, trg in chosen:
        task = tasks.get(tid)
        msg = None
        if task is not None and task.done() and not task.cancelled() and task.exception() is None:
            msg = task.result()
        if msg is None:   # not ready in budget -> deterministic fallback right now (no LLM)
            if task is not None and not task.done():
                task.cancel()     # frees its place in the LLM queue for messages not yet sent
            inp = _inputs_for(tid)
            if not inp:
                continue
            _k, category, merchant, _t, customer = inp
            msg = await compose(category, merchant, trg, customer, now=now, timeout=0, use_llm=False)
        mid = trg.get("merchant_id")
        cid = trg.get("customer_id")
        conv_id = conversation_id_for(trg, mid, cid)
        if STORE.has_conversation(conv_id) and STORE.conversation(conv_id).turns:
            conv_id = f"{conv_id}_{int(now.timestamp())}"
        conv = STORE.conversation(conv_id, mid, cid)
        conv.trigger_id = tid
        conv.send_as = msg["send_as"]
        conv.language = (msg.get("_facts") or {}).get("customer", {}).get("language") or \
            (msg.get("_facts") or {}).get("merchant", {}).get("language")
        conv.turns.append({"role": "bot", "body": msg["body"]})
        STORE.sent_suppression_keys.add(msg["suppression_key"])
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": cid,
            "send_as": msg["send_as"],
            "trigger_id": tid,
            "template_name": msg["template_name"],
            "template_params": msg["template_params"],
            "body": msg["body"],
            "cta": msg["cta"],
            "suppression_key": msg["suppression_key"],
            "rationale": msg["rationale"],
        })
        log.info("tick send %s via %s: %s", tid, msg.get("_source"), msg["body"][:90])
    if skipped:
        log.info("tick skipped: %s", skipped)
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"action": "end", "rationale": "malformed request"})
    try:
        return await asyncio.wait_for(handle_reply(body), timeout=12.0)
    except asyncio.TimeoutError:
        return {"action": "wait", "wait_seconds": 600, "rationale": "Composer timed out; backing off briefly instead of sending a weak reply."}


@app.post("/v1/teardown")
async def teardown():
    STORE.teardown()
    _jobs.clear()
    return {"ok": True}


# ---------------------------------------------------------------- keep-alive (Render free tier)
async def _keepalive():
    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("KEEPALIVE_URL")
    if not url:
        return
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(600)
            try:
                await client.get(f"{url.rstrip('/')}/v1/healthz")
            except Exception:
                pass


@app.on_event("startup")
async def _startup():
    log.info("Vera bot starting. LLM: %s", llm.model_label())
    asyncio.create_task(_keepalive())
