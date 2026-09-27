"""Turn the 4 raw contexts into a compact, *grounded* fact sheet.

Both the LLM prompt and the rule-based fallback read from this. Keeping it small
and explicit is what stops hallucination: the model only sees facts we can verify,
and the validator later checks every number in the output against this sheet.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

CUSTOMER_KINDS = {
    "recall_due", "customer_lapsed_soft", "customer_lapsed_hard", "appointment_tomorrow",
    "chronic_refill_due", "trial_followup", "wedding_package_followup", "bridal_followup",
    "unplanned_slot_open", "birthday", "customer_winback",
}

# Hindi words in Roman script that signal a Hinglish speaker (per-turn language detection)
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "kya", "aap", "aapka", "aapki", "karo", "kar", "karna", "mujhe",
    "haan", "han", "ji", "accha", "acha", "theek", "thik", "bhai", "kaise", "kab", "kitna", "kyun",
    "chahiye", "bolo", "batao", "abhi", "baad", "mein", "hum", "humara", "hamara", "yeh", "woh",
    "bhejo", "dekho", "samajh", "chalo", "chalega", "zaroor", "shukriya", "dhanyavad", "namaste",
    "matlab", "lekin", "par", "sab", "kuch", "bahut", "jaldi", "kal", "aaj",
}


# ---------------------------------------------------------------- small helpers
def parse_iso(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def pct(x: Any) -> str | None:
    """0.18 -> '18%', -0.5 -> '50%' (sign handled by caller wording)."""
    if isinstance(x, (int, float)):
        v = abs(x) * 100
        return f"{v:.0f}%" if abs(v - round(v)) < 0.05 else f"{v:.1f}%"
    return None


def inr(v: Any) -> str:
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return str(v)
    s = str(n)
    if len(s) > 3:  # Indian grouping: 1,23,456
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        s = head + "," + tail
    return "₹" + s


def humanize(slug: Any) -> str:
    return str(slug).replace("_", " ").strip() if slug is not None else ""


def window_words(w: Any) -> str:
    """'7d' -> '7 days', '30d' -> '30 days'."""
    m = re.fullmatch(r"(\d+)\s*d", str(w or "").strip())
    return f"{m.group(1)} days" if m else humanize(w or "week")


def nice_date(s: Any) -> str:
    """'2026-05-12' -> '12 May 2026' (leaves anything unparseable as-is)."""
    dt = parse_iso(str(s)) if s else None
    if not dt:
        return str(s)
    return f"{dt.day} {MONTHS[dt.month - 1]} {dt.year}"


def detect_language(text: str) -> str:
    """'hi-en' if the text is Hindi / Hinglish, else 'en'."""
    if re.search(r"[ऀ-ॿ]", text or ""):
        return "hi-en"
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    if not words:
        return "en"
    hits = sum(1 for w in words if w in HINGLISH_MARKERS)
    return "hi-en" if hits >= 2 or (hits >= 1 and len(words) <= 4) else "en"


def salutation(merchant: dict, category_slug: str) -> str:
    ident = merchant.get("identity", {}) or {}
    first = (ident.get("owner_first_name") or "").strip()
    if not first:
        return ""
    if category_slug == "dentists" or merchant.get("category_slug") == "dentists":
        return first if first.lower().startswith("dr") else f"Dr. {first}"
    return first


def merchant_language(merchant: dict, customer: dict | None = None) -> str:
    if customer:
        pref = str((customer.get("identity") or {}).get("language_pref", "")).lower()
        if "hi" in pref:
            return "hi-en"
        return "en"
    langs = (merchant.get("identity") or {}).get("languages") or []
    return "hi-en" if "hi" in langs else "en"


# ---------------------------------------------------------------- digest lookup
def resolve_digest_item(category: dict, trigger: dict) -> dict | None:
    payload = trigger.get("payload") or {}
    digest = category.get("digest") or []
    by_id = {d.get("id"): d for d in digest}
    for key in ("top_item_id", "digest_item_id", "alert_id", "item_id"):
        if payload.get(key) in by_id:
            return by_id[payload[key]]
    kind = trigger.get("kind", "")
    want = {
        "research_digest": "research", "regulation_change": "compliance", "cde_opportunity": "cde",
        "category_trend_movement": "trend", "supply_alert": "alert", "tech_update": "tech",
    }.get(kind)
    if want:
        for d in digest:
            if d.get("kind") == want:
                return d
    return None


def seasonal_now(category: dict, now: datetime) -> list[dict]:
    """Seasonal beats whose month range covers `now`."""
    m = now.month
    out = []
    for beat in category.get("seasonal_beats") or []:
        rng = str(beat.get("month_range", ""))
        parts = [p.strip()[:3].title() for p in rng.split("-")]
        try:
            idx = [MONTHS.index(p) + 1 for p in parts if p in MONTHS]
        except ValueError:
            continue
        if not idx:
            continue
        a, b = idx[0], idx[-1]
        inside = a <= m <= b if a <= b else (m >= a or m <= b)
        if inside:
            out.append(beat)
    return out


# ---------------------------------------------------------------- the fact sheet
def build_facts(category: dict, merchant: dict, trigger: dict, customer: dict | None,
                now: datetime) -> dict:
    cat_slug = category.get("slug") or merchant.get("category_slug", "")
    ident = merchant.get("identity") or {}
    perf = merchant.get("performance") or {}
    peer = category.get("peer_stats") or {}
    voice = category.get("voice") or {}
    placeholder = bool((trigger.get("payload") or {}).get("placeholder"))
    payload = {} if placeholder else dict(trigger.get("payload") or {})
    if "vs_baseline" in payload:   # models misread this as a peer average
        payload["merchant_own_usual_value"] = payload.pop("vs_baseline")

    active_offers = [o.get("title") for o in merchant.get("offers") or [] if o.get("status") == "active"]
    other_offers = [f'{o.get("title")} ({o.get("status")})' for o in merchant.get("offers") or []
                    if o.get("status") != "active"]

    derived: dict[str, Any] = {}
    ctr, peer_ctr = perf.get("ctr"), peer.get("avg_ctr")
    if isinstance(ctr, (int, float)) and isinstance(peer_ctr, (int, float)) and peer_ctr:
        derived["ctr_pct"] = round(ctr * 100, 1)
        derived["peer_ctr_pct"] = round(peer_ctr * 100, 1)
        derived["ctr_vs_peer"] = "below" if ctr < peer_ctr else "above_or_equal"
    if isinstance(perf.get("views"), (int, float)) and isinstance(peer.get("avg_views_30d"), (int, float)):
        derived["views_vs_peer"] = "below" if perf["views"] < peer["avg_views_30d"] else "above_or_equal"
    if isinstance(perf.get("calls"), (int, float)) and isinstance(peer.get("avg_calls_30d"), (int, float)):
        derived["calls_vs_peer"] = "below" if perf["calls"] < peer["avg_calls_30d"] else "above_or_equal"
    d7 = perf.get("delta_7d") or {}
    for k, v in d7.items():
        if isinstance(v, (int, float)):
            derived[f"{k}_7d_readable"] = ("+" if v >= 0 else "-") + (pct(v) or "")

    facts: dict[str, Any] = {
        "today": now.strftime("%Y-%m-%d (%a)"),
        "category": cat_slug,
        "voice": {
            "tone": voice.get("tone"), "register": voice.get("register"),
            "code_mix": voice.get("code_mix"),
            "vocab_allowed": (voice.get("vocab_allowed") or [])[:12],
            "taboo_words": voice.get("vocab_taboo") or voice.get("taboos") or [],
            "tone_examples": (voice.get("tone_examples") or [])[:3],
        },
        "merchant": {
            "salutation": salutation(merchant, cat_slug),
            "business_name": ident.get("name"),
            "locality": ident.get("locality"), "city": ident.get("city"),
            "established_year": ident.get("established_year"),
            "verified_on_google": ident.get("verified"),
            "language": merchant_language(merchant),
            "subscription": merchant.get("subscription"),
            "performance_30d": {k: v for k, v in perf.items() if k != "delta_7d"},
            "delta_7d": d7,
            "active_offers": active_offers,
            "past_offers": other_offers,
            "customer_aggregate": merchant.get("customer_aggregate"),
            "signals": merchant.get("signals"),
            "review_themes": merchant.get("review_themes"),
            "recent_conversation": [
                {"from": t.get("from"), "body": t.get("body"), "engagement": t.get("engagement")}
                for t in (merchant.get("conversation_history") or [])[-4:]
            ],
        },
        "peer_benchmarks": peer,
        "derived": derived,
        "category_offer_catalog": [o.get("title") for o in category.get("offer_catalog") or []][:8],
        "trigger": {
            "kind": trigger.get("kind"), "scope": trigger.get("scope"), "source": trigger.get("source"),
            "urgency": trigger.get("urgency"), "payload": payload,
            "payload_is_empty": placeholder or not payload,
            "expires_at": trigger.get("expires_at"),
        },
    }

    item = resolve_digest_item(category, trigger)
    if item:
        facts["relevant_digest_item"] = item
    beats = seasonal_now(category, now)
    if beats:
        facts["seasonal_beats_now"] = beats
    trends = category.get("trend_signals") or []
    if trends:
        facts["trend_signals"] = trends[:3]
    if trigger.get("kind") in {"research_digest", "curious_ask_due", "dormant_with_vera"}:
        lib = category.get("patient_content_library") or []
        if lib:
            facts["shareable_content"] = [{"title": c.get("title")} for c in lib[:3]]

    if customer:
        cid = customer.get("identity") or {}
        facts["customer"] = {
            "name": cid.get("name"),
            "language_pref": cid.get("language_pref"),
            "language": merchant_language(merchant, customer),
            "age_band": cid.get("age_band"),
            "relationship": customer.get("relationship"),
            "state": customer.get("state"),
            "preferences": customer.get("preferences"),
            "consent_scope": (customer.get("consent") or {}).get("scope"),
        }
        facts["send_as"] = "merchant_on_behalf"
        # The reader is the customer: keep the merchant's internal numbers out of reach so the
        # model can't leak them ("our repeat rate is 62%") or write to the owner by mistake.
        m = facts["merchant"]
        for k in ("salutation", "subscription", "performance_30d", "delta_7d", "customer_aggregate",
                  "signals", "review_themes", "recent_conversation", "past_offers", "verified_on_google"):
            m.pop(k, None)
        for k in ("peer_benchmarks", "derived", "trend_signals", "category_offer_catalog"):
            facts.pop(k, None)
    else:
        facts["send_as"] = "vera"
    return facts


# ---------------------------------------------------------------- fact-check support
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _walk_numbers(obj: Any, out: set[str]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        _add_number(float(obj), out)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                _add_number(float(m.replace(",", "")), out)
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_numbers(k, out)
            _walk_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, out)


def _add_number(v: float, out: set[str]) -> None:
    for x in {v, abs(v)}:
        out.add(_canon(x))
        if abs(x) < 1.5:                       # fractions like 0.18 -> 18 (%)
            out.add(_canon(round(x * 100, 1)))
            out.add(_canon(round(x * 100)))
        out.add(_canon(round(x)))


def _canon(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}".rstrip("0").rstrip(".")


def allowed_numbers(facts: Any, extra: list[Any] | None = None, generic: bool = True) -> set[str]:
    out: set[str] = set()
    _walk_numbers(facts, out)
    if extra:
        _walk_numbers(extra, out)
    if not generic:
        return out
    # small counts, durations and reply options ("2-min", "Reply 1", "5 min") are always fine
    for n in range(0, 11):
        out.add(str(n))
    for n in (15, 20, 24, 30, 45, 48, 60, 90):
        out.add(str(n))
    return out


def numbers_in(text: str) -> list[str]:
    found = []
    for m in _NUM.findall(text or ""):
        try:
            found.append(_canon(float(m.replace(",", ""))))
        except ValueError:
            pass
    return found


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
