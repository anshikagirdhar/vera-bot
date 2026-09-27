"""compose(category, merchant, trigger, customer?) -> message.

Pipeline:
  1. build_facts()      - compact, grounded fact sheet (facts.py)
  2. LLM draft          - kind-specific playbook + strict JSON output
  3. validate()         - taboo words, URLs, fabricated numbers, CTA shape, salutation
  4. one repair retry   - feed the validator's complaints back to the model
  5. rule-based fallback if anything still fails (fallback.py)
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from . import llm
from .facts import CUSTOMER_KINDS, _canon, parse_iso, allowed_numbers, build_facts, compact_json, numbers_in
from .fallback import attach_draft, compose_fallback

PROMPT_VERSION = "composer_v7"

SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp assistant for Indian local merchants (dentists, salons, restaurants, gyms, pharmacies).
You write ONE WhatsApp message for the situation described in the FACTS JSON.

GOAL: a message this specific merchant (or their customer) would actually reply to.

HARD RULES
1. Use ONLY facts in the FACTS JSON. Never invent numbers, dates, prices, slots, names, competitors, research, sources or customer counts. Every number you write must appear in FACTS. Do not do arithmetic that creates new numbers.
2. If trigger.payload_is_empty is true, do NOT describe event details you were not given; anchor the message on the merchant's own data (performance, offers, signals, review themes, customer history) instead.
3. Lead with WHY NOW: the first sentence must make the trigger clear.
4. Exactly ONE call to action, and it is the last sentence. Prefer a binary "Reply YES" style ask, or one easy question. Never offer multiple options, except booking slots for a customer (e.g. "Reply 1 for Wed, 2 for Thu").
5. No URLs or links. No long preamble ("I hope you're doing well"). Don't introduce yourself as Vera. No hype, no ALL CAPS, at most one emoji.
6. Never use any word in voice.taboo_words.
7. Service + price ("Dental Cleaning @ ₹299") beats "% off". Use the merchant's active offers when relevant; never claim they run an offer that is not in merchant.active_offers, and never invent a new offer, combo, price or time slot. If an offer names days (e.g. "(Tue-Thu)"), only push it for those days; today's weekday is in FACTS.today.
8. Language: if the target's language is "hi-en", write natural Hindi-English code-mix in Roman script (e.g. "Aapke liye draft ready kar doon?"). Otherwise English.
8b. Dates and days: never name a weekday, "this weekend", "tomorrow" or a duration ("5 months", "a week away") unless FACTS gives it. Today's date is FACTS.today.
9. Keep it tight: 2-5 sentences plus the draft, under about 600 characters.
11. DO THE WORK FIRST, but never write the draft yourself: when the best next step is a ready Google post, a review-request message or a public reply to reviews, set "attach" to "post", "review_request" or "public_reply" and end your body with a short lead-in such as "Here's a ready post:". The system then appends a verified draft (built from the merchant's own name, locality and live offer) plus the "Reply YES" CTA. With "attach" set, do NOT write your own CTA and do NOT put any draft text in quotes. Otherwise use "attach": "none" and end with your own single CTA.
10. Numbers: quote them as given. "merchant_own_usual_value" is the merchant's OWN usual level, not a peer figure. Never compute before/after values, gaps or totals.

VOICE BY CATEGORY
- dentists: clinical peer-to-peer, technical vocabulary welcome, address as "Dr. <name>", cite sources.
- salons: warm, practical, fellow-operator.
- restaurants: operator-to-operator ("covers", "AOV", "delivery radius").
- gyms: coach-like, motivating, never guilt-tripping customers.
- pharmacies: trustworthy, precise, respectful (esp. seniors).

SENDER
- send_as "vera": you are writing TO the merchant. Address them by merchant.salutation.
- send_as "merchant_on_behalf": you are writing AS the merchant's business TO their customer. The READER IS THE CUSTOMER, not the owner: open with the customer's name, name the business, never address the owner, never mention business metrics. No medical claims. Respect the customer's state (no guilt for lapsed customers) and preferences.

COMPULSION LEVERS (use 1-3): a specific verifiable number or source; loss aversion ("you're missing X"); social proof from peer_benchmarks; effort externalisation ("I've drafted it - just say YES"); curiosity; asking the merchant a question; a single binary commitment.

JUDGEMENT: add a genuinely useful, category-aware point of view (e.g. a seasonal dip is normal - save ad spend; a weekend match night hurts dine-in - push delivery). Don't just restate the trigger.

OUTPUT: JSON only, with keys:
{"body": "<the WhatsApp message>",
 "attach": "post" | "review_request" | "public_reply" | "none",
 "cta": "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | "open_ended" | "none",
 "rationale": "<1-2 sentences: which facts you used, which levers, why this is the best message now>"}"""

PLAYBOOKS: dict[str, str] = {
    "research_digest": "Cite the digest item's source and key number (trial size, effect). Tie it to the merchant's own patients/customers (customer_aggregate, signals). Offer to pull the abstract or draft a shareable customer note.",
    "regulation_change": "Compliance heads-up: what changed, the deadline, what they must check. Calm, precise, cite the circular. Offer a checklist/audit.",
    "cde_opportunity": "Professional-development invite: title, date/time, credits, fee from the digest item. Offer to register/block the calendar.",
    "supply_alert": "Urgent but calm: molecule, batch numbers, manufacturer, what customers need. Offer to draft the customer notice + replacement workflow.",
    "perf_dip": "State the exact drop (metric, %, window). Add one peer benchmark or merchant signal that explains it. Propose ONE concrete fix using their live offer or a stale-posts signal.",
    "seasonal_perf_dip": "Reframe: this dip is expected for the season (say so plainly), so don't panic or spend on ads now. Redirect to retention of existing customers/members.",
    "perf_spike": "Celebrate the specific lift and its likely driver, then create urgency to capitalise before it fades; propose one follow-up action.",
    "milestone_reached": "Name the exact milestone and the gap (if imminent). Offer a drafted review-request or celebration post.",
    "review_theme_emerged": "Quote the theme/common_quote and count. Offer a drafted public reply and one operational fix.",
    "competitor_opened": "Name the competitor, distance and their offer exactly as given. Advise against a blind price war; lean on the merchant's strengths (rating, reviews, live offer). Offer a drafted post.",
    "festival_upcoming": "Festival + date + days until. Timing: more than 60 days away = too early to spend, offer to draft a dated plan; 15-60 days = teaser/early-bird; under 15 = booking window, act now. Build the idea on their existing offer.",
    "ipl_match_today": "Match, venue, time. Use judgement: weekend matches pull people home (push delivery), weeknight matches bring groups out (dine-in). Only use a live offer valid today (check its days against FACTS.today); otherwise suggest a push without a new price. Offer drafted creatives.",
    "renewal_due": "Days left, plan, amount. Recap the value delivered using their own performance numbers. Single YES to renew.",
    "winback_eligible": "Days since expiry and what they've lost since (views dip, lapsed customers). Offer to reactivate + run a win-back.",
    "dormant_with_vera": "Re-open gently with one fresh, specific insight from their data, then ask ONE easy question about their business.",
    "curious_ask_due": "Ask the merchant one low-effort question about their business this week (e.g. most-asked service), and promise a concrete artifact in return (Google post + ready WhatsApp reply). Can offer a guess based on their data.",
    "gbp_unverified": "Profile is unverified; state the estimated uplift and the verification path. Offer to start it for them.",
    "active_planning_intent": "The merchant ALREADY said yes. Do not ask qualifying questions. Deliver a concrete starter draft (bullet-style lines are fine) using their catalog/offers, then ask for one go-ahead.",
    "category_seasonal": "List the demand shifts exactly as given. Give one concrete shelf/stocking action. Offer a drafted plan/post.",
    "recall_due": "Customer recall: service due, last visit, real slots if given, live offer price. Honour language and preferred time. Slot-choice CTA allowed.",
    "customer_lapsed_soft": "Warm, no-guilt nudge to a lapsed customer referencing their history; live offer; zero-commitment YES.",
    "customer_lapsed_hard": "Warm, no-guilt win-back referencing their past goal/history; something new or the live offer; zero-commitment YES.",
    "appointment_tomorrow": "Friendly reminder of tomorrow's appointment (only use a time if given); confirm or reschedule in one reply.",
    "chronic_refill_due": "Respectful refill reminder: exact medicines and run-out date if given; delivery; single CONFIRM. If this category is not a pharmacy, turn it into a sensible follow-up/check-up reminder for this business instead - never mention medicines.",
    "trial_followup": "Thank them for the trial (date), offer the real next session option, single YES.",
    "wedding_package_followup": "Countdown to the wedding date, the next-step window, the relevant offer, single YES.",
}

# ---- AI triage: the free LLM tier allows ~3 calls/min, so spend them where judgement matters.
# Routine customer reminders are already excellent as templates and never use the LLM.
TEMPLATE_ONLY_KINDS = {"appointment_tomorrow", "chronic_refill_due", "recall_due", "gbp_unverified"}
KIND_WEIGHT = {
    "research_digest": 5, "regulation_change": 5, "competitor_opened": 5, "active_planning_intent": 5,
    "supply_alert": 5, "review_theme_emerged": 4, "cde_opportunity": 4, "perf_dip": 4, "seasonal_perf_dip": 4,
    "winback_eligible": 4, "customer_lapsed_hard": 3, "customer_lapsed_soft": 3, "dormant_with_vera": 3,
    "festival_upcoming": 3, "ipl_match_today": 3, "category_seasonal": 3, "curious_ask_due": 3, "perf_spike": 3,
    "renewal_due": 2, "milestone_reached": 2,
}


def llm_priority(trigger: dict) -> int:
    """Higher = gets an LLM slot first. Kind value dominates; the trigger's urgency breaks ties;
    a thin placeholder payload is worth less LLM effort."""
    kind = trigger.get("kind", "")
    thin = bool((trigger.get("payload") or {}).get("placeholder"))
    return KIND_WEIGHT.get(kind, 2) * 10 + int(trigger.get("urgency") or 0) - (15 if thin else 0)


GENERIC_PLAYBOOK = "Explain clearly why you are messaging now using the trigger. Anchor on one verifiable fact from the merchant's data and propose one concrete, low-effort next step."

TABOO_DEFAULT = ["guaranteed", "100% safe", "miracle", "best in city", "cure"]
URL_RE = re.compile(r"(https?://|www\.)\S+|\b\S+\.(com|in|org|net|io|co)\b(/\S*)?", re.I)
_GENERIC_BIZ = {"the", "and", "dental", "clinic", "care", "pharmacy", "medicos", "medical", "studio", "salon",
                "gym", "fitness", "restaurant", "cafe", "kitchen", "health", "plus", "centre", "center", "by"}
QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]


def _kind_for_prompt(trigger: dict, category_slug: str) -> str:
    kind = trigger.get("kind", "")
    pb = PLAYBOOKS.get(kind, GENERIC_PLAYBOOK)
    if kind in CUSTOMER_KINDS and kind != "chronic_refill_due" and category_slug == "pharmacies":
        pb += " Keep medical claims out entirely."
    return pb


def _canon_num(n: str) -> str:
    return _canon(float(n))


def validate(draft: dict | None, facts: dict, extra_allowed: list | None = None) -> list[str]:
    """Return a list of problems (empty = OK)."""
    if not isinstance(draft, dict):
        return ["no JSON object returned"]
    body = draft.get("body")
    if not isinstance(body, str) or len(body.strip()) < 20:
        return ["body missing or too short"]
    problems = []
    low = body.lower()
    taboos = [t.split("(")[0].strip().lower() for t in (facts["voice"].get("taboo_words") or [])] + TABOO_DEFAULT
    hit = [t for t in taboos if t and re.search(r"\b" + re.escape(t) + r"\b", low)]
    if hit:
        problems.append(f"uses taboo words {hit}")
    if URL_RE.search(body):
        problems.append("contains a URL/link - remove it")
    allowed = allowed_numbers(facts, extra_allowed)
    bad = sorted({n for n in numbers_in(body) if n not in allowed})
    if bad:
        problems.append(f"numbers not present in FACTS (fabricated?): {bad}")
    strict = allowed_numbers(facts, extra_allowed, generic=False)
    own = allowed_numbers({"offers": facts["merchant"].get("active_offers"), "payload": facts["trigger"].get("payload"),
                           "digest": facts.get("relevant_digest_item"), "sub": facts["merchant"].get("subscription"),
                           "customer": facts.get("customer")}, extra_allowed, generic=False)
    rupees = sorted({n for n in numbers_in(" ".join(re.findall(r"(?:₹|rs\.?|inr)\s*[\d,]+", low))) if n not in own})
    if rupees:
        problems.append(f"prices not in the merchant's own offers/trigger (invented offer?): {rupees}")
    durs = sorted({n for n, _ in re.findall(r"(\d+)\s*[- ]?\s*(month|mahin|week|hafte|year|saal|day|din)", low)
                   if _canon_num(n) not in strict})
    if durs:
        problems.append(f"durations not in FACTS: {durs} - don't compute or guess time spans")
    rel = (facts.get("customer") or {}).get("relationship") or {}
    last, today = parse_iso(rel.get("last_visit")), parse_iso(str(facts.get("today", ""))[:10])
    if last and today:
        gap = (today - last).days
        for m in re.finditer(r"(\d+)\s*[- ]?\s*(month|mahin|week|hafte|day|din)\w*", low):
            n, unit = m.group(1), m.group(2)
            around = low[max(0, m.start() - 20):m.end() + 25]
            if not re.search(r"\b(been|since|ago|pehle|ho gaye|ho gya|back|se)\b", around):
                continue      # "6 month cleaning" is a service name, not a time gap
            per = 30 if unit in ("month", "mahin") else 7 if unit in ("week", "hafte") else 1
            if abs(int(n) * per - gap) > max(3, per):
                problems.append(f"'{n} {unit}' is wrong - the last visit was {gap} days ago; state the date instead")
    ftext = compact_json(facts).lower()
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        if re.search(r"\b" + day + r"\b", low) and day not in ftext and f"({day[:3]})" not in ftext and \
                not re.search(r"\b" + day[:3] + r"\b", ftext):
            problems.append(f"'{day}' is not in FACTS - don't invent days")
    if re.search(r"\bweekend\b", low) and "weekend" not in ftext and '"is_weeknight":false' not in ftext:
        problems.append("'weekend' is not supported by FACTS")
    if re.search(r"\b(a|one|ek)\s+(week|hafta|hafte)\s+(away|door|mein)\b", low) and facts["trigger"].get("payload_is_empty"):
        problems.append("no event date in FACTS - don't say how far away it is")
    if len(body) > 700:
        problems.append("too long - keep under 600 characters")
    sal = facts["merchant"].get("salutation")
    if facts.get("send_as") == "vera" and sal:
        if sal.split()[-1].lower() not in low:
            problems.append(f"address the merchant as '{sal}'")
        if not sal.lower().startswith("dr") and re.search(r"\bdr\.?\s", low):
            problems.append(f"this merchant is not a doctor - address them as '{sal}', no 'Dr.'")
    cname = (facts.get("customer") or {}).get("name")
    if facts.get("send_as") == "merchant_on_behalf":
        if cname and cname.lower() not in low:
            problems.append(f"address the customer as '{cname}'")
        biz = facts["merchant"].get("business_name")
        key = [w for w in re.findall(r"[a-z]+", biz.lower().replace("'s", "")) if len(w) > 2 and w not in _GENERIC_BIZ] if biz else []
        if key and not any(re.search(r"\b" + w + r"\b", low) for w in key):
            problems.append(f"the customer must see which business this is - name '{biz}'")
        if re.search(r"\b(ctr|peers?|benchmark|footfall|repeat[- ]customer|views|calls)\b", low):
            problems.append("this goes to the CUSTOMER - no business metrics, peer data or internal stats")
    if body.count("?") > 2:
        problems.append("too many questions - one CTA only")
    tail = re.split(r"(?<=[.!?])\s+", body.strip())[-1].lower()
    if not ("?" in tail or re.search(
            r"\b(reply|confirm|yes|batao|bataiye|batayein|bolo|let me know|tell me)\b", tail)):
        problems.append("the call to action must be the LAST sentence (e.g. 'Reply YES ...')")
    return problems


def _self_written_quotes(draft: dict | None, facts: dict) -> list[str]:
    """Quoted drafts must come from attach_draft(); a quote the model wrote is only OK
    if it is copied verbatim from FACTS (e.g. a review's common_quote)."""
    if not isinstance(draft, dict) or not isinstance(draft.get("body"), str):
        return []
    ftext = compact_json(facts).lower()
    bad = [q for q in re.findall(r"[“\"]([^”\"]{15,})[”\"]", draft["body"]) if q.lower().strip() not in ftext]
    return ["don't write draft text in quotes yourself - set \"attach\" and the system adds a verified draft"] if bad else []


def _assemble(draft: dict | None, facts: dict) -> dict | None:
    """LLM body + (optional) verified draft and CTA appended by the bot."""
    if not isinstance(draft, dict) or not isinstance(draft.get("body"), str):
        return draft
    att = attach_draft(facts, str(draft.get("attach") or "none"))
    if not att:
        return draft
    text, cta_line = att
    return {**draft, "body": f"{draft['body'].strip()} {text} {cta_line}", "cta": "binary_yes_no"}


def _check(draft: dict | None, facts: dict) -> tuple[dict | None, list[str]]:
    quotes = _self_written_quotes(draft, facts)
    full = _assemble(draft, facts)
    return full, quotes + validate(full, facts)


def _tidy(draft: dict) -> dict:
    body = " ".join(str(draft.get("body", "")).replace("\r", "").split(" "))
    body = re.sub(r"[ \t]+", " ", body).strip()
    cta = draft.get("cta") if draft.get("cta") in {
        "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "open_ended", "none"} else "open_ended"
    rationale = str(draft.get("rationale") or "").strip()[:400]
    return {"body": body, "cta": cta, "rationale": rationale}


def conversation_id_for(trigger: dict, merchant_id: str, customer_id: str | None) -> str:
    who = customer_id or merchant_id
    tid = trigger.get("id", "trg")
    return f"conv_{who}_{tid}"[:120]


def template_for(kind: str, send_as: str, body: str, name: str) -> tuple[str, list[str]]:
    prefix = "merchant" if send_as == "merchant_on_behalf" else "vera"
    sentences = re.split(r"(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<=[.!?])\s+", body)
    cta = sentences[-1] if sentences else ""
    middle = " ".join(sentences[1:-1]) if len(sentences) > 2 else (sentences[0] if sentences else "")
    return f"{prefix}_{kind or 'generic'}_v1", [name or "", middle[:900], cta[:300]]


async def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None,
                  now: datetime | None = None, timeout: float = 12.0, use_llm: bool = True) -> dict:
    now = now or datetime.now(timezone.utc)
    facts = build_facts(category, merchant, trigger, customer, now)
    cat_slug = facts["category"]
    source = "fallback"
    result = None

    prio = llm_priority(trigger)
    if use_llm and llm.enabled() and trigger.get("kind") not in TEMPLATE_ONLY_KINDS:
        user = (f"TRIGGER PLAYBOOK ({trigger.get('kind')}): {_kind_for_prompt(trigger, cat_slug)}\n\n"
                f"FACTS:\n{compact_json(facts)}\n\nWrite the message now. JSON only.")
        raw = await llm.complete_json(SYSTEM_PROMPT, user, timeout=timeout * 0.55, priority=prio)
        draft, problems = _check(raw, facts) if raw is not None else (None, ["no draft"])
        if draft is not None and problems:
            repair = (user + f"\n\nYour previous draft:\n{compact_json(raw)}\n"
                      f"It has these problems: {problems}. Fix ALL of them and return the corrected JSON.")
            draft2 = await llm.complete_json(SYSTEM_PROMPT, repair, timeout=timeout * 0.4, priority=prio + 1)
            full2, problems2 = _check(draft2, facts)
            if draft2 is not None and not problems2:
                draft, problems = full2, []
        if draft is not None and not problems:
            result = _tidy(draft)
            source = "llm"

    if result is None:
        result = compose_fallback(facts)

    send_as = facts["send_as"]
    name = (facts.get("customer") or {}).get("name") if send_as == "merchant_on_behalf" else facts["merchant"].get("salutation")
    tname, tparams = template_for(trigger.get("kind", ""), send_as, result["body"], name or "")
    result.update({
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key") or f"{trigger.get('kind')}:{merchant.get('merchant_id')}",
        "template_name": tname,
        "template_params": tparams,
        "_source": source,
        "_facts": facts,
    })
    if not result.get("rationale"):
        result["rationale"] = f"{trigger.get('kind')} trigger composed from merchant + category context."
    return result
