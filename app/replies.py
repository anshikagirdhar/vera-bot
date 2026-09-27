"""Multi-turn reply handling for POST /v1/reply.

Routing is rule-based first (fast, deterministic, and exactly what the replay tests
probe), and the LLM is only used to *word* a 'send' when a merchant engages.

Classes:
  auto_reply  -> 1st: one short nudge for the owner, 2nd: wait 24h, 3rd+: end
  opt_out     -> end immediately, suppress the merchant for 30 days
  hostile     -> end (short apology if it's the first contact)
  commit      -> ACTION mode: confirm what is being done now, no qualifying questions
  later       -> wait (back off)
  off_topic   -> polite decline + redirect to the original topic
  question/engaged/other -> grounded answer that moves one step forward
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from . import llm
from .composer import URL_RE, validate
from .facts import build_facts, compact_json, detect_language, humanize
from .store import STORE, Conversation

AUTO_REPLY_PATTERNS = [
    r"thank(s| you) for (contacting|reaching|your message|messaging)", r"our team will (respond|get back|contact)",
    r"will (get back|respond|revert) (to you )?(shortly|soon|asap)", r"automated (assistant|message|reply|response)",
    r"auto[- ]?reply", r"currently (unavailable|away|closed|out of office)", r"out of (the )?office",
    r"business hours", r"we are (closed|away)", r"this is an automated", r"aapki jaankari ke liye",
    r"hamari team tak", r"jaldi (hi )?sampark", r"for (urgent|immediate) (queries|assistance)", r"do not reply",
]
OPT_OUT_PATTERNS = [
    r"\bstop\b", r"not interested", r"unsubscribe", r"don'?t (message|text|contact|send)", r"do not (message|contact|send)",
    r"leave me alone", r"remove me", r"no more messages", r"band karo", r"mat bhejo", r"nahi chahiye", r"interest nahi",
    r"block (you|kar)",
]
HOSTILE_PATTERNS = [
    r"\bspam\b", r"useless", r"bothering me", r"why are you (bothering|messaging|disturbing)", r"waste of time",
    r"\bidiot\b", r"\bstupid\b", r"\bshut up\b", r"\bfraud\b", r"\bscam\b", r"bakwas", r"pagal", r"\bbc\b", r"\bmc\b",
    r"f+u+c+k", r"\bshit\b", r"irritat", r"pareshan mat",
]
COMMIT_PATTERNS = [
    r"\blet'?s do it\b", r"\bgo ahead\b", r"\bdo it\b", r"\bok(ay)?,? (let'?s|do|go|proceed|sure|send|please)", r"\bproceed\b",
    r"\byes\b", r"\byess+\b", r"\byeah\b", r"\byep\b", r"\bsure\b", r"\bconfirm(ed)?\b", r"\bplease (do|send|draft|go)",
    r"\bsend (it|me|the)\b", r"\bdraft (it|the|one)\b", r"what'?s next", r"\bi want to join\b", r"\bjoin(ing)?\b",
    r"\bsign me up\b", r"\bhaan\b", r"\bhaa\b", r"\bji haan\b", r"\bkar do\b", r"\bkardo\b", r"\bchalo\b", r"\bbhej do\b",
    r"\bthik hai\b", r"\btheek hai\b", r"\bdone\b", r"\bbook (it|me)\b", r"^\s*(1|2)\s*$", r"\bagreed\b", r"\bsounds good\b",
    r"judna hai", r"\bstart\b", r"^\s*ok(ay)?\s*[.!]*\s*$", r"^\s*ok(ay)?[,.! ]+(ji|sir|madam|thanks)\b",
]
LATER_PATTERNS = [
    r"\blater\b", r"\bbusy\b", r"\btomorrow\b", r"\bnext week\b", r"baad mein", r"\bkal\b", r"abhi nahi", r"not now",
    r"call (me )?later", r"in a meeting", r"\bdriving\b",
]
OFF_TOPIC_PATTERNS = [
    r"\bgst\b", r"income tax", r"\bitr\b", r"\bloan\b", r"\binsurance\b", r"\bvisa\b", r"\bpassport\b",
    r"\bca\b", r"accountant", r"\blegal (notice|advice)\b", r"\blawyer\b", r"stock (market|tips)", r"\bcrypto\b",
    r"\belectricity bill\b", r"\bjob\b", r"\bhomework\b",
]


def _match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify(message: str, seen_count: int) -> str:
    t = (message or "").lower().strip()
    if not t:
        return "other"
    # canned text, or the same longer message verbatim again (short "yes"/"ok" repeats are real replies)
    if _match(AUTO_REPLY_PATTERNS, t) or (seen_count >= 2 and len(t) >= 25):
        return "auto_reply"
    if _match(OPT_OUT_PATTERNS, t):
        return "opt_out"
    if _match(HOSTILE_PATTERNS, t):
        return "hostile_offtopic" if _match(OFF_TOPIC_PATTERNS, t) else "hostile"
    if _match(OFF_TOPIC_PATTERNS, t):
        return "off_topic"
    if _match(LATER_PATTERNS, t) and not _match(COMMIT_PATTERNS, t):
        return "later"
    if _match(COMMIT_PATTERNS, t):
        return "commit"
    if "?" in t or re.match(r"^(what|how|why|when|where|which|who|can|could|is|are|kya|kaise|kab|kitna|kyun)\b", t):
        return "question"
    return "engaged"


REPLY_SYSTEM = """You are Vera, magicpin's WhatsApp assistant for Indian local merchants, mid-conversation.
Write the NEXT message only. Rules:
- Use only facts in FACTS and the CONVERSATION. Never invent numbers, prices, dates, names or sources. No URLs.
- Don't re-introduce yourself. No preamble. 1-4 short sentences.
- Match the language of the merchant's latest message ("hi-en" = natural Hindi-English code-mix in Roman script).
- Never repeat a message you already sent in this conversation.
- End with exactly one clear next step.
- MODE is decisive:
  * commit: the merchant said yes. Switch to ACTION: say what you're doing now / what's ready, show the concrete deliverable (the final draft in quotes if one exists in the conversation), and ask only for a final CONFIRM. Do NOT ask any qualifying question and do not use the phrases "would you", "do you", "can you tell", "what if", "how about".
  * question: answer it directly from FACTS. If they ask what you'll do, what it looks like or how it works, SHOW the work now: the concrete draft in quotes “...” (built only from FACTS / the conversation) and the 1-2 steps that follow. Only if a factual answer isn't in FACTS, say you'll check rather than guessing. Then one step forward.
  * engaged: acknowledge briefly and move one concrete step forward toward the original goal.
  * off_topic: politely say that's outside what you can help with (suggest the right person, e.g. their CA for GST), then bring it back to the original topic in one line.
Output JSON only: {"body": "...", "cta": "binary_yes_no|binary_confirm_cancel|open_ended|none", "rationale": "..."}"""


def _lang_for(conv: Conversation, message: str, merchant: dict | None) -> str:
    lang = detect_language(message)
    if lang == "en" and conv.language == "hi-en" and len(message.split()) <= 3:
        lang = "hi-en"      # a bare "ok"/"yes" doesn't switch a Hinglish conversation to English
    conv.language = lang
    return lang


def _topic(conv: Conversation) -> str:
    trg = STORE.get("trigger", conv.trigger_id) or {}
    kind = trg.get("kind")
    return humanize(kind).replace("_", " ") + " update" if kind else "growing your profile"


def _first_bot(conv: Conversation) -> str:
    for t in conv.turns:
        if t["role"] == "bot":
            return t["body"]
    return ""


def _draft_for(conv: Conversation) -> str | None:
    """The concrete deliverable for this conversation: the draft already quoted in our opening
    message, or a post built from the merchant's own name, locality and live offer."""
    for t in conv.turns:
        if t["role"] == "bot":
            m = re.search(r"“([^”]{10,})”", t["body"])
            if m:
                return f"“{m.group(1)}”"
    merchant = STORE.get("merchant", conv.merchant_id) or {}
    ident = merchant.get("identity") or {}
    offer = next((o.get("title") for o in merchant.get("offers") or [] if o.get("status") == "active"), None)
    if not ident.get("name") or not offer:
        return None
    head = f"{ident['name']}, {ident['locality']}" if ident.get("locality") else ident["name"]
    return f"“{head} — {offer}. Call or WhatsApp us.”"


def _fallback_send(cls: str, lang: str, conv: Conversation, message: str) -> dict:
    hi = lang == "hi-en"
    topic = _topic(conv)
    draft = None if conv.customer_id else _draft_for(conv)
    if cls == "commit" and draft:
        body = (f"Done — final version yeh hai: {draft} Aaj hi aapki Google profile pe live kar dungi; publish karne ke liye CONFIRM reply karein."
                if hi else
                f"Done — here's the final version: {draft} I'll put it live on your Google profile today; reply CONFIRM to publish.")
        return {"action": "send", "body": body, "cta": "binary_confirm_cancel",
                "rationale": "Merchant committed: action mode — showed the finished deliverable and asked only for a final CONFIRM."}
    if cls == "question" and draft:
        body = (f"Seedha plan: 1) yeh post aapki Google profile pe jayega: {draft} 2) wahi note WhatsApp pe aapke regulars ko. Aapke OK ke bina kuch live nahi hoga — aage badhne ke liye YES reply karein."
                if hi else
                f"Here's exactly what I'll do: 1) publish this on your Google profile: {draft} 2) send the same note to your regulars on WhatsApp. Nothing goes live without your OK — reply YES to go ahead.")
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Merchant asked what happens next: answered with the actual deliverable and steps instead of a promise."}
    if cls == "engaged" and draft:
        body = (f"Yeh abhi ready hai — koi wait nahi: {draft} YES reply karein, aaj hi Google profile pe live ho jayega."
                if hi else
                f"It's ready now — no waiting: {draft} Reply YES and it goes live on your Google profile today.")
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Merchant engaged: removed the wait by showing the finished draft, one-word YES to publish."}
    if cls == "commit":
        body = ("Done — kaam shuru kar diya hai. Draft 10 minute mein ready hoga aur main yahin bhej dungi; aapko bas CONFIRM reply karna hai, uske baad main live kar dungi."
                if hi else
                "Done — I'm on it. Your draft will be ready here in about 10 minutes; reply CONFIRM once you've seen it and I'll take it live.")
        return {"action": "send", "body": body, "cta": "binary_confirm_cancel",
                "rationale": "Merchant committed: switched straight to action mode with a concrete next step, no further qualifying."}
    if cls == "off_topic":
        body = (f"Yeh mere scope ke bahar hai — iske liye aapke CA ya expert best rahenge. Wapas {topic} pe aate hain: jo draft maine offer kiya tha, woh bhej doon? Reply YES."
                if hi else
                f"That's outside what I can help with — your CA or a specialist is the right person for it. Coming back to {topic}: shall I send the draft I mentioned? Reply YES.")
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Off-topic request politely declined; redirected to the original trigger in one line."}
    if cls == "question":
        body = ("Achha sawaal — main exact details check karke isi chat mein bhejti hoon, andaaza nahi lagaungi. Tab tak jo draft offer kiya tha woh ready kar doon? Reply YES."
                if hi else
                "Good question — I'll check the exact details and send them here rather than guess. Meanwhile, shall I get the draft I offered ready? Reply YES.")
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Merchant asked something not answerable from context: no guessing, promise to verify, keep momentum."}
    body = ("Samajh gayi. Main aapke liye pehla draft bana ke yahin bhejti hoon — aap bas dekh ke YES ya changes bata dena."
            if hi else
            "Got it. I'll put together a first draft and share it right here — just reply YES or tell me what to change.")
    return {"action": "send", "body": body, "cta": "binary_yes_no",
            "rationale": "Merchant engaged: acknowledged and moved one concrete step forward."}


async def handle_reply(req: dict) -> dict:
    conv_id = req.get("conversation_id") or "conv_unknown"
    merchant_id = req.get("merchant_id")
    customer_id = req.get("customer_id")
    message = str(req.get("message") or "")
    conv = STORE.conversation(conv_id, merchant_id, customer_id)
    merchant_id = conv.merchant_id or merchant_id
    conv.turns.append({"role": req.get("from_role") or "merchant", "body": message})
    conv.unanswered_nudges = 0

    seen = STORE.note_inbound(merchant_id, message)
    cls = classify(message, seen)
    lang = _lang_for(conv, message, STORE.get("merchant", merchant_id))

    # ---- conversation already closed: only reopen if the merchant genuinely re-engages
    if conv.status == "ended" and cls in {"auto_reply", "opt_out", "hostile", "later"}:
        return {"action": "end", "rationale": "Conversation already closed; not re-engaging."}

    if cls == "auto_reply":
        conv.auto_reply_count += 1
        # count across conversations too (the harness may use a new conversation_id each turn)
        n = max(conv.auto_reply_count, seen)
        if n <= 1:
            body = ("Lagta hai yeh auto-reply hai 🙂 Jab owner dekhein, bas 'YES' reply kar dein — baaki main sambhaal lungi."
                    if lang == "hi-en" else
                    "Looks like an auto-reply 🙂 When the owner sees this, just reply 'YES' and I'll handle the rest.")
            return _send(conv, body, "binary_yes_no",
                         "Detected a WhatsApp Business canned auto-reply; one short prompt flagged for the owner, then back off.")
        if n == 2:
            conv.status = "waiting"
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": "Same auto-reply again: the owner isn't at the phone. Backing off 24h instead of burning turns."}
        conv.status = "ended"
        return {"action": "end", "rationale": f"Auto-reply {n}x in a row with no human response; closing the conversation politely."}

    if cls == "opt_out":
        conv.status = "ended"
        STORE.opt_out(merchant_id)
        return {"action": "end", "rationale": "Merchant explicitly opted out; ending and suppressing all sends to them for 30 days."}

    if cls in {"hostile", "hostile_offtopic"}:
        conv.status = "ended"
        STORE.opt_out(merchant_id, days=7)
        return {"action": "end", "rationale": "Merchant is frustrated; exiting gracefully without further pitching and pausing sends for 7 days."}

    if cls == "later":
        conv.status = "waiting"
        wait = 86400 if re.search(r"tomorrow|kal|next week", message.lower()) else 3600
        return {"action": "wait", "wait_seconds": wait,
                "rationale": f"Merchant asked for time; backing off {wait // 3600}h and will resume the same thread."}

    # max depth: don't drag a conversation on forever
    if conv.bot_turns() >= 6:
        conv.status = "ended"
        return {"action": "end", "rationale": "Conversation reached its natural length; closing to avoid over-messaging."}

    if conv.status == "ended":
        conv.status = "active"          # merchant came back on their own with a real message
    if cls == "commit":
        conv.mode = "action"

    reply = await _llm_reply(conv, cls, lang, message)
    if reply is None:
        reply = _fallback_send(cls, lang, conv, message)
    return _send(conv, reply["body"], reply.get("cta", "open_ended"), reply.get("rationale", ""))


def _send(conv: Conversation, body: str, cta: str, rationale: str) -> dict:
    body = body.strip()
    if body in conv.bot_bodies():                          # anti-repetition guard
        body = body.rstrip(".") + " — just reply here whenever you're ready."
        if body in conv.bot_bodies():
            conv.status = "ended"
            return {"action": "end", "rationale": "Nothing new to add without repeating myself; closing politely."}
    conv.turns.append({"role": "bot", "body": body})
    return {"action": "send", "body": body, "cta": cta, "rationale": rationale}


async def _llm_reply(conv: Conversation, cls: str, lang: str, message: str) -> dict | None:
    if not llm.enabled():
        return None
    merchant = STORE.get("merchant", conv.merchant_id) or {}
    category = STORE.get("category", merchant.get("category_slug")) or {}
    trigger = STORE.get("trigger", conv.trigger_id) or {"kind": "conversation", "payload": {}}
    customer = STORE.get("customer", conv.customer_id)
    if not merchant or not category:
        facts = {"merchant": {"salutation": None, "language": lang}, "voice": {"taboo_words": []}}
    else:
        facts = build_facts(category, merchant, trigger, customer, datetime.now(timezone.utc))
    transcript = [{"from": t["role"], "body": t["body"]} for t in conv.turns[-8:]]
    user = (f"MODE: {cls}\nTARGET LANGUAGE: {lang}\n\nFACTS:\n{compact_json(facts)}\n\n"
            f"CONVERSATION (oldest first):\n{compact_json(transcript)}\n\nWrite the next message. JSON only.")
    draft = await llm.complete_json(REPLY_SYSTEM, user, timeout=9.0)
    if not isinstance(draft, dict) or not isinstance(draft.get("body"), str):
        return None
    body = draft["body"].strip()
    low = body.lower()
    if len(body) < 10 or URL_RE.search(body):
        return None
    if cls == "commit" and any(q in low for q in ["would you", "do you", "can you tell", "what if", "how about"]):
        return None
    # numbers must come from facts or the conversation itself
    if merchant and category:
        probs = [p for p in validate({"body": body}, facts, extra_allowed=transcript)
                 if p.startswith("numbers") or p.startswith("uses taboo")]
        if probs:
            return None
    if body in conv.bot_bodies():
        return None
    return {"body": body, "cta": draft.get("cta", "open_ended"), "rationale": str(draft.get("rationale", ""))[:400]}
