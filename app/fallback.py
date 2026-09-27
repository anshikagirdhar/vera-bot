"""Deterministic, rule-based composer.

Used when no LLM key is set, when the free-tier rate limit is hit, or when an LLM
draft fails validation. Every template only uses fields that actually exist in the
fact sheet, so it can never hallucinate; if a field is missing the sentence that
needs it is simply dropped.
"""
from __future__ import annotations

import re
from typing import Callable

from .facts import humanize, inr, nice_date, parse_iso, pct, window_words


def _hi(facts: dict) -> bool:
    lang = (facts.get("customer") or {}).get("language") or facts["merchant"].get("language")
    return lang == "hi-en"


def _L(facts: dict, en: str, hi: str) -> str:
    return hi if _hi(facts) else en


def _greet(facts: dict) -> str:
    m = facts["merchant"]
    return m.get("salutation") or m.get("business_name") or "Hi"


_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _offer_days(title: str) -> set[int] | None:
    """Weekdays an offer title restricts itself to ('... (Tue-Thu)' -> {1,2,3}); None = any day."""
    low = title.lower()
    days: set[int] = set()
    for a, b in re.findall(r"\b(mon|tue|wed|thu|fri|sat|sun)\w*\s*[-–to ]+\s*(mon|tue|wed|thu|fri|sat|sun)\w*", low):
        i, j = _DAYS.index(a), _DAYS.index(b)
        days.update((i + k) % 7 for k in range((j - i) % 7 + 1))
    if not days:
        days.update(_DAYS.index(d) for d in re.findall(r"\b(mon|tue|wed|thu|fri|sat|sun)(?:day|sday|nesday|rsday|urday)?s?\b", low))
    if not days and re.search(r"\bweekdays?\b", low):
        days = {0, 1, 2, 3, 4}
    if not days and re.search(r"\bweekends?\b", low):
        days = {5, 6}
    return days or None


def _offer(facts: dict, today_only: bool = False) -> str | None:
    """First active offer; with today_only, skip offers not valid today (e.g. a Tue-Thu deal on a Sunday)."""
    offers = facts["merchant"].get("active_offers") or []
    if not today_only:
        return offers[0] if offers else None
    m = re.search(r"\((mon|tue|wed|thu|fri|sat|sun)", str(facts.get("today", "")).lower())
    today = _DAYS.index(m.group(1)) if m else None
    for o in offers:
        days = _offer_days(str(o))
        if today is None or days is None or today in days:
            return o
    return None


def _lc_first(s: str | None) -> str | None:
    """Lower-case a sentence's first letter when it's spliced mid-sentence (keeps 'CTR', 'I')."""
    if not s or len(s) < 2 or s[1].isupper():
        return s
    return s[0].lower() + s[1:]


def _join(*parts: str | None) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _metric_line(facts: dict) -> str | None:
    """One verifiable performance anchor vs peers."""
    d = facts.get("derived") or {}
    perf = facts["merchant"].get("performance_30d") or {}
    if d.get("ctr_vs_peer") == "below":
        return f"Your profile CTR is {d['ctr_pct']}% vs {d['peer_ctr_pct']}% for peers nearby."
    if perf.get("views") and perf.get("calls") is not None:
        return f"Last 30 days: {perf['views']:,} profile views, {perf['calls']} calls."
    return None


def _lapsed_line(facts: dict) -> str | None:
    agg = facts["merchant"].get("customer_aggregate") or {}
    if agg.get("lapsed_180d_plus"):
        return f"{agg['lapsed_180d_plus']} of your customers haven't returned in 6+ months."
    return None


def _draft(f: dict, *lines: str | None) -> str:
    """A ready-to-post draft built only from merchant fields, quoted inside the message,
    so the merchant approves finished work instead of an idea."""
    m = f["merchant"]
    name, loc = m.get("business_name"), m.get("locality")
    head = f"{name}, {loc}" if name and loc else (name or "")
    text = " ".join(x.strip() for x in (head + " —" if head else None, *lines) if x and x.strip())
    return f"“{text}”"


def _since(f: dict) -> str | None:
    y = f["merchant"].get("established_year")
    return f"Trusted locally since {y}." if y else None


def _svc(driver: str) -> str:
    """'kids yoga post' -> 'Kids yoga' (the driver names the post; the draft names the service)."""
    d = re.sub(r"\s*\b(post|posts|campaign|ad|ads|update)\b\s*$", "", driver.strip(), flags=re.I)
    return d[:1].upper() + d[1:]


def _book(f: dict) -> str:
    return {"restaurants": "Call or WhatsApp to order.", "pharmacies": "Call or WhatsApp us."}.get(
        f.get("category"), "Call or WhatsApp to book.")


def _post_cta(f: dict) -> str:
    return _L(f, "Reply YES and I'll post it on your Google profile today.",
              "YES reply karein, aaj hi aapki Google profile pe post kar dungi.")


def _res(body: str, cta: str, rationale: str) -> dict:
    return {"body": " ".join(body.split()), "cta": cta, "rationale": rationale}


def attach_draft(f: dict, kind: str) -> tuple[str, str] | None:
    """Verified draft + closing CTA that the composer appends to an LLM-written body.
    The LLM chooses WHAT to attach; the words inside the quotes always come from here."""
    if f.get("send_as") != "vera":
        return None
    if kind == "review_request":
        return (f"“Thank you for choosing {f['merchant'].get('business_name') or 'us'}! If we made your day, "
                "a 1-line Google review would mean a lot to us.”",
                _L(f, "Reply YES and I'll send it to you as a ready-to-forward message.",
                   "YES reply karein, main ise forward-ready message bana ke bhej dungi."))
    if kind == "public_reply":
        p = f["trigger"]["payload"] or {}
        theme = p.get("theme") or next((t.get("theme") for t in f["merchant"].get("review_themes") or []
                                        if t.get("sentiment") == "neg"), None)
        if theme:
            return (f"“Thank you for telling us about the {humanize(theme)}. We're looking into it and would love "
                    "to make it right — please WhatsApp us.”",
                    _L(f, "Reply YES and I'll post it under those reviews.",
                       "YES reply karein, main ise un reviews ke neeche post kar dungi."))
        kind = "post"
    if kind == "post":
        today_only = f["trigger"].get("kind") == "ipl_match_today"
        offer = _offer(f, today_only=today_only)
        return _draft(f, f"{offer}." if offer else _since(f), _book(f)), _post_cta(f)
    return None


# ================================================================ merchant-facing
def research_like(f: dict) -> dict:
    item = f.get("relevant_digest_item") or {}
    g = _greet(f)
    if not item:
        return generic_merchant(f)
    kind = item.get("kind")
    src = item.get("source")
    title = item.get("title", "")
    n = item.get("trial_n")
    agg = f["merchant"].get("customer_aggregate") or {}
    cohort = None
    if item.get("patient_segment") and "high_risk" in str(item.get("patient_segment")) and agg.get("high_risk_adult_count"):
        cohort = f"Relevant for your {agg['high_risk_adult_count']} high-risk adult patients."
    if kind == "compliance":
        payload = f["trigger"]["payload"]
        deadline = payload.get("deadline_iso")
        body = _join(
            f"{g}, compliance heads-up: {title}" + (f" ({src})." if src else "."),
            item.get("summary"),
            f"Deadline: {nice_date(deadline)}." if deadline and str(deadline) not in title else None,
            _L(f, "Want me to prep a 1-page audit checklist for your setup? Reply YES.",
               "Aapke setup ke liye 1-page audit checklist bana doon? Reply YES."))
        return _res(body, "binary_yes_no", f"Regulation trigger: surfaced {src or 'the circular'} with its deadline; "
                                           "loss-aversion (compliance risk) + effort externalisation (I prep the checklist).")
    if kind == "cde":
        dt = parse_iso(item.get("date"))
        when = dt.strftime("%a %d %b, %I:%M%p").replace(" 0", " ") if dt else None
        body = _join(
            f"{g}, {title}" + (f" — {when}" if when else "") + ".",
            f"{item['credits']} CDE credits." if item.get("credits") else None,
            item.get("summary"),
            item.get("actionable") + "." if item.get("actionable") else None,
            _L(f, "Want me to block your calendar and send the registration details? Reply YES.",
               "Calendar block karke registration details bhej doon? Reply YES."))
        return _res(body, "binary_yes_no", "CDE opportunity from the category digest; specific date/credits; single YES CTA.")
    if kind == "alert":
        return supply_alert(f)
    body = _join(
        f"{g}, {src + ' is out' if src else 'new item this week'} — {title}" + (f" ({n:,}-patient trial)." if isinstance(n, int) else "."),
        item.get("summary"), cohort,
        _L(f, "Want me to pull the abstract and draft a patient-friendly WhatsApp you can share? Reply YES.",
           "Abstract nikaal ke ek patient-friendly WhatsApp draft kar doon jo aap share kar sakein? Reply YES."))
    return _res(body, "binary_yes_no", f"Digest trigger: cited {src}; tied to the merchant's own patient base; "
                                       "curiosity + reciprocity (I draft the share).")


def supply_alert(f: dict) -> dict:
    p = f["trigger"]["payload"]
    item = f.get("relevant_digest_item") or {}
    g = _greet(f)
    batches = ", ".join(p.get("affected_batches") or [])
    body = _join(
        f"{g}, urgent: recall notice on {p.get('molecule', 'a molecule you stock')}" +
        (f" batches {batches}" if batches else "") + (f" by {p.get('manufacturer')}" if p.get("manufacturer") else "") + ".",
        item.get("summary"),
        _L(f, "Want me to draft the customer WhatsApp note and a replacement-pickup checklist? Reply YES.",
           "Customers ke liye WhatsApp note aur replacement-pickup checklist draft kar doon? Reply YES."))
    return _res(body, "binary_yes_no", "Supply/recall alert: exact batch numbers for verification; urgent; "
                                       "offer to take the work off the merchant.")


def _own_delta(f: dict, negative: bool) -> dict | None:
    """Build a perf payload from the merchant's own delta_7d when the trigger is thin."""
    d7 = f["merchant"].get("delta_7d") or {}
    best = None
    for k, v in d7.items():
        if not isinstance(v, (int, float)) or k.startswith("ctr"):
            continue
        if (negative and v < 0) or (not negative and v > 0):
            if best is None or abs(v) > abs(best[1]):
                best = (k.replace("_pct", ""), v)
    return {"metric": best[0], "delta_pct": best[1], "window": "7d"} if best else None


def perf_dip(f: dict) -> dict:
    p = f["trigger"]["payload"] or _own_delta(f, negative=True) or {}
    g = _greet(f)
    if not p.get("delta_pct"):
        return generic_merchant(f)
    metric = humanize(p.get("metric", "calls"))
    d = pct(p.get("delta_pct"))
    seasonal = p.get("is_expected_seasonal") or f["trigger"]["kind"] == "seasonal_perf_dip"
    offer = _offer(f)
    if seasonal:
        body = _join(
            f"{g}, your {metric} are down {d} this week" + (f" ({humanize(p.get('season_note'))})" if p.get("season_note") else "") + ".",
            "This is the expected seasonal lull, not a problem with your listing — so no need to spend on ads right now.",
            _members_line(f),
            _L(f, "Want me to draft a retention challenge for your existing members to carry you through the dip? Reply YES.",
               "Existing members ke liye ek retention challenge draft kar doon taaki dip mein bhi engagement bana rahe? Reply YES."))
        return _res(body, "binary_yes_no", "Seasonal dip: pre-empted anxiety with the 'expected' framing, redirected to retention.")
    base = p.get("merchant_own_usual_value", p.get("vs_baseline"))
    body = _join(
        f"{g}, your {metric} dropped {d} in the last {window_words(p.get('window', '7d'))}" +
        (f" (baseline {base})." if base is not None else "."),
        _metric_line(f),
        "A fresh Google post is usually the fastest fix — here's one ready:",
        _draft(f, f"{offer}." if offer else _since(f), _book(f)),
        _post_cta(f))
    return _res(body, "binary_yes_no", "Perf dip: exact drop + peer anchor (loss aversion), one concrete fix using their live offer.")


def _members_line(f: dict) -> str | None:
    agg = f["merchant"].get("customer_aggregate") or {}
    for k in ("active_members", "active_count", "total_active_members"):
        if agg.get(k):
            return f"Focus on keeping your {agg[k]} active members engaged."
    return None


def perf_spike(f: dict) -> dict:
    p = f["trigger"]["payload"] or _own_delta(f, negative=False) or {}
    g = _greet(f)
    if not p.get("delta_pct"):
        return generic_merchant(f)
    metric = humanize(p.get("metric", "views"))
    driver = humanize(p.get("likely_driver")) if p.get("likely_driver") else None
    body = _join(
        f"{g}, nice — your {metric} are up {pct(p.get('delta_pct'))} over the last {window_words(p.get('window', '7d'))}" +
        (f", most likely from your {driver}." if driver else "."),
        "Momentum like this fades in about a week unless you follow it up — here's a follow-up post ready:",
        _draft(f, (_svc(driver) + " — now taking bookings.") if driver and _svc(driver) else None,
               f"{_offer(f)}." if _offer(f) else None, _book(f)),
        _post_cta(f))
    return _res(body, "binary_yes_no", "Perf spike: named the metric + likely driver; urgency (momentum fades); one follow-up action.")


def milestone(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    if not p.get("value_now") and not p.get("milestone_value"):
        return generic_merchant(f)
    metric = humanize(p.get("metric", "reviews")).replace("review count", "reviews")
    now, target = p.get("value_now"), p.get("milestone_value")
    gap = (target - now) if isinstance(now, int) and isinstance(target, int) else None
    body = _join(
        f"{g}, you're at {now} {metric} — just {gap} away from {target}." if gap and gap > 0 else
        f"{g}, you just hit {target or now} {metric}!",
        "A short thank-you ask to this week's happy customers usually closes the gap in days. Here it is:",
        f"“Thank you for choosing {f['merchant'].get('business_name') or 'us'}! If we made your day, a 1-line Google review would mean a lot to us.”",
        _L(f, "Reply YES and I'll send it to you as a ready-to-forward message.",
           "YES reply karein, main ise forward-ready message bana ke bhej dungi."))
    return _res(body, "binary_yes_no", "Milestone: exact count + gap (goal-gradient effect); low-effort drafted ask.")


def review_theme(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    theme = humanize(p.get("theme")) if p.get("theme") else None
    if not theme:
        themes = f["merchant"].get("review_themes") or []
        neg = [t for t in themes if t.get("sentiment") == "neg"]
        if not neg:
            return generic_merchant(f)
        p = neg[0]
        theme = humanize(p.get("theme"))
    body = _join(
        f"{g}, {p.get('occurrences_30d', 'a few')} reviews in the last 30 days mention '{theme}'" +
        (f" — e.g. \"{p.get('common_quote')}\"." if p.get("common_quote") else "."),
        "Replying publicly usually turns this into a trust signal instead of a red flag. Suggested reply:",
        f"“Thank you for telling us about the {theme}. We're looking into it and would love to make it right — please WhatsApp us.”",
        _L(f, "Reply YES and I'll post it under those reviews.",
           "YES reply karein, main ise un reviews ke neeche post kar dungi."))
    return _res(body, "binary_yes_no", "Review theme: verbatim quote + count; loss aversion (reputation); drafted reply offered.")


def competitor(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    if not p.get("competitor_name"):
        return generic_merchant(f)
    offer = _offer(f)
    body = _join(
        f"{g}, a new {f['category'].rstrip('s')} opened {p.get('distance_km')} km away — {p['competitor_name']}" +
        (f", launching with '{p['their_offer']}'." if p.get("their_offer") else "."),
        f"Your live offer is '{offer}'." if offer else None,
        "Rather than a price war, highlighting what they can't copy (your reviews and experience) holds share better. Ready post:",
        _draft(f, _since(f), f"{offer}." if offer else None, _book(f)),
        _post_cta(f))
    return _res(body, "binary_yes_no", "Competitor opened: name/distance/offer from the trigger; contrarian advice (don't discount).")


def festival(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    if not p.get("festival"):
        return generic_merchant(f)
    offer = _offer(f)
    days = p.get("days_until")
    lead = f"{g}, {p['festival']} is on {nice_date(p.get('date'))}" + (f" — {days} days out." if days else ".")
    far = isinstance(days, int) and days > 60
    if far:
        advice = "Too early to spend on promotion, but the right time to lock the plan so you're ready before searches pick up."
    elif isinstance(days, int) and days > 30:
        advice = "Early planners book first, so a teaser now beats a rush later."
    else:
        advice = "This is the booking window — people decide in the next few days."
    body = _join(lead, advice,
                 f"Your '{offer}' can anchor a festive bundle." if offer and far else None,
                 _L(f, f"Want me to draft a simple {p['festival']} plan with dates for each step? Reply YES.",
                    f"{p['festival']} ke liye ek simple plan (har step ki date ke saath) draft kar doon? Reply YES.") if far else
                 "Ready post: " + _draft(f, f"{p['festival']} special: {offer}." if offer else f"Celebrate {p['festival']} with us.",
                                         "Book early — " + _book(f)[0].lower() + _book(f)[1:]) + " " + _post_cta(f))
    return _res(body, "binary_yes_no", "Festival trigger: date + days-until; timing advice; builds on their existing offer.")


def ipl(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    offer = _offer(f, today_only=True)
    t = parse_iso(p.get("match_time_iso"))
    when = t.strftime("%I:%M%p").lstrip("0").lower() if t else "tonight"
    match = p.get("match", "the match")
    if p.get("is_weeknight") is False:
        angle = "Weekend match nights usually pull people home, so dine-in dips — delivery is where the orders are."
        action = "delivery-only match-night push"
        post = _draft(f, f"{match} tonight? Order in and don't miss a ball.", f"{offer}." if offer else None)
    else:
        angle = "Weeknight matches bring groups out — a watch-party deal fills tables."
        action = "match-night dine-in offer"
        post = _draft(f, f"Watch {match} with us tonight — bring the gang.", f"{offer}." if offer else None)
    body = _join(f"{g}, {p.get('match', 'IPL match')} at {p.get('venue', 'the stadium')} today, {when}.", angle,
                 f"Worth a {action} — ready post:", post,
                 _L(f, "Reply YES and I'll post it as your Google update before the match.",
                    "YES reply karein, match se pehle Google update pe post kar dungi."))
    return _res(body, "binary_yes_no", "Local event: match details; weekday/weekend judgement; leverages their live offer.")


def renewal(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    perf = f["merchant"].get("performance_30d") or {}
    days = p.get("days_remaining") or (f["merchant"].get("subscription") or {}).get("days_remaining")
    body = _join(
        f"{g}, your {p.get('plan', 'Vera')} plan renews in {days} days" +
        (f" ({inr(p['renewal_amount'])})." if p.get("renewal_amount") else "."),
        f"In the last 30 days it brought you {perf['views']:,} profile views and {perf['calls']} calls." if perf.get("views") else None,
        _L(f, "Want me to lock in the renewal so there's no gap in visibility? Reply YES.",
           "Renewal abhi lock kar doon taaki visibility mein gap na aaye? Reply YES."))
    return _res(body, "binary_yes_no", "Renewal: days left + amount; value recap from their own numbers; loss aversion.")


def winback(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    body = _join(
        f"{g}, it's been {p.get('days_since_expiry')} days since your plan lapsed" + (
            f", and your views are down {pct(p.get('perf_dip_pct'))}." if p.get("perf_dip_pct") else "."),
        f"{p['lapsed_customers_added_since_expiry']} customers have lapsed since then." if p.get("lapsed_customers_added_since_expiry") else None,
        _L(f, "Want me to reactivate and run a win-back message to them this week? Reply YES.",
           "Plan reactivate karke is hafte unhe win-back message bhej doon? Reply YES."))
    return _res(body, "binary_yes_no", "Win-back: concrete loss since expiry (views, lapsed customers); single YES.")


def dormant(f: dict) -> dict:
    g = _greet(f)
    anchor = _metric_line(f) or _lapsed_line(f)
    body = _join(
        f"{g}, quick one —", _lc_first(anchor) or "I was looking at your profile this week.",
        _L(f, "I'll build this month's plan around your answer: what do you want more of — calls, walk-ins or repeat customers?",
           "Aapke jawab ke hisaab se is mahine ka plan bana deti hoon — sabse zyada kya chahiye: calls, walk-ins ya repeat customers?"))
    return _res(body, "open_ended", "Dormant merchant: one fresh data point (reciprocity) + an easy asking-the-merchant question.")


def curious_ask(f: dict) -> dict:
    g = _greet(f)
    offer = _offer(f)
    body = _join(
        f"{g}, quick question —",
        _L(f, "which service are customers asking about most this week?",
           "is hafte customers sabse zyada kis service ke baare mein pooch rahe hain?"),
        (_L(f, f"If it's your '{offer}', I already have an angle for it.",
            f"Agar '{offer}' hai, toh uske liye mere paas ek angle ready hai.") if offer else None),
        _L(f, "Tell me and I'll turn it into a Google post plus a ready WhatsApp reply for price enquiries — takes 5 min.",
           "Batao, main use ek Google post aur price enquiries ke liye ready WhatsApp reply bana dungi — 5 min ka kaam."))
    return _res(body, "open_ended", "Curious-ask cadence: low-stakes question + reciprocity (I create the post).")


def gbp_unverified(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    body = _join(
        f"{g}, your Google profile is still unverified",
        f"— verified listings typically see about {pct(p['estimated_uplift_pct'])} more visibility." if p.get("estimated_uplift_pct") else ".",
        f"Verification is via {humanize(p['verification_path']).replace(' or ', ' or a ')}." if p.get("verification_path") else None,
        _L(f, "Want me to start it and walk you through the one step you need to do? Reply YES.",
           "Main process start kar doon aur aapko sirf ek step batana hoga? Reply YES."))
    return _res(body, "binary_yes_no", "GBP unverified: uplift estimate from trigger; effort externalisation.")


def planning(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    topic = humanize(p.get("intent_topic", "the plan"))
    offer = _offer(f)
    body = _join(
        f"{g}, here's a starter for the {topic} — edit anything:",
        f"1) Anchor it on your existing '{offer}' so pricing stays familiar." if offer else "1) One clear service + price, no discount maths.",
        "2) A fixed weekly slot so it's easy to commit to.",
        "3) Announce it — ready post:",
        _draft(f, f"Now launching: {topic}.", _book(f)),
        _L(f, "Reply YES and I'll post it and send the same note to your regulars.",
           "YES reply karein, main post kar dungi aur yahi note aapke regulars ko bhej dungi."))
    return _res(body, "binary_yes_no", "Active planning intent: merchant already said yes, so I move straight to a draft (no re-qualifying).")


def category_seasonal(f: dict) -> dict:
    p = f["trigger"]["payload"]
    g = _greet(f)
    trends = p.get("trends") or []
    pretty, rising = [], []
    for t in trends[:4]:
        parts = str(t).rsplit("_", 1)
        name = humanize(parts[0]).replace(" demand", "")
        pretty.append(f"{name} {parts[1]}%" if len(parts) == 2 else humanize(t))
        if len(parts) == 2 and parts[1].startswith("+"):
            rising.append(name[:1].upper() + name[1:] if name.islower() and len(name) > 3 else name)
    if not pretty:
        return generic_merchant(f)
    body = _join(
        f"{g}, {humanize(p.get('season', 'this season'))} demand shift in your category: " + ", ".join(pretty) + ".",
        "Moving the rising items to eye level and the counter usually lifts basket size within a week. Ready post:",
        _draft(f, (", ".join(rising) + " — in stock now.") if rising else None, _book(f)),
        _post_cta(f))
    return _res(body, "binary_yes_no", "Seasonal category shift: exact demand deltas from the trigger; concrete shelf action.")


THIN_LEADS = {
    "competitor_opened": "a new competitor has just listed near {loc} on Google, so nearby searchers now have one more option.",
    "festival_upcoming": "the festive season is coming up, and this is when people start planning where to go.",
    "milestone_reached": "your profile just crossed a milestone worth using.",
    "perf_dip": "your numbers softened this week.",
    "perf_spike": "your profile picked up this week.",
    "review_theme_emerged": "a pattern is showing up in your recent reviews.",
    "renewal_due": "your plan is due for renewal soon.",
    "research_digest": "this week's category digest has something relevant for you.",
}


def generic_merchant(f: dict) -> dict:
    g = _greet(f)
    kind = humanize(f["trigger"]["kind"])
    anchor = _metric_line(f) or _lapsed_line(f)
    offer = _offer(f)
    lead = THIN_LEADS.get(f["trigger"]["kind"], "a quick update on your profile.").format(
        loc=f["merchant"].get("locality") or "you")
    body = _join(
        f"{g}, {lead}", anchor,
        "A fresh Google post is the quickest lever right now — here's one ready:",
        _draft(f, f"{offer}." if offer else _since(f), _book(f)),
        _post_cta(f))
    return _res(body, "binary_yes_no", f"{kind} trigger with no detailed payload: anchored on the merchant's own "
                                       "metrics instead of inventing event details.")


# ================================================================ customer-facing
def _cust(f: dict) -> dict:
    return f.get("customer") or {}


def _cust_greet(f: dict) -> str:
    name = _cust(f).get("name")
    return f"Hi {name}" if name else "Hi"


def _from(f: dict) -> str:
    return f"{f['merchant'].get('business_name')} here"


def customer_recall(f: dict) -> dict:
    p = f["trigger"]["payload"]
    c = _cust(f)
    rel = c.get("relationship") or {}
    slots = [s.get("label") for s in p.get("available_slots") or [] if s.get("label")]
    offer = _offer(f)
    service = humanize(p.get("service_due")) if p.get("service_due") else None
    last = p.get("last_service_date") or rel.get("last_visit")
    body = _join(
        f"{_cust_greet(f)}, {_from(f)}.",
        f"Your {service} is due" + (f" (last visit {nice_date(last)})." if last else ".") if service else
        (f"It's been a while since your last visit on {nice_date(last)}." if last else "It's time for your routine check-in."),
        (_L(f, "Two slots are open: ", "Aapke liye 2 slots ready hain: ") + " or ".join(slots) + ".") if slots else None,
        f"{offer} applies." if offer else None,
        ("Reply 1 or 2 to book, or tell us a time that works." if len(slots) >= 2 else
         _L(f, "Reply YES and we'll book you in.", "Reply YES karein, hum slot book kar denge.")))
    return _res(body, "multi_choice_slot" if len(slots) >= 2 else "binary_yes_no",
                "Customer recall: real slots + live offer from merchant data; language preference honoured.")


def customer_lapsed(f: dict) -> dict:
    p = f["trigger"]["payload"]
    c = _cust(f)
    rel = c.get("relationship") or {}
    offer = _offer(f)
    days = p.get("days_since_last_visit")
    focus = humanize(p.get("previous_focus")) if p.get("previous_focus") else None
    body = _join(
        f"{_cust_greet(f)}, {_from(f)}.",
        f"It's been about {days} days since we last saw you — no pressure at all." if days else
        (f"We haven't seen you since {nice_date(rel['last_visit'])} — no pressure at all." if rel.get("last_visit") else "It's been a while — no pressure at all."),
        f"Whenever you're ready to get back to your {focus} goal, we'd love to help." if focus else None,
        f"Right now we're running: {offer}." if offer else None,
        _L(f, "Want us to hold a slot for you this week? Reply YES — no commitment.",
           "Is hafte aapke liye ek slot hold kar dein? Reply YES — koi commitment nahi."))
    return _res(body, "binary_yes_no", "Customer win-back: no-guilt tone, their own history, live offer, zero-commitment YES.")


def customer_appointment(f: dict) -> dict:
    p = f["trigger"]["payload"]
    label = p.get("slot_label") or p.get("time_label")
    body = _join(
        f"{_cust_greet(f)}, {_from(f)}.",
        "Reminder: your appointment is tomorrow" + (f", {label}." if label else "."),
        _L(f, "Reply YES to confirm, or tell us if you need to reschedule.",
           "Confirm karne ke liye YES reply karein, ya reschedule karna ho toh batayein."))
    return _res(body, "binary_yes_no", "Appointment reminder: confirm/reschedule in one reply; no invented time if not provided.")


def customer_refill(f: dict) -> dict:
    p = f["trigger"]["payload"]
    mols = p.get("molecule_list") or []
    if not mols or f["category"] != "pharmacies":
        return customer_generic(f)
    out = parse_iso(p.get("stock_runs_out_iso"))
    name = _cust(f).get("name")
    hello = f"Namaste {name}" if name else "Namaste"
    body = _join(
        _L(f, f"{hello}, {_from(f)}.", f"{hello} — {f['merchant'].get('business_name')} yahan."),
        _L(f, f"The monthly medicines ({', '.join(mols)}) are due to run out on {out.strftime('%d %b') if out else 'soon'}.",
           f"Monthly medicines ({', '.join(mols)}) {out.strftime('%d %b') if out else 'jaldi'} ko khatam hongi."),
        _L(f, "Same pack is ready.", "Same pack ready hai."),
        _L(f, "Home delivery to your saved address." if p.get("delivery_address_saved") else "",
           "Saved address pe home delivery." if p.get("delivery_address_saved") else ""),
        f"{_offer(f)} applies." if _offer(f) else None,
        _L(f, "Reply CONFIRM to dispatch.", "Dispatch ke liye CONFIRM reply karein."))
    return _res(body, "binary_confirm_cancel", "Chronic refill: exact molecules + run-out date; respectful tone; single CONFIRM.")


def customer_trial(f: dict) -> dict:
    p = f["trigger"]["payload"]
    opts = [o.get("label") for o in p.get("next_session_options") or [] if o.get("label")]
    body = _join(
        f"{_cust_greet(f)}, {_from(f)}.",
        f"Thanks for coming to the trial on {nice_date(p['trial_date'])}." if p.get("trial_date") else "Thanks for trying us out.",
        f"Next session: {opts[0]}." if opts else None,
        _L(f, "Want us to save your spot? Reply YES.", "Aapka spot save kar dein? Reply YES."))
    return _res(body, "binary_yes_no", "Trial follow-up: references the actual trial + the real next slot.")


def customer_wedding(f: dict) -> dict:
    p = f["trigger"]["payload"]
    body = _join(
        f"{_cust_greet(f)}, {_from(f)}.",
        f"{p['days_to_wedding']} days to your wedding" + (f" on {nice_date(p['wedding_date'])}" if p.get("wedding_date") else "") + " —"
        if p.get("days_to_wedding") else None,
        f"this is the right window to start your {humanize(p.get('next_step_window_open', 'prep plan'))}.",
        f"{_offer(f)} is available." if _offer(f) else None,
        _L(f, "Want us to block your first session next week? Reply YES.",
           "Agle hafte pehla session block kar dein? Reply YES."))
    return _res(body, "binary_yes_no", "Bridal follow-up: countdown from real wedding date; next-step window; single YES.")


def customer_generic(f: dict) -> dict:
    c = _cust(f)
    rel = c.get("relationship") or {}
    offer = _offer(f)
    body = _join(
        f"{_cust_greet(f)}, {_from(f)}.",
        f"Thanks for visiting us {rel['visits_total']} times!" if rel.get("visits_total") else None,
        f"We haven't seen you since {nice_date(rel['last_visit'])}." if rel.get("last_visit") and c.get("state") in {"lapsed_soft", "lapsed_hard", "churned"} else None,
        f"{offer} is on right now." if offer else None,
        _L(f, "Want us to book you in this week? Reply YES.", "Is hafte aapka slot book kar dein? Reply YES."))
    return _res(body, "binary_yes_no", "Customer message with thin trigger payload: grounded only in their visit history + live offer.")


DISPATCH: dict[str, Callable[[dict], dict]] = {
    "research_digest": research_like, "regulation_change": research_like, "cde_opportunity": research_like,
    "category_research_digest_release": research_like, "category_trend_movement": research_like,
    "supply_alert": supply_alert,
    "perf_dip": perf_dip, "seasonal_perf_dip": perf_dip, "perf_spike": perf_spike,
    "milestone_reached": milestone, "review_theme_emerged": review_theme,
    "competitor_opened": competitor, "festival_upcoming": festival, "ipl_match_today": ipl,
    "renewal_due": renewal, "winback_eligible": winback, "dormant_with_vera": dormant,
    "curious_ask_due": curious_ask, "scheduled_recurring": curious_ask, "gbp_unverified": gbp_unverified,
    "active_planning_intent": planning, "category_seasonal": category_seasonal,
    "recall_due": customer_recall, "customer_lapsed_soft": customer_lapsed, "customer_lapsed_hard": customer_lapsed,
    "appointment_tomorrow": customer_appointment, "chronic_refill_due": customer_refill,
    "trial_followup": customer_trial, "wedding_package_followup": customer_wedding, "bridal_followup": customer_wedding,
}


def compose_fallback(facts: dict) -> dict:
    kind = facts["trigger"]["kind"]
    fn = DISPATCH.get(kind)
    if fn is None:
        fn = customer_generic if facts.get("customer") else generic_merchant
    try:
        out = fn(facts)
    except Exception:  # a malformed payload must never crash the endpoint
        out = customer_generic(facts) if facts.get("customer") else generic_merchant(facts)
    if facts.get("customer") and fn in (generic_merchant,):
        out = customer_generic(facts)
    return out
