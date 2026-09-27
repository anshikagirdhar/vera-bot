# Vera bot — magicpin AI Challenge

A stateful HTTP bot that composes grounded, category-aware WhatsApp messages from the four context layers (category, merchant, trigger, customer), then handles the conversation after that.

## The idea in one line

**Vera does the work before it asks.** Every message follows one pattern: *why now → one real number → the work, already done → reply YES.* The merchant approves a finished Google post, review request or reply, not an idea. Nothing is invented: every number, price, day and duration is checked against the merchant's own data.

## Approach

```
context push ──► versioned in-memory store ──► background compose job (per trigger)
                                                   │
      build_facts()  →  LLM draft (per-kind playbook, temp 0, JSON)  →  validator  →  repair retry
                                                   │                        │ fails
                                                   ▼                        ▼
/v1/tick  ◄── collect finished jobs within 8s ◄── message          rule-based composer
```

1. **Grounded fact sheet.** Each compose starts from a compact fact sheet built only from the pushed contexts: the salutation, the merchant's 30-day numbers and 7-day deltas, live offers, peer benchmarks, the digest item the trigger points to, seasonal beats for the current month, and the customer's history and consent. The model never sees anything else.
2. **Per-trigger playbooks.** There are 25 trigger kinds, and each gets a short instruction for what "good" means. For example, a seasonal dip means "reframe, don't panic"; `active_planning_intent` means "the merchant already said yes, so deliver the draft". Unknown kinds get a generic "why now + one fact + one step" playbook.
3. **Validator.** Every number in the draft must appear in the fact sheet, which is how fabrication is detected. It also rejects: prices that aren't in the merchant's own offers or the trigger; durations ("5 months since your visit") that don't match the real dates; weekdays or "this weekend" the data doesn't support; a "Dr." for non-doctors; customer messages that leak business metrics or omit the business name; and any message whose last sentence isn't the single call to action. It also blocks taboo words and URLs. A failing draft gets one repair retry with the problems listed. If that also fails, the bot uses a **deterministic rule-based composer**, which only writes sentences whose fields exist.
4. **Thin or placeholder payloads.** When the trigger carries no detail, the bot anchors on the merchant's own data (7-day deltas, CTR vs peers, lapsed customers) and never makes up event specifics.
5. **Finished work, verified.** The LLM decides *what* to attach (`post`, `review_request`, `public_reply` or nothing); the bot builds the quoted draft itself from verified fields (business name, locality, year established, live offer valid today) and appends the "Reply YES" CTA. So the model can't invent an offer or detail inside a draft, and any quote it writes itself is rejected unless copied verbatim from the context. The rule-based composer follows the same pattern.
6. **Latency.** Composition starts the moment a trigger or merchant update is pushed, so `/v1/tick` only collects results. Anything not ready within the budget falls back to the rule-based composer, so the bot never times out.
7. **Replies.** A deterministic router runs before any LLM call:
   - **Auto-reply:** canned-phrase detection plus verbatim repeats across conversations. The first gets one nudge for the owner, the second a 24h wait, the third ends the conversation.
   - **Opt-out or hostile:** end the conversation and suppress sends to that merchant.
   - **Commitment** ("ok let's do it", "haan kar do"): switch to action mode with no qualifying questions.
   - **"Later":** wait.
   - **Off-topic** (GST, loans): decline politely and redirect.

   The language is re-detected on every turn (Hinglish vs English). An anti-repetition guard stops the bot sending the same body twice in a conversation.
   When a merchant asks "what exactly will you do?" or "how long will it take?", Vera answers with the actual draft and the steps ("It's ready now"), not a promise to check.
8. **Restraint.** No re-sends of the same `suppression_key`, at most one action per (merchant, conversation) per tick, and no customer sends without a customer context or with `reminder_opt_in: false`.

## Model choice

**`openai/gpt-oss-120b` on Groq (free tier)** at temperature 0 with low reasoning effort, a fixed seed and an in-process response cache. Identical inputs return identical outputs within a run. Groq's free tier allows 8K tokens/minute and 200K tokens/day, and each compose uses about 2.7K. So the limiter budgets **tokens** (7K/min, corrected with the real usage Groq reports) as well as requests, and on a 429 it pauses for exactly the "try again in" time instead of retrying. **AI triage:** when capacity is scarce, the most valuable messages get the model first (compliance, research, competitor, planning), ranked by kind and urgency; routine reminders (appointments, refills, recalls with real slots) always use templates, which already do them well. Anything the model can't reach in time goes to the rule-based composer, so the bot never times out. Gemini, Anthropic, other OpenAI-compatible APIs and a no-LLM mode can be switched in with `LLM_PROVIDER` and `LLM_MODEL`.

## Tradeoffs

- The fact-check is strict: it rejects any number not in context, including correct arithmetic such as totals. I chose that because fabrication is penalised far more than a missing calculation.
- Rule-based reply routing is predictable and fast, but a keyword router can misread unusual phrasing. The LLM only writes the wording after routing.
- State lives in memory with a single worker, as the brief allows. A restart during the test would lose pushed context.
- When the judge lists a trigger in `available_triggers`, the bot treats it as active even if `expires_at` has passed, because the judge's simulated clock is the source of truth.

## What extra context would help most

Real appointment slots per merchant, which services each merchant actually offers beyond their promo offers, customer-level consent scopes mapped to trigger kinds, and a record of what was already sent to each merchant in the last 7 days to plan the cadence.

## Run locally

```bash
pip install -r requirements.txt
export LLM_PROVIDER=groq LLM_API_KEY=<your Groq key>   # or LLM_PROVIDER=none for rule-based only
uvicorn app.main:app --port 8080
python tests/local_harness.py --bot http://localhost:8080 --data <path to expanded dataset>
```
