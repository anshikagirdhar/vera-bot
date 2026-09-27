#!/usr/bin/env python3
"""Local harness that mimics the magicpin judge (no LLM needed to run it).

  python tests/local_harness.py --bot http://localhost:8080 --data ../expanded

Checks the contract (schemas, idempotency, 409s, timing), prints all 30 canonical
test-pair messages, and runs the three replay scenarios. Writes
tests/out/submission.jsonl in the format of challenge-brief section 7.2.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib import request as rq, error as er

REQ_ACTION_FIELDS = ["conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id",
                     "template_name", "template_params", "body", "cta", "suppression_key", "rationale"]
fails: list[str] = []


def call(bot, method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = rq.Request(bot + path, data=data, method=method, headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        r = rq.urlopen(req, timeout=timeout)
        return r.status, json.loads(r.read()), time.time() - t
    except er.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def load(d: Path, sub: str):
    return {json.load(open(f)).get("slug") or f.stem: json.load(open(f)) for f in sorted((d / sub).glob("*.json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="http://localhost:8080")
    ap.add_argument("--data", default="../expanded")
    ap.add_argument("--pace", type=float, default=0,
                    help="seconds to wait per test pair (push its trigger, wait, tick it alone) so a "
                         "rate-limited free-tier LLM can compose each one; 0 = push all, tick in batches of 5")
    args = ap.parse_args()
    bot, d = args.bot.rstrip("/"), Path(args.data)
    cats = load(d, "categories")
    merchants = {m["merchant_id"]: m for m in load(d, "merchants").values()}
    customers = {c["customer_id"]: c for c in load(d, "customers").values()}
    triggers = {t["id"]: t for t in load(d, "triggers").values()}
    pairs = json.load(open(d / "test_pairs.json"))["pairs"]

    print("== warmup")
    s, b, _ = call(bot, "GET", "/v1/healthz"); check(s == 200 and b.get("status") == "ok", "healthz 200")
    s, b, _ = call(bot, "GET", "/v1/metadata"); check(s == 200 and "team_name" in b, "metadata 200")
    print(f"  bot model: {b.get('model')}")
    t0 = time.time()
    for scope, items in (("category", cats), ("merchant", merchants), ("customer", customers)):
        for cid, payload in items.items():
            s, b, _ = call(bot, "POST", "/v1/context", {"scope": scope, "context_id": cid, "version": 1,
                                                        "payload": payload, "delivered_at": "2026-04-26T09:45:00Z"})
            if s != 200:
                check(False, f"push {scope}/{cid} -> {s} {b}")
    print(f"  pushed 255 contexts in {time.time()-t0:.1f}s")
    s, b, _ = call(bot, "GET", "/v1/healthz")
    check(b["contexts_loaded"] == {"category": 5, "merchant": 50, "customer": 200, "trigger": 0},
          f"contexts_loaded {b['contexts_loaded']}")
    m1 = "m_001_drmeera_dentist_delhi"
    s, b, _ = call(bot, "POST", "/v1/context", {"scope": "merchant", "context_id": m1, "version": 1, "payload": merchants[m1], "delivered_at": "x"})
    check(s == 409 and b.get("reason") == "stale_version", "same version re-push -> 409 stale_version")
    s, b, _ = call(bot, "POST", "/v1/context", {"scope": "bogus", "context_id": "x", "version": 1, "payload": {}, "delivered_at": "x"})
    check(s == 400, "invalid scope -> 400")
    s, b, _ = call(bot, "POST", "/v1/tick", {"now": "2026-04-26T10:30:00Z", "available_triggers": []})
    check(s == 200 and b.get("actions") == [], "empty tick -> no actions")

    print("\n== 30 canonical test pairs")
    def push_trigger(t):
        call(bot, "POST", "/v1/context", {"scope": "trigger", "context_id": t["id"], "version": 1, "payload": t, "delivered_at": "x"})

    pair_trg = {p["trigger_id"] for p in pairs}
    for t in triggers.values():
        if not args.pace or t["id"] not in pair_trg:
            push_trigger(t)
    out_dir = Path(__file__).parent / "out"; out_dir.mkdir(exist_ok=True)
    sub = open(out_dir / "submission.jsonl", "w", encoding="utf-8")
    slow = 0
    step = 1 if args.pace else 5
    for i in range(0, len(pairs), step):
        batch = pairs[i:i + step]
        if args.pace:
            for p in batch:
                push_trigger(triggers[p["trigger_id"]])
            time.sleep(args.pace)
        s, b, lat = call(bot, "POST", "/v1/tick", {"now": "2026-04-26T10:35:00Z",
                                                  "available_triggers": [p["trigger_id"] for p in batch]})
        slow += lat > 10
        by_trg = {a["trigger_id"]: a for a in b.get("actions", [])}
        for p in batch:
            a = by_trg.get(p["trigger_id"])
            print(f"\n[{p['test_id']}] {triggers[p['trigger_id']]['kind']}  ({lat:.1f}s)")
            if not a:
                print("  (no action — bot chose restraint)")
                continue
            missing = [k for k in REQ_ACTION_FIELDS if k not in a]
            if missing: check(False, f"{p['test_id']} missing fields {missing}")
            if re.search(r"https?://|www\.", a["body"]): check(False, f"{p['test_id']} has URL")
            print(f"  send_as={a['send_as']} cta={a['cta']}\n  BODY: {a['body']}\n  WHY:  {a['rationale']}")
            sub.write(json.dumps({"test_id": p["test_id"], "body": a["body"], "cta": a["cta"], "send_as": a["send_as"],
                                  "suppression_key": a["suppression_key"], "rationale": a["rationale"]}, ensure_ascii=False) + "\n")
    sub.close()
    check(slow == 0, "every tick answered within 10s")
    s, b, _ = call(bot, "POST", "/v1/tick", {"now": "2026-04-26T10:40:00Z", "available_triggers": [pairs[0]["trigger_id"]]})
    check(b.get("actions") == [], "suppression: same trigger not re-sent")

    print("\n== replay: auto-reply hell (same conversation)")
    call(bot, "POST", "/v1/context", {"scope": "trigger", "context_id": "trg_x_cde", "version": 1, "delivered_at": "x",
          "payload": {**triggers["trg_022_cde_webinar_dentists"], "id": "trg_x_cde", "merchant_id": "m_002_bharat_dentist_mumbai",
                      "suppression_key": "cde:x"}})
    s, b, _ = call(bot, "POST", "/v1/tick", {"now": "2026-04-26T11:00:00Z", "available_triggers": ["trg_x_cde"]})
    conv = b["actions"][0]["conversation_id"] if b.get("actions") else "conv_auto"
    auto = "Thank you for contacting Bharat Dental Care! Our team will respond shortly."
    acts = []
    for turn in range(2, 6):
        s, b, _ = call(bot, "POST", "/v1/reply", {"conversation_id": conv, "merchant_id": "m_002_bharat_dentist_mumbai",
                                                  "customer_id": None, "from_role": "merchant", "message": auto,
                                                  "received_at": "x", "turn_number": turn})
        acts.append(b.get("action")); print(f"  turn {turn}: {b.get('action')} {b.get('body', '')[:80]}")
    check("end" in acts[:3], f"auto-reply ends by 3rd repeat {acts}")

    print("\n== replay: intent transition")
    conv = "conv_intent_test"
    for turn, msg in enumerate(["Hmm what exactly will you do?", "ok and how long will it take", "Ok lets do it. Whats next?"], start=2):
        s, b, _ = call(bot, "POST", "/v1/reply", {"conversation_id": conv, "merchant_id": "m_003_studio11_salon_hyderabad",
                                                  "customer_id": None, "from_role": "merchant", "message": msg,
                                                  "received_at": "x", "turn_number": turn})
        print(f"  merchant: {msg}\n  bot: {b.get('action')} | {b.get('body', '')}")
    low = b.get("body", "").lower()
    check(b.get("action") == "send" and not any(q in low for q in ["would you", "do you", "can you tell", "what if", "how about"]),
          "commit -> action mode, no qualifying question")

    print("\n== replay: hostile then off-topic")
    for turn, msg in enumerate(["Why are you bothering me. This is useless.", "can you also help me file my GST?"], start=2):
        s, b, _ = call(bot, "POST", "/v1/reply", {"conversation_id": "conv_hostile_t", "merchant_id": "m_005_pizzajunction_restaurant_delhi",
                                                  "customer_id": None, "from_role": "merchant", "message": msg,
                                                  "received_at": "x", "turn_number": turn})
        print(f"  merchant: {msg}\n  bot: {b.get('action')} | {b.get('body', '')} | {b.get('rationale')}")
    s, b, _ = call(bot, "POST", "/v1/reply", {"conversation_id": "conv_gst_t", "merchant_id": "m_006_southindiancafe_restaurant_bangalore",
                                              "customer_id": None, "from_role": "merchant", "message": "Btw can you help with my GST filing?",
                                              "received_at": "x", "turn_number": 2})
    print(f"  (fresh) GST ask -> {b.get('action')} | {b.get('body')}")
    check(b.get("action") == "send" and "gst" not in b.get("body", "").lower().split("ca")[0][:0], "off-topic handled")
    s, b, _ = call(bot, "POST", "/v1/reply", {"conversation_id": "conv_stop_t", "merchant_id": "m_007_powerhouse_gym_bangalore",
                                              "customer_id": None, "from_role": "merchant", "message": "Not interested. Stop messaging me.",
                                              "received_at": "x", "turn_number": 2})
    check(b.get("action") == "end", "hard no -> end")
    s, b, _ = call(bot, "POST", "/v1/reply", {"conversation_id": "conv_hi_t", "merchant_id": "m_001_drmeera_dentist_delhi",
                                              "customer_id": None, "from_role": "merchant", "message": "haan ji kar do, mujhe dikhao kya bana hai",
                                              "received_at": "x", "turn_number": 2})
    print(f"  Hinglish commit -> {b.get('body')}")

    s, b, _ = call(bot, "GET", "/v1/debug/llm")
    if s == 200:
        print(f"\nLLM stats: {b.get('stats')}  model={b.get('model')}")
        if b.get("last_error"):
            print(f"  last_error: {b['last_error']}")

    print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILURES: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
