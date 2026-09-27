#!/usr/bin/env python3
"""Try the bot yourself: send one situation, see the message, reply as the owner.

  python tests/try_it.py --list                           # show the situations you can try
  python tests/try_it.py                                  # default: Dr. Bharat's calls dropped
  python tests/try_it.py --trigger trg_010_ipl_match_delhi --reply "what exactly will you do?"

Wipes the bot's memory before and after (POST /v1/teardown), so only use it
BEFORE the real evaluation starts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request as rq, error as er

LIVE = "https://vera-bot-77pe.onrender.com"


def call(bot, method, path, body=None):
    req = rq.Request(bot + path, data=json.dumps(body).encode() if body is not None else None,
                     method=method, headers={"Content-Type": "application/json"})
    try:
        return json.loads(rq.urlopen(req, timeout=90).read() or b"{}")
    except er.HTTPError as e:
        return json.loads(e.read() or b"{}")


def load(d: Path, sub: str, key: str):
    out = {}
    for f in sorted((d / sub).glob("*.json")):
        x = json.load(open(f))
        out[x.get(key) or f.stem] = x
    return out


def push(bot, scope, cid, payload):
    call(bot, "POST", "/v1/context", {"scope": scope, "context_id": cid, "version": 1,
                                      "payload": payload, "delivered_at": "2026-04-26T10:00:00Z"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default=LIVE)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "challenge" / "expanded"))
    ap.add_argument("--trigger", default="trg_004_perf_dip_bharat")
    ap.add_argument("--reply", default="what exactly will you do?")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    d, bot = Path(a.data), a.bot.rstrip("/")
    triggers = load(d, "triggers", "id")
    if a.list:
        for tid, t in triggers.items():
            print(f"{tid:55s} {t.get('kind')}")
        return
    t = triggers[a.trigger]
    merchant = load(d, "merchants", "merchant_id")[t["merchant_id"]]
    category = load(d, "categories", "slug")[merchant["category_slug"]]
    customer = load(d, "customers", "customer_id").get(t.get("customer_id")) if t.get("customer_id") else None

    print(f"Bot: {bot}   (waking it up can take ~30s)")
    print("Health:", call(bot, "GET", "/v1/healthz").get("status"))
    call(bot, "POST", "/v1/teardown")

    print(f"\n1) SENDING THE FACTS")
    print(f"   shop:     {merchant['identity']['name']} ({merchant['category_slug']}, {merchant['identity'].get('locality')})")
    print(f"   reason:   {t['kind']}  {json.dumps(t.get('payload'), ensure_ascii=False)[:150]}")
    push(bot, "category", category["slug"], category)
    push(bot, "merchant", merchant["merchant_id"], merchant)
    if customer:
        print(f"   customer: {customer['identity'].get('name')}")
        push(bot, "customer", customer["customer_id"], customer)
    push(bot, "trigger", t["id"], t)

    print("\n2) ASKING: anything to send?   (up to ~10s)")
    acts = call(bot, "POST", "/v1/tick", {"now": "2026-04-26T10:35:00Z", "available_triggers": [t["id"]]}).get("actions", [])
    if not acts:
        print("   Vera chose not to send anything (restraint).")
    else:
        m = acts[0]
        print(f"   send as:  {m['send_as']}")
        print(f"   MESSAGE:  {m['body']}")
        print(f"   why:      {m['rationale']}")
        print(f"\n3) YOU REPLY AS THE OWNER: \"{a.reply}\"")
        r = call(bot, "POST", "/v1/reply", {"conversation_id": m["conversation_id"], "merchant_id": m["merchant_id"],
                                            "customer_id": m["customer_id"], "from_role": "merchant",
                                            "message": a.reply, "received_at": "2026-04-26T10:40:00Z", "turn_number": 2})
        print(f"   Vera does: {r.get('action')}")
        if r.get("body"):
            print(f"   VERA:      {r['body']}")

    call(bot, "POST", "/v1/teardown")
    print("\n(bot memory wiped again, ready for the real judge)")


if __name__ == "__main__":
    main()
