"""Re-run validate() on tests/out/submission.jsonl offline (no LLM): python tests/revalidate.py"""
import json, glob, sys
from datetime import datetime, timezone
sys.path.insert(0, ".")
from app.facts import build_facts
from app.composer import validate
D = "../challenge/expanded"
def ld(sub, k):
    return {(d := json.load(open(f))).get(k) or f.split("/")[-1][:-5]: d for f in glob.glob(f"{D}/{sub}/*.json")}
cats = {c["slug"]: c for c in ld("categories", "slug").values()}
M, C, T = ld("merchants", "merchant_id"), ld("customers", "customer_id"), ld("triggers", "id")
pairs = {p["test_id"]: p for p in json.load(open(f"{D}/test_pairs.json"))["pairs"]}
now = datetime(2026, 4, 26, 10, 35, tzinfo=timezone.utc)
for line in open("tests/out/submission.jsonl"):
    r = json.loads(line); t = T[pairs[r["test_id"]]["trigger_id"]]; m = M[t["merchant_id"]]
    f = build_facts(cats[m["category_slug"]], m, t, C.get(t.get("customer_id")), now)
    print(r["test_id"], validate({"body": r["body"], "cta": r["cta"]}, f) or "ok")
