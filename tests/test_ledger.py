"""Adversarial tests for the ledger. These are the trust guarantee."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ledger import Ledger, verify, canonical, read_payloads

KEY = b"test-signing-key-not-the-real-one"
WRONG = b"attacker-key"

def _fresh(n=3):
    d = Path(tempfile.mkdtemp()); p = d / "ledger.jsonl"
    L = Ledger(p, KEY)
    for i in range(n):
        L.append({"kind": "score", "i": i, "ndcg@10": 0.0301 + i / 1000, "note": "ünïcode"})
    return p

def _lines(p): return [l for l in p.read_text().split("\n") if l.strip()]
def _write(p, lines): p.write_text("\n".join(lines) + "\n")

R = []
def check(name, cond):
    R.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")

# 1 happy path
p = _fresh(3); v = verify(p, KEY)
check("clean ledger verifies", v.ok and v.n_verified == 3 and not v.incomplete_tail)

# 2 resume after reopen keeps the chain intact
L2 = Ledger(p, KEY); L2.append({"kind": "note", "msg": "resumed"})
v = verify(p, KEY); check("resume preserves chain", v.ok and v.n_verified == 4)

# 3 payload tampering in the middle
p = _fresh(3); ls = _lines(p)
e = json.loads(ls[1]); e["payload"]["ndcg@10"] = 0.9999; ls[1] = canonical(e).decode()
_write(p, ls); v = verify(p, KEY)
check("edited interior payload -> bad signature", (not v.ok) and v.bad_seq == 1)

# 4 reordering
p = _fresh(3); ls = _lines(p); ls[1], ls[2] = ls[2], ls[1]
_write(p, ls); check("reordered entries -> rejected", not verify(p, KEY).ok)

# 5 interior deletion
p = _fresh(3); ls = _lines(p); del ls[1]
_write(p, ls); v = verify(p, KEY)
check("deleted interior entry -> sequence gap", (not v.ok) and "sequence gap" in (v.error or ""))

# 6 truncated final line (crash / disk full) must be TOLERATED
p = _fresh(3); txt = p.read_text(); p.write_text(txt[: -25])
v = verify(p, KEY)
check("truncated tail tolerated, not called fraud", v.ok and v.incomplete_tail and v.n_verified == 2)

# 7 forged append by a process without the key
p = _fresh(2)
forged = {"seq": 2, "prev": "0" * 64, "payload": {"kind": "score", "ndcg@10": 0.99}, "sig": "de" * 32}
with p.open("a") as fh: fh.write(canonical(forged).decode() + "\n")
check("unsigned forged append -> rejected", not verify(p, KEY).ok)

# 8 wrong key
p = _fresh(2); check("wrong key -> rejected", not verify(p, WRONG).ok)

# 9 float + unicode canonicalisation is stable through a round trip
vals = [0.0, 1.0, 0.1, 1e-17, 0.030100000000000002, 1 / 3, 1e308, -0.0]
s = canonical({"v": vals, "u": "ünïcode ✓"})
check("canonical form round-trips exactly", canonical(json.loads(s)) == s)

# 10 empty key refused
try:
    Ledger(Path(tempfile.mkdtemp()) / "x.jsonl", b""); check("empty key refused", False)
except ValueError: check("empty key refused", True)

# 11 NaN/Inf refused rather than silently written as invalid JSON
try:
    canonical({"x": float("nan")}); check("NaN refused by canonical", False)
except ValueError: check("NaN refused by canonical", True)

# 12 DOCUMENTED LIMIT: truncating COMPLETE trailing entries is undetectable
#    without an external anchor. Assert the known behaviour so it stays known.
p = _fresh(4); ls = _lines(p); _write(p, ls[:2])
v = verify(p, KEY)
check("[known limit] whole-entry truncation verifies as a shorter chain",
      v.ok and v.n_verified == 2)

# 13 read_payloads stops at the first incomplete line
p = _fresh(3); txt = p.read_text(); p.write_text(txt[:-25])
check("read_payloads ignores truncated tail", len(read_payloads(p)) == 2)

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
