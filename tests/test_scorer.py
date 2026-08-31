"""End-to-end scorer test for the SUPERSEDED task (full-catalog NDCG@10/Recall@50).
Started with --task legacy: the scorer now defaults to the CURRENT task. The
current-task path is covered by test_official_metric / test_official_data.

Run against a live serve process over the socket.
The refusal cases are the point: they are what makes enforcement real."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Must precede the core.scorer import: LEDGER_PATH is resolved from the environment
# at import time. This suite used to unlink the shared runs/ledger.jsonl, i.e.
# running the tests destroyed whatever audit trail was there.
os.environ["TRACK2_LEDGER"] = str(ROOT / "runs" / "t" / "test_ledger.jsonl")
import numpy as np
import pandas as pd
from core import splits as S
from core.scorer import request, SOCKET_PATH, LEDGER_PATH
from core.ledger import verify

KEY = "a" * 64
R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  ' + extra) if extra else ''}")

# clean slate, then start the scorer as an INDEPENDENT process holding the key
LEDGER_PATH.unlink(missing_ok=True); SOCKET_PATH.unlink(missing_ok=True)
env = {**os.environ, "TRACK2_SIGNING_KEY": KEY}
env.pop("TRACK2_ORCHESTRATOR", None)
proc = subprocess.Popen([sys.executable, "-m", "core.scorer", "serve", "--task", "legacy"], cwd=ROOT, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for _ in range(120):
    if SOCKET_PATH.exists(): break
    if proc.poll() is not None: print(proc.stdout.read()); sys.exit("scorer died on startup")
    time.sleep(0.25)
check("scorer came up and is listening", SOCKET_PATH.exists())
check("ping responds with the manifest hash", request({"op": "ping"}).get("ok"))

# build a real popularity prediction file
tr, part = S.load_for_strategy()
view = S.load_for_scorer()
users = np.sort(view.users["search_val"]).astype(np.int64)
pop = tr.loc[tr.is_click == 1, "video_id"].value_counts().index.to_numpy()[:100].astype(np.int64)
tmp = ROOT / "runs" / "t"; tmp.mkdir(parents=True, exist_ok=True)

def w(name, u, it):
    p = tmp / name; np.savez_compressed(p, users=u, items=it); return str(p) + ".npz"

good = w("pop", users, np.tile(pop, (users.size, 1)))
r = request({"preds": good, "split": "search_val", "config_hash": "popcfg", "mask": "none"})
check("valid predictions accepted", r.get("ok"), r.get("error", ""))
pl = r["entry"]["payload"]
check("metrics returned", "ndcg@10" in pl["metrics"])
check("preds hash bound into the entry", len(pl["preds_sha256"]) == 64)
check("manifest hash bound into the entry", pl["manifest_sha256"] == view.manifest_sha256)
check("no flags on a sane baseline", pl["flags"] == [], str(pl["flags"]))
ref = pl["reference"]
check("actual beats the measured random floor",
      pl["metrics"]["recall@50"] > ref["random_ranking"]["recall@50"],
      f"actual={pl['metrics']['recall@50']:.5f} random={ref['random_ranking']['recall@50']:.5f}")
check("row_shifted == actual for popularity (identical lists, as expected)",
      abs(ref["row_shifted"]["recall@50"] - pl["metrics"]["recall@50"]) < 1e-9)

# ---- refusals ----
bad = request({"preds": w("short", users, np.tile(pop[:20], (users.size, 1))),
               "split": "search_val", "config_hash": "x", "mask": "none"})
check("N < 50 refused", not bad["ok"] and "N >= 50" in bad["error"], bad.get("error", ""))

bad = request({"preds": w("cov", users[:-5], np.tile(pop, (users.size - 5, 1))),
               "split": "search_val", "config_hash": "x", "mask": "none"})
check("missing users refused", not bad["ok"] and "coverage" in bad["error"])

dupe = np.tile(pop, (users.size, 1)); dupe[:, 1] = dupe[:, 0]
bad = request({"preds": w("dupe", users, dupe), "split": "search_val",
               "config_hash": "x", "mask": "none"})
check("duplicate items in a ranking refused", not bad["ok"] and "duplicate" in bad["error"])

# THE enforcement test: claim exclude_train_seen while including a seen item
seeded = np.tile(pop, (users.size, 1)).copy()
u0 = int(users[0]); seen0 = view.train_seen[u0]
# must not ALSO be a duplicate, or the structural check fires first
inject = next(int(x) for x in seen0 if int(x) not in set(pop.tolist()))
seeded[0, 0] = inject
bad = request({"preds": w("mask", users, seeded), "split": "search_val",
               "config_hash": "x", "mask": "exclude_train_seen"})
check("FALSELY claimed masking is REFUSED (enforced, not trusted)",
      not bad["ok"] and "violated" in bad["error"], bad.get("error", ""))

bad = request({"preds": good, "split": "test_quarantine", "config_hash": "x", "mask": "none"})
check("quarantined split not scoreable", not bad["ok"] and "non-scoreable" in bad["error"])

bad = request({"preds": str(tmp / "nope.npz"), "split": "search_val",
               "config_hash": "x", "mask": "none"})
check("missing file refused", not bad["ok"] and "not found" in bad["error"])

# honest masking path must work
EMPTY = np.array([], dtype=np.int64)
cat = view.catalog_items
def masked_row(u):
    seen = view.train_seen.get(int(u), EMPTY)
    row = np.setdiff1d(pop, seen)                 # popular items the user has not seen
    if row.size < 50:                             # pad from the wider catalog
        fill = np.setdiff1d(np.setdiff1d(cat, seen), row)
        row = np.concatenate([row, fill[: 50 - row.size]])
    return row[:50]
honest = np.stack([masked_row(u) for u in users])
assert honest.shape == (users.size, 50)
assert all(np.unique(r).size == 50 for r in honest[:200]), "padding introduced duplicates"
r2 = request({"preds": w("honest", users, honest), "split": "search_val",
              "config_hash": "honestcfg", "mask": "exclude_train_seen"})
check("honest masking accepted", r2.get("ok"), r2.get("error", ""))
if r2.get("ok"):
    p2 = r2["entry"]["payload"]
    check("masked run reports users whose relevance was fully masked",
          "n_users_all_relevant_masked" in p2,
          f"n={p2.get('n_users_all_relevant_masked')}")

# ledger integrity, and that the orchestrator cannot forge
v = verify(LEDGER_PATH, KEY.encode())
check("ledger verifies with the scorer's key", v.ok, str(v))
check("ledger verification fails under a wrong key",
      not verify(LEDGER_PATH, b"b" * 64).ok)
with LEDGER_PATH.open("a") as fh:
    fh.write(json.dumps({"seq": 999, "prev": "0" * 64,
                         "payload": {"kind": "score", "metrics": {"ndcg@10": 0.99}},
                         "sig": "ff" * 32}) + "\n")
check("orchestrator-style forged append is rejected", not verify(LEDGER_PATH, KEY.encode()).ok)

# ---- SUPERSET COVERAGE ---------------------------------------------------------
# An isolated strategy cannot be told which users are scoreable (that set is derived
# from validation labels), so it ranks every user and the scorer keeps the ones it
# scores. This asserts the two give BYTE-IDENTICAL metrics -- if they ever diverge,
# every number measured before isolation existed becomes incomparable to every
# number measured after it.
all_users = np.sort(pd.read_csv(S.DATA / "user_features_pure.csv",
                                usecols=["user_id"]).user_id.unique()).astype(np.int64)
sup = w("pop_superset", all_users, np.tile(pop, (all_users.size, 1)))
r_exact = request({"preds": good, "split": "search_val", "mask": "none",
                   "config_hash": "cov_exact"})
r_super = request({"preds": sup, "split": "search_val", "mask": "none",
                   "config_hash": "cov_superset", "isolation": "seatbelt"})
check("a superset of the scoreable users is accepted", r_super.get("ok"),
      str(r_super.get("error", ""))[:80])
if r_exact.get("ok") and r_super.get("ok"):
    pe, ps = r_exact["entry"]["payload"], r_super["entry"]["payload"]
    check("superset coverage scores IDENTICALLY to exact coverage",
          pe["metrics"] == ps["metrics"],
          f"{pe['metrics']} vs {ps['metrics']}")
    check("the payload records supplied vs scored separately",
          ps["n_users_supplied"] == all_users.size
          and ps["n_users_scored"] == pe["n_users_scored"],
          f"supplied={ps['n_users_supplied']:,} scored={ps['n_users_scored']:,}")
    check("isolation is recorded as CLAIMED, never as verified",
          ps["isolation_claimed"] == "seatbelt" and pe["isolation_claimed"] == "unspecified")

# PAIRING ACROSS DIFFERENT COVERAGE. The gate pairs per user, so an isolated run
# (every user) compared against an in-process run (scoreable users only) must align
# exactly, or every mixed comparison is silently wrong. Same rankings on both sides,
# so the paired delta must be EXACTLY zero -- not merely small.
mixed = request({"op": "compare", "a_preds": [good], "b_preds": [sup],
                 "split": "search_val", "mask": "none",
                 "min_effect": {"composite_mean": 0.002},
                 "a_label": "exact_coverage", "b_label": "superset_coverage"})
if mixed.get("ok"):
    cmx = mixed["entry"]["payload"]["paired"]["composite_mean"]
    check("pairing aligns exactly across different coverage",
          cmx["delta"] == 0.0 and cmx["p_two_sided"] == 1.0
          and cmx["frac_users_unchanged"] == 1.0,
          f"delta={cmx['delta']} p={cmx['p_two_sided']} n={cmx['n_users']:,}")
    check("both sides score the same user count",
          cmx["n_users"] == r_exact["entry"]["payload"]["n_users_scored"])
else:
    check("pairing aligns exactly across different coverage", False,
          str(mixed.get("error"))[:80])

# a MISSING scoreable user is still a refusal -- superset does not mean "anything goes"
short = w("pop_short", users[:-1], np.tile(pop, (users.size - 1, 1)))
rs = request({"preds": short, "split": "search_val", "mask": "none", "config_hash": "cov_short"})
check("a missing scoreable user is still refused",
      not rs.get("ok") and "coverage" in rs.get("error", ""), str(rs.get("error"))[:70])

# duplicate user ids must be refused: they would double-weight a user in the mean
dup_u = np.concatenate([users, users[:1]])
dup = w("pop_dupuser", dup_u, np.tile(pop, (dup_u.size, 1)))
rd = request({"preds": dup, "split": "search_val", "mask": "none", "config_hash": "cov_dup"})
check("duplicate user ids are refused",
      not rd.get("ok") and "duplicate user" in rd.get("error", ""), str(rd.get("error"))[:70])

# the anti-subprocess guard
g = subprocess.run([sys.executable, "-m", "core.scorer", "serve", "--task", "legacy"], cwd=ROOT,
                   env={**os.environ, "TRACK2_SIGNING_KEY": KEY, "TRACK2_ORCHESTRATOR": "1"},
                   capture_output=True, text=True, timeout=60)
check("refuses to start when launched by the orchestrator",
      g.returncode != 0 and "voids the trust model" in (g.stdout + g.stderr))

proc.terminate(); proc.wait(timeout=15)
print()
bad_t = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad_t)}/{len(R)} passed" + (f"  FAILURES: {bad_t}" if bad_t else ""))
sys.exit(1 if bad_t else 0)
