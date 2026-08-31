"""Stage-2 diagnostics: correct numbers, and incapable of becoming a label oracle.

The safety property is the one worth testing hardest. A diagnostic that could report on
the held-out split would be a metric the agent could query directly, which is precisely
what the scorer exists to prevent.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from agent import inspect as INS
from core.task import get
from core.official_data import load_for_strategy, load_for_scorer

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

spec = get("kuairand_pure")
sp, _ = load_for_strategy()

# 1 -- SAFETY: a diagnostic must not be runnable where held-out labels exist ------
full, _ = load_for_scorer()
try:
    INS.run("label_rates", full, spec)
    check("refuses to run in a process holding held-out labels", False)
except RuntimeError as e:
    check("refuses to run in a process holding held-out labels", "oracle" in str(e))
try:
    INS.run("nope", sp, spec); check("unknown diagnostic raises rather than guessing", False)
except KeyError:
    check("unknown diagnostic raises rather than guessing", True)

# no diagnostic may mention the held-out split at all
held = spec.split.held_out_split
for name in INS.DIAGNOSTICS:
    res = INS.run(name, sp, spec)
    flat = str(res)
    check(f"{name}: reports nothing about the held-out split",
          held not in flat and "test" not in {k.split(":")[0] for k in
                                              (res.get("by_date") or {})})

# 2 -- CORRECTNESS: numbers must match an independent computation -----------------
lr = INS.run("label_rates", sp, spec)
tr, dv = sp["train"], sp[spec.split.dev_split]
check("train rate matches a direct mean", abs(lr["train_rate"] - float(tr.label.mean())) < 1e-5,
      f"{lr['train_rate']}")
check("dev rate matches a direct mean", abs(lr["dev_rate"] - float(dv.label.mean())) < 1e-5,
      f"{lr['dev_rate']}")
check("shift is dev minus train", abs(lr["shift"] - (lr["dev_rate"] - lr["train_rate"])) < 1e-9)
check("by_date covers every date in both splits",
      len(lr["by_date"]) == len(np.unique(tr.date)) + len(np.unique(dv.date)))

gs = INS.run("group_sizes", spec=spec, splits=sp)
u, inv = np.unique(dv.user_id, return_inverse=True)
sizes = np.bincount(inv); pos = np.bincount(inv, weights=dv.label)
check("user count matches", gs["users"] == u.size)
check("mean impressions matches", abs(gs["impressions_per_user_mean"] - sizes.mean()) < 1e-3)
check("the three user classes sum to 100%",
      abs(gs["all_negative_users_pct"] + gs["all_positive_users_pct"]
          + gs["discriminative_users_pct"] - 100.0) < 0.05,
      f"{gs['all_negative_users_pct']}+{gs['all_positive_users_pct']}+{gs['discriminative_users_pct']}")
check("all-negative count matches a direct count",
      abs(gs["all_negative_users_pct"] - 100.0 * float(np.mean(pos == 0))) < 0.01)

ax = INS.run("aux_signals", sp, spec)
check("aux report covers the auxiliary signals present on train",
      set(ax["signals"]) == {k for k in INS.AUX if k in tr.aux})
ic = ax["signals"]["is_click"]
check("is_click's correlation with the label is reported and high",
      ic["corr_with_label"] > 0.5, f"{ic['corr_with_label']}")
check("and its conditional rates are reported",
      ic["P_label_given_1"] > ic["P_label_given_0"])
check("play_time is normalised by duration, not raw ms",
      "rate" not in ax["signals"]["play_time_ms"])

cov = INS.run("item_coverage", sp, spec)
# Tolerance is 1e-3 because the diagnostic stores round(pct, 3). Comparing a 3-dp value
# at 1e-6 can never pass and says nothing -- the same mistake as test_task's 1e-9 against
# a 6-dp reference.
_direct = 100.0 * float(np.mean(~np.isin(dv.video_id, np.unique(tr.video_id))))
check("unseen-item share matches a direct computation",
      abs(cov["dev_rows_with_unseen_item_pct"] - _direct) < 1e-3,
      f"{cov['dev_rows_with_unseen_item_pct']} vs {_direct:.6f}")
check("coverage percentages are in range",
      0 <= cov["dev_rows_with_unseen_item_pct"] <= 100
      and 0 <= cov["dev_rows_with_unseen_user_pct"] <= 100)

dp = INS.run("duration_profile", sp, spec)
check("duration profile has at most 10 deciles", len(dp["by_duration_decile"]) <= 10)
rates = [v["label_rate"] for v in dp["by_duration_decile"].values()]
check("spread equals max minus min label rate",
      abs(dp["spread"] - (max(rates) - min(rates))) < 1e-9)
check("every decile reports a row count and a rate",
      all({"rows", "label_rate", "median_duration_s"} <= set(v)
          for v in dp["by_duration_decile"].values()))

# 3 -- every diagnostic explains how to read it -----------------------------------
for name in INS.DIAGNOSTICS:
    res = INS.run(name, sp, spec)
    check(f"{name}: carries a 'reading' the agent can act on",
          isinstance(res.get("reading"), str) and len(res["reading"]) > 40)
    check(f"{name}: is JSON-serialisable for the ledger",
          __import__("json").dumps(res) is not None)

# 4 -- cheap enough to be worth offering ------------------------------------------
import time
t0 = time.perf_counter()
for name in INS.DIAGNOSTICS:
    INS.run(name, sp, spec)
el = time.perf_counter() - t0
check("all diagnostics together cost less than one model fit (~14s)", el < 5.0,
      f"{el:.2f}s for {len(INS.DIAGNOSTICS)}")

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
