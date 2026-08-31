"""The official metric: GAUC + nDCG@5 over within-user rankings.

This suite exists because the metric is not ours. It is vendored verbatim from the
Starter Kit, and the only thing worth testing about vendored code is that it is still
verbatim and that our wrapper does not distort it. The conventions asserted below are
the three that move the number by tens of a percent when wrong.
"""
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.metrics import (primary, OFFICIAL_TEST, OFFICIAL_VALID,
                          OFFICIAL_FM_SEED_STD, OFFICIAL_EPSILON, OFFICIAL_N)
from core import official_evaluate as OE

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

KIT = ROOT.parent / "kuairand-starter-kit" / "evaluate.py"
MARKER = "# ---------------------------------------------------------------------------- vendored\n"

# 1 -- the vendored copy must still be byte-identical to the kit ------------------
if not KIT.exists():
    print(f"  [skip] kit not present at {KIT}; cannot verify vendoring")
else:
    body = Path(ROOT / "core/official_evaluate.py").read_text().split(MARKER, 1)[1]
    kit = KIT.read_text()
    check("vendored evaluate.py is BYTE-IDENTICAL to the kit", body == kit,
          f"sha {hashlib.sha256(body.encode()).hexdigest()[:12]}")
    check("we did not smuggle an edit in above the marker",
          "def evaluate" not in Path(ROOT / "core/official_evaluate.py").read_text().split(MARKER)[0])

# 2 -- our wrapper must not distort it -------------------------------------------
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(500):
    nu = int(rng.integers(1, 6))
    us, ls, ss = [], [], []
    for u in range(nu):
        n = int(rng.integers(1, 12))
        us += [f"u{u}"] * n
        ls += list(rng.integers(0, 2, size=n).astype(int))
        ss += list(rng.normal(size=n))
    a = primary(us, ls, ss)
    b = OE.evaluate(us, ls, ss, k=5)
    for key in ("GAUC", "nDCG@5", "primary"):
        worst = max(worst, abs(a[key] - b[key]))
check("primary() agrees with the official evaluate() on 500 random cases",
      worst == 0.0, f"max abs diff {worst:.1e}")

# 3 -- THE THREE TRAP CONVENTIONS ------------------------------------------------
# (a) zero-positive users score 0 and ARE counted in the nDCG mean.
z = primary(["a"] * 4, [0, 0, 0, 0], [4.0, 3.0, 2.0, 1.0])
check("a zero-positive user scores nDCG 0.0", z["nDCG@5"] == 0.0)
mixed = primary(["a", "a", "b", "b"], [0, 0, 1, 0], [1.0, 2.0, 2.0, 1.0])
check("zero-positive users are COUNTED in the nDCG mean (not dropped)",
      abs(mixed["nDCG@5"] - 0.5) < 1e-12,
      f"got {mixed['nDCG@5']:.4f}; 0.5 = mean(0.0, 1.0), dropping would give 1.0")

# (b) GAUC covers only users with 0 < npos < n, weighted by npos.
allpos = primary(["a"] * 3, [1, 1, 1], [3.0, 2.0, 1.0])
check("an all-positive user is excluded from GAUC (falls back to 0.5)",
      allpos["GAUC"] == 0.5, f"got {allpos['GAUC']}")
check("an all-positive user still scores nDCG 1.0", abs(allpos["nDCG@5"] - 1.0) < 1e-12)
# weighting: user A has 1 positive and is perfectly ranked; user B has 3 and is inverted
us = ["A"] * 4 + ["B"] * 6
ls = [1, 0, 0, 0] + [0, 0, 0, 1, 1, 1]
ss = [9.0, 3, 2, 1] + [9.0, 8, 7, 3, 2, 1]
w = primary(us, ls, ss)
check("GAUC is weighted by positive count, not by user",
      w["GAUC"] < 0.5, f"GAUC {w['GAUC']:.4f} -- B's 3 positives outweigh A's 1")

# (c) gain 2^rel - 1, discount log2(rank+1), ideal over the user's own list.
one_at_2 = primary(["a"] * 2, [0, 1], [2.0, 1.0])
import math
check("nDCG uses log2(rank+1) discount and a within-list ideal",
      abs(one_at_2["nDCG@5"] - (1 / math.log2(3)) / 1.0) < 1e-12,
      f"got {one_at_2['nDCG@5']:.6f}, expected {(1/math.log2(3)):.6f}")

# 4 -- perfect ordering must be the ceiling, and reversing must be the floor ------
# NOTE: the scores have to actually PRODUCE the perfect order. An earlier version of
# this test used descending scores over labels [1,0,1,0,0,1], which is not a perfect
# ordering at all (positives land at ranks 1, 3, 6) -- the test failed and the metric
# was right.
lab = [1, 0, 1, 0, 0, 1]
perfect = primary(["a"] * 6, lab, [10, 3, 9, 2, 1, 8])     # all positives on top
reversed_ = primary(["a"] * 6, lab, [1, 8, 2, 9, 10, 3])   # all positives at the bottom
check("a perfectly ordered user reaches GAUC 1.0 and nDCG 1.0",
      abs(perfect["GAUC"] - 1.0) < 1e-12 and abs(perfect["nDCG@5"] - 1.0) < 1e-12,
      f"GAUC {perfect['GAUC']:.4f} nDCG {perfect['nDCG@5']:.4f}")
check("a fully reversed ordering reaches GAUC 0.0",
      abs(reversed_["GAUC"]) < 1e-12, f"GAUC {reversed_['GAUC']:.4f}")
check("perfect ordering beats reversed ordering",
      perfect["primary"] > reversed_["primary"],
      f"{perfect['primary']:.4f} vs {reversed_['primary']:.4f}")

# 5 -- the published reference points we will be measured against ----------------
check("official test references recorded",
      OFFICIAL_TEST["fm_official"] == 0.5946 and OFFICIAL_TEST["random"] == 0.4753)
check("official valid references recorded", OFFICIAL_VALID["fm_official"] == 0.6016)
check("the GIVEN convergence rule is recorded, not invented",
      (OFFICIAL_EPSILON, OFFICIAL_N) == (0.002, 3))
check("eps is ~2.5 sigma of the published FM seed std",
      abs(OFFICIAL_EPSILON / OFFICIAL_FM_SEED_STD - 2.5) < 0.01,
      f"{OFFICIAL_EPSILON}/{OFFICIAL_FM_SEED_STD} = {OFFICIAL_EPSILON/OFFICIAL_FM_SEED_STD:.2f} sigma")
check("the nDCG ceiling is BELOW 1.0 and is recorded",
      OFFICIAL_TEST["oracle_ceiling"] < 1.0)

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
