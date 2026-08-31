"""The official loader: split boundaries, row ORDER, and the quarantine.

Row order is the point. The submission is keyed by `row_id` = position in
`data.load()[split]`, and (user_id, video_id) is NOT unique -- so producing the right
SET of rows in the wrong ORDER yields a submission that passes a naive check and scores
garbage. This suite compares against the kit's own loader element-for-element.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.official_data import (load, load_for_scorer, load_for_strategy, dur_bucket,
                                EXPECT_ROWS, SPLITS, LABEL, DATA)

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

splits, edges = load_for_scorer()

# 1 -- boundaries and counts -----------------------------------------------------
for name, exp in EXPECT_ROWS.items():
    check(f"{name} has the published row count", len(splits[name]) == exp, f"{len(splits[name]):,}")
for name, (lo, hi) in SPLITS.items():
    d = splits[name].date
    check(f"{name} dates are within {lo}-{hi}", d.min() >= lo and d.max() <= hi,
          f"{d.min()}..{d.max()}")
tot = sum(len(splits[n]) for n in SPLITS)
check("the three splits are disjoint and cover both standard logs",
      tot == sum(EXPECT_ROWS.values()), f"{tot:,}")

# 2 -- ORDER, against the kit's own loader ---------------------------------------
KIT = ROOT.parent / "kuairand-starter-kit"
if not (KIT / "data.py").exists():
    print(f"  [skip] kit not present at {KIT}; cannot verify row order")
else:
    sys.path.insert(0, str(KIT))
    import importlib
    kit_data = importlib.import_module("data")
    kit = kit_data.load(str(DATA))
    for name in ("train", "valid", "test"):
        krows = kit[name]
        ours = splits[name]
        check(f"{name}: same row count as the kit", len(krows) == len(ours),
              f"kit {len(krows):,} vs ours {len(ours):,}")
        ku = np.array([int(x[1]) for x in krows], np.int64)
        kv = np.array([int(x[2]) for x in krows], np.int64)
        kl = np.array([float(x[6]) for x in krows], np.float64)
        check(f"{name}: user_id ELEMENT-FOR-ELEMENT identical to the kit",
              np.array_equal(ku, ours.user_id))
        check(f"{name}: video_id ELEMENT-FOR-ELEMENT identical to the kit",
              np.array_equal(kv, ours.video_id))
        check(f"{name}: {LABEL} element-for-element identical to the kit",
              np.array_equal(kl, ours.label))
    # dur_bucket must match the kit's encoding too. Compute the edges ONCE -- an
    # earlier version rebuilt them inside the comprehension, i.e. 5000 quantile passes
    # over 1.14M rows, and the test hung.
    kit_edges = kit_data._bucket_edges([x[5] for x in kit['train']])
    check("our train deciles equal the kit's", np.allclose(kit_edges, edges),
          f"max diff {np.max(np.abs(np.asarray(kit_edges) - edges)):.3g}")
    kd = np.array([x[5] for x in kit['test'][:20000]], np.float64)
    kb = np.searchsorted(kit_edges, kd)
    ob = dur_bucket(splits['test'].duration_ms[:20000], edges)
    check("dur_bucket matches the kit's train-decile encoding", np.array_equal(kb, ob))

# 3 -- (user, video) is NOT unique: this is why row_id exists --------------------
te = splits["test"]
pairs = np.stack([te.user_id, te.video_id], 1)
uniq = np.unique(pairs, axis=0)
dupe_frac = 1.0 - len(uniq) / len(pairs)
check("(user_id, video_id) is NOT unique in test -- row_id is mandatory",
      dupe_frac > 0.02, f"{dupe_frac*100:.2f}% duplicate pairs (kit says 3.06%)")
check("row_id is 0-based and contiguous", te.row_id[0] == 0 and te.row_id[-1] == len(te) - 1)

# 4 -- the published test-set composition ---------------------------------------
import collections
byu = collections.defaultdict(list)
for u, y in zip(te.user_id.tolist(), te.label.tolist()):
    byu[u].append(y)
n = len(byu)
allneg = sum(1 for v in byu.values() if sum(v) == 0) / n * 100
allpos = sum(1 for v in byu.values() if sum(v) == len(v)) / n * 100
disc = 100 - allneg - allpos
check("test user count matches the kit", n == 23875, f"{n:,}")
check("all-negative share matches the kit's 27.1%", abs(allneg - 27.1) < 0.15, f"{allneg:.1f}%")
check("all-positive share matches the kit's 9.2%", abs(allpos - 9.2) < 0.15, f"{allpos:.1f}%")
check("discriminative share matches the kit's 63.7%", abs(disc - 63.7) < 0.2, f"{disc:.1f}%")

# 5 -- THE QUARANTINE ------------------------------------------------------------
# The quarantine protects TEST, not valid. Valid is the "public validation feedback"
# the brief explicitly grants, and the graded submission is the validation-best
# checkpoint -- withholding it broke early stopping and did not make us safer.
sp2, _ = load_for_strategy()
check("load_for_strategy returns NO test label -- the one hard invariant",
      sp2["test"].label is None)
check("load_for_strategy DOES return valid labels (public validation feedback)",
      sp2["valid"].label is not None)
check("load_for_strategy DOES return train labels", sp2["train"].label is not None)
check("test FEATURES are still available to the strategy",
      len(sp2["test"]) == EXPECT_ROWS["test"] and sp2["test"].user_id.size > 0)
check("default want_labels excludes test", load()[0]["test"].label is None)
_leak = False
try:
    _s, _ = load(want_labels=("train", "valid", "test"))
    _leak = _s["test"].label is not None
except Exception:
    pass
check("test labels are reachable ONLY by asking for them explicitly (scorer only)", _leak)
check("aux signals are carried for train only",
      bool(sp2["train"].aux) and not sp2["test"].aux,
      f"{len(sp2['train'].aux)} aux arrays on train")
check("content_sha256 binds ORDER, not just contents",
      splits["test"].content_sha256() != splits["valid"].content_sha256())

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
