"""TaskSpec: the declared contract, and proof the refactor moved no number.

A refactor that changes a score is not a refactor. The recorded seed-0 figures below
come from experiments/official_seeds.json, produced BEFORE core/task.py existed, so
reproducing them through the spec is evidence the seam is inert.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.task import get, REGISTRY, TaskSpec, MetricSpec, SplitSpec, SubmissionSpec

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

t = get("kuairand_pure")

# 1 -- the spec declares what the problem owns -----------------------------------
check("registry exposes the competition task", "kuairand_pure" in REGISTRY)
check("label is declared by the spec, not the agent", t.label == "long_view")
check("metric is a function owned by the spec", callable(t.metric.fn))
check("primary key names which number ranks", t.metric.primary_key == "primary")
check("split declares the held-out split explicitly", t.split.held_out_split == "test")
check("split declares the dev split the agent MAY see", t.split.dev_split == "valid")
check("dev and held-out are different splits",
      t.split.dev_split != t.split.held_out_split)
check("expected row counts are pinned", t.split.expect_rows["test"] == 170588)
check("submission requires a positional key", t.submission.positional_required)
check("submission header matches the kit", t.submission.header ==
      ("row_id", "user_id", "video_id", "score"))
check("the organisers' convergence rule is recorded WITH its source",
      t.convergence["epsilon"] == 0.002 and t.convergence["N"] == 3
      and "organiser" in t.convergence["source"])
check("baseline to beat is declared", abs(t.baseline["test"] - 0.5946) < 1e-9)
check("the trap conventions are written down, not assumed",
      len(t.metric.conventions) >= 4)
check("measured dead ends are carried as knowledge",
      set(t.knowledge["dead_ends"]) >= {"more_capacity", "user_side_only"})
try:
    get("nope"); _raised = False
except KeyError:
    _raised = True
check("unknown task raises rather than guessing", _raised)

# 2 -- THE REFACTOR MUST MOVE NO NUMBER ------------------------------------------
recorded = json.loads((ROOT / "experiments" / "official_seeds.json").read_text())
splits, edges = t.loader()
check("the spec's loader still quarantines the held-out labels",
      splits[t.split.held_out_split].label is None)
check("the spec's loader still exposes the dev labels",
      splits[t.split.dev_split].label is not None)

# score the held-out split through the spec, using labels the SCORER would hold
from core.official_data import load_for_scorer
full, _ = load_for_scorer()
from strategy.ranker import Config, run as fit

for arm in ("pointwise", "pairwise"):
    scores, _info = fit(splits, edges, Config(loss=arm, seed=0))
    for split in ("valid", "test"):
        via_spec = t.primary(full[split], scores[split])
        want = recorded["arms"][arm][split][0]        # seed 0, recorded pre-refactor
        # Tolerance is 1e-6 because that is the precision the reference was STORED at
        # (experiments/official_seeds.json writes round(x, 6)). Asserting 1e-9 against a
        # 6-dp reference can never pass and says nothing about the refactor.
        check(f"{arm} {split}: spec reproduces the pre-refactor number",
              abs(via_spec - want) < 1e-6,
              f"{via_spec:.9f} vs recorded {want:.6f} (diff {abs(via_spec-want):.2e})")

# 3 -- the spec's metric IS the vendored official one, not a copy -----------------
from core.metrics import primary as official
rng = np.random.default_rng(0)
u = np.array([f"u{i//3}" for i in range(60)])
y = rng.integers(0, 2, 60).astype(float)
s = rng.normal(size=60)


class _Rows:
    user_id = u
    label = y


check("spec.score delegates to the vendored official metric",
      t.score(_Rows(), s) == official(u, y, s))
check("spec.primary returns the ranking number",
      t.primary(_Rows(), s) == official(u, y, s)["primary"])

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
