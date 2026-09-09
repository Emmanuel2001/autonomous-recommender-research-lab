"""TaskSpec — the declared contract between a problem and the agent.

WHAT THIS IS FOR
----------------
The agent decides model, features, loss, and what to try next. It does NOT decide the
metric, the split, or the submission format. That division is not a limitation on the
LLM's intelligence, it is what makes its numbers mean anything:

  * an agent that picks its own metric can pick an easy one -- "the agent decided to
    optimise AUC instead" is not a result, it is the reward-hacking failure the scorer
    and ledger exist to prevent;
  * an agent that defines its own split can train on the test set, not maliciously but
    by getting it wrong once, silently;
  * the problem owner owns these. The organisers say so outright: the conventions are
    fixed, do not change them.

So a TaskSpec is DECLARED. The LLM may DRAFT one from a data directory plus a one-line
goal (see agent/draft_task.py), and a human confirms it once. That confirmation is a
setup step, not a per-iteration one, so it costs nothing on the autonomy metric, which
counts interventions DURING a run.

WHAT A SPEC OWNS vs WHAT THE STRATEGY OWNS
------------------------------------------
    spec owns            data location, split boundaries, the label column, the metric
                         function, the submission schema, the published baseline
    strategy owns        features, model, loss, optimisation -- everything the agent
                         may change

REFACTOR DISCIPLINE
-------------------
This module was factored OUT of core/official_data.py and core/metrics.py, which
continue to work unchanged. The KuaiRand spec below must reproduce the recorded numbers
EXACTLY (valid 0.601696 / test 0.594922 pointwise, seed 0). tests/test_task.py asserts
it. If a number moves, the refactor is wrong -- not the number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class MetricSpec:
    """How a score vector becomes a number. Owned by the problem, never the agent."""
    name: str                                  # e.g. "primary = mean(GAUC, nDCG@5)"
    fn: Callable                               # (user_ids|None, labels, scores) -> dict
    primary_key: str                           # which key in the returned dict ranks
    higher_is_better: bool = True
    #: grouping column for group-wise metrics (GAUC, per-user nDCG). None = global.
    group_by: str | None = "user_id"
    #: conventions that are easy to get wrong, recorded so a reader can check them
    conventions: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitSpec:
    """Which rows are train / dev / held-out. Owned by the problem."""
    name: str
    kind: str                                  # "date_range" | "row_fraction" | "column"
    bounds: dict                               # e.g. {"train": (20220408, 20220421), ...}
    column: str | None = None                  # the column `bounds` refers to
    #: split whose labels the agent may see for model selection ("public validation")
    dev_split: str = "valid"
    #: split whose labels the agent must NEVER see
    held_out_split: str = "test"
    expect_rows: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SubmissionSpec:
    """The shape of a submission file. Owned by the problem."""
    header: tuple[str, ...]
    row_key: str                               # the positional key, e.g. "row_id"
    #: True when (natural key) is not unique and a positional id is therefore mandatory
    positional_required: bool = True
    note: str = ""


@dataclass(frozen=True)
class TaskSpec:
    """One problem, fully declared."""
    slug: str
    title: str
    data_dir: Path
    label: str
    metric: MetricSpec
    split: SplitSpec
    submission: SubmissionSpec
    loader: Callable                           # () -> (dict[str, Rows], aux)
    #: scorer-side loader with held-out labels present. Optional only for historical specs.
    scorer_loader: Callable | None = None
    #: published reference points. `baseline` is what must be beaten.
    baseline: dict = field(default_factory=dict)
    #: the organisers' convergence rule when they give one, else ours, declared
    convergence: dict = field(default_factory=dict)
    #: facts handed to the agent as seed knowledge, including measured dead ends
    knowledge: dict = field(default_factory=dict)
    #: Whether this task requires a user-conditioned prediction pathway. Declared by the
    #: problem owner, never guessed from a column name or metric.
    personalization: dict = field(default_factory=lambda: {"required": False})

    def score(self, rows, scores) -> dict:
        """Apply the metric. The one place a number is produced from a score vector."""
        group = (getattr(rows, self.metric.group_by) if self.metric.group_by else None)
        return self.metric.fn(group, rows.label, scores)

    def primary(self, rows, scores) -> float:
        return float(self.score(rows, scores)[self.metric.primary_key])


# --------------------------------------------------------------- the KuaiRand-Pure spec
def kuairand_pure() -> TaskSpec:
    """The competition task, assembled from the pieces already verified against the kit."""
    from core import official_data as D
    from core.metrics import (primary as official_primary, OFFICIAL_TEST, OFFICIAL_VALID,
                              OFFICIAL_EPSILON, OFFICIAL_N, OFFICIAL_FM_SEED_STD)
    from agent import official_actions as OA

    return TaskSpec(
        slug="kuairand_pure",
        title="KuaiRand-Pure — within-user ranking of logged impressions",
        data_dir=D.DATA,
        label=D.LABEL,
        metric=MetricSpec(
            name="primary = mean(GAUC, nDCG@5)",
            fn=lambda g, y, s: official_primary(g, y, s),
            primary_key="primary",
            group_by="user_id",
            conventions=(
                "zero-positive users score nDCG 0.0 and ARE counted in the mean "
                "(27.1% of test users; dropping them inflates the score ~37%)",
                "GAUC covers only users with 0 < npos < n_impressions, weighted by npos",
                "nDCG gain is 2^rel - 1, discount log2(rank+1), ideal over the user's "
                "own candidate list",
                "the nDCG@5 ceiling is 0.7289, not 1.0",
            )),
        split=SplitSpec(
            name="official date split", kind="date_range", column="date",
            bounds=dict(D.SPLITS), dev_split="valid", held_out_split="test",
            expect_rows=dict(D.EXPECT_ROWS)),
        submission=SubmissionSpec(
            header=("row_id", "user_id", "video_id", "score"), row_key="row_id",
            positional_required=True,
            note="(user_id, video_id) is NOT unique -- 3.06% duplicate pairs in test, "
                 "up to 12 repeats -- so row_id is mandatory"),
        loader=D.load_for_strategy,
        scorer_loader=D.load_for_scorer,
        baseline={"name": "FM (organiser-provided)",
                  "valid": OFFICIAL_VALID["fm_official"],
                  "test": OFFICIAL_TEST["fm_official"],
                  "seed_std": OFFICIAL_FM_SEED_STD,
                  "ladder": {"random": OFFICIAL_TEST["random"],
                             "item_popularity": OFFICIAL_TEST["item_popularity"],
                             "oracle_ceiling": OFFICIAL_TEST["oracle_ceiling"]}},
        convergence={"epsilon": OFFICIAL_EPSILON, "N": OFFICIAL_N,
                     "source": "organisers (baseline_scores.json)",
                     "note": "epsilon is ~2.5 sigma of the published seed std; it is a "
                             "keep-searching threshold, NOT a significance test"},
        knowledge={
            "dead_ends": dict(OA.DEAD_ENDS),
            "unexplored": OA.__doc__.split("Encouraged")[-1].strip(),
            "time_col": "date", "item_col": "video_id",
            "loaded_sources": [
                "log_standard_4_08_to_4_21_pure.csv",
                "log_standard_4_22_to_5_08_pure.csv",
                "video_features_basic_pure.csv: author_id only",
            ],
            "excluded_sources": [
                {"source": "user_features_pure.csv",
                 "reason": "present in the dataset directory but not joined by the "
                           "current loader"},
                {"source": "video_features_basic_pure.csv: remaining columns",
                 "reason": "the current loader joins author_id only"},
                {"source": "video_features_statistic_pure.csv",
                 "reason": "period aggregates cover the full logging window, including "
                           "held-out dates; a naive join leaks future information"},
                {"source": "log_random_4_22_to_5_08_pure.csv",
                 "reason": "different exposure policy and excluded from the declared "
                           "official task loader"},
            ],
        },
        personalization={
            "required": True,
            "user_role": "user_id",
            "require_interaction_not_user_bias_only": True,
            "non_personalized_allowed_only_as_declared_ablation": True,
        },
    )


REGISTRY: dict[str, Callable[[], TaskSpec]] = {"kuairand_pure": kuairand_pure}


def get(slug: str) -> TaskSpec:
    if slug not in REGISTRY:
        raise KeyError(f"unknown task {slug!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[slug]()


def load(path_or_slug: str | Path) -> TaskSpec:
    """Load a registered task or a confirmed portable JSON task specification."""
    p = Path(path_or_slug)
    if p.is_file():
        from core.tabular_task import load_task_spec
        return load_task_spec(p)
    return get(str(path_or_slug))
