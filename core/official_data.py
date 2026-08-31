"""Loading and splitting for the CURRENT task, aligned to the Starter Kit.

WHY THIS IS A NEW MODULE RATHER THAN AN EDIT TO core/splits.py
--------------------------------------------------------------
`core/splits.py` implements the SUPERSEDED task: full-catalog retrieval, `is_click`,
a positional cut of the eval file, and a four-way partition with an audit split. Its
experiments and its 37 tests are still reproducible and still cited in STATE.md, so it
is left intact. This module is the current task and nothing here is inherited.

ROW ORDER IS LOAD-BEARING
-------------------------
The submission format is `row_id,user_id,video_id,score` where `row_id` is the
POSITION of the row in `data.load()[split]`. It cannot be a (user, video) key because
**that pair is not unique** -- the kit measures 3.06% duplicate pairs in test, up to 12
repeats. So our loader must produce the kit's order element-for-element, not merely the
same set. `tests/test_official_data.py` asserts that against the kit's own loader.

The kit's order is: read `log_standard_4_08_to_4_21_pure.csv`, then
`log_standard_4_22_to_5_08_pure.csv`, appending rows in file order; then filter by date
per split, preserving that order.

WHAT THE QUARANTINE ACTUALLY PROTECTS -- corrected 2026-08-27
------------------------------------------------------------
An earlier version of this module withheld BOTH `valid` and `test` labels from the
strategy. That was wrong, and it broke early stopping. The brief says the agent
develops using "only the training split and **the public validation feedback**" -- so:

  train  labels: available. Training data.
  valid  labels: **available.** This IS the public validation feedback. The kit's own
                 baseline early-stops on it, and the graded submission is the
                 "validation-best checkpoint". Withholding it does not make us safer,
                 it makes us unable to do the task.
  test   labels: **quarantined.** Never returned by `load_for_strategy()`, under any
                 argument. This is the set the score comes from.

The audit property is preserved by a different rule, which is the one that actually
matters: **the strategy may use valid internally for model selection, but every number
that enters the LEDGER comes from the scorer process.** Early stopping is model
selection; a recorded metric is a claim. Only the second needs a signature.
"""
from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("TRACK2_DATA_ROOT")
            or Path(__file__).resolve().parents[1] / "KuaiRand-Pure" / "data")

LABEL = "long_view"                    # official: native 0/1 column
LOG_FILES = ("log_standard_4_08_to_4_21_pure.csv",
             "log_standard_4_22_to_5_08_pure.csv")
RANDOM_LOG = "log_random_4_22_to_5_08_pure.csv"     # unbiased validation, opt-in
SPLITS = {"train": (20220408, 20220421),
          "valid": (20220422, 20220428),
          "test":  (20220429, 20220508)}
#: Published row counts. Asserted on load -- a mismatch means wrong data or wrong order.
EXPECT_ROWS = {"train": 1_141_112, "valid": 124_909, "test": 170_588}

#: The kit's 5 baseline feature fields. `dur_bucket` is derived from duration_ms by
#: train-set deciles. Extra RAW columns we also carry so the agent can build features
#: from them without another pass over 1.4M rows.
KIT_FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
AUX_SIGNALS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward",
               "is_hate", "is_profile_enter", "long_view")
AUX_CONTINUOUS = ("play_time_ms", "duration_ms", "profile_stay_time",
                  "comment_stay_time")


@dataclass(frozen=True)
class Rows:
    """One split. Arrays are parallel and in KIT ORDER; index == row_id."""
    split: str
    user_id: np.ndarray            # int64
    video_id: np.ndarray           # int64
    author_id: np.ndarray          # int64, -1 for UNK
    tab: np.ndarray                # int64
    duration_ms: np.ndarray        # float64
    date: np.ndarray               # int64
    hourmin: np.ndarray            # int64
    label: np.ndarray | None       # float64 long_view, or None when quarantined
    aux: dict = field(default_factory=dict, repr=False)   # other signals, train only

    def __len__(self) -> int:
        return int(self.user_id.size)

    @property
    def row_id(self) -> np.ndarray:
        return np.arange(len(self), dtype=np.int64)

    def content_sha256(self) -> str:
        """Identity of this split, order included. Binds a score file to a split."""
        h = hashlib.sha256()
        for a in (self.user_id, self.video_id, self.date):
            h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
        return h.hexdigest()


def _read_logs(include_random: bool = False) -> list[dict]:
    """All log rows, in the kit's file-then-line order. csv.DictReader like the kit."""
    files = list(LOG_FILES) + ([RANDOM_LOG] if include_random else [])
    out: list[dict] = []
    for f in files:
        with open(DATA / f, newline="") as fh:
            for r in csv.DictReader(fh):
                out.append(r)
    return out


def _vid2author() -> dict[str, str]:
    m = {}
    with open(DATA / "video_features_basic_pure.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            m[r["video_id"]] = r["author_id"]
    return m


def _bucket_edges(durations: np.ndarray, n: int = 10) -> np.ndarray:
    """Train-set deciles, exactly the kit's `_bucket_edges`."""
    return np.quantile(np.asarray(durations, dtype=np.float64),
                       np.linspace(0, 1, n + 1)[1:-1])


#: Splits whose labels the strategy may see. `test` is absent and must stay absent.
STRATEGY_LABELS = ("train", "valid")


def load(include_random: bool = False,
         want_labels: tuple[str, ...] = STRATEGY_LABELS) -> dict[str, Rows]:
    """Load and split. `want_labels` names the splits whose label is returned.

    Defaults to train + valid, which is exactly what the brief permits the agent to
    develop against. `test` must be requested explicitly, and only the scorer does so.
    """
    raw = _read_logs(include_random)
    v2a = _vid2author()

    date = np.fromiter((int(r["date"]) for r in raw), np.int64, len(raw))
    out: dict[str, Rows] = {}
    # duration bucket edges come from TRAIN only, like the kit
    tr_mask = (date >= SPLITS["train"][0]) & (date <= SPLITS["train"][1])
    dur_all = np.fromiter((float(r["duration_ms"]) for r in raw), np.float64, len(raw))
    edges = _bucket_edges(dur_all[tr_mask])

    # stable string->id maps, built from TRAIN order so ids are reproducible
    author_vocab: dict[str, int] = {}
    tab_vocab: dict[str, int] = {}

    def _id(vocab: dict[str, int], key: str, allow_new: bool) -> int:
        if key in vocab:
            return vocab[key]
        if not allow_new:
            return -1                       # UNK
        vocab[key] = len(vocab)
        return vocab[key]

    for name, (lo, hi) in SPLITS.items():
        sel = np.flatnonzero((date >= lo) & (date <= hi))
        if name != "train" and len(sel) == 0:
            continue
        is_train = name == "train"
        rr = [raw[i] for i in sel]
        n = len(rr)
        u = np.fromiter((int(r["user_id"]) for r in rr), np.int64, n)
        v = np.fromiter((int(r["video_id"]) for r in rr), np.int64, n)
        a = np.fromiter((_id(author_vocab, v2a.get(r["video_id"], "UNK"), is_train)
                         for r in rr), np.int64, n)
        t = np.fromiter((_id(tab_vocab, r["tab"], is_train) for r in rr), np.int64, n)
        d = dur_all[sel]
        dt = date[sel]
        hm = np.fromiter((int(r["hourmin"]) for r in rr), np.int64, n)
        lab = np.fromiter((0.0 if r[LABEL] == "0" else 1.0 for r in rr), np.float64, n)
        aux = {}
        if is_train:
            for c in AUX_SIGNALS:
                if c == LABEL:
                    continue
                aux[c] = np.fromiter((0.0 if r[c] == "0" else 1.0 for r in rr),
                                     np.float64, n)
            for c in AUX_CONTINUOUS:
                aux[c] = np.fromiter((float(r[c]) for r in rr), np.float64, n)
        out[name] = Rows(name, u, v, a, t, d, dt, hm,
                         lab if name in want_labels else None, aux)

    for name, exp in EXPECT_ROWS.items():
        got = len(out[name])
        if got != exp:
            raise ValueError(f"split {name}: expected {exp:,} rows, got {got:,}. "
                             f"Wrong data directory, or the kit's split changed.")
    return out, edges


def dur_bucket(duration_ms: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The kit's `dur_bucket`: np.searchsorted against train deciles."""
    return np.searchsorted(edges, np.asarray(duration_ms, dtype=np.float64)).astype(np.int64)


def load_for_strategy(include_random: bool = False):
    """What the (sandboxed) strategy may see: train + valid labels, test features only.

    The `test` check is not decoration. It is the one invariant that cannot be relaxed
    without invalidating the submission, so it is asserted on every load rather than
    trusted to the call site.
    """
    splits, edges = load(include_random=include_random, want_labels=STRATEGY_LABELS)
    if splits["test"].label is not None:
        raise RuntimeError("quarantine violated: test labels were returned to the strategy")
    if splits["valid"].label is None:
        raise RuntimeError("valid labels are the public validation feedback and are "
                           "required for early stopping; see this module's header")
    return splits, edges


def load_for_scorer():
    """Labels for valid and test live here and nowhere else."""
    return load(want_labels=("train", "valid", "test"))
