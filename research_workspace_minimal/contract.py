"""THE CONTRACT between agent-written strategy code and the harness. Immutable.

The agent may rewrite `model.py`, `features.py`, `train.py` and `config.json`. It may not
change this file: everything the harness relies on to score a candidate, compare it to its
parent, and refuse an invalid one is declared here. A run pins one contract, and the
contract's own sha256 is recorded beside every candidate's tree hash -- so "the rules
changed mid-run" is a visible fact rather than an argument.

WHAT A CANDIDATE MUST PROVIDE
----------------------------
`train.py` defines exactly one entry point:

    def fit_predict(data: Data, params: dict) -> dict

Returning:

    {"dev":          1-D float array, one score per data.dev row, in row order,
     "heldout":      1-D float array, one score per data.heldout row, in row order,
     "train_subset": 1-D float array, one score per row of train_subset(data)[0],
     "telemetry":    {...}}       # optional; per-epoch curves, notes, anything else

`train_subset` is REQUIRED, and it is three lines to produce -- predict on the rows
`train_subset(data)` names. It exists so the train-versus-validation gap is computed BY
THE HARNESS with the official metric, rather than being a number the candidate reports
about itself. Overfitting is the thing this project could not previously see; a candidate
that could omit it by rewriting its own telemetry would put that back.

Scores are finite, non-constant, higher = better. Row ORDER is the contract: the
submission is keyed by position, so a permuted return is silently wrong and is refused.

WHY `data` AND NOT (train, dev, heldout, params)
------------------------------------------------
The older ProposePatch contract passed three row objects positionally, which meant
anything the loader carried but the signature did not mention was unreachable -- the
`edges` needed for the baseline's own `dur_bucket` feature were computed by the harness
and then dropped on the floor. `Data` is one object with room to carry them, so a
candidate can use every feature the baseline uses, and `data.columns` lets it discover
what a DIFFERENT dataset provides instead of hard-coding KuaiRand's column names.

ON COMPUTING THE METRIC YOURSELF
--------------------------------
`primary()` is exposed, and candidates are expected to use it for early stopping and for
the train-versus-validation curve. This is not a hole:

  * `data.heldout.label` is None -- the labels that decide the competition are not in
    this process at all, and the harness refuses to run if they are;
  * `data.train.label` and `data.dev.label` ARE here, because you cannot fit or early-stop
    without them. Selecting an epoch on dev is early stopping, which is what the official
    baseline does too;
  * the number that COUNTS is produced by the scorer -- a separate process holding the
    labels and the signing key. A candidate's self-reported metric is telemetry. It is
    recorded as self-reported. The headline train and dev metrics are computed by the
    HARNESS from the score vectors the candidate returns, so the train/validation gap is
    never self-reported; per-epoch curves in `telemetry` are, and are labelled that way
    wherever they are shown.

The trust boundary is the scorer's signature, not the candidate's ignorance of arithmetic.

THE ONE LIMIT OF THE TRAIN/DEV GAP
----------------------------------
`check_scores` verifies that the `train_subset` vector has one entry per row that
`train_subset(data)` named. It CANNOT verify that those entries were computed for those
rows -- nothing in a returned array records which rows produced it. A candidate that scored
different rows would therefore report a meaningless gap.

That is a diagnostic limit, not a scoring hole: promotion and the submission use only the
scorer's signed dev number, so a wrong gap misleads the research narrative and cannot
inflate the result. It is written down here because the alternative -- presenting the gap as
though it were verified end to end -- would be the more expensive mistake.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields as _dc_fields
from typing import Any

import numpy as np

#: Exactly the files a candidate tree consists of, and the only ones the agent may edit.
#: The harness reads a workspace through this list, so a stray file the agent leaves
#: behind cannot change a candidate's identity (see core.artifacts.read_dir).
MUTABLE_FILES = ("features.py", "model.py", "train.py", "config.json")

#: The entry point `train.py` must define.
ENTRY_POINT = "fit_predict"

#: Users sampled for the TRAIN half of the overfitting curve. Fixed, and deliberately not
#: derived from the candidate's seed: every candidate must be measured on the same rows or
#: the curves are not comparable across a run.
TRAIN_SUBSET_USERS = 5000
TRAIN_SUBSET_SEED = 0xC0FFEE


class ContractError(RuntimeError):
    """A candidate ran but did not honour the contract."""


@dataclass(frozen=True)
class Spec:
    """Task identity, so a candidate is not written against KuaiRand by accident."""
    task: str
    dev_split: str
    held_out_split: str
    label: str


@dataclass(frozen=True)
class Data:
    """Everything a candidate is given. Frozen: a candidate cannot rebind the splits.

    train   -- rows with `.label`
    dev     -- rows with `.label` (the public validation split; early-stop on this)
    heldout -- rows with `.label is None`, by construction
    aux     -- dict[str, array] of additional TRAIN-only targets, may be empty
    edges   -- precomputed bucket edges from TRAIN only (duration quantiles for KuaiRand);
               may be None for a task that has none
    spec    -- Spec above
    """
    train: Any
    dev: Any
    heldout: Any
    aux: dict
    edges: Any
    spec: Spec

    @property
    def columns(self) -> tuple[str, ...]:
        """Feature columns the loader actually carries, discovered not assumed.

        Use this instead of hard-coding column names: it is what makes a candidate
        portable to another dataset, and it is how a candidate can tell that a field it
        wanted does not exist BEFORE spending a fit finding out.
        """
        return columns_of(self.train)


def columns_of(rows) -> tuple[str, ...]:
    """Column names on a row object, excluding the label and non-feature members."""
    if hasattr(rows, "__dataclass_fields__"):
        names = [f.name for f in _dc_fields(rows)]
    else:
        names = [n for n in dir(rows) if not n.startswith("_")]
    return tuple(n for n in names
                 if n not in ("label", "aux", "split")
                 and not callable(getattr(rows, n, None)))


def primary(user_ids, labels, scores, k: int = 5) -> dict:
    """The graded metric: {'GAUC', 'nDCG@5', 'primary', 'users', 'rows'}.

    One import site, forwarding to the vendored organiser code via core.metrics -- the
    same function the harness and the scorer use. Deliberately not reimplemented here:
    three of its conventions (zero-positive users counted at nDCG 0, GAUC restricted to
    users with 0 < npos < n, gain 2^rel - 1) each move the number by whole percent when
    got wrong.
    """
    from core.metrics import primary as _p
    return _p(user_ids, labels, scores, k=k)


def train_subset(data: Data) -> tuple[np.ndarray, str]:
    """Deterministic row indices for the train half of the overfitting curve.

    Returns (row_indices, identity_hash). Sampling by USER, not by row, because the
    metric is computed within a user: half a user's impressions would score a group that
    does not exist. The hash identifies the subset in the run log, so a curve can never
    be compared against one measured on different rows.
    """
    users = np.unique(np.asarray(data.train.user_id))
    n = min(TRAIN_SUBSET_USERS, users.size)
    rng = np.random.default_rng(TRAIN_SUBSET_SEED)
    picked = np.sort(rng.choice(users, size=n, replace=False))
    idx = np.flatnonzero(np.isin(np.asarray(data.train.user_id), picked))
    h = hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()
    return idx, h


def check_scores(name: str, arr, n: int) -> np.ndarray:
    """Validate one returned score vector. Raises ContractError with a usable reason."""
    if arr is None:
        raise ContractError(f"{ENTRY_POINT} returned no scores for {name!r}")
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size != n:
        raise ContractError(
            f"{name!r}: returned {a.size:,} scores for {n:,} rows. One score per row, in "
            f"row order -- the submission is keyed by position.")
    bad = ~np.isfinite(a)
    if bad.any():
        raise ContractError(
            f"{name!r}: {int(bad.sum())} non-finite scores (NaN/Inf), first at row "
            f"{int(np.flatnonzero(bad)[0])}. Check the learning rate, the loss, and any "
            f"log/divide in the forward pass.")
    if np.all(a == a[0]):
        raise ContractError(
            f"{name!r}: every score is identical ({a[0]:g}). A constant score cannot rank "
            f"anything -- within-user order would be arbitrary.")
    return a
