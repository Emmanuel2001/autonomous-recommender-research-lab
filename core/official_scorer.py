"""Scoring for the CURRENT task. Enforcement, not trust -- same stance as before.

WHAT CHANGED FROM THE LEGACY SCORER
-----------------------------------
The unit of submission is no longer a per-user top-K list over a 7,583-item catalog; it
is ONE SCORE PER EVAL ROW, keyed by `row_id` (its position in the official row order).
So the checks change shape:

  legacy                          current
  --------------------------      ------------------------------------------------
  user coverage == split users     row COUNT and ORDER match the split exactly
  no duplicate items in a row      (n/a)
  N >= 50 ranked items             (n/a)
  mask compliance verified         (n/a -- no retrieval, nothing to mask)
  random-ranking floor             random floor, anchored to the ORGANISERS' 0.4753

The last one is a genuine upgrade: our negative control is now a number the organisers
published, not one we derived. If a scored `random` submission does not land near
0.4753 the harness is broken, and that is their own stated self-check.

WHY ORDER IS CHECKED AND NOT JUST LENGTH
----------------------------------------
`(user_id, video_id)` is not unique -- 3.06% of test rows are duplicate pairs, up to 12
repeats. A submission with the right rows in the wrong order would pass any set-based
check and score garbage. So the split's `content_sha256` (which hashes user, video and
date IN ORDER) is bound into every signed entry, and the redundant user/video columns
are verified row by row when a CSV is supplied.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class OfficialScoringError(Exception):
    """Refusal. The orchestrator gets this instead of a signed entry."""


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HEADER = ["row_id", "user_id", "video_id", "score"]


def read_scores(path: str | Path, rows) -> tuple[np.ndarray, str]:
    """Load a score vector for `rows`. Accepts .npy or the official submission CSV.

    Returns (scores, fmt). Every alignment failure raises with a readable reason -- the
    same failures the kit's `submit.py --check` rejects, so ours cannot be laxer.
    """
    p = Path(path)
    if not p.exists():
        raise OfficialScoringError(f"score file not found: {p}")

    if p.suffix == ".npy":
        s = np.load(p)
        if s.ndim != 1:
            raise OfficialScoringError(f"expected a 1-D score vector, got shape {s.shape}")
        if len(s) != len(rows):
            raise OfficialScoringError(
                f"score count {len(s):,} != {rows.split} rows {len(rows):,}")
        fmt = "npy"
    else:
        import csv
        with p.open(newline="") as fh:
            r = csv.reader(fh)
            head = next(r, None)
            if head != HEADER:
                raise OfficialScoringError(f"header must be {','.join(HEADER)}, got {head}")
            s = np.empty(len(rows), dtype=np.float64)
            n = 0
            for line in r:
                if n >= len(rows):
                    raise OfficialScoringError(f"more rows than {rows.split} ({len(rows):,})")
                if len(line) != 4:
                    raise OfficialScoringError(f"row {n}: expected 4 fields, got {len(line)}")
                rid, u, v, sc = line
                if int(rid) != n:
                    raise OfficialScoringError(f"row_id must be 0-based and contiguous; "
                                               f"expected {n}, got {rid}")
                if int(u) != int(rows.user_id[n]) or int(v) != int(rows.video_id[n]):
                    raise OfficialScoringError(
                        f"row {n} misaligned: submission says user {u} video {v}, "
                        f"split says user {rows.user_id[n]} video {rows.video_id[n]}")
                s[n] = float(sc)
                n += 1
            if n != len(rows):
                raise OfficialScoringError(f"row count {n:,} != {rows.split} {len(rows):,}")
        fmt = "csv"

    s = np.asarray(s, dtype=np.float64)
    bad = ~np.isfinite(s)
    if bad.any():
        raise OfficialScoringError(f"{int(bad.sum())} non-finite scores (NaN/Inf) at rows "
                                   f"{np.flatnonzero(bad)[:5].tolist()}...")
    return s, fmt


def score_official(path, split_name, rows, config_hash, ledger, *,
                   kind="score", isolation="unspecified", seed=None,
                   reference=True, isolation_available=None) -> dict:
    """Score one submission and append a SIGNED ledger entry.

    `isolation_available` is the scorer's own map of which sandbox backends work on this
    machine (empty string = available, otherwise the reason it is not). Passing it lets
    this cross-check the client's CLAIM, which nothing did before: a client could send
    `isolation="docker"` on a machine whose genesis entry says *"docker daemon not
    responding"*, and the scorer would sign `isolation_claimed: "docker"` without comment.
    One chain then carried both statements -- contradictory signed evidence, unflagged.
    The scorer still cannot verify that a claimed sandbox was really used; the field is
    named `isolation_claimed` for that reason. But it can refuse to let a claim contradict
    what it measured itself, which is the difference between an unverified claim and an
    impossible one.
    """
    from core.metrics import primary, OFFICIAL_TEST, OFFICIAL_VALID

    if rows.label is None:
        raise OfficialScoringError(f"split {split_name!r} has no labels in this process; "
                                   f"the scorer must be started with them")
    s, fmt = read_scores(path, rows)
    res = primary(rows.user_id, rows.label, s)

    ref = {}
    flags = []
    if reference:
        # Negative control anchored to the ORGANISERS' published self-check, not ours.
        rng = np.random.default_rng(int(_sha256_file(Path(path))[:8], 16) % (2 ** 31))
        rnd = primary(rows.user_id, rows.label, rng.random(len(rows)))
        pub = (OFFICIAL_TEST if split_name == "test" else OFFICIAL_VALID)
        ref = {"random_control": round(rnd["primary"], 6),
               "published_random": pub["random"],
               "published_baseline": pub["fm_official"],
               "published_oracle": pub["oracle_ceiling"]}
        if abs(rnd["primary"] - pub["random"]) > 0.01:
            flags.append("random_control_off_published_value")
        if res["primary"] < rnd["primary"]:
            flags.append("below_random_control")

    # Does the claimed sandbox exist on this machine? Flagged rather than refused: the
    # score itself is still a real measurement of a real prediction file, and throwing it
    # away would lose evidence. What must not happen is signing the claim silently.
    if isolation_available is not None and isolation not in ("unspecified", None):
        why = isolation_available.get(isolation)
        if why is None:
            flags.append(f"isolation_claimed_unknown_backend:{isolation}")
        elif why:
            flags.append(f"isolation_claimed_unavailable:{isolation}")

    payload = {
        "kind": kind,
        "task": "official_within_user_ranking",
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split_name,
        "label": "long_view",
        "metric": "primary = mean(GAUC, nDCG@5)",
        "config_hash": config_hash,
        "seed": seed,
        "isolation_claimed": isolation,
        "scores_sha256": _sha256_file(Path(path)),
        "scores_format": fmt,
        "split_content_sha256": rows.content_sha256(),
        "n_rows": int(len(rows)),
        "metrics": {"GAUC": round(res["GAUC"], 6),
                    "nDCG@5": round(res["nDCG@5"], 6),
                    "primary": round(res["primary"], 6)},
        "n_users": int(res["users"]),
        "reference": ref,
        "flags": flags,
    }
    if split_name != "test":
        payload["delta_vs_official_baseline"] = round(
            res["primary"] - OFFICIAL_VALID["fm_official"], 6)
    else:
        payload["delta_vs_official_baseline"] = round(
            res["primary"] - OFFICIAL_TEST["fm_official"], 6)
    return ledger.append(payload)
