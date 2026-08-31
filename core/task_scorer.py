"""TaskSpec-owned scoring for portable recommender tasks."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class TaskScoringError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_task(path, split_name: str, rows, spec, config_hash: str, ledger, *,
               kind="score", isolation="unspecified", seed=None,
               isolation_available=None):
    p = Path(path)
    if not p.is_file():
        raise TaskScoringError(f"score file not found: {p}")
    if p.suffix != ".npy":
        raise TaskScoringError("portable scorer currently accepts a positional .npy vector")
    scores = np.asarray(np.load(p), dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(rows):
        raise TaskScoringError(
            f"expected {len(rows):,} positional scores, got shape {scores.shape}")
    if not np.isfinite(scores).all():
        raise TaskScoringError("scores contain NaN or infinity")
    if rows.label is None:
        raise TaskScoringError(f"scorer has no labels for split {split_name!r}")
    metrics = dict(spec.score(rows, scores))
    primary = float(metrics[spec.metric.primary_key])
    metrics.setdefault("primary", primary)
    clean = {k: (round(float(v), 6) if isinstance(v, (int, float, np.number)) else v)
             for k, v in metrics.items()}
    flags = []
    if isolation_available is not None and isolation not in (None, "unspecified"):
        why = isolation_available.get(isolation)
        if why is None:
            flags.append(f"isolation_claimed_unknown_backend:{isolation}")
        elif why:
            flags.append(f"isolation_claimed_unavailable:{isolation}")
    baseline = spec.baseline.get(split_name)
    payload = {
        "kind": kind, "task": spec.slug,
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split_name, "label": spec.label, "metric": spec.metric.name,
        "config_hash": config_hash, "seed": seed,
        "isolation_claimed": isolation, "scores_sha256": _sha(p),
        "scores_format": "npy", "split_content_sha256": rows.content_sha256(),
        "n_rows": int(len(rows)), "metrics": clean, "flags": flags,
        "reference": {"declared_baseline": baseline},
    }
    if baseline is not None:
        # `delta_vs_declared_baseline` only. This used to be duplicated into
        # `delta_vs_official_baseline` so existing readers would find it, but that made a
        # shop_clicks_demo record assert a delta against KuaiRand's official baseline, which
        # is not a thing that exists for that task. The two display readers fall back to this
        # key instead; core/scorer.py's official print path never sees a portable payload.
        payload["delta_vs_declared_baseline"] = round(primary - float(baseline), 6)
    return ledger.append(payload)
