"""The noise gate: is an observed movement larger than the noise?

Two variance components, measured, and which one binds depends on the model:

    sigma_seed   BPR-MF .00058 ndcg / .00113 recall ; EASE exactly 0 (closed form)
    sigma_eval   .00116 ndcg / .00261 recall  (user bootstrap, B=2000)

Per-user Recall sd is 0.3416 -- with ~2 relevant items a user's recall is usually
0 or 1 -- so comparing two configs via independent confidence intervals is very
weak. PAIRING cancels that between-user variance, which is most of sigma_eval.

Hence: paired bootstrap over users on the DIFFERENCE, with seed replication folded
in by averaging each user's value across seeds first (which shrinks sigma_seed by
sqrt(n_seeds) before the pairing happens).

These are pure functions over per-user metric arrays. They hold no labels; the
scorer calls them on data it already has.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 4000,
                     seed: int = 0) -> dict:
    """b vs a, aligned per user. Positive delta means b is better.

    Returns the point estimate, a percentile CI, and a two-sided bootstrap p-value
    for H0: mean difference == 0.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"unaligned inputs: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("no users to compare")
    d = b - a
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    point = float(d.mean())
    # two-sided bootstrap p: how often does the resampled mean fall on, or across,
    # the null side of zero
    p_lo = float((means <= 0).mean())
    p_hi = float((means >= 0).mean())
    p = float(min(1.0, 2.0 * min(p_lo, p_hi)))
    return {
        "delta": round(point, 6),
        "ci95": [round(float(np.percentile(means, 2.5)), 6),
                 round(float(np.percentile(means, 97.5)), 6)],
        "se": round(float(means.std(ddof=1)), 6),
        "p_two_sided": round(p, 5),
        "n_users": int(n),
        "n_boot": int(n_boot),
        "frac_users_improved": round(float((d > 0).mean()), 4),
        "frac_users_unchanged": round(float((d == 0).mean()), 4),
    }


def average_over_seeds(per_seed: list[np.ndarray]) -> np.ndarray:
    """Mean per-user value across replicate seeds of ONE config.

    Averaging before pairing is what folds sigma_seed into the comparison; doing
    it after would leave seed noise in the difference.
    """
    if not per_seed:
        raise ValueError("no seeds supplied")
    shapes = {x.shape for x in per_seed}
    if len(shapes) != 1:
        raise ValueError(f"seed replicates disagree in shape: {shapes}")
    return np.mean(np.stack(per_seed, axis=0), axis=0)


def decide(stats: dict, min_effect: float, alpha: float = 0.05) -> dict:
    """Accept only if the effect is BOTH statistically distinguishable from zero
    and larger than the pre-declared minimum effect.

    min_effect must come from a measurement (see experiments/noise_floor.json),
    not from taste. Two configs can differ significantly by an amount too small
    to be worth carrying, which is how a search fills up with noise it believes in.
    """
    ci_excludes_zero = (stats["ci95"][0] > 0) or (stats["ci95"][1] < 0)
    significant = stats["p_two_sided"] < alpha and ci_excludes_zero
    big_enough = abs(stats["delta"]) >= min_effect
    if significant and big_enough:
        verdict = "improved" if stats["delta"] > 0 else "regressed"
    elif significant and not big_enough:
        verdict = "significant_but_below_min_effect"
    else:
        verdict = "indistinguishable"
    return {"verdict": verdict, "significant": significant,
            "above_min_effect": big_enough, "min_effect": round(min_effect, 6),
            "alpha": alpha}
