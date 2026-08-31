"""Stage 2 of the MLE loop: inspect the data. Diagnostics the agent can run.

WHY THIS EXISTS
---------------
Mapped against the brief's Figure 1, this project automated stage 4 (train+tune) well,
stage 5 (evaluate) fully, and stage 2 (inspect data) with a single action. The agent could
not ask a question about the data -- every fact it had was one a human put in its prompt.
That is the widest gap in the "iterates autonomously across the full stack" requirement.

WHAT MAKES A DIAGNOSTIC SAFE TO OFFER
-------------------------------------
Every function here reports on TRAIN and the DEV split only. None of them touches the
held-out split, and the harness could not honour a request that did: the strategy-side
loader returns `None` for the held-out label and the sandbox has the outcome columns
redacted out of the bytes. So an inspection action cannot become a label oracle -- which
is the obvious way a feature like this goes wrong.

They are also CHEAP. A diagnostic that cost a full fit would compete with the thing it is
meant to inform; each of these is a pass or two over arrays already in memory.
"""
from __future__ import annotations

import numpy as np

#: Signals a diagnostic may look at on train. `long_view` is the label; the rest are the
#: auxiliary outcomes the brief lists as available for multi-task work.
AUX = ("is_click", "is_like", "is_follow", "is_comment", "is_forward",
       "is_profile_enter", "play_time_ms")


def _rate(a: np.ndarray) -> float:
    return float(np.mean(a)) if a.size else float("nan")


def label_rates(splits, spec) -> dict:
    """Base rate of the label on train vs dev, and how it moves by date.

    A drifting base rate is the difference between "my model is wrong" and "the world
    moved", and the agent has no way to tell those apart without looking.
    """
    tr, dv = splits["train"], splits[spec.split.dev_split]
    out = {"train_rate": round(_rate(tr.label), 5),
           "dev_rate": round(_rate(dv.label), 5)}
    out["shift"] = round(out["dev_rate"] - out["train_rate"], 5)
    by = {}
    for rows, name in ((tr, "train"), (dv, "dev")):
        for d in np.unique(rows.date):
            by[f"{name}:{int(d)}"] = round(_rate(rows.label[rows.date == d]), 4)
    out["by_date"] = by
    out["reading"] = ("A large train-vs-dev gap means the dev score is not measuring the "
                      "same population the model was fitted on; a within-split trend by "
                      "date means time is a usable feature.")
    return out


def group_sizes(splits, spec) -> dict:
    """Impressions per user on dev, and what fraction of users can be scored at all.

    The metric is computed within a user, so a user with one impression contributes no
    ordering information, and a user with no positive contributes a fixed 0 to nDCG. Both
    bound what any model can achieve, and neither is visible from a score.
    """
    dv = splits[spec.split.dev_split]
    u, inv = np.unique(dv.user_id, return_inverse=True)
    sizes = np.bincount(inv)
    pos = np.bincount(inv, weights=dv.label)
    all_neg = int(np.sum(pos == 0))
    all_pos = int(np.sum(pos == sizes))
    return {"users": int(u.size),
            "impressions_per_user_mean": round(float(sizes.mean()), 3),
            "impressions_per_user_median": int(np.median(sizes)),
            "singleton_users_pct": round(100.0 * float(np.mean(sizes == 1)), 2),
            "all_negative_users_pct": round(100.0 * all_neg / u.size, 2),
            "all_positive_users_pct": round(100.0 * all_pos / u.size, 2),
            "discriminative_users_pct": round(100.0 * (u.size - all_neg - all_pos) / u.size, 2),
            "reading": ("Only discriminative users carry GAUC. All-negative users hold "
                        "nDCG at 0 and all-positive at 1 whatever the model does, so the "
                        "attainable range is narrower than [0,1].")}


def aux_signal_overlap(splits, spec) -> dict:
    """How each auxiliary signal relates to the label, on TRAIN.

    Multi-task learning is only worth the compute when an auxiliary target carries
    information the label does not. A signal that is nearly identical to the label adds
    nothing; one that is nearly independent adds noise. This shows which is which instead
    of leaving it to be guessed.
    """
    tr = splits["train"]
    y = tr.label
    out = {}
    for name in AUX:
        a = tr.aux.get(name)
        if a is None:
            continue
        if name == "play_time_ms":
            dur = np.maximum(tr.duration_ms, 1.0)
            a = np.clip(a / dur, 0.0, 5.0)
        sd_a, sd_y = float(np.std(a)), float(np.std(y))
        corr = (float(np.corrcoef(a, y)[0, 1]) if sd_a > 0 and sd_y > 0 else float("nan"))
        row = {"corr_with_label": round(corr, 4)}
        if set(np.unique(a)) <= {0.0, 1.0}:
            row["rate"] = round(_rate(a), 5)
            p1 = _rate(y[a == 1]) if np.any(a == 1) else float("nan")
            p0 = _rate(y[a == 0]) if np.any(a == 0) else float("nan")
            row["P_label_given_1"] = round(p1, 4)
            row["P_label_given_0"] = round(p0, 4)
        out[name] = row
    return {"signals": out,
            "reading": ("A signal with |corr| near 1 duplicates the label. One near 0 is "
                        "unrelated. The useful range is in between, and the size of "
                        "P(label|signal=1) - P(label|signal=0) is the discrimination it "
                        "would add.")}


def item_coverage(splits, spec) -> dict:
    """How much of dev the model has actually seen in training.

    An item or user absent from train gets whatever the model does with an unknown id.
    If that share is large, cold-start handling matters more than tuning does.
    """
    tr, dv = splits["train"], splits[spec.split.dev_split]
    tr_items, tr_users = np.unique(tr.video_id), np.unique(tr.user_id)
    seen_i = np.isin(dv.video_id, tr_items)
    seen_u = np.isin(dv.user_id, tr_users)
    return {"dev_rows_with_unseen_item_pct": round(100.0 * float(np.mean(~seen_i)), 3),
            "dev_rows_with_unseen_user_pct": round(100.0 * float(np.mean(~seen_u)), 3),
            "train_items": int(tr_items.size), "train_users": int(tr_users.size),
            "dev_items": int(np.unique(dv.video_id).size),
            "label_rate_seen_item": round(_rate(dv.label[seen_i]), 5) if seen_i.any() else None,
            "label_rate_unseen_item": round(_rate(dv.label[~seen_i]), 5) if (~seen_i).any() else None,
            "reading": ("If the unseen share is small, cold-start work cannot move the "
                        "score much no matter how good it is.")}


def duration_profile(splits, spec) -> dict:
    """Label rate by video-duration decile, on TRAIN.

    Duration is the one continuous item feature, and `long_view` is defined against watch
    time -- so if the label rate varies strongly with duration, that is a first-order
    effect rather than an interaction, and bucketing choices matter.
    """
    tr = splits["train"]
    edges = np.quantile(tr.duration_ms, np.linspace(0, 1, 11)[1:-1])
    b = np.searchsorted(edges, tr.duration_ms)
    rows = {}
    for k in range(int(b.max()) + 1):
        m = b == k
        if m.any():
            rows[f"decile_{k}"] = {"rows": int(m.sum()),
                                   "label_rate": round(_rate(tr.label[m]), 4),
                                   "median_duration_s": round(
                                       float(np.median(tr.duration_ms[m])) / 1000.0, 1)}
    rates = [v["label_rate"] for v in rows.values()]
    return {"by_duration_decile": rows,
            "spread": round(max(rates) - min(rates), 4) if rates else None,
            "reading": ("A large spread means duration alone predicts the label, so the "
                        "bucketing granularity is a real choice, not a detail.")}


#: name -> (function, one-line description shown to the agent)
DIAGNOSTICS = {
    "label_rates": (label_rates,
                    "label base rate on train vs dev, and by date -- detects drift"),
    "group_sizes": (group_sizes,
                    "impressions per user on dev, and the share of users that can carry "
                    "each metric at all"),
    "aux_signals": (aux_signal_overlap,
                    "how each auxiliary outcome relates to the label on train -- which "
                    "ones could add information for multi-task work"),
    "item_coverage": (item_coverage,
                      "share of dev rows whose item or user never appears in train"),
    "duration_profile": (duration_profile,
                         "label rate by video-duration decile on train"),
}


def run(name: str, splits, spec) -> dict:
    """Run one diagnostic. Raises KeyError for an unknown name rather than guessing."""
    if name not in DIAGNOSTICS:
        raise KeyError(f"unknown diagnostic {name!r}; choose from {sorted(DIAGNOSTICS)}")
    held = splits.get(spec.split.held_out_split)
    if held is not None and held.label is not None:
        raise RuntimeError("refusing to run a diagnostic in a process holding held-out "
                           "labels: a diagnostic must not be able to become an oracle")
    return DIAGNOSTICS[name][0](splits, spec)
