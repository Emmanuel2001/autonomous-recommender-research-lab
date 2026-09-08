"""The training loop and the contract entry point. AGENT-EDITABLE.

This copy keeps the parent's WITHIN-USER PAIRWISE (BPR) path, sampler, FM
hyper-parameters, dev early stopping, patience, best-checkpoint restore, auto-field
resolution and wall-clock guard EXACTLY as they were. The only change in this candidate is
the INPUT REPRESENTATION, and it lives in features.py + config.json: four WITHIN-USER
relative context columns (`dur_pct_bucket`, `dur_dev_bucket`, `user_nimp_bucket`,
`upos_bucket`) are added to the base field list, so the ten auto-admitted columns the
parent used are unchanged and the difference is attributable to the new columns alone.

WHY WITHIN-USER NORMALISATION AT ALL
------------------------------------
The metric -- mean(GAUC, nDCG@5) -- is computed inside each user, so only variation across
a user's OWN impressions can move it. Absolute buckets give every user the same global
bucket-to-score map; the percentile / median-deviation / session-position encodings put
the feature in exactly the units the objective ranks in.

LEAKAGE
-------
Admission is whitelist-first and blacklist-checked: features.POST_IMPRESSION can never be
admitted, and raw timestamps, date and the logging-policy flag are declined on purpose.
Vocabularies and continuous quantile edges come from TRAIN rows only, as in the seed. The
new within-user statistics are computed from the rows of the split being encoded using
only `user_id`, `duration_ms` and row order -- no labels are touched (heldout has none in
this process) and no train statistic is carried across splits.

WHAT MUST SURVIVE A REWRITE
---------------------------
`fit_predict(data, params)` returning `dev`, `heldout` and `train_subset` score vectors --
see contract.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from contract import ContractError, primary, train_subset
from features import WITHIN_USER_FIELDS, available_columns, encode, resolve_fields
from model import FM

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(params: dict | None = None) -> dict:
    """config.json, overridden by anything the driver passed in `params`."""
    cfg = json.loads(CONFIG_PATH.read_text())
    if params:
        unknown = sorted(set(params) - set(cfg))
        if unknown:
            raise ContractError(
                f"params override unknown config keys {unknown}; config.json defines "
                f"{sorted(cfg)}. Add the key to config.json in the same edit.")
        cfg.update(params)
    return cfg


def group_bounds(users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(order, starts) for rows grouped by user. `order` sorts rows by user, stably."""
    order = np.argsort(users, kind="stable")
    su = users[order]
    starts = np.r_[0, np.flatnonzero(np.diff(su)) + 1, len(su)]
    return order, starts


def build_pair_index(users: np.ndarray, labels: np.ndarray):
    """Static part of the within-user pair sampler.

    Returns (pos_rows, neg_start, neg_count, n_users_used):
      pos_rows[i]  -- train row index of the i-th positive of a mixed-label user
      neg_start[i] -- position in `order` where that user's negatives begin
      neg_count[i] -- how many negatives that user has
    plus `order`, the (user, label) lexsort permutation the offsets index into.
    """
    y = (np.asarray(labels) > 0.5).astype(np.int64)
    # lexsort keys are ordered last-key-major: sorts by user, then by label within user.
    order = np.lexsort((y, np.asarray(users)))
    su = np.asarray(users)[order]
    sy = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(su)) + 1, len(su)]
    gs, ge = starts[:-1], starts[1:]

    cum = np.r_[0, np.cumsum(sy)]
    npos = (cum[ge] - cum[gs]).astype(np.int64)
    nneg = (ge - gs).astype(np.int64) - npos
    keep = (npos > 0) & (nneg > 0)
    gs_v, npos_v, nneg_v = gs[keep].astype(np.int64), npos[keep], nneg[keep]

    tot = int(npos_v.sum())
    if tot == 0:
        return order, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), 0

    pos_block_start = gs_v + nneg_v                      # positives follow the negatives
    rep_start = np.repeat(pos_block_start, npos_v)
    run_base = np.repeat(np.r_[0, np.cumsum(npos_v)[:-1]], npos_v)
    offs = np.arange(tot, dtype=np.int64) - run_base
    pos_rows = order[rep_start + offs]

    neg_start = np.repeat(gs_v, npos_v)
    neg_count = np.repeat(nneg_v, npos_v)
    return order, pos_rows, neg_start, neg_count, int(keep.sum())


def fit_predict(data, params: dict) -> dict:
    t_start = time.perf_counter()
    cfg = load_config(params)
    loss = cfg["loss"]
    if loss not in ("pointwise", "bpr"):
        raise ContractError(
            f"unknown loss {loss!r}. This candidate implements 'pointwise' (the seed) and "
            f"'bpr' (within-user pairwise). Add a step_* method in model.py and a branch "
            f"here for anything else.")

    splits = {"train": data.train, "dev": data.dev, "heldout": data.heldout}

    # ---------------------------------------------------------- representation
    cont_bins = int(cfg.get("cont_bins", 16))
    base_fields = tuple(cfg["fields"])
    fields, admitted, field_report = base_fields, [], []
    if bool(cfg.get("auto_fields", False)):
        fields, admitted, field_report = resolve_fields(
            data.train, data.edges, base_fields,
            max_card=int(cfg.get("max_field_cardinality", 2000)),
            max_new=int(cfg.get("max_new_fields", 10)),
            cont_bins=cont_bins,
            exclude=tuple(cfg.get("exclude_fields") or ()))

    t_enc = time.perf_counter()
    enc, dim = encode(splits, data.edges, fields, cont_bins=cont_bins)
    encode_seconds = time.perf_counter() - t_enc
    Xtr, ytr = enc["train"], np.asarray(data.train.label, dtype=np.float64)
    Xdev = enc["dev"]
    udev = np.asarray(data.dev.user_id)
    ydev = np.asarray(data.dev.label, dtype=np.float64)
    utr = np.asarray(data.train.user_id)

    # The train half of the overfitting curve, on rows the HARNESS chose.
    sub_idx, sub_hash = train_subset(data)
    Xsub, usub, ysub = Xtr[sub_idx], utr[sub_idx], ytr[sub_idx]

    m = FM(dim, k=cfg["k"], lr=cfg["lr"], l2=cfg["l2"], seed=cfg["seed"])
    # A SECOND generator, seeded the same. Kept separate from the model's init generator
    # so that changing the number of epochs cannot change the initial weights.
    rng = np.random.default_rng(cfg["seed"])

    ppp = max(1, int(cfg.get("pairs_per_pos", 1)))
    pos_rows = neg_start = neg_count = None
    n_pair_users = 0
    if loss == "bpr":
        _order, pos_rows, neg_start, neg_count, n_pair_users = build_pair_index(utr, ytr)
        if len(pos_rows) == 0:
            raise ContractError(
                "no within-user positive/negative pairs exist in train; the BPR objective "
                "has nothing to order. Fall back to loss='pointwise'.")
        if ppp > 1:
            pos_rows = np.tile(pos_rows, ppp)
            neg_start = np.tile(neg_start, ppp)
            neg_count = np.tile(neg_count, ppp)
        neg_lookup = _order

    best, best_state, bad = -1.0, None, 0
    tel = {"loss": loss, "epochs": [], "train_loss": [], "train_primary": [],
           "dev_primary": [], "gap": [], "epoch_seconds": [], "self_reported": True,
           "train_subset_users": int(np.unique(usub).size),
           "train_subset_rows": int(sub_idx.size),
           "train_subset_hash": sub_hash,
           "auto_fields": bool(cfg.get("auto_fields", False)),
           "base_fields": list(base_fields),
           "fields_used": list(fields),
           "fields_admitted": list(admitted),
           "n_fields": int(len(fields)),
           "within_user_fields": [f for f in fields if f in WITHIN_USER_FIELDS],
           "encode_seconds": round(encode_seconds, 2),
           "field_report": field_report,
           "dataset_columns": available_columns(data.train)}
    if loss == "bpr":
        tel["pairs_per_epoch"] = int(len(pos_rows))
        tel["pair_users"] = int(n_pair_users)
        tel["pairs_per_pos"] = ppp
    every = int(cfg.get("train_curve_every", 1))
    max_seconds = float(cfg.get("max_seconds", 0) or 0)
    stopped_on_time = False

    for ep in range(1, int(cfg["epochs"]) + 1):
        t_ep = time.perf_counter()
        batch = int(cfg["batch"])

        if loss == "pointwise":
            idx = rng.permutation(len(ytr))
            losses = [m.step_pointwise(Xtr[idx[i:i + batch]], ytr[idx[i:i + batch]])
                      for i in range(0, len(idx), batch)]
        else:
            # Fresh negative for every positive, uniform within that positive's own user.
            u = rng.random(len(pos_rows))
            draw = neg_start + np.minimum((u * neg_count).astype(np.int64), neg_count - 1)
            neg_rows = neg_lookup[draw]
            perm = rng.permutation(len(pos_rows))
            p_rows, n_rows = pos_rows[perm], neg_rows[perm]
            losses = [m.step_bpr(Xtr[p_rows[i:i + batch]], Xtr[n_rows[i:i + batch]])
                      for i in range(0, len(p_rows), batch)]

        dev_m = primary(udev, ydev, m.predict(Xdev))
        dev_p = dev_m["primary"]
        # Recorded as None rather than NaN: json.dumps writes a bare NaN, which is not
        # valid JSON and would break every reader of the record it lands in.
        ml = float(np.mean(losses)) if len(losses) else 0.0
        tr_p = None
        if every > 0 and (ep % every == 0):
            tr_p = primary(usub, ysub, m.predict(Xsub))["primary"]
        tel["epochs"].append(ep)
        tel["train_loss"].append(round(ml, 6) if np.isfinite(ml) else None)
        tel["dev_primary"].append(round(dev_p, 6))
        tel["train_primary"].append(None if tr_p is None else round(tr_p, 6))
        tel["gap"].append(None if tr_p is None else round(tr_p - dev_p, 6))
        tel["epoch_seconds"].append(round(time.perf_counter() - t_ep, 3))

        if dev_p > best + 1e-5:
            best, bad = dev_p, 0
            best_state = m.state()
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break

        if max_seconds > 0 and (time.perf_counter() - t_start) > max_seconds:
            stopped_on_time = True
            break

    if best_state is not None:
        m.load(best_state)

    tel["epochs_run"] = len(tel["epochs"])
    tel["stopped_on_time"] = bool(stopped_on_time)
    tel["dev_primary_best"] = round(best, 6)
    tel["n_features"] = int(dim)
    tel["fit_seconds"] = round(time.perf_counter() - t_start, 2)
    return {"dev": m.predict(Xdev),
            "heldout": m.predict(enc["heldout"]),
            "train_subset": m.predict(Xsub),
            "telemetry": tel}
