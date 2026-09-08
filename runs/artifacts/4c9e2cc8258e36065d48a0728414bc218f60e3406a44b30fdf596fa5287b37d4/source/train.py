"""The training loop and the contract entry point. AGENT-EDITABLE.

This copy adds a within-user PAIRWISE (BPR) objective alongside the seed's pointwise
logloss. `group_bounds` -- retained but unused in the seed -- is now what builds the
per-user positive/negative index lists the sampler draws from.

Everything the contract cares about is unchanged: `fit_predict(data, params)` returns
`dev`, `heldout` and `train_subset` score vectors in row order, `train_subset` is computed
on the rows the harness named, early stopping reads `data.dev` exactly as the baseline
does, and pairs are sampled from TRAIN rows only (no dev structure enters the fit).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from contract import ContractError, primary, train_subset
from features import encode
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


def build_pair_index(users: np.ndarray, y: np.ndarray) -> dict:
    """Per-user positive / negative TRAIN row indices, flattened for cheap sampling.

    Users whose impressions are all one class carry no within-user ordering signal and
    are skipped -- they are also exactly the users GAUC cannot score.

    Returns flat arrays:
      pos          all usable positive row indices, grouped by user
      pos_slot     for each entry of `pos`, its user's slot in neg_start/neg_count
      neg          all usable negative row indices, grouped by user slot
      neg_start    offset of each slot's negatives inside `neg`
      neg_count    number of negatives in each slot
    """
    order, starts = group_bounds(users)
    pos_chunks, slot_chunks, neg_chunks = [], [], []
    neg_start, neg_count = [], []
    off = 0
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        m = y[rows] > 0.5
        p, n = rows[m], rows[~m]
        if p.size == 0 or n.size == 0:
            continue
        slot = len(neg_start)
        pos_chunks.append(p.astype(np.int64))
        slot_chunks.append(np.full(p.size, slot, dtype=np.int64))
        neg_chunks.append(n.astype(np.int64))
        neg_start.append(off)
        neg_count.append(int(n.size))
        off += int(n.size)
    if not pos_chunks:
        raise ContractError(
            "pairwise loss needs at least one user with both a positive and a negative "
            "train impression; none found. Fall back to loss='pointwise'.")
    return {
        "pos": np.concatenate(pos_chunks),
        "pos_slot": np.concatenate(slot_chunks),
        "neg": np.concatenate(neg_chunks),
        "neg_start": np.asarray(neg_start, dtype=np.int64),
        "neg_count": np.asarray(neg_count, dtype=np.int64),
        "n_users": len(neg_start),
    }


def sample_pairs(pix: dict, rng: np.random.Generator, n: int):
    """n (positive, negative) row-index pairs, both members from the same user.

    The positive is drawn uniformly over all usable positives (standard BPR sampling,
    so a user's weight is its positive count); the negative is drawn uniformly among
    THAT user's negatives.
    """
    i = rng.integers(0, pix["pos"].size, size=n)
    slot = pix["pos_slot"][i]
    cnt = pix["neg_count"][slot]
    j = (rng.random(n) * cnt).astype(np.int64)
    np.minimum(j, cnt - 1, out=j)                     # guard the rng.random()==1.0 edge
    return pix["pos"][i], pix["neg"][pix["neg_start"][slot] + j]


def fit_predict(data, params: dict) -> dict:
    cfg = load_config(params)
    loss = cfg["loss"]
    if loss not in ("pointwise", "pairwise"):
        raise ContractError(
            f"unknown loss {loss!r}. This tree implements 'pointwise' (the kit baseline) "
            f"and 'pairwise' (within-user BPR). Add a step_* method in model.py and a "
            f"branch here for anything else.")

    splits = {"train": data.train, "dev": data.dev, "heldout": data.heldout}
    enc, dim = encode(splits, data.edges, cfg["fields"])
    Xtr, ytr = enc["train"], np.asarray(data.train.label, dtype=np.float64)
    Xdev = enc["dev"]
    udev = np.asarray(data.dev.user_id)
    ydev = np.asarray(data.dev.label, dtype=np.float64)

    # The train half of the overfitting curve, on rows the HARNESS chose.
    sub_idx, sub_hash = train_subset(data)
    Xsub, usub, ysub = Xtr[sub_idx], np.asarray(data.train.user_id)[sub_idx], ytr[sub_idx]

    m = FM(dim, k=cfg["k"], lr=cfg["lr"], l2=cfg["l2"], seed=cfg["seed"])
    # A SECOND generator, seeded the same. Kept separate from the model's init generator
    # so that changing the number of epochs cannot change the initial weights.
    rng = np.random.default_rng(cfg["seed"])
    utr = np.asarray(data.train.user_id)

    pix = None
    n_pairs = 0
    if loss == "pairwise":
        pix = build_pair_index(utr, ytr)
        n_pairs = int(cfg.get("pairs_per_epoch", 0) or 0)
        if n_pairs <= 0:
            # Each pairwise step forwards TWO rows per pair, so half as many pairs as
            # train rows keeps per-epoch cost at parity with the pointwise baseline.
            n_pairs = max(1, len(ytr) // 2)

    best, best_state, bad = -1.0, None, 0
    tel = {"epochs": [], "train_loss": [], "train_primary": [], "dev_primary": [],
           "gap": [], "epoch_seconds": [], "self_reported": True,
           "loss": loss,
           "pairs_per_epoch": int(n_pairs),
           "pairwise_users": (None if pix is None else int(pix["n_users"])),
           "pairwise_positives": (None if pix is None else int(pix["pos"].size)),
           "train_subset_users": int(np.unique(usub).size),
           "train_subset_rows": int(sub_idx.size),
           "train_subset_hash": sub_hash}
    every = int(cfg.get("train_curve_every", 1))

    for ep in range(1, int(cfg["epochs"]) + 1):
        t_ep = time.perf_counter()
        batch = int(cfg["batch"])
        losses = []
        if loss == "pointwise":
            idx = rng.permutation(len(ytr))
            losses = [m.step_pointwise(Xtr[idx[i:i + batch]], ytr[idx[i:i + batch]])
                      for i in range(0, len(idx), batch)]
        else:
            done = 0
            while done < n_pairs:
                bsz = min(batch, n_pairs - done)
                pidx, nidx = sample_pairs(pix, rng, bsz)
                losses.append(m.step_pairwise(Xtr[pidx], Xtr[nidx]))
                done += bsz

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

    if best_state is not None:
        m.load(best_state)

    tel["epochs_run"] = len(tel["epochs"])
    tel["dev_primary_best"] = round(best, 6)
    tel["n_features"] = int(dim)
    return {"dev": m.predict(Xdev),
            "heldout": m.predict(enc["heldout"]),
            "train_subset": m.predict(Xsub),
            "telemetry": tel}
