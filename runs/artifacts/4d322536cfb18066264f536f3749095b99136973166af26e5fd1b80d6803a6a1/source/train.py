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
    order = np.argsort(users, kind="stable")
    su = users[order]
    starts = np.r_[0, np.flatnonzero(np.diff(su)) + 1, len(su)]
    return order, starts


def compute_user_ranks(users: np.ndarray, preds: np.ndarray) -> np.ndarray:
    order, starts = group_bounds(users)
    ranks = np.empty_like(preds, dtype=np.float64)
    for i in range(len(starts) - 1):
        st, en = starts[i], starts[i + 1]
        if st == en:
            continue
        idx = order[st:en]
        user_preds = preds[idx]
        # Higher score gets a smaller rank number, or we can use argsort twice
        # Let's map to percentile or rank from 0 to 1 within user
        sort_idx = np.argsort(user_preds)
        r = np.empty_like(sort_idx, dtype=np.float64)
        r[sort_idx] = np.arange(en - st, dtype=np.float64)
        if en - st > 1:
            r /= (en - st - 1)
        else:
            r[0] = 0.5
        ranks[idx] = r
    return ranks


def fit_predict(data, params: dict) -> dict:
    cfg = load_config(params)
    loss = cfg["loss"]
    if loss != "pointwise":
        raise ContractError(
            f"unknown loss {loss!r}. This MINIMAL seed implements only 'pointwise'.")

    splits = {"train": data.train, "dev": data.dev, "heldout": data.heldout}
    enc, dim = encode(splits, data.edges, cfg["fields"])
    Xtr, ytr = enc["train"], np.asarray(data.train.label, dtype=np.float64)
    Xdev = enc["dev"]
    udev = np.asarray(data.dev.user_id)
    ydev = np.asarray(data.dev.label, dtype=np.float64)

    sub_idx, sub_hash = train_subset(data)
    Xsub, usub, ysub = Xtr[sub_idx], np.asarray(data.train.user_id)[sub_idx], ytr[sub_idx]

    m = FM(dim, k=cfg["k"], lr=cfg["lr"], l2=cfg["l2"], seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    best, best_state, bad = -1.0, None, 0
    checkpoints = []
    max_top_k = int(cfg.get("ensemble_top_k", 3))

    tel = {"epochs": [], "train_loss": [], "train_primary": [], "dev_primary": [],
           "gap": [], "epoch_seconds": [], "self_reported": True,
           "train_subset_users": int(np.unique(usub).size),
           "train_subset_rows": int(sub_idx.size),
           "train_subset_hash": sub_hash}
    every = int(cfg.get("train_curve_every", 1))

    for ep in range(1, int(cfg["epochs"])+1):
        t_ep = time.perf_counter()
        batch = int(cfg["batch"])
        idx = rng.permutation(len(ytr))
        losses = [m.step_pointwise(Xtr[idx[i:i + batch]], ytr[idx[i:i + batch]])
                  for i in range(0, len(idx), batch)]

        dev_m = primary(udev, ydev, m.predict(Xdev))
        dev_p = dev_m["primary"]
        ml = float(np.mean(losses)) if len(losses) else 0.0
        tr_p = None
        if every > 0 and (ep % every == 0):
            tr_p = primary(usub, ysub, m.predict(Xsub))["primary"]

        tel["epochs"].append(ep)
        tel["train_loss"].append(round(ml, 6) if np.isfinite(ml) else None)
        tel["dev_primary"].append(round(dev_p, 6))
        tel["train_primary"].append(none_or_round(tr_p))
        tel["gap"].append(none_or_round(tr_p - dev_p if tr_p is not None else None))
        tel["epoch_seconds"].append(round(time.perf_counter() - t_ep, 3))

        # Keep track of checkpoints for ensembling
        checkpoints.append((dev_p, m.state()))
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        if len(checkpoints) > max_top_k:
            checkpoints.pop()

        if dev_p > best + 1e-5:
            best, bad = dev_p, 0
            best_state = m.state()
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break

    if best_state is not None:
        m.load(best_state)

    # Ensemble predictions from top K checkpoints using rank averaging within users
    def ensemble_predict(X, users):
        if not checkpoints:
            return m.predict(X)
        accum_ranks = np.zeros(len(X), dtype=np.float64)
        for _, st in checkpoints:
            m.load(st)
            preds = m.predict(X)
            ranks = compute_user_ranks(users, preds)
            accum_ranks += ranks
        accum_ranks /= len(checkpoints)
        return accum_ranks

    dev_preds = ensemble_predict(Xdev, udev)
    heldout_preds = ensemble_predict(enc["heldout"], np.asarray(data.heldout.user_id))
    sub_preds = ensemble_predict(Xsub, usub)

    # Restore best state finally
    if best_state is not None:
        m.load(best_state)

    tel["epochs_run"] = len(tel["epochs"])
    tel["dev_primary_best"] = round(best, 6)
    tel["n_features"] = int(dim)
    return {"dev": dev_preds,
            "heldout": heldout_preds,
            "train_subset": sub_preds,
            "telemetry": tel}


def none_or_round(v):
    return None if v is None else round(v, 6)
