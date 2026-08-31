"""The training loop and the contract entry point. AGENT-EDITABLE.

MINIMAL PROFILE. Pointwise logloss only -- no pairwise sampler, no listwise batching. The
`group_bounds` helper is retained because a within-user objective will need it, but nothing
here uses it. Any ranking-aligned loss must be written by the agent.

Seeded with the official baseline's procedure: minibatch Adam over shuffled impressions,
early stopping on the dev ranking metric with patience, and the best-dev checkpoint
restored before predicting. Changing the objective, the sampling, the schedule, or adding
a second stage all happen here.

WHAT MUST SURVIVE A REWRITE
---------------------------
`fit_predict(data, params)` returning `dev`, `heldout` and `train_subset` score vectors --
see contract.py. `train_subset` is what makes overfitting visible, and it is computed on
rows the harness names, so it cannot drift into a self-flattering subset.

Early stopping reads `data.dev`. That is the public validation split and selecting an
epoch on it is ordinary practice, but it is also why the train/dev GAP matters: a
candidate can buy dev primary with capacity and lose it on the held-out split, and the gap
is the only in-run warning of that.
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


def fit_predict(data, params: dict) -> dict:
    cfg = load_config(params)
    loss = cfg["loss"]
    if loss != "pointwise":
        raise ContractError(
            f"unknown loss {loss!r}. This MINIMAL seed implements only 'pointwise'. A "
            f"ranking-aligned objective is a legitimate hypothesis, but it has to be "
            f"written: add a step_* method in model.py and a branch here, then set it in "
            f"config.json. Nothing is pre-supplied.")

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
    order, starts = group_bounds(np.asarray(data.train.user_id))

    best, best_state, bad = -1.0, None, 0
    tel = {"epochs": [], "train_loss": [], "train_primary": [], "dev_primary": [],
           "gap": [], "epoch_seconds": [], "self_reported": True,
           "train_subset_users": int(np.unique(usub).size),
           "train_subset_rows": int(sub_idx.size),
           "train_subset_hash": sub_hash}
    every = int(cfg.get("train_curve_every", 1))
    max_epochs = int(cfg["epochs"])
    lr_init = float(cfg["lr"])
    lr_min = float(cfg.get("lr_min", 1e-5))
    use_cosine = cfg.get("scheduler", None) == "cosine"

    for ep in range(1, max_epochs + 1):
        t_ep = time.perf_counter()
        batch = int(cfg["batch"])

        if use_cosine:
            # Cosine annealing schedule
            progress = (ep - 1) / max(1, max_epochs - 1)
            m.lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * progress))

        idx = rng.permutation(len(ytr))
        losses = [m.step_pointwise(Xtr[idx[i:i + batch]], ytr[idx[i:i + batch]])
                  for i in range(0, len(idx), batch)]

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
