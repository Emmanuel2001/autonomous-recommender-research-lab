from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from contract import ContractError, primary, train_subset
from features import encode
from model import DeepFM

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


def fit_predict(data, params: dict) -> dict:
    cfg = load_config(params)
    loss = cfg["loss"]
    if loss != "pointwise":
        raise ContractError(
            f"unknown loss {loss!r}. This DeepFM model implements only 'pointwise'.")

    splits = {"train": data.train, "dev": data.dev, "heldout": data.heldout}
    enc, dim = encode(splits, data.edges, cfg["fields"])
    Xtr, ytr = enc["train"], np.asarray(data.train.label, dtype=np.float64)
    Xdev = enc["dev"]
    udev = np.asarray(data.dev.user_id)
    ydev = np.asarray(data.dev.label, dtype=np.float64)

    sub_idx, sub_hash = train_subset(data)
    Xsub, usub, ysub = Xtr[sub_idx], np.asarray(data.train.user_id)[sub_idx], ytr[sub_idx]

    m = DeepFM(dim=dim, k=cfg["k"], hidden_dims=cfg.get("hidden_dims", [64, 32]),
               dropout=cfg.get("dropout", 0.2), l2=cfg["l2"], lr=cfg["lr"], seed=cfg["seed"])

    rng = np.random.default_rng(cfg["seed"])

    best, best_state, bad = -1.0, None, 0
    tel = {"epochs": [], "train_loss": [], "train_primary": [], "dev_primary": [],
           "gap": [], "epoch_seconds": [], "self_reported": True,
           "train_subset_users": int(np.unique(usub).size),
           "train_subset_rows": int(sub_idx.size),
           "train_subset_hash": sub_hash}
    every = int(cfg.get("train_curve_every", 1))

    for ep in range(1, int(cfg["epochs"]) + 1):
        t_ep = time.perf_counter()
        batch = int(cfg["batch"])
        idx = rng.permutation(len(ytr))
        losses = [m.step_pointwise(Xtr[idx[i:i + batch]], ytr[idx[i:i + batch]])
                  for i in range(0, len(idx), batch)]

        dev_preds = m.predict_tensor(Xdev)
        dev_m = primary(udev, ydev, dev_preds)
        dev_p = dev_m["primary"]

        ml = float(np.mean(losses)) if len(losses) else 0.0
        tr_p = None
        if every > 0 and (ep % every == 0):
            tr_preds = m.predict_tensor(Xsub)
            tr_p = primary(usub, ysub, tr_preds)["primary"]

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
    return {"dev": m.predict_tensor(Xdev),
            "heldout": m.predict_tensor(enc["heldout"]),
            "train_subset": m.predict_tensor(Xsub),
            "telemetry": tel}
