"""Device-portable PyTorch training loop. AGENT-EDITABLE.

The harness sets TRACK2_DEVICE after measuring hardware.  The candidate never guesses that
CUDA or MPS exists and never silently changes devices itself.
"""
# requires: torch
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from contract import ContractError, primary, train_subset
from features import encode
from model import TorchFM

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(params=None):
    cfg = json.loads(CONFIG_PATH.read_text())
    if params:
        unknown = sorted(set(params) - set(cfg))
        if unknown:
            raise ContractError(f"unknown config overrides {unknown}")
        cfg.update(params)
    return cfg


def group_pairs(users, labels, rng, neg_per_pos):
    """Within-user positive/negative pairs for the ranking-aligned branch."""
    order = np.argsort(users, kind="stable")
    su = users[order]
    starts = np.r_[0, np.flatnonzero(np.diff(su)) + 1, len(su)]
    pos_all, neg_all = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        pos, neg = idx[labels[idx] == 1], idx[labels[idx] == 0]
        if not len(pos) or not len(neg):
            continue
        pos = np.repeat(pos, int(neg_per_pos))
        pos_all.append(pos)
        neg_all.append(rng.choice(neg, size=len(pos), replace=True))
    if not pos_all:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(pos_all), np.concatenate(neg_all)


def fit_predict(data, params):
    cfg = load_config(params)
    device = os.environ.get("TRACK2_DEVICE", "cpu")
    if device not in ("cpu", "mps", "cuda"):
        raise ContractError(f"harness supplied unsupported device {device!r}")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ContractError("harness selected MPS but torch reports it unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ContractError("harness selected CUDA but torch reports it unavailable")

    splits = {"train": data.train, "dev": data.dev, "heldout": data.heldout}
    enc, dim = encode(splits, data.edges, cfg["fields"])
    xtr = enc["train"]
    ytr = np.asarray(data.train.label, dtype=np.float32)
    sub_idx, sub_hash = train_subset(data)
    xsub = xtr[sub_idx]
    usub = np.asarray(data.train.user_id)[sub_idx]
    ysub = ytr[sub_idx]
    udev = np.asarray(data.dev.user_id)
    ydev = np.asarray(data.dev.label, dtype=np.float32)

    torch.manual_seed(int(cfg["seed"]))
    model = TorchFM(dim, cfg["k"], cfg["seed"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                           weight_decay=float(cfg["l2"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    best, best_state, bad = -float("inf"), None, 0
    tel = {"epochs": [], "train_loss": [], "train_primary": [], "dev_primary": [],
           "gap": [], "epoch_seconds": [], "self_reported": True,
           "framework": "torch", "device": device,
           "train_subset_hash": sub_hash}

    for epoch in range(1, int(cfg["epochs"]) + 1):
        started = time.perf_counter()
        losses = []
        if cfg["loss"] == "pointwise":
            left = rng.permutation(len(ytr))
            batches = ((left[i:i + int(cfg["batch"])], None)
                       for i in range(0, len(left), int(cfg["batch"])))
        elif cfg["loss"] == "pairwise":
            p, n = group_pairs(np.asarray(data.train.user_id), ytr, rng,
                               cfg["neg_per_pos"])
            perm = rng.permutation(len(p))
            batches = ((p[perm[i:i + int(cfg["batch"])]],
                        n[perm[i:i + int(cfg["batch"])]])
                       for i in range(0, len(perm), int(cfg["batch"])))
        else:
            raise ContractError(f"unknown loss {cfg['loss']!r}")

        model.train()
        for a, b in batches:
            opt.zero_grad(set_to_none=True)
            xa = torch.as_tensor(xtr[a], dtype=torch.long, device=device)
            za = model(xa)
            if b is None:
                yy = torch.as_tensor(ytr[a], dtype=torch.float32, device=device)
                loss = F.binary_cross_entropy_with_logits(za, yy)
            else:
                xb = torch.as_tensor(xtr[b], dtype=torch.long, device=device)
                loss = -F.logsigmoid(za - model(xb)).mean()
            loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))

        dev_scores = model.predict_numpy(enc["dev"], device)
        dev_p = primary(udev, ydev, dev_scores)["primary"]
        train_scores = model.predict_numpy(xsub, device)
        train_p = primary(usub, ysub, train_scores)["primary"]
        tel["epochs"].append(epoch)
        tel["train_loss"].append(round(float(np.mean(losses)) if losses else 0.0, 6))
        tel["train_primary"].append(round(train_p, 6))
        tel["dev_primary"].append(round(dev_p, 6))
        tel["gap"].append(round(train_p - dev_p, 6))
        tel["epoch_seconds"].append(round(time.perf_counter() - started, 3))
        if dev_p > best + 1e-5:
            best, bad = dev_p, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best_state:
        model.load_state_dict(best_state)
    tel.update(epochs_run=len(tel["epochs"]), dev_primary_best=round(best, 6),
               n_features=int(dim))
    return {"dev": model.predict_numpy(enc["dev"], device),
            "heldout": model.predict_numpy(enc["heldout"], device),
            "train_subset": model.predict_numpy(xsub, device), "telemetry": tel}
