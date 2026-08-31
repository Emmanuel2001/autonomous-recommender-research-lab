"""What the strategy process runs. The entry point inside the sandbox.

Reads ONLY the files a sandboxed process is permitted to open (see
core/isolation.py) and writes exactly one file: the predictions npz. Emits one
JSON object on stdout, which is the entire channel back to the driver.

WHY THE CONFIG ARRIVES AS JSON
------------------------------
This used to take one flag per parameter, and the flags covered nine of Config's
twenty-two fields. `content_alpha` and `signal_weights` -- the two axes that
produced the only real improvement measured so far -- had no flag at all, so an
isolated run would have silently trained a DIFFERENT model from the in-process
run of the same config and reported it under the same config hash. Passing the
whole dataclass removes the class of bug rather than adding two more flags.

WHY IT RANKS EVERY USER
-----------------------
The scoreable user set is derived from validation labels. Rather than be told it,
this process ranks all 27,285 users and lets the scorer keep the ones it scores;
the metrics are identical. See core.splits.load_train_only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core import splits as S
from strategy.mle_strategy import Config, run

# Fields that must be restored to a tuple after a JSON round-trip. JSON has no
# tuple, and Config.hash() serialises tuples and lists identically, so the hash is
# unaffected -- but code that does `set(cfg.content_fields)` should not start
# seeing a list. Asserted in tests/test_isolation.py.
_TUPLE_FIELDS = {k for k, v in vars(Config()).items() if isinstance(v, tuple)}


def config_from_json(blob: str) -> Config:
    d = json.loads(blob)
    known = {f.name for f in fields(Config)}
    unknown = set(d) - known
    if unknown:
        raise SystemExit(f"unknown config fields: {sorted(unknown)}")
    for k in _TUPLE_FIELDS & set(d):
        d[k] = tuple(d[k])
    return Config(**d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", required=True,
                    help="the full Config as a JSON object")
    ap.add_argument("--split", default="search_val",
                    help="recorded only; this process ranks every user either way")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = config_from_json(a.config_json)

    t0 = time.perf_counter()
    train, meta = S.load_train_only(use_log_random=cfg.use_log_random)
    t_load = time.perf_counter() - t0

    users = meta["rank_users"]
    t0 = time.perf_counter()
    ranked, info = run(train, users, cfg, meta["n_items_catalog"],
                       n_users=meta["n_users_cap"], device=a.device)
    t_fit = time.perf_counter() - t0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, users=users, items=ranked)

    print(json.dumps({
        "preds": str(out),
        "config_hash": cfg.hash(),
        "config": asdict(cfg),
        "split": a.split,
        "mask": cfg.mask,
        "n_users_ranked": int(users.size),
        "n_items_catalog": meta["n_items_catalog"],
        "data_root": meta["data_root"],
        "sandboxed": os.environ.get("TRACK2_SANDBOX") == "1",
        "seconds": {"load": round(t_load, 2), "fit_and_rank": round(t_fit, 2)},
        **{k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))},
    }))


if __name__ == "__main__":
    main()
