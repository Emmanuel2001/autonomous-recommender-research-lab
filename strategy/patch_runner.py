"""Runs an LLM-written strategy patch INSIDE the sandbox. The entry point for generated code.

WHY THIS IS SAFE TO RUN AT ALL
------------------------------
This executes code an LLM wrote, against a dataset we are scored on. That is only
defensible because of core/isolation.py: under `seatbelt` the process cannot read the
held-out labels, the ledger, the signing key or the network, and cannot write outside its
own output directory. The sandbox is not insurance here -- it is the precondition.

THE CONTRACT the patch must satisfy
-----------------------------------
    def fit_predict(train, dev, heldout, params) -> dict

`train` / `dev` / `heldout` are deliberately NOT called train/valid/test: the names come
from the TaskSpec, so a patch written against this contract is not KuaiRand-shaped. Each
is an object exposing parallel numpy arrays -- one entry per row, index == row_id:

    .user_id .video_id .author_id .tab   int64
    .duration_ms                          float64
    .date .hourmin                        int64
    .label                                float64 on train and dev; **None on heldout**
    .aux                                  dict[str, float64 array], TRAIN ONLY

Return `{"dev": <1-D float array>, "heldout": <1-D float array>}`, one score per row of
that split, finite, higher = better. Anything else is a refusal with a readable reason,
recorded as an event rather than a crash.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not score. It writes score vectors; the scorer -- a different process, holding
the labels and the signing key -- is still the only thing that produces a number.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

MAX_SECONDS_DEFAULT = 900


class PatchContractError(RuntimeError):
    """The patch ran but did not honour the contract."""


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("agent_patch", path)
    if spec is None or spec.loader is None:
        raise PatchContractError(f"cannot import a module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                    # the generated code runs here
    return mod


def _check(name: str, arr, n: int) -> np.ndarray:
    if arr is None:
        raise PatchContractError(f"fit_predict returned no scores for {name!r}")
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size != n:
        raise PatchContractError(
            f"{name!r}: returned {a.size:,} scores for {n:,} rows. One score per row, "
            f"in row order -- the submission is keyed by position.")
    bad = ~np.isfinite(a)
    if bad.any():
        raise PatchContractError(
            f"{name!r}: {int(bad.sum())} non-finite scores (NaN/Inf), first at row "
            f"{int(np.flatnonzero(bad)[0])}")
    if np.all(a == a[0]):
        raise PatchContractError(
            f"{name!r}: every score is identical ({a[0]:g}). A constant score cannot "
            f"rank anything -- within-group order would be arbitrary.")
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True, help="the generated strategy module")
    ap.add_argument("--task", default="kuairand_pure")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS_DEFAULT)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"patch": str(a.patch), "task": a.task, "ok": False}
    t0 = time.perf_counter()
    try:
        from core.task import get
        spec = get(a.task)
        splits, _aux = spec.loader()
        dev_name, held_name = spec.split.dev_split, spec.split.held_out_split
        train, dev, heldout = splits["train"], splits[dev_name], splits[held_name]
        if heldout.label is not None:
            raise PatchContractError("held-out labels are present in this process; "
                                     "refusing to run generated code against them")

        mod = _load_module(Path(a.patch))
        if not hasattr(mod, "fit_predict"):
            raise PatchContractError(
                "the patch must define fit_predict(train, dev, heldout, params). "
                "See strategy/patch_runner.py for the contract.")

        t_fit = time.perf_counter()
        res = mod.fit_predict(train, dev, heldout, json.loads(a.params))
        fit_s = time.perf_counter() - t_fit
        if not isinstance(res, dict):
            raise PatchContractError(
                f"fit_predict must return a dict with keys 'dev' and 'heldout'; "
                f"got {type(res).__name__}")

        dv = _check("dev", res.get("dev"), len(dev))
        hv = _check("heldout", res.get("heldout"), len(heldout))
        np.save(out / "dev_scores.npy", dv)
        np.save(out / "heldout_scores.npy", hv)
        meta.update(ok=True, dev_rows=int(dv.size), heldout_rows=int(hv.size),
                    fit_seconds=round(fit_s, 2),
                    dev_split=dev_name, heldout_split=held_name,
                    info=getattr(mod, "INFO", None) if isinstance(
                        getattr(mod, "INFO", None), dict) else None)
    except Exception as e:                       # any failure is DATA, not a crash
        tb = traceback.extract_tb(e.__traceback__)
        frame = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else "?"
        meta.update(ok=False, error_type=type(e).__name__,
                    error=str(e)[:900], error_frame=frame,
                    contract_violation=isinstance(e, PatchContractError))
    meta["total_seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
