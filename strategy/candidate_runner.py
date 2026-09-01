"""Runs an agent-written CANDIDATE TREE inside the sandbox. Entry point for generated code.

Successor to strategy/patch_runner.py, which took one module and three positional row
objects. Three things changed and each one was a real limitation:

  * a candidate is now a TREE (features/model/train/config), so an architecture change and
    the feature change it needs are one candidate rather than one file doing everything;
  * the row objects are wrapped in `contract.Data`, which carries `edges` and exposes
    `columns`. patch_runner loaded `(splits, _aux)` and DROPPED the second element, so
    generated code could not build the baseline's own `dur_bucket` feature;
  * `train_subset` scores are required back, so the harness computes the train/dev gap
    itself with the official metric instead of trusting a self-reported curve.

WHY THIS IS SAFE TO RUN
-----------------------
core/isolation.py: under `seatbelt` this process cannot read the held-out labels, the
ledger, the signing key or the network, and -- since agent/candidate.py passes
`out_dir=<this candidate's own directory>` -- cannot write into a sibling candidate's
directory either. The held-out labels are additionally absent from the loaded data, and
that is asserted, not assumed.

It does not score. It writes score vectors; the scorer, a separate process holding the
labels and the signing key, is the only thing that produces a signed number.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np

MAX_SECONDS_DEFAULT = 900


def _import_entry(cand_dir: Path, workspace: Path, task_primary=None):
    """Import the candidate's train.py with the candidate's own modules taking priority.

    Order matters: `cand_dir` first so `features`/`model` resolve to the candidate's
    copies, `workspace` next for the immutable `contract`, ROOT last for `core`. The
    candidate cannot shadow `contract` -- agent/candidate.py refuses a tree containing it.
    """
    for p in (str(ROOT), str(workspace), str(cand_dir)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(cand_dir))
    # Drop anything a previous candidate left in the module cache in this process.
    for name in ("train", "features", "model", "contract"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    # Candidate seeds import `primary` from the immutable contract for early stopping.
    # Bind that name to the active TaskSpec before train.py imports it.  The scorer still
    # owns the authoritative number; this is only the public-dev feedback candidates need
    # to train correctly on a non-KuaiRand task.
    contract = importlib.import_module("contract")
    if task_primary is not None:
        contract.primary = task_primary
    return importlib.import_module("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="directory holding the tree")
    ap.add_argument("--workspace", required=True, help="directory holding contract.py")
    ap.add_argument("--task", default="kuairand_pure")
    ap.add_argument("--task-spec", default=None,
                    help="confirmed portable JSON TaskSpec; absent for registered tasks")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS_DEFAULT)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"candidate": str(a.candidate), "task": a.task, "ok": False,
            "execution": {
                "framework": os.environ.get("TRACK2_FRAMEWORK", "numpy"),
                "device": os.environ.get("TRACK2_DEVICE", "cpu"),
            }}
    t0 = time.perf_counter()
    try:
        from core.task import load
        spec = load(a.task_spec or a.task)
        splits, edges = spec.loader()
        dev_name, held_name = spec.split.dev_split, spec.split.held_out_split
        train, dev, heldout = splits["train"], splits[dev_name], splits[held_name]
        if heldout.label is not None:
            raise RuntimeError("held-out labels are present in this process; refusing to "
                               "run generated code against them")
        if dev.label is None:
            raise RuntimeError(f"the {dev_name!r} split has no labels; a candidate cannot "
                               f"early-stop without them")

        sys.path.insert(0, str(Path(a.workspace)))
        from contract import ContractError, Data, Spec, check_scores, train_subset

        data = Data(train=train, dev=dev, heldout=heldout,
                    aux=dict(getattr(train, "aux", {}) or {}), edges=edges,
                    spec=Spec(task=a.task, dev_split=dev_name,
                              held_out_split=held_name, label=spec.label))
        sub_idx, sub_hash = train_subset(data)

        # `k` must be FORWARDED. The contract's primary() and core.metrics.primary() both
        # take k and pass it on; an earlier wrapper accepted k and dropped it, so a candidate
        # doing nDCG@10 early stopping silently got the nDCG@5 number instead (measured:
        # 0.5429 vs 0.4463 on a synthetic split -- a 0.097 discrepancy, on the official task
        # too, since this wrapper is installed for every task). Portable metrics such as
        # global AUC take no k, so forward it only where the signature accepts it.
        import inspect as _inspect
        try:
            _takes_k = "k" in _inspect.signature(spec.metric.fn).parameters
        except (TypeError, ValueError):
            _takes_k = False

        def task_primary(group, labels, scores, k=None):
            fn = spec.metric.fn
            if _takes_k:
                measured = dict(fn(group, labels, scores, k=(5 if k is None else k)))
            else:
                # MetricSpec.fn is contractually (ids, labels, scores) -- the official task's
                # is a 3-arg lambda. Silently returning the k=5 number for a candidate that
                # asked for k=10 is the failure mode being closed here; refuse instead.
                if k is not None and int(k) != 5:
                    raise ContractError(
                        f"this task's metric is fixed at its published cutoff and cannot "
                        f"honour k={k}. The scorer's number is the graded one; a candidate "
                        f"must not early-stop on a different cutoff than it reports.")
                measured = dict(fn(group, labels, scores))
            measured.setdefault("primary", measured[spec.metric.primary_key])
            return measured

        mod = _import_entry(Path(a.candidate), Path(a.workspace), task_primary)
        entry = getattr(mod, "fit_predict", None)
        if entry is None or not callable(entry):
            raise ContractError(
                "train.py must define fit_predict(data, params). See contract.py.")

        t_fit = time.perf_counter()
        res = entry(data, json.loads(a.params))
        fit_s = time.perf_counter() - t_fit
        if not isinstance(res, dict):
            raise ContractError(
                f"fit_predict must return a dict with 'dev', 'heldout' and "
                f"'train_subset'; got {type(res).__name__}")

        dv = check_scores("dev", res.get("dev"), len(dev))
        hv = check_scores("heldout", res.get("heldout"), len(heldout))
        # Not optional: this is the train half of the overfitting curve, and it is scored
        # by the harness rather than reported by the candidate.
        sv = check_scores("train_subset", res.get("train_subset"), int(sub_idx.size))

        np.save(out / "dev_scores.npy", dv)
        np.save(out / "heldout_scores.npy", hv)
        np.save(out / "train_subset_scores.npy", sv)
        np.save(out / "train_subset_index.npy", np.ascontiguousarray(sub_idx, np.int64))

        tel = res.get("telemetry")
        if tel is not None:
            try:
                (out / "telemetry.json").write_text(
                    json.dumps(tel, indent=2, sort_keys=True, allow_nan=False,
                               default=str) + "\n")
            except (TypeError, ValueError) as e:
                # Telemetry is a courtesy; a candidate must not fail its whole fit
                # because a numpy scalar could not be serialised. Recorded, not hidden.
                meta["telemetry_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        # The harness can prove which device it made available to the candidate, but an
        # arbitrary generated program can still ignore TRACK2_DEVICE.  Preserve the
        # candidate's own claim separately; never relabel the resolved target as observed
        # execution.  A missing claim is useful evidence (and was exposed by a generated
        # DeepFM which received mps but created every tensor on CPU).
        if isinstance(tel, dict):
            claimed_framework = tel.get("framework")
            claimed_device = tel.get("device")
            if claimed_framework is not None:
                meta["execution"]["candidate_reported_framework"] = str(claimed_framework)
            if claimed_device is not None:
                meta["execution"]["candidate_reported_device"] = str(claimed_device)
        target_framework = meta["execution"]["framework"]
        target_device = meta["execution"]["device"]
        claimed_framework = meta["execution"].get("candidate_reported_framework")
        claimed_device = meta["execution"].get("candidate_reported_device")
        if claimed_framework is None or claimed_device is None:
            meta["execution"]["candidate_report_status"] = "unreported"
        elif claimed_framework == target_framework and claimed_device == target_device:
            meta["execution"]["candidate_report_status"] = "reported_match"
        else:
            meta["execution"]["candidate_report_status"] = "reported_mismatch"
        meta["execution"]["candidate_report_note"] = (
            "candidate telemetry is self-reported; the harness only attests the resolved target")
        meta.update(ok=True, dev_rows=int(dv.size), heldout_rows=int(hv.size),
                    train_subset_rows=int(sv.size), train_subset_hash=sub_hash,
                    fit_seconds=round(fit_s, 2), dev_split=dev_name,
                    heldout_split=held_name,
                    has_telemetry=bool(tel),
                    info=getattr(mod, "INFO", None) if isinstance(
                        getattr(mod, "INFO", None), dict) else None)
    except Exception as e:                        # any failure is DATA, not a crash
        tb = traceback.extract_tb(e.__traceback__)
        # Report the deepest frame that belongs to the CANDIDATE, not the runner: the
        # agent needs the line in the file it wrote, and the last frame is often numpy.
        cand = Path(a.candidate).resolve()
        own = [f for f in tb if str(Path(f.filename).resolve().parent) == str(cand)]
        pick = own[-1] if own else (tb[-1] if tb else None)
        # WHOSE code failed. A contract violation detected by the harness's own return-value
        # check has no candidate frame at all, and reporting the harness's line number as
        # though it were the agent's is actively misleading during a repair.
        meta.update(ok=False, error_type=type(e).__name__, error=str(e)[:900],
                    error_in="candidate" if own else "harness_check",
                    error_frame=f"{Path(pick.filename).name}:{pick.lineno}" if pick else "?",
                    error_line=(pick.line or "").strip()[:200] if pick else "",
                    contract_violation=type(e).__name__ == "ContractError",
                    traceback="".join(traceback.format_exception(
                        type(e), e, e.__traceback__))[-2500:])
    meta["total_seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
