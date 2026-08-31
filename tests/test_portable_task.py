"""End-to-end proof that the platform runs a second declared recommender task."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import candidate as CAND
from core import isolation as ISO
from core.ledger import Ledger, verify
from core.task import load
from core.task_scorer import score_task

R = []


def check(name, condition, extra=""):
    ok = bool(condition)
    R.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


FIXTURE = ROOT / "tests" / "fixtures" / "portable_recommender" / "task.json"
spec = load(FIXTURE)
strategy, _ = spec.loader()
scorer, _ = spec.scorer_loader()

check("portable JSON becomes a TaskSpec", spec.slug == "shop_clicks_demo")
check("custom feature names survive", hasattr(strategy["train"], "account_id"))
check("canonical user alias is added", np.array_equal(
    strategy["train"].user_id, strategy["train"].account_id))
check("canonical item alias is added", np.array_equal(
    strategy["train"].video_id, strategy["train"].product_id))
check("strategy sees train labels", strategy["train"].label is not None)
check("strategy sees public-dev labels", strategy["valid"].label is not None)
check("strategy cannot see held-out labels", strategy["test"].label is None)
check("scorer alone sees held-out labels", scorer["test"].label is not None)
check("strategy/scorer content identity ignores quarantined outcomes",
      strategy["test"].content_sha256() == scorer["test"].content_sha256())
check("task-specific convergence is loaded",
      spec.convergence["epsilon"] == 0.01 and spec.convergence["N"] == 2)

# A deterministic score vector proves the declared metric can score original column names.
probe = np.linspace(-1.0, 1.0, len(scorer["valid"]))
metric = spec.score(scorer["valid"], probe)
check("declared within-account metric executes", set(metric) >= {"GAUC", "nDCG@5", "primary"})

with tempfile.TemporaryDirectory(prefix="track2-portable-") as td:
    temp = Path(td)
    prepared = ISO.prepare_portable_data_root(spec, temp / "redacted")
    with (prepared.path / "events.csv").open(newline="") as fh:
        redacted = list(csv.DictReader(fh))
    held = [r for r in redacted if int(r["day"]) == 6]
    train = [r for r in redacted if int(r["day"]) <= 3]
    check("held-out outcomes are physically redacted", held and
          all(r["converted"] == "0" for r in held))
    check("train outcomes remain available", any(r["converted"] == "1" for r in train))
    check("original portable data root is kernel-denied",
          str(spec.data_dir.resolve()) in prepared.deny_paths)

    files = CAND.seed_tree(CAND.WORKSPACE)
    report = CAND.run_candidate(
        files,
        {"fields": ["account_id", "product_id", "context"],
         "epochs": 4, "batch": 16, "patience": 2},
        run_dir=temp / "candidate",
        task=spec.slug,
        backend="none",
        max_seconds=120,
        artifact_root=temp / "artifacts",
        workspace=CAND.WORKSPACE,
        execution_profile={"framework": "numpy", "device": "cpu",
                           "accelerator_policy": "off"},
        task_spec_path=FIXTURE)
    check("same candidate harness trains on second dataset", report.get("ok"),
          json.dumps({k: report.get(k) for k in ("error_type", "error")}))
    if report.get("ok"):
        source_path = Path(report["candidate"]).resolve()
        check("candidate source is outside its writable output directory",
              not source_path.is_relative_to(Path(report["run_dir"]).resolve()))
        dev_scores = np.load(report["dev_scores"])
        check("portable candidate returns one score per dev row",
              dev_scores.shape == (len(strategy["valid"]),))
        check("portable candidate used declared CPU execution",
              report["execution"]["device"] == "cpu")

        key = b"portable-task-test-key"
        ledger_path = temp / "portable.jsonl"
        ledger = Ledger(ledger_path, key)
        entry = score_task(report["dev_scores"], "valid", scorer["valid"], spec,
                           report["tree"], ledger, isolation="none")
        payload = entry["payload"]
        check("generic scorer signs task-owned metric",
              payload["task"] == spec.slug and "primary" in payload["metrics"])
        check("generic scorer reports declared-baseline delta",
              "delta_vs_declared_baseline" in payload)
        check("portable score ledger verifies", verify(ledger_path, key).ok)

    accelerated = CAND.run_candidate(
        CAND.seed_tree(CAND.WORKSPACE_ACCELERATED),
        {"fields": ["account_id", "product_id", "context"],
         "epochs": 3, "batch": 16, "patience": 2},
        run_dir=temp / "accelerated_candidate",
        task=spec.slug,
        backend="none",
        max_seconds=120,
        artifact_root=temp / "accelerated_artifacts",
        workspace=CAND.WORKSPACE_ACCELERATED,
        execution_profile={"framework": "torch", "device": "auto",
                           "accelerator_policy": "prefer", "allow_fallback": True},
        task_spec_path=FIXTURE)
    check("PyTorch training path runs end-to-end on resolved hardware",
          accelerated.get("ok"),
          json.dumps({k: accelerated.get(k) for k in ("error_type", "error")}))
    if accelerated.get("ok"):
        plan = accelerated["execution"]
        check("accelerated run records framework and actual device",
              plan["framework"] == "torch" and plan["device"] in ("cpu", "mps", "cuda"))
        telemetry = json.loads((Path(accelerated["run_dir"]) / "telemetry.json").read_text())
        check("training telemetry records the same resolved device",
              telemetry["device"] == plan["device"] and telemetry["framework"] == "torch")

print()
print("== a task declaration cannot choose inputs outside its declared data_dir")
# Without containment, files: ["../x.csv"] was staged into the redacted root the candidate
# reads -- a spec picking its own inputs from outside the directory the operator confirmed.
import json as _json, shutil as _shutil, tempfile as _tempfile
from core.tabular_task import PortableTaskError as _PTE
_fx = ROOT / "tests" / "fixtures" / "portable_recommender" / "task.json"
with _tempfile.TemporaryDirectory() as _td:
    _d = Path(_td) / "pr"
    _shutil.copytree(_fx.parent, _d)
    _shutil.copy(_d / "events.csv", Path(_td) / "outside.csv")
    _cfg = _json.loads((_d / "task.json").read_text())
    _cfg["files"] = ["../outside.csv"]
    (_d / "task.json").write_text(_json.dumps(_cfg))
    try:
        load(_d / "task.json")
        _refused = False
    except _PTE as _e:
        _refused = "outside data_dir" in str(_e)
    check("a traversing declared file is refused", _refused)
    # And the legitimate spelling still works, so the check is not simply rejecting everything.
    _cfg["files"] = ["events.csv"]
    (_d / "task.json").write_text(_json.dumps(_cfg))
    check("the legitimate in-directory spelling still loads",
          load(_d / "task.json").slug == "shop_clicks_demo")

print()
bad = [name for name, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
raise SystemExit(1 if bad else 0)
