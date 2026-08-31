#!/usr/bin/env python3
"""Fast policy checks for opt-in architecture discovery. No scorer or API calls."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import control_launch as CL
from agent import research_driver as RD
from agent.research_agent import (code_prompt, judge_prompt, tournament_prompt,
                                  tournament_schema)

R = []


def check(name, condition, extra=""):
    R.append((name, bool(condition)))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")


def raises(fn, text):
    try:
        fn()
    except (ValueError, CL.LaunchRefused) as exc:
        return text in str(exc)
    return False


def candidate(cid, lane, files):
    return {
        "id": cid, "lane": lane, "hypothesis": f"test hypothesis for {cid}",
        "mechanism": f"causal mechanism for {cid}", "evidence": "synthetic test",
        "expected_direction": "increase", "expected_magnitude": 0.001,
        "information_value": "settles a research question",
        "novelty_vs_history": "not tried", "files_to_change": files,
        "runtime_estimate": "one minute", "memory_estimate": "one GB",
        "implementation_risk": "bounded", "leakage_risk": "none",
        "success_criteria": "dev metric rises",
        "failure_interpretation": "direction unsupported", "parent": "champion",
    }


lanes = ["feature_preprocessing", "objective_loss", "sampling_training",
         "optimization_regularization", "model_architecture", "ensemble_calibration"]
cands = [candidate(f"C{i}", lane, ["model.py"] if lane == "model_architecture"
                   else ["train.py"] if lane == "sampling_training"
                   else ["config.json"])
         for i, lane in enumerate(lanes, 1)]
arch = next(c for c in cands if c["lane"] == "model_architecture")

print("== discovery tournament contract")
_, standard_cov = RD.validate_tournament({"candidates": cands}, 6)
check("standard mode preserves the old coverage contract",
      standard_cov["lane_coverage_ok"], str(standard_cov["problems"]))
_, discovery_bad = RD.validate_tournament({"candidates": cands}, 6, discovery=True)
check("discovery rejects an unnamed architecture family",
      not discovery_bad["lane_coverage_ok"] and
      any("model_family" in p for p in discovery_bad["problems"]),
      str(discovery_bad["problems"]))
arch["model_family"] = "DeepFM"
arch["architecture_change"] = "add a neural MLP interaction tower beside the FM branch"
_, discovery_ok = RD.validate_tournament({"candidates": cands}, 6, discovery=True)
check("a substantive model.py architecture passes discovery validation",
      discovery_ok["lane_coverage_ok"], str(discovery_ok["problems"]))
arch["files_to_change"] = ["config.json"]
_, config_only = RD.validate_tournament({"candidates": cands}, 6, discovery=True)
check("a config-only width change cannot masquerade as architecture",
      any("must include model.py" in p for p in config_only["problems"]),
      str(config_only["problems"]))
arch["files_to_change"] = ["model.py"]

print("== architecture funding floor")
check("standard mode never forces architecture",
      not RD.discovery_architecture_required("standard", 3, 0))
check("discovery decisions one and two remain open",
      not RD.discovery_architecture_required("discovery", 1, 0)
      and not RD.discovery_architecture_required("discovery", 2, 0))
check("the third discovery decision requires architecture",
      RD.discovery_architecture_required("discovery", 3, 0))
check("the floor switches off once architecture was selected",
      not RD.discovery_architecture_required("discovery", 3, 1)
      and not RD.discovery_architecture_required("discovery", 8, 1))

def verdict(selected):
    return {
        "selected_id": selected,
        "ranking": [{"id": c["id"], "rank": i + 1} for i, c in enumerate(cands)],
        "architecture_verdicts": [{"candidate_id": arch["id"],
                                    "verdict": ("selected" if selected == arch["id"]
                                                else "rejected"),
                                    "reason": "a sufficiently explicit test reason"}],
    }

check("a non-architecture judge choice is refused when the floor is due",
      any("floor is due" in p for p in
          RD.validate_judge(cands, verdict("C1"), require_architecture=True)))
check("an architecture judge choice satisfies the floor",
      RD.validate_judge(cands, verdict(arch["id"]), require_architecture=True) == [])
check("the same non-architecture choice remains valid in standard mode",
      RD.validate_judge(cands, verdict("C1")) == [])

print("== prompts expose capability without prescribing a winner")
state = {"task": {"name": "test"}, "budget": {
    "search_mode": "discovery", "discovery": {
        "architecture_selected_so_far": 0,
        "architecture_selection_deadline": 3,
        "architecture_required_now": True}}}
ts, tp = tournament_prompt(state, {"model.py": "# model"}, None, 6)
_, jp = judge_prompt(cands, state, {"model.py": "# model"}, None)
check("planner sees the actual allowed tree and neural libraries",
      all(name in "\n".join(ts) + tp for name in ("lightgbm", "xgboost", "torch")))
check("planner is told config-only changes are not architecture",
      "config constant" in tp and "MUST include `model.py`" in tp)
check("directions are examples rather than a forced model menu",
      "examples, not a mandatory menu" in tp)
check("judge sees the explicit architecture deadline", "required on THIS decision: True" in jp)
check("the old pro-FM judge bias is gone", "was the right call so far" not in jp)
check("the incumbent is evidence rather than a mandate", "evidence,\nnot a mandate" in jp)
props = tournament_schema()["properties"]["candidates"]["items"]["properties"]
check("structured output permits architecture identity fields",
      {"model_family", "architecture_change"} <= set(props))
_, cp = code_prompt({"hypothesis": "test"}, {"model.py": "# model"}, None, 0, state)
check("coder gets the executable dependency declaration syntax",
      "# requires: package-name" in cp and "field alone does not install" in cp)

print("== CLI and console launcher keep discovery opt-in")
parser = RD.build_parser()
check("research CLI defaults to the standard search path",
      parser.parse_args([]).search_mode == "standard")
check("research CLI exposes exactly standard and discovery",
      tuple(next(a for a in parser._actions if a.dest == "search_mode").choices)
      == RD.SEARCH_MODES)
valid = CL.parse({"driver": "research", "backend": "gemini", "model": "m",
                  "search_mode": "discovery", "research_policy": "tournament",
                  "convergence": "off", "iterations": "12"})
check("UI validator accepts an explicit discovery run", valid.search_mode == "discovery")
check("UI argv carries the discovery policy",
      valid.argv()[valid.argv().index("--search-mode") + 1] == "discovery")
check("discovery cannot silently retain official convergence",
      raises(lambda: CL.parse({"driver": "research", "backend": "gemini", "model": "m",
                              "search_mode": "discovery", "convergence": "official"}),
             "convergence=off"))
check("discovery cannot use the single-hypothesis policy",
      raises(lambda: CL.parse({"driver": "research", "backend": "gemini", "model": "m",
                              "search_mode": "discovery", "convergence": "off",
                              "research_policy": "single"}), "tournament"))
check("discovery cannot claim an impossible two-iteration floor",
      raises(lambda: CL.parse({"driver": "research", "backend": "gemini", "model": "m",
                              "search_mode": "discovery", "convergence": "off",
                              "iterations": "2"}), "at least 3"))
check("the official driver cannot receive discovery-only policy",
      raises(lambda: CL.parse({"proposer": "random", "search_mode": "discovery"}),
             "code-first"))
check("ordinary research launches still default to standard",
      CL.parse({"driver": "research", "backend": "gemini", "model": "m"}).search_mode
      == "standard")

print("== discovery evidence stays on a separate signed chain")
with tempfile.TemporaryDirectory() as td:
    ledger = Path(td) / "ledger.jsonl"
    ledger.write_text(json.dumps({"payload": {"kind": "run_start", "body": {
        "run_id": "old", "search_mode": "standard"}}}) + "\n")
    check("a discovery run refuses a standard ledger",
          raises(lambda: RD.require_compatible_search_ledger(ledger, "discovery"),
                 "separate scorer processes"))
    RD.require_compatible_search_ledger(ledger, "standard")
    check("another standard run may share a standard ledger", True)
    ledger.write_text(json.dumps({"payload": {"kind": "run_start", "body": {
        "run_id": "disc", "search_mode": "discovery"}}}) + "\n")
    check("a standard run refuses a discovery ledger",
          raises(lambda: RD.require_compatible_search_ledger(ledger, "standard"),
                 "separate scorer processes"))
    RD.require_compatible_search_ledger(ledger, "discovery")
    check("another discovery run may share a discovery ledger", True)

passed = sum(v for _, v in R)
print(f"\n{passed}/{len(R)} assertions passed")
if passed != len(R):
    raise SystemExit(1)
