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
from agent import tournament_view as TV
from agent import research_driver as RD
from agent.research_agent import (code_prompt, judge_prompt, tournament_prompt,
                                  tournament_schema)
from core.ledger import canonical
from core.scorer import MAX_RECORD_BYTES

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
        "data_columns": ["user_id", "video_id"],
        "derived_features": [],
        "personalization": "preserved",
        "personalization_pathway": "user-specific embeddings interact with each item",
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

print("== data and realization contracts")
loader_contract = {
    "columns": [{"name": "user_id"}, {"name": "video_id"}, {"name": "duration_ms"}],
    "personalization": {
        "required": True,
        "non_personalized_allowed_only_as_declared_ablation": True,
    },
}
contract_cands = [dict(c) for c in cands]
_, contract_ok = RD.validate_tournament(
    {"candidates": contract_cands}, 6, data_contract=loader_contract)
check("valid loader columns and a user pathway pass before coding",
      contract_ok["lane_coverage_ok"], str(contract_ok["problems"]))
contract_cands[0]["data_columns"] = ["user_id", "invented_profile_field"]
_, bad_column = RD.validate_tournament(
    {"candidates": contract_cands}, 6, data_contract=loader_contract)
check("an invented raw column is refused before coding",
      any("invented_profile_field" in p for p in bad_column["problems"]),
      str(bad_column["problems"]))
no_pathway = dict(cands[0], personalization_pathway="")
check("personalized work without a user-conditioned pathway is refused",
      any("personalization_pathway" in p for p in
          RD.plan_contract_problems(no_pathway, loader_contract)))
ablation = dict(cands[0], personalization="non_personalized_ablation",
                personalization_pathway="")
check("an explicitly declared non-personalized ablation is accepted",
      RD.plan_contract_problems(ablation, loader_contract) == [])
check("a declared model.py edit with a config-only patch is refused",
      any("model.py" in p for p in
          RD.realization_problems(arch, ["config.json"])))
check("an architecture patch that really changes model.py passes realization",
      RD.realization_problems(arch, ["model.py"]) == [])

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

print("== eight rich hypotheses fit the signed record cap")
rich = [dict(c) for c in cands] + [candidate("C7", "feature_preprocessing", ["features.py"]),
                                   candidate("C8", "objective_loss", ["train.py"])]
for c in rich:
    for k in ("hypothesis", "mechanism", "information_value", "novelty_vs_history",
              "architecture_change", "implementation_risk", "leakage_risk"):
        c[k] = (str(c.get(k) or k) + " long recorded explanation") * 80
rv = verdict(arch["id"])
rv["ranking"] = [{"id": c["id"], "rank": i + 1,
                  "score_evidence": 1, "score_expected_impact": 1,
                  "score_novelty": 1, "score_information_value": 1,
                  "score_cpu_feasibility": 1, "score_implementation_risk": 1,
                  "note": "long judge note " * 200}
                 for i, c in enumerate(rich)]
rv["why_selected"] = "long selection reason " * 500
rv["architecture_verdicts"][0]["reason"] = "long architecture reason " * 500

# THE FIELDS THAT ACTUALLY BLEW THE CAP. The previous fixture inflated only the prose
# fields and left data_columns at two short names and personalization_pathway at one short
# sentence, so it passed at ~fixture-minimal values while the first real eight-candidate
# Opus tournament produced a 13,004-byte decision and was refused. Every field a candidate
# can fill is now padded PAST its truncation limit, so the test bounds the worst case
# rather than a typical one.
for c in rich:
    c["data_columns"] = [f"{n}_column_with_a_realistically_long_name" for n in range(16)]
    c["derived_features"] = [f"derived_feature_{n}_long_name" for n in range(16)]
    c["personalization"] = "non_personalized_ablation"
    c["personalization_pathway"] = "user-conditioned pathway description " * 40
    c["evidence"] = "iteration 3 evidence with a long citation " * 40
    c["expected_magnitude"] = 0.00123456789
    c["files_to_change"] = [f"generated_module_{n}_with_long_name.py" for n in range(8)]
    c["runtime_estimate"] = "about forty minutes on cpu " * 30
    c["memory_estimate"] = "about six gigabytes resident " * 30
    c["success_criteria"] = "signed dev primary rises by at least 0.002 " * 30
    c["failure_interpretation"] = "a negative result would settle the question " * 30
discovery_ok_max = dict(discovery_ok)
discovery_ok_max["coverage_note"] = "coverage rationale " * 60
discovery_ok_max["problems"] = [f"a recorded coverage problem {n} " * 8 for n in range(4)]

proposal_envelope = {"run_id": "test", "iteration": 3,
                     "what": "tournament_proposed", "policy": "tournament",
                     "tournament_size": 8}
proposal_body = {**proposal_envelope,
                 **RD.bounded_tournament_proposal_record(
                     cands=rich, coverage=discovery_ok_max, search_mode="discovery",
                     architecture_required=True, envelope=proposal_envelope)}
proposal_bytes = len(canonical(proposal_body))
check("1. maximal eight-candidate tournament_proposed fits the signed cap",
      proposal_bytes < MAX_RECORD_BYTES, f"{proposal_bytes} < {MAX_RECORD_BYTES}")

meta = RD.bounded_tournament_record(
    cands=rich, coverage=discovery_ok_max, verdict=rv,
    plan={"_candidate_id": arch["id"]}, selection_how="judge_selected_id",
    search_mode="discovery", architecture_required=True, proposal_seq=41)
record_body = {"run_id": "test", "iteration": 3, "what": "tournament",
               "policy": "tournament", "tournament_size": 8, **meta}
record_bytes = len(canonical(record_body))
check("2. maximal compact tournament decision fits the signed cap",
      record_bytes < MAX_RECORD_BYTES, f"{record_bytes} < {MAX_RECORD_BYTES}")

serialized = canonical(record_body).decode()
check("3. the decision duplicates no candidate prose",
      not any(marker in serialized for marker in
              ("long recorded explanation", "user-conditioned pathway description",
               "realistically_long_name", "long citation")),
      f"decision {record_bytes} B vs proposal {proposal_bytes} B")
check("the decision references all eight candidates by id and lane",
      len(meta["candidate_refs"]) == 8
      and all(set(r) == {"id", "lane"} for r in meta["candidate_refs"]))
check("the decision keeps the complete judge ranking",
      len(meta["judge"]["ranking"]) == 8)
check("the decision names its proposal record", meta["proposal_seq"] == 41)
check("compaction freed real headroom against the old duplicated record",
      record_bytes < proposal_bytes,
      f"decision {record_bytes} B < proposal {proposal_bytes} B")

print("== a compact decision hydrates, and an unresolvable one says so")
proposal_record = {"seq": 41, **proposal_body}
view = TV.hydrate(meta, [proposal_record])
check("a compact decision resolves every candidate from its proposal",
      view["_hydration"]["state"] == "resolved" and len(view["candidates"]) == 8
      and all(c.get("hypothesis") for c in view["candidates"]))
check("hydration never mutates the signed decision",
      "candidates" not in meta and meta["candidate_refs"] is not None)
embedded = {"iteration": 3, "candidates": [dict(c) for c in view["candidates"]],
            "judge": meta["judge"], "selected_id": meta["selected_id"]}
check("a historical embedded record still renders without a proposal",
      TV.hydrate(embedded, [])["_hydration"]["state"] == "embedded"
      and len(TV.hydrate(embedded, [])["candidates"]) == 8)
check("a decision naming a missing proposal is an INTEGRITY state, not empty content",
      TV.hydrate(meta, [])["_hydration"]["state"] == "unresolved"
      and TV.hydrate(meta, [])["candidates"] == []
      and "INTEGRITY" in TV.unavailable_note(TV.hydrate(meta, [])))
no_seq = {k: v for k, v in meta.items() if k != "proposal_seq"}
check("a decision naming no proposal is an unavailable-data state, not fabricated",
      TV.hydrate(no_seq, [])["_hydration"]["state"] == "missing_seq"
      and TV.hydrate(no_seq, [])["candidates"] == []
      and "unavailable" in TV.unavailable_note(TV.hydrate(no_seq, [])))
check("lane accounting works on a compact decision with no proposal available",
      sum(TV.lane_counts(meta).values()) == 8)

state_for_summary = RD.ResearchState({"name": "test"}, {})
for i in range(10):
    state_for_summary.beliefs[f"mechanism {i} " + "m" * 400] = {
        "mechanism": f"mechanism {i} " + "m" * 400,
        "lane": "model_architecture", "belief": "b" * 1600,
        "llm_confidence": 0.7, "evidence_grade": "replicated",
        "supporting_iterations": list(range(20)),
        "contradicting_iterations": list(range(20)),
        "affected_files": ["model.py" * 30] * 20,
        "effect_range": [-0.01, 0.01], "unresolved": "u" * 1000,
        "retry_only_if": "r" * 1000, "last_updated_iteration": i,
    }
summary_exp = RD.Experiment(
    iteration=8, tree="a" * 64, parent="b" * 64,
    hypothesis="large reflection", lane="model_architecture",
    reflection={key: "x" * 5000 for key in (
        "what_happened", "mechanism_supported", "why", "recommendation",
        "prediction_comment", "remaining_uncertainty", "next_question",
        "belief_updates")})
summary = RD.bounded_iteration_result_record(
    run_id="test", exp=summary_exp, state=state_for_summary,
    cost={"wall_seconds": 1.0, "tokens_total": 10})
summary_bytes = len(canonical(summary))
check("maximal iteration summary stays under MAX_RECORD_BYTES",
      summary_bytes < MAX_RECORD_BYTES, f"{summary_bytes} < {MAX_RECORD_BYTES}")
check("iteration summary omits duplicated belief updates",
      "belief_updates" not in summary["reflection"])
check("iteration summary keeps a bounded useful belief view",
      0 < len(summary["beliefs_after"]) <= 4)
run_end = RD.bounded_run_end_record({
    "run_id": "test", "termination": "budget",
    "stats": {"planned": 8, "evaluated": 8},
    "budget": {"tokens_total": 999999, "stage_seconds": {"code": 1200.0}},
    "champion": {"tree": "a" * 64, "primary": 0.604},
    "usage_by_role": {role: {"calls": 8, "models": ["claude-opus-5"],
                              "total_tokens": 100000}
                      for role in ("plan", "judge", "code", "reflect")},
    "lane_stats": {"by_lane": {lane: {"considered": 8, "selected": 2,
                                         "executed": 1, "promoted": 0}
                                for lane in lanes}},
    "beliefs": state_for_summary.beliefs,
    "hardware_capabilities": {"cpu": {"available": True},
                              "mps": {"available": False},
                              "cuda": {"available": False}},
    "manual_interventions": [],
})
run_end_bytes = len(canonical(run_end))
check("maximal run-end summary stays under MAX_RECORD_BYTES",
      run_end_bytes < MAX_RECORD_BYTES, f"{run_end_bytes} < {MAX_RECORD_BYTES}")
check("run-end summary discloses omitted beliefs",
      run_end["beliefs_omitted"] == 6 and len(run_end["beliefs"]) == 4)

# Reproduce the shape that failed in anthropic_discovery_cont1. The earlier fixture was
# only 9 KB and therefore never entered the compaction paths that matter.
oversized_run_end = RD.bounded_run_end_record({
    "run_id": "anthropic_discovery_cont1", "termination": "budget",
    "stats": {
        "planned": 8, "evaluated": 8, "architecture_considered": 24,
        "candidate_diagnostics": ["long diagnostic " * 300 for _ in range(8)],
    },
    "budget": {
        "tokens_total": 999999, "tokens_total_from_iterations": 999999,
        "stage_seconds": {f"stage_{i}": 1000.0 + i for i in range(40)},
        "iteration_wall_seconds": [1000.0 + i for i in range(50)],
        "remaining": {"tokens": 1, "wall_seconds": 2},
    },
    "champion": {"tree": "c" * 64, "primary": 0.604,
                 "hypothesis": "large champion hypothesis " * 1000},
    "usage_by_role": {
        role: {"calls": 20, "input_tokens": 200000, "output_tokens": 100000,
               "thinking_tokens": 50000, "total_tokens": 350000,
               "seconds": 1000.0, "retries": 3, "refusals": 1,
               "truncations": 2, "models": ["claude-opus-5"],
               "provider_detail": "not needed " * 1000}
        for role in ("plan", "judge", "code", "reflect")
    },
    "lane_stats": {"by_lane": {
        lane: {"considered": 100, "selected": 30, "executed": 20,
               "promoted": 2, "notes": "lane detail " * 500}
        for lane in lanes},
        "model_py_edits": 3, "train_py_edits": 4,
        "features_py_edits": 2, "config_json_edits": 5},
    "beliefs": state_for_summary.beliefs,
    "hardware_capabilities": {
        "platform": "platform " * 500, "python": "python " * 500,
        "torch_available": True, "torch_version": "version " * 500,
        "cpu": {"available": True, "name": "cpu " * 1000,
                "reason": "reason " * 1000},
        "mps": {"available": False, "reason": "reason " * 1000},
        "cuda": {"available": False, "reason": "reason " * 1000},
    },
    "seed_profile": "minimal", "research_policy": "tournament",
    "search_mode": "discovery", "evidence_class": "diagnostic_discovery",
    "submittable": False, "execution_profile": {"framework": "auto"},
    "manual_interventions": [{"what": "none " * 1000, "why": "none " * 1000}],
})
oversized_bytes = len(canonical(oversized_run_end))
check("the real oversized run-end shape compacts below MAX_RECORD_BYTES",
      oversized_bytes < MAX_RECORD_BYTES,
      f"{oversized_bytes} < {MAX_RECORD_BYTES}")
check("run-end compaction preserves the completion and champion identity",
      oversized_run_end["run_id"] == "anthropic_discovery_cont1"
      and oversized_run_end["termination"] == "budget"
      and oversized_run_end["champion"]["tree"] == "c" * 64)
check("run-end compaction is explicit rather than silent",
      bool(oversized_run_end.get("run_end_compaction")))

print("== CLI and console launcher keep discovery opt-in")
parser = RD.build_parser()
check("research CLI defaults to the standard search path",
      parser.parse_args([]).search_mode == "standard")
continuation_args = parser.parse_args([
    "--llm-max-output-tokens", "64000", "--seed-tree", "0" * 64])
check("research CLI separates per-call output from the run token cap",
      continuation_args.llm_max_output_tokens == 64000)
check("research CLI accepts an explicit continuation artifact",
      continuation_args.seed_tree == "0" * 64)
provider_ceiling_args = parser.parse_args([
    "--llm-max-output-tokens", "128000"])
check("research CLI accepts Opus 5's synchronous output ceiling",
      provider_ceiling_args.llm_max_output_tokens == 128000)
continued_provenance = RD.provenance_for_seed_files("minimal", {
    "model.py": "class FM:\n    def step_bpr(self):\n        pass\n",
    "train.py": "def build_pair_index():\n    pass\n",
})
check("continuation provenance detects agent-written BPR naming",
      continued_provenance["pairwise_code_present"] is True)
check("a BPR method is not mislabelled as a new model architecture",
      continued_provenance["alternative_architecture_code_present"] is False)
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
