#!/usr/bin/env python3
"""Signed evidence is reconciled per IMPLEMENTATION, not collapsed per mechanism.

The case this suite exists for is real and is in the repository twice over:

    research_trials seq 116   loss pointwise->pairwise, neg_per_pos 2
                              +0.001888 over the official baseline, 20/20 seeds, CONFIRMED
    anthropic_contract_gate2  loss pointwise->pairwise, pairs_per_epoch 600000
                              -0.000738 on ONE seed

The old grader returned "contradicted" for that pair, so a single-seed negative buried a
twenty-seed confirmation. No API calls, no training, no ledger is written or modified.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.research_state import Experiment, ResearchState
from core import evidence as EV

R = []


def check(name, condition, extra=""):
    ok = bool(condition)
    R.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


POINTWISE = {"loss": "pointwise", "fields": ["user_id", "video_id"], "k": 16,
             "lr": 0.001, "neg_per_pos": 1}
PAIRWISE_GOOD = {**POINTWISE, "loss": "pairwise", "neg_per_pos": 2}
PAIRWISE_BAD = {"loss": "pairwise", "fields": ["user_id", "video_id"], "k": 16,
                "lr": 0.001, "pairs_per_epoch": 600000}
SCOPE = {"name": "kuairand_pure", "metric": "primary", "label": "long_view",
         "dev_split": "valid", "held_out_split": "test",
         "dataset_fingerprint": "f" * 64, "data_contract_sha256": "d" * 64,
         "workspace_contract_sha256": "w" * 64}


def obs(tree, cfg, delta, *, parent_cfg=POINTWISE, files=("config.json",),
        confirmation=None, scope=None):
    return EV.Observation(
        mechanism=EV.mechanism_key(parent_cfg, cfg, files), tree=tree,
        parent_tree="p" * 64, changed_files=list(files), config=cfg,
        config_delta=EV.config_delta(parent_cfg, cfg), delta=delta,
        seed_confirmation=confirmation,
        provenance={"scope": scope or SCOPE})


print("== the mechanism join key is derived from facts, not prose")
good = obs("a" * 64, PAIRWISE_GOOD, +0.001888)
bad = obs("b" * 64, PAIRWISE_BAD, -0.000738)
check("both implementations derive the same mechanism key",
      good.mechanism == bad.mechanism == "loss=pointwise->pairwise", good.mechanism)
check("a numeric knob does not create a new mechanism",
      EV.mechanism_key(PAIRWISE_GOOD, {**PAIRWISE_GOOD, "lr": 0.01}, ["config.json"])
      != "loss=pointwise->pairwise")
check("implementations stay distinguishable",
      EV.implementation_summary(good) != EV.implementation_summary(bad),
      f"{EV.implementation_summary(good)!r} vs {EV.implementation_summary(bad)!r}")
check("a pure code rewrite still gets a stable key",
      EV.mechanism_key(POINTWISE, POINTWISE, ["model.py"]) == "code:model.py")

print()
print("== one positive and one negative implementation produce MIXED")
check("grade is mixed, not contradicted", EV.grade([good, bad]) == "mixed",
      EV.grade([good, bad]))
check("'contradicted' is not a grade this project emits",
      "contradicted" not in EV.GRADES)
check("order does not change the grade", EV.grade([bad, good]) == "mixed")
check("two agreeing implementations replicate",
      EV.grade([good, obs("c" * 64, PAIRWISE_GOOD, +0.0011)]) == "replicated")
check("one observation alone is a single signed result",
      EV.grade([bad]) == "single_signed_result")
check("no observations is unreplicated", EV.grade([]) == "unreplicated")

print()
print("== a seed-confirmed implementation stays visibly the strongest")
CONF = {"n": 20, "mean": 0.001888, "wins": 20, "verdict": "CONFIRMED",
        "champion_tree": "a" * 64, "comparison_tree": "z" * 64,
        "ledger": "runs/ledgers/research_trials.jsonl", "seq": 116}
confirmed = obs("a" * 64, PAIRWISE_GOOD, +0.001888, confirmation=CONF)
check("a confirmed positive against a single-seed negative is still mixed",
      EV.grade([confirmed, bad]) == "mixed")
check("the seed-confirmed implementation is the strongest evidence",
      EV.strongest([confirmed, bad]).tree == "a" * 64)
check("a single-seed negative can never outrank a confirmation",
      EV.strongest([bad, confirmed]).confirmed is True)
check("a bigger single-seed effect still loses to a confirmation",
      EV.strongest([confirmed, obs("e" * 64, PAIRWISE_BAD, -0.9)]).tree == "a" * 64)
summary = EV.conflict_summary("loss=pointwise->pairwise", [confirmed, bad])
check("the summary names the strongest implementation",
      summary["strongest_evidence"]["tree"] == "a" * 64
      and summary["strongest_evidence"]["seeds"] == 20)
check("the summary reports mixed evidence rather than a settled negative",
      summary["evidence_grade"] == "mixed"
      and "not a settled negative" in summary["interpretation"])
check("the synthesised sentence is labelled as synthesised",
      summary["interpretation_is_synthesised"] is True)
check("procedure claims are labelled LLM-authored",
      summary["procedure_claims_are_llm_authored"] is True)

print()
print("== the planner sees BOTH sides of a conflict")
view = EV.planner_view(EV.index_by_mechanism([confirmed, bad]))
block = next(b for b in view if b["mechanism"] == "loss=pointwise->pairwise")
text = json.dumps(block)
check("the confirmed positive is present with its paired mean and seed count",
      "+0.001888" in text and "20/20 seeds" in text and "CONFIRMED" in text)
check("the single-seed negative is present with its delta",
      "-0.000738" in text)
check("both sides are populated, not just the newest",
      len(block["supporting"]) == 1 and len(block["contradicting"]) == 1)
check("conflicts are ordered first for the planner",
      EV.planner_view(EV.index_by_mechanism(
          [confirmed, bad, obs("f" * 64, {**POINTWISE, "loss": "listwise"}, +0.002)]
      ))[0]["evidence_grade"] == "mixed")

print()
print("== a confounded candidate is evidence for each mechanism, and says so")
multi = obs("g" * 64, {**PAIRWISE_GOOD, "fields": ["user_id", "video_id", "hist_len"]},
            +0.001888, confirmation=CONF)
check("a candidate changing two things carries a compound key",
      multi.confounded and len(multi.components) == 2, multi.mechanism)
idx = EV.index_by_mechanism([multi, bad])
check("it is indexed under every mechanism it changed",
      "loss=pointwise->pairwise" in idx and "fields+hist_len" in idx, str(sorted(idx)))
check("without component indexing the two BPR results would never meet",
      len(idx["loss=pointwise->pairwise"]) == 2)
check("the confounding is disclosed, naming only the OTHER mechanism",
      "CONFOUNDED" in multi.short("loss=pointwise->pairwise")
      and "fields+hist_len" in multi.short("loss=pointwise->pairwise")
      and "loss=pointwise->pairwise" not in
      multi.short("loss=pointwise->pairwise").split("CONFOUNDED")[1])

print()
print("== confirmations attach only to their exact ordered tree pair")
idxc = EV.ConfirmationIndex()
idxc.add({"champion_tree": "a" * 64, "comparison_tree": "z" * 64, "n": 20,
          "mean": 0.001888, "wins": 20, "verdict": "CONFIRMED"},
         ledger="L1", seq=116)
idxc.add({"champion_tree": "q" * 64, "comparison_tree": "a" * 64, "n": 20,
          "mean": 0.000223, "wins": 14, "verdict": "NOT CONFIRMED"},
         ledger="L2", seq=42)
check("a confirmation is found by its exact ordered pair",
      idxc.for_pair("a" * 64, "z" * 64)["verdict"] == "CONFIRMED")
check("the reversed pair is a different record",
      idxc.for_pair("z" * 64, "a" * 64) is None)
check("a tree used as a COMPARISON does not inherit that record",
      [r["seq"] for r in idxc.for_champion("a" * 64)] == [116])
check("a NOT CONFIRMED record never counts as confirmation",
      idxc.best_for("q" * 64) is None)
check("the verdict is stored verbatim",
      idxc.for_pair("q" * 64, "a" * 64)["verdict"] == "NOT CONFIRMED")
check("provenance of the signed record is preserved",
      idxc.for_pair("a" * 64, "z" * 64)["ledger"] == "L1"
      and idxc.for_pair("a" * 64, "z" * 64)["seq"] == 116)
check("an incomplete confirmation record is refused",
      idxc.add({"champion_tree": "a" * 64, "n": 5}) is False)

print()
print("== incompatible datasets or contracts cannot import evidence")
ok, why = EV.compatible(SCOPE, SCOPE)
check("an identical scope is compatible", ok and why == [])
for key in ("dataset_fingerprint", "data_contract_sha256",
            "workspace_contract_sha256", "metric", "label"):
    other = {**SCOPE, key: "x" * 64}
    ok2, why2 = EV.compatible(SCOPE, other)
    check(f"a differing {key} blocks import", not ok2 and any(key in r for r in why2))
missing = {k: v for k, v in SCOPE.items() if k != "data_contract_sha256"}
ok3, why3 = EV.compatible(SCOPE, missing)
check("an unrecorded scope field fails closed rather than assuming a match",
      not ok3 and any("unrecorded" in r for r in why3))

print()
print("== legacy unscoped beliefs stay excluded")
current = ResearchState({**SCOPE, "run_id": "new"})
prior = ResearchState({**SCOPE, "run_id": "old"})
prior.beliefs = {
    "scoped mechanism": {"mechanism": "scoped mechanism", "belief": "kept",
                         "supporting_iterations": [1], "contradicting_iterations": [],
                         "scope": {"data_contract_sha256": SCOPE["data_contract_sha256"]}},
    "legacy prose that merely sounds relevant": {
        "mechanism": "legacy prose that merely sounds relevant", "belief": "dropped",
        "supporting_iterations": [1], "contradicting_iterations": []},
}
report = current.import_memory(prior, mode="import", source="prior/state.json")
check("a scoped belief is imported", "scoped mechanism" in current.beliefs)
check("an unscoped legacy belief is refused",
      "legacy prose that merely sounds relevant" not in current.beliefs)
check("the refusal is counted, not silent", report["beliefs_skipped_unscoped"] == 1)
check("imported beliefs are marked as imported",
      current.beliefs["scoped mechanism"]["imported"] is True)
check("imported beliefs keep their origin",
      current.beliefs["scoped mechanism"]["imported_from"] == "prior/state.json")

real = json.loads((ROOT / "runs/candidates/research_submission1/state.json").read_text())
legacy = [b for b in (real.get("beliefs") or {}).values() if not b.get("scope")]
check("the repository really does contain unscoped legacy beliefs to exclude",
      len(legacy) >= 1, f"{len(legacy)} unscoped")
prior2 = ResearchState({**SCOPE, "run_id": "old2"})
prior2.beliefs = {b["mechanism"]: b for b in legacy}
fresh = ResearchState({**SCOPE, "run_id": "new2"})
rep2 = fresh.import_memory(prior2, mode="import", source="research_submission1")
check("no real legacy belief crosses into a new run",
      fresh.beliefs == {} and rep2["beliefs_skipped_unscoped"] == len(legacy))

print()
print("== observation ingestion carries provenance and refuses foreign scope")
donor = ResearchState({**SCOPE, "run_id": "donor"})
donor.note_observation(confirmed)
donor.note_observation(bad)
receiver = ResearchState({**SCOPE, "run_id": "receiver"})
n_in, n_skip = receiver.import_evidence(donor.observations, source="donor/state.json")
check("compatible observations are imported", n_in == 2 and n_skip == 0)
check("provenance records where they came from",
      all(o["provenance"]["imported"] is True for o in receiver.observations))
foreign = ResearchState({**SCOPE, "run_id": "foreign"})
foreign.note_observation(obs("h" * 64, PAIRWISE_BAD, -0.01,
                             scope={**SCOPE, "dataset_fingerprint": "other"}))
target = ResearchState({**SCOPE, "run_id": "target"})
n_in2, n_skip2 = target.import_evidence(foreign.observations, source="foreign")
check("an observation from another dataset is refused",
      n_in2 == 0 and n_skip2 == 1)
check("two implementations of one mechanism both survive ingestion",
      len({o["tree"] for o in receiver.observations}) == 2)
check("the receiving run grades the mechanism as mixed",
      any(b["evidence_grade"] == "mixed" for b in receiver.mechanism_evidence()))
check("the planner view reaches for_prompt()",
      "mechanism_evidence" in receiver.for_prompt())
check("for_prompt shows both sides",
      any(b["supporting"] and b["contradicting"]
          for b in receiver.for_prompt()["mechanism_evidence"]))

print()
print("== a single-seed negative cannot overwrite a confirmed result")
state = ResearchState({**SCOPE, "run_id": "r"})
state.note_observation(confirmed)
state.note_observation(bad)
block2 = next(b for b in state.mechanism_evidence()
              if b["mechanism"] == "loss=pointwise->pairwise")
check("the mechanism is mixed after the negative arrives",
      block2["evidence_grade"] == "mixed")
check("the confirmed implementation is still named strongest",
      block2["strongest_evidence"]["grade"] == "seed_confirmed")
state.note_observation(bad)
check("re-recording the same implementation does not duplicate it",
      len(state.observations) == 2)

print()
print("== belief grading: strength and direction are separate")
mixed_belief = {"supporting_iterations": [1], "contradicting_iterations": [2]}
check("a belief with both signs grades mixed",
      ResearchState.evidence_grade(mixed_belief) == "mixed")
check("its direction is conflicting",
      ResearchState.belief_direction(mixed_belief) == "conflicting")
check("a purely negative belief is graded by strength, not verdict",
      ResearchState.evidence_grade(
          {"supporting_iterations": [], "contradicting_iterations": [2]})
      == "single_signed_result")
check("its direction is against",
      ResearchState.belief_direction(
          {"supporting_iterations": [], "contradicting_iterations": [2]}) == "against")

print()
print("== every recorded run still loads, and no historical file is touched")
before = {p: Path(p).read_bytes() for p in glob.glob("runs/candidates/*/state.json")}
loaded = 0
for path in sorted(before):
    st = ResearchState.load(path)
    for b in st.beliefs.values():
        assert b.get("evidence_grade") in EV.GRADES, (path, b.get("evidence_grade"))
    loaded += 1
check(f"all {loaded} recorded states load and re-grade into the new vocabulary",
      loaded >= 10, f"{loaded} states")
check("loading never rewrites a state file",
      all(Path(p).read_bytes() == b for p, b in before.items()))
check("states written before observations existed default cleanly",
      all(isinstance(ResearchState.load(p).observations, list) for p in sorted(before)))

print()
print("== the real repository evidence reconciles the Gate 2 conflict")
conf_idx = EV.read_confirmations(sorted(glob.glob("runs/ledgers/*.jsonl")))
CHAMP = "fe985741a8368e082be4c71e0d9e3be06ca333f2aac1b26216203cbf2283c3b5"
BASE = "e1d994dedd110658b8c1c130486fb980b98df84e6fabafcad664b94d38ed693c"
best = conf_idx.best_for(CHAMP)
check("the submitted champion's 20-seed confirmation is found",
      best and best["n"] == 20 and best["wins"] == 20
      and abs(best["mean"] - 0.001888) < 1e-9 and best["verdict"] == "CONFIRMED",
      json.dumps({k: best.get(k) for k in ("n", "mean", "wins", "verdict")})
      if best else "not found")
check("its confirmation cites the signed record it came from",
      best["ledger"].endswith("research_trials.jsonl") and best["seq"] == 116)
check("the opus comparison does NOT confirm the submitted champion",
      conf_idx.for_pair(
          "0375004028b7690ad09a75c2d44e8bec4ac125eaae68a32ea17337f3121aa602",
          CHAMP)["verdict"] == "NOT CONFIRMED")
check("the champion is never confirmed against itself",
      all(r["champion_tree"] == CHAMP for r in conf_idx.for_champion(CHAMP)))

print()
bad_checks = [n for n, ok in R if not ok]
print(f"{len(R) - len(bad_checks)}/{len(R)} passed"
      + (f"  FAILURES: {bad_checks}" if bad_checks else ""))
raise SystemExit(1 if bad_checks else 0)
