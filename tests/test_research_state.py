"""Cross-run memory is explicit, scoped and honest about confidence."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.research_state import Experiment, ResearchState

R = []


def check(name, condition, extra=""):
    ok = bool(condition)
    R.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as error:
        return str(error)
    except Exception as error:
        return f"WRONG-TYPE:{type(error).__name__}:{error}"
    return ""


def task(run="r1", data="d" * 64):
    return {"name": "demo", "metric": "primary", "label": "clicked",
            "dev_split": "valid", "held_out_split": "test",
            "dataset_fingerprint": data,
            "data_contract_sha256": "c" * 64,
            "workspace_contract_sha256": "w" * 64,
            "run_id": run}


prior = ResearchState(task("prior"))
positive = Experiment(1, "a" * 64, None, files_changed=["train.py"])
prior.add(positive)
prior.update_beliefs(positive, [{
    "mechanism": "pairwise ordering", "lane": "objective_loss",
    "belief": "likely useful", "llm_confidence": 0.99, "supports": True,
}])
belief = prior.beliefs["pairwise ordering"]
check("LLM confidence is stored under an honest name", belief["llm_confidence"] == 0.99)
check("one result yields a categorical evidence grade",
      belief["evidence_grade"] == "single_signed_result")
check("numeric LLM confidence does not determine evidence grade",
      ResearchState.evidence_grade({"llm_confidence": 0.01,
                                    "supporting_iterations": [1]})
      == ResearchState.evidence_grade({"llm_confidence": 0.99,
                                      "supporting_iterations": [1]}))

negative = Experiment(2, "b" * 64, "a" * 64, files_changed=["train.py"])
prior.add(negative)
prior.update_beliefs(negative, [{
    "mechanism": "pairwise ordering", "lane": "objective_loss",
    "belief": "mixed evidence", "llm_confidence": 0.5, "supports": False,
}])
# This assertion used to require "contradicted", which is the defect
# anthropic_contract_gate2 exposed: one contradicting iteration outranked everything, so a
# single-seed negative could bury the submitted BPR champion's 20/20 seed confirmation of
# the same mechanism by a different implementation. A grade now reports STRENGTH and
# `belief_direction` reports which way it points.
check("one supporting and one contradicting result grade as mixed, not contradicted",
      prior.beliefs["pairwise ordering"]["evidence_grade"] == "mixed",
      prior.beliefs["pairwise ordering"]["evidence_grade"])
check("the conflict is visible as a direction rather than a verdict",
      ResearchState.belief_direction(prior.beliefs["pairwise ordering"]) == "conflicting")
check("'contradicted' is no longer a grade this project can emit",
      "contradicted" not in __import__("core.evidence", fromlist=["x"]).GRADES)

imported = ResearchState(task("new"))
summary = imported.import_memory(prior, mode="import", source="prior/state.json")
check("compatible import carries scoped beliefs", summary["beliefs_imported"] == 1)
check("import provenance remains visible",
      imported.beliefs["pairwise ordering"]["imported"] is True)
check("import does not pretend prior experiments ran again", imported.experiments == [])
check("import does not carry dedup history", imported.seen_trees == {})

continued = ResearchState(task("continued"))
prior.note_seen("f" * 64, 7)
prior.failures["Example"] = [{"iteration": 7}]
continued.import_memory(prior, mode="continue", source="prior/state.json")
check("continue carries dedup history", "f" * 64 in continued.seen_trees)
check("continue carries failure history", "Example" in continued.failures)

wrong = ResearchState(task("wrong", data="e" * 64))
message = raises(ValueError, wrong.import_memory, prior,
                 mode="import", source="prior/state.json")
check("incompatible dataset memory is refused",
      "dataset_fingerprint" in message, message)

legacy_scoped = ResearchState(task("legacy"))
legacy_scoped.beliefs["old"] = {"mechanism": "old", "belief": "old claim",
                                 "confidence": 0.8,
                                 "supporting_iterations": [1],
                                 "contradicting_iterations": []}
target = ResearchState(task("target"))
legacy_summary = target.import_memory(legacy_scoped, mode="import", source="old.json")
check("legacy unscoped beliefs are not made permanent",
      legacy_summary["beliefs_imported"] == 0
      and legacy_summary["beliefs_skipped_unscoped"] == 1)

with tempfile.TemporaryDirectory(prefix="track2-state-") as td:
    path = Path(td) / "state.json"
    payload = prior.to_json()
    payload["experiments"][0]["future_field"] = "newer writer"
    payload["beliefs"]["pairwise ordering"]["confidence"] = 0.4
    payload["beliefs"]["pairwise ordering"].pop("llm_confidence", None)
    path.write_text(json.dumps(payload))
    loaded = ResearchState.load(path)
check("old state accepts future experiment fields", len(loaded.experiments) == 2)
check("legacy confidence loads as explicitly LLM-authored",
      loaded.beliefs["pairwise ordering"]["llm_confidence"] == 0.4
      and "uncalibrated" in loaded.beliefs["pairwise ordering"]["llm_confidence_note"])

print()
bad = [name for name, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
