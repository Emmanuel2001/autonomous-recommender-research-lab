"""The planner's data contract is measured, general and held-out safe."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import data_contract as DC
from core.task import get, load

R = []


def check(name, condition, extra=""):
    ok = bool(condition)
    R.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


@dataclass(frozen=True)
class TinyRows:
    split: str
    user_id: np.ndarray
    video_id: np.ndarray
    temperature: np.ndarray
    date: np.ndarray
    label: np.ndarray | None
    aux: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.user_id)

    def content_sha256(self):
        import hashlib
        h = hashlib.sha256()
        for value in (self.user_id, self.video_id, self.temperature, self.date):
            h.update(np.ascontiguousarray(value).tobytes())
        return h.hexdigest()


def rows(name, offset, label=True):
    return TinyRows(
        name,
        np.array([1, 2]), np.array([10, 20]),
        np.array([1.5 + offset, 2.5 + offset]),
        np.array([20260101 + offset, 20260101 + offset]),
        np.array([0.0, 1.0]) if label else None,
        {"secondary": np.array([1.0, 0.0])} if name == "train" else {})


spec = get("kuairand_pure")
splits = {"train": rows("train", 0), "valid": rows("valid", 1),
          "test": rows("test", 987654321, label=False)}
contract = DC.build(spec, splits)

names = DC.column_names(contract)
check("contract lists exactly the loader columns",
      names == set(DC.columns_of(splits["train"])), str(sorted(names)))
check("contract records exact dtypes", all(c["dtype"] for c in contract["columns"]))
check("contract records row counts", contract["split_rows"] ==
      {"train": 2, "valid": 2, "test": 2})
check("held-out labels are explicitly unavailable",
      contract["label"]["held_out_values_visible_to_agent"] is False)
check("held-out feature values never enter the prompt contract",
      "987654321" not in json.dumps(contract, sort_keys=True))
check("official personalization is declared, not inferred",
      contract["personalization"]["required"] is True
      and contract["personalization"]["user_role"] == "user_id")
excluded = {x["source"]: x["reason"] for x in contract["excluded_sources"]}
check("unjoined user profiles are disclosed",
      "user_features_pure.csv" in excluded and bool(excluded["user_features_pure.csv"]))
check("full-window statistic leakage is disclosed",
      "held-out" in excluded["video_features_statistic_pure.csv"])
check("contract carries a stable content address", len(contract["sha256"]) == 64)
check("dataset fingerprint is distinct from contract hash",
      contract["dataset_fingerprint"] != contract["sha256"])

changed = dict(splits)
changed["train"] = rows("train", 3)
changed_contract = DC.build(spec, changed)
check("dataset fingerprint changes when loaded content changes",
      changed_contract["dataset_fingerprint"] != contract["dataset_fingerprint"])

good = {"id": "good", "data_columns": ["user_id", "temperature"]}
bad = {"id": "bad", "data_columns": ["user_id", "imaginary_profile"]}
check("declared real columns pass", DC.candidate_column_problems(good, contract) == [])
problems = DC.candidate_column_problems(bad, contract)
check("invented columns are refused before coding",
      problems and "imaginary_profile" in problems[0], str(problems))
check("missing declarations are refused",
      bool(DC.candidate_column_problems({"id": "missing"}, contract)))

print()
print("== D1: the label and the auxiliary outcomes are declarable, held-out never is")


def decl(*names):
    return {"id": "d", "data_columns": list(names)}


avail = DC.declarable_names(contract)
check("the training label is declarable",
      DC.candidate_column_problems(decl("user_id", "long_view"), contract) == [],
      str(DC.candidate_column_problems(decl("user_id", "long_view"), contract)))
check("the label is declarable on train and dev only",
      set(avail["long_view"]) == {"train", "valid"}, str(avail["long_view"]))
aux_names = [a["name"] for a in contract["auxiliary_training_outcomes"]]
check("every published auxiliary outcome is declarable",
      aux_names and all(DC.candidate_column_problems(decl("user_id", n), contract) == []
                        for n in aux_names), str(aux_names))
check("an auxiliary outcome is declarable on its own split",
      DC.candidate_column_problems(decl("train:secondary"), contract) == [])
check("an auxiliary outcome is REFUSED on a split that does not carry it",
      any("available on ['train'] only" in p
          for p in DC.candidate_column_problems(decl("valid:secondary"), contract)),
      str(DC.candidate_column_problems(decl("valid:secondary"), contract)))
check("a feature is declarable on any split that carries it",
      DC.candidate_column_problems(decl("test:temperature"), contract) == [])
held = DC.candidate_column_problems(decl("test:long_view"), contract)
check("held-out label values are refused however they are qualified",
      held and "never visible to the agent" in held[0], str(held))
check("the held-out alias is refused too",
      bool(DC.candidate_column_problems(decl("heldout:long_view"), contract)))
check("no declarable name carries the held-out split's labels",
      "test" not in avail["long_view"])
invented = DC.candidate_column_problems(decl("onehot_feat3"), contract)
check("invented and unjoined columns are still refused", bool(invented))
check("the refusal names the declarable surfaces so a repair can succeed",
      "the label" in invented[0] and "auxiliary outcomes" in invented[0], invented[0])
check("an unknown split qualifier is refused explicitly",
      any("unknown split" in p
          for p in DC.candidate_column_problems(decl("holdout2:user_id"), contract)),
      str(DC.candidate_column_problems(decl("holdout2:user_id"), contract)))
check("the Gate 2 false refusal of the task's own label is gone",
      not any("absent from the loader contract" in p
              for p in DC.candidate_column_problems(decl("long_view"), contract)))

print()
print("== D1: the declaration is not the boundary -- executable source still is")
from agent import candidate as CAND

seed = CAND.seed_tree(CAND.workspace_for("minimal"))
leaky = dict(seed)
leaky["train.py"] = seed["train.py"].replace(
    "def fit_predict(", "def fit_predict(", 1) + "\n\ndef _peek(data):\n    return data.heldout.label\n"
try:
    CAND.validate_tree(leaky, CAND.workspace_for("minimal"))
    refused = ""
except CAND.CandidateError as exc:
    refused = str(exc)
check("candidate source reading heldout.label is refused by the static validator",
      "heldout.label" in refused, refused[:120])
check("declaring the label in a hypothesis does NOT let source read held-out labels",
      DC.candidate_column_problems(decl("long_view"), contract) == []
      and "heldout.label" in refused)
check("the held-out split carries no label values at all",
      splits["test"].label is None)

portable = load(ROOT / "tests" / "fixtures" / "portable_recommender" / "task.json")
portable_splits, _ = portable.loader()
portable_contract = DC.build(portable, portable_splits)
portable_names = DC.column_names(portable_contract)
check("alternate schema is discovered without KuaiRand names",
      {"account_id", "product_id"} <= portable_names, str(sorted(portable_names)))
check("portable tasks do not become personalized by column-name guessing",
      portable_contract["personalization"]["required"] is False)

print()
bad_checks = [name for name, ok in R if not ok]
print(f"{len(R)-len(bad_checks)}/{len(R)} passed" +
      (f"  FAILURES: {bad_checks}" if bad_checks else ""))
sys.exit(1 if bad_checks else 0)
