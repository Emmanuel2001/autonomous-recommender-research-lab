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
