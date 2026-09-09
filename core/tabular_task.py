"""Runnable JSON TaskSpecs for numeric/categorical recommender event tables.

This closes the gap documented by agent/draft_task.py: a confirmed task declaration can
now become a real loader and scorer input without editing Python.  The first supported
portable family is logged-impression ranking/classification over CSV files.  It is narrow
on purpose and validates every assumption instead of claiming arbitrary AutoML.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from core.task import MetricSpec, SplitSpec, SubmissionSpec, TaskSpec

METRICS = ("within_user_ranking_gauc_ndcg5", "global_auc")


class PortableTaskError(ValueError):
    pass


class TabularRows:
    """Array-backed rows with dynamic, schema-declared feature columns."""
    def __init__(self, split: str, columns: dict[str, np.ndarray], label):
        self.split = split
        self.label = None if label is None else np.asarray(label, dtype=np.float64)
        self.aux = {}
        self._column_names = tuple(columns)
        for name, values in columns.items():
            setattr(self, name, np.asarray(values))
        lengths = {len(v) for v in columns.values()}
        if self.label is not None:
            lengths.add(len(self.label))
        if len(lengths) != 1:
            raise PortableTaskError(f"split {split!r} columns have inconsistent lengths")
        self._length = next(iter(lengths), 0)

    def __len__(self):
        return self._length

    def content_sha256(self) -> str:
        h = hashlib.sha256(self.split.encode())
        for name in self._column_names:
            a = np.ascontiguousarray(getattr(self, name))
            h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
        return h.hexdigest()


def _value(text: str):
    """Stable numeric representation without learning a vocabulary from held-out rows."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            # A deterministic hash preserves equality without exposing test categories to
            # a train-fitted vocabulary.  63 bits fit safely in signed NumPy integers.
            return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def validate_config(d: dict, source: Path) -> dict:
    required = {"slug", "title", "data_dir", "files", "label", "time_col",
                "splits", "metric", "feature_cols"}
    missing = sorted(required - set(d))
    if missing:
        raise PortableTaskError(f"{source}: missing fields {missing}")
    if d["metric"] not in METRICS:
        raise PortableTaskError(f"metric must be one of {list(METRICS)}")
    if not isinstance(d["files"], list) or not d["files"]:
        raise PortableTaskError("files must be a non-empty list")
    if not isinstance(d["feature_cols"], list) or not d["feature_cols"]:
        raise PortableTaskError("feature_cols must be a non-empty list")
    if d["metric"] == "within_user_ranking_gauc_ndcg5" and not d.get("group_by"):
        raise PortableTaskError("within-user ranking requires group_by")
    if "personalization" in d and not isinstance(d["personalization"], dict):
        raise PortableTaskError("personalization must be an object when provided")
    for name in ("train", "valid", "test"):
        bounds = (d.get("splits") or {}).get(name)
        if not (isinstance(bounds, list) and len(bounds) == 2
                and int(bounds[0]) <= int(bounds[1])):
            raise PortableTaskError(f"splits.{name} must be [low, high]")
    base = Path(d["data_dir"])
    if not base.is_absolute():
        base = (source.parent / base).resolve()
    for name in d["files"]:
        p = (base / name).resolve()
        # Containment: a declared filename must stay inside data_dir. Without this,
        # files: ["../something.csv"] would be staged into the redacted root that the
        # candidate reads, i.e. a task declaration choosing its own inputs from outside the
        # directory the operator confirmed. resolve() first so "a/../../b" cannot slip past.
        try:
            p.relative_to(base)
        except ValueError:
            raise PortableTaskError(
                f"declared file {name!r} resolves outside data_dir ({p}); portable task "
                f"files must live inside the declared data directory") from None
        if not p.is_file():
            raise PortableTaskError(f"declared data file does not exist: {p}")
        with p.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        need = set(d["feature_cols"]) | {d["label"], d["time_col"]}
        if d.get("group_by"):
            need.add(d["group_by"])
        if d.get("item_col"):
            need.add(d["item_col"])
        absent = sorted(need - set(header))
        if absent:
            raise PortableTaskError(f"{p.name} is missing columns {absent}")
    out = dict(d); out["data_dir"] = str(base)
    return out


def _read(config: dict, *, scorer: bool) -> tuple[dict[str, TabularRows], None]:
    # Inside a strategy sandbox, read only the harness-prepared/redacted root.  The scorer
    # has no TRACK2_SANDBOX and deliberately reads the originals with labels intact.
    root = (Path(os.environ["TRACK2_DATA_ROOT"]) if os.environ.get("TRACK2_SANDBOX")
            else Path(config["data_dir"]))
    names = list(dict.fromkeys(config["feature_cols"] +
                              [config["time_col"], config["label"]]
                              + ([config["group_by"]] if config.get("group_by") else [])
                              + ([config["item_col"]] if config.get("item_col") else [])))
    raw = {n: [] for n in names}
    for filename in config["files"]:
        with (root / filename).open(newline="") as fh:
            for row in csv.DictReader(fh):
                for name in names:
                    raw[name].append(_value(row[name]))
    arrays = {n: np.asarray(v) for n, v in raw.items()}
    time_values = arrays[config["time_col"]].astype(np.int64)
    out = {}
    for split, bounds in config["splits"].items():
        lo, hi = map(int, bounds)
        mask = (time_values >= lo) & (time_values <= hi)
        columns = {n: a[mask] for n, a in arrays.items() if n != config["label"]}
        group = config.get("group_by")
        item = config.get("item_col")
        # The existing safe candidate contract uses canonical recommender aliases for its
        # deterministic train subset. Preserve original names and add aliases if needed.
        if group and "user_id" not in columns:
            columns["user_id"] = arrays[group][mask].astype(np.int64)
        if item and "video_id" not in columns:
            columns["video_id"] = arrays[item][mask].astype(np.int64)
        label = arrays[config["label"]][mask]
        if split == "test" and not scorer:
            label = None
        out[split] = TabularRows(split, columns, label)
        if len(out[split]) == 0:
            raise PortableTaskError(f"declared split {split!r} has no rows")
    return out, None


def load_task_spec(path: str | Path) -> TaskSpec:
    path = Path(path).resolve()
    cfg = validate_config(json.loads(path.read_text()), path)
    if cfg["metric"] == "within_user_ranking_gauc_ndcg5":
        from core.metrics import primary
        metric = MetricSpec("primary = mean(GAUC, nDCG@5)", primary, "primary",
                            group_by=cfg["group_by"],
                            conventions=("within-group logged-impression ranking",))
    else:
        from core.official_evaluate import auc
        def global_auc(_group, labels, scores):
            value = float(auc(labels, scores))
            return {"AUC": value, "primary": value, "rows": len(labels)}
        metric = MetricSpec("global AUC", global_auc, "primary", group_by=None)

    loader = lambda: _read(cfg, scorer=False)
    scorer_loader = lambda: _read(cfg, scorer=True)
    expect = {k: len(v) for k, v in loader()[0].items()}
    return TaskSpec(
        slug=cfg["slug"], title=cfg["title"], data_dir=Path(cfg["data_dir"]),
        label=cfg["label"], metric=metric,
        split=SplitSpec("declared time ranges", "date_range", cfg["splits"],
                        column=cfg["time_col"], dev_split="valid",
                        held_out_split="test", expect_rows=expect),
        submission=SubmissionSpec(tuple(cfg.get("submission_header") or
                                        ("row_id", "score")), "row_id", True,
                                  "portable positional score submission"),
        loader=loader, scorer_loader=scorer_loader,
        baseline=dict(cfg.get("baseline") or {}),
        convergence=dict(cfg.get("convergence") or {"epsilon": 0.002, "N": 3,
                                                     "source": "task specification"}),
        knowledge={"task_spec_path": str(path),
                   "feature_cols": list(cfg["feature_cols"]),
                   "source_files": list(cfg["files"]),
                   "loaded_sources": list(cfg["files"]),
                   "time_col": cfg["time_col"],
                   "outcome_columns": list(cfg.get("outcome_columns") or [cfg["label"]]),
                   "item_col": cfg.get("item_col")},
        personalization=dict(cfg.get("personalization") or {"required": False}))
