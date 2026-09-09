"""Deterministic, prompt-safe description of the data a candidate can actually use.

The task declaration says what the problem means.  This module measures the narrower
runtime surface produced by that task's loader.  It never profiles the raw directory and
never reads held-out labels.  The resulting contract is therefore suitable for an LLM
prompt and, unlike prose or candidate telemetry, is available before the first paid call.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields as dataclass_fields, is_dataclass

import numpy as np


VERSION = 1


def columns_of(rows) -> tuple[str, ...]:
    """Return public array columns carried by a loaded row object.

    This deliberately mirrors the candidate contract without importing a particular seed
    workspace.  Properties such as ``row_id`` and bookkeeping members are excluded.
    """
    if hasattr(rows, "_column_names"):
        names = list(getattr(rows, "_column_names"))
        # Portable loaders may add canonical aliases after their original columns.
        for alias in ("user_id", "video_id"):
            if hasattr(rows, alias) and alias not in names:
                names.append(alias)
    elif is_dataclass(rows):
        names = [f.name for f in dataclass_fields(rows)]
    else:
        names = [n for n in dir(rows) if not n.startswith("_")]
    out = []
    for name in names:
        if name in ("label", "aux", "split", "row_id"):
            continue
        value = getattr(rows, name, None)
        if callable(value):
            continue
        try:
            if len(value) != len(rows):
                continue
        except (TypeError, AttributeError):
            continue
        out.append(str(name))
    return tuple(dict.fromkeys(out))


def _dtype(value) -> str:
    return str(np.asarray(value).dtype)


def _hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _content_identity(rows) -> str | None:
    fn = getattr(rows, "content_sha256", None)
    if not callable(fn):
        return None
    return str(fn())


def build(spec, splits: dict) -> dict:
    """Build the authoritative contract from strategy-visible loaded splits only."""
    dev = spec.split.dev_split
    held = spec.split.held_out_split
    required_splits = ("train", dev, held)
    missing = [name for name in required_splits if name not in splits]
    if missing:
        raise ValueError(f"cannot build data contract; loader omitted splits {missing}")
    if getattr(splits[held], "label", None) is not None:
        raise ValueError("cannot build prompt contract while held-out labels are visible")

    split_columns = {name: columns_of(splits[name]) for name in required_splits}
    all_columns = sorted(set().union(*(set(v) for v in split_columns.values())))
    roles = {
        "group": spec.metric.group_by,
        "user": (spec.personalization or {}).get("user_role"),
        "item": spec.knowledge.get("item_col"),
        "time": spec.knowledge.get("time_col") or spec.split.column,
    }
    columns = []
    for name in all_columns:
        available = [s for s in required_splits if name in split_columns[s]]
        role = [key for key, value in roles.items() if value == name]
        # Portable loaders normally expose the same schema on every split, but the
        # contract should describe a legitimate dev-only field instead of crashing while
        # assuming that train owns it. Values still never enter the contract.
        source_split = next(s for s in required_splits if name in split_columns[s])
        columns.append({
            "name": name,
            "dtype": _dtype(getattr(splits[source_split], name)),
            "role": role or ["feature"],
            "available_on": available,
        })

    auxiliary = getattr(splits["train"], "aux", {}) or {}
    aux_columns = [{"name": str(name), "dtype": _dtype(value),
                    "role": ["auxiliary_training_outcome"],
                    "available_on": ["train"]}
                   for name, value in sorted(auxiliary.items())]

    identity = {
        "task": spec.slug,
        "split_rows": {name: len(splits[name]) for name in required_splits},
        "split_content_sha256": {
            name: _content_identity(splits[name]) for name in required_splits},
        "columns": columns,
    }
    contract = {
        "version": VERSION,
        "task": spec.slug,
        "dataset_fingerprint": _hash(identity),
        "columns": columns,
        "auxiliary_training_outcomes": aux_columns,
        "split_rows": identity["split_rows"],
        "split": {
            "name": spec.split.name,
            "kind": spec.split.kind,
            "column": spec.split.column,
            "bounds": spec.split.bounds,
            "development": dev,
            "held_out": held,
        },
        "label": {
            "name": spec.label,
            "visible_on": ["train", dev],
            "held_out_values_visible_to_agent": False,
        },
        "roles": roles,
        "metric": {
            "name": spec.metric.name,
            "primary_key": spec.metric.primary_key,
            "higher_is_better": bool(spec.metric.higher_is_better),
        },
        "personalization": dict(spec.personalization or {"required": False}),
        "loaded_sources": list(spec.knowledge.get("loaded_sources") or []),
        "excluded_sources": list(spec.knowledge.get("excluded_sources") or []),
        "authority_note": (
            "Columns describe the loader output, not every file in the dataset directory. "
            "This contract overrides conflicting LLM-authored beliefs. Excluded sources "
            "need an explicitly validated, leakage-safe loader change before use."
        ),
    }
    contract["sha256"] = _hash(contract)
    return contract


def column_names(contract: dict) -> set[str]:
    return {str(c.get("name")) for c in (contract.get("columns") or [])
            if isinstance(c, dict) and c.get("name")}


def candidate_column_problems(candidate: dict, contract: dict) -> list[str]:
    """Validate a hypothesis' explicit raw-column declaration.

    Generated features belong in ``derived_features`` and are not treated as raw columns.
    Parsing free prose for identifiers would create false refusals, so the declaration is
    required and checked directly.
    """
    cid = str(candidate.get("id") or "candidate")
    declared = candidate.get("data_columns")
    if not isinstance(declared, list):
        return [f"candidate {cid} must declare data_columns as an array"]
    bad_types = [x for x in declared if not isinstance(x, str) or not x.strip()]
    if bad_types:
        return [f"candidate {cid} has invalid data_columns entries"]
    unknown = sorted(set(declared) - column_names(contract))
    if unknown:
        return [f"candidate {cid} references columns absent from the loader contract: "
                f"{unknown}"]
    return []
