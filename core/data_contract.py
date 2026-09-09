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
    """Ordinary FEATURE column names. Not the label, not the auxiliary outcomes."""
    return {str(c.get("name")) for c in (contract.get("columns") or [])
            if isinstance(c, dict) and c.get("name")}


def declarable_names(contract: dict) -> dict[str, tuple[str, ...]]:
    """Every name a hypothesis may declare, mapped to the splits that actually carry it.

    Three surfaces, and all three are already published by the contract itself:

      * feature columns, on the splits listed in their ``available_on``;
      * the training LABEL, on the splits listed in ``label.visible_on`` -- train and dev.
        A ranking objective cannot be written without it, and withholding it from the
        declaration vocabulary made the harness refuse correct pairwise and listwise
        candidates while the contract listed the label three keys away;
      * auxiliary training outcomes, on their own ``available_on`` -- train only.

    The held-out split is absent from every entry by construction: no held-out label value
    is in this process, and ``build()`` refuses to produce a contract if one were.
    """
    out: dict[str, tuple[str, ...]] = {}
    for c in (contract.get("columns") or []):
        if isinstance(c, dict) and c.get("name"):
            out[str(c["name"])] = tuple(str(s) for s in (c.get("available_on") or ()))
    label = contract.get("label") or {}
    if label.get("name"):
        out[str(label["name"])] = tuple(str(s) for s in (label.get("visible_on") or ()))
    for a in (contract.get("auxiliary_training_outcomes") or []):
        if isinstance(a, dict) and a.get("name"):
            name = str(a["name"])
            splits = tuple(str(s) for s in (a.get("available_on") or ()))
            # duration_ms is both a feature and a train-only aux outcome. The feature
            # entry is the wider one, so it wins rather than being narrowed to train.
            out[name] = tuple(dict.fromkeys(out.get(name, ()) + splits))
    return out


def _held_out_aliases(contract: dict) -> set[str]:
    """Names that mean "the split whose labels decide the competition"."""
    held = str((contract.get("split") or {}).get("held_out") or "")
    return {s for s in (held, "heldout", "held_out", "test") if s}


def _parse_declaration(entry: str) -> tuple[str | None, str]:
    """``"train:is_click"`` -> ``("train", "is_click")``; ``"user_id"`` -> ``(None, ...)``.

    The qualifier is OPTIONAL and exists only so a declaration can be REFUSED for using a
    name on a split that does not carry it. It never widens access: an unqualified name is
    checked against the splits the contract publishes, and the executable boundary is
    agent/candidate.py's static refusal of `heldout.label`, not this declaration.
    """
    split, sep, name = entry.partition(":")
    if not sep:
        return None, entry.strip()
    return split.strip(), name.strip()


def candidate_column_problems(candidate: dict, contract: dict) -> list[str]:
    """Validate a hypothesis' explicit data declaration against the loader contract.

    Generated features belong in ``derived_features`` and are not treated as raw columns.
    Parsing free prose for identifiers would create false refusals, so the declaration is
    required and checked directly.

    Every refusal says WHY, because the message is what the planner repairs against. An
    earlier version reported "absent from the loader contract" for the task's own label,
    which taught the model something false about the dataset.
    """
    cid = str(candidate.get("id") or candidate.get("_candidate_id") or "candidate")
    declared = candidate.get("data_columns")
    if not isinstance(declared, list):
        return [f"candidate {cid} must declare data_columns as an array"]
    bad_types = [x for x in declared if not isinstance(x, str) or not x.strip()]
    if bad_types:
        return [f"candidate {cid} has invalid data_columns entries"]

    available = declarable_names(contract)
    held_out = _held_out_aliases(contract)
    label_name = str((contract.get("label") or {}).get("name") or "")
    known_splits = {s for splits in available.values() for s in splits} | held_out
    problems: list[str] = []
    unknown: list[str] = []

    for entry in declared:
        split, name = _parse_declaration(entry)
        if name not in available:
            unknown.append(name)
            continue
        if split is None:
            continue
        if split not in known_splits:
            problems.append(
                f"candidate {cid} qualifies {name!r} with an unknown split {split!r}; "
                f"the declared splits are {sorted(known_splits)}")
        elif split not in available[name]:
            if name == label_name and split in held_out:
                problems.append(
                    f"candidate {cid} declares {entry!r}, but held-out label values are "
                    f"never visible to the agent and are not in this process at all. "
                    f"Train and early-stop on {list(available[name])} only")
            else:
                problems.append(
                    f"candidate {cid} declares {entry!r}, but {name!r} is available on "
                    f"{list(available[name])} only")

    if unknown:
        feature = sorted(column_names(contract))
        aux = sorted(str(a.get("name")) for a in
                     (contract.get("auxiliary_training_outcomes") or [])
                     if isinstance(a, dict) and a.get("name"))
        problems.insert(0, (
            f"candidate {cid} references names absent from the loader contract: "
            f"{sorted(set(unknown))}. Declarable names are the loader feature columns "
            f"{feature}, the label {label_name!r} (train and dev), and the train-only "
            f"auxiliary outcomes {aux}. Computed quantities belong in derived_features"))
    return problems
