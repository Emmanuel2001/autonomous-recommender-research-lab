"""The typed action space, and the JSON contract a proposer must satisfy.

DESIGN
------
An ENVELOPE with a free-form `params` dict, validated against the chosen action's
schema in a SECOND step. Not an N-way discriminated union in one decode: a single
static envelope schema means the output contract never changes mid-run, so it
cannot invalidate a cached prompt prefix, and a malformed action produces a
targeted error rather than a whole-union rejection.

Every action is a pure function Config -> Config. That is what makes an
iteration reproducible from its ledger entry: replay the action list, get the
config back.

Every action carries a REQUIRED rationale. Innovation is scored on *what the
agent chose to try and why*, explicitly not on implementation, so an action with
no stated mechanism is worth nothing even if it works. The rationale also gates
the action: an expected_magnitude below the measured noise floor is refused, which
forces noise-floor reasoning on every proposal.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

# ---- vocabulary the proposer must draw from --------------------------------
MODELS = ("ease", "itemknn", "als", "bpr_mf", "popularity")
SIGNALS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward",
           "is_hate", "long_view", "is_profile_enter", "play_ratio")
METRICS = ("ndcg@10", "recall@50", "both")
SYMPTOMS = ("lambda_plateau", "click_only_features", "weak_model_class",
            "content_unused", "signals_unused", "below_noise_floor",
            "exposure_bias", "train_data_limited")

# ---- the action schema: name -> {param: (type, lo, hi) or ("enum", choices)} --
SPEC: dict[str, dict[str, Any]] = {
    "SwapModel":        {"model": ("enum", MODELS)},
    "ChangeEaseLambda": {"value": ("float", 1.0, 1e5)},
    "SetSignalWeights": {"weights": ("weights", SIGNALS)},
    "SetContentAlpha":  {"alpha": ("float", 0.0, 1.0)},
    "SetContentTopK":   {"topk": ("int", 1, 7583)},
    "ChangeKnnK":       {"k": ("int", 1, 7583)},
    "ChangeDim":        {"dim": ("int", 4, 512)},
    "ChangeEpochs":     {"epochs": ("int", 1, 100)},
    "ChangeLr":         {"lr": ("float", 1e-5, 1.0)},
    "ChangeAlsIters":   {"iters": ("int", 1, 50)},
    "SetMask":          {"mask": ("enum", ("none", "exclude_train_seen"))},
    # gated by --allow-log-random on the driver; the proposer may ask, the harness
    # refuses unless the operator has enabled it
    "SetUseLogRandom":  {"enabled": ("enum", (True, False))},
}
ACTION_TYPES = tuple(SPEC)

# ---- how each action mutates the Config -----------------------------------
_APPLY = {
    "SwapModel":        lambda c, p: replace(c, model=p["model"]),
    "ChangeEaseLambda": lambda c, p: replace(c, ease_lambda=float(p["value"])),
    "SetSignalWeights": lambda c, p: replace(c, signal_weights=dict(p["weights"])),
    "SetContentAlpha":  lambda c, p: replace(c, content_alpha=float(p["alpha"])),
    "SetContentTopK":   lambda c, p: replace(c, content_topk=int(p["topk"])),
    "ChangeKnnK":       lambda c, p: replace(c, knn_k=int(p["k"])),
    "ChangeDim":        lambda c, p: replace(c, dim=int(p["dim"])),
    "ChangeEpochs":     lambda c, p: replace(c, epochs=int(p["epochs"])),
    "ChangeLr":         lambda c, p: replace(c, lr=float(p["lr"])),
    "ChangeAlsIters":   lambda c, p: replace(c, als_iters=int(p["iters"])),
    "SetMask":          lambda c, p: replace(c, mask=p["mask"]),
    "SetUseLogRandom":  lambda c, p: replace(c, use_log_random=bool(p["enabled"])),
}


class ActionError(ValueError):
    """Rejected proposal. Cheap to re-sample; never a 'repair'."""


def _check_param(name: str, value: Any, spec: tuple) -> Any:
    kind = spec[0]
    if kind == "enum":
        if value not in spec[1]:
            raise ActionError(f"{name}={value!r} not in {spec[1]}")
        return value
    if kind == "weights":
        if not isinstance(value, dict) or not value:
            raise ActionError(f"{name} must be a non-empty object")
        bad = set(value) - set(spec[1])
        if bad:
            raise ActionError(f"unknown signal(s) {sorted(bad)}; choose from {spec[1]}")
        if "is_click" not in value:
            raise ActionError("is_click must be present: it is the SCORED signal")
        out = {}
        for k, w in value.items():
            w = float(w)
            if not 0.0 <= w <= 10.0:
                raise ActionError(f"weight {k}={w} outside [0,10]")
            out[k] = w
        return out
    lo, hi = spec[1], spec[2]
    try:
        v = int(value) if kind == "int" else float(value)
    except (TypeError, ValueError):
        raise ActionError(f"{name}={value!r} is not a {kind}") from None
    if not lo <= v <= hi:
        raise ActionError(f"{name}={v} outside [{lo}, {hi}]")
    return v


def validate_rationale(r: Any, min_effect: dict[str, float]) -> dict:
    """A proposal must say what it expects, and expect something detectable."""
    if not isinstance(r, dict):
        raise ActionError("rationale must be an object")
    for f in ("mechanism", "target_metric", "expected_direction", "expected_magnitude"):
        if f not in r:
            raise ActionError(f"rationale missing required field {f!r}")
    if not isinstance(r["mechanism"], str) or len(r["mechanism"].strip()) < 20:
        raise ActionError("rationale.mechanism must be a real sentence (>=20 chars)")
    if r["target_metric"] not in METRICS:
        raise ActionError(f"target_metric must be one of {METRICS}")
    if r["expected_direction"] not in ("increase", "decrease"):
        raise ActionError("expected_direction must be 'increase' or 'decrease'")
    try:
        mag = abs(float(r["expected_magnitude"]))
    except (TypeError, ValueError):
        raise ActionError("expected_magnitude must be a number") from None
    # the floor the action must clear to be worth an evaluation at all
    tm = r["target_metric"]
    floor = (min(min_effect.values()) if tm == "both"
             else float(min_effect.get(tm, 0.0)))
    if mag < floor:
        raise ActionError(
            f"expected_magnitude {mag:.5f} is below the measured minimum "
            f"detectable effect for {tm} ({floor:.5f}); this experiment cannot "
            f"resolve an effect that small -- propose something larger or request "
            f"more seeds")
    bad = set(r.get("symptoms_addressed", [])) - set(SYMPTOMS)
    if bad:
        raise ActionError(f"unknown symptom(s) {sorted(bad)}; choose from {SYMPTOMS}")
    return {"mechanism": r["mechanism"].strip(), "target_metric": tm,
            "expected_direction": r["expected_direction"],
            "expected_magnitude": mag,
            "cites": list(r.get("cites", [])),
            "symptoms_addressed": list(r.get("symptoms_addressed", []))}


def validate(envelope: Any, min_effect: dict[str, float]) -> dict:
    """Full validation of one proposal. Raises ActionError with a usable message."""
    if isinstance(envelope, str):
        try:
            envelope = json.loads(envelope)
        except json.JSONDecodeError as e:
            raise ActionError(f"not valid JSON: {e}") from None
    if not isinstance(envelope, dict):
        raise ActionError("envelope must be a JSON object")
    extra = set(envelope) - {"action_type", "params", "rationale", "parent_node_id"}
    if extra:
        raise ActionError(f"unexpected field(s) {sorted(extra)}")
    at = envelope.get("action_type")
    if at not in SPEC:
        raise ActionError(f"action_type {at!r} unknown; choose from {ACTION_TYPES}")
    params = envelope.get("params")
    if not isinstance(params, dict):
        raise ActionError("params must be an object")
    spec = SPEC[at]
    missing = set(spec) - set(params)
    if missing:
        raise ActionError(f"{at} missing param(s) {sorted(missing)}")
    unknown = set(params) - set(spec)
    if unknown:
        raise ActionError(f"{at} got unexpected param(s) {sorted(unknown)}")
    clean = {k: _check_param(k, params[k], spec[k]) for k in spec}
    return {"action_type": at, "params": clean,
            "rationale": validate_rationale(envelope.get("rationale"), min_effect),
            "parent_node_id": envelope.get("parent_node_id")}


def apply(cfg, action: dict):
    """Config -> Config. Pure; the same action always yields the same config."""
    return _APPLY[action["action_type"]](cfg, action["params"])


def _params_schema() -> dict:
    """Every param name across every action, as TYPED OPTIONAL properties.

    Not a free-form object. A schema-constrained decode uses the schema, not the
    prose, to decide what keys to emit -- measured: with `params` left as an
    unconstrained object, gemini-3.5-flash-lite chose SetSignalWeights three times
    in a row and omitted `weights` every time, so 3/3 proposals were rejected.

    Not an N-way oneOf union either: that changes the output contract per action and
    is poorly supported across schema subsets. One static shape, all names visible,
    with `validate()` enforcing that exactly the right subset is present.
    """
    sig_props = {s: {"type": "number", "minimum": 0, "maximum": 10} for s in SIGNALS}
    return {
        "type": "object", "additionalProperties": False,
        "description": ("Include ONLY the parameters this action_type takes -- see "
                        "the action table in the system prompt. Omit all others."),
        "properties": {
            "model": {"type": "string", "enum": list(MODELS),
                      "description": "SwapModel"},
            "value": {"type": "number", "description": "ChangeEaseLambda"},
            "weights": {"type": "object", "additionalProperties": False,
                        "properties": sig_props, "required": ["is_click"],
                        "description": "SetSignalWeights. is_click is REQUIRED "
                                       "(it is the scored signal)."},
            "alpha": {"type": "number", "minimum": 0, "maximum": 1,
                      "description": "SetContentAlpha"},
            "topk": {"type": "integer", "description": "SetContentTopK"},
            "k": {"type": "integer", "description": "ChangeKnnK"},
            "dim": {"type": "integer", "description": "ChangeDim"},
            "epochs": {"type": "integer", "description": "ChangeEpochs"},
            "lr": {"type": "number", "description": "ChangeLr"},
            "iters": {"type": "integer", "description": "ChangeAlsIters"},
            "mask": {"type": "string", "enum": ["none", "exclude_train_seen"],
                     "description": "SetMask"},
            "enabled": {"type": "boolean", "description": "SetUseLogRandom"},
        },
    }


def json_schema() -> dict:
    """The output contract handed to an LLM backend. Static across the whole run."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["action_type", "params", "rationale"],
        "properties": {
            "action_type": {"type": "string", "enum": list(ACTION_TYPES)},
            "params": _params_schema(),
            "parent_node_id": {"type": ["string", "null"]},
            "rationale": {
                "type": "object", "additionalProperties": False,
                "required": ["mechanism", "target_metric", "expected_direction",
                             "expected_magnitude"],
                "properties": {
                    "mechanism": {"type": "string", "minLength": 20},
                    "target_metric": {"type": "string", "enum": list(METRICS)},
                    "expected_direction": {"type": "string",
                                           "enum": ["increase", "decrease"]},
                    "expected_magnitude": {"type": "number"},
                    "cites": {"type": "array", "items": {"type": "string"}},
                    "symptoms_addressed": {"type": "array",
                                           "items": {"type": "string",
                                                     "enum": list(SYMPTOMS)}},
                },
            },
        },
    }
