"""The agent's typed action space for the CURRENT task.

WHY A NEW MODULE
----------------
`agent/actions.py` targets the superseded task: EASE lambda, content blending, ItemKNN
k, masking policy. None of those exist here. That file and its 30 tests stay for
reproducing the old runs.

WHAT MAKES THIS ONE DIFFERENT: THE ORGANISERS PUBLISHED THEIR DEAD ENDS
----------------------------------------------------------------------
The Starter Kit lists, with measured numbers, two things that do NOT work and seven
directions nobody has tried. That converts a chunk of the search from exploration into
*known* territory, and the honest thing to do with a published negative result is refuse
to spend an iteration on it -- and say why, with their number, in the refusal.

Refused with citation:
  * more static feature fields   .5940 with 13 fields vs .5950 with 5 -- no gain
  * more embedding capacity      k=8/16/32 -> .5895/.5902/.5887 -- no gain
  * pure user-side features      PROVABLY inert, not merely unhelpful: ranking happens
                                 within a user, so any term constant across that user's
                                 rows cannot change their order. This is the strongest
                                 kind of rejection available -- it is arithmetic, not a
                                 measurement that might not replicate.

Encouraged (their own priority order): loss replacement, history sequences,
multi-objective, censored watch-time, then bigger models, then time/drift, then
unbiased validation via log_random.
"""
from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------- known dead ends
#: field -> (their measured number, the citation). Used to refuse, with evidence.
DEAD_ENDS = {
    "more_static_fields": (
        "primary .5940 with all 13 CWM feature fields vs .5950 with the baseline 5 -- "
        "inside noise and slightly worse. The organisers measured this; "
        "user_id x video_id already carries most of the learnable signal."),
    "more_capacity": (
        "k = 8 / 16 / 32 gives primary .5895 / .5902 / .5887 -- it barely moves. "
        "1.14M rows will not support more capacity. The organisers measured this."),
    "side_table_fields": (
        "MEASURED 2026-08-27, 11-arm ablation on valid, pairwise, seed 0. Joining the "
        "user_features / video_features side tables adds NOTHING: music_id +0.00028, "
        "video_type +0.00016, upload_type +0.00010, follow_user_num_range +0.00008, "
        "fans_user_num_range +0.00014, friend_user_num_range +0.00032, "
        "register_days_range -0.00007, user_active_degree -0.00009 -- every one inside the "
        "0.0008 seed sd, two negative. All 14 fields together (+0.00093) scored WORSE than "
        "the baseline 5 plus hourmin_bucket alone (+0.00113), so they actively dilute. Do "
        "not propose a side-table join; it has been done and measured."),
    "field_choice_rule": (
        "A field can only help if it VARIES WITHIN A USER -- ranking is within-user, so a "
        "value constant across a user's own rows cannot reorder them. Measured: all five "
        "*_num_range / user_active_degree fields vary in 0.0% of users and all scored inside "
        "noise, exactly as the arithmetic predicts. Varying is necessary but NOT sufficient: "
        "music_id (81.8% varying), upload_type (74.2%) and video_type (4.6%) also failed, "
        "because they are item attributes already carried by video_id. The one field that "
        "pays is hourmin_bucket -- it varies (79.1%) AND it is CONTEXT rather than item "
        "identity: +0.001192 paired over 5 seeds, t=14.04, 5/5 wins, which takes the "
        "combined valid delta past the organisers' epsilon=0.002 for the first time. Ask of "
        "any candidate feature: does it change within one user's impressions, and does it "
        "say something video_id does not?"),
    "user_side_only": (
        "PROVABLY inert: ranking is within-user, so a term that is constant across a "
        "user's own rows cannot reorder them. The organisers confirmed it empirically "
        "(item_pop x user-bias scores identically to plain item_pop), but the argument "
        "is arithmetic, not empirical. User signal must enter through a CROSS with the "
        "item side."),
}

#: The five baseline fields. Adding *raw categorical* fields beyond these is the
#: measured dead end; derived/crossed/sequence features are NOT and are encouraged.
BASELINE_FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
#: Only fields the LOADER can actually provide. This list previously advertised nine more,
#: drawn from the user_features and video_features side tables, and every one of them raised
#: at encode time because `core.official_data.Rows` does not carry them -- the action space
#: offered a lever wired to nothing. No agent happened to propose one, so it never fired, but
#: it would have cost an iteration out of a budget the epsilon/N rule caps near three.
#: The nine were then MEASURED (11-arm ablation, valid, pairwise, seed 0) before being
#: dropped rather than after: eight are worth nothing and dilute, and the ninth,
#: hourmin_bucket, is worth +0.001192 paired over 5 seeds (t=14.04) and is now implemented
#: in strategy.ranker.encode. See DEAD_ENDS["side_table_fields"] and ["field_choice_rule"].
KNOWN_FIELDS = BASELINE_FIELDS + ("hourmin_bucket",)

#: Fields that were offered, MEASURED, and removed -- kept here so proposing one still gets
#: the reason rather than a bare "unknown field". Deleting a lever without keeping its
#: measurement means the next agent (or the next me) pays to rediscover it.
#: Non-baseline fields that are MEASURED WINS, so the piling-on dead end must not refuse
#: them. The dead-ends note has always said derived features are encouraged; this is the
#: first one measured, and the guard had to be told.
ENCOURAGED_EXTRA = ("hourmin_bucket",)

REMOVED_FIELDS = {
    "music_id": "+0.00028 valid (81.8% within-user varying, but video_id already identifies "
                "the item)",
    "video_type": "+0.00016 valid (varies for only 4.6% of users, 3 distinct values)",
    "upload_type": "+0.00010 valid (74.2% varying but subsumed by video_id)",
    "follow_user_num_range": "+0.00008 valid; PROVABLY inert -- 0.0% within-user variation",
    "register_days_range": "-0.00007 valid; PROVABLY inert -- 0.0% within-user variation",
    "fans_user_num_range": "+0.00014 valid; PROVABLY inert -- 0.0% within-user variation",
    "friend_user_num_range": "+0.00032 valid; PROVABLY inert -- 0.0% within-user variation",
    "user_active_degree": "-0.00009 valid; PROVABLY inert -- 0.0% within-user variation",
}
#: Fields that are pure user-side. On their own they cannot change a within-user order.
USER_SIDE_FIELDS = ("user_id", "follow_user_num_range", "register_days_range",
                    "fans_user_num_range", "friend_user_num_range",
                    "user_active_degree")

AUX_LABELS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward",
              "is_profile_enter", "play_time_ms")
#: A minimal patch that satisfies the contract. Shown to the model verbatim, because
#: an action the prompt cannot DESCRIBE is an action the agent does not have -- measured
#: three times now: an unconstrained params object gave 3/3 rejections, a stale worked
#: example gave 12/12, and rendering this action's bounds as the raw tuple
#: "{'code': ('code',)}" gave 0 uses across two full runs. Deliberately trivial: it
#: shows the SHAPE, so the idea still has to come from the model.
MINIMAL_PATCH_EXAMPLE = '\n'.join([
    '"""One-line description of the idea."""',
    'import numpy as np',
    '',
    'def fit_predict(train, dev, heldout, params):',
    '    # train.label and dev.label are float 0/1 arrays. heldout.label is None.',
    '    # Every split exposes parallel int64 arrays: user_id, video_id, author_id,',
    '    # tab, date, hourmin, plus float64 duration_ms. train.aux holds extra signals.',
    '    prior = float(train.label.mean())',
    '    num, den = {}, {}',
    '    for v, y in zip(train.video_id, train.label):',
    '        k = int(v)',
    '        num[k] = num.get(k, 0.0) + y',
    '        den[k] = den.get(k, 0.0) + 1.0',
    '',
    '    def score(rows):',
    '        out = np.empty(len(rows.video_id), dtype=float)',
    '        for i, v in enumerate(rows.video_id):',
    '            k = int(v)',
    '            out[i] = (num.get(k, 0.0) + 20 * prior) / (den.get(k, 0.0) + 20)',
    '        return out',
    '',
    '    return {"dev": score(dev), "heldout": score(heldout)}',
])

#: How each param KIND is described to the model. The raw spec tuple is meaningless to
#: a reader, and an action described meaninglessly does not get used.
KIND_HELP = {
    "code": ("a complete Python module (string) defining "
             "fit_predict(train, dev, heldout, params), returning "
             "{'dev': one finite float per dev row, 'heldout': one per heldout row}. "
             "numpy is available. It runs sandboxed and the held-out labels are redacted "
             "out of the data it can read, so do not attempt to use them. If you need "
             "another library, declare it on its own line as `# requires: <name>` (at "
             "most 4). Permitted: numpy, scipy, pandas, scikit-learn, lightgbm, xgboost, "
             "torch, implicit -- anything else is refused."),
    "params": "a free-form object handed to your fit_predict as `params`. May be {}.",
}

SYMPTOMS = ("loss_metric_mismatch", "no_sequence_features", "single_objective",
            "watch_time_unmodelled", "no_time_features", "exposure_bias_unchecked",
            "capacity_bound", "feature_bound", "seed_variance")

#: name -> {param: (kind, *bounds)}
SPEC: dict[str, dict] = {
    # direction 1 -- the organisers' top pick
    "SetLoss":            {"loss": ("enum", ("pointwise", "pairwise", "listwise"))},
    "SetNegPerPos":       {"neg_per_pos": ("int", 1, 16)},
    "SetListwiseTemp":    {"listwise_temp": ("float", 0.1, 10.0)},
    # direction 3 -- multi-objective
    "SetAuxWeights":      {"aux_weights": ("weights", AUX_LABELS)},
    # feature engineering (crosses and derived fields are fine; raw piling-on is not)
    "SetFields":          {"fields": ("fields", KNOWN_FIELDS)},
    # optimisation
    "ChangeLr":           {"lr": ("float", 1e-5, 0.05)},
    "ChangeL2":           {"l2": ("float", 1e-8, 1e-2)},
    "ChangeBatch":        {"batch": ("int", 512, 65536)},
    "ChangeEpochs":       {"epochs": ("int", 1, 80)},
    "ChangePatience":     {"patience": ("int", 1, 10)},
    "ChangeK":            {"k": ("int", 4, 64)},
    # model class -- deprioritised by the organisers, still reachable
    "SwapModel":          {"model": ("enum", ("fm", "popularity"))},
    # VARIANCE, not bias. Absent from the organisers' list because that list is
    # organised around features, losses and models -- but the published seed std (0.0008)
    # is comparable to the entire winning margin, so noise is worth attacking directly.
    # Costs n_seeds x the fit time; the agent is told that so the trade is explicit.
    "SetEnsemble":        {"n_seeds": ("int", 1, 8),
                           "combine": ("enum", ("rank", "score"))},
    # STAGE 2 of the loop: ask a question about the data instead of guessing.
    # An Inspect costs no fit and changes no config -- it returns a finding that is put
    # into the next prompt. Each diagnostic may be run once per run, because its answer
    # does not change, so repeating one is a duplicate in the same sense a repeated
    # config is.
    "Inspect":            {"diagnostic": ("diagnostic",)},
    # direction 7 -- unbiased validation
    "SetUseRandomLog":    {"use_random_log": ("enum", (True, False))},
    # THE ESCAPE HATCH. Everything above is a parameter change inside a model we wrote.
    # This is the agent writing the model. It clears the brief's "writing the code for
    # each stage is part of the agent's job" bar, and it is only safe because the patch
    # runs behind the kernel sandbox with the held-out labels redacted out of the bytes.
    "ProposePatch":       {"code": ("code",), "params": ("params",)},
}
ACTION_TYPES = tuple(SPEC)


class ActionError(ValueError):
    """Rejection. Cheap, recorded, and always explains itself."""


def _num(v, lo, hi, name, integer=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ActionError(f"{name} must be a number, got {type(v).__name__}")
    if integer and int(v) != v:
        raise ActionError(f"{name} must be an integer, got {v}")
    if not (lo <= v <= hi):
        raise ActionError(f"{name}={v} is outside [{lo}, {hi}]")
    return int(v) if integer else float(v)


def validate_params(action_type: str, params: Any) -> dict:
    if action_type not in SPEC:
        raise ActionError(f"unknown action_type {action_type!r}; "
                          f"choose from {list(SPEC)}")
    if not isinstance(params, dict):
        raise ActionError("params must be an object")
    spec = SPEC[action_type]
    # The JSON schema we hand the model lists EVERY param across EVERY action as an
    # optional property (a single flat object -- an unconstrained one gave 3/3 rejections,
    # see the module history). That design invites the model to attach params belonging to
    # other actions, and refusing it for doing what the schema permits cost a real
    # discovery: a SetEnsemble proposal was rejected for carrying `use_random_log`.
    #
    # So params known to ANOTHER action are ignored, and recorded as ignored rather than
    # dropped silently. Params in no action at all are still a refusal, because that is a
    # genuine misunderstanding rather than schema noise.
    known_anywhere = {q for sp in SPEC.values() for q in sp}
    unknown = set(params) - known_anywhere
    if unknown:
        raise ActionError(f"{action_type}: {sorted(unknown)} is not a parameter of any "
                          f"action. {action_type} takes {list(spec)}.")
    ignored = sorted(set(params) - set(spec))
    missing = set(spec) - set(params)
    if missing:
        raise ActionError(f"{action_type} requires {sorted(missing)}; got "
                          f"{sorted(set(params) & set(spec))}")

    out = {}
    for name, s in spec.items():
        v = params[name]
        kind = s[0]
        if kind == "enum":
            if v not in s[1]:
                raise ActionError(f"{name}={v!r} not in {list(s[1])}")
            out[name] = v
        elif kind == "int":
            out[name] = _num(v, s[1], s[2], name, integer=True)
        elif kind == "float":
            out[name] = _num(v, s[1], s[2], name)
        elif kind == "weights":
            if not isinstance(v, dict) or not v:
                raise ActionError(f"{name} must be a non-empty object of label->weight")
            bad = set(v) - set(s[1])
            if bad:
                raise ActionError(f"{name} has unknown labels {sorted(bad)}; "
                                  f"choose from {list(s[1])}")
            out[name] = {k: _num(w, 0.0, 5.0, f"{name}[{k}]") for k, w in v.items()}
        elif kind == "diagnostic":
            from agent.inspect import DIAGNOSTICS
            if v not in DIAGNOSTICS:
                raise ActionError(f"{name}={v!r} is not a diagnostic; choose from "
                                  f"{sorted(DIAGNOSTICS)}")
            out[name] = v
        elif kind == "code":
            from agent.patch import validate as _vp, PatchError as _PE
            try:
                _vp(v)
            except _PE as e:
                raise ActionError(f"patch rejected: {e}") from None
            out[name] = v
        elif kind == "params":
            if not isinstance(v, dict):
                raise ActionError(f"{name} must be an object (may be empty)")
            out[name] = v
        elif kind == "fields":
            if not isinstance(v, (list, tuple)) or not v:
                raise ActionError(f"{name} must be a non-empty list of field names")
            bad = [f for f in v if f not in s[1]]
            if bad:
                # A field that was offered, measured and removed gets its NUMBER back, not
                # just "unknown". The refusal is the only feedback the agent receives, so it
                # is the only place the measurement can teach anything.
                measured = {f: REMOVED_FIELDS[f] for f in bad if f in REMOVED_FIELDS}
                if measured:
                    det = "; ".join(f"{f}: {why}" for f, why in measured.items())
                    raise ActionError(
                        f"{name}: these fields were offered, MEASURED and removed -- {det}. "
                        f"All are inside the 0.0008 seed sd, and all 14 fields together "
                        f"(.5940-equivalent, +0.00093) scored WORSE than the baseline 5 plus "
                        f"hourmin_bucket (.5950-equivalent, +0.00113). A field can only "
                        f"reorder a within-user list if it VARIES within that user, and must "
                        f"also say something video_id does not. Choose from {list(s[1])}.")
                raise ActionError(f"{name} has unknown fields {bad}; "
                                  f"choose from {list(s[1])}")
            out[name] = tuple(v)
    if ignored:
        out["_ignored_params"] = ignored
    return out


def check_dead_ends(action_type: str, params: dict, cfg) -> None:
    """Refuse the organisers' published dead ends, citing their numbers.

    Raised as ActionError so it lands in the same 'rejected_schema' path the driver
    already records -- the refusal and its reason end up in the ledger.
    """
    if action_type == "SetEnsemble":
        return                                   # nothing measured rules it out
    if action_type == "SetFields":
        fields = tuple(params["fields"])
        # `hourmin_bucket` is a DERIVED CONTEXT field, not raw piling-on, and the
        # distinction is measured rather than stylistic: it varies within 79.1% of users and
        # is worth +0.001192 paired over 5 seeds (t=14.04, 5/5), which is more than any
        # hyperparameter found so far and takes the combined valid delta past epsilon=0.002.
        # Without this exemption the dead-end guard refused the single best field available --
        # the guard was written when "extra field" could only mean the side-table columns.
        raw_extra = [f for f in fields
                     if f not in BASELINE_FIELDS and f not in ENCOURAGED_EXTRA]
        if len(fields) > len(BASELINE_FIELDS) and raw_extra:
            raise ActionError(
                f"REFUSED (published dead end -- more static fields): {DEAD_ENDS['more_static_fields']} "
                f"You proposed adding {raw_extra}. Derived or crossed features are still "
                f"open; piling on more raw categorical fields is not.")
        if fields and all(f in USER_SIDE_FIELDS for f in fields):
            raise ActionError(
                f"REFUSED (provably inert -- user-side only): {DEAD_ENDS['user_side_only']} "
                f"You proposed fields={list(fields)}, all of which are user-side.")
        if "video_id" not in fields:
            raise ActionError(
                "REFUSED: dropping video_id leaves nothing item-side to cross against, "
                "so within-user ordering becomes constant. " + DEAD_ENDS["user_side_only"])
    if action_type == "ChangeK":
        k = params["k"]
        if k in (8, 16, 32):
            raise ActionError(
                f"REFUSED (published dead end -- capacity): {DEAD_ENDS['more_capacity']} "
                f"k={k} is one of the three they measured.")


def validate_rationale(r: Any, epsilon: float, require_magnitude: bool = True) -> dict:
    """Every action carries a rationale. Innovation is scored on the WHY.

    `require_magnitude` is False for actions that do not predict a score change (an
    Inspect asks a question; it does not claim a delta).
    """
    if not isinstance(r, dict):
        raise ActionError("rationale must be an object")
    needed = (("mechanism", "expected_direction", "expected_magnitude")
              if require_magnitude else ("mechanism",))
    for f in needed:
        if f not in r:
            raise ActionError(f"rationale missing required field {f!r}")
    if not isinstance(r["mechanism"], str) or len(r["mechanism"].strip()) < 20:
        raise ActionError("rationale.mechanism must be a real sentence (>=20 chars)")
    if not require_magnitude:
        bad0 = set(r.get("symptoms_addressed", [])) - set(SYMPTOMS)
        if bad0:
            raise ActionError(f"unknown symptoms {sorted(bad0)}; choose from {list(SYMPTOMS)}")
        return {"mechanism": r["mechanism"].strip(),
                "expected_direction": r.get("expected_direction", "increase"),
                "expected_magnitude": None,
                "cites": list(r.get("cites", [])),
                "symptoms_addressed": list(r.get("symptoms_addressed", []))}
    if r["expected_direction"] not in ("increase", "decrease"):
        raise ActionError("rationale.expected_direction must be increase|decrease")
    try:
        mag = abs(float(r["expected_magnitude"]))
    except (TypeError, ValueError):
        raise ActionError("expected_magnitude must be a number") from None
    if mag < epsilon:
        raise ActionError(
            f"expected_magnitude {mag:.5f} is below the organisers' own convergence "
            f"threshold epsilon={epsilon}. A change too small to clear epsilon cannot "
            f"move the incumbent even if it works -- propose something bigger or say "
            f"why this is worth an iteration.")
    bad = set(r.get("symptoms_addressed", [])) - set(SYMPTOMS)
    if bad:
        raise ActionError(f"unknown symptoms {sorted(bad)}; choose from {list(SYMPTOMS)}")
    return {"mechanism": r["mechanism"].strip(),
            "expected_direction": r["expected_direction"],
            "expected_magnitude": mag,
            "cites": list(r.get("cites", [])),
            "symptoms_addressed": list(r.get("symptoms_addressed", []))}


def validate(envelope: Any, epsilon: float, cfg=None) -> dict:
    if not isinstance(envelope, dict):
        raise ActionError("proposal must be an object")
    extra = set(envelope) - {"action_type", "params", "rationale", "parent_node_id"}
    if extra:
        raise ActionError(f"unexpected top-level keys {sorted(extra)}")
    at = envelope.get("action_type")
    params = validate_params(at, envelope.get("params"))
    check_dead_ends(at, params, cfg)
    # An Inspect makes no prediction about the score, so gating it on
    # expected_magnitude >= epsilon would refuse every well-formed diagnostic. It still
    # needs a stated mechanism -- "why is this worth asking" is the point of it.
    rat = validate_rationale(envelope.get("rationale"), epsilon,
                             require_magnitude=(at != "Inspect"))
    return {"action_type": at, "params": params, "rationale": rat,
            "parent_node_id": envelope.get("parent_node_id")}


def apply(cfg, action: dict):
    """Return a NEW Config with the action applied. Never mutates."""
    from dataclasses import replace
    p = action["params"]
    if action["action_type"] == "Inspect":
        return cfg                               # a query, not a change
    if action["action_type"] == "SetEnsemble":
        return replace(cfg, ensemble_seeds=p["n_seeds"], ensemble_combine=p["combine"])
    if action["action_type"] == "ProposePatch":
        # a patch replaces the model wholesale; the typed fields stop applying
        return replace(cfg, model="patch", patch_code=p["code"],
                       patch_params=dict(p.get("params") or {}))
    kw = {}
    for k, v in p.items():
        if k == "_ignored_params":
            continue
        kw[k] = tuple(v) if k == "fields" else v
    return replace(cfg, **kw)


def _params_schema() -> dict:
    """Every param across every action as a TYPED OPTIONAL property.

    Flattened rather than a free-form object, because a constrained decode reads the
    SCHEMA, not the prose: measured on the previous task, an unconstrained `params`
    produced 3/3 rejected proposals (STATE.md bug 9).
    """
    props = {
        "loss": {"type": "string", "enum": list(SPEC["SetLoss"]["loss"][1])},
        "neg_per_pos": {"type": "integer", "minimum": 1, "maximum": 16},
        "listwise_temp": {"type": "number", "minimum": 0.1, "maximum": 10.0},
        "aux_weights": {"type": "object", "additionalProperties": {"type": "number"}},
        "fields": {"type": "array", "items": {"type": "string", "enum": list(KNOWN_FIELDS)}},
        "lr": {"type": "number", "minimum": 1e-5, "maximum": 0.05},
        "l2": {"type": "number", "minimum": 1e-8, "maximum": 1e-2},
        "batch": {"type": "integer", "minimum": 512, "maximum": 65536},
        "epochs": {"type": "integer", "minimum": 1, "maximum": 80},
        "patience": {"type": "integer", "minimum": 1, "maximum": 10},
        "k": {"type": "integer", "minimum": 4, "maximum": 64},
        "model": {"type": "string", "enum": ["fm", "popularity"]},
        "use_random_log": {"type": "boolean"},
        "code": {"type": "string", "minLength": 60},
        "n_seeds": {"type": "integer", "minimum": 1, "maximum": 8},
        "diagnostic": {"type": "string"},
        "combine": {"type": "string", "enum": ["rank", "score"]},
    }
    props["params"] = {"type": "object", "additionalProperties": True}
    return {"type": "object", "properties": props, "additionalProperties": False}


def json_schema() -> dict:
    return {
        "type": "object",
        "required": ["action_type", "params", "rationale"],
        "additionalProperties": False,
        "properties": {
            "action_type": {"type": "string", "enum": list(ACTION_TYPES)},
            "params": _params_schema(),
            "rationale": {
                "type": "object",
                "required": ["mechanism"],
                "additionalProperties": False,
                "properties": {
                    "mechanism": {"type": "string", "minLength": 20},
                    "expected_direction": {"type": "string",
                                           "enum": ["increase", "decrease"]},
                    "expected_magnitude": {"type": "number"},
                    "cites": {"type": "array", "items": {"type": "string"}},
                    "symptoms_addressed": {"type": "array",
                                           "items": {"type": "string",
                                                     "enum": list(SYMPTOMS)}},
                },
            },
            "parent_node_id": {"type": ["string", "null"]},
        },
    }
