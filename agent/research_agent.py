"""Planner, coder and reflector: the three LLM stages of one research iteration.

WHY THREE STAGES AND NOT ONE
----------------------------
The typed driver asked for one thing per iteration -- an action -- and got, measured over
34 runs, mostly hyperparameter nudges. Splitting the work makes each call answerable:

  planner    given the history, what is worth trying and WHY? Commits to a mechanism and
             a predicted magnitude BEFORE any code exists, so the prediction can be
             pre-registered below the score that tests it and cannot be retrofitted.
  coder      given that hypothesis and the parent's source, write the code. Returns whole
             files; the harness computes the diff. It is never asked for a diff, because a
             diff the model authors can describe a change it did not make, and the run log
             is what judges read.
  reflector  given the signed result, was the mechanism supported? This is the stage that
             turns a number into something the next planner call can use.

A repair is a further `code` call inside the SAME iteration -- one hypothesis, possibly
several attempts. Repairs are counted and their diffs recorded; they are not hidden,
because "how a failure was handled" is what Robustness is scored on.

WHAT THE STAGES ARE NOT TOLD
----------------------------
The held-out split, in any form. And no gain this project has already measured by hand:
the seed workspace states mechanisms (how to derive a column, where a loss lives) but
carries no measured values, so a result the agent reports is one it established.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "research_workspace"

STAGES = ("features", "model", "training", "config", "evaluation")

#: The research lanes a tournament must span. The point is BREADTH OF CONSIDERATION: two
#: 50-iteration runs and five compliant ones spent 115 of 126 proposals in
#: feature_preprocessing, and the reason was policy, not judgement -- one LLM call was asked
#: for one idea, so whatever came to mind first became the experiment. A tournament forces
#: alternatives from every lane to exist before one is chosen.
#:
#: Requiring an architecture candidate to be CONSIDERED is not the same as requiring one to
#: WIN. The judge may reject it, and must say why.
LANES = ("feature_preprocessing", "objective_loss", "sampling_training",
         "optimization_regularization", "model_architecture", "ensemble_calibration")

#: Lanes a tournament must always include at least one candidate from.
REQUIRED_LANES = ("model_architecture", "sampling_training")

#: The roles a run may route to different models. `plan` and `judge` carry the reasoning;
#: `code` writes it; `reflect` summarises. A run may point them all at one model.
ROLES = ("plan", "judge", "code", "reflect")


def _hypothesis_properties() -> dict:
    """The fields every tournament candidate must carry.

    `information_value` and `novelty_vs_history` exist so the judge can prefer an experiment
    that RESOLVES something over one that merely might win: a decisive negative result on an
    architecture question is worth more than another feature cross that nudges the metric.
    """
    return {
        "id": {"type": "string", "minLength": 1,
               "description": "short stable id, e.g. 'A1'"},
        "lane": {"type": "string", "enum": list(LANES)},
        "hypothesis": {"type": "string", "minLength": 10},
        "mechanism": {"type": "string", "minLength": 10,
                      "description": "WHY this should move the metric. Causal, not a "
                                     "restatement of the change."},
        "evidence": {"type": "string",
                     "description": "what in the history, telemetry or beliefs motivates "
                                    "it. Cite iteration numbers."},
        "expected_direction": {"type": "string",
                               "enum": ["increase", "decrease", "unclear"]},
        "expected_magnitude": {"type": "number"},
        "information_value": {"type": "string",
                              "description": "what is learned even if it LOSES"},
        "novelty_vs_history": {"type": "string",
                               "description": "how this differs from what has been tried; "
                                              "if it revisits a rejected mechanism, what "
                                              "has materially changed"},
        "files_to_change": {"type": "array", "items": {"type": "string"}},
        "runtime_estimate": {"type": "string",
                             "description": "rough fit time; this runs on a laptop CPU"},
        "memory_estimate": {"type": "string"},
        "implementation_risk": {"type": "string"},
        "leakage_risk": {"type": "string",
                         "description": "any way this could see dev or held-out labels it "
                                        "should not; 'none' is an acceptable answer"},
        "success_criteria": {"type": "string"},
        "failure_interpretation": {"type": "string",
                                   "description": "what a negative result would mean"},
        "parent": {"type": "string",
                   "description": "'champion' or a 12-char tree prefix"},
    }


#: The tournament policy is meaningless below one candidate per lane, so six is a floor, not
#: a default. `--tournament-size` is clamped up to it and the loop refuses to proceed on an
#: under-sized set.
MIN_TOURNAMENT_SIZE = len(LANES)


def tournament_schema(size: int = 6) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array", "minItems": MIN_TOURNAMENT_SIZE, "maxItems": 12,
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["id", "lane", "hypothesis", "mechanism",
                                       "expected_direction", "expected_magnitude",
                                       "information_value", "files_to_change", "parent"],
                          "properties": _hypothesis_properties()},
            },
            "lane_coverage_note": {"type": "string"},
        },
    }


def judge_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["selected_id", "ranking", "why_selected",
                     "architecture_verdicts", "exploit_or_explore"],
        "properties": {
            "selected_id": {"type": "string", "minLength": 1},
            "ranking": {
                "type": "array", "minItems": 1, "maxItems": 12,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "rank", "score_evidence", "score_expected_impact",
                                 "score_novelty", "score_information_value",
                                 "score_cpu_feasibility", "score_implementation_risk"],
                    "properties": {
                        "id": {"type": "string"},
                        "rank": {"type": "integer", "minimum": 1},
                        "score_evidence": {"type": "number"},
                        "score_expected_impact": {"type": "number"},
                        "score_novelty": {"type": "number"},
                        "score_information_value": {"type": "number"},
                        "score_cpu_feasibility": {"type": "number"},
                        "score_implementation_risk": {"type": "number"},
                        "note": {"type": "string"},
                    },
                },
            },
            "why_selected": {"type": "string", "minLength": 10},
            "architecture_verdicts": {
                "type": "array", "minItems": 1, "maxItems": 12,
                "description": "ONE ENTRY PER model_architecture candidate, by exact id. "
                               "Rejecting architecture work is a legitimate judgement; "
                               "leaving a candidate unaddressed is not, and a missing or "
                               "unknown id is rejected by the harness before anything runs.",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["candidate_id", "verdict", "reason"],
                    "properties": {
                        "candidate_id": {"type": "string", "minLength": 1},
                        "verdict": {"type": "string",
                                    "enum": ["selected", "rejected"]},
                        "reason": {"type": "string", "minLength": 10},
                    },
                },
            },
            "exploit_or_explore": {"type": "string",
                                   "enum": ["exploit", "explore"]},
            "convergence_note": {"type": "string"},
        },
    }


def _mutable() -> list[str]:
    from research_workspace.contract import MUTABLE_FILES
    return list(MUTABLE_FILES)


# ------------------------------------------------------------------------ schemas
def plan_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["hypothesis", "evidence", "stage", "mechanism",
                     "expected_direction", "expected_magnitude", "parent",
                     "files_to_change", "risk", "success_criteria"],
        "properties": {
            "hypothesis": {"type": "string", "minLength": 10, "maxLength": 600,
                           "description": "One concrete, falsifiable change to try."},
            "evidence": {"type": "string", "maxLength": 1200,
                         "description": "What in the history, diagnostics or telemetry "
                                        "motivates this. Cite iteration numbers."},
            "stage": {"type": "string", "enum": list(STAGES)},
            "mechanism": {"type": "string", "maxLength": 1200,
                          "description": "WHY this should move the metric. The causal "
                                         "story, not a restatement of the change."},
            "expected_direction": {"type": "string",
                                   "enum": ["increase", "decrease", "unclear"]},
            "expected_magnitude": {
                "type": "number",
                "description": "Predicted change in dev primary. Pre-registered before "
                               "the code exists. A value below epsilon is fine -- small "
                               "gains accumulate; epsilon is the convergence rule, not "
                               "an admission gate."},
            "parent": {"type": "string", "maxLength": 64,
                       "description": "'champion', or the 12-char tree prefix of an "
                                      "earlier candidate to fork instead."},
            "files_to_change": {"type": "array", "minItems": 1, "maxItems": 8,
                                "items": {"type": "string"}},
            "risk": {"type": "string", "maxLength": 600},
            "fallback": {"type": "string", "maxLength": 600},
            "success_criteria": {"type": "string", "maxLength": 600},
        },
    }


def code_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["reasoning", "files"],
        "properties": {
            "reasoning": {"type": "string", "maxLength": 2000},
            "files": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "description": "Whole replacement files. Omit a file to inherit the "
                               "parent's copy unchanged. The harness computes the diff.",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string", "maxLength": 64},
                        "content": {"type": "string", "maxLength": 60000},
                    },
                },
            },
            "dependencies": {
                "type": "array", "maxItems": 4, "items": {"type": "string"},
                "description": "Extra packages needed. Must be on the harness allowlist; "
                               "anything else is refused before the fit runs."},
        },
    }


def reflect_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["what_happened", "mechanism_supported", "why", "recommendation",
                     "next_question"],
        "properties": {
            "what_happened": {"type": "string", "maxLength": 1200},
            "mechanism_supported": {"type": "boolean"},
            "why": {"type": "string", "maxLength": 1500,
                    "description": "Why it improved or failed. If the prediction was "
                                   "wrong, say what the wrong model of the problem was."},
            "recommendation": {"type": "string", "enum": ["promote", "discard"]},
            "prediction_comment": {"type": "string", "maxLength": 600},
            "remaining_uncertainty": {"type": "string", "maxLength": 800},
            "next_question": {"type": "string", "maxLength": 600,
                              "description": "The most informative next experiment, and "
                                             "what it would settle."},
            "belief_updates": {
                "type": "array", "maxItems": 6,
                "description": "Structured beliefs to carry forward. These are what the next "
                               "tournament reasons over, so state the MECHANISM, not this "
                               "candidate. A belief that a mechanism is unhelpful is as "
                               "valuable as one that it works. Note that these are YOUR "
                               "interpretation of a signed number, not measurements: say what "
                               "you believe and how confident you are, and do not overstate.",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["mechanism", "lane", "confidence", "belief"],
                    "properties": {
                        "mechanism": {"type": "string", "minLength": 4,
                                      "description": "the mechanism or research question, "
                                                     "stable across iterations so updates "
                                                     "accumulate on one record"},
                        "lane": {"type": "string", "enum": list(LANES)},
                        "belief": {"type": "string", "minLength": 4,
                                   "description": "what is now believed, e.g. 'harmful from "
                                                  "a pointwise baseline'"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "supports": {"type": "boolean",
                                     "description": "did THIS iteration support the belief"},
                        "affected_files": {"type": "array", "items": {"type": "string"}},
                        "effect_low": {"type": "number"},
                        "effect_high": {"type": "number"},
                        "unresolved": {"type": "string"},
                        "retry_only_if": {"type": "string",
                                          "description": "what would have to change for this "
                                                         "to be worth revisiting"},
                    },
                },
            },
        },
    }


# ------------------------------------------------------------------------ prompts
def _contract_text() -> str:
    """The contract's own docstring, read from the file. Never a paraphrase: a paraphrase
    drifts from the code that enforces it, and then the agent is coding against fiction."""
    import ast
    doc = ast.get_docstring(ast.parse((WORKSPACE / "contract.py").read_text()))
    return (doc or "").strip()


def _rules_block(task: dict) -> str:
    return f"""You are the research engine of an autonomous ML research agent.

THE TASK
{json.dumps(task, indent=2, sort_keys=True, default=str)}

THE CONTRACT YOUR CODE MUST SATISFY
{_contract_text()}

RULES THAT ARE ENFORCED, NOT ADVISORY
  * You may edit only these files: {_mutable()}. contract.py is immutable and a tree
    containing it is refused.
  * The held-out split's labels are not in the process. Do not try to reach them; the
    kernel denies it and the attempt is recorded.
  * Every candidate runs sandboxed with no network. Extra packages must be declared and
    must be on the allowlist.
  * The number that decides anything is produced by a separate signing scorer. Your own
    metric calls are for early stopping and for the train/dev curve.
  * A tree identical to one already evaluated is refused for free -- vary something real.
  * Read `budget` before deciding. It is LIVE: it tells you how many evaluations remain, and
    when a convergence check falls. If `convergence.mode` is "official" the run STOPS at the
    next check unless the champion's CUMULATIVE gain across the window exceeds epsilon -- so
    with few iterations until that check, an idea whose expected magnitude cannot contribute
    to clearing it is a wasted evaluation, and two smaller improvements that add up are
    enough. If convergence is disabled, spend the budget: finishing early earns nothing."""


def plan_prompt(state: dict, parent_source: dict, parent_telemetry: dict | None) -> tuple:
    sysb = [_rules_block(state.get("task", {}))]
    tel = _telemetry_view(parent_telemetry)
    user = f"""RESEARCH STATE
{json.dumps(state, indent=2, sort_keys=True, default=str)[:12000]}

THE CURRENT CHAMPION'S SOURCE
{_render_source(parent_source)}

THE CHAMPION'S PER-EPOCH TELEMETRY (self-reported by the candidate, not signed)
{tel}

Propose ONE hypothesis for the next experiment.

Read the telemetry before deciding. The train and dev curves are there so that
overfitting, underfitting and a collapsed learning rate look different from each other
rather than all looking like "the number did not move".

Do not repeat an idea in the failure registry without saying what you are changing about
it. Prefer a change whose mechanism you can state; a hyperparameter nudge with no
mechanism is the least informative experiment available and this project has already run
many of them.

`reflections` holds your OWN conclusions from previous iterations, including the one behind
the current champion. Either build on the `next_question` you already identified, or state
why you are departing from it -- but do not silently ignore your own analysis, and do not
re-derive a conclusion you already reached."""
    return sysb, user


def tournament_prompt(state: dict, parent_source: dict, parent_telemetry: dict | None,
                      size: int, deficiency: list | None = None) -> tuple:
    """Ask for SEVERAL competing hypotheses spanning every research lane.

    The instruction is breadth, not a predetermined answer. Nothing here says a particular
    lane should win; it says every lane must be represented so the judge has a real choice.
    """
    sysb = [_rules_block(state.get("task", {}))]
    tel = _telemetry_view(parent_telemetry)
    lanes = "\n".join(f"    {i+1}. {l}" for i, l in enumerate(LANES))
    user = f"""RESEARCH STATE
{json.dumps(state, indent=2, sort_keys=True, default=str)[:14000]}

THE CURRENT CHAMPION'S SOURCE
{_render_source(parent_source)}

THE CHAMPION'S PER-EPOCH TELEMETRY (self-reported by the candidate, not signed)
{tel}

Propose **{size} COMPETING hypotheses**, not one. They will be ranked by a judge and exactly
one will be evaluated; the rest cost nothing and are recorded as considered.

Span these research lanes:
{lanes}

Hard requirements on the SET, checked by the harness before anything is coded or fitted:
  * EVERY one of the six lanes must have at least one candidate;
  * `model_architecture` (rewriting `model.py`) and `sampling_training` (rewriting
    `train.py`) must each be present -- these two are named explicitly because they are the
    ones this project has historically neglected;
  * a second candidate in a lane is only allowed once all six lanes are covered;
  * no two candidates that are the same change with a different constant.

If the set does not satisfy this, you will be asked once more with the specific deficiency,
and if it still does not, the iteration is abandoned without spending an evaluation.

This is a laptop CPU with numpy. A candidate must be fittable in minutes, so prefer a
structured change over raw capacity. `information_value` matters: an experiment that settles
an open question is worth more than one that might nudge the metric, because a decisive
negative result removes a whole direction from future consideration.

Read `beliefs` before proposing. If a candidate revisits a mechanism already recorded as
unhelpful, `novelty_vs_history` must say what has materially changed -- otherwise do not
propose it."""
    if deficiency:
        user += ("\n\nYOUR PREVIOUS SET WAS REFUSED. Fix exactly these problems and return a "
                 "complete set:\n  * " + "\n  * ".join(str(d) for d in deficiency)
                 + "\n\nKeep the candidates that were fine; add or re-lane only what is "
                   "needed. This is the last attempt before the iteration is abandoned "
                   "without spending an evaluation.")
    return sysb, user


def judge_prompt(candidates: list, state: dict, parent_source: dict,
                 parent_telemetry: dict | None, problems: list | None = None) -> tuple:
    """Rank the tournament and pick one. The architecture verdict is mandatory."""
    sysb = [_rules_block(state.get("task", {}))]
    user = f"""THE TOURNAMENT CANDIDATES
{json.dumps(candidates, indent=2, sort_keys=True, default=str)[:16000]}

RESEARCH STATE (history, beliefs, failures, calibration, live budget)
{json.dumps(state, indent=2, sort_keys=True, default=str)[:12000]}

THE CURRENT CHAMPION'S SOURCE
{_render_source(parent_source)}

CHAMPION TELEMETRY (self-reported)
{_telemetry_view(parent_telemetry)}

Rank every candidate and select exactly one to evaluate. Score each on evidence, expected
impact, novelty, information value, CPU feasibility and implementation risk.

`architecture_verdicts` is REQUIRED: one entry for EVERY `model_architecture` candidate,
keyed by its exact id, each with `verdict` of "selected" or "rejected" and a reason. Rejecting
architecture work is a legitimate judgement -- this project's champion keeps a fixed
Factorization Machine and that was the right call so far. Passing over a candidate in silence
is not, and a missing or unrecognised id is refused before anything runs.

`ranking` must list EVERY candidate exactly once, with unique consecutive ranks starting at 1,
and no id that is not in the tournament. `selected_id` must be one of them.

Use `budget`. If a convergence check is close, prefer a candidate whose expected magnitude
could plausibly contribute to clearing it, and remember the window measures CUMULATIVE gain
so two smaller improvements can be enough. If there is plenty of budget, an informative
experiment that may lose is the better spend.

Say whether you are exploiting established evidence or exploring uncertainty."""
    if problems:
        user += ("\n\nYOUR PREVIOUS RESPONSE WAS REFUSED. Fix exactly these problems:\n  * "
                 + "\n  * ".join(str(x) for x in problems)
                 + "\n\nThis is the last attempt before the iteration is abandoned without "
                   "spending an evaluation.")
    return sysb, user


def _telemetry_view(parent_telemetry: dict | None) -> str:
    if not parent_telemetry:
        return "none"
    keep = ("epochs_run", "dev_primary_best", "train_primary", "dev_primary", "gap",
            "train_loss", "epoch_seconds", "n_features")
    return json.dumps({k: v for k, v in parent_telemetry.items() if k in keep},
                      indent=2, default=str)[:4000]


def code_prompt(plan: dict, parent_source: dict, error: dict | None,
                attempt: int, state: dict) -> tuple:
    sysb = [_rules_block(state.get("task", {}))]
    head = f"""THE HYPOTHESIS YOU ARE IMPLEMENTING
{json.dumps(plan, indent=2, sort_keys=True, default=str)}

THE PARENT SOURCE YOU ARE EDITING
{_render_source(parent_source)}"""
    if error is None:
        tail = """Return whole replacement files for everything you change. Omit a file to
inherit the parent's copy. Change only what the hypothesis needs -- a candidate that
changes four things at once cannot attribute its own result."""
    else:
        tail = f"""YOUR PREVIOUS ATTEMPT FAILED. This is repair attempt {attempt}.

{json.dumps(error, indent=2, sort_keys=True, default=str)[:5000]}

Fix the cause. If the error shows the approach cannot work as written, say so in
`reasoning` and return the smallest change that makes the hypothesis testable -- do not
quietly abandon the hypothesis and return something unrelated."""
    return sysb, head + "\n\n" + tail


def reflect_prompt(plan: dict, result: dict, state: dict) -> tuple:
    sysb = [_rules_block(state.get("task", {}))]
    user = f"""THE HYPOTHESIS
{json.dumps(plan, indent=2, sort_keys=True, default=str)}

WHAT HAPPENED
{json.dumps(result, indent=2, sort_keys=True, default=str)[:8000]}

RESEARCH STATE
{json.dumps(state, indent=2, sort_keys=True, default=str)[:6000]}

Was the mechanism supported? Note that `signed` is the scorer's number and decides
promotion; `train_primary` / `dev_primary` / `gap` are computed by the harness from your
returned score vectors; anything under `telemetry` is self-reported by your own code.

Your `recommendation` is advice. The driver promotes on the signed dev primary alone."""
    return sysb, user


def _render_source(files: dict) -> str:
    if not files:
        return "(no parent -- this is the first candidate)"
    out = []
    for path in sorted(files):
        out.append(f"----- {path} -----\n{files[path]}")
    return "\n".join(out)


# --------------------------------------------------------------------- the client
#: Schema keywords VERIFIED to pass Gemini's structured-output validator -- precisely the
#: set agent/official_actions.json_schema() already uses in real runs. Our schemas also
#: carry maxLength / minItems / maxItems, which are NOT in that proven set, so they are
#: stripped for that backend rather than gambling a whole iteration on an untested keyword.
#: Nothing is lost: every one of those bounds is enforced again on our side (the driver
#: truncates the recorded fields, and agent/candidate.validate_tree enforces file sizes).
GEMINI_SAFE_KEYWORDS = {"type", "enum", "items", "properties", "required",
                        "additionalProperties", "minLength", "minimum", "maximum",
                        "description"}


def strip_schema(node, allowed: set):
    """Recursively drop schema keywords a backend has not been shown to accept."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in ("properties",):
                out[k] = {pk: strip_schema(pv, allowed) for pk, pv in v.items()}
            elif k in allowed:
                out[k] = strip_schema(v, allowed) if isinstance(v, (dict, list)) else v
        return out
    if isinstance(node, list):
        return [strip_schema(v, allowed) for v in node]
    return node


class ResearchAgent:
    """Planner, tournament, judge, coder and reflector over one or more LLM backends.

    ROLE ROUTING
    ------------
    Reasoning and implementation are different jobs and do not need the same model. Each role
    resolves its model as: role-specific flag -> `--model` -> the backend's own default. A run
    may therefore point a stronger model at `plan` and `judge` (where the research decisions
    are made) and a cheaper one at `reflect`, or point everything at one model and behave
    exactly as before. No vendor's identifier is hard-coded here; the model strings come from
    the CLI and the backend supplies the default.

    COST ACCOUNTING
    --------------
    Every call records role, backend, model, input/output/thinking tokens, latency, retries
    and whether it was refused or truncated. `usage_by_role()` aggregates it, because "which
    role spent the budget" is the question a compute report has to answer and a single total
    cannot.
    """
    name = "research"

    def __init__(self, backend: str = "gemini", model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 effort: str = "high", thinking_level: str | None = None,
                 timeout: float = 600.0, max_retries: int = 2,
                 max_tokens: int = 32000,
                 planner_model: str | None = None, judge_model: str | None = None,
                 coder_model: str | None = None, reflector_model: str | None = None):
        self.backend, self.effort, self.max_tokens = backend, effort, max_tokens
        self.thinking_level = thinking_level
        self.timeout, self.max_retries, self.base_url = timeout, max_retries, base_url
        self.usage: list[dict] = []
        self._key = (api_key or os.environ.get("TRACK2_LLM_API_KEY")
                     or os.environ.get("ANTHROPIC_API_KEY") or "")
        #: role -> requested model, or None to fall back to `model`, then the backend default
        self.role_model = {
            "plan": planner_model or model,
            "judge": judge_model or model,
            "code": coder_model or model,
            "reflect": reflector_model or model,
        }
        self.model = model                       # the fallback, kept for reporting
        self._clients: dict = {}                 # (backend, model) -> client/backend object
        # Build the fallback eagerly so a missing key or SDK fails at startup, not at
        # iteration 1 of an unattended run.
        self._client_for("plan")
        self.name = f"research:{backend}:{self.model or 'default'}"

    # ------------------------------------------------------------------ clients
    def _client_for(self, role: str):
        want = self.role_model.get(role)
        key = (self.backend, want)
        if key in self._clients:
            return self._clients[key]
        if self.backend == "anthropic":
            try:
                import anthropic                              # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "the `anthropic` SDK is not installed. `pip install anthropic`, or use "
                    "--backend gemini, which needs no SDK.") from None
            if not self._key:
                raise RuntimeError(
                    "no LLM key. Set TRACK2_LLM_API_KEY to a PROJECT-SCOPED key with a "
                    "console spend limit; that limit is the only cap a harness bug cannot "
                    "exceed.")
            import anthropic
            obj = ("anthropic",
                   anthropic.Anthropic(api_key=self._key, base_url=self.base_url,
                                       timeout=self.timeout,
                                       max_retries=self.max_retries),
                   want or "claude-opus-5")
        else:
            from agent.backends import make_backend
            be = make_backend(self.backend, api_key=self._key, model=want,
                              base_url=self.base_url,
                              thinking_level=self.thinking_level)
            obj = ("backend", be, getattr(be, "model", want or "?"))
        self._clients[key] = obj
        return obj

    def model_for(self, role: str) -> str:
        """The model this role will actually use, after fallback. Reported in run_start."""
        return self._client_for(role)[2]

    def routing(self) -> dict:
        return {r: self.model_for(r) for r in ROLES}

    def usage_by_role(self) -> dict:
        """Tokens, calls, latency and failures per role."""
        agg: dict = {}
        for u in self.usage:
            a = agg.setdefault(u["role"], {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "thinking_tokens": 0, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0, "total_tokens": 0,
                "seconds": 0.0, "retries": 0, "refusals": 0, "truncations": 0,
                "models": set()})
            a["calls"] += 1
            for k in ("input_tokens", "output_tokens", "thinking_tokens",
                      "cache_read_input_tokens", "cache_creation_input_tokens"):
                a[k] += int(u.get(k) or 0)
            a["total_tokens"] += int(u.get("total_tokens") or 0)
            a["seconds"] += float(u.get("seconds") or 0.0)
            a["retries"] += int(u.get("retries") or 0)
            a["refusals"] += 1 if u.get("refused") else 0
            a["truncations"] += 1 if u.get("truncated") else 0
            a["models"].add(u.get("model"))
        for a in agg.values():
            a["models"] = sorted(m for m in a["models"] if m)
            a["seconds"] = round(a["seconds"], 2)
        return agg

    # ------------------------------------------------------------------ one call
    def _ask(self, role: str, sysb: list, user: str, schema: dict) -> tuple[dict, dict]:
        from agent.proposer import ProposerError, ProposerRefusal, ProposerTruncated
        kind, client, model = self._client_for(role)
        t0 = time.perf_counter()
        usage = {"role": role, "backend": self.backend, "model": model,
                 "effort": self.effort, "retries": 0, "refused": False,
                 "truncated": False}

        if kind == "backend":
            from agent.backends import BackendError
            system = "\n\n".join(sysb)
            sch = (strip_schema(schema, GEMINI_SAFE_KEYWORDS)
                   if self.backend == "gemini" else schema)
            try:
                resp, u = client.complete_json(system, user, sch)
                text = client.text_of(resp)
            except BackendError as e:
                usage["seconds"] = round(time.perf_counter() - t0, 2)
                usage["error"] = str(e)[:300]
                self.usage.append(usage)
                raise ProposerError(f"{self.name} at role {role}: {e}") from None
            usage.update({k: u.get(k, 0) for k in
                          ("input_tokens", "output_tokens", "cache_read_input_tokens",
                           "cache_creation_input_tokens")})
            usage["thinking_tokens"] = int(u.get("thinking_tokens")
                                           or u.get("total_thought_tokens") or 0)
            usage["stop_reason"] = u.get("stop_reason")
        else:
            import anthropic
            req = dict(
                model=model, max_tokens=self.max_tokens,
                system=[{"type": "text", "text": t} for t in sysb],
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema},
                               "effort": self.effort},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
            try:
                resp = client.beta.messages.create(**req)
            except anthropic.BadRequestError:
                # A 400 is never retryable; most often the beta/fallback pair is unavailable
                # on this account. Retry once without it, then let it raise.
                usage["retries"] += 1
                req.pop("betas", None); req.pop("fallbacks", None)
                resp = client.beta.messages.create(**req)
            u = resp.usage
            usage.update({
                "input_tokens": getattr(u, "input_tokens", 0),
                "output_tokens": getattr(u, "output_tokens", 0),
                "cache_creation_input_tokens":
                    getattr(u, "cache_creation_input_tokens", 0) or 0,
                "cache_read_input_tokens":
                    getattr(u, "cache_read_input_tokens", 0) or 0,
                "thinking_tokens": getattr(u, "thinking_tokens", 0) or 0,
                "stop_reason": getattr(resp, "stop_reason", None)})
            text = next((b.text for b in resp.content
                         if getattr(b, "type", "") == "text"), "")

        usage["total_tokens"] = sum(int(usage.get(k) or 0) for k in
                                    ("input_tokens", "output_tokens", "thinking_tokens"))
        usage["seconds"] = round(time.perf_counter() - t0, 2)
        stop = usage.get("stop_reason")
        if stop == "refusal":
            usage["refused"] = True
            self.usage.append(usage)
            raise ProposerRefusal(
                f"model declined at role {role}; recorded as an infrastructure event, not "
                f"an agent failure", usage)
        if stop == "max_tokens":
            usage["truncated"] = True
            self.usage.append(usage)
            raise ProposerTruncated(
                f"role {role} hit max_tokens before closing its JSON; retry at lower effort "
                f"or with a smaller parent", usage)
        self.usage.append(usage)
        try:
            return json.loads(text), usage
        except json.JSONDecodeError as e:
            raise ProposerError(
                f"{self.name} role {role} returned non-JSON despite a schema constraint: "
                f"{e}; first 200 chars: {text[:200]!r}") from None

    # -------------------------------------------------------------------- stages
    def plan(self, state, parent_source, parent_telemetry=None):
        s, u = plan_prompt(state, parent_source, parent_telemetry)
        return self._ask("plan", s, u, plan_schema())

    def tournament(self, state, parent_source, parent_telemetry=None, size: int = 6,
                   deficiency=None):
        s, u = tournament_prompt(state, parent_source, parent_telemetry, size, deficiency)
        return self._ask("plan", s, u, tournament_schema(size))

    def judge(self, candidates, state, parent_source, parent_telemetry=None,
              problems=None):
        s, u = judge_prompt(candidates, state, parent_source, parent_telemetry, problems)
        return self._ask("judge", s, u, judge_schema())

    def code(self, plan, parent_source, state, error=None, attempt=0):
        s, u = code_prompt(plan, parent_source, error, attempt, state)
        return self._ask("code", s, u, code_schema())

    def reflect(self, plan, result, state):
        s, u = reflect_prompt(plan, result, state)
        return self._ask("reflect", s, u, reflect_schema())
