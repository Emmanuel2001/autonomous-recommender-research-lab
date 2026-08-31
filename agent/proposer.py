"""Proposers. Same interface, swappable, so the control arm is free.

  RandomProposer   zero tokens, zero credentials. Samples uniformly from the SAME
                   bounds an LLM reads, so "does the LLM beat random" is a fair
                   comparison with one source of truth for the search space.
  LLMProposer      the real thing. Assembles a budgeted context and requests a
                   schema-constrained action. Requires the caller to supply a key.

The context assembler is shared, so the LLM backend is a drop-in once a key exists.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

from agent import actions as A

SYNTHETIC_MECHANISM = ("SYNTHETIC RATIONALE -- this proposal came from uniform random "
                       "sampling of the action space, not from reasoning. Present so "
                       "the random control arm satisfies the same schema as the LLM arm.")


class ProposerError(RuntimeError):
    """Infrastructure failure, not an agent failure. Carries usage so a refused or
    truncated call is still counted against the token budget."""
    def __init__(self, msg: str, usage: dict | None = None):
        super().__init__(msg)
        self.usage = usage or {}


class ProposerRefusal(ProposerError):
    pass


class ProposerTruncated(ProposerError):
    pass


class Proposer:
    name = "base"
    def propose(self, state: dict) -> tuple[dict, dict]:
        raise NotImplementedError


class RandomProposer(Proposer):
    name = "random"

    # Param kinds this arm knows how to draw from. An action is sampleable only if
    # EVERY one of its params has a distribution here.
    SAMPLEABLE_KINDS = frozenset({"enum", "weights", "fields", "int", "float",
                                  "diagnostic"})

    def __init__(self, seed: int = 0, actions=None):
        self.rng = random.Random(seed)
        if actions is None:
            from agent import actions as actions
        self.A = actions
        # Partition the action space up front instead of discovering an unsamplable
        # kind mid-run. Before this existed, drawing Inspect or ProposePatch raised
        # IndexError out of the numeric branch and ended the run -- and, worse for the
        # comparison it exists to support, the control arm was silently competing over
        # a SMALLER action space than the LLM arm.
        self.pool, self.unsampleable = [], {}
        for at in actions.ACTION_TYPES:
            unknown = [ps[0] for ps in actions.SPEC[at].values()
                       if ps[0] not in self.SAMPLEABLE_KINDS]
            if unknown:
                self.unsampleable[at] = sorted(set(unknown))
            else:
                self.pool.append(at)
        if not self.pool:
            raise RuntimeError(
                "the random control arm cannot sample ANY action in this space; kinds "
                f"present: {sorted({ps[0] for at in actions.ACTION_TYPES for ps in actions.SPEC[at].values()})}")

    def coverage_note(self) -> str:
        """What this arm cannot reach, in words, so the asymmetry is reported rather
        than hidden. `code` is the honest case: there is no uniform distribution over
        program texts, so a random arm cannot write a model. Any comparison against the
        LLM arm is therefore over the TYPED actions only, and saying so is the
        difference between a control and a flattering one."""
        if not self.unsampleable:
            return f"random arm samples all {len(self.pool)} actions uniformly"
        det = "; ".join(f"{k} (kind {'/'.join(v)})"
                        for k, v in sorted(self.unsampleable.items()))
        return (f"random arm samples {len(self.pool)} of "
                f"{len(self.pool) + len(self.unsampleable)} actions uniformly; cannot "
                f"reach {det} -- no uniform distribution exists over those, so the "
                f"control is over the typed actions only")

    def propose(self, state: dict) -> tuple[dict, dict]:
        A_ = self.A
        at = self.rng.choice(self.pool)
        params = {}
        for pname, spec in A_.SPEC[at].items():
            kind = spec[0]
            if kind == "diagnostic":
                from agent.inspect import DIAGNOSTICS
                params[pname] = self.rng.choice(sorted(DIAGNOSTICS))
            elif kind == "enum":
                params[pname] = self.rng.choice(list(spec[1]))
            elif kind == "weights":
                pool = list(spec[1])
                w = {pool[0]: 1.0}
                for sig in self.rng.sample(pool[1:], self.rng.randint(0, min(3, len(pool) - 1))):
                    w[sig] = round(self.rng.uniform(0.1, 2.0), 3)
                params[pname] = w
            elif kind == "fields":
                pool = list(spec[1])
                params[pname] = pool[:self.rng.randint(2, min(5, len(pool)))]
            elif kind == "int":
                lo, hi = spec[1], spec[2]
                params[pname] = self.rng.randint(lo, min(hi, max(lo, 1024)))
            else:
                lo, hi = spec[1], spec[2]
                # log-uniform over the same bounds the LLM is shown
                import math
                lo_ = max(lo, 1e-6)
                params[pname] = round(math.exp(self.rng.uniform(math.log(lo_),
                                                                math.log(min(hi, 1e4)))), 6)
        me = state["min_effect"]
        env = {"action_type": at, "params": params,
               "rationale": {"mechanism": SYNTHETIC_MECHANISM,
                             "target_metric": "both",
                             "expected_direction": "increase",
                             "expected_magnitude": round(min(me.values()) * 1.5, 6),
                             "cites": [], "symptoms_addressed": []},
               "parent_node_id": state.get("parent_node_id")}
        return env, {"input_tokens": 0, "output_tokens": 0,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


class LLMProposer(Proposer):
    name = "llm"

    def __init__(self, model: str = "claude-opus-5", api_key: str | None = None,
                 base_url: str | None = None, effort: str = "high",
                 timeout: float = 300.0, max_retries: int = 2):
        try:
            import anthropic                          # noqa: F401
        except ImportError:
            raise RuntimeError(
                "the `anthropic` SDK is not installed. `pip install anthropic`, then "
                "supply a key: ANTHROPIC_API_KEY=... (a project-scoped key with a "
                "console spend limit -- never the key that runs your shell)") from None
        key = api_key or os.environ.get("TRACK2_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "no LLM key. Set TRACK2_LLM_API_KEY to a PROJECT-SCOPED key with a "
                "console spend limit; that limit is the only cap a harness bug "
                "cannot exceed.")
        import anthropic
        # Python SDK timeout is in SECONDS. Timeouts are themselves retried, so
        # worst-case wall clock is timeout x (max_retries+1) -- 15 min here, which
        # the driver's --max-seconds cap must exceed.
        self.client = anthropic.Anthropic(api_key=key, base_url=base_url,
                                          timeout=timeout, max_retries=max_retries)
        self.model, self.effort = model, effort

    def propose(self, state: dict) -> tuple[dict, dict]:
        import anthropic
        blocks = assemble_context(state)
        # NOTE ON CACHING: the minimum cacheable prefix is ~1024 tokens. Our static
        # blocks are ~530 tokens, so a cache_control breakpoint would SILENTLY not
        # fire. We therefore do not set one, and we do not claim caching. If the
        # knowledge block grows past ~1024 tokens, add the breakpoint after it and
        # verify with usage.cache_read_input_tokens > 0 -- never assume.
        sys_blocks = [{"type": "text", "text": b["text"]}
                      for b in blocks if b["role"] == "system"]
        user = "\n\n".join(b["text"] for b in blocks if b["role"] == "user")

        req = dict(
            model=self.model, max_tokens=16000, system=sys_blocks,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": A.json_schema()},
                           "effort": self.effort},
            # Opus 5 can decline a request with HTTP 200 + stop_reason "refusal".
            # Server-side fallback routes by category so a 3am refusal does not
            # dead-end an unattended run.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        try:
            resp = self.client.beta.messages.create(**req)
        except anthropic.BadRequestError:
            # a 400 is never retryable -- most likely the beta/fallback pair is
            # unavailable on this account. Retry once without it, then give up.
            req.pop("betas", None); req.pop("fallbacks", None)
            resp = self.client.beta.messages.create(**req)

        u = resp.usage
        usage = {"input_tokens": getattr(u, "input_tokens", 0),
                 "output_tokens": getattr(u, "output_tokens", 0),
                 # cache reads STILL COUNT toward the rubric's total input tokens,
                 # so all three input fields must be summed
                 "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "stop_reason": getattr(resp, "stop_reason", None),
                 "model": self.model, "effort": self.effort}

        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            # stop_details is populated ONLY for refusals
            det = getattr(resp, "stop_details", None)
            cat = getattr(det, "category", None)
            raise ProposerRefusal(f"model declined (category={cat}); logged as an "
                                  f"infrastructure event, not an agent failure", usage)
        if stop == "max_tokens":
            raise ProposerTruncated("response hit max_tokens before producing an "
                                    "action; retry at lower effort", usage)

        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        return json.loads(text), usage


# ------------------------------------------------------------------- context
def assemble_context(state: dict, A=None) -> list[dict]:
    """Fresh every call. No conversation object, no growing history.

    Volatile blocks are AGGREGATED (per action type) rather than listed per
    variant, so context size is flat in iteration count instead of linear. The
    cache breakpoint sits after the static blocks.
    """
    if A is None:                      # default: the legacy action space
        from agent import actions as A
    task = state.get("knowledge", {}).get("task")
    header = task or ("Scored metrics: NDCG@10 and Recall@50 over the FULL 7,583-item "
                      "catalog, relevance = is_click. The reported score is the MEAN of "
                      "the two absolute improvements over a fixed baseline.")
    static = f"""You are proposing ONE experiment for an autonomous recommender-systems
research loop on KuaiRand-Pure.

{header}

You may emit exactly one action. Each action takes EXACTLY the params listed --
include those and nothing else:
{_render_spec(A)}

Example of a well-formed response (using an action that really exists above):
{json.dumps(_example(A), indent=1)}
{_patch_block(A)}

Rules that will REJECT your proposal:
  * an action_type not in the list above -- there are exactly {len(A.ACTION_TYPES)}
  * params other than the ones that action takes
  * expected_magnitude below the stated minimum effect
  * any parameter outside its stated bounds
  * anything the MEASURED_DEAD_ENDS below already rule out
Rejections are cheap; a vague mechanism is not. Innovation is judged on what you
chose to try AND WHY, not on the implementation."""

    measured = f"""MEASURED FACTS about this dataset -- treat as ground truth, they came
from runs on this exact split:
{json.dumps(state.get('knowledge', {}), indent=1)}"""

    hist = f"""CURRENT STATE
incumbent config : {json.dumps(state.get('incumbent', {}), default=str)}
incumbent score  : {json.dumps(state.get('incumbent_metrics', {}))}
minimum effect   : {json.dumps(state['min_effect'])}
iterations done  : {state.get('iteration', 0)}

ALREADY EVALUATED -- do NOT propose any config on this list again. A duplicate is
detected before training and wastes the iteration entirely.
{chr(10).join('  ' + c for c in state.get('population', {}).get('configs', [])) or '  (none yet)'}
({state.get('population', {}).get('total_evaluated', 0)} configs evaluated so far; showing
{state.get('population', {}).get('shown', 0)} -- the best and the most recent.)

Counts by action type:
{json.dumps(state.get('action_summary', {}), indent=1)}

FAILURES seen so far, by signature:
{json.dumps(state.get('failure_registry', {}), indent=1)}

DATA YOU HAVE ALREADY INSPECTED this run (each diagnostic answers once; asking again is
a wasted iteration). Use these numbers -- they are why you asked:
{json.dumps(state.get('findings', {}), indent=1) if state.get('findings') else '  (nothing inspected yet -- Inspect is free of any model fit)'}"""

    return [{"role": "system", "text": static, "cache": False},
            {"role": "system", "text": measured, "cache": True},
            {"role": "user", "text": hist, "cache": False}]


def _patch_block(A) -> str:
    """Show a working patch when the action space has a code escape hatch."""
    ex = getattr(A, "MINIMAL_PATCH_EXAMPLE", None)
    if not ex or "ProposePatch" not in getattr(A, "ACTION_TYPES", ()):
        return ""
    return ("\nIf you want to change something the parameters above cannot express, "
            "WRITE THE MODEL. A minimal patch that satisfies the contract looks like "
            "this -- it is deliberately trivial, the idea has to come from you:\n"
            "```python\n" + ex.strip() + "\n```\n"
            "Pass it as ProposePatch with params={\"code\": \"<the module>\", "
            "\"params\": {...}}.")


def _render_spec(A) -> str:
    """Describe the action space in words the model can act on.

    The previous version printed the raw spec tuple -- `bounds: {'code': ('code',)}` --
    which is meaningless, and the action it described went unused across two full runs.
    Kinds that need prose get it from A.KIND_HELP; numeric kinds get their real bounds.
    """
    help_map = dict(getattr(A, "KIND_HELP", {}))
    if "Inspect" in getattr(A, "SPEC", {}):
        try:
            from agent.inspect import DIAGNOSTICS
            help_map["diagnostic"] = ("one of " + ", ".join(
                f"{k} ({v[1]})" for k, v in DIAGNOSTICS.items()))
        except Exception:
            pass
    lines = []
    for name, spec in A.SPEC.items():
        parts = []
        for pname, ps in spec.items():
            kind = ps[0]
            if kind in help_map:
                parts.append(f"{pname}: {help_map[kind]}")
            elif kind == "enum":
                parts.append(f"{pname}: one of {list(ps[1])}")
            elif kind in ("int", "float"):
                parts.append(f"{pname}: {kind} in [{ps[1]}, {ps[2]}]")
            elif kind == "weights":
                parts.append(f"{pname}: object mapping any of {list(ps[1])} to a weight 0-5")
            elif kind == "fields":
                parts.append(f"{pname}: list from {list(ps[1])}")
            else:
                parts.append(f"{pname}: {kind}")
        lines.append(f"  {name}\n      " + "\n      ".join(parts))
    return "\n".join(lines)


def _example(A) -> dict:
    """A worked example built from the LIVE action space.

    Hardcoding one was a real bug: the example described the previous task's EASE
    actions while validation used the current space, and the model copied the example
    -- 12 of 12 proposals rejected for naming models that no longer exist.
    """
    at = A.ACTION_TYPES[0]
    params = {}
    for name, spec in A.SPEC[at].items():
        kind = spec[0]
        if kind == "enum":
            params[name] = spec[1][0]
        elif kind == "int":
            params[name] = int(spec[1])
        elif kind == "float":
            params[name] = float(spec[1])
        elif kind == "weights":
            params[name] = {spec[1][0]: 1.0}
        elif kind == "fields":
            params[name] = list(spec[1][:3])
    return {"action_type": at, "params": params,
            "rationale": {"mechanism": "State the mechanism in one or two real "
                                       "sentences: what is currently wrong, and why "
                                       "this change should fix it.",
                          "expected_direction": "increase",
                          "expected_magnitude": 0.005,
                          "symptoms_addressed": list(getattr(A, "SYMPTOMS", ())[:1])}}


class BackendProposer(Proposer):
    """Any HTTP backend from agent.backends: gemini, ollama, groq.

    Same context, same schema, same validation as every other arm -- only the
    endpoint differs, so "does the LLM beat random" stays a fair comparison with
    one source of truth for the search space.
    """

    def __init__(self, kind: str, api_key: str = "", model: str | None = None,
                 base_url: str | None = None, thinking_level: str | None = None,
                 actions=None):
        from agent.backends import make_backend
        if actions is None:
            from agent import actions as actions
        self.A = actions
        self.be = make_backend(kind, api_key=api_key, model=model,
                               base_url=base_url, thinking_level=thinking_level)
        self.name = f"{kind}:{getattr(self.be, 'model', '?')}"

    def propose(self, state: dict) -> tuple[dict, dict]:
        from agent.backends import BackendError
        blocks = assemble_context(state, self.A)
        system = "\n\n".join(b["text"] for b in blocks if b["role"] == "system")
        user = "\n\n".join(b["text"] for b in blocks if b["role"] == "user")
        try:
            resp, usage = self.be.complete_json(system, user, self.A.json_schema())
            text = self.be.text_of(resp)
        except BackendError as e:
            # infrastructure, not an agent failure -- the driver records it and
            # continues rather than ending an unattended run
            raise ProposerError(f"{self.name}: {e}") from None
        usage.setdefault("cache_creation_input_tokens", 0)
        usage.setdefault("cache_read_input_tokens", 0)
        usage["model"] = getattr(self.be, "model", "?")
        try:
            return json.loads(text), usage
        except json.JSONDecodeError as e:
            raise ProposerError(f"{self.name} returned non-JSON despite a schema "
                                f"constraint: {e}; first 200 chars: {text[:200]!r}") from None


def make(kind: str, **kw) -> Proposer:
    """`actions` selects the action-space module. Defaults to the legacy one so old
    runs reproduce; the official driver passes agent.official_actions."""
    if kind == "random":
        return RandomProposer(seed=kw.get("seed", 0), actions=kw.get("actions"))
    if kind == "anthropic":
        return LLMProposer(**{k: v for k, v in kw.items()
                              if k in ("model", "api_key", "base_url", "effort")})
    if kind in ("gemini", "ollama", "groq"):
        return BackendProposer(kind, api_key=kw.get("api_key", ""),
                               model=kw.get("model"), base_url=kw.get("base_url"),
                               thinking_level=kw.get("thinking_level"),
                               actions=kw.get("actions"))
    raise ValueError(f"unknown proposer {kind!r}; choose "
                     f"random | gemini | ollama | groq | anthropic")
