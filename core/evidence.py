"""Signed experimental evidence, reconciled at the level the experiment actually tested.

THE DEFECT THIS EXISTS TO FIX
-----------------------------
`anthropic_contract_gate2` ran one BPR implementation, lost by 0.000738 on one seed, and
its reflector concluded that objective-metric misalignment "is not the binding constraint".
The signed chain already said otherwise: `research_trials` seq 116 confirms the submitted
pairwise champion at +0.001888 over the official baseline across twenty paired seeds,
20/20 positive, t = 11.5, CONFIRMED.

Both are correct about what they ran. They ran different things:

    tree fe985741  loss pointwise -> pairwise, neg_per_pos 2      +0.001888, 20 seeds
    tree bd3d24c5  loss pointwise -> pairwise, pairs_per_epoch    -0.000738,  1 seed
                   600000

An experiment supports or contradicts the IMPLEMENTATION it executed. It is evidence
about the mechanism only in combination with every other implementation of that mechanism.
The old grader collapsed that distinction -- one contradicting iteration returned
"contradicted" outright -- so a single-seed negative could bury a twenty-seed confirmation.

WHY BELIEFS COULD NOT RECONCILE AT ALL
--------------------------------------
Beliefs were keyed by the reflector's PROSE. Two runs describing the same mechanism in
different words produced two unrelated beliefs, so no grader could ever have seen the
conflict. The join key here is derived from facts instead: the CATEGORICAL transitions in
`config.json` between a candidate and its parent. A categorical change (pointwise ->
pairwise, a field added) selects a different mechanism; a numeric change (neg_per_pos 1 ->
2, a learning rate) tunes the same one. That rule is general -- it names no KuaiRand
column -- and on the real evidence it puts both BPR runs under one key while keeping their
implementations distinct.

WHAT IS FACT HERE AND WHAT IS NOT
---------------------------------
Facts, and the only things this module will vouch for: the tree and parent hashes, the
files the harness computed as changed, the parsed `config.json`, the scorer-signed delta,
and confirmation records read from a signed ledger.

NOT facts: how the pairs were sampled, how the optimizer applied the update, whether the
bias was frozen. Those are readable only from generated source, and static analysis cannot
establish what code does. They travel as `procedure_claims`, labelled LLM-authored, and no
grade is ever computed from them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


#: Grades, weakest to strongest evidentiary standing. `mixed` is not a rung on this ladder
#: -- it means the observations disagree and the reader must look at which implementation
#: carries which result.
GRADES = ("unreplicated", "single_signed_result", "replicated", "mixed", "seed_confirmed")

#: Facts that must match before signed evidence may cross from one run into another.
#: Anything absent on either side fails closed: unknown provenance is not compatibility.
COMPATIBILITY_KEYS = ("name", "metric", "label", "dev_split", "held_out_split",
                      "dataset_fingerprint", "data_contract_sha256",
                      "workspace_contract_sha256")


# --------------------------------------------------------------------- mechanism keys
def _categorical(value) -> bool:
    """Whether a config value names a KIND of experiment rather than a magnitude.

    Booleans are categorical even though Python calls them numbers, which is why the
    isinstance check tests bool first.
    """
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return False
    return True


def _render(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def config_delta(parent: dict, child: dict) -> dict:
    """Structured difference between two candidate configurations."""
    parent = parent or {}
    child = child or {}
    changed = {k: (parent[k], child[k]) for k in sorted(set(parent) & set(child))
               if parent[k] != child[k]}
    return {
        "changed": changed,
        "added": {k: child[k] for k in sorted(set(child) - set(parent))},
        "removed": {k: parent[k] for k in sorted(set(parent) - set(child))},
    }


def mechanism_key(parent_config: dict, child_config: dict,
                  changed_files=()) -> str:
    """A join key two runs can independently derive for the SAME mechanism.

    Categorical transitions only. `loss=pointwise->pairwise` is the mechanism; the
    `neg_per_pos` and `pairs_per_epoch` that differ between the two BPR implementations are
    magnitudes and belong to the implementation, not to the key -- which is exactly why the
    two runs join here and are still told apart below.

    A list-valued field reports its membership change (`fields+user_hist_len`) rather than
    its whole new value, so adding one feature to a six-field model does not read as a
    different mechanism from adding it to a five-field one.

    With no categorical config change the key falls back to the source files the harness
    saw change, so an architecture rewrite that touches no config still gets a stable key.
    """
    delta = config_delta(parent_config, child_config)
    parts = []
    for key, (old, new) in delta["changed"].items():
        if not (_categorical(old) or _categorical(new)):
            continue
        if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
            gained = [v for v in new if v not in old]
            lost = [v for v in old if v not in new]
            if gained:
                parts.append(f"{key}+{_render(gained)}")
            if lost:
                parts.append(f"{key}-{_render(lost)}")
        else:
            parts.append(f"{key}={_render(old)}->{_render(new)}")
    for key, value in delta["added"].items():
        # An added categorical switch is a mechanism; an added numeric knob is tuning.
        if _categorical(value) and not isinstance(value, (list, tuple)):
            parts.append(f"{key}=+{_render(value)}")
    if parts:
        return "|".join(sorted(parts))
    code = sorted(f for f in (changed_files or ()) if str(f).endswith(".py"))
    if code:
        return "code:" + ",".join(code)
    return "unclassified"


def implementation_key(child_config: dict, changed_files=()) -> str:
    """Distinguishes implementations WITHIN one mechanism.

    The numeric and structural detail the mechanism key deliberately drops: this is what
    separates `neg_per_pos=2` from `pairs_per_epoch=600000` when both are "pairwise".
    """
    cfg = child_config or {}
    numeric = [f"{k}={cfg[k]}" for k in sorted(cfg)
               if not _categorical(cfg[k])]
    files = sorted(str(f) for f in (changed_files or ()))
    return "|".join(numeric + [f"files:{','.join(files)}"] if files else numeric)


# ------------------------------------------------------------------------ observations
@dataclass
class Observation:
    """One implementation of one mechanism, and what the scorer said about it.

    Every field above `procedure_claims` is a fact the harness or the scorer produced.
    `procedure_claims` is LLM-authored prose kept beside them and never graded.
    """
    mechanism: str                      # the derived join key
    tree: str = ""
    parent_tree: str = ""
    changed_files: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    config_delta: dict = field(default_factory=dict)
    feature_fields: list = field(default_factory=list)
    model_family: str = ""
    signed_metrics: dict = field(default_factory=dict)
    delta: float | None = None          # scorer-signed, versus the stated comparison
    delta_versus: str = ""              # "parent" | "champion"
    #: {n, mean, wins, verdict, champion_tree, comparison_tree, ledger, seq}
    seed_confirmation: dict | None = None
    #: LLM-authored, never a fact: sampling procedure, optimizer/update path, notes.
    procedure_claims: dict = field(default_factory=dict)
    #: where this came from, so a reader can go back to the signed record
    provenance: dict = field(default_factory=dict)
    mechanism_label: str = ""           # the reflector's prose, kept as a label only

    @property
    def direction(self) -> str:
        if self.delta is None:
            return "unknown"
        if self.delta > 0:
            return "positive"
        if self.delta < 0:
            return "negative"
        return "neutral"

    @property
    def components(self) -> list[str]:
        """The individual mechanisms this candidate changed.

        A candidate that flips the loss AND adds a feature carries the compound key
        `fields+user_hist_len|loss=pointwise->pairwise`. It is real evidence about each
        component and clean evidence about neither, so it is indexed under both and marked
        confounded. Requiring an exact key match instead would have hidden the twenty-seed
        BPR confirmation from the very mechanism it is about.
        """
        return [p for p in str(self.mechanism or "").split("|") if p]

    @property
    def confounded(self) -> bool:
        """True when this observation changed more than one mechanism at once."""
        return len(self.components) > 1

    @property
    def confirmed(self) -> bool:
        """A CONFIRMED multi-seed result for THIS tree. NOT CONFIRMED is not confirmation."""
        c = self.seed_confirmation or {}
        return str(c.get("verdict", "")).upper().startswith("CONFIRMED")

    @property
    def seeds(self) -> int:
        return int((self.seed_confirmation or {}).get("n") or 1)

    def to_record(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction
        d["confirmed"] = self.confirmed
        d["confounded"] = self.confounded
        d["components"] = self.components
        d["seeds"] = self.seeds
        return d

    def short(self, context: str = "") -> str:
        impl = implementation_summary(self)
        got = "no signed delta" if self.delta is None else f"{self.delta:+.6f}"
        if self.seed_confirmation:
            c = self.seed_confirmation
            got = (f"paired mean {c.get('mean'):+.6f}, {c.get('wins')}/{c.get('n')} seeds, "
                   f"{c.get('verdict')}")
        note = ""
        others = [c for c in self.components if c != context]
        if self.confounded and others:
            note = (f" [CONFOUNDED: the same candidate also changed "
                    f"{', '.join(others)}]")
        return f"tree {self.tree[:12]} ({impl}): {got}{note}"


def implementation_summary(obs: "Observation") -> str:
    """The handful of config values that actually distinguish this implementation.

    Deliberately short: this is read inside a prompt, and the full config is one ledger
    lookup away via the tree hash.
    """
    cfg = obs.config or {}
    delta = (obs.config_delta or {})
    interesting = list((delta.get("added") or {})) + [
        k for k in (delta.get("changed") or {}) if not _categorical(cfg.get(k))]
    seen, parts = set(), []
    for k in interesting:
        if k in seen or k in ("max_seconds", "train_curve_every", "seed"):
            continue
        seen.add(k)
        parts.append(f"{k}={_render(cfg.get(k))}")
    return ", ".join(parts[:4]) or "baseline configuration"


# ----------------------------------------------------------------------- confirmations
class ConfirmationIndex:
    """Seed confirmations, keyed by the ORDERED pair they actually compared.

    Keying by champion alone would be wrong in a way this repository can demonstrate:
    `research_trials` seq 116 confirms fe985741 against the official baseline, while
    `opus_vs_gemini_20seed` seq 42 reports NOT CONFIRMED for 0375004 against fe985741. The
    same tree is the champion of one and the comparison of the other. Attaching by tree
    alone would let the second read as evidence against the first.
    """

    def __init__(self):
        self._by_pair: dict[tuple, dict] = {}

    def add(self, body: dict, *, ledger: str = "", seq=None, run_id: str = "") -> bool:
        champ = str(body.get("champion_tree") or "")
        comp = str(body.get("comparison_tree") or "")
        if not champ or not comp or body.get("n") is None:
            return False
        record = {k: body.get(k) for k in
                  ("n", "mean", "sd", "se", "t", "wins", "verdict", "seeds",
                   "comparison", "method")}
        record.update({"champion_tree": champ, "comparison_tree": comp,
                       "ledger": str(ledger), "seq": seq, "run_id": str(run_id)})
        self._by_pair[(champ, comp)] = record
        return True

    def for_pair(self, champion_tree: str, comparison_tree: str) -> dict | None:
        return self._by_pair.get((str(champion_tree), str(comparison_tree)))

    def for_champion(self, tree: str) -> list[dict]:
        """Confirmations where `tree` is the CHAMPION. Never where it is the comparison."""
        return [v for (champ, _), v in sorted(self._by_pair.items())
                if champ == str(tree)]

    def best_for(self, tree: str) -> dict | None:
        """The strongest CONFIRMED record for this tree as champion, else None."""
        confirmed = [r for r in self.for_champion(tree)
                     if str(r.get("verdict", "")).upper().startswith("CONFIRMED")]
        if not confirmed:
            return None
        return max(confirmed, key=lambda r: (int(r.get("n") or 0),
                                             abs(float(r.get("mean") or 0.0))))

    def __len__(self) -> int:
        return len(self._by_pair)


# ----------------------------------------------------------------------------- grading
def grade(observations) -> str:
    """Categorical strength for one mechanism, from signed outcomes only.

    The rule that matters: observations of BOTH signs are `mixed`, whether or not either
    side is seed-confirmed. Two single-seed results disagreeing is the common case and is
    just as much a conflict as a confirmed result meeting a negative -- the old grader
    returned "contradicted" for both, which discarded the positive evidence entirely.

    Seed confirmation decides which implementation is STRONGEST (see `strongest`), not
    whether the mechanism is contested.
    """
    obs = list(observations or [])
    if not obs:
        return "unreplicated"
    positive = [o for o in obs if o.direction == "positive"]
    negative = [o for o in obs if o.direction == "negative"]
    if positive and negative:
        return "mixed"
    same = positive or negative
    if not same:
        return "unreplicated"
    if any(o.confirmed for o in same):
        return "seed_confirmed"
    if len({o.tree for o in same}) >= 2:
        return "replicated"
    return "single_signed_result"


def strongest(observations) -> "Observation | None":
    """The observation a reader should weigh most, by evidence, never by recency.

    Ordering: a CONFIRMED seed result beats anything unconfirmed; among confirmed, more
    seeds wins; among equals, the larger absolute effect. An unconfirmed single-seed result
    can never outrank a confirmation, which is the property requirement 3 asks for.
    """
    obs = [o for o in (observations or []) if o.delta is not None or o.seed_confirmation]
    if not obs:
        return None
    return max(obs, key=lambda o: (o.confirmed, o.seeds,
                                   abs(float(o.delta or (o.seed_confirmation or {}).get("mean") or 0.0))))


def conflict_summary(mechanism: str, observations, *, label: str = "") -> dict:
    """The planner-facing reconciliation for one mechanism.

    Everything except `interpretation` is rendered from signed records. `interpretation` is
    the only synthesised sentence and says so, because the point of this module is that a
    reader can tell measurement from commentary at a glance.
    """
    obs = list(observations or [])
    g = grade(obs)
    best = strongest(obs)
    positive = [o for o in obs if o.direction == "positive"]
    negative = [o for o in obs if o.direction == "negative"]
    out = {
        "mechanism": mechanism,
        "label": label or (obs[0].mechanism_label if obs else ""),
        "evidence_grade": g,
        "implementations": len({o.tree for o in obs}),
        "supporting": [o.short(mechanism) for o in positive],
        "contradicting": [o.short(mechanism) for o in negative],
        "strongest_evidence": (None if best is None else {
            "tree": best.tree,
            "implementation": implementation_summary(best),
            "grade": "seed_confirmed" if best.confirmed else "single_signed_result",
            "seeds": best.seeds,
            "detail": best.short(mechanism),
        }),
        "procedure_claims_are_llm_authored": True,
    }
    out["confounded_implementations"] = sorted(
        {o.tree[:12] for o in obs if o.confounded})
    if g == "mixed" and best is not None:
        out["interpretation"] = (
            f"MIXED EVIDENCE, not a settled negative. This mechanism has both supporting "
            f"and contradicting implementations, so it is implementation-sensitive. The "
            f"strongest evidence is tree {best.tree[:12]} "
            f"({implementation_summary(best)}). Differences in the untested implementation "
            f"details remain unresolved; a new attempt must say what it changes about the "
            f"implementation, not merely restate the mechanism.")
    elif g == "seed_confirmed":
        out["interpretation"] = (
            "Confirmed across paired seeds for the implementation named above. This is the "
            "strongest class of evidence this project produces.")
    else:
        out["interpretation"] = (
            f"Evidence grade {g}: derived from signed deltas, not from reflector "
            f"confidence.")
    out["interpretation_is_synthesised"] = True
    return out


# --------------------------------------------------------------------------- ingestion
def compatible(current: dict, candidate: dict) -> tuple[bool, list[str]]:
    """Whether signed evidence may cross from `candidate` provenance into `current`.

    Fails closed on absence. A run whose scope was never recorded -- every belief written
    before the scope stamp existed -- cannot be shown to describe the same task, dataset,
    split, metric, label or contracts, and prose that merely sounds relevant is not
    evidence that it is.
    """
    reasons = []
    for key in COMPATIBILITY_KEYS:
        a, b = (current or {}).get(key), (candidate or {}).get(key)
        if a is None or b is None:
            reasons.append(f"{key} is unrecorded on one side; refusing to assume a match")
        elif a != b:
            reasons.append(f"{key} differs ({a!r} vs {b!r})")
    return (not reasons), reasons


def observations_from_state(state_json: dict, *, confirmations: ConfirmationIndex | None,
                            parent_configs: dict | None = None,
                            provenance: dict | None = None) -> list[Observation]:
    """Derive observations from a recorded run state.

    `parent_configs` maps a tree hash to its parsed config.json when the caller can supply
    it (the artifact store can). Without it the mechanism key falls back to changed files,
    which is weaker but never wrong -- it simply will not join across implementations.
    """
    parent_configs = parent_configs or {}
    out = []
    for exp in (state_json.get("experiments") or []):
        tree = str(exp.get("tree") or "")
        parent = str(exp.get("parent") or "")
        if not tree or not exp.get("signed"):
            continue
        child_cfg = parent_configs.get(tree) or {}
        parent_cfg = parent_configs.get(parent) or {}
        changed = list(exp.get("files_changed") or [])
        key = mechanism_key(parent_cfg, child_cfg, changed)
        delta = exp.get("delta_vs_parent")
        versus = "parent"
        if delta is None:
            delta, versus = exp.get("delta_vs_champion"), "champion"
        conf = confirmations.best_for(tree) if confirmations else None
        refl = exp.get("reflection") or {}
        out.append(Observation(
            mechanism=key, tree=tree, parent_tree=parent, changed_files=changed,
            config=child_cfg, config_delta=config_delta(parent_cfg, child_cfg),
            feature_fields=list(child_cfg.get("fields") or []),
            model_family=str(child_cfg.get("model") or child_cfg.get("model_family") or ""),
            signed_metrics=dict((exp.get("signed") or {}).get("metrics") or {}),
            delta=delta, delta_versus=versus, seed_confirmation=conf,
            procedure_claims={k: str(refl.get(k))[:400] for k in
                              ("what_happened", "remaining_uncertainty") if refl.get(k)},
            provenance={**(provenance or {}), "iteration": exp.get("iteration"),
                        "outcome": exp.get("outcome")},
            mechanism_label=str(exp.get("mechanism") or exp.get("hypothesis") or "")[:200],
        ))
    return out


def index_by_mechanism(observations, *, expand: bool = True) -> dict[str, list[Observation]]:
    """Group observations by mechanism.

    With `expand` a compound observation appears under every component it changed, so
    evidence about "pairwise" is found whether the candidate changed only the loss or the
    loss together with a feature. `Observation.confounded` keeps the distinction visible;
    it is never silently treated as a clean single-mechanism result.
    """
    out: dict[str, list[Observation]] = {}
    for o in observations or ():
        keys = o.components if (expand and o.components) else [o.mechanism]
        for key in keys:
            out.setdefault(key, []).append(o)
    return out


def planner_view(by_mechanism: dict, *, limit: int = 8,
                 conflicts_first: bool = True) -> list[dict]:
    """Bounded reconciliation blocks for the prompt, conflicts first.

    Conflicts lead because a mechanism whose evidence disagrees is the one a planner is
    most likely to get wrong -- it is where a confident single result reads as settled.
    """
    blocks = [conflict_summary(k, v) for k, v in sorted(by_mechanism.items())]
    if conflicts_first:
        order = {"mixed": 0, "seed_confirmed": 1, "replicated": 2,
                 "single_signed_result": 3, "unreplicated": 4}
        blocks.sort(key=lambda b: (order.get(b["evidence_grade"], 9), b["mechanism"]))
    return blocks[:limit]


def read_confirmations(ledger_paths) -> ConfirmationIndex:
    """Collect every confirmation_result from signed ledgers. Read-only, never writes."""
    idx = ConfirmationIndex()
    for path in ledger_paths or ():
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            payload = entry.get("payload") or {}
            body = payload.get("body") or {}
            if body.get("what") != "confirmation_result":
                continue
            idx.add(body, ledger=str(path), seq=entry.get("seq"),
                    run_id=str(body.get("run_id") or ""))
    return idx
