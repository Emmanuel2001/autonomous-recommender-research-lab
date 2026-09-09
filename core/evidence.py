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
from pathlib import Path


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


# =====================================================================================
#                        VERIFIED INGESTION FROM SIGNED LEDGERS
# =====================================================================================
# Everything below consumes the bounded bundle that core.scorer.evidence_bundle returns
# after verifying a ledger with the key this process does not hold. Nothing here reads a
# ledger directly, and nothing here trusts an editable file: state.json is not evidence.

import hashlib
import statistics


#: A recomputed confirmation must reproduce the recorded one this closely. The scorer
#: rounds its metrics to 6dp, so the deltas are exact to that; anything larger is a real
#: disagreement and is reported rather than smoothed over.
CONFIRMATION_TOLERANCE = 1e-6

#: Scalar facts a legacy ledger must match exactly. The split-level facts (names, row
#: counts, content hashes) are compared separately and in full, because a matching label
#: and metric over DIFFERENT rows is not the same experiment.
LEGACY_KEYS = ("task", "label", "metric", "workspace_contract_sha256")


def seedless_identity(files: dict) -> str | None:
    """Content address of a candidate tree with the model seed removed.

    A per-seed refit is a different tree by design -- `seed` lives in config.json and is
    part of candidate identity, so a deliberate replicate is not silently deduplicated.
    That makes the 40 per-seed records of a 20-seed confirmation 40 distinct trees. This
    strips the seed so each refit can be bound BY CONTENT to the lineage it belongs to,
    rather than by its position in the ledger.
    """
    if not files or "config.json" not in files:
        return None
    try:
        cfg = json.loads(files["config.json"])
    except (ValueError, TypeError):
        return None
    cfg.pop("seed", None)
    rest = {k: v for k, v in files.items() if k != "config.json"}
    payload = json.dumps({**rest, "config.json": json.dumps(cfg, sort_keys=True)},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verified_config(tree: str, reader) -> tuple[dict | None, str]:
    """Read a candidate's config AFTER re-deriving its content address.

    Requirement: a configuration is only evidence about a mechanism if it is the
    configuration that tree actually holds. `reader` returns the stored files for a tree
    and is expected to re-hash them; a mismatch yields no config and a stated reason, and
    the caller must then refuse the observation rather than guess.
    """
    if not tree:
        return None, "no tree hash"
    try:
        files = reader(tree)
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        return None, f"artifact unreadable: {type(exc).__name__}"
    if not files:
        return None, "artifact missing from the store"
    try:
        from core import artifacts as AR
        recomputed = AR.tree_hash(files)
    except Exception as exc:                       # noqa: BLE001
        return None, f"tree hash could not be recomputed: {type(exc).__name__}"
    if recomputed != tree:
        return None, f"artifact hash mismatch: stored content hashes to {recomputed[:12]}"
    if "config.json" not in files:
        return None, "tree carries no config.json"
    try:
        return json.loads(files["config.json"]), ""
    except ValueError:
        return None, "config.json is not valid JSON"


def recompute_confirmation(per_seed, lineage_of) -> dict:
    """Recompute a paired-seed confirmation from the SIGNED per-seed score records.

    `confirmation_result` is a derived summary written by the same process that ran the
    comparison. It is convenient, and it is not a measurement -- so it is recomputed here
    from the per-seed records, each bound to its lineage by content through
    `lineage_of`, and the two are compared. A forged summary cannot survive because the
    numbers come from elsewhere.

    Returns the recomputed statistics plus `pairs` actually used. Seeds that do not yield
    exactly one champion and one comparison record are dropped and counted, never guessed.
    """
    by_seed: dict = {}
    for rec in per_seed or ():
        seed = rec.get("seed")
        primary = ((rec.get("metrics") or {}).get("primary"))
        if seed is None or primary is None:
            continue
        lineage = lineage_of(rec.get("config_hash") or "")
        if lineage not in ("champion", "comparison"):
            continue
        by_seed.setdefault(seed, {}).setdefault(lineage, []).append(float(primary))
    deltas, dropped = [], []
    for seed in sorted(by_seed):
        sides = by_seed[seed]
        champ, comp = sides.get("champion") or [], sides.get("comparison") or []
        if len(champ) != 1 or len(comp) != 1:
            dropped.append(seed)
            continue
        deltas.append(round(champ[0] - comp[0], 6))
    n = len(deltas)
    if n == 0:
        return {"n": 0, "pairs": 0, "dropped_seeds": dropped,
                "reason": "no seed produced exactly one champion and one comparison fit"}
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    se = (sd / (n ** 0.5)) if n > 1 else 0.0
    return {
        "n": n, "pairs": n, "mean": round(mean, 6), "sd": round(sd, 6),
        "se": round(se, 6), "t": (round(mean / se, 2) if se else None),
        "wins": sum(1 for d in deltas if d > 0),
        "deltas": deltas, "dropped_seeds": dropped,
        "recomputed_from": "signed per-seed score records",
    }


def compare_confirmation(recorded: dict, recomputed: dict) -> dict:
    """Whether a recorded confirmation summary matches what the signed records say."""
    if not recomputed or not recomputed.get("n"):
        return {"agrees": False, "reason": "nothing could be recomputed",
                "detail": recomputed.get("reason", "")}
    discrepancies = []
    for field in ("n", "wins"):
        a, b = recorded.get(field), recomputed.get(field)
        if a is not None and a != b:
            discrepancies.append(f"{field}: recorded {a}, recomputed {b}")
    for field in ("mean", "sd"):
        a, b = recorded.get(field), recomputed.get(field)
        if a is not None and abs(float(a) - float(b)) > CONFIRMATION_TOLERANCE:
            discrepancies.append(f"{field}: recorded {a}, recomputed {b}")
    return {"agrees": not discrepancies, "discrepancies": discrepancies}


def verdict_from(recomputed: dict, *, epsilon: float = 0.0) -> str:
    """The verdict this project's own protocol implies, derived not copied.

    Matches confirm_champion.py: a paired t over the seed deltas, CONFIRMED when the mean
    is positive and t exceeds 2.5. The recorded verdict is compared against this rather
    than believed.
    """
    n, mean, t = recomputed.get("n") or 0, recomputed.get("mean"), recomputed.get("t")
    if n < 2 or mean is None or t is None:
        return "INSUFFICIENT"
    return "CONFIRMED" if (mean > epsilon and t > 2.5) else "NOT CONFIRMED"


def legacy_compatibility(current: dict, facts: dict) -> tuple[bool, list[str], dict]:
    """Permit import from a pre-contract ledger ONLY on exact authoritative facts.

    An older ledger has no data-contract or workspace-contract hash, so `compatible()`
    fails it closed -- correctly, because absence is not a match. This adapter is the
    explicit, recorded alternative, and it is deliberately demanding: the scalar facts
    AND the full split manifest must agree.

    Splits are compared by NAME, ROW COUNT and CONTENT HASH, all three, for every split
    on either side. A label and metric that match over different rows describe a different
    experiment, and a train/dev/held-out boundary that moved silently is exactly the error
    a content hash exists to catch. Nothing missing is filled in from the other side --
    the whole point is that the two were recorded independently.
    """
    reasons, matched = [], {}
    for key in LEGACY_KEYS:
        a, b = (current or {}).get(key), (facts or {}).get(key)
        if a is None or b is None:
            reasons.append(f"{key} is unavailable on one side; the legacy adapter "
                           f"requires it to be recorded, not assumed")
        elif a != b:
            reasons.append(f"{key} differs ({str(a)[:24]!r} vs {str(b)[:24]!r})")
        else:
            matched[key] = a

    cur_splits = (current or {}).get("splits") or {}
    old_splits = (facts or {}).get("splits") or {}
    if not cur_splits or not old_splits:
        reasons.append("split manifest is unavailable on one side; the legacy adapter "
                       "requires row counts and content hashes for every split")
        return False, reasons, matched
    if set(cur_splits) != set(old_splits):
        reasons.append(f"split names differ ({sorted(cur_splits)} vs {sorted(old_splits)})")
        return False, reasons, matched
    for name in sorted(cur_splits):
        a, b = cur_splits[name] or {}, old_splits[name] or {}
        for field in ("rows", "content_sha256"):
            av, bv = a.get(field), b.get(field)
            if av is None or bv is None:
                reasons.append(f"split {name}: {field} unavailable on one side")
            elif av != bv:
                reasons.append(f"split {name}: {field} differs "
                               f"({str(av)[:20]} vs {str(bv)[:20]})")
    if not reasons:
        matched["splits"] = {k: dict(v) for k, v in sorted(cur_splits.items())}
    return (not reasons), reasons, matched


def code_mechanism_key(changed_files, tree: str) -> str:
    """A deliberately NON-mergeable key for a candidate with no categorical config change.

    Two unrelated architecture rewrites both touch model.py and train.py. Nothing
    structured distinguishes them -- only the reflector's prose, which is not a verified
    fact and must never become a join key. So a code-only mechanism is scoped to its own
    candidate and will not merge with another. Under-merging costs a missed connection;
    over-merging would fabricate agreement between two unrelated experiments.
    """
    files = ",".join(sorted(str(f) for f in (changed_files or ()) if str(f).endswith(".py")))
    return f"code:{files or 'none'}@{str(tree)[:12]}"


def _primary_by_tree(records, *, split: str | None = None) -> tuple[dict, dict, dict]:
    """tree -> scorer-signed primary, plus conflicts and deduplication disclosure.

    A cross-run delta is derived from the signed number the scorer produced, never from a
    `delta_vs_parent` in an editable state.json.

    AMBIGUITY IS REFUSED, NOT RESOLVED. The same tree may legitimately be scored more than
    once -- a rerun, a re-registration -- and identical values are then deduplicated and
    disclosed. But two DIFFERENT primaries for the same tree in the same evaluation
    context mean the ledger does not say what that candidate scored, and picking the first
    would silently choose one reading of contradictory evidence. Returns
    (primaries, conflicts, duplicates); the caller must refuse any tree in `conflicts`.

    Context is (split, split content hash, metric, label): the same tree scored on a
    different split, or under a different metric, is not a contradiction.
    """
    seen: dict = {}
    for r in records:
        if r.get("kind") not in ("score", "confirmation"):
            continue
        tree = str(r.get("config_hash") or "")
        primary = (r.get("metrics") or {}).get("primary")
        if not tree or primary is None:
            continue
        # Deltas are compared within ONE split. A tree legitimately scored on both the
        # dev and the held-out split has two primaries and that is not a contradiction,
        # so anything outside the split under comparison is skipped rather than conflated.
        if split is not None and r.get("split") not in (None, split):
            continue
        context = (r.get("split"), r.get("split_content_sha256"), r.get("metric"),
                   r.get("label"))
        seen.setdefault((tree, context), []).append(
            {"seq": r.get("seq"), "primary": float(primary)})

    primaries, conflicts, duplicates = {}, {}, {}
    for (tree, context), rows in seen.items():
        values = {row["primary"] for row in rows}
        if len(values) > 1:
            conflicts.setdefault(tree, []).append({
                "split": context[0], "metric": context[2],
                "values": sorted(values),
                "sequences": sorted(r["seq"] for r in rows if r["seq"] is not None)})
            continue
        value = rows[0]["primary"]
        if len(rows) > 1:
            duplicates[tree] = {
                "identical_scores": len(rows), "primary": value,
                "sequences": sorted(r["seq"] for r in rows if r["seq"] is not None)}
        if tree in primaries and primaries[tree] != value:
            conflicts.setdefault(tree, []).append({
                "reason": "the same tree scored differently in two evaluation contexts "
                          "on the same split",
                "values": sorted({primaries[tree], value})})
            continue
        primaries[tree] = value
    for tree in conflicts:
        primaries.pop(tree, None)
    return primaries, conflicts, duplicates


def ingest_bundle(bundle: dict, *, current_scope: dict, reader,
                  allow_legacy: bool = False) -> tuple[list, dict]:
    """Turn a scorer-verified bundle into observations, refusing everything unproven.

    The refusals are the substance of this function. An observation is produced only when
    the ledger verified, the lineage is compatible, both the candidate and its parent
    resolve to stored trees whose content re-hashes to the recorded address, and the delta
    comes from two signed score records. Anything else is counted with a reason and
    dropped.

    Returns (observations, summary) where the summary is what run_start records.
    """
    observations, rejected = [], []
    summary = {"ledgers": [], "observations_imported": 0, "observations_rejected": 0,
               "confirmations_recomputed": 0, "confirmation_discrepancies": [],
               "compatibility_method": None, "imported_sequences": []}

    def reject(reason, **ctx):
        rejected.append({"reason": reason, **ctx})

    for led in (bundle or {}).get("ledgers", []):
        entry = {"ledger": led.get("ledger"), "sha256": led.get("sha256"),
                 "verified": bool(led.get("verified")),
                 "verified_entries": led.get("entries", 0),
                 "imported_sequences": [], "compatibility_method": None,
                 "observations_imported": 0, "observations_rejected": 0}
        if not led.get("verified"):
            entry["refused"] = led.get("refused") or "ledger did not verify"
            reject(entry["refused"], ledger=led.get("ledger"))
            summary["ledgers"].append(entry)
            continue

        records = led.get("records") or []
        starts = [r for r in records if r.get("kind") == "run_start"]
        scores = [r for r in records if r.get("kind") in ("score", "confirmation")]

        # ---- compatibility, strict first and legacy only when asked for explicitly
        ledger_scope = {}
        for r in starts:
            for key, src in (("name", "task"), ("dataset_fingerprint", "dataset_fingerprint"),
                             ("data_contract_sha256", "data_contract_sha256"),
                             ("workspace_contract_sha256", "contract_sha256")):
                if r.get(src) is not None:
                    ledger_scope.setdefault(key, r.get(src))
        for r in scores:
            for key in ("label", "metric"):
                if r.get(key) is not None:
                    ledger_scope.setdefault(key, r.get(key))
            if r.get("split") == current_scope.get("dev_split"):
                ledger_scope.setdefault("dev_split", r.get("split"))
        ledger_scope.setdefault("held_out_split", current_scope.get("held_out_split"))

        ok, why = compatible(current_scope, ledger_scope)
        method = "contract"
        if not ok:
            # BOTH sides are built independently. The historical side comes from the
            # ledger's own sanitised genesis manifest; the current side comes from the
            # splits this run actually loaded, plus the running scorer's own manifest for
            # the task name the driver cannot invent. Nothing is copied across.
            genesis = led.get("genesis_manifest") or {}
            current_manifest = (bundle or {}).get("current_manifest") or {}
            current_facts = {
                "task": current_manifest.get("task"),
                "label": current_scope.get("label"),
                "metric": current_scope.get("metric"),
                "workspace_contract_sha256":
                    current_scope.get("workspace_contract_sha256"),
                "splits": current_scope.get("splits"),
            }
            facts = {
                "task": genesis.get("task"),
                "label": genesis.get("label"),
                "metric": genesis.get("metric"),
                "workspace_contract_sha256": next(
                    (r.get("contract_sha256") for r in starts
                     if r.get("contract_sha256")), None),
                "splits": genesis.get("splits"),
            }
            if not allow_legacy:
                entry["refused"] = ("incompatible and --evidence-allow-legacy was not "
                                    "given: " + "; ".join(why[:3]))
                reject(entry["refused"], ledger=led.get("ledger"))
                summary["ledgers"].append(entry)
                continue
            ok, legacy_why, matched = legacy_compatibility(current_facts, facts)
            method = "legacy_adapter"
            if not ok:
                entry["refused"] = "legacy adapter refused: " + "; ".join(legacy_why[:3])
                reject(entry["refused"], ledger=led.get("ledger"))
                summary["ledgers"].append(entry)
                continue
            entry["legacy_facts_matched"] = matched
        entry["compatibility_method"] = method
        summary["compatibility_method"] = method

        # ---- confirmations, recomputed from signed per-seed records
        primaries, score_conflicts, score_duplicates = _primary_by_tree(
            records, split=current_scope.get("dev_split"))
        entry["ambiguous_score_trees"] = len(score_conflicts)
        entry["deduplicated_identical_scores"] = len(score_duplicates)
        for tree, detail in sorted(score_conflicts.items()):
            reject("the same tree has disagreeing signed primaries in one evaluation "
                   "context; refusing rather than selecting one",
                   ledger=led.get("ledger"), tree=tree[:12], conflict=detail)
        confirmations = ConfirmationIndex()
        per_seed_all = [r for r in records if r.get("kind") == "confirmation"]
        opened = [r for r in records if r.get("kind") == "confirmation_protocol"]
        protocols = [r for r in records if r.get("kind") == "confirmation_result"]
        for res in protocols:
            # Scope the per-seed records to THIS confirmation's sequence window. A ledger
            # may hold several confirmations of the same tree pair -- research_trials has
            # a 5-seed run and a 20-seed run for fe985741 -- and pooling their per-seed
            # records makes seeds appear twice, which the pairing then drops. The window
            # runs from the protocol that opened this comparison to the result that
            # closed it, both of which are signed entries in the same chain.
            res_seq = res.get("seq")
            opener = max(
                (o for o in opened
                 if o.get("champion_tree") == res.get("champion_tree")
                 and o.get("comparison_tree") == res.get("comparison_tree")
                 and (o.get("seq") or -1) < (res_seq or 10 ** 9)),
                key=lambda o: o.get("seq") or -1, default=None)
            lo = (opener or {}).get("seq", -1)
            per_seed = [r for r in per_seed_all
                        if lo < (r.get("seq") or -1) < (res_seq or 10 ** 9)]
            if opener is None:
                reject("confirmation_result has no opening confirmation_protocol; its "
                       "per-seed records cannot be scoped and it is not imported",
                       ledger=led.get("ledger"), seq=res_seq)
                continue
            champ, comp = str(res.get("champion_tree") or ""), str(res.get("comparison_tree") or "")
            champ_files = _safe_read(reader, champ)
            comp_files = _safe_read(reader, comp)
            champ_id, comp_id = seedless_identity(champ_files), seedless_identity(comp_files)
            if not champ_id or not comp_id:
                reject("confirmation trees are not in the artifact store; its summary "
                       "cannot be recomputed and is not imported",
                       ledger=led.get("ledger"), seq=res.get("seq"))
                continue

            def lineage_of(tree, _c=champ_id, _p=comp_id):
                ident = seedless_identity(_safe_read(reader, tree))
                return ("champion" if ident == _c else
                        "comparison" if ident == _p else "unrelated")

            recomputed = recompute_confirmation(per_seed, lineage_of)
            agreement = compare_confirmation(res, recomputed)
            summary["confirmations_recomputed"] += 1
            derived_verdict = verdict_from(recomputed)
            if not agreement["agrees"]:
                summary["confirmation_discrepancies"].append({
                    "ledger": led.get("ledger"), "seq": res.get("seq"),
                    "champion_tree": champ[:12], "comparison_tree": comp[:12],
                    "discrepancies": agreement.get("discrepancies")
                    or [agreement.get("detail", "")],
                })
                # A summary the signed records do not support is not evidence. The
                # recomputation stands; the recorded numbers are discarded.
                if not recomputed.get("n"):
                    continue
            if derived_verdict != str(res.get("verdict") or ""):
                summary["confirmation_discrepancies"].append({
                    "ledger": led.get("ledger"), "seq": res.get("seq"),
                    "discrepancies": [f"verdict: recorded {res.get('verdict')!r}, "
                                      f"derived {derived_verdict!r}"]})
            # A confirmation measures champion VERSUS comparison, so it is evidence
            # about every mechanism that separates those two trees -- not about the
            # champion's last incremental edit. fe985741's own parent delta is the feature
            # add; its +0.001888 over the official baseline is BPR *and* the feature, and
            # attaching it only to the feature would hide the confirmation from the very
            # mechanism the reconciliation is about.
            champ_cfg, _wc = verified_config(champ, reader)
            comp_cfg, _wp = verified_config(comp, reader)
            if champ_cfg is not None and comp_cfg is not None and recomputed.get("n"):
                cum_key = mechanism_key(comp_cfg, champ_cfg, [])
                if cum_key not in ("unclassified",) and not cum_key.startswith("code:"):
                    observations.append(Observation(
                        mechanism=cum_key, tree=champ, parent_tree=comp,
                        changed_files=[], config=champ_cfg,
                        config_delta=config_delta(comp_cfg, champ_cfg),
                        feature_fields=list(champ_cfg.get("fields") or []),
                        signed_metrics={},
                        delta=recomputed["mean"], delta_versus="comparison_tree",
                        seed_confirmation={
                            "champion_tree": champ, "comparison_tree": comp,
                            "n": recomputed["n"], "mean": recomputed["mean"],
                            "sd": recomputed.get("sd"), "se": recomputed.get("se"),
                            "t": recomputed.get("t"), "wins": recomputed["wins"],
                            "verdict": derived_verdict, "ledger": led.get("ledger"),
                            "seq": res.get("seq"),
                            "method": "recomputed from signed per-seed score records"},
                        procedure_claims={
                            "note": "paired-seed comparison; the delta is the recomputed "
                                    "mean over signed per-seed scores"},
                        provenance={"ledger": led.get("ledger"),
                                    "ledger_sha256": led.get("sha256"),
                                    "seq": res.get("seq"),
                                    "compatibility_method": method,
                                    "delta_source": "recomputed paired seed mean",
                                    "scope": dict(current_scope)},
                        mechanism_label="paired-seed confirmation of the cumulative tree",
                    ))
                    entry["imported_sequences"].append(res.get("seq"))
                    entry["observations_imported"] += 1
            confirmations.add({
                "champion_tree": champ, "comparison_tree": comp,
                "comparison": res.get("comparison"),
                "n": recomputed["n"], "mean": recomputed["mean"],
                "sd": recomputed.get("sd"), "se": recomputed.get("se"),
                "t": recomputed.get("t"), "wins": recomputed["wins"],
                "verdict": derived_verdict,
                "method": "recomputed from signed per-seed score records",
            }, ledger=led.get("ledger"), seq=res.get("seq"),
                run_id=str(res.get("run_id") or ""))

        # ---- observations, one per pre-registered candidate with two signed scores
        for prop in [r for r in records if r.get("kind") == "candidate_proposal"]:
            tree, parent = str(prop.get("tree") or ""), str(prop.get("parent_tree") or "")
            if not tree or not parent:
                reject("proposal names no parent; a delta needs both sides",
                       ledger=led.get("ledger"), seq=prop.get("seq"))
                continue
            child_cfg, why_child = verified_config(tree, reader)
            parent_cfg, why_parent = verified_config(parent, reader)
            if child_cfg is None or parent_cfg is None:
                reject(f"artifact verification failed: {why_child or why_parent}",
                       ledger=led.get("ledger"), seq=prop.get("seq"), tree=tree[:12])
                continue
            if tree in score_conflicts or parent in score_conflicts:
                reject("candidate or parent has an ambiguous signed score",
                       ledger=led.get("ledger"), seq=prop.get("seq"), tree=tree[:12])
                continue
            if tree not in primaries or parent not in primaries:
                reject("no signed score record for both the candidate and its parent; "
                       "a state.json delta is not accepted as cross-run evidence",
                       ledger=led.get("ledger"), seq=prop.get("seq"), tree=tree[:12])
                continue
            changed = list(prop.get("files_changed") or [])
            key = mechanism_key(parent_cfg, child_cfg, changed)
            if not changed and key == "unclassified":
                reject("candidate changed nothing; a seed re-registration is not an "
                       "experiment", ledger=led.get("ledger"), seq=prop.get("seq"))
                continue
            if key.startswith("code:") or key == "unclassified":
                key = code_mechanism_key(changed, tree)
            delta = round(primaries[tree] - primaries[parent], 6)
            observations.append(Observation(
                mechanism=key, tree=tree, parent_tree=parent, changed_files=changed,
                config=child_cfg, config_delta=config_delta(parent_cfg, child_cfg),
                feature_fields=list(child_cfg.get("fields") or []),
                model_family=str(child_cfg.get("model") or ""),
                signed_metrics={"primary": primaries[tree]},
                delta=delta, delta_versus="parent",
                seed_confirmation=confirmations.best_for(tree),
                procedure_claims={
                    "label": str(prop.get("mechanism") or prop.get("hypothesis") or "")[:400],
                    "note": "LLM-authored; never used as a join key or graded",
                },
                provenance={"ledger": led.get("ledger"), "ledger_sha256": led.get("sha256"),
                            "seq": prop.get("seq"), "run_id": prop.get("run_id"),
                            "iteration": prop.get("iteration"),
                            "compatibility_method": method,
                            "delta_source": "signed score records",
                            "scope": dict(current_scope)},
                mechanism_label=str(prop.get("mechanism") or prop.get("hypothesis") or "")[:200],
            ))
            entry["imported_sequences"].append(prop.get("seq"))
            entry["observations_imported"] += 1
        entry["observations_rejected"] = sum(
            1 for r in rejected if r.get("ledger") == led.get("ledger"))
        summary["ledgers"].append(entry)
        summary["imported_sequences"].extend(entry["imported_sequences"])

    summary["observations_imported"] = len(observations)
    summary["observations_rejected"] = len(rejected)
    summary["rejections"] = rejected[:20]
    return observations, summary


def _safe_read(reader, tree: str):
    if not tree:
        return None
    try:
        return reader(tree)
    except Exception:                              # noqa: BLE001 - absence is a fact here
        return None


#: Caps for the summary that must fit INSIDE run_start alongside everything else it
#: carries. run_start is already the largest routine record this project writes, so the
#: evidence summary gets a small, fixed share of the 12,288-byte body.
SUMMARY_MAX_LEDGERS = 8
SUMMARY_MAX_SEQUENCES = 4
SUMMARY_MAX_REJECTIONS = 4
SUMMARY_REASON_CHARS = 160


def _digest(values) -> str:
    """A deterministic address for a list that is too long to record in full."""
    payload = json.dumps(list(values), sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _logical_name(path, root=None) -> str:
    """`runs/ledgers/x.jsonl` rather than a machine-specific absolute path.

    An absolute path is both larger and less portable: it names this laptop, not the
    evidence. The relative form is what a reader on another machine can actually follow.
    """
    text = str(path or "")
    if not text:
        return ""
    try:
        base = str(root) if root else str(Path(__file__).resolve().parents[1])
        if text.startswith(base):
            return text[len(base):].lstrip("/")
    except Exception:                              # noqa: BLE001
        pass
    # A ledger recorded on another machine will not share this root. The logical name is
    # still recoverable from the approved directory marker, which is the portable part.
    marker = "runs/ledgers/"
    idx = text.rfind(marker)
    if idx >= 0:
        return text[idx:]
    return text


def bounded_evidence_summary(summary: dict, *, root=None) -> dict:
    """Shrink an import summary to a signed-record-sized index of itself.

    Every long list becomes a count plus a sha256 digest of the full list, so the record
    stays small and a reader can still prove which sequences were imported by digesting
    the same list from the ledger. Representative samples are kept because a digest alone
    tells a human nothing.
    """
    if not summary:
        return None
    out = {
        "requested": [_logical_name(x, root) for x in (summary.get("requested") or [])
                      ][:SUMMARY_MAX_LEDGERS],
        "compatibility_method": summary.get("compatibility_method"),
        "observations_imported": summary.get("observations_imported", 0),
        "observations_rejected": summary.get("observations_rejected", 0),
        "confirmations_recomputed": summary.get("confirmations_recomputed", 0),
    }
    seqs = list(summary.get("imported_sequences") or [])
    out["imported_sequence_count"] = len(seqs)
    out["imported_sequences_sha256"] = _digest(seqs)
    out["imported_sequences_sample"] = seqs[:SUMMARY_MAX_SEQUENCES]

    reasons = {}
    for r in (summary.get("rejections") or []):
        reasons[str(r.get("reason"))[:SUMMARY_REASON_CHARS]] = \
            reasons.get(str(r.get("reason"))[:SUMMARY_REASON_CHARS], 0) + 1
    out["rejection_reason_counts"] = dict(sorted(reasons.items())[:SUMMARY_MAX_REJECTIONS])
    out["rejections_sha256"] = _digest(
        [str(r.get("reason")) for r in (summary.get("rejections") or [])])

    discrepancies = summary.get("confirmation_discrepancies") or []
    out["confirmation_discrepancy_count"] = len(discrepancies)
    out["confirmation_discrepancies_sha256"] = _digest(discrepancies)
    out["confirmation_discrepancies_sample"] = [
        {"seq": d.get("seq"), "detail": str(d.get("discrepancies"))[:SUMMARY_REASON_CHARS]}
        for d in discrepancies[:2]]

    ledgers = []
    for led in (summary.get("ledgers") or [])[:SUMMARY_MAX_LEDGERS]:
        seq_list = list(led.get("imported_sequences") or [])
        ledgers.append({
            "ledger": _logical_name(led.get("ledger"), root),
            "sha256": led.get("sha256"),
            "verified": led.get("verified"),
            "verified_entries": led.get("verified_entries"),
            "compatibility_method": led.get("compatibility_method"),
            "observations_imported": led.get("observations_imported", 0),
            "observations_rejected": led.get("observations_rejected", 0),
            "ambiguous_score_trees": led.get("ambiguous_score_trees", 0),
            "deduplicated_identical_scores": led.get("deduplicated_identical_scores", 0),
            "imported_sequence_count": len(seq_list),
            "imported_sequences_sha256": _digest(seq_list),
            "imported_sequences_sample": seq_list[:3],
            "refused": str(led.get("refused"))[:SUMMARY_REASON_CHARS]
            if led.get("refused") else None,
        })
    out["ledgers"] = ledgers
    out["ledgers_omitted"] = max(0, len(summary.get("ledgers") or []) - len(ledgers))
    out["note"] = ("bounded index of the full import; long lists are replaced by counts "
                   "plus sha256 digests of the complete list")
    return out
