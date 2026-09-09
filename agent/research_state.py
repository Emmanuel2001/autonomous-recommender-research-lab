"""The research agent's memory: champion, experiment history, failures, diagnostics.

WHY A STATE OBJECT AND NOT A PROMPT STRING
------------------------------------------
The typed driver assembled its context fresh from a dict each iteration and, measured over
34 runs, the weak tier used 3 of 16 available actions and repeated 6 of 12 proposals. A
leaderboard does not tell an agent what has already been tried and why it failed; it just
shows the best number. This holds the things that let the next decision differ from the
last one: which trees have been evaluated (so a repeat is refused for free), which ideas
FAILED and with what error (so the same syntax mistake is not made three times), and what
each experiment predicted versus what happened (so predictions can be calibrated).

BOUNDED BY CONSTRUCTION
-----------------------
`for_prompt()` returns a bounded view -- a full 50-iteration history with sources would
run to tens of thousands of tokens and push the contract itself out of the model's
attention. Everything is kept on disk; only a summary is sent.

CANDIDATE IDENTITY IS THE TREE HASH
-----------------------------------
Including config.json, so the model seed is part of identity: two candidates differing only
in seed are two candidates, and a deliberate seed replicate is not silently rejected as a
duplicate. That was a real defect in the typed path, where `seed` was in no dedup tuple and
the agent therefore could not evaluate one configuration twice.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path

#: How much history each stage is shown. Enough to see a trend, bounded enough that the
#: contract and the parent source stay in view.
PROMPT_RECENT = 12
PROMPT_FAILURES = 8
#: How many reflections the planner is shown in full. These are the reflector's own
#: conclusions, and they were WRITE-ONLY until 2026-08-29: `for_prompt()` sent the compact
#: experiment lines, the failure registry and the diagnostics, but no reflection field. So
#: roughly a third of the LLM budget was spent producing analysis that reached a file and
#: never reached the next decision. Six is enough to carry a thread without displacing the
#: contract from the model's attention.
PROMPT_REFLECTIONS = 6


@dataclass
class Experiment:
    """One hypothesis, its code, and what happened. The unit the run log is made of."""
    iteration: int
    tree: str
    parent: str | None
    hypothesis: str = ""
    stage: str = ""
    #: research lane (see agent.research_agent.LANES). Empty for a single-policy run that
    #: predates lanes, which is why lane_stats() buckets those as "unspecified".
    lane: str = ""
    #: how many tournament candidates competed in this lane for this iteration
    considered_in_lane: int = 0
    #: total tournament size for this iteration, 0 for the single-hypothesis policy
    tournament_size: int = 0
    mechanism: str = ""
    files_changed: list = field(default_factory=list)
    diff_hash: str = ""
    diff_bytes: int = 0
    expected_direction: str = ""
    expected_magnitude: float | None = None
    #: signed metrics from the scorer -- the only numbers that decide anything
    signed: dict | None = None
    #: harness-computed from the candidate's returned score vectors, NOT self-reported
    train_primary: float | None = None
    dev_primary: float | None = None
    gap: float | None = None
    outcome: str = ""                      # promoted | rejected | failed | refused
    error_type: str = ""
    error: str = ""
    repairs: int = 0
    wall_seconds: float | None = None
    #: Harness-resolved numerical target plus any candidate-reported execution evidence.
    #: Resolution is measured outside the candidate, but arbitrary generated code can ignore
    #: the supplied target; it must never be presented as proof that MPS/CUDA was used.
    execution: dict | None = None
    delta_vs_parent: float | None = None
    delta_vs_champion: float | None = None
    reflection: dict | None = None

    def to_record(self) -> dict:
        return asdict(self)

    def one_line(self) -> str:
        """Compact history line for a prompt. Deliberately includes the FAILURES."""
        head = f"#{self.iteration} [{self.outcome or 'pending'}] {self.stage or '?'}"
        if self.signed:
            p = self.signed.get("metrics", {}).get("primary")
            got = f"primary {p:.5f}" if isinstance(p, (int, float)) else "unscored"
            if self.delta_vs_champion is not None:
                got += f" ({self.delta_vs_champion:+.5f} vs champion)"
            if self.gap is not None:
                got += f", train-dev gap {self.gap:+.4f}"
        elif self.error_type:
            got = f"FAILED {self.error_type}: {self.error[:120]}"
        else:
            got = "no result"
        pred = ""
        if self.expected_magnitude is not None:
            pred = f" predicted {self.expected_magnitude:+.5f}"
        files = ",".join(self.files_changed) if self.files_changed else "-"
        return f"{head}: {self.hypothesis[:150]} | changed {files} |{pred} -> {got}"


class ResearchState:
    """Everything the agent knows. Serialisable, resumable, and bounded for prompts."""

    def __init__(self, task: dict, budget: dict | None = None):
        self.task = dict(task)
        self.budget = dict(budget or {})
        self.champion: dict | None = None
        self.experiments: list[Experiment] = []
        #: mechanism -> structured belief. Accumulated across iterations so the next
        #: tournament reasons over conclusions rather than re-reading prose. A belief keyed
        #: by MECHANISM (not by candidate) is what lets "high-cardinality crosses are
        #: harmful from a pointwise baseline" survive the candidate that established it.
        #:
        #: These are REFLECTOR-AUTHORED interpretations of scorer-signed numbers. The delta is
        #: signed; the belief about what it means is not. See update_beliefs().
        self.beliefs: dict[str, dict] = {}
        #: Reflections imported from a compatible earlier run. Kept separate so a report
        #: never mistakes them for experiments executed in this run.
        self.imported_reflections: list[dict] = []
        self.memory_import: dict | None = None
        self.diagnostics: dict = {}
        #: idea -> list of failures, so the same mistake is visible before it repeats
        self.failures: dict[str, list[dict]] = {}
        self.seen_trees: dict[str, int] = {}          # tree -> iteration first evaluated
        self.manual_interventions: list[dict] = []
        self.started_epoch = time.time()

    # -------------------------------------------------------------------- writes
    #: A tree is a full sha256 content address. Anything else -- a synthetic key for an
    #: iteration that was refused before it ever produced a tree -- must not occupy a slot
    #: in the dedup map, which is consulted to decide whether a candidate has been fitted.
    @staticmethod
    def is_tree(tree) -> bool:
        return (isinstance(tree, str) and len(tree) == 64
                and all(c in "0123456789abcdef" for c in tree))

    def note_seen(self, tree: str, iteration: int) -> None:
        if not self.is_tree(tree):
            return
        self.seen_trees.setdefault(tree, iteration)

    def already_seen(self, tree: str) -> int | None:
        """The iteration that first evaluated this exact tree, or None."""
        return self.seen_trees.get(tree)

    def add(self, exp: Experiment) -> None:
        self.experiments.append(exp)
        self.note_seen(exp.tree, exp.iteration)
        if exp.error_type:
            self.failures.setdefault(exp.error_type, []).append(
                {"iteration": exp.iteration, "stage": exp.stage,
                 "hypothesis": exp.hypothesis[:200], "error": exp.error[:400],
                 "files": list(exp.files_changed)})

    def promote(self, exp: Experiment, tree_files: dict | None = None) -> None:
        """Advance the champion. Called ONLY with a signed improvement -- the driver
        enforces that; this records it."""
        m = (exp.signed or {}).get("metrics", {})
        self.champion = {"tree": exp.tree, "iteration": exp.iteration,
                         "primary": m.get("primary"), "metrics": dict(m),
                         "hypothesis": exp.hypothesis,
                         "train_primary": exp.train_primary,
                         "dev_primary": exp.dev_primary, "gap": exp.gap}
        exp.outcome = "promoted"

    # ------------------------------------------------------------------- beliefs
    def update_beliefs(self, exp: "Experiment", updates: list | None) -> list[str]:
        """Merge the reflector's belief updates. Returns the mechanisms touched.

        WHAT IS SIGNED HERE AND WHAT IS NOT -- the distinction matters
        -------------------------------------------------------------
        The METRIC behind a belief is scorer-signed: the delta an experiment produced came
        from a separate process holding the labels and the signing key, and it cannot be
        forged by the loop.

        The INTERPRETATION is not. `belief`, `llm_confidence`, and whether a result
        *supports* or *contradicts* a mechanism are all authored by the reflector, which is
        an LLM reading that signed number and forming an opinion about what it means. A
        belief is therefore LLM-authored commentary attached to signed evidence, and it
        must never be presented as signed evidence itself.

        What this method contributes is only bookkeeping the loop can vouch for: which
        iteration an update came from, and which files that iteration actually changed. The
        `supports` flag that decides which list an iteration lands in is the reflector's
        judgement, taken as given.
        """
        touched = []
        for u in (updates or []):
            if not isinstance(u, dict):
                continue
            mech = str(u.get("mechanism") or "").strip()
            if not mech:
                continue
            b = self.beliefs.setdefault(mech, {
                "mechanism": mech, "lane": u.get("lane"), "belief": "",
                "llm_confidence": None,
                "llm_confidence_note": "reflector-authored and uncalibrated",
                "evidence_grade": "unreplicated", "supporting_iterations": [],
                "contradicting_iterations": [], "affected_files": [],
                "effect_range": [None, None], "unresolved": "",
                "retry_only_if": "", "last_updated_iteration": None,
                "scope": {
                    "data_contract_sha256": self.task.get("data_contract_sha256"),
                    "workspace_contract_sha256":
                        self.task.get("workspace_contract_sha256"),
                },
                "source_run_id": self.task.get("run_id")})
            b["lane"] = u.get("lane") or b["lane"]
            b["belief"] = str(u.get("belief") or b["belief"])[:400]
            confidence = u.get("llm_confidence", u.get("confidence"))
            if isinstance(confidence, (int, float)):
                b["llm_confidence"] = round(float(confidence), 3)
            key = ("supporting_iterations" if u.get("supports", True)
                   else "contradicting_iterations")
            if exp.iteration not in b[key]:
                b[key].append(exp.iteration)
            for f in (u.get("affected_files") or exp.files_changed or []):
                if f not in b["affected_files"]:
                    b["affected_files"].append(f)
            lo, hi = u.get("effect_low"), u.get("effect_high")
            if isinstance(lo, (int, float)):
                b["effect_range"][0] = round(float(lo), 6)
            if isinstance(hi, (int, float)):
                b["effect_range"][1] = round(float(hi), 6)
            b["unresolved"] = str(u.get("unresolved") or b["unresolved"])[:300]
            b["retry_only_if"] = str(u.get("retry_only_if") or b["retry_only_if"])[:300]
            b["last_updated_iteration"] = exp.iteration
            b["evidence_grade"] = self.evidence_grade(b)
            touched.append(mech)
        return touched

    @staticmethod
    def evidence_grade(belief: dict) -> str:
        """Categorical strength from recorded outcomes, never the LLM's confidence."""
        if belief.get("seed_confirmation"):
            return "seed_confirmed"
        contradicting = len(set(belief.get("contradicting_iterations") or []))
        supporting = len(set(belief.get("supporting_iterations") or []))
        if contradicting:
            return "contradicted"
        if supporting >= 2:
            return "replicated"
        if supporting == 1:
            return "single_signed_result"
        return "unreplicated"

    @staticmethod
    def compatibility(task: dict) -> dict:
        """Facts that must match before conclusions move between runs."""
        return {key: task.get(key) for key in (
            "name", "metric", "label", "dev_split", "held_out_split",
            "dataset_fingerprint", "data_contract_sha256",
            "workspace_contract_sha256")}

    def import_memory(self, prior: "ResearchState", *, mode: str,
                      source: str) -> dict:
        """Import only contract-scoped conclusions from a compatible prior run."""
        if mode not in ("continue", "import"):
            raise ValueError("memory mode must be continue or import")
        current = self.compatibility(self.task)
        previous = self.compatibility(prior.task)
        required = ("dataset_fingerprint", "data_contract_sha256",
                    "workspace_contract_sha256")
        if any(not current.get(k) or not previous.get(k) for k in required):
            raise ValueError("memory import requires versioned data and workspace contracts")
        mismatches = [k for k in current if current.get(k) != previous.get(k)]
        if mismatches:
            raise ValueError(f"incompatible research memory fields: {mismatches}")

        imported, skipped = 0, 0
        for mechanism, raw in prior.beliefs.items():
            belief = deepcopy(raw)
            scope = belief.get("scope") or {}
            if scope.get("data_contract_sha256") != current["data_contract_sha256"]:
                skipped += 1
                continue
            belief["imported"] = True
            belief["imported_from"] = str(source)
            belief["evidence_grade"] = self.evidence_grade(belief)
            if "llm_confidence" not in belief and isinstance(
                    belief.get("confidence"), (int, float)):
                belief["llm_confidence"] = belief.pop("confidence")
            belief["llm_confidence_note"] = "reflector-authored and uncalibrated"
            self.beliefs[str(mechanism)] = belief
            imported += 1

        self.imported_reflections = [
            {**deepcopy(item), "imported": True, "imported_from": str(source)}
            for item in prior.recent_reflections()[-PROMPT_REFLECTIONS:]]
        if mode == "continue":
            self.seen_trees.update(prior.seen_trees)
            self.failures = deepcopy(prior.failures)
        self.memory_import = {"mode": mode, "source": str(source),
                              "beliefs_imported": imported,
                              "beliefs_skipped_unscoped": skipped}
        return dict(self.memory_import)

    def beliefs_for_prompt(self, n: int = 12) -> list[dict]:
        """Most-recently-updated beliefs first, bounded."""
        ordered = sorted(self.beliefs.values(),
                         key=lambda b: (b.get("last_updated_iteration") or -1),
                         reverse=True)
        out = []
        for raw in ordered[:n]:
            b = dict(raw)
            if "llm_confidence" not in b and isinstance(b.get("confidence"), (int, float)):
                b["llm_confidence"] = b.pop("confidence")
            b["llm_confidence_note"] = "reflector-authored and uncalibrated"
            b["evidence_grade"] = self.evidence_grade(b)
            out.append(b)
        return out

    # --------------------------------------------------------------- lane stats
    def lane_stats(self) -> dict:
        """Per-lane research distribution, plus per-file edit counts.

        This is the number that showed the old policy was narrow: proposals considered
        versus selected versus promoted, by lane. `considered` is only non-zero once a
        tournament runs, so a single-hypothesis run reports selected == considered.
        """
        out: dict = {}
        for e in self.experiments:
            lane = e.lane or "unspecified"
            a = out.setdefault(lane, {
                "considered": 0, "selected": 0, "executed": 0, "promoted": 0,
                "rejected": 0, "failed": 0, "refused": 0, "repairs": 0,
                "deltas": []})
            a["selected"] += 1
            a["considered"] += max(1, int(e.considered_in_lane or 1))
            if e.signed:
                a["executed"] += 1
                if e.delta_vs_champion is not None:
                    a["deltas"].append(e.delta_vs_champion)
            if e.outcome == "promoted":
                a["promoted"] += 1
            elif e.outcome == "rejected":
                a["rejected"] += 1
            elif e.outcome == "failed":
                a["failed"] += 1
            elif e.outcome == "refused":
                a["refused"] += 1
            a["repairs"] += int(e.repairs or 0)
        for a in out.values():
            d = a.pop("deltas")
            a["mean_signed_delta"] = (round(sum(d) / len(d), 6) if d else None)
        files: dict = {}
        for e in self.experiments:
            for f in (e.files_changed or []):
                files[f] = files.get(f, 0) + 1
        return {"by_lane": out, "file_edits": dict(sorted(files.items())),
                "model_py_edits": files.get("model.py", 0),
                "train_py_edits": files.get("train.py", 0),
                "features_py_edits": files.get("features.py", 0),
                "config_json_edits": files.get("config.json", 0)}

    def note_intervention(self, what: str, why: str) -> None:
        """A human touched the run. Recorded immediately, because the rubric asks how many
        interventions a run needed and an unrecorded one is indistinguishable from none."""
        self.manual_interventions.append(
            {"epoch": time.time(), "what": what, "why": why})

    # --------------------------------------------------------------------- reads
    @property
    def champion_primary(self) -> float | None:
        return None if self.champion is None else self.champion.get("primary")

    def calibration(self) -> dict:
        """Predicted versus actual, across scored experiments. The agent's own track
        record: an agent that consistently over-predicts should be told so."""
        pairs = [(e.expected_magnitude, e.delta_vs_parent) for e in self.experiments
                 if e.expected_magnitude is not None and e.delta_vs_parent is not None]
        if not pairs:
            return {"n": 0}
        errs = [abs(p - a) for p, a in pairs]
        over = sum(1 for p, a in pairs if p > a)
        return {"n": len(pairs), "mean_abs_error": round(sum(errs) / len(errs), 6),
                "over_predicted": over, "under_predicted": len(pairs) - over,
                "directional_hits": sum(1 for p, a in pairs
                                        if (p > 0) == (a > 0))}

    def recent_reflections(self, n: int = PROMPT_REFLECTIONS) -> list[dict]:
        """The last `n` reflections, bounded per field.

        Includes the CHAMPION's reflection even if it has scrolled out of the window: the
        reasoning behind the configuration being built on is the single most relevant piece
        of analysis the planner can have.
        """
        picked = [e for e in self.experiments if e.reflection][-n:]
        champ_it = (self.champion or {}).get("iteration")
        if champ_it is not None and not any(e.iteration == champ_it for e in picked):
            champ = next((e for e in self.experiments
                          if e.iteration == champ_it and e.reflection), None)
            if champ is not None:
                picked = [champ] + picked[1:] if picked else [champ]
        out = []
        for e in picked:
            r = e.reflection or {}
            out.append({
                "iteration": e.iteration, "outcome": e.outcome,
                "is_current_champion": e.iteration == champ_it,
                "mechanism_supported": r.get("mechanism_supported"),
                "why": str(r.get("why") or "")[:420],
                "remaining_uncertainty": str(r.get("remaining_uncertainty") or "")[:260],
                "next_question": str(r.get("next_question") or "")[:260],
            })
        imported = self.imported_reflections[-max(0, n - len(out)):]
        return imported + out

    def for_prompt(self) -> dict:
        """The bounded view every stage is built from."""
        scored = [e for e in self.experiments if e.signed]
        return {
            "task": self.task,
            "budget": self.budget,
            "champion": self.champion,
            "iterations_done": len(self.experiments),
            "scored": len(scored),
            "failed": sum(1 for e in self.experiments if e.error_type),
            "recent_experiments": [e.one_line() for e in self.experiments[-PROMPT_RECENT:]],
            # The reflector's own conclusions, so the next plan can build on them or say
            # why it is departing from them. Without this the loop reflects into a void.
            "reflections": self.recent_reflections(),
            # Structured conclusions the next tournament reasons over, and the lane
            # distribution so it can see which directions it has been neglecting.
            "beliefs": self.beliefs_for_prompt(),
            "lane_stats": self.lane_stats(),
            "failure_registry": {
                k: v[-PROMPT_FAILURES:] for k, v in sorted(self.failures.items())},
            "diagnostics": self.diagnostics,
            "calibration": self.calibration(),
            "trees_already_evaluated": len(self.seen_trees),
            "manual_interventions": len(self.manual_interventions),
        }

    # ---------------------------------------------------------------- persistence
    def to_json(self) -> dict:
        return {"task": self.task, "budget": self.budget, "champion": self.champion,
                "beliefs": self.beliefs,
                "imported_reflections": self.imported_reflections,
                "memory_import": self.memory_import,
                "experiments": [e.to_record() for e in self.experiments],
                "diagnostics": self.diagnostics, "failures": self.failures,
                "seen_trees": self.seen_trees,
                "manual_interventions": self.manual_interventions,
                "started_epoch": self.started_epoch}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True,
                                  allow_nan=False, default=str) + "\n")
        tmp.replace(path)                    # atomic: a crash cannot truncate the state

    @classmethod
    def load(cls, path: Path) -> "ResearchState":
        d = json.loads(Path(path).read_text())
        s = cls(d.get("task", {}), d.get("budget", {}))
        s.champion = d.get("champion")
        allowed = {f.name for f in dataclass_fields(Experiment)}
        s.experiments = [Experiment(**{k: v for k, v in e.items() if k in allowed})
                         for e in d.get("experiments", []) if isinstance(e, dict)]
        s.beliefs = d.get("beliefs", {})
        for belief in s.beliefs.values():
            if "llm_confidence" not in belief and isinstance(
                    belief.get("confidence"), (int, float)):
                belief["llm_confidence"] = belief.pop("confidence")
            belief["llm_confidence_note"] = "reflector-authored and uncalibrated"
            belief["evidence_grade"] = s.evidence_grade(belief)
        s.imported_reflections = d.get("imported_reflections", [])
        s.memory_import = d.get("memory_import")
        s.diagnostics = d.get("diagnostics", {})
        s.failures = d.get("failures", {})
        s.seen_trees = d.get("seen_trees", {})
        s.manual_interventions = d.get("manual_interventions", [])
        s.started_epoch = d.get("started_epoch", time.time())
        return s
