"""The code-first autonomous research loop.

Sits BESIDE agent/official_driver.py rather than replacing it. The typed driver stays as a
control arm -- a measured comparison between choosing from a menu of hyperparameters and
writing the code -- and, more practically, the graded submission is produced through that
path and this one must not be able to disturb it.

ONE ITERATION
-------------
    plan        the agent commits to a hypothesis, a mechanism, and a PREDICTED magnitude
    code        the agent returns whole files; the harness applies them to the parent tree
    diff        computed by the harness from the two trees, never taken from the model
    register    hypothesis + hashes appended to the signed ledger. THIS MUST SUCCEED.
    fit         the tree runs sandboxed; failures come back as data
    repair      on failure the error goes back to the coder, bounded, every diff recorded
    score       dev predictions go to the scorer; the signed number is the only one that
                decides anything
    reflect     the agent says whether its mechanism was supported
    promote     champion advances only on a signed dev improvement

WHY REGISTRATION IS STRICT HERE
------------------------------
The typed driver's `rec()` is fire-and-forget: a refused record increments a counter and
the run continues. Combined with a record cap (12,288 bytes) below the size of a legal
candidate (24,000), that produced a specific hazard -- a scored experiment with no
pre-registered hypothesis, which is exactly the evidence a pre-registration is for. So here
the proposal record is checked, and a candidate whose registration fails is ABANDONED
rather than run. Nothing is ever executed unrecorded.

The source itself does not go in the ledger; hashes and a bounded diff do, and the bytes
live in the content-addressed store (core/artifacts.py). `verify()` recomputes the address
from the stored bytes, so the signed chain covers the source without carrying it.

THE CHAMPION CANNOT REGRESS
---------------------------
It advances only on a signed improvement, so the worst case for this loop is a run that
spends its budget and leaves the incumbent standing. Given that the one previous
free-rein code generation this project measured scored WORSE than the numpy baseline
(-0.019), that property is doing real work and is not decoration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from agent import candidate as CAND            # noqa: E402
from agent.research_state import Experiment, ResearchState   # noqa: E402
from core import artifacts as AR               # noqa: E402
from core import budgets as BUD                # noqa: E402
from core import execution as EXEC             # noqa: E402
# OFFICIAL_TEST is deliberately NOT imported here. The code-first driver has no legitimate
# use for any test figure, and not importing it makes that checkable by grep rather than by
# reading every line.
from core.metrics import OFFICIAL_EPSILON, OFFICIAL_N  # noqa: E402

#: Repair attempts per hypothesis. Bounded because an agent that cannot fix its own code in
#: three tries is not converging on a fix, and each attempt costs a fit.
MAX_REPAIRS = 3

#: Leave room in a 12,288-byte record for everything that is NOT the diff.
INLINE_DIFF_BUDGET = 6000


# --------------------------------------------------------------------------- helpers
def validate_tournament(out: dict, size: int) -> tuple[list, dict]:
    """Normalise a tournament and decide whether its coverage is ACCEPTABLE.

    Coverage is now enforced, not merely reported. The rules, in the order a reader would
    check them:

      * at least `MIN_TOURNAMENT_SIZE` (six) candidates -- one per lane is the floor below
        which "tournament" means nothing;
      * every one of the six lanes has at least one candidate;
      * `model_architecture` and `sampling_training` are present. They are named separately
        even though the previous rule implies them, because they are the two this project
        historically neglected and the error message should say so;
      * a SECOND candidate in a lane only once all six lanes are covered -- otherwise a
        planner can satisfy the count while still ignoring a direction.

    A failing set is repairable: `problems` is written for the planner to read. It is never
    silently accepted, and the caller abandons the iteration rather than spend an evaluation
    on an unbalanced set.
    """
    from agent.research_agent import LANES, MIN_TOURNAMENT_SIZE, REQUIRED_LANES
    cands, seen = [], set()
    for c in (out.get("candidates") or []):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip() or f"C{len(cands) + 1}"
        if cid in seen:
            cid = f"{cid}_{len(cands) + 1}"
        seen.add(cid)
        lane = str(c.get("lane") or "").strip()
        cands.append({**c, "id": cid,
                      "lane": lane if lane in LANES else "unspecified"})

    per_lane = {l: sum(1 for c in cands if c["lane"] == l) for l in LANES}
    unknown_lane = [c["id"] for c in cands if c["lane"] == "unspecified"]
    missing = [l for l, n in per_lane.items() if n == 0]
    missing_required = [l for l in REQUIRED_LANES if per_lane.get(l, 0) == 0]
    dup_lanes = [l for l, n in per_lane.items() if n > 1]

    problems = []
    if len(cands) < MIN_TOURNAMENT_SIZE:
        problems.append(
            f"only {len(cands)} candidates; the minimum is {MIN_TOURNAMENT_SIZE}, one per "
            f"research lane")
    if missing:
        problems.append(f"no candidate in these lanes: {missing}")
    if missing_required:
        problems.append(
            f"the required lanes {missing_required} are empty. These two are mandatory: "
            f"model_architecture rewrites model.py, sampling_training rewrites train.py")
    if dup_lanes and missing:
        problems.append(
            f"lanes {dup_lanes} have more than one candidate while {missing} have none. A "
            f"second candidate in a lane is only allowed once every lane is covered")
    if unknown_lane:
        problems.append(
            f"candidates {unknown_lane} declare a lane outside {list(LANES)}")

    return cands, {
        "size_requested": size, "size_returned": len(cands),
        "min_size": MIN_TOURNAMENT_SIZE,
        "lanes_present": sorted({c["lane"] for c in cands}),
        "per_lane": per_lane,
        "required_lanes": list(REQUIRED_LANES),
        "missing_lanes": missing,
        "missing_required_lanes": missing_required,
        "duplicate_lanes": dup_lanes,
        "unknown_lane_candidates": unknown_lane,
        "problems": problems,
        "lane_coverage_ok": not problems,
        "coverage_note": str(out.get("lane_coverage_note") or "")[:400],
    }


def validate_judge(cands: list, verdict: dict) -> list:
    """Return the list of problems with a judge response. Empty means acceptable.

    Checked exactly, because every one of these has a silent-failure mode. A ranking that
    omits a candidate lets the judge duck a direction without saying so; an unknown id means
    it ranked something that does not exist; non-consecutive ranks make "rank 1" meaningless;
    and an architecture verdict keyed to the wrong id reads as coverage while addressing
    nothing.
    """
    ids = [c["id"] for c in cands]
    idset = set(ids)
    arch = [c["id"] for c in cands if c.get("lane") == "model_architecture"]
    problems = []

    sel = str(verdict.get("selected_id") or "").strip()
    if sel not in idset:
        problems.append(f"selected_id {sel!r} is not a candidate id; expected one of {ids}")

    ranking = [r for r in (verdict.get("ranking") or []) if isinstance(r, dict)]
    rids = [str(r.get("id") or "").strip() for r in ranking]
    unknown = sorted(set(rids) - idset)
    missing = sorted(idset - set(rids))
    dupes = sorted({i for i in rids if rids.count(i) > 1})
    if unknown:
        problems.append(f"ranking contains ids that are not candidates: {unknown}")
    if missing:
        problems.append(f"ranking omits these candidates: {missing}. Rank every one")
    if dupes:
        problems.append(f"ranking lists these candidates more than once: {dupes}")
    ranks = [r.get("rank") for r in ranking]
    if not all(isinstance(x, int) for x in ranks):
        problems.append("every ranking entry needs an integer rank")
    elif sorted(ranks) != list(range(1, len(ranking) + 1)):
        problems.append(
            f"ranks must be unique and consecutive from 1 to {len(ranking)}; got "
            f"{sorted(ranks)}")

    avs = [v for v in (verdict.get("architecture_verdicts") or []) if isinstance(v, dict)]
    av_ids = [str(v.get("candidate_id") or "").strip() for v in avs]
    av_unknown = sorted(set(av_ids) - idset)
    av_missing = sorted(set(arch) - set(av_ids))
    av_nonarch = sorted(set(av_ids) & idset - set(arch))
    if av_unknown:
        problems.append(f"architecture_verdicts name unknown ids: {av_unknown}")
    if av_missing:
        problems.append(
            f"no architecture verdict for {av_missing}. Every model_architecture candidate "
            f"needs one, selected or rejected, with a reason")
    if av_nonarch:
        problems.append(
            f"architecture_verdicts name non-architecture candidates: {av_nonarch}")
    for v in avs:
        if v.get("verdict") not in ("selected", "rejected"):
            problems.append(
                f"verdict for {v.get('candidate_id')!r} must be 'selected' or 'rejected'")
    if sel in arch:
        chosen = [v for v in avs if v.get("candidate_id") == sel]
        if chosen and chosen[0].get("verdict") != "selected":
            problems.append(
                f"{sel} was selected but its architecture verdict says "
                f"{chosen[0].get('verdict')!r}")
    return problems


#: lane -> the `stage` label the older proposal record and the coder prompt already use, so
#: a tournament run's ledger entries stay readable next to a single-policy run's.
LANE_TO_STAGE = {
    "feature_preprocessing": "features", "objective_loss": "training",
    "sampling_training": "training", "optimization_regularization": "config",
    "model_architecture": "model", "ensemble_calibration": "evaluation",
}


#: A summary record must fit inside core.scorer.MAX_RECORD_BYTES (12288). Each summary was
#: capped at 1800 chars but the COUNT was not, so seven or more in one call pushed the body
#: past the cap -- and an oversized record is REFUSED, which the caller used to discard by
#: ignoring rec()'s return value. The console would then be silently missing the very
#: decisions it exists to show. Module-level and pure so the bound is testable; the previous
#: version lived in a closure, which is why the suite claimed "bounded" without checking it.
THOUGHT_TOTAL_BUDGET = 6000
THOUGHT_MAX_ITEMS = 6
THOUGHT_ITEM_CHARS = 1800


def bound_thought_summaries(raw) -> tuple[list[str], int]:
    """Return (summaries that fit, how many were dropped)."""
    # isinstance, not str(): str(None) is "None", which is truthy, so a provider
    # returning a null in this list recorded the literal text "None" as a summary.
    items = [s.strip() for s in (raw or []) if isinstance(s, str) and s.strip()]
    kept, used, dropped = [], 0, 0
    for item in items:
        item = item[:THOUGHT_ITEM_CHARS]
        if len(kept) >= THOUGHT_MAX_ITEMS or used + len(item) > THOUGHT_TOTAL_BUDGET:
            dropped += 1
            continue
        kept.append(item)
        used += len(item)
    return kept, dropped


def candidate_to_plan(cand: dict) -> dict:
    """The selected tournament candidate, in the shape the coder and recorder expect."""
    return {
        "hypothesis": cand.get("hypothesis", ""),
        "mechanism": cand.get("mechanism", ""),
        "evidence": cand.get("evidence", ""),
        "stage": LANE_TO_STAGE.get(cand.get("lane"), "config"),
        "lane": cand.get("lane", "unspecified"),
        "expected_direction": cand.get("expected_direction", "unclear"),
        "expected_magnitude": cand.get("expected_magnitude"),
        "parent": cand.get("parent", "champion"),
        "files_to_change": cand.get("files_to_change") or [],
        "risk": cand.get("implementation_risk", ""),
        "fallback": cand.get("failure_interpretation", ""),
        "success_criteria": cand.get("success_criteria", ""),
        "information_value": cand.get("information_value", ""),
        "novelty_vs_history": cand.get("novelty_vs_history", ""),
        "leakage_risk": cand.get("leakage_risk", ""),
        "_candidate_id": cand.get("id"),
    }


def select_from_tournament(cands: list, verdict: dict) -> tuple[dict, str]:
    """The judge's pick, or the top-ranked candidate if the id does not resolve."""
    want = str(verdict.get("selected_id") or "").strip()
    by_id = {c["id"]: c for c in cands}
    if want in by_id:
        return by_id[want], "judge_selected_id"
    ranked = sorted((r for r in (verdict.get("ranking") or [])
                     if isinstance(r, dict) and r.get("id") in by_id),
                    key=lambda r: r.get("rank", 10 ** 6))
    if ranked:
        return by_id[ranked[0]["id"]], "fell_back_to_top_rank"
    return cands[0], "fell_back_to_first_candidate"


def merge_tree(parent: dict, edits: list) -> tuple[dict, list]:
    """Apply whole-file replacements to a parent tree. Returns (child, changed_paths).

    A file the coder omits is INHERITED unchanged. An edit whose content already equals
    the parent's is not counted as changed, so `files_changed` in the run log reflects
    what actually differs rather than what was resent.
    """
    child = dict(parent)
    changed = []
    for e in edits or []:
        path = AR.safe_relpath(str(e["path"]))
        content = e.get("content")
        if not isinstance(content, str):
            raise CAND.CandidateError(
                f"{path}: 'content' must be a string, got {type(content).__name__}")
        if AR.canon_text(content) != AR.canon_text(parent.get(path, "")):
            changed.append(path)
        child[path] = content
    return child, sorted(changed)


def harness_metrics(splits, spec, run_dir: Path) -> dict:
    """Train and dev metrics, computed HERE with the task-owned metric.

    Not self-reported. The candidate returns score vectors; the labels for train and dev
    are in this process; so the train/dev gap -- the thing this project previously could
    not see at all -- is a harness measurement.
    """
    run_dir = Path(run_dir)
    idx = np.load(run_dir / "train_subset_index.npy")
    sub = np.load(run_dir / "train_subset_scores.npy")
    tr = splits["train"]
    group_name = spec.metric.group_by
    tg = getattr(tr, group_name)[idx] if group_name else None
    tm = spec.metric.fn(tg, tr.label[idx], sub)
    dv = np.load(run_dir / "dev_scores.npy")
    va = splits[spec.split.dev_split]
    vg = getattr(va, group_name) if group_name else None
    dm = spec.metric.fn(vg, va.label, dv)
    pk = spec.metric.primary_key
    # The state/report schema calls the decision value "primary" for every task. Preserve
    # the task's native key as well, so a global-AUC task carries both auc and primary.
    tm, dm = dict(tm), dict(dm)
    tm.setdefault("primary", float(tm[pk]))
    dm.setdefault("primary", float(dm[pk]))
    return {"train": tm, "dev": dm,
            "gap": round(float(tm[pk]) - float(dm[pk]), 6),
            "train_rows": int(idx.size), "train_users": int(tm.get("users", 0))}


def proposal_body(*, run_id: str, iteration: int, attempt: int, plan: dict,
                  tree: str, parent: str | None, diff: str, static: dict,
                  artifact_dir: str) -> dict:
    """The pre-registration record. Hashes always; the diff inline only if it fits.

    A record refused for being oversized is the failure mode this shape exists to avoid:
    the cap is real, a legal candidate can exceed it, and the honest response is to record
    an address rather than to silently skip the registration.
    """
    body = {
        "what": "candidate_proposal", "run_id": run_id, "iteration": iteration,
        "attempt": attempt,
        "tree": tree, "parent_tree": parent, "artifact_dir": artifact_dir,
        "contract_sha256": static.get("contract_sha256"),
        "tree_bytes": static.get("bytes"),
        "files": static.get("files"),
        "diff_hash": AR.diff_hash(diff), "diff_bytes": len(diff.encode("utf-8")),
        # The pre-registered claim. Recorded at a sequence number BELOW the score that
        # tests it, so it cannot be retrofitted once the result is known.
        "hypothesis": str(plan.get("hypothesis", ""))[:600],
        "mechanism": str(plan.get("mechanism", ""))[:800],
        "evidence": str(plan.get("evidence", ""))[:800],
        "stage": plan.get("stage"),
        "expected_direction": plan.get("expected_direction"),
        "expected_magnitude": plan.get("expected_magnitude"),
        "files_changed": plan.get("_files_changed"),
        "config": static.get("config"),
    }
    if body["diff_bytes"] <= INLINE_DIFF_BUDGET:
        body["diff"] = diff
    else:
        body["diff_omitted"] = (
            f"{body['diff_bytes']:,} bytes exceeds the {INLINE_DIFF_BUDGET:,}-byte inline "
            f"budget; the full diff is reproducible from parent_tree -> tree via "
            f"core.artifacts.tree_diff")
    return body


# ------------------------------------------------------------------------- the loop
def research_loop(*, agent, splits, edges, spec, state: ResearchState, budget: BUD.Budget,
                  run_id: str, rec, score, run_root: Path, backend: str,
                  max_seconds: float, convergence: str = "official",
                  max_repairs: int = MAX_REPAIRS, verbose: bool = True,
                  seed_files: dict | None = None, seed_note: str = "",
                  policy: str = "tournament", tournament_size: int = 6,
                  workspace: Path | None = None,
                  max_coverage_repairs: int = 1,
                  execution_profile=None,
                  task_spec_path: str | Path | None = None) -> dict:
    """Run the loop. `rec` and `score` are injected so this is testable against a fake
    scorer as well as the real one.

        rec(kind, body)  -> (ok: bool, entry_or_error)
        score(path, tree)-> (ok: bool, payload_or_error)
    """
    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def record_public_thoughts(role: str, usage: dict, iteration: int, stage: str) -> None:
        """Record provider-authored summaries when available, never hidden reasoning."""
        summaries, dropped = bound_thought_summaries(usage.get("thought_summaries"))
        if not summaries:
            return
        body = {"run_id": run_id, "iteration": iteration,
                "what": "llm_thought_summary", "role": role, "stage": stage,
                "model": usage.get("model"), "summaries": summaries,
                "thinking_tokens": usage.get("thinking_tokens", 0),
                "note": "provider-generated public summary; not chain-of-thought"}
        if dropped:
            body["summaries_omitted_for_record_size"] = dropped
        ok, entry = rec("event", body)
        if not ok:
            # Not fatal -- summaries are commentary, not evidence -- but never silent.
            say(f"       note: thought-summary record refused ({entry}); "
                f"{len(summaries)} summarie(s) not recorded")

    # Convergence and the reference point belong to the task contract.  The official
    # constants remain the backwards-compatible default for historical callers, but a
    # portable task must never silently inherit KuaiRand's stopping rule or baseline.
    conv_epsilon = float(spec.convergence.get("epsilon", OFFICIAL_EPSILON))
    conv_n = int(spec.convergence.get("N", OFFICIAL_N))
    baseline_valid = spec.baseline.get(spec.split.dev_split)

    stats = {"planned": 0, "refused": 0, "duplicate": 0, "registered": 0,
             "evaluated": 0, "failed": 0, "repaired": 0, "promoted": 0,
             "unregistered_aborts": 0, "infra_events": 0, "metric_disagreements": 0,
             "tournaments": 0, "candidates_considered": 0,
             "lane_coverage_violations": 0, "architecture_considered": 0,
             "architecture_selected": 0, "beliefs_updated": 0,
             "coverage_repairs": 0, "coverage_aborts": 0,
             "judge_violations": 0, "judge_aborts": 0}

    # ---- the starting champion: the seed workspace, which reproduces the baseline
    say("[research] evaluating the seed candidate (the official baseline, as code) ...")
    # Overridable so a test can seed a cheap configuration; the real run uses the
    # committed workspace, which reproduces the official baseline bit-for-bit.
    ws = Path(workspace) if workspace else CAND.WORKSPACE
    seed_files = dict(seed_files) if seed_files else CAND.seed_tree(ws)
    seed_dir = Path(run_root) / "iter000_seed"
    seed_static = CAND.validate_tree(seed_files, ws)
    seed_art = AR.put(seed_files, meta={"role": "seed",
                                        "contract_sha256": seed_static["contract_sha256"]})
    ok, entry = rec("proposal", proposal_body(
        run_id=run_id, iteration=0, attempt=0,
        plan={"hypothesis": ("the official baseline, expressed as an editable candidate "
                             "tree; the starting champion"
                             + (f". STARTING POINT OVERRIDDEN: {seed_note}. This run does "
                                f"NOT begin from the unmodified official baseline."
                                if seed_note else "")), "stage": "config",
              "mechanism": "reference point", "expected_magnitude": 0.0,
              "expected_direction": "unclear", "_files_changed": []},
        tree=seed_art.tree, parent=None,
        diff=AR.tree_diff(None, seed_files), static=seed_static,
        artifact_dir=str(seed_art.path)))
    if not ok:
        raise SystemExit(f"the seed candidate could not be pre-registered: {entry}. "
                         f"Nothing is run unrecorded, so the run stops here.")
    rep = CAND.run_candidate(seed_files, {}, run_dir=seed_dir, task=spec_task(spec),
                             backend=backend, max_seconds=max_seconds, workspace=ws,
                             execution_profile=execution_profile,
                             task_spec_path=task_spec_path)
    if not rep.get("ok"):
        raise SystemExit(f"the seed candidate failed to run: "
                         f"{rep.get('error_type')}: {rep.get('error')}")
    sok, spay = score(rep["dev_scores"], seed_art.tree)
    if not sok:
        raise SystemExit(f"the scorer refused the seed candidate: {spay}")
    hm = harness_metrics(splits, spec, rep["run_dir"])
    seed_exp = Experiment(
        iteration=0, tree=seed_art.tree, parent=None,
        # Must carry the override too. The signed PROPOSAL discloses it, but state.json and
        # every lineage table generated from it read this field -- so without the note here a
        # pairwise-seeded run describes its starting point as "the official baseline".
        hypothesis=("the official baseline as an editable candidate tree"
                    + (f" -- STARTING POINT OVERRIDDEN: {seed_note}; NOT the unmodified "
                       f"official baseline" if seed_note else "")),
        stage="config", files_changed=[], signed=spay,
        train_primary=hm["train"]["primary"], dev_primary=hm["dev"]["primary"],
        gap=hm["gap"], outcome="promoted", wall_seconds=rep.get("total_seconds"))
    seed_exp.execution = rep.get("execution")
    state.add(seed_exp)
    state.promote(seed_exp)
    state.diagnostics["seed_baseline"] = {
        "signed_dev_primary": spay["metrics"]["primary"],
        "harness_dev_primary": hm["dev"]["primary"],
        "harness_train_primary": hm["train"]["primary"],
        "train_dev_gap": hm["gap"],
        "official_baseline_valid": baseline_valid,
        "declared_baseline_valid": baseline_valid,
        # The published TEST baseline is deliberately NOT put in front of the agent. It is a
        # number the organisers printed, not a measurement of our predictions, so including
        # it leaks nothing -- but the agent has no legitimate use for any test figure, and
        # an external review was right that its presence in the prompt state is indefensible
        # to explain. Removed rather than argued for.
        "note": "gap = train primary - dev primary on a fixed 5,000-user train subsample. "
                "Both computed by the harness with the official metric.",
    }
    # The seed candidate is scored every run, so reproduction is MEASURED per run rather
    # than asserted from a test that ran once. Tolerance is 1e-6: the scorer rounds to 6dp.
    seed_p = spay["metrics"]["primary"]
    ref = baseline_valid
    repro = {"seed_signed_primary": seed_p,
             "published_baseline_valid": ref,
             "delta": None if ref is None else round(seed_p - ref, 6),
             "seed_profile_tree": seed_art.tree,
             "note": ("the published figure is rounded to 4dp, so a small delta is the "
                      "rounding, not a difference in the model")}
    state.diagnostics["baseline_reproduction"] = repro
    rec("event", {"run_id": run_id, "what": "seed_baseline_reproduction", **repro})
    say(f"[research] seed: signed dev primary {seed_p:.6f} "
        f"(train {hm['train']['primary']:.6f}, gap {hm['gap']:+.4f})")

    best_window = deque([spay["metrics"]["primary"]], maxlen=conv_n + 1)
    converged = None
    tree_cache: dict[str, dict] = {seed_art.tree: seed_files}
    telem_cache: dict[str, dict] = {}
    tf = Path(rep["run_dir"]) / "telemetry.json"
    if tf.exists():
        try:
            telem_cache[seed_art.tree] = json.loads(tf.read_text())
        except json.JSONDecodeError:
            pass

    # ---------------------------------------------------------------- iterations
    while True:
        why = budget.stop_reason()
        if why is not None:
            say(f"[research] stopping: budget {why}")
            break
        if converged:
            say(f"[research] stopping: {converged}")
            break

        it_no = budget.iterations_done + 1
        with budget.iteration() as it:
            say(f"\n[research] ---- iteration {it_no} "
                f"(champion {state.champion_primary:.6f}, "
                f"{budget.remaining()['wall_seconds']/60:.0f} min left) ----")
            # LIVE decision context. state.budget was set once at construction, so the
            # planner saw only the static maxima -- it could not know it had three
            # evaluations before a convergence check, nor how much gain the window still
            # needed. An agent told "the budget is meant to be spent" while a rule can stop
            # it after three evaluations is being actively misinformed.
            rem = budget.remaining()
            wgain = (best_window[-1] - best_window[0]) if len(best_window) > 1 else 0.0
            until_check = max(0, (conv_n + 1) - len(best_window))
            state.budget = {
                "max_iterations": budget.max_iterations,
                "iterations_done": budget.iterations_done,
                "evaluations_remaining": rem["iterations"],
                "wall_seconds_remaining": round(rem["wall_seconds"]),
                "convergence": {
                    "mode": convergence,
                    "epsilon": conv_epsilon, "N": conv_n,
                    "window_gain_so_far": round(wgain, 6),
                    "iterations_until_next_check": until_check,
                    "cumulative_gain_still_needed_to_survive_the_check":
                        (None if convergence != "official"
                         else round(max(0.0, conv_epsilon - wgain), 6)),
                    "note": ("the window measures CUMULATIVE champion gain, so two "
                             "sub-epsilon improvements inside it are enough to continue"
                             if convergence == "official" else
                             "convergence is disabled for this run"),
                },
                "stages_attempted_recently": [
                    e.stage for e in state.experiments[-6:] if e.stage],
            }
            # The PARENT of this iteration's candidate. Starts as the champion and may be
            # reassigned below if the plan asks to fork an earlier candidate instead, which
            # is why it is not called champ_tree.
            parent_tree = state.champion["tree"]
            parent_files = tree_cache.get(parent_tree) or AR.read(parent_tree)

            # ---- plan: one hypothesis, or a tournament judged down to one
            tourn_meta = None
            try:
                if policy == "tournament":
                    # ---- tournament, with a BOUNDED repair call on incomplete coverage.
                    # An unbalanced set is never accepted: after the bound the iteration is
                    # abandoned before any code is written, fitted or scored, because
                    # spending an evaluation on a set that ignored two lanes would let the
                    # report claim breadth the run did not have.
                    cands, cov, deficiency = [], {}, None
                    for cattempt in range(max_coverage_repairs + 1):
                        with it.stage("tournament"):
                            traw, usage = agent.tournament(
                                state.for_prompt(), parent_files,
                                telem_cache.get(parent_tree), tournament_size,
                                deficiency=deficiency)
                        it.add_tokens(usage.get("input_tokens", 0),
                                      usage.get("output_tokens", 0),
                                      usage.get("thinking_tokens", 0))
                        record_public_thoughts("plan", usage, it_no, "tournament")
                        cands, cov = validate_tournament(traw, tournament_size)
                        if cov["lane_coverage_ok"]:
                            break
                        stats["lane_coverage_violations"] += 1
                        deficiency = cov["problems"]
                        rec("event", {"run_id": run_id, "iteration": it_no,
                                      "what": "tournament_incomplete",
                                      "attempt": cattempt, "coverage": cov,
                                      "repair_available":
                                          cattempt < max_coverage_repairs})
                        say(f"       tournament coverage REFUSED (attempt {cattempt}): "
                            f"{cov['problems'][:2]}")
                        if cattempt < max_coverage_repairs:
                            stats["coverage_repairs"] += 1
                            say("       asking once more with the deficiency")
                    if not cov.get("lane_coverage_ok"):
                        stats["coverage_aborts"] += 1
                        rec("event", {"run_id": run_id, "iteration": it_no,
                                      "what": "iteration_abandoned",
                                      "reason": "tournament coverage incomplete after "
                                                f"{max_coverage_repairs} repair call(s)",
                                      "coverage": cov, "coded": False, "fitted": False,
                                      "scored": False})
                        say("       ABANDONED before coding: coverage still incomplete. "
                            "No candidate was written, fitted or scored.")
                        continue

                    # ---- judge, also bounded. An invalid ranking or a missing architecture
                    # verdict is refused the same way, for the same reason.
                    verdict, jproblems = None, None
                    for jattempt in range(max_coverage_repairs + 1):
                        with it.stage("judge"):
                            verdict, usage = agent.judge(
                                cands, state.for_prompt(), parent_files,
                                telem_cache.get(parent_tree), problems=jproblems)
                        it.add_tokens(usage.get("input_tokens", 0),
                                      usage.get("output_tokens", 0),
                                      usage.get("thinking_tokens", 0))
                        record_public_thoughts("judge", usage, it_no, "judge")
                        jproblems = validate_judge(cands, verdict)
                        if not jproblems:
                            break
                        stats["judge_violations"] += 1
                        rec("event", {"run_id": run_id, "iteration": it_no,
                                      "what": "judge_invalid", "attempt": jattempt,
                                      "problems": jproblems,
                                      "repair_available":
                                          jattempt < max_coverage_repairs})
                        say(f"       judge response REFUSED (attempt {jattempt}): "
                            f"{jproblems[:2]}")
                    if jproblems:
                        stats["judge_aborts"] += 1
                        rec("event", {"run_id": run_id, "iteration": it_no,
                                      "what": "iteration_abandoned",
                                      "reason": "judge response invalid after "
                                                f"{max_coverage_repairs} repair call(s)",
                                      "problems": jproblems, "coded": False,
                                      "fitted": False, "scored": False})
                        say("       ABANDONED before coding: judge response invalid. "
                            "No candidate was written, fitted or scored.")
                        continue
                    chosen, how = select_from_tournament(cands, verdict)
                    plan = candidate_to_plan(chosen)
                    arch = [c for c in cands if c["lane"] == "model_architecture"]
                    stats["tournaments"] += 1
                    stats["candidates_considered"] += len(cands)
                    stats["architecture_considered"] += len(arch)
                    if plan.get("lane") == "model_architecture":
                        stats["architecture_selected"] += 1
                    if not cov["lane_coverage_ok"]:
                        stats["lane_coverage_violations"] += 1
                        say(f"       tournament missed required lanes: "
                            f"{cov['missing_required_lanes']} (recorded, not hidden)")
                    tourn_meta = {
                        "coverage": cov, "selection_how": how,
                        "selected_id": plan.get("_candidate_id"),
                        "candidates": [
                            {k: c.get(k) for k in
                             ("id", "lane", "hypothesis", "mechanism",
                              "expected_direction", "expected_magnitude",
                              "information_value", "novelty_vs_history",
                              "files_to_change", "runtime_estimate",
                              "implementation_risk", "leakage_risk")}
                            for c in cands],
                        "judge": {k: verdict.get(k) for k in
                                  ("selected_id", "ranking", "why_selected",
                                   "architecture_verdicts", "exploit_or_explore",
                                   "convergence_note")},
                    }
                    say(f"       tournament: {len(cands)} candidates "
                        f"({sum(1 for c in cands if c['lane']=='model_architecture')} "
                        f"architecture) -> chose {plan.get('_candidate_id')} "
                        f"[{plan.get('lane')}] ({how})")
                else:
                    with it.stage("plan"):
                        plan, usage = agent.plan(state.for_prompt(), parent_files,
                                                 telem_cache.get(parent_tree))
                    it.add_tokens(usage.get("input_tokens", 0),
                                  usage.get("output_tokens", 0),
                                  usage.get("thinking_tokens", 0))
                    record_public_thoughts("plan", usage, it_no, "plan")
            except Exception as e:
                stats["infra_events"] += 1
                rec("event", {"run_id": run_id, "iteration": it_no,
                              "what": "planner_error", "policy": policy,
                              "error_type": type(e).__name__, "error": str(e)[:600]})
                say(f"       planner failed: {type(e).__name__}: {str(e)[:160]}")
                continue
            stats["planned"] += 1

            # ---- record the tournament BEFORE any candidate code exists or runs, so the
            # set considered and the judge's reasoning cannot be reconstructed after the
            # result is known. A refused record aborts the iteration: an unrecorded
            # tournament would let the report claim breadth with nothing behind it.
            if tourn_meta is not None:
                ok, entry = rec("event", {
                    "run_id": run_id, "iteration": it_no, "what": "tournament",
                    "policy": policy, "tournament_size": tournament_size,
                    **tourn_meta})
                if not ok:
                    stats["unregistered_aborts"] += 1
                    say(f"       ABANDONED: the tournament could not be recorded "
                        f"({entry}). Nothing runs unrecorded.")
                    continue
                say(f"       tournament recorded at seq {entry}")
            say(f"       hypothesis: {str(plan.get('hypothesis',''))[:150]}")
            say(f"       stage={plan.get('stage')} "
                f"predicted={plan.get('expected_magnitude')}")

            # ---- a parent other than the champion, if the plan asked for one
            want = str(plan.get("parent", "champion") or "champion")
            if want != "champion":
                match = [t for t in state.seen_trees if t.startswith(want)]
                if len(match) == 1:
                    parent_files = tree_cache.get(match[0]) or AR.read(match[0])
                    parent_tree = match[0]
                    say(f"       forking {want} instead of the champion")

            # ---- code, then repair on failure. Every executed tree is registered.
            error_ctx, exp, result = None, None, None
            for attempt in range(max_repairs + 1):
                try:
                    with it.stage("code"):
                        out, usage = agent.code(plan, parent_files, state.for_prompt(),
                                                error=error_ctx, attempt=attempt)
                    it.add_tokens(usage.get("input_tokens", 0),
                                  usage.get("output_tokens", 0),
                                  usage.get("thinking_tokens", 0))
                    record_public_thoughts("code", usage, it_no, "code")
                except Exception as e:
                    stats["infra_events"] += 1
                    rec("event", {"run_id": run_id, "iteration": it_no,
                                  "what": "coder_error", "attempt": attempt,
                                  "error_type": type(e).__name__, "error": str(e)[:600]})
                    say(f"       coder failed: {type(e).__name__}: {str(e)[:160]}")
                    break

                # What the coder actually sent. A repair prompt otherwise showed the
                # PARENT plus an error whose line numbers refer to the file the coder
                # wrote and was never shown -- "syntax error on line 157" against a file
                # it cannot see is not an actionable error.
                sent = {str(f.get("path")): str(f.get("content", ""))[:12000]
                        for f in (out.get("files") or []) if isinstance(f, dict)}

                # ---- apply + validate. A refused tree costs nothing and is recorded.
                try:
                    child, changed = merge_tree(parent_files, out.get("files"))
                    static = CAND.validate_tree(child, ws)
                except (CAND.CandidateError, AR.ArtifactError) as e:
                    stats["refused"] += 1
                    rec("event", {"run_id": run_id, "iteration": it_no,
                                  "what": "candidate_refused", "attempt": attempt,
                                  "error_type": type(e).__name__, "error": str(e)[:700]})
                    say(f"       REFUSED: {str(e)[:170]}")
                    error_ctx = {"phase": "static_validation",
                                 "error_type": type(e).__name__, "error": str(e)[:900],
                                 "your_previous_attempt": sent}
                    if attempt < max_repairs:
                        it.note_repair(); stats["repaired"] += 1
                        continue
                    break

                if not changed:
                    stats["refused"] += 1
                    rec("event", {"run_id": run_id, "iteration": it_no,
                                  "what": "candidate_unchanged", "attempt": attempt})
                    say("       REFUSED: the returned tree is identical to its parent")
                    error_ctx = {"phase": "static_validation",
                                 "error_type": "NoChange",
                                 "error": "the tree you returned is byte-identical to the "
                                          "parent after canonicalisation; nothing would be "
                                          "tested. Change something real.",
                                 "your_previous_attempt": sent}
                    if attempt < max_repairs:
                        it.note_repair(); stats["repaired"] += 1
                        continue
                    break

                tree = static["tree"]
                seen = state.already_seen(tree)
                if seen is not None:
                    stats["duplicate"] += 1
                    rec("event", {"run_id": run_id, "iteration": it_no,
                                  "what": "candidate_duplicate", "tree": tree,
                                  "first_seen_iteration": seen})
                    say(f"       DUPLICATE of iteration {seen}; not refitted")
                    error_ctx = {"phase": "dedup", "error_type": "Duplicate",
                                 "error": f"this exact tree was already evaluated at "
                                          f"iteration {seen}. Vary something real.",
                                 "your_previous_attempt": sent}
                    if attempt < max_repairs:
                        it.note_repair(); stats["repaired"] += 1
                        continue
                    break

                diff = AR.tree_diff(parent_files, child)
                art = AR.put(child, meta={"contract_sha256": static["contract_sha256"],
                                          "iteration": it_no, "attempt": attempt})
                plan["_files_changed"] = changed

                # ---- PRE-REGISTER. Nothing runs unrecorded.
                ok, entry = rec("proposal", proposal_body(
                    run_id=run_id, iteration=it_no, attempt=attempt, plan=plan,
                    tree=tree, parent=parent_tree, diff=diff, static=static,
                    artifact_dir=str(art.path)))
                if not ok:
                    stats["unregistered_aborts"] += 1
                    say(f"       ABANDONED: pre-registration refused ({entry}). A "
                        f"candidate is never run unrecorded.")
                    break
                stats["registered"] += 1
                say(f"       tree {tree[:12]} changed {changed} "
                    f"({len(diff.encode()):,}-byte diff, seq {entry})")

                # ---- fit, sandboxed
                cdir = Path(run_root) / f"iter{it_no:03d}_a{attempt}_{tree[:12]}"
                with it.stage("fit"):
                    rep = CAND.run_candidate(child, {}, run_dir=cdir,
                                             task=spec_task(spec), backend=backend,
                                             max_seconds=max_seconds, workspace=ws,
                                             execution_profile=execution_profile,
                                             task_spec_path=task_spec_path)
                state.note_seen(tree, it_no)
                tree_cache[tree] = child

                if not rep.get("ok"):
                    stats["failed"] += 1
                    rec("event", {"run_id": run_id, "iteration": it_no,
                                  "what": "candidate_failed", "attempt": attempt,
                                  "tree": tree, "error_type": rep.get("error_type"),
                                  "error": str(rep.get("error"))[:700],
                                  "error_frame": rep.get("error_frame"),
                                  "error_line": rep.get("error_line"),
                                  "error_in": rep.get("error_in"),
                                  "contract_violation": rep.get("contract_violation"),
                                  "isolation": rep.get("isolation"),
                                  "execution": rep.get("execution")})
                    say(f"       FIT FAILED [{rep.get('error_type')}] at "
                        f"{rep.get('error_frame')}: {str(rep.get('error'))[:130]}")
                    error_ctx = {"phase": "execution",
                                 "error_type": rep.get("error_type"),
                                 "error": str(rep.get("error"))[:900],
                                 "error_frame": rep.get("error_frame"),
                                 "error_line": rep.get("error_line"),
                                 "error_in": rep.get("error_in"),
                                 "traceback": str(rep.get("traceback") or "")[-1800:],
                                 "contract_violation": rep.get("contract_violation"),
                                 "your_previous_attempt": sent}
                    exp = Experiment(
                        iteration=it_no, tree=tree, parent=parent_tree,
                        hypothesis=str(plan.get("hypothesis", ""))[:600],
                        stage=str(plan.get("stage", "")), files_changed=changed,
                        lane=str(plan.get("lane", "")),
                        considered_in_lane=(0 if tourn_meta is None else sum(
                            1 for c in tourn_meta["candidates"]
                            if c.get("lane") == plan.get("lane"))),
                        tournament_size=(0 if tourn_meta is None
                                         else len(tourn_meta["candidates"])),
                        mechanism=str(plan.get("mechanism", ""))[:600],
                        diff_hash=AR.diff_hash(diff), diff_bytes=len(diff.encode()),
                        expected_direction=str(plan.get("expected_direction", "")),
                        expected_magnitude=_num(plan.get("expected_magnitude")),
                        outcome="failed", error_type=str(rep.get("error_type") or ""),
                        error=str(rep.get("error") or "")[:600],
                        wall_seconds=rep.get("total_seconds"))
                    if attempt < max_repairs:
                        it.note_repair(); stats["repaired"] += 1
                        say(f"       returning the error to the coder "
                            f"(repair {attempt + 1}/{max_repairs})")
                        continue
                    break

                # ---- score: the only number that decides anything
                with it.stage("score"):
                    sok, spay = score(rep["dev_scores"], tree)
                if not sok:
                    stats["failed"] += 1
                    rec("event", {"run_id": run_id, "iteration": it_no,
                                  "what": "scorer_refused", "tree": tree,
                                  "error": str(spay)[:600]})
                    say(f"       scorer REFUSED: {str(spay)[:160]}")
                    break

                hm = harness_metrics(splits, spec, rep["run_dir"])
                # The harness and the scorer compute the SAME metric on the SAME rows, so
                # they must agree. A disagreement means the predictions and the labels are
                # not aligned, whose only other symptom is a slightly low score.
                # The scorer rounds its metrics to 6dp (core/official_scorer.py:171), so
                # the comparison is exact after the same rounding rather than a tolerance:
                # same function, same rows, same file -> bit-identical floats. Any
                # difference means the predictions and the labels are not aligned, whose
                # only other symptom is a slightly low score.
                drift = abs(hm["dev"]["primary"] - spay["metrics"]["primary"])
                if round(hm["dev"]["primary"], 6) != spay["metrics"]["primary"]:
                    stats["metric_disagreements"] += 1
                    rec("event", {"run_id": run_id, "iteration": it_no,
                                  "what": "metric_disagreement", "tree": tree,
                                  "harness_dev_primary": hm["dev"]["primary"],
                                  "scorer_dev_primary": spay["metrics"]["primary"],
                                  "abs_difference": drift})
                    say(f"       WARNING: harness and scorer disagree by {drift:.2e}")

                tel = None
                tfp = Path(rep["run_dir"]) / "telemetry.json"
                if tfp.exists():
                    try:
                        tel = json.loads(tfp.read_text())
                        telem_cache[tree] = tel
                    except json.JSONDecodeError:
                        pass

                p = spay["metrics"]["primary"]
                parent_p = _experiment_primary(state, parent_tree)
                exp = Experiment(
                    iteration=it_no, tree=tree, parent=parent_tree,
                    hypothesis=str(plan.get("hypothesis", ""))[:600],
                    stage=str(plan.get("stage", "")), files_changed=changed,
                    lane=str(plan.get("lane", "")),
                    considered_in_lane=(0 if tourn_meta is None else sum(
                        1 for c in tourn_meta["candidates"]
                        if c.get("lane") == plan.get("lane"))),
                    tournament_size=(0 if tourn_meta is None
                                     else len(tourn_meta["candidates"])),
                    mechanism=str(plan.get("mechanism", ""))[:600],
                    diff_hash=AR.diff_hash(diff), diff_bytes=len(diff.encode()),
                    expected_direction=str(plan.get("expected_direction", "")),
                    expected_magnitude=_num(plan.get("expected_magnitude")),
                    signed=spay, train_primary=hm["train"]["primary"],
                    dev_primary=hm["dev"]["primary"], gap=hm["gap"],
                    repairs=attempt, wall_seconds=rep.get("total_seconds"),
                    execution=rep.get("execution"),
                    delta_vs_parent=None if parent_p is None else round(p - parent_p, 6),
                    delta_vs_champion=round(p - state.champion_primary, 6))
                stats["evaluated"] += 1
                result = {"signed": spay, "train_primary": hm["train"]["primary"],
                          "dev_primary": hm["dev"]["primary"], "gap": hm["gap"],
                          "delta_vs_champion": exp.delta_vs_champion,
                          "repairs": attempt, "fit_seconds": rep.get("fit_seconds"),
                          "telemetry": tel, "execution": rep.get("execution")}
                say(f"       signed dev primary {p:.6f} "
                    f"({exp.delta_vs_champion:+.6f} vs champion), "
                    f"train {hm['train']['primary']:.6f}, gap {hm['gap']:+.4f}")
                break

            # A refused or duplicated candidate never produced an Experiment, so nothing
            # reached state.experiments or the failure registry and the next planner saw no
            # sign the idea had been tried. The ledger had the event; the agent's own memory
            # did not, which is how an agent proposes the same refused idea twice.
            if exp is None and error_ctx is not None:
                exp = Experiment(
                    iteration=it_no, tree=f"refused:{it_no}", parent=parent_tree,
                    hypothesis=str(plan.get("hypothesis", ""))[:600],
                    stage=str(plan.get("stage", "")),
                    lane=str(plan.get("lane", "")),
                    tournament_size=(0 if tourn_meta is None
                                     else len(tourn_meta["candidates"])),
                    mechanism=str(plan.get("mechanism", ""))[:600],
                    files_changed=list(plan.get("_files_changed") or []),
                    expected_direction=str(plan.get("expected_direction", "")),
                    expected_magnitude=_num(plan.get("expected_magnitude")),
                    outcome="refused", repairs=stats_repairs_this_iter(it),
                    error_type=str(error_ctx.get("error_type") or error_ctx.get("phase")),
                    error=str(error_ctx.get("error") or "")[:600],
                    wall_seconds=None)
                # Deliberately NOT added here: the block below adds exactly one experiment
                # per iteration. Adding it in both places recorded it twice AND relabelled
                # it "rejected", which is how a refusal became indistinguishable from an
                # evaluated candidate that simply lost.
                say("       recorded as refused so the next plan can see it")

            # ---- reflect on whatever happened, including a failure
            if exp is not None:
                try:
                    with it.stage("reflect"):
                        refl, usage = agent.reflect(
                            plan, result or {"failed": True,
                                             "error_type": exp.error_type,
                                             "error": exp.error, "repairs": exp.repairs},
                            state.for_prompt())
                    it.add_tokens(usage.get("input_tokens", 0),
                                  usage.get("output_tokens", 0),
                                  usage.get("thinking_tokens", 0))
                    record_public_thoughts("reflect", usage, it_no, "reflect")
                    exp.reflection = refl
                    # Structured beliefs, keyed by MECHANISM, so the next tournament reasons
                    # over conclusions rather than re-reading prose. The supporting and
                    # contradicting iteration lists come from the signed outcome.
                    touched = state.update_beliefs(exp, refl.get("belief_updates"))
                    if touched:
                        stats["beliefs_updated"] += len(touched)
                        say(f"       beliefs updated: {touched}")
                except Exception as e:
                    stats["infra_events"] += 1
                    rec("event", {"run_id": run_id, "iteration": it_no,
                                  "what": "reflector_error",
                                  "error_type": type(e).__name__, "error": str(e)[:600]})

                # ---- promote on the SIGNED number, never on the reflection's advice
                if exp.signed and exp.delta_vs_champion is not None \
                        and exp.delta_vs_champion > 0:
                    state.add(exp)
                    state.promote(exp)
                    stats["promoted"] += 1
                    say(f"       PROMOTED: champion is now {exp.tree[:12]} "
                        f"at {exp.signed['metrics']['primary']:.6f}")
                else:
                    # "failed" (it ran and broke) and "refused" (it never ran) are both
                    # already accurate; only an evaluated candidate that lost is "rejected".
                    if exp.outcome not in ("failed", "refused"):
                        exp.outcome = "rejected"
                    state.add(exp)

                rec("event", {"run_id": run_id, "iteration": it_no,
                              "what": "iteration_result",
                              "tree": exp.tree, "outcome": exp.outcome,
                              "signed_primary": (exp.signed or {}).get(
                                  "metrics", {}).get("primary"),
                              "train_primary": exp.train_primary,
                              "dev_primary": exp.dev_primary, "gap": exp.gap,
                              "expected_magnitude": exp.expected_magnitude,
                              "delta_vs_champion": exp.delta_vs_champion,
                              "repairs": exp.repairs,
                              "lane": exp.lane,
                              "tournament_size": exp.tournament_size,
                              "considered_in_lane": exp.considered_in_lane,
                              "beliefs_after": state.beliefs_for_prompt(6),
                              "reflection": _trim(exp.reflection),
                              # record_now(), not cost.to_record(): this entry is written
                              # INSIDE the iteration context, where __exit__ has not yet
                              # filled wall_seconds. Using the raw record wrote 0.0 into the
                              # ledger while the stages beneath it totalled ~54 s.
                              "cost": it.record_now()})

            # ---- convergence, on the organisers' rule
            best_window.append(state.champion_primary)
            if convergence == "official" and len(best_window) > conv_n:
                gain = best_window[-1] - best_window[0]
                if gain <= conv_epsilon:
                    converged = (f"converged: the validation best improved by "
                                 f"{gain:+.6f} over {conv_n} iterations, which does "
                                 f"not exceed epsilon={conv_epsilon}")

    return {"stats": stats, "converged": converged, "budget": budget.summary()}


def stats_repairs_this_iter(it) -> int:
    """Repairs recorded on this iteration's cost object so far."""
    return int(getattr(getattr(it, "cost", None), "repairs", 0) or 0)


def spec_task(spec) -> str:
    """The task slug to hand the in-sandbox runner.

    This read `getattr(spec, "name", None) or "kuairand_pure"`, and TaskSpec has no
    `name` -- it has `slug`. So every run silently passed the hard-coded default, and
    `--task something_else` would have loaded one spec here and a DIFFERENT dataset inside
    the sandbox. No fallback now: an unrecognisable spec is an error, not a default.
    """
    slug = getattr(spec, "slug", None)
    if not slug:
        raise ValueError(
            f"task spec {type(spec).__name__} exposes no `slug`, so the sandboxed runner "
            f"cannot be told which task to load. Refusing to guess.")
    return str(slug)


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _trim(d, limit: int = 900):
    if not isinstance(d, dict):
        return d
    return {k: (v[:limit] if isinstance(v, str) else v) for k, v in d.items()}


def _experiment_primary(state: ResearchState, tree: str):
    for e in reversed(state.experiments):
        if e.tree == tree and e.signed:
            return e.signed["metrics"]["primary"]
    return None


# ------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Code-first autonomous research loop (Track 2).")
    ap.add_argument("--iterations", type=int, default=BUD.OFFICIAL_MAX_ITERATIONS,
                    help=f"hard cap; the organisers' limit is "
                         f"{BUD.OFFICIAL_MAX_ITERATIONS}")
    ap.add_argument("--max-wall-seconds", type=float,
                    default=float(BUD.OFFICIAL_MAX_WALL_SECONDS),
                    help="hard agent wall-clock cap; the organisers' limit is 6 h")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--backend", default="gemini",
                    choices=["gemini", "ollama", "groq", "anthropic"],
                    help="same backends as the typed proposer, so the two arms stay "
                         "comparable. 'anthropic' needs its SDK installed.")
    ap.add_argument("--model", default=None,
                    help="backend-specific model id; each backend has its own default")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--thinking-level", default=None)
    ap.add_argument("--thought-summaries", action="store_true",
                    help="request provider-generated public thought summaries when the "
                         "backend supports them; never exposes hidden chain-of-thought")
    ap.add_argument("--isolation", default="seatbelt",
                    choices=["seatbelt", "docker", "subprocess", "none"])
    ap.add_argument("--fit-seconds", type=float, default=1800.0,
                    help="per-candidate timeout inside the sandbox")
    ap.add_argument("--framework", default="auto", choices=list(EXEC.FRAMEWORKS),
                    help="operator execution constraint; auto infers NumPy or torch from "
                         "the candidate source")
    ap.add_argument("--device", default="auto", choices=list(EXEC.DEVICES),
                    help="numerical device for candidates: auto, cpu, Apple mps, or cuda")
    ap.add_argument("--accelerator-policy", default="prefer",
                    choices=list(EXEC.ACCELERATOR_POLICIES),
                    help="off/allow/prefer/require accelerator use for torch candidates")
    ap.add_argument("--no-device-fallback", action="store_true",
                    help="refuse a candidate when its requested device is unavailable")
    ap.add_argument("--max-repairs", type=int, default=MAX_REPAIRS)
    ap.add_argument("--convergence", default="official", choices=["official", "off"])
    ap.add_argument("--task", default="kuairand_pure")
    ap.add_argument("--task-spec", default=None,
                    help="confirmed portable JSON task specification; overrides --task")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--research-policy", default="tournament",
                    choices=["tournament", "single"],
                    help="'tournament' (default) proposes several competing hypotheses "
                         "across every research lane and has a judge pick one; 'single' is "
                         "the original one-hypothesis-per-iteration policy, kept for "
                         "reproduction of the recorded runs.")
    ap.add_argument("--tournament-size", type=int, default=6,
                    help="candidates requested per tournament. The MINIMUM is 6 -- one per "
                         "research lane -- because a smaller set cannot cover them and the "
                         "coverage check would abandon every iteration. Maximum 12. Only "
                         "the winner consumes an evaluation.")
    ap.add_argument("--max-coverage-repairs", type=int, default=1,
                    help="repair calls allowed when a tournament's lane coverage is "
                         "incomplete, or a judge response is invalid. After the bound the "
                         "iteration is abandoned WITHOUT coding, fitting or scoring.")
    ap.add_argument("--seed-profile", default="inherited",
                    choices=["inherited", "minimal", "accelerated"],
                    help="'inherited' is research_workspace/, which contains a prepared "
                         "pairwise switch and is required to reproduce the recorded runs. "
                         "'minimal' is research_workspace_minimal/: the pointwise official "
                         "baseline only, with no pairwise, listwise or alternative "
                         "architecture supplied, so a ranking objective must be written; "
                         "'accelerated' is an explicitly non-official PyTorch FM seed for "
                         "CPU/MPS/CUDA execution research.")
    ap.add_argument("--planner-model", default=None,
                    help="model for hypothesis generation; falls back to --model")
    ap.add_argument("--judge-model", default=None,
                    help="model for tournament judging; falls back to --model")
    ap.add_argument("--coder-model", default=None,
                    help="model that writes the candidate; falls back to --model")
    ap.add_argument("--reflector-model", default=None,
                    help="model for reflection; falls back to --model")
    ap.add_argument("--seed-config", default=None,
                    help='JSON object merged into the seed tree\'s config.json before the '
                         'run, e.g. \'{"loss": "pairwise"}\'. Recorded in run_start and in '
                         'the seed proposal: starting from anything other than the '
                         'unmodified official baseline is a DISCLOSURE, not a detail, so it '
                         'goes in the signed chain rather than in prose.')
    return ap


def main() -> None:
    a = build_parser().parse_args()
    from agent.research_agent import MIN_TOURNAMENT_SIZE
    if a.research_policy == "tournament":
        if a.tournament_size < MIN_TOURNAMENT_SIZE:
            raise SystemExit(
                f"--tournament-size {a.tournament_size} is below the minimum "
                f"{MIN_TOURNAMENT_SIZE}. The tournament policy requires one candidate per "
                f"research lane; a smaller set cannot satisfy the coverage check and every "
                f"iteration would be abandoned. Either raise it or use "
                f"--research-policy single.")
        if a.tournament_size > 12:
            raise SystemExit(f"--tournament-size {a.tournament_size} exceeds the maximum 12")
    # ---- optional seed override, validated against the committed config's own keys
    seed_files, seed_note = None, ""
    if a.seed_config:
        try:
            overrides = json.loads(a.seed_config)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--seed-config is not valid JSON: {e}") from None
        if not isinstance(overrides, dict) or not overrides:
            raise SystemExit("--seed-config must be a non-empty JSON object")
        base = CAND.seed_tree(CAND.workspace_for(a.seed_profile))
        cfg0 = json.loads(base["config.json"])
        unknown = sorted(set(overrides) - set(cfg0))
        if unknown:
            raise SystemExit(
                f"--seed-config names keys the workspace config does not define: "
                f"{unknown}. Known keys: {sorted(cfg0)}")
        cfg0.update(overrides)
        seed_files = {**base, "config.json": json.dumps(cfg0, indent=2) + "\n"}
        seed_note = json.dumps(overrides, sort_keys=True)
        print(f"[research] SEED OVERRIDE {seed_note} -- this run does NOT start from the "
              f"unmodified official baseline")

    from core.handshake import (HandshakeError, require_same_data, require_task,
                               require_verified_chain)
    from core.scorer import request
    from core.task import load as get_task

    t0 = time.time()
    run_id = (a.run_name
              or f"research_{a.backend}_{str(a.model or 'default').replace('/', '_')}"
                 f"_{int(t0)}")

    spec = get_task(a.task_spec or a.task)
    ping = request({"op": "ping"})
    try:
        require_task(ping, spec.slug if a.task_spec else "official")
        require_verified_chain(ping)
    except HandshakeError as e:
        raise SystemExit(str(e)) from None
    print(f"[research] scorer task={ping['task']} "
          f"manifest {str(ping.get('manifest_sha256',''))[:16]} "
          f"ledger {Path(str(ping.get('ledger') or '?')).name}")

    splits, edges = spec.loader()
    held = spec.split.held_out_split
    dev = spec.split.dev_split
    if splits[held].label is not None:
        raise SystemExit(f"refusing to run: {held} labels reached the driver")
    print(f"[research] train {len(splits['train']):,} / {dev} {len(splits[dev]):,} "
          f"rows; {held} labels absent from this process")

    execution_profile = EXEC.ExecutionProfile(
        framework=a.framework, device=a.device,
        accelerator_policy=a.accelerator_policy,
        allow_fallback=not a.no_device_fallback,
        max_fit_seconds=a.fit_seconds)
    capabilities = EXEC.public_capabilities()
    print(f"[research] execution framework={a.framework} device={a.device} "
          f"policy={a.accelerator_policy}; "
          f"mps={capabilities['mps']['available']} "
          f"cuda={capabilities['cuda']['available']}")

    budget = BUD.Budget(max_iterations=a.iterations,
                        max_wall_seconds=a.max_wall_seconds,
                        max_tokens=a.max_tokens)
    state = ResearchState(
        task={"name": spec.slug, "metric": spec.metric.name,
              "label": spec.label,
              "framing": spec.title,
              "group_by": spec.metric.group_by,
              "dev_split": spec.split.dev_split,
              "held_out_split": spec.split.held_out_split,
              "official_baseline_valid_primary": spec.baseline.get(dev),
              "declared_baseline_valid_primary": spec.baseline.get(dev),
              "convergence": dict(spec.convergence),
              "execution_profile": execution_profile.to_dict(),
              "hardware_capabilities": capabilities},
        budget={"max_iterations": a.iterations, "max_wall_seconds": a.max_wall_seconds})

    ws = CAND.workspace_for(a.seed_profile)
    provenance = CAND.profile_provenance(a.seed_profile)
    if not provenance.get("official_numpy_seed") and not seed_note:
        seed_note = ("accelerated PyTorch reference profile; this is NOT the byte-identical "
                     "official NumPy baseline")
    print(f"[research] seed profile {a.seed_profile!r} ({ws.name}): "
          f"pairwise_present={provenance['pairwise_code_present']} "
          f"listwise_present={provenance['listwise_code_present']} "
          f"alt_architecture_present="
          f"{provenance['alternative_architecture_code_present']}")

    from agent.research_agent import ResearchAgent
    agent = ResearchAgent(backend=a.backend, model=a.model, effort=a.effort,
                          thinking_level=a.thinking_level,
                          thinking_summaries=a.thought_summaries,
                          planner_model=a.planner_model, judge_model=a.judge_model,
                          coder_model=a.coder_model, reflector_model=a.reflector_model)
    print(f"[research] policy={a.research_policy}"
          + (f" size={a.tournament_size}" if a.research_policy == "tournament" else "")
          + f"  model routing: {agent.routing()}")

    run_root = ROOT / "runs" / "candidates" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "state.json"

    def rec(kind: str, body: dict):
        r = request({"op": "record", "kind": kind, "body": body})
        if not r.get("ok"):
            return False, r.get("error")
        return True, r["entry"]["seq"]

    def score(path: str, tree: str):
        r = request({"op": "score_task" if a.task_spec else "score_official",
                     "scores": str(path), "split": dev,
                     "config_hash": tree, "isolation": a.isolation,
                     "kind": "score"})
        if not r.get("ok"):
            return False, r.get("error")
        pl = r["entry"]["payload"]
        try:
            require_same_data(splits[dev].content_sha256(),
                              pl.get("split_content_sha256"), dev)
        except HandshakeError as e:
            return False, str(e)
        return True, pl

    ok, seq = rec("run_start", {
        "run_id": run_id, "task": spec.slug, "driver": "research_driver",
        "task_spec_path": str(Path(a.task_spec).resolve()) if a.task_spec else None,
        "mode": "code_first", "backend": a.backend, "model": agent.model,
        "effort": a.effort, "thinking_level": a.thinking_level,
        "thought_summaries_requested": a.thought_summaries,
        "isolation": a.isolation, "max_iterations": a.iterations,
        "max_wall_seconds": a.max_wall_seconds,
        "execution_profile": execution_profile.to_dict(),
        "hardware_capabilities": capabilities,
        "contract_sha256": CAND.contract_sha256(ws),
        "research_policy": a.research_policy,
        "tournament_size": (a.tournament_size if a.research_policy == "tournament" else 0),
        "model_routing": agent.routing(),
        "seed_profile": a.seed_profile,
        "seed_provenance": provenance,
        "seed_config_overrides": (json.loads(a.seed_config) if a.seed_config else None),
        "starts_from_official_baseline": (not bool(seed_note)
                                           and provenance.get("official_numpy_seed", False)),
        "convergence": {"epsilon": float(spec.convergence.get("epsilon", OFFICIAL_EPSILON)),
                        "N": int(spec.convergence.get("N", OFFICIAL_N)),
                        "mode": a.convergence},
        "started_epoch": t0})
    if not ok:
        raise SystemExit(f"could not record run_start: {seq}")

    try:
        out = research_loop(
            agent=agent, splits=splits, edges=edges, spec=spec, state=state,
            budget=budget, run_id=run_id, rec=rec, score=score, run_root=run_root,
            backend=a.isolation, max_seconds=a.fit_seconds,
            convergence=a.convergence, max_repairs=a.max_repairs,
            seed_files=seed_files, seed_note=seed_note,
            policy=a.research_policy, tournament_size=a.tournament_size,
            workspace=ws, max_coverage_repairs=a.max_coverage_repairs,
            execution_profile=execution_profile, task_spec_path=a.task_spec)
    finally:
        state.save(state_path)

    rec("run_end", {"run_id": run_id, "termination": out["converged"] or "budget",
                    "stats": out["stats"], "budget": out["budget"],
                    "champion": state.champion,
                    "usage_by_role": agent.usage_by_role(),
                    "lane_stats": state.lane_stats(),
                    "beliefs": state.beliefs,
                    "seed_profile": a.seed_profile,
                    "research_policy": a.research_policy,
                    "execution_profile": execution_profile.to_dict(),
                    "hardware_capabilities": capabilities,
                    "manual_interventions": state.manual_interventions})
    state.save(state_path)

    print("\n[research] ================= run complete =================")
    print(f"  champion tree      {str(state.champion['tree'])[:12]} "
          f"at dev primary {state.champion['primary']:.6f}")
    declared_baseline = spec.baseline.get(dev)
    if declared_baseline is not None:
        print(f"  vs declared base   "
              f"{state.champion['primary'] - float(declared_baseline):+.6f}")
    else:
        print("  vs declared base   unavailable (not provided by task spec)")
    print(f"  iterations         {out['budget']['iterations']}/"
          f"{out['budget']['max_iterations']}")
    print(f"  wall clock         {out['budget']['wall_hours']:.2f} h of "
          f"{a.max_wall_seconds/3600:.1f} h")
    print(f"  tokens             {out['budget']['tokens_total']:,}")
    print(f"  stats              {json.dumps(out['stats'])}")
    print(f"  state              {state_path}")


if __name__ == "__main__":
    main()
