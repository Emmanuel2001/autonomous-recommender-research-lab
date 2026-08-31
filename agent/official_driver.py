"""The loop, for the CURRENT task. propose -> validate -> fit -> score -> gate -> record.

Deliberately a sibling of `agent/driver.py` rather than a rewrite of it: that file runs
the superseded task and its recorded runs are still cited. Everything structural is
carried over unchanged, because none of it depended on the task --

  * the driver holds NO signing key; every number comes from the scorer socket
  * proposals are PRE-REGISTERED: the rationale is countersigned at a sequence number
    below the score that tests it, so `expected_magnitude` cannot be retrofitted
  * population admits every non-crashed config; the gate only moves the incumbent
  * failure registry + circuit breaker, so a bad config cannot end an unattended run
  * canonical config hash over model-relevant fields only, so no-ops cost nothing

What IS new: the convergence rule is no longer ours to declare. The organisers publish
epsilon = 0.002 and N = 3, derived from the baseline's own seed std of 0.0008. We use
theirs and report it as theirs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import official_actions as A
from agent.proposer import make as make_proposer
from core.metrics import (OFFICIAL_EPSILON, OFFICIAL_N, OFFICIAL_FM_SEED_STD,
                          OFFICIAL_VALID, OFFICIAL_TEST)
from core.official_data import load_for_strategy
from core.scorer import request
from strategy.ranker import Config, run as fit_and_score

SEED_KNOWLEDGE = {
    "task": "Within-user ranking of each user's own logged impressions. NOT retrieval. "
            "Label long_view. Metrics GAUC and nDCG@5; the score is their mean.",
    "official_baseline": {"valid_primary": OFFICIAL_VALID["fm_official"],
                          "test_primary": OFFICIAL_TEST["fm_official"],
                          "model": "FM k=16 lr=0.001, 5 fields, pointwise logloss"},
    "ceiling": f"nDCG@5 cannot exceed 0.729 because 27.1% of users have no positive at "
               f"all. Oracle primary is {OFFICIAL_TEST['oracle_ceiling']}, and the "
               f"baseline already holds 30.7% of the attainable span.",
    "seed_noise": f"The baseline's std over 5 seeds is {OFFICIAL_FM_SEED_STD}. epsilon="
                  f"{OFFICIAL_EPSILON} is about 2.5 sigma of that. Anything smaller is "
                  f"noise. Note this std is comparable in size to the entire margin any "
                  f"single change has produced so far -- draw your own conclusion.",
    "COSTS": "SetEnsemble(n_seeds=k) fits k models instead of 1, so it costs k times the "
             "wall clock (a single fit is ~15s, k=5 is ~75s). Feasibility is scored on "
             "total compute, so spend it deliberately.",
    "MEASURED_DEAD_ENDS": {
        "more_static_fields": A.DEAD_ENDS["more_static_fields"],
        "more_capacity": A.DEAD_ENDS["more_capacity"],
        "user_side_only": A.DEAD_ENDS["user_side_only"]},
    "YOU_MAY_WRITE_CODE": (
        "ProposePatch lets you write the strategy yourself instead of tuning ours. "
        "Define fit_predict(train, dev, heldout, params) returning "
        "{'dev': scores, 'heldout': scores} -- one finite float per row, in row order, "
        "higher is better. train/dev expose .label; heldout.label is None by "
        "construction and the held-out labels are redacted out of the data the patch can "
        "read. numpy is available. Use this when the change you want is not a parameter."),
    "UNEXPLORED_in_organiser_priority_order": [
        "1. Replace the loss. Pointwise logloss is misaligned with ranking metrics; "
        "pairwise (BPR) or listwise (softmax over the user's impressions). The "
        "organisers call this the most promising direction.",
        "2. User history sequences -- completely untouched (DIN/SIM-style).",
        "3. Multi-objective: is_click / is_like / is_follow / is_comment / is_forward / "
        "play_time_ms as auxiliary tasks for long_view.",
        "4. Watch-time modelling as censored regression.",
        "5. Bigger models (DeepFM/DCN) -- AFTER 1-4, capacity is not the bottleneck.",
        "6. Time features and train-test drift (hourmin, date).",
        "7. log_random as a second, unbiased validation set."],
}


def _sig(exc: BaseException) -> str:
    import hashlib
    tb = traceback.extract_tb(exc.__traceback__)
    frame = f"{tb[-1].filename}:{tb[-1].name}" if tb else "?"
    msg = "".join(ch for ch in str(exc)[:120] if not ch.isdigit())
    return hashlib.sha1(f"{type(exc).__name__}|{frame}|{msg}".encode()).hexdigest()[:8]


def _cfg_line(cfg: Config, primary: float) -> str:
    if cfg.model == "patch":
        from agent.patch import patch_hash
        note = (cfg.patch_code.strip().splitlines() or [""])[0][:46]
        return (f"patch {patch_hash(cfg.patch_code)[:10]} {note:<46s} "
                f"valid_primary={primary:.5f}")
    bits = [cfg.model]
    if cfg.model == "fm":
        bits += [f"loss={cfg.loss}", f"k={cfg.k}", f"lr={cfg.lr:g}", f"l2={cfg.l2:g}",
                 f"ep={cfg.epochs}", f"bs={cfg.batch}"]
        if cfg.loss == "pairwise":
            bits.append(f"neg={cfg.neg_per_pos}")
        if cfg.loss == "listwise":
            bits.append(f"T={cfg.listwise_temp:g}")
        if cfg.aux_weights:
            bits.append("aux=" + ",".join(f"{k}:{v:g}" for k, v in sorted(cfg.aux_weights.items())))
        if tuple(cfg.fields) != tuple(A.BASELINE_FIELDS):
            bits.append("fields=" + "+".join(cfg.fields))
    return f"{' '.join(bits):<74s} valid_primary={primary:.5f}"


def population_digest(pop: dict, incumbent_hash: str, keep: int = 12) -> dict:
    rows = [(h, d, d["primary"]) for h, d in pop.items()]
    best = sorted(rows, key=lambda r: -r[2])[: keep // 2]
    recent = list(rows)[-(keep // 2):]
    seen, lines = set(), []
    for h, d, pr in best + recent:
        if h in seen:
            continue
        seen.add(h)
        lines.append(_cfg_line(d["cfg"], pr)
                     + ("  <== CURRENT INCUMBENT" if h == incumbent_hash else ""))
    return {"total_evaluated": len(rows), "shown": len(lines), "configs": lines}


def build_parser() -> argparse.ArgumentParser:
    """Exposed so the UI's allowlist can be tested against the REAL choices rather than
    a copy of them. A drift between the two is then a test failure at build time instead
    of an argparse error at iteration 1 of a launched run."""
    ap = argparse.ArgumentParser(description="autonomous MLE loop, official task")
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--proposer", default="random",
                    choices=["random", "gemini", "ollama", "groq", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--thinking-level", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="SEARCH seed. Seeds the driver's RNG and the proposer's "
                         "sampling; it does NOT seed the model. See --model-seed.")
    ap.add_argument("--model-seed", type=int, default=0,
                    help="MODEL seed. Reaches Config.seed and so the fit's RNG "
                         "(strategy/ranker.py). Default 0 is exactly what every run "
                         "before this flag used, so behaviour is unchanged.")
    ap.add_argument("--epsilon-greedy", type=float, default=0.25)
    ap.add_argument("--max-seconds", type=float, default=5400.0)
    ap.add_argument("--max-tokens", type=int, default=5_000_000)
    ap.add_argument("--convergence", default="official", choices=["official", "off"],
                    help="'official' applies the organisers' epsilon/N and is the only "
                         "setting a SUBMISSION run may use. 'off' runs the full iteration "
                         "budget and exists to answer diagnostic questions about search "
                         "behaviour -- its result is not a submittable converged result, "
                         "and the ledger records which was used.")
    ap.add_argument("--start-loss", default="pointwise",
                    choices=["pointwise", "pairwise", "listwise"],
                    help="starting incumbent's loss. The SUBMISSION run starts at "
                         "'pointwise' so the agent must find the improvement itself. "
                         "Starting elsewhere is for experiments that ask what the agent "
                         "does once the cheap win is already taken.")
    ap.add_argument("--patch-isolation", default="seatbelt",
                    choices=["seatbelt", "docker", "subprocess", "none"],
                    help="sandbox for LLM-written patches. `none` runs generated code "
                         "with NO boundary and is never the right answer outside a "
                         "debug session.")
    ap.add_argument("--patch-seconds", type=float, default=900.0,
                    help="wall-clock cap on a single generated patch")
    ap.add_argument("--allow-random-log", action="store_true",
                    help="permit training on log_random. The kit endorses it as an "
                         "unbiased VALIDATION set; using it to TRAIN is a rules "
                         "judgement, so it stays opt-in and refusals are recorded.")
    return ap


def main() -> None:
    a = build_parser().parse_args()

    t0 = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    # Assigned HERE rather than beside run_start: evaluate() records a patch failure
    # under this id, and the STARTING incumbent is evaluated before run_start is
    # appended. With the assignment further down, a patch used as the starting point
    # raised NameError inside its own failure handler -- unreachable while the
    # incumbent is always typed, and a crash the moment a code-first champion is the
    # starting point. Same value either way; t0 is assigned directly above.
    run_id = f"official_{a.proposer}_s{a.seed}_{int(t0)}"
    print(f"[loop] task=official proposer={a.proposer} iterations={a.iterations}")
    if a.convergence == "official":
        print(f"[loop] convergence is the ORGANISERS': epsilon={OFFICIAL_EPSILON} "
              f"N={OFFICIAL_N} (their FM seed std {OFFICIAL_FM_SEED_STD})")
    else:
        print(f"[loop] CONVERGENCE DISABLED -- diagnostic run, NOT submittable. "
              f"Running the full budget of {a.iterations} iterations.")

    # Preconditions live in core/handshake.py so they are shared with
    # make_official_submission.py -- the graded path had the same gap and needed it more.
    from core.handshake import HandshakeError, require_task, require_verified_chain
    ping = request({"op": "ping"})
    try:
        require_task(ping, "official")
        require_verified_chain(ping)
    except HandshakeError as e:
        raise SystemExit(str(e)) from None
    print(f"[loop] scorer task={ping['task']} manifest {str(ping.get('manifest_sha256',''))[:16]} "
          f"ledger {Path(str(ping.get('ledger') or '?')).name}")

    from core.task import get as get_task
    spec = get_task("kuairand_pure")
    splits, edges = load_for_strategy()
    if splits["test"].label is not None:
        raise SystemExit("refusing to run: test labels reached the driver")
    print(f"[loop] train {len(splits['train']):,} / valid {len(splits['valid']):,} rows; "
          f"test labels absent from this process")

    outdir = ROOT / "runs" / "official"
    outdir.mkdir(parents=True, exist_ok=True)
    stats = {"proposed": 0, "rejected_schema": 0, "rejected_policy": 0, "duplicate": 0,
             "evaluated": 0, "improved": 0, "crashed": 0, "tokens_in": 0,
             "tokens_out": 0, "tokens_thinking": 0, "interventions": 0,
             "unrecorded": 0, "inspected": 0}

    def rec(kind: str, body: dict):
        r = request({"op": "record", "kind": kind, "body": body})
        if not r.get("ok"):
            stats["unrecorded"] += 1
            print(f"       LEDGER RECORD FAILED ({kind}): {r.get('error')}")

    def record_curve(cfg: Config, pl: dict, info: dict) -> None:
        """Record the per-epoch training telemetry for one evaluation.

        These numbers are the STRATEGY's own, not the scorer's: the ledger's `event`
        kind is attested as receipt-order only, which is exactly the right status for
        them. Keeping them out of the signed `score` payload is deliberate -- that
        payload means "the scorer measured this", and a self-reported curve does not.

        `self_reported_best` is kept alongside the scorer's `primary` precisely so the
        two can be compared: for a single fit they must agree, because the ranker
        restores its best epoch before predicting. A gap means early stopping and
        scoring disagree about which checkpoint won, which would be a real bug.
        """
        body = {"what": "train_curve", "config_hash": cfg.hash(),
                "scorer_primary": pl["metrics"]["primary"],
                "split": pl.get("split")}
        for k in ("model", "loss", "epochs_run", "valid_primary_best",
                  "valid_history", "train_loss_history", "ensemble_seeds",
                  "ensemble_combine", "member_valid_primary", "member_valid_history",
                  "member_train_loss_history", "fit_seconds", "patch_hash"):
            v = info.get(k)
            if v is not None:
                body["self_reported_best" if k == "valid_primary_best" else k] = v
        rec("event", body)

    def evaluate(cfg: Config) -> dict | None:
        """Fit, write scores, get a SIGNED valid score. Test is never scored here.

        Two paths, one contract. A TYPED config is fitted in-process by strategy/ranker
        (fast, and the whole reason parameter changes dedup for free). A PATCH config is
        LLM-written code, so it runs behind the sandbox -- a failing patch comes back as
        data, never as an exception that ends the run.
        """
        if cfg.model == "patch":
            from agent.patch import run_sandboxed
            rep = run_sandboxed(cfg.patch_code, cfg.patch_params,
                                backend=a.patch_isolation, max_seconds=a.patch_seconds)
            if not rep.get("ok"):
                rec("event", {"run_id": run_id, "what": "patch_failed",
                              "patch_hash": rep.get("patch_hash", "")[:16],
                              "error_type": rep.get("error_type"),
                              "error": str(rep.get("error"))[:600],
                              "contract_violation": rep.get("contract_violation"),
                              "isolation": rep.get("isolation")})
                print(f"       PATCH FAILED [{rep.get('error_type')}] "
                      f"{str(rep.get('error'))[:110]}")
                return None
            f = Path(rep["dev_scores"])
            info = {"model": "patch", "patch_hash": rep["patch_hash"],
                    "fit_seconds": rep.get("fit_seconds"), "patch_info": rep.get("info")}
            np.save(outdir / f"loop_{cfg.hash()[:12]}_test.npy",
                    np.load(rep["heldout_scores"]))
            r = request({"op": "score_official", "scores": str(f), "split": "valid",
                         "config_hash": cfg.hash(), "isolation": rep.get("isolation"),
                         "seed": cfg.seed})
            if not r.get("ok"):
                print(f"       scorer REFUSED: {r['error']}")
                return None
            pl = r["entry"]["payload"]
            record_curve(cfg, pl, info)
            return {"primary": pl["metrics"]["primary"], "payload": pl, "info": info}

        scores, info = fit_and_score(splits, edges, cfg)
        f = outdir / f"loop_{cfg.hash()[:12]}_valid.npy"
        np.save(f, scores["valid"])
        r = request({"op": "score_official", "scores": str(f), "split": "valid",
                     "config_hash": cfg.hash(), "isolation": "none", "seed": cfg.seed})
        if not r.get("ok"):
            print(f"       scorer REFUSED: {r['error']}")
            return None
        pl = r["entry"]["payload"]
        # keep the test-split scores on disk unscored, so the final submission can be
        # produced from the designated checkpoint without refitting
        np.save(outdir / f"loop_{cfg.hash()[:12]}_test.npy", scores["test"])
        record_curve(cfg, pl, info)
        return {"primary": pl["metrics"]["primary"], "payload": pl, "info": info}

    # seed= was previously left at Config's default, so the model was fitted at seed 0
    # whatever --seed said. The default is still 0, so this changes no past behaviour.
    incumbent = Config(loss=a.start_loss, seed=a.model_seed)
    print(f"[loop] evaluating the starting incumbent (the official baseline) ...")
    inc = evaluate(incumbent)
    if inc is None:
        raise SystemExit("the starting incumbent could not be scored")

    # Do the two processes agree on WHAT they are scoring, order included? See
    # core/handshake.py for why a row-count check is not enough.
    from core.handshake import require_same_data
    mine = splits["valid"].content_sha256()
    try:
        require_same_data(mine, inc["payload"].get("split_content_sha256"), "valid")
    except HandshakeError as e:
        raise SystemExit(str(e)) from None
    print(f"[loop] data agreement confirmed: valid split {mine[:16]} on both sides")
    print(f"       valid primary {inc['primary']:.5f}  "
          f"(official {OFFICIAL_VALID['fm_official']:.4f}, "
          f"delta {inc['primary']-OFFICIAL_VALID['fm_official']:+.5f})")

    pop = {incumbent.hash(): {"cfg": incumbent, "primary": inc["primary"]}}
    key = (os.environ.get("TRACK2_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or "")
    proposer = make_proposer(a.proposer, seed=a.seed, api_key=key, model=a.model,
                             thinking_level=a.thinking_level, actions=A)
    print(f"[loop] proposer backend: {proposer.name}")
    # The control arm cannot draw from every action the LLM arm can see (there is no
    # uniform distribution over program texts). That asymmetry belongs in the ledger,
    # not in a docstring, because it bounds what "the LLM beat random" can mean.
    coverage = getattr(proposer, "coverage_note", lambda: None)()
    if coverage:
        print(f"[loop] {coverage}")

    rec("run_start", {"run_id": run_id, "task": "official", "proposer": proposer.name,
                      "proposer_coverage": coverage,
                      "start_loss": a.start_loss,
                      "patch_isolation": a.patch_isolation,
                      "model": a.model or "", "seed": a.seed,
                      "model_seed": a.model_seed,
                      "iterations_requested": a.iterations,
                      "epsilon": OFFICIAL_EPSILON, "N": OFFICIAL_N,
                      "epsilon_source": "organisers (baseline_scores.json)",
                      "convergence_mode": a.convergence,
                      "submittable": a.convergence == "official",
                      "start_config_hash": incumbent.hash(),
                      "start_valid_primary": inc["primary"],
                      "allow_random_log": a.allow_random_log})

    action_summary: dict = {}
    failures: dict = {}
    #: diagnostic name -> result. Grows as the agent asks questions, and is put into every
    #: subsequent prompt so an Inspect actually informs the next proposal.
    findings: dict = {}
    # TWO SEPARATE RULES, which an earlier version of this file conflated:
    #
    #   incumbent selection -- OURS. Move to any config that scores better on
    #     validation, by any margin. This is hill climbing and it must be allowed to
    #     chain small steps. Requiring each step to clear epsilon stalled the search:
    #     measured, pairwise BPR gained +0.00188, epsilon is 0.002, so the incumbent
    #     never moved and the agent re-proposed the same action five times.
    #
    #   convergence -- THEIRS. "validation score has not improved by more than epsilon
    #     over the last N consecutive iterations". That is a statement about the BEST
    #     SCORE SO FAR, not about whether any individual step was accepted.
    #
    # The submitted checkpoint is the validation-best one either way, which is what the
    # brief asks for.
    converged, policy_refusals = None, {}
    best_hash, best_primary = incumbent.hash(), inc["primary"]
    # Convergence is a WINDOW, not a per-step test. "validation score has not improved
    # by more than epsilon over the last N consecutive iterations" compares the best
    # now against the best N iterations ago -- so N sub-epsilon steps that ACCUMULATE
    # past epsilon correctly keep the run alive.
    #
    # The first version compared against "best at the last >epsilon jump", which could
    # never be reset by a smaller gain: measured, a +0.00188 step left the reference
    # pinned and the run died after exactly 3 evaluations regardless of progress.
    from collections import deque
    best_window = deque([inc["primary"]], maxlen=OFFICIAL_N + 1)

    for it in range(1, a.iterations + 1):
        cap = None
        if time.perf_counter() - t0 > a.max_seconds:
            cap = f"wall-clock {a.max_seconds}s"
        elif stats["tokens_in"] + stats["tokens_out"] > a.max_tokens:
            cap = f"tokens {a.max_tokens}"
        if cap:
            converged = f"budget: {cap}"
            print(f"[loop] CAP REACHED ({cap}) at iteration {it}")
            break
        # THE ORGANISERS' convergence rule, not ours
        if (a.convergence == "official" and len(best_window) > OFFICIAL_N
                and best_window[-1] - best_window[0] <= OFFICIAL_EPSILON):
            converged = (f"official rule: best validation primary improved by only "
                         f"{best_window[-1] - best_window[0]:+.5f} over the last "
                         f"N={OFFICIAL_N} evaluated iterations, which does not exceed "
                         f"epsilon={OFFICIAL_EPSILON}")
            print(f"[loop] CONVERGED ({converged}) at iteration {it}")
            break

        restart = rng.random() < a.epsilon_greedy and len(pop) > 1
        parent_hash = str(rng.choice(list(pop))) if restart else incumbent.hash()
        parent = pop[parent_hash]["cfg"]

        state = {"min_effect": {"primary": OFFICIAL_EPSILON},
                 "knowledge": SEED_KNOWLEDGE,
                 "incumbent": asdict(parent),
                 "incumbent_metrics": {"valid_primary": pop[parent_hash]["primary"]},
                 "iteration": it, "action_summary": action_summary,
                 "population": population_digest(pop, incumbent.hash()),
                 "failure_registry": failures, "findings": findings,
                 "parent_node_id": parent_hash[:12]}
        try:
            env, usage = proposer.propose(state)
        except Exception as e:
            sg = _sig(e)
            f = failures.setdefault(sg, {"count": 0, "type": type(e).__name__,
                                         "msg": str(e)[:200], "actions": ["<proposer>"]})
            f["count"] += 1; stats["crashed"] += 1
            rec("event", {"run_id": run_id, "iteration": it, "what": "proposer_error",
                          "signature": sg, "error": str(e)[:400]})
            print(f"  it{it:>3} PROPOSER-ERROR [{sg}] {str(e)[:90]}")
            if f["count"] >= 5:
                converged = f"proposer failing repeatedly ({sg})"
                print(f"[loop] CIRCUIT BREAKER: {converged}")
                break
            continue
        stats["proposed"] += 1
        stats["tokens_in"] += (usage.get("input_tokens", 0)
                               + usage.get("cache_creation_input_tokens", 0)
                               + usage.get("cache_read_input_tokens", 0))
        stats["tokens_out"] += usage.get("output_tokens", 0)
        stats["tokens_thinking"] += usage.get("thinking_tokens", 0)

        try:
            action = A.validate(env, OFFICIAL_EPSILON, parent)
            cand = A.apply(parent, action)
        except A.ActionError as e:
            stats["rejected_schema"] += 1
            rec("event", {"run_id": run_id, "iteration": it, "what": "schema_rejected",
                          "error": str(e)[:500],
                          "raw_action_type": str(env.get("action_type"))[:64]})
            print(f"  it{it:>3} REFUSED   {str(e)[:96]}")
            continue

        at = action["action_type"]
        rec_at = action_summary.setdefault(at, {"tried": 0, "improved": 0, "duplicate": 0})
        rec_at["tried"] += 1

        # ---- Inspect: a question about the data, not a change to the model.
        # Costs no fit, so it does not count as an evaluation and must not touch the
        # convergence window -- that window measures score progress and an Inspect makes
        # no score. It IS capped at once per diagnostic: the answer does not change, so
        # asking twice is a duplicate in the same sense a repeated config is.
        if at == "Inspect":
            name = action["params"]["diagnostic"]
            base = {"run_id": run_id, "iteration": it, "task": "official",
                    "action_type": at, "params": action["params"],
                    "rationale": action["rationale"],
                    "parent_config_hash": parent_hash,
                    "candidate_config_hash": parent_hash,
                    "selection": "inspect", "proposer": proposer.name}
            if name in findings:
                stats["duplicate"] += 1; rec_at["duplicate"] += 1
                rec("proposal", {**base, "disposition": "duplicate"})
                print(f"  it{it:>3} DUPLICATE Inspect({name}) -> already answered")
                continue
            from agent import inspect as INS
            try:
                result = INS.run(name, splits, spec)
            except Exception as e:
                stats["crashed"] += 1
                rec("event", {"run_id": run_id, "iteration": it,
                              "what": "inspect_failed", "diagnostic": name,
                              "error": str(e)[:400]})
                print(f"  it{it:>3} INSPECT FAILED {name}: {str(e)[:80]}")
                continue
            findings[name] = result
            rec("proposal", {**base, "disposition": "answered"})
            rec("event", {"run_id": run_id, "iteration": it, "what": "inspect_result",
                          "diagnostic": name, "result": json.loads(json.dumps(result))})
            stats["inspected"] += 1
            head = {k: v for k, v in result.items()
                    if not isinstance(v, dict) and k != "reading"}
            print(f"  it{it:>3} INSPECT {name:18s} {json.dumps(head)[:96]}")
            continue

        if cand.use_random_log and not a.allow_random_log:
            disposition = "policy_refused"
        elif cand.hash() in pop:
            disposition = "duplicate"
        else:
            disposition = "evaluated"

        rec("proposal", {"run_id": run_id, "iteration": it, "task": "official",
                         "action_type": at, "params": action["params"],
                         "rationale": action["rationale"],
                         "parent_config_hash": parent_hash,
                         "candidate_config_hash": cand.hash(),
                         "selection": "restart" if restart else "mutate",
                         "disposition": disposition, "proposer": proposer.name,
                         "parent_valid_primary": round(pop[parent_hash]["primary"], 6),
                         "incumbent_valid_primary": round(pop[incumbent.hash()]["primary"], 6)})

        if disposition == "policy_refused":
            stats["rejected_policy"] += 1
            policy_refusals["use_random_log"] = policy_refusals.get("use_random_log", 0) + 1
            print(f"  it{it:>3} POLICY-REFUSED {at}: training on log_random is not "
                  f"enabled (--allow-random-log); the kit endorses it for VALIDATION")
            continue
        if disposition == "duplicate":
            stats["duplicate"] += 1; rec_at["duplicate"] += 1
            print(f"  it{it:>3} DUPLICATE {at} -> config already evaluated")
            continue

        try:
            res = evaluate(cand)
        except Exception as e:
            s = _sig(e)
            f = failures.setdefault(s, {"count": 0, "type": type(e).__name__,
                                        "msg": str(e)[:150], "actions": []})
            f["count"] += 1
            if at not in f["actions"]:
                f["actions"].append(at)
            stats["crashed"] += 1
            rec("event", {"run_id": run_id, "iteration": it, "what": "strategy_crashed",
                          "action_type": at, "signature": s, "error": str(e)[:400]})
            print(f"  it{it:>3} CRASHED   {at} -> {type(e).__name__} [{s}] {str(e)[:60]}")
            continue
        if res is None:
            stats["crashed"] += 1
            continue
        stats["evaluated"] += 1
        pop[cand.hash()] = {"cfg": cand, "primary": res["primary"]}

        delta = res["primary"] - pop[incumbent.hash()]["primary"]
        took_step = delta > 0.0                       # incumbent selection: ours
        if took_step:
            incumbent = cand
            stats["improved"] += 1; rec_at["improved"] += 1
        if res["primary"] > best_primary:
            best_hash, best_primary = cand.hash(), res["primary"]
        best_window.append(best_primary)      # one entry per EVALUATED iteration
        window_gain = best_window[-1] - best_window[0]
        tag = ("STEP" if took_step else "no")
        tag += (f" (best {best_primary:.5f}, window {window_gain:+.5f} vs eps "
                f"{OFFICIAL_EPSILON})")
        print(f"  it{it:>3} {'restart' if restart else 'mutate ':7s} {at:17s} "
              f"valid {res['primary']:.5f} d={delta:+.5f} -> {tag}")

    el = time.perf_counter() - t0
    if converged is None:
        converged = f"iteration limit ({a.iterations})"
    rec("run_end", {"run_id": run_id, "termination": converged,
                    "epsilon_source": "organisers", "epsilon": OFFICIAL_EPSILON,
                    "N": OFFICIAL_N,
                    "rule_note": "incumbent moves on any validation gain (hill "
                                 "climbing); convergence uses the organisers' "
                                 "epsilon/N on the BEST score so far",
                    "seconds": round(el, 1), "stats": stats,
                    "policy_refusals": policy_refusals,
                    "final_incumbent_config_hash": incumbent.hash(),
                    "final_valid_primary": round(pop[incumbent.hash()]["primary"], 6),
                    "best_config_hash": best_hash,
                    "best_valid_primary": round(best_primary, 6),
                    "population_size": len(pop), "action_summary": action_summary,
                    "diagnostics_run": sorted(findings)})

    print("\n" + "=" * 78)
    print(f"loop finished in {el:.0f}s  ({el/max(stats['evaluated'],1):.0f}s per evaluation)")
    print(f"  TERMINATION        {converged}")
    for k, v in stats.items():
        print(f"  {k:18s} {v}")
    print(f"  TOKENS REPORTED    {stats['tokens_in']+stats['tokens_out']+stats['tokens_thinking']:,}")
    if findings:
        print(f"\n  diagnostics run     {sorted(findings)}")
    print(f"\n  official baseline  valid {OFFICIAL_VALID['fm_official']:.4f}")
    print(f"  final incumbent    valid {pop[incumbent.hash()]['primary']:.5f}  "
          f"delta {pop[incumbent.hash()]['primary']-OFFICIAL_VALID['fm_official']:+.5f}")
    print(f"  best found         valid {best_primary:.5f}  "
          f"delta {best_primary-OFFICIAL_VALID['fm_official']:+.5f}  cfg {best_hash[:12]}")
    print(f"  submission source  runs/official/loop_{best_hash[:12]}_test.npy")

    # Write the report as part of the run, not as a step someone has to remember. A
    # report that only exists when a human runs a second command is a report that is
    # stale exactly when it matters. Failing to render must never fail the run -- the
    # scored artifacts are already on disk and signed by this point.
    try:
        from agent.official_report import build as build_report, load as load_ledger
        # ASK the scorer which file it is writing. Reading TRACK2_LEDGER here was wrong:
        # the scorer resolves the ledger from ITS OWN environment at startup, so a driver
        # started with a different value would render a path that does not exist. The
        # scorer owns the ledger; the only correct source for its location is the scorer.
        pong = request({"op": "ping"})
        led = Path(pong.get("ledger") or (ROOT / "runs" / "ledger.jsonl"))
        rp = ROOT / "runs" / f"report_{led.stem}.html"
        rp.write_text(build_report(load_ledger(led)), encoding="utf-8")
        print(f"  report             {rp.relative_to(ROOT)}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  report             NOT WRITTEN ({type(exc).__name__}: {exc})")
    print("=" * 78)


if __name__ == "__main__":
    main()
