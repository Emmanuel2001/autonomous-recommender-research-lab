"""The loop. propose -> validate -> apply -> train -> score -> gate -> record.

The driver holds NO signing key. It talks to the scorer over the unix socket, so
every number in the ledger was produced by a process the driver cannot forge.

Controller: epsilon-greedy with restart. The population records EVERY evaluated
config (per the DGM correction -- gating admission on significance guarantees an
empty archive, since a stepping stone is by definition not yet significantly
better). The gate decides the INCUMBENT and nothing else.

ISOLATION: --isolation selects the boundary the strategy runs behind, and it
defaults to the strongest one this machine can enforce. Under `seatbelt` the
strategy process cannot open the eval-window csv, the ledger, or the signing key,
and cannot write anywhere but its own output directory -- measured, not asserted:
`python3 -m core.isolation probe` runs a hostile strategy for real and records
what the kernel refused. `none` keeps the in-process path (one data load instead
of one per iteration) and is what every number measured before 2026-08-26 used;
it is still available, and the choice is recorded in every signed entry as
`isolation_claimed` -- claimed, because the scorer cannot verify it. The probe is
the verifiable part.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from agent import actions as A
from agent.proposer import make as make_proposer
from core import isolation as I
from core import splits as S
from core.ledger import read_payloads
from core.scorer import LEDGER_PATH, request
from strategy.mle_strategy import Config, run as run_strategy

SEED_KNOWLEDGE = {
    "lambda_plateau": "EASE lambda in [250,1000] is a measured PLATEAU; l=500 vs l=250 "
                      "is indistinguishable (p=0.36 ndcg, p=0.29 recall). Do not spend "
                      "iterations there.",
    "model_ladder_recall@50": {"ease_l500": 0.2549, "als": 0.2307, "itemknn": 0.2175,
                               "bpr_mf": 0.1616, "popularity": 0.1306},
    "unexplored": "Every model measured so far uses ONLY (user, video, is_click). "
                  "31 user features, 12 video features and 11 other feedback signals "
                  "are untouched. long_view is 33.7% dense, nearly as dense as click.",
    "noise": "sigma_seed: BPR-MF .00058 ndcg / .00113 recall; EASE exactly 0 "
             "(closed form). sigma_eval (user bootstrap) .00116 / .00261.",
    "caution": "video_features_statistic (52 aggregate count columns) may have been "
               "computed over the whole period INCLUDING the eval window. Unverified "
               "leakage risk -- do not use until checked.",
}


def _sig(exc: BaseException) -> str:
    import hashlib
    tb = traceback.extract_tb(exc.__traceback__)
    frame = f"{tb[-1].filename}:{tb[-1].name}" if tb else "?"
    msg = "".join(ch for ch in str(exc)[:120] if not ch.isdigit())
    return hashlib.sha1(f"{type(exc).__name__}|{frame}|{msg}".encode()).hexdigest()[:8]


def _cfg_line(cfg, pl) -> str:
    """One compact line per evaluated config. Only the fields that matter for the
    active model, so the line stays short and comparable."""
    m = pl["metrics"]
    comp = 0.5 * (m["ndcg@10"] + m["recall@50"])
    bits = [cfg.model]
    if cfg.model == "ease":
        bits.append(f"lam={cfg.ease_lambda:g}")
        bits.append(f"content_a={cfg.content_alpha:g}")
        if set(cfg.signal_weights) != {"is_click"}:
            bits.append("sig=" + ",".join(f"{k}:{v:g}" for k, v in
                                          sorted(cfg.signal_weights.items())))
    elif cfg.model == "itemknn":
        bits.append(f"k={cfg.knn_k}")
    elif cfg.model == "als":
        bits.append(f"dim={cfg.dim} iters={cfg.als_iters}")
    elif cfg.model == "bpr_mf":
        bits.append(f"dim={cfg.dim} ep={cfg.epochs} lr={cfg.lr:g}")
    if cfg.mask != "exclude_train_seen":
        bits.append(f"mask={cfg.mask}")
    return f"{' '.join(bits):<58s} composite={comp:.5f}"


def population_digest(population: dict, incumbent_hash: str, keep: int = 12) -> dict:
    """What the proposer sees about prior work.

    Aggregating only by ACTION TYPE was not enough -- measured: gemini-3.5-flash-lite
    proposed SetSignalWeights three times running and iterations 2 and 3 were caught
    as duplicates, i.e. it re-proposed configs whose results it had already been
    given. The type-level summary told it "SetSignalWeights: tried 1" without saying
    WHICH weights or WHAT happened.

    Bounded rather than flat: best `keep/2` plus most recent `keep/2`, with the total
    stated, so context does not grow linearly with iterations.
    """
    rows = [(h, d, 0.5 * (d["pl"]["metrics"]["ndcg@10"] + d["pl"]["metrics"]["recall@50"]))
            for h, d in population.items()]
    best = sorted(rows, key=lambda r: -r[2])[: keep // 2]
    recent = list(rows)[-(keep // 2):]
    seen, lines = set(), []
    for h, d, _ in best + recent:
        if h in seen:
            continue
        seen.add(h)
        tag = "  <== CURRENT INCUMBENT" if h == incumbent_hash else ""
        lines.append(_cfg_line(d["cfg"], d["pl"]) + tag)
    return {"total_evaluated": len(rows), "shown": len(lines), "configs": lines}


# ---------------------------------------------------------------- parent selection
#
# THE ABLATION THIS EXISTS FOR
# ----------------------------
# The DGM claim we borrow is that keeping an ARCHIVE of every evaluated variant and
# branching from it beats a single CHAIN of successive improvements, because a
# stepping stone that looks like a regression can still be the best parent. We had
# been asserting that by analogy and running only epsilon-greedy. These three
# policies make it measurable on this task, with the proposer held FIXED so the
# controller is the only thing that varies:
#
#   chain           always mutate the incumbent. Pure hill climbing, no archive.
#   epsilon_greedy  mutate the incumbent; with probability epsilon branch from a
#                   uniformly random archive member.
#   archive         always select from the archive, weighted by performance and by
#                   how little that node has been explored.
#
# `archive` is DGM-INSPIRED, not DGM. Sakana's published selection rule is not
# reimplemented here, so it is not claimed: this is a softmax over composite score
# with an explicit under-exploration bonus, and the parameters are ours.
SELECTIONS = ("chain", "epsilon_greedy", "archive")


def select_parent(population: dict, incumbent_hash: str, children: dict,
                  policy: str, epsilon: float, rng) -> tuple[str, str]:
    """Returns (parent_hash, tag). `children` maps a hash to how many times it has
    been used as a parent."""
    if policy == "chain":
        return incumbent_hash, "chain"
    if len(population) == 1:
        # Nothing to branch to yet. The tag reports the ACTION taken, not the policy
        # name, so this must not read as "chain" -- the policy is still whatever was
        # asked for, the archive is simply too small to exercise it.
        return incumbent_hash, "mutate"

    if policy == "epsilon_greedy":
        if rng.random() < epsilon:
            return str(rng.choice(list(population))), "restart"
        return incumbent_hash, "mutate"

    if policy == "archive":
        hashes = list(population)
        comps = np.array([0.5 * (population[h]["pl"]["metrics"]["ndcg@10"]
                                 + population[h]["pl"]["metrics"]["recall@50"])
                          for h in hashes], dtype=np.float64)
        # Temperature is set from the SPREAD of the archive, not fixed: composites
        # here differ in the third decimal, so any constant temperature would either
        # flatten the softmax to uniform or collapse it onto the argmax depending on
        # the run. Scaling by the observed sd keeps the policy comparable across
        # runs and archives of different maturity.
        sd = float(comps.std())
        z = (comps - comps.max()) / (sd if sd > 1e-9 else 1.0)
        w = np.exp(z)
        # under-exploration bonus: a node used as a parent k times is damped by
        # 1/(1+k), so the archive keeps branching instead of hammering one node.
        w = w / np.array([1.0 + children.get(h, 0) for h in hashes])
        w = w / w.sum()
        pick = hashes[int(rng.choice(len(hashes), p=w))]
        return pick, ("archive-best" if pick == incumbent_hash else "archive-branch")

    raise ValueError(f"unknown selection policy {policy!r}; choose from {SELECTIONS}")


def rebuild_population(split: str) -> dict:
    """Derive the population from the LEDGER, not from driver memory.

    If this cannot reconstruct the run, the run is not auditable.
    """
    if not LEDGER_PATH.exists():
        return {}
    pop = {}
    for pl in read_payloads(LEDGER_PATH):
        if pl.get("kind") == "score" and pl.get("split") == split:
            pop[pl["config_hash"]] = {"metrics": pl["metrics"],
                                      "preds": pl.get("preds_sha256")}
    return pop


def main() -> None:
    ap = argparse.ArgumentParser(description="autonomous MLE loop")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--proposer", default="random",
                    choices=["random", "gemini", "ollama", "groq", "anthropic"])
    ap.add_argument("--model", default=None, help="backend model id")
    ap.add_argument("--thinking-level", default=None, help="gemini only")
    ap.add_argument("--epsilon", type=float, default=0.25,
                    help="restart probability (epsilon_greedy only)")
    ap.add_argument("--selection", default="epsilon_greedy", choices=list(SELECTIONS),
                    help="parent-selection policy. This is the archive-vs-chain "
                         "ablation axis; see select_parent.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--split", default="search_val")
    ap.add_argument("--start-model", default="ease")
    ap.add_argument("--start-lambda", type=float, default=500.0)
    ap.add_argument("--max-tokens", type=int, default=5_000_000)
    ap.add_argument("--max-evaluations", type=int, default=1000)
    ap.add_argument("--allow-log-random", action="store_true",
                    help="permit training on log_random. OFF by default: the brief's "
                         "split names only the two log_standard files, so this is a "
                         "rules question. Refusals are recorded.")
    ap.add_argument("--isolation", default="auto",
                    choices=["auto", "none", "subprocess", "seatbelt", "docker"],
                    help="boundary the strategy runs behind. `auto` picks the "
                         "strongest available and says so. `none` is in-process and "
                         "enforces nothing -- use it only to reproduce numbers "
                         "measured before isolation existed.")
    ap.add_argument("--patience", type=int, default=8,
                    help="DECLARED convergence rule: stop after this many consecutive "
                         "gated comparisons without an incumbent replacement. The "
                         "organisers' epsilon/N are unpublished, so we declare ours.")
    a = ap.parse_args()

    t_start = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    print(f"[loop] proposer={a.proposer} iterations={a.iterations} "
          f"selection={a.selection} epsilon={a.epsilon}")

    ping = request({"op": "ping"})
    if not ping.get("ok"):
        raise SystemExit("scorer is not reachable; start it in its own terminal first")
    print(f"[loop] scorer manifest {ping['manifest_sha256'][:16]}")

    iso = I.detect_backend() if a.isolation == "auto" else a.isolation
    sb = None
    train = users = None
    n_users_cap = 27285
    if iso == "none":
        # NOTE this path loads the eval csv: load_for_strategy -> build_manifest ->
        # _load_raw reads all three files. It does not RETURN a validation label,
        # but they are in this process's memory. That is exactly why isolation
        # exists, and why `none` is not the default.
        train, part = S.load_for_strategy()
        users = {"search_val": part.search_val_users,
                 "audit_val": part.audit_val_users}[a.split].astype(np.int64)
        n_items = part.manifest["n_items_catalog"]
        print(f"[loop] ISOLATION none -- IN-PROCESS, nothing is enforced. The label "
              f"quarantine here is a property of the code, not of the OS.")
        print(f"[loop] train={len(train):,} rows, ranking {len(users):,} users "
              f"over {n_items:,} items")
    else:
        sb = I.make_sandbox(iso, allow_log_random=a.allow_log_random)
        # n_items comes from the item catalogue, which is label-free. The driver
        # deliberately does NOT call load_for_strategy here: that would read the
        # eval csv into this process for no reason.
        n_items = int(__import__("pandas").read_csv(
            S.DATA / "video_features_basic_pure.csv", usecols=["video_id"]
        ).video_id.nunique())
        print(f"[loop] ISOLATION {iso} -- {sb.describe()['enforced_by']}")
        if iso == "subprocess":
            print("[loop] WARNING: `subprocess` prepares paths but ENFORCES NOTHING. "
                  "The real eval path still resolves from inside the strategy. "
                  "Run `python3 -m core.isolation probe` to see this measured.")
        print(f"[loop] strategy sees {len(sb.data.present)} files; "
              f"{len(sb.denied)} paths read-denied; ranking every user over "
              f"{n_items:,} items")

    nf = json.loads((ROOT / "experiments" / "noise_floor.json").read_text())
    min_effect = {m: nf["splits"]["search_val"][m]["mde"]["1"]
                  for m in ("ndcg@10", "recall@50")}
    # the gate decides on the composite; the per-metric floors are diagnostics
    min_effect["composite_mean"] = nf["composite_calibration"]["min_effect_composite"]

    stats = {"proposed": 0, "rejected_schema": 0, "rejected_policy": 0,
             "duplicate": 0, "evaluated": 0, "improved": 0, "crashed": 0,
             "tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0,
             "interventions": 0, "unrecorded": 0}

    def evaluate(cfg: Config) -> dict | None:
        """Train, rank, and get a SIGNED score. Returns the payload or None."""
        if sb is None:
            ranked, _ = run_strategy(train, users, cfg, n_items, n_users_cap)
            out = ROOT / "runs" / f"loop_{cfg.hash()[:12]}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, users=users, items=ranked)
        else:
            out = sb.out_dir / f"loop_{cfg.hash()[:12]}.npz"
            # The child may see a different path than we do (docker bind mount), so
            # ask the sandbox to translate. The file still lands at `out` on the host.
            m = sb.run(["-m", "strategy.run_iteration",
                        "--config-json", json.dumps(asdict(cfg)),
                        "--split", a.split, "--out", sb.translate_out(out)])
            # The sandbox is a separate process, so verify it ran the config we
            # asked for rather than trusting that it did. A mismatch means the
            # config did not survive serialisation, and the score would be
            # attributed to the wrong experiment.
            if m["config_hash"] != cfg.hash():
                raise RuntimeError(f"sandbox ran a different config: got "
                                   f"{m['config_hash'][:12]}, asked for {cfg.hash()[:12]}")
            if not out.exists():
                # Catches a path-translation mistake immediately instead of letting
                # the scorer report a missing file with no explanation of why.
                raise RuntimeError(f"the sandbox reported success but {out} does not "
                                   f"exist on the host (backend={sb.backend}; the "
                                   f"child wrote to {sb.translate_out(out)})")
        r = request({"preds": str(out), "split": a.split,
                     "config_hash": cfg.hash(), "mask": cfg.mask,
                     "isolation": iso})
        if not r.get("ok"):
            print(f"       scorer REFUSED: {r['error']}")
            return None
        return {**r["entry"]["payload"], "_preds": str(out)}

    def rec(kind: str, body: dict) -> str:
        """Ask the scorer to countersign an orchestrator assertion.

        The driver holds no key, so it cannot sign anything itself. Failure here is
        NOT fatal to the run -- an unrecorded iteration is a gap in the audit trail,
        which must be visible rather than silently tolerated, so it is counted.
        """
        r = request({"op": "record", "kind": kind, "body": body})
        if not r.get("ok"):
            stats["unrecorded"] = stats.get("unrecorded", 0) + 1
            print(f"       LEDGER RECORD FAILED ({kind}): {r.get('error')}")
            return ""
        return r["entry"]["seq"]

    incumbent = Config(model=a.start_model, ease_lambda=a.start_lambda, seed=0)
    print(f"[loop] evaluating the starting incumbent ({incumbent.model}) ...")
    inc_pl = evaluate(incumbent)
    if inc_pl is None:
        raise SystemExit("the starting incumbent could not be scored")
    print(f"       ndcg@10={inc_pl['metrics']['ndcg@10']:.5f} "
          f"recall@50={inc_pl['metrics']['recall@50']:.5f}")

    population = {incumbent.hash(): {"cfg": incumbent, "pl": inc_pl}}
    import os
    api_key = (os.environ.get("TRACK2_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GROQ_API_KEY") or "")
    proposer = make_proposer(a.proposer, seed=a.seed, api_key=api_key,
                             model=a.model, thinking_level=a.thinking_level)
    print(f"[loop] proposer backend: {proposer.name}")
    action_summary: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    run_id = f"{a.proposer}_s{a.seed}_{int(t_start)}"
    rec("run_start", {"run_id": run_id, "proposer": proposer.name,
                      "model": a.model or "", "seed": a.seed,
                      "iterations_requested": a.iterations, "epsilon": a.epsilon,
                      "patience": a.patience, "split": a.split,
                      "selection": a.selection,
                      "isolation_claimed": iso, "allow_log_random": a.allow_log_random,
                      "start_config_hash": incumbent.hash(),
                      "min_effect": min_effect})
    since_improvement = 0
    children: dict[str, int] = {}          # parent hash -> times used as a parent
    converged_reason = None
    policy_refusals: dict[str, int] = {}

    for it in range(1, a.iterations + 1):
        # hard caps, checked BEFORE any spend. Exceeding one is a clean
        # converged-by-budget exit, not a crash.
        cap = None
        if time.perf_counter() - t_start > a.max_seconds:
            cap = f"wall-clock {a.max_seconds}s"
        elif stats["tokens_in"] + stats["tokens_out"] > a.max_tokens:
            cap = f"tokens {a.max_tokens}"
        elif stats["evaluated"] >= a.max_evaluations:
            cap = f"evaluations {a.max_evaluations}"
        if cap:
            converged_reason = f"budget: {cap}"
            print(f"[loop] CAP REACHED ({cap}); converged-by-budget at iteration {it}")
            break
        # DECLARED convergence: the organisers' epsilon/N are unpublished, so we
        # declare and report our own rather than pretend to theirs.
        if since_improvement >= a.patience:
            converged_reason = (f"plateau: {a.patience} consecutive gated comparisons "
                                f"without an incumbent replacement")
            print(f"[loop] CONVERGED ({converged_reason}) at iteration {it}")
            break

        parent_hash, tag = select_parent(population, incumbent.hash(), children,
                                         a.selection, a.epsilon, rng)
        children[parent_hash] = children.get(parent_hash, 0) + 1
        restart = tag in ("restart", "archive-branch")
        parent = population[parent_hash]["cfg"]

        state = {"min_effect": min_effect, "knowledge": SEED_KNOWLEDGE,
                 "incumbent": asdict(parent),
                 "incumbent_metrics": population[parent_hash]["pl"]["metrics"],
                 "iteration": it, "action_summary": action_summary,
                 "population": population_digest(population, incumbent.hash()),
                 "failure_registry": failures, "parent_node_id": parent_hash[:12]}

        try:
            env, usage = proposer.propose(state)
        except Exception as e:                    # ProposerError and anything below it
            sg = _sig(e)
            f = failures.setdefault(sg, {"count": 0, "type": type(e).__name__,
                                         "msg": str(e)[:200], "actions": ["<proposer>"]})
            f["count"] += 1
            stats["crashed"] += 1
            rec("event", {"run_id": run_id, "iteration": it, "what": "proposer_error",
                          "signature": sg, "type": type(e).__name__, "error": str(e)[:400]})
            print(f"  it{it:>3} PROPOSER-ERROR [{sg}] {str(e)[:100]}")
            if f["count"] >= 5:
                converged_reason = f"proposer failing repeatedly ({sg}); halting"
                print(f"[loop] CIRCUIT BREAKER: {converged_reason}")
                break
            continue
        stats["proposed"] += 1
        # The rubric scores TOTAL input + output tokens. Cache reads still count as
        # input, and THOUGHT tokens are generated tokens -- omitting either
        # under-reports the quantity we are graded on.
        stats["tokens_in"] += (usage.get("input_tokens", 0)
                              + usage.get("cache_creation_input_tokens", 0)
                              + usage.get("cache_read_input_tokens", 0))
        stats["tokens_out"] += usage.get("output_tokens", 0)
        stats["tokens_thinking"] += usage.get("thinking_tokens", 0)

        try:
            action = A.validate(env, min_effect)
            cand = A.apply(parent, action)
        except A.ActionError as e:
            stats["rejected_schema"] += 1
            # Recorded: "error / recovery per iteration" is a required column of the
            # Run & Iteration Logs, and a rejection is the commonest error there is.
            rec("event", {"run_id": run_id, "iteration": it, "what": "schema_rejected",
                          "error": str(e)[:400],
                          "raw_action_type": str(env.get("action_type"))[:64]})
            print(f"  it{it:>3} REJECTED  {str(e)[:88]}")
            continue

        at = action["action_type"]
        rec_at = action_summary.setdefault(at, {"tried": 0, "improved": 0, "duplicate": 0})
        rec_at["tried"] += 1

        # operator policy gate. The agent may PROPOSE a rules-ambiguous action; the
        # harness refuses unless it was explicitly enabled, and the refusal is
        # recorded so the write-up can say the agent found the lever and we declined.
        if cand.use_log_random and not a.allow_log_random:
            disposition = "policy_refused"
        elif cand.hash() in population:
            disposition = "duplicate"
        else:
            disposition = "evaluated"

        # PRE-REGISTRATION. Written before the evaluation, so expected_magnitude is
        # on the signed record before the delta that tests it. This is the only
        # place the rationale is persisted; before it existed the hypothesis was
        # validated and then dropped on the floor.
        rec("proposal", {
            "run_id": run_id, "iteration": it,
            "action_type": at, "params": action["params"],
            "rationale": action["rationale"],
            "parent_config_hash": parent_hash, "candidate_config_hash": cand.hash(),
            "selection": "restart" if restart else "mutate",
            "disposition": disposition,
            "proposer": proposer.name,
            # BOTH baselines, because they differ on a restart and confusing them
            # corrupts any calibration measurement. `parent_composite` is what the
            # proposer was shown and what expected_magnitude predicts about;
            # `incumbent_composite` is what the gate will compare against.
            "parent_composite": round(0.5 * (
                population[parent_hash]["pl"]["metrics"]["ndcg@10"]
                + population[parent_hash]["pl"]["metrics"]["recall@50"]), 6),
            "incumbent_composite": round(0.5 * (
                population[incumbent.hash()]["pl"]["metrics"]["ndcg@10"]
                + population[incumbent.hash()]["pl"]["metrics"]["recall@50"]), 6),
        })

        if disposition == "policy_refused":
            stats["rejected_policy"] += 1
            policy_refusals["use_log_random"] = policy_refusals.get("use_log_random", 0) + 1
            print(f"  it{it:>3} POLICY-REFUSED {at}: training on log_random is not "
                  f"enabled (--allow-log-random); rules status unconfirmed")
            continue

        if disposition == "duplicate":
            stats["duplicate"] += 1; rec_at["duplicate"] += 1
            print(f"  it{it:>3} DUPLICATE {at} -> config already evaluated")
            continue

        try:
            pl = evaluate(cand)
        except Exception as e:                       # a bad config must not end the run
            s = _sig(e)
            f = failures.setdefault(s, {"count": 0, "type": type(e).__name__,
                                        "msg": str(e)[:150], "actions": []})
            f["count"] += 1
            if at not in f["actions"]:
                f["actions"].append(at)
            stats["crashed"] += 1
            rec("event", {"run_id": run_id, "iteration": it, "what": "strategy_crashed",
                          "action_type": at, "signature": s, "type": type(e).__name__,
                          "error": str(e)[:400], "config_hash": cand.hash()})
            print(f"  it{it:>3} CRASHED   {at} -> {type(e).__name__} [{s}] {str(e)[:60]}")
            continue
        if pl is None:
            stats["crashed"] += 1
            continue
        stats["evaluated"] += 1

        # the gate decides the incumbent, and it is computed where the labels live
        g = request({"op": "compare",
                     "a_preds": [population[incumbent.hash()]["pl"]["_preds"]],
                     "b_preds": [pl["_preds"]], "split": a.split, "mask": cand.mask,
                     "min_effect": min_effect, "a_label": "incumbent", "b_label": at,
                     "a_config": incumbent.hash(), "b_config": cand.hash()})
        verdict = g["entry"]["payload"]["overall_verdict"] if g.get("ok") else "ungated"

        population[cand.hash()] = {"cfg": cand, "pl": pl}    # ALWAYS admitted
        if verdict == "improved":
            incumbent = cand
            stats["improved"] += 1; rec_at["improved"] += 1
            since_improvement = 0
        else:
            since_improvement += 1

        print(f"  it{it:>3} {tag:7s} {at:18s} ndcg={pl['metrics']['ndcg@10']:.5f} "
              f"recall={pl['metrics']['recall@50']:.5f}  -> {verdict.upper()}")

    # ------------------------------------------------------------------ summary
    el = time.perf_counter() - t_start
    best = max(population.values(), key=lambda d: d["pl"]["metrics"]["recall@50"])
    if converged_reason is None:
        converged_reason = f"iteration limit ({a.iterations}) reached"
    print("\n" + "=" * 76)
    print(f"loop finished in {el:.0f}s  ({el/max(stats['evaluated'],1):.1f}s per evaluation)")
    print(f"  TERMINATION        {converged_reason}")
    print(f"  selection          {a.selection}"
          + (f" (epsilon={a.epsilon})" if a.selection == "epsilon_greedy" else ""))
    print(f"  distinct parents   {len(children)} of {len(population)} archive members")
    print(f"  isolation          {iso}"
          + ("  (nothing enforced)" if iso in ("none", "subprocess") else ""))
    print(f"  declared patience  {a.patience} (organisers' epsilon/N unpublished)")
    if policy_refusals:
        print(f"  policy refusals    {policy_refusals}  <- the agent proposed these; "
              f"the harness declined pending a rules answer")
    for k, v in stats.items():
        print(f"  {k:18s} {v}")
    print(f"  TOKENS REPORTED    {stats['tokens_in'] + stats['tokens_out'] + stats['tokens_thinking']:,}"
          f"  (in+out+thinking, the rubric figure)")
    print(f"  population size    {len(population)}")
    print(f"\n  final incumbent    {incumbent.model} "
          f"ndcg={population[incumbent.hash()]['pl']['metrics']['ndcg@10']:.5f} "
          f"recall={population[incumbent.hash()]['pl']['metrics']['recall@50']:.5f}")
    print(f"  best in population {best['cfg'].model} "
          f"ndcg={best['pl']['metrics']['ndcg@10']:.5f} "
          f"recall={best['pl']['metrics']['recall@50']:.5f}")
    rec("run_end", {"run_id": run_id, "termination": converged_reason,
                    "selection": a.selection,
                    "distinct_parents_used": len(children),
                    "seconds": round(el, 1), "stats": stats,
                    "policy_refusals": policy_refusals,
                    "final_incumbent_config_hash": incumbent.hash(),
                    "final_metrics": population[incumbent.hash()]["pl"]["metrics"],
                    "population_size": len(population),
                    "action_summary": action_summary})
    print(f"\n  population rebuilt from the LEDGER: "
          f"{len(rebuild_population(a.split))} configs "
          f"(driver memory: {len(population)}) -- these must agree")
    print("=" * 76)


if __name__ == "__main__":
    main()
