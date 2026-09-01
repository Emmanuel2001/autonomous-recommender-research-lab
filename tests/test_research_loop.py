"""End-to-end test of the code-first loop against a REAL signing scorer.

A deterministic scripted agent drives one seed plus three iterations that between them
exercise every path the run log has to be able to show:

    iteration 1   a syntax error, refused statically, then REPAIRED and promoted
    iteration 2   a runtime contract violation, repaired into a deliberately WORSE
                  candidate, which must be rejected without touching the champion
    iteration 3   a candidate identical to the champion, refused as a duplicate

The property tested hardest is pre-registration: for every candidate that got SCORED,
a proposal record naming its tree must exist at a LOWER ledger sequence number than the
score. That ordering is the whole claim of a pre-registered prediction, and it is the one
thing a fire-and-forget record cannot guarantee.

Speed: the seed is given a 1-epoch configuration and isolation is 'none'. The sandbox is
measured separately (tests/test_isolation.py, and the cross-candidate write probe in
test_candidate.py); what is under test here is the LOOP.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Per-process scratch. run_all.sh already points the ledger and socket at a temp dir, but
# the candidate/submission directories below were hard-coded and rmtree'd on start, so two
# concurrent runs of this suite deleted each other's working directories mid-fit -- which
# shows up as a crash, not as a failed assertion, and looks like a real defect. Same
# hazard run_all.sh's own header documents for test_scorer.
SCRATCH = ROOT / "runs" / "t" / f"research_{os.getpid()}"
os.environ.setdefault("TRACK2_LEDGER", str(SCRATCH / "ledger.jsonl"))
os.environ.setdefault("TRACK2_SOCKET", str(SCRATCH / "scorer.sock"))

import numpy as np
from core import artifacts as AR
from core import budgets as BUD
from core.ledger import verify
from core.scorer import LEDGER_PATH, SOCKET_PATH, request

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

KEY = "b" * 64
LEDGER_PATH.unlink(missing_ok=True); SOCKET_PATH.unlink(missing_ok=True)
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "TRACK2_SIGNING_KEY": KEY}
env.pop("TRACK2_ORCHESTRATOR", None)
proc = subprocess.Popen([sys.executable, "-m", "core.scorer", "serve", "--task", "official"],
                        cwd=ROOT, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True)
try:
    for _ in range(400):
        if SOCKET_PATH.exists(): break
        if proc.poll() is not None:
            print(proc.stdout.read()); sys.exit("scorer died on startup")
        time.sleep(0.25)
    check("a real official scorer is listening", SOCKET_PATH.exists())

    from agent import candidate as CAND
    from agent import research_driver as RD
    from agent.research_agent import plan_prompt as RA_plan_prompt
    from agent.research_state import ResearchState
    from core.official_data import load_for_strategy
    from core.task import get as get_task

    splits, edges = load_for_strategy()
    spec = get_task("kuairand_pure")

    # ---- a cheap seed: one epoch, so the whole suite runs in a couple of minutes
    seed = dict(CAND.seed_tree())
    cfg = json.loads(seed["config.json"]); cfg.update(epochs=1, patience=1)
    seed["config.json"] = json.dumps(cfg, indent=2) + "\n"

    def cfg_edit(**kw):
        c = json.loads(seed["config.json"]); c.update(kw)
        return [{"path": "config.json", "content": json.dumps(c, indent=2) + "\n"}]

    BROKEN_SYNTAX = [{"path": "train.py",
                      "content": seed["train.py"] + "\ndef oops(  # unclosed paren\n"}]
    WRONG_SHAPE = [{"path": "train.py", "content": seed["train.py"].replace(
        'return {"dev": m.predict(Xdev),',
        'return {"dev": m.predict(Xdev)[:10],')}]

    class ScriptedAgent:
        """Deterministic stand-in. Same three methods the real agent exposes."""
        name = "scripted"
        def __init__(self):
            self.calls = {"plan": 0, "code": 0, "reflect": 0}
            self.saw_errors = []
        def _u(self):
            return {"input_tokens": 100, "output_tokens": 50}
        def plan(self, state, parent_source, parent_telemetry=None):
            self.calls["plan"] += 1
            n = self.calls["plan"]
            base = {"stage": "training", "parent": "champion",
                    "evidence": f"scripted step {n}", "risk": "none",
                    "success_criteria": "dev primary improves",
                    "expected_direction": "increase", "files_to_change": ["config.json"]}
            if n == 1:
                base.update(hypothesis="one epoch is undertrained; train for three",
                            mechanism="more gradient steps from an undertrained start",
                            expected_magnitude=0.01)
            elif n == 2:
                base.update(hypothesis="a much smaller learning rate",
                            mechanism="deliberately crippled, to test rejection",
                            expected_magnitude=-0.05, expected_direction="decrease")
            else:
                base.update(hypothesis="resubmit the champion unchanged",
                            mechanism="duplicate, to test refusal",
                            expected_magnitude=0.0)
            return base, self._u()
        def code(self, plan, parent_source, state, error=None, attempt=0):
            self.calls["code"] += 1
            n = self.calls["plan"]
            if error is not None:
                self.saw_errors.append(error)
            if n == 1:
                files = BROKEN_SYNTAX if error is None else cfg_edit(epochs=3, patience=3)
            elif n == 2:
                files = WRONG_SHAPE if error is None else cfg_edit(lr=1e-9, epochs=1)
            else:
                # The SEED tree: differs from the champion (1 epoch vs 3) but has already
                # been evaluated, so this exercises dedup rather than the
                # identical-to-parent check.
                files = [{"path": k, "content": v} for k, v in sorted(seed.items())]
            return {"reasoning": f"scripted {n}/{attempt}", "files": files}, self._u()
        def reflect(self, plan, result, state):
            self.calls["reflect"] += 1
            return {"what_happened": "scripted", "mechanism_supported": True,
                    "why": "scripted", "recommendation": "promote",
                    "next_question": "scripted"}, self._u()

    def rec(kind, body):
        r = request({"op": "record", "kind": kind, "body": body})
        return (True, r["entry"]["seq"]) if r.get("ok") else (False, r.get("error"))

    def score(path, tree):
        r = request({"op": "score_official", "scores": str(path), "split": "valid",
                     "config_hash": tree, "isolation": "none", "kind": "score"})
        return (True, r["entry"]["payload"]) if r.get("ok") else (False, r.get("error"))

    agent = ScriptedAgent()
    budget = BUD.Budget(max_iterations=3, max_wall_seconds=3600)
    state = ResearchState({"name": "kuairand_pure"}, {"max_iterations": 3})
    run_root = SCRATCH / "research_loop"
    import shutil; shutil.rmtree(run_root, ignore_errors=True)

    t0 = time.perf_counter()
    out = RD.research_loop(agent=agent, splits=splits, edges=edges, spec=spec, state=state,
                           budget=budget, run_id="test_research", rec=rec, score=score,
                           run_root=run_root, backend="none", max_seconds=600,
                           convergence="off", max_repairs=1, verbose=True,
                           seed_files=seed, policy="single")
    el = time.perf_counter() - t0
    st = out["stats"]
    print(f"\n  [timing] loop finished in {el:.0f}s; stats {json.dumps(st)}")

    # ---- 1. the loop ran the scripted path -----------------------------------
    check("the seed was scored and became the champion", state.champion is not None)
    check("three iterations were spent", budget.iterations_done == 3,
          str(budget.iterations_done))
    check("the planner was called once per iteration", agent.calls["plan"] == 3)
    check("a static refusal happened (the syntax error)", st["refused"] >= 1, str(st))
    check("a runtime failure happened (the wrong-shape return)", st["failed"] >= 1, str(st))
    check("repairs were attempted", st["repaired"] >= 2, str(st["repaired"]))
    check("the coder was given the error to repair from", len(agent.saw_errors) >= 2)
    check("a static refusal reached the coder as a repair context",
          any(e.get("phase") == "static_validation" for e in agent.saw_errors))
    check("an execution failure reached the coder as a repair context",
          any(e.get("phase") == "execution" for e in agent.saw_errors))
    ex = [e for e in agent.saw_errors if e.get("phase") == "execution"]
    check("an execution error says whose code it is in",
          all(e.get("error_in") in ("candidate", "harness_check") for e in ex),
          str([e.get("error_in") for e in ex]))
    check("a harness-detected contract violation is labelled as such, not as the "
          "agent's line", all(e["error_in"] == "harness_check" for e in ex
                              if e.get("contract_violation")),
          str([(e.get("error_in"), e.get("error_frame")) for e in ex]))
    check("the contract violation message names the actual mismatch",
          any("124,909 rows" in str(e.get("error", "")) for e in ex))
    check("a repair is shown the source IT wrote, not only the parent",
          all(isinstance(e.get("your_previous_attempt"), dict)
              and e["your_previous_attempt"] for e in agent.saw_errors),
          str([sorted(e.get("your_previous_attempt") or {}) for e in agent.saw_errors]))
    check("the syntax-error repair context contains the file the error line refers to",
          any("train.py" in (e.get("your_previous_attempt") or {})
              for e in agent.saw_errors if e.get("phase") == "static_validation"))
    check("a duplicate was refused", st["duplicate"] >= 1, str(st["duplicate"]))
    check("two candidates were scored", st["evaluated"] == 2, str(st["evaluated"]))
    check("nothing was aborted for an unrecorded proposal",
          st["unregistered_aborts"] == 0)

    # ---- 2. promotion is driven by the signed number only --------------------
    scored = [e for e in state.experiments if e.signed]
    seed_exp = state.experiments[0]
    promoted = [e for e in state.experiments if e.outcome == "promoted"]
    rejected = [e for e in state.experiments if e.outcome == "rejected"]
    check("exactly one candidate was promoted after the seed", len(promoted) == 2,
          f"{[e.iteration for e in promoted]}")
    check("the champion is the repaired 3-epoch candidate",
          state.champion["iteration"] == 1, str(state.champion.get("iteration")))
    check("the champion improved on the seed",
          state.champion["primary"] > seed_exp.signed["metrics"]["primary"],
          f"{state.champion['primary']:.6f} vs {seed_exp.signed['metrics']['primary']:.6f}")
    check("the deliberately worse candidate was rejected", len(rejected) == 1)
    check("the rejected candidate really was worse",
          rejected and rejected[0].delta_vs_champion < 0,
          str(rejected[0].delta_vs_champion) if rejected else "")
    check("a rejected candidate did not become the champion",
          rejected and state.champion["tree"] != rejected[0].tree)
    check("the reflector's 'promote' advice did NOT promote a worse candidate",
          rejected and rejected[0].reflection is not None
          and rejected[0].reflection["recommendation"] == "promote"
          and rejected[0].outcome == "rejected")

    # ---- 2b. a refused iteration is remembered, so it can inform the next plan -
    refused_exps = [e for e in state.experiments if e.outcome == "refused"]
    check("an iteration that never produced a fit is still recorded", len(refused_exps) >= 1,
          str([e.outcome for e in state.experiments]))
    check("the refusal reaches the failure registry the planner is shown",
          any(k for k in state.for_prompt()["failure_registry"]),
          str(sorted(state.failures)))
    check("the refusal appears in the recent history the planner is shown",
          any("refused" in line for line in state.for_prompt()["recent_experiments"]))
    check("a synthetic refusal key never enters the dedup map",
          all(state.is_tree(t) for t in state.seen_trees),
          str([t for t in state.seen_trees if not state.is_tree(t)]))
    check("is_tree accepts a real address and rejects a synthetic one",
          state.is_tree("a" * 64) and not state.is_tree("refused:3")
          and not state.is_tree("A" * 64))

    # ---- 2c. the reflector must FEED the next plan, not just be recorded ------
    # Until 2026-08-29 reflections were write-only: stored on the Experiment and saved to
    # state.json, but `for_prompt()` sent none of their fields, so a third of the LLM budget
    # produced analysis the next planner never saw. An external review caught it.
    fp = state.for_prompt()
    check("for_prompt carries a reflections block", "reflections" in fp)
    check("reflections actually reach the planner", len(fp["reflections"]) >= 1,
          str(len(fp.get("reflections", []))))
    _blob = json.dumps(fp, default=str)
    _r = fp["reflections"][-1]
    for _f in ("why", "next_question", "remaining_uncertainty"):
        check(f"the planner sees reflection.{_f}",
              _f in _r and (not _r[_f] or _r[_f][:30] in _blob))
    check("the champion's own reflection is always included",
          any(r["is_current_champion"] for r in fp["reflections"]),
          str([(r["iteration"], r["is_current_champion"]) for r in fp["reflections"]]))
    check("the planner is told to use its own reflections",
          "do not silently ignore your own analysis"
          in RA_plan_prompt(fp, {"train.py": "x = 1\n"}, None)[1])

    # ---- 2d. the planner must see LIVE budget + convergence state ------------
    # state.budget was set once at construction, so the planner saw the static maxima only:
    # it could not know how many evaluations remained or that a convergence check was one
    # iteration away, while its prompt told it the budget was "meant to be spent".
    b = state.budget
    for k in ("evaluations_remaining", "wall_seconds_remaining", "iterations_done"):
        check(f"budget carries live {k}", k in b, str(sorted(b)))
    cv = b.get("convergence") or {}
    for k in ("mode", "epsilon", "N", "window_gain_so_far",
              "iterations_until_next_check",
              "cumulative_gain_still_needed_to_survive_the_check"):
        check(f"convergence state carries {k}", k in cv, str(sorted(cv)))
    # Written at the START of each iteration, so after a 3-iteration run it reports the
    # headroom as of iteration 3 (1 left), not 0. What matters is that it MOVED off the
    # static maximum -- that is what "live" means here.
    check("evaluations_remaining is live, not the static maximum",
          b["evaluations_remaining"] < b["max_iterations"]
          and b["iterations_done"] == budget.iterations_done - 1,
          f"remaining={b['evaluations_remaining']} max={b['max_iterations']} "
          f"done_at_write={b['iterations_done']} done_now={budget.iterations_done}")
    check("the window note states that gain is CUMULATIVE",
          "CUMULATIVE" in str(cv.get("note")) or cv.get("mode") != "official")
    check("recent stages are exposed so the planner can see it is repeating itself",
          isinstance(b.get("stages_attempted_recently"), list))
    _u = RA_plan_prompt(state.for_prompt(), {"train.py": "x = 1\n"}, None)[1]
    check("the live budget reaches the planner's prompt", "evaluations_remaining" in _u)
    check("the prompt no longer tells it unconditionally to spend the budget",
          "Budget is meant to be SPENT" not in _u)

    # ---- 3. the train/dev gap is present and harness-computed ----------------
    check("every scored experiment has a train primary",
          all(e.train_primary is not None for e in scored))
    check("every scored experiment has a train/dev gap",
          all(e.gap is not None for e in scored))
    check("the harness dev primary equals the scorer's signed primary EXACTLY once "
          "rounded the same way",
          all(round(e.dev_primary, 6) == e.signed["metrics"]["primary"] for e in scored),
          str([(round(e.dev_primary, 6), e.signed['metrics']['primary'])
               for e in scored]))
    check("no metric disagreement was flagged", st["metric_disagreements"] == 0)
    trained = [e for e in scored if e.outcome in ("promoted",)]
    check("a TRAINED model fits its training rows better than dev (positive gap)",
          all(e.gap > 0 for e in trained), str([e.gap for e in trained]))
    check("the crippled candidate's gap is not assumed positive (it barely trained)",
          any(e.gap <= 0 for e in scored if e.outcome == "rejected")
          or all(e.gap > 0 for e in scored))

    # ---- 4. PRE-REGISTRATION: proposal before score, in the signed chain -----
    entries = [json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip()]
    props, scores_ = {}, {}
    for e in entries:
        pl = e.get("payload", {})
        if pl.get("kind") == "proposal":
            b = pl.get("body", {})
            props.setdefault(b.get("tree"), []).append(e["seq"])
        if "config_hash" in pl and "metrics" in pl:
            scores_.setdefault(pl["config_hash"], []).append(e["seq"])
    check("every scored tree has a proposal record", 
          all(t in props for t in scores_), str([t[:8] for t in scores_ if t not in props]))
    check("every proposal precedes the score it predicts",
          all(min(props[t]) < min(scores_[t]) for t in scores_ if t in props),
          str({t[:8]: (min(props.get(t, [10**9])), min(v)) for t, v in scores_.items()}))
    check("the failed candidates were registered too (not only the successful ones)",
          len(props) >= len(scores_) + 1, f"{len(props)} proposals, {len(scores_)} scores")
    v = verify(LEDGER_PATH, KEY.encode())
    check("the ledger hash chain verifies", v.ok, str(getattr(v, "reason", "")))
    check("every entry was verified", v.n_verified >= len(entries) - 1,
          f"{v.n_verified} of {len(entries)}")

    # ---- 5. the diff is recorded, and computed by the harness ----------------
    dbodies = [json.loads(l)["payload"]["body"] for l in LEDGER_PATH.read_text().splitlines()
               if l.strip() and json.loads(l)["payload"].get("kind") == "proposal"]
    withdiff = [b for b in dbodies if b.get("diff")]
    check("proposals carry a unified diff", len(withdiff) >= 2, str(len(withdiff)))
    check("a diff shows the config change as a hunk",
          any("@@" in b["diff"] and "config.json" in b["diff"] for b in withdiff))
    check("every proposal records a diff hash and byte count",
          all(b.get("diff_hash") and isinstance(b.get("diff_bytes"), int)
              for b in dbodies))
    check("every proposal records the contract hash",
          all(len(str(b.get("contract_sha256", ""))) == 64 for b in dbodies))
    check("the recorded diff hash matches the recorded diff text",
          all(AR.diff_hash(b["diff"]) == b["diff_hash"] for b in withdiff))
    check("no ledger record exceeded the size cap (nothing silently unrecorded)",
          st["unregistered_aborts"] == 0 and len(props) >= 4)

    # ---- 6. cost accounting: wall-clock per iteration, stages, tokens --------
    bs = out["budget"]
    check("per-iteration wall-clock was recorded for every iteration",
          len(bs["iteration_wall_seconds"]) == 3)
    check("every iteration has a positive wall-clock",
          all(x > 0 for x in bs["iteration_wall_seconds"]))
    check("stage timings were recorded", set(bs["stage_seconds"]) >= {"plan", "code"},
          str(sorted(bs["stage_seconds"])))
    check("fit time was recorded as its own stage", "fit" in bs["stage_seconds"])
    check("tokens were accumulated from the agent", bs["tokens_total"] > 0)
    # Against the RAW cost, not the summary: summary() rounds to 2dp and a scripted
    # agent's code stage is sub-millisecond, so a rounded assertion here would be
    # measuring the rounding rather than the accumulation.
    rep_iters = [c for c in budget.costs if c.repairs >= 1]
    check("the repairing iteration recorded its repairs", len(rep_iters) >= 2,
          str([c.repairs for c in budget.costs]))
    check("a repairing iteration timed its code stage more than once",
          all("code" in c.stages and c.stages["code"] > 0 for c in rep_iters),
          str([(c.iteration, sorted(c.stages)) for c in rep_iters]))
    check("iteration wall-clock is at least the sum of its own stages",
          all(c.wall_seconds >= sum(c.stages.values()) - 1e-6 for c in budget.costs))

    # ---- 7. artifacts: every recorded tree exists and verifies ---------------
    bad = []
    for t in props:
        try:
            AR.verify(t)
        except AR.ArtifactError as e:
            bad.append((t[:8], str(e)[:60]))
    check("every tree named in the ledger exists in the store and verifies",
          not bad, str(bad))
    check("the champion's tree can be read back", 
          set(AR.read(state.champion["tree"])) == set(CAND.seed_tree()))

    # ---- 8. state survives a save/load round trip ----------------------------
    sp = run_root / "state.json"
    state.save(sp)
    from agent.research_state import ResearchState as RS2
    back = RS2.load(sp)
    check("state round-trips through JSON",
          back.champion["tree"] == state.champion["tree"]
          and len(back.experiments) == len(state.experiments))
    check("seen trees survive the round trip", back.seen_trees == state.seen_trees)

    # -- the per-iteration cost RECORDED IN THE LEDGER must carry a real wall time.
    # This is the assertion that was missing: the old one read budget.costs after the loop,
    # where __exit__ had filled the field, so it passed while every ledger entry said
    # wall_seconds: 0.0.
    _ents = [json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip()]
    _costs = [(e["payload"]["body"].get("iteration"), e["payload"]["body"]["cost"])
              for e in _ents
              if (e["payload"].get("body") or {}).get("what") == "iteration_result"
              and (e["payload"].get("body") or {}).get("cost")]
    check("every ledger iteration record has a non-zero wall time",
          all(c["wall_seconds"] > 0 for _i, c in _costs),
          str([(i, c["wall_seconds"]) for i, c in _costs]))
    check("every ledger iteration wall_seconds >= its own stage sum (within rounding)",
          all(c["wall_seconds"] + 0.0005 * (len(c["stages"]) + 1)
              >= sum(c["stages"].values()) for _i, c in _costs),
          str([(i, c["wall_seconds"], round(sum(c["stages"].values()), 3))
               for i, c in _costs]))
    check("the ledger records a tokens_total per iteration",
          all("tokens_total" in c for _i, c in _costs))

    # ---- 9. the submission is BOUND to the promoted artifact ------------------
    import os as _os
    env2 = {**_os.environ, "TRACK2_LEDGER": str(LEDGER_PATH)}
    def submit(*extra):
        return subprocess.run(
            [sys.executable, "make_research_submission.py", "--state", str(sp),
             "--ledger", str(LEDGER_PATH), "--outdir",
             str(SCRATCH / "sub")] + list(extra),
            cwd=ROOT, env=env2, capture_output=True, text=True)

    # The test champion is deliberately undertrained, so it is BELOW the official
    # baseline -- which is exactly the case the refusal exists for.
    r0 = submit()
    check("a champion below the official baseline is refused by default",
          r0.returncode != 0 and "does not beat the official baseline" in
          (r0.stdout + r0.stderr), (r0.stdout + r0.stderr)[-160:])

    r1 = submit("--allow-below-baseline")
    check("with the override it builds", r1.returncode == 0,
          (r1.stdout + r1.stderr)[-200:])
    sd = SCRATCH / "sub"
    check("the artifact was verified, not trusted",
          "artifact VERIFIED" in r1.stdout)
    check("the dev scores were bound to the signed hash",
          "matches the signed value" in r1.stdout)
    check("a submission CSV was written", (sd / "submission_test.csv").exists())
    check("the submission has one row per held-out row",
          sum(1 for _ in (sd / "submission_test.csv").open()) - 1 == len(splits["test"]))
    man = json.loads((sd / "MANIFEST.json").read_text())
    check("the manifest names the champion tree",
          man["champion"]["tree"] == state.champion["tree"])
    check("the manifest carries the signed sequence number",
          isinstance(man["signed_validation"]["seq"], int))
    check("the manifest records the contract hash",
          len(str(man["champion"]["contract_sha256"])) == 64)
    check("the manifest states the submission is not a refit",
          "NOT a refit" in man["binding"]["how"])
    check("the manifest carries the full experiment lineage",
          len(man["lineage"]) == len(state.experiments))
    check("the manifest records the harness-measured train/dev gap",
          man["harness_measured"]["train_dev_gap"] is not None)
    check("the ORGANISERS' own format checker passed on this submission",
          man["kit_check_passed"] is True, str(man["kit_check_passed"]))

    # The binding property: corrupt the dev scores and the submission must REFUSE,
    # because the directory can no longer be shown to be the run that was signed.
    champ_dir = None
    for d in sorted(run_root.iterdir()):
        if d.is_dir() and state.champion["tree"][:12] in d.name:
            champ_dir = d; break
    check("the champion's run directory was located", champ_dir is not None)
    if champ_dir:
        orig = (champ_dir / "dev_scores.npy").read_bytes()
        arr = np.load(champ_dir / "dev_scores.npy"); arr[0] += 1.0
        np.save(champ_dir / "dev_scores.npy", arr)
        r2 = submit("--allow-below-baseline")
        check("tampered dev scores make the submission REFUSE",
              r2.returncode != 0 and "not the run that produced the signed number"
              in (r2.stdout + r2.stderr), (r2.stdout + r2.stderr)[-200:])
        (champ_dir / "dev_scores.npy").write_bytes(orig)
        check("restoring the scores makes it build again",
              submit("--allow-below-baseline").returncode == 0)

    # And a tampered ARTIFACT must be caught before anything else happens.
    art_src = AR.ARTIFACTS / state.champion["tree"] / "source" / "config.json"
    if art_src.exists():
        keep = art_src.read_text()
        art_src.write_text(keep.replace("\n}", ',\n  "sneaked": 1\n}'))
        r3 = submit("--allow-below-baseline")
        check("a tampered champion artifact makes the submission REFUSE",
              r3.returncode != 0 and "does not match its contents" in
              (r3.stdout + r3.stderr), (r3.stdout + r3.stderr)[-200:])
        art_src.write_text(keep)
    # ---- 10. nothing is ever executed unrecorded -----------------------------
    # A scorer that refuses to record a proposal must stop the candidate BEFORE it runs.
    # The typed driver's rec() is fire-and-forget, which combined with a record cap below
    # the size of a legal candidate could produce a scored experiment with no
    # pre-registered hypothesis -- the exact thing pre-registration exists to prevent.
    refused_kinds = []
    def rec_refusing(kind, body):
        if kind == "proposal" and body.get("iteration", 0) >= 1:
            refused_kinds.append(kind)
            return False, "simulated: record exceeds MAX_RECORD_BYTES"
        return rec(kind, body)

    budget2 = BUD.Budget(max_iterations=1, max_wall_seconds=1800)
    state2 = ResearchState({"name": "kuairand_pure"}, {"max_iterations": 1})
    root2 = SCRATCH / "research_abort"
    shutil.rmtree(root2, ignore_errors=True)
    out2 = RD.research_loop(agent=ScriptedAgent(), splits=splits, edges=edges, spec=spec,
                            state=state2, budget=budget2, run_id="test_abort",
                            rec=rec_refusing, score=score, run_root=root2,
                            # max_repairs=1: the scripted agent's first tree is the
                            # deliberate syntax error, which is refused statically before
                            # pre-registration is ever reached. The repair produces a
                            # valid tree, which is what must then be abandoned.
                            backend="none", max_seconds=600, convergence="off",
                            max_repairs=1, verbose=False, seed_files=seed,
                            policy="single")
    st2 = out2["stats"]
    check("a refused pre-registration was encountered", len(refused_kinds) >= 1)
    check("the candidate was ABANDONED, not run", st2["unregistered_aborts"] >= 1,
          str(st2))
    check("no candidate beyond the seed was scored", st2["evaluated"] == 0, str(st2))
    check("the champion is still the seed", state2.champion["iteration"] == 0)
    ran = [d.name for d in root2.iterdir() if d.is_dir() and d.name.startswith("iter001")]
    check("no working directory was created for the unregistered candidate", not ran,
          str(ran))
    check("the abandoned iteration still cost wall-clock (not hidden)",
          len(out2["budget"]["iteration_wall_seconds"]) == 1)

finally:
    proc.terminate()
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: proc.kill()
    shutil.rmtree(SCRATCH, ignore_errors=True)

print()
bad_l = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad_l)}/{len(R)} passed" + (f"  FAILURES: {bad_l}" if bad_l else ""))
sys.exit(1 if bad_l else 0)
