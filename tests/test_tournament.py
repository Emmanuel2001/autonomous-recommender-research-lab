"""Tests for the hypothesis tournament, structured beliefs and role routing.

WHAT THIS SUITE IS FOR
----------------------
The old policy asked one LLM call for one idea, and across 126 recorded proposals 115 edited
`features.py` while `train.py` was never touched. That was a property of the POLICY, not of
the model's judgement. The tournament makes alternatives from every research lane exist
before one is chosen, and a judge must say in writing why it passed over the architecture
candidate.

So the assertions here are about the MECHANISM, not about any lane winning:
  * every lane is represented, and architecture/sampling candidates are always present;
  * exactly one candidate consumes an evaluation -- the others must never be presentable as
    completed experiments;
  * the tournament and the judge's reasoning are in the signed chain BEFORE any candidate
    code runs, so neither can be reconstructed once the result is known;
  * an architecture candidate that LOSES is recorded and reflected on, not hidden.

A fake backend drives it. No API credits are spent.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRATCH = ROOT / "runs" / "t" / f"tourn_{os.getpid()}"
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

KEY = "c" * 64
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
    from agent.research_agent import LANES, REQUIRED_LANES, ROLES, ResearchAgent
    from agent.research_state import ResearchState
    from core.official_data import load_for_strategy
    from core.task import get as get_task

    splits, edges = load_for_strategy()
    spec = get_task("kuairand_pure")
    WS = CAND.workspace_for("inherited")

    seed = dict(CAND.seed_tree(WS))
    cfg = json.loads(seed["config.json"]); cfg.update(epochs=1, patience=1)
    seed["config.json"] = json.dumps(cfg, indent=2) + "\n"

    def cfg_edit(**kw):
        c = json.loads(seed["config.json"]); c.update(kw)
        return [{"path": "config.json", "content": json.dumps(c, indent=2) + "\n"}]

    # A SYNTHETIC fake-agent architecture change, written by hand for this test: scale the
    # interaction term. No LLM produced it. It is genuinely a model.py edit rather than a
    # config flip, which is what makes it a valid exercise of the model-replacement path --
    # but nothing here demonstrates that a real agent would propose such a thing.
    ARCH_MODEL = seed["model.py"].replace(
        "        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))",
        "        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))\n"
        "        inter = inter * 0.85       # synthetic fake-agent interaction damping")
    # A SYNTHETIC fake-agent training change, likewise hand-written for this test.
    TRAIN_EDIT = seed["train.py"].replace(
        "        dev_m = primary(udev, ydev, m.predict(Xdev))",
        "        dev_m = primary(udev, ydev, m.predict(Xdev))\n"
        "        # synthetic fake-agent edit: clipped loss alongside the raw one\n"
        "        _ = float(np.clip(np.mean(losses) if losses else 0.0, 0.0, 10.0))")

    def tourn_candidates(n):
        """One candidate per lane, so coverage is satisfied deterministically."""
        base = [
            ("F1", "feature_preprocessing", "add a duration x tab cross", ["features.py"]),
            ("O1", "objective_loss", "switch to a pairwise ranking objective",
             ["config.json"]),
            ("S1", "sampling_training", "sample harder negatives within a user",
             ["train.py"]),
            ("R1", "optimization_regularization", "raise l2 to curb the train/dev gap",
             ["config.json"]),
            ("M1", "model_architecture", "damp the second-order interaction term",
             ["model.py"]),
            ("E1", "ensemble_calibration", "rank-average two seeds", ["config.json"]),
        ]
        out = []
        for cid, lane, hyp, files in base[:n]:
            out.append({
                "id": cid, "lane": lane, "hypothesis": hyp,
                "mechanism": f"mechanism for {cid}", "evidence": "scripted",
                "expected_direction": "increase", "expected_magnitude": 0.001,
                "information_value": f"settles whether {lane} helps here",
                "novelty_vs_history": "not previously tried",
                "files_to_change": files, "runtime_estimate": "~30s",
                "memory_estimate": "<1GB", "implementation_risk": "low",
                "leakage_risk": "none", "success_criteria": "dev primary improves",
                "failure_interpretation": f"{lane} is not the lever here",
                "parent": "champion"})
        return out

    class FakeAgent:
        """Deterministic stand-in exposing the real agent's surface, including role usage."""
        name = "fake"
        def __init__(self, pick_order):
            self.pick_order = list(pick_order)
            self.calls = {r: 0 for r in ROLES}
            self.usage = []
            self.tournaments_seen = 0
            self.judge_states = []
            self.code_plans = []
            self.role_model = {r: f"model-{r}" for r in ROLES}
        def _u(self, role, model=None):
            u = {"role": role, "backend": "fake", "model": model or self.role_model[role],
                 "input_tokens": 100, "output_tokens": 50, "thinking_tokens": 10,
                 "total_tokens": 160, "seconds": 0.01, "retries": 0,
                 "refused": False, "truncated": False}
            self.usage.append(u); self.calls[role] += 1
            return u
        def routing(self): return dict(self.role_model)
        def usage_by_role(self):
            return ResearchAgent.usage_by_role(self)
        def tournament(self, state, parent_source, parent_telemetry=None, size=6,
                       deficiency=None):
            self.tournaments_seen += 1
            return {"candidates": tourn_candidates(size),
                    "lane_coverage_note": "one per lane"}, self._u("plan")
        def judge(self, candidates, state, parent_source, parent_telemetry=None,
                  problems=None):
            self.judge_states.append(state)
            want = self.pick_order[min(self.tournaments_seen - 1,
                                       len(self.pick_order) - 1)]
            return {"selected_id": want,
                    "ranking": [{"id": c["id"], "rank": i + 1, "score_evidence": 1.0,
                                 "score_expected_impact": 1.0, "score_novelty": 1.0,
                                 "score_information_value": 1.0,
                                 "score_cpu_feasibility": 1.0,
                                 "score_implementation_risk": 1.0}
                                for i, c in enumerate(candidates)],
                    "why_selected": f"scripted pick {want}",
                    "architecture_verdicts": [
                        {"candidate_id": c["id"],
                         "verdict": "selected" if c["id"] == want else "rejected",
                         "reason": "scripted reason, long enough to pass"}
                        for c in candidates if c["lane"] == "model_architecture"],
                    "exploit_or_explore": "explore"}, self._u("judge")
        def code(self, plan, parent_source, state, error=None, attempt=0):
            self.code_plans.append(plan)
            cid = plan.get("_candidate_id")
            if cid == "M1":
                files = [{"path": "model.py", "content": ARCH_MODEL}]
            elif cid == "S1":
                files = [{"path": "train.py", "content": TRAIN_EDIT}]
            elif cid == "O1":
                files = cfg_edit(loss="pairwise", epochs=2, patience=2)
            else:
                files = cfg_edit(l2=5e-6, epochs=2, patience=2)
            return {"reasoning": f"scripted for {cid}", "files": files}, self._u("code")
        def reflect(self, plan, result, state):
            lane = plan.get("lane", "unspecified")
            return {"what_happened": "scripted", "mechanism_supported": False,
                    "why": "scripted why", "recommendation": "discard",
                    "next_question": f"is {lane} exhausted?",
                    "remaining_uncertainty": "scripted",
                    "belief_updates": [{
                        "mechanism": f"{lane} mechanism", "lane": lane,
                        "belief": "no gain observed", "confidence": 0.6,
                        "supports": False, "affected_files": plan.get("files_to_change"),
                        "effect_low": -0.002, "effect_high": 0.001,
                        "unresolved": "untested at higher capacity",
                        "retry_only_if": "with regularization"}]}, self._u("reflect")

    def rec(kind, body):
        r = request({"op": "record", "kind": kind, "body": body})
        return (True, r["entry"]["seq"]) if r.get("ok") else (False, r.get("error"))

    def score(path, tree):
        r = request({"op": "score_official", "scores": str(path), "split": "valid",
                     "config_hash": tree, "isolation": "none", "kind": "score"})
        return (True, r["entry"]["payload"]) if r.get("ok") else (False, r.get("error"))

    # M1 (architecture) first, then S1 (training), then R1 -- so both required lanes are
    # actually EXECUTED at least once in this suite, not merely considered.
    agent = FakeAgent(["M1", "S1", "R1"])
    budget = BUD.Budget(max_iterations=3, max_wall_seconds=3600)
    state = ResearchState({"name": "kuairand_pure"}, {"max_iterations": 3})
    run_root = SCRATCH / "loop"
    shutil.rmtree(run_root, ignore_errors=True)

    t0 = time.perf_counter()
    out = RD.research_loop(agent=agent, splits=splits, edges=edges, spec=spec, state=state,
                           budget=budget, run_id="test_tourn", rec=rec, score=score,
                           run_root=run_root, backend="none", max_seconds=900,
                           convergence="off", max_repairs=1, verbose=True,
                           seed_files=seed, policy="tournament", tournament_size=6,
                           workspace=WS)
    st = out["stats"]
    print(f"\n  [timing] {time.perf_counter()-t0:.0f}s  stats {json.dumps(st)}\n")

    entries = [json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip()]
    tourns = [e for e in entries
              if (e["payload"].get("body") or {}).get("what") == "tournament"]
    props = [e for e in entries if e["payload"].get("kind") == "proposal"]
    scores_ = [e for e in entries if "metrics" in e["payload"]]

    # 1 -- the tournament happened, spans every lane, and includes the required ones ------
    check("a tournament ran for every iteration", len(tourns) == 3, str(len(tourns)))
    cov = tourns[0]["payload"]["body"]["coverage"]
    check("every research lane is represented",
          sorted(cov["lanes_present"]) == sorted(LANES), str(cov["lanes_present"]))
    check("no required lane is missing", cov["missing_required_lanes"] == [],
          str(cov["missing_required_lanes"]))
    check("coverage is reported as OK", cov["lane_coverage_ok"] is True)
    for lane in REQUIRED_LANES:
        check(f"a candidate was considered in {lane}", cov["per_lane"][lane] >= 1)
    cands = tourns[0]["payload"]["body"]["candidates"]
    check("bounded signed candidates carry every executable hypothesis field",
          all(all(k in c for k in ("id", "lane", "hypothesis", "mechanism",
                                  "expected_direction", "expected_magnitude",
                                  "information_value", "files_to_change",
                                  "implementation_risk", "model_family",
                                  "architecture_change"))
              for c in cands), str(sorted(cands[0])))
    check("stats count architecture candidates considered",
          st["architecture_considered"] == 3, str(st["architecture_considered"]))
    check("stats count total candidates considered",
          st["candidates_considered"] == 18, str(st["candidates_considered"]))

    # 2 -- only the WINNER consumes an evaluation ----------------------------------------
    check("one proposal per iteration, not one per candidate",
          len(props) == 4, f"{len(props)} (1 seed + 3 iterations)")
    check("scored trees equal iterations plus the seed",
          len(scores_) == 4, str(len(scores_)))
    check("18 candidates were considered but only 3 were evaluated",
          st["candidates_considered"] == 18 and st["evaluated"] == 3,
          f"considered {st['candidates_considered']} evaluated {st['evaluated']}")

    # 3 -- the tournament is recorded BEFORE the candidate exists or runs ----------------
    ok_order = True
    for t in tourns:
        it_no = t["payload"]["body"]["iteration"]
        pr = [e for e in props if e["payload"]["body"].get("iteration") == it_no]
        if not pr or min(x["seq"] for x in pr) <= t["seq"]:
            ok_order = False
    check("every tournament precedes its own proposal in the chain", ok_order)
    check("every proposal precedes the score it predicts",
          all(min(p["seq"] for p in props
                  if p["payload"]["body"]["tree"] == s_["payload"]["config_hash"])
              < s_["seq"] for s_ in scores_
              if any(p["payload"]["body"]["tree"] == s_["payload"]["config_hash"]
                     for p in props)))
    jr = tourns[0]["payload"]["body"]["judge"]
    check("the judge's rationale is in the chain", bool(jr.get("why_selected")))
    avs = jr.get("architecture_verdicts") or []
    check("the judge's structured architecture verdicts are in the chain", bool(avs))
    check("each verdict names a candidate id, a verdict and a reason",
          all({"candidate_id", "verdict", "reason"} <= set(v) for v in avs), str(avs[:1]))
    check("there is one verdict per architecture candidate",
          len(avs) == sum(1 for c in cands if c["lane"] == "model_architecture"),
          f"{len(avs)} verdicts")
    check("the judge's full ranking is in the chain", len(jr.get("ranking") or []) == 6)
    check("exploit-or-explore is recorded", jr.get("exploit_or_explore") in
          ("exploit", "explore"))

    # 4 -- an ARCHITECTURE experiment actually ran, and a TRAINING one too ---------------
    exps = state.experiments
    lanes_run = {e.lane for e in exps if e.lane}
    check("the architecture candidate was executed, not just considered",
          "model_architecture" in lanes_run, str(sorted(lanes_run)))
    check("the sampling/training candidate was executed",
          "sampling_training" in lanes_run, str(sorted(lanes_run)))
    arch = [e for e in exps if e.lane == "model_architecture"]
    check("a synthetic fake-agent model.py replacement executed and was scored",
          arch and arch[0].signed is not None,
          str(arch[0].outcome if arch else "no arch experiment"))
    check("the architecture experiment edited model.py",
          arch and "model.py" in (arch[0].files_changed or []),
          str(arch[0].files_changed if arch else []))
    tr = [e for e in exps if e.lane == "sampling_training"]
    check("a synthetic fake-agent train.py replacement executed and was scored",
          tr and tr[0].signed is not None)
    check("the training experiment edited train.py",
          tr and "train.py" in (tr[0].files_changed or []),
          str(tr[0].files_changed if tr else []))
    check("a losing architecture experiment is recorded, not hidden",
          arch and arch[0].outcome in ("rejected", "promoted", "failed"),
          str(arch[0].outcome if arch else ""))
    check("it was reflected on", arch and arch[0].reflection is not None)

    # 5 -- beliefs are updated from signed results and reach the next tournament --------
    check("beliefs were updated", st["beliefs_updated"] >= 2, str(st["beliefs_updated"]))
    check("beliefs are keyed by mechanism", all("mechanism" in b for b in
                                                state.beliefs.values()))
    b0 = next(iter(state.beliefs.values()))
    for f in ("lane", "confidence", "supporting_iterations", "contradicting_iterations",
              "affected_files", "effect_range", "unresolved", "retry_only_if",
              "last_updated_iteration"):
        check(f"a belief carries {f}", f in b0, str(sorted(b0)))
    check("a non-supporting result lands in contradicting_iterations",
          any(b["contradicting_iterations"] for b in state.beliefs.values()))
    fp = state.for_prompt()
    check("beliefs reach the next tournament's state", bool(fp.get("beliefs")))
    check("lane stats reach the next tournament's state", "lane_stats" in fp)
    later = agent.judge_states[-1]
    check("the judge saw beliefs from earlier iterations", bool(later.get("beliefs")))
    check("the judge saw earlier reflections", bool(later.get("reflections")))

    # 6 -- lane statistics -----------------------------------------------------------
    ls = state.lane_stats()
    check("lane stats count model.py edits", ls["model_py_edits"] >= 1,
          str(ls["model_py_edits"]))
    check("lane stats count train.py edits", ls["train_py_edits"] >= 1,
          str(ls["train_py_edits"]))
    check("lane stats separate considered from selected",
          all("considered" in v and "selected" in v for v in ls["by_lane"].values()))
    check("per-lane mean signed delta is computed",
          any(v["mean_signed_delta"] is not None for v in ls["by_lane"].values()))

    # 7 -- role-based token accounting ------------------------------------------------
    ur = agent.usage_by_role()
    check("usage is aggregated by role", set(ur) >= {"plan", "judge", "code", "reflect"},
          str(sorted(ur)))
    check("each role reports its own model", all(u["models"] for u in ur.values()))
    check("thinking tokens are counted", ur["plan"]["thinking_tokens"] > 0)
    check("totals sum input+output+thinking",
          ur["judge"]["total_tokens"] == 160 * ur["judge"]["calls"],
          str(ur["judge"]))
    check("the judge was called once per tournament", ur["judge"]["calls"] == 3)

    # 8 -- the SINGLE policy still works, and does not run a tournament ---------------
    agent2 = FakeAgent(["M1"])
    class SingleAgent(FakeAgent):
        def plan(self, state, parent_source, parent_telemetry=None):
            return {"hypothesis": "single-policy hypothesis", "mechanism": "m",
                    "evidence": "e", "stage": "config", "expected_direction": "increase",
                    "expected_magnitude": 0.001, "parent": "champion",
                    "files_to_change": ["config.json"], "risk": "", 
                    "success_criteria": ""}, self._u("plan")
        def code(self, plan, parent_source, state, error=None, attempt=0):
            return {"reasoning": "single", "files": cfg_edit(l2=2e-6, epochs=1)}, \
                   self._u("code")
    a3 = SingleAgent(["x"])
    b3 = BUD.Budget(max_iterations=1, max_wall_seconds=1800)
    s3 = ResearchState({"name": "kuairand_pure"}, {})
    root3 = SCRATCH / "single"; shutil.rmtree(root3, ignore_errors=True)
    n_t_before = len([e for e in
                      [json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip()]
                      if (e["payload"].get("body") or {}).get("what") == "tournament"])
    o3 = RD.research_loop(agent=a3, splits=splits, edges=edges, spec=spec, state=s3,
                          budget=b3, run_id="test_single", rec=rec, score=score,
                          run_root=root3, backend="none", max_seconds=900,
                          convergence="off", max_repairs=0, verbose=False,
                          seed_files=seed, policy="single", workspace=WS)
    n_t_after = len([e for e in
                     [json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip()]
                     if (e["payload"].get("body") or {}).get("what") == "tournament"])
    check("the single policy runs no tournament", n_t_after == n_t_before)
    check("the single policy still evaluates its hypothesis",
          o3["stats"]["evaluated"] == 1, str(o3["stats"]))
    check("the single policy reports zero candidates considered",
          o3["stats"]["candidates_considered"] == 0)
    check("a single-policy experiment has no lane", all(not e.lane for e in s3.experiments))

    # 10 -- ADVERSARIAL: a malformed tournament or judge must never reach coding ------
    # Each case is driven through the real loop. The assertion is not "an error was logged"
    # but "no candidate was written, fitted or scored" -- an unbalanced set that still spends
    # an evaluation would let the report claim breadth the run did not have.
    LANE_LIST = list(LANES)

    def bad_set(kind):
        if kind == "no_arch":
            return [c for c in tourn_candidates(6) if c["lane"] != "model_architecture"]
        if kind == "no_sampling":
            return [c for c in tourn_candidates(6) if c["lane"] != "sampling_training"]
        if kind == "one_lane_x6":
            return [{**c, "lane": "feature_preprocessing"} for c in tourn_candidates(6)]
        if kind == "too_few":
            return tourn_candidates(3)
        return tourn_candidates(6)

    class AdversarialAgent(FakeAgent):
        """Emits a specific malformation. `repair_on_retry` mends it on the second call."""
        def __init__(self, kind, repair_on_retry=False):
            super().__init__(["M1"])
            self.kind, self.repair_on_retry = kind, repair_on_retry
            self.tournament_calls = 0
            self.judge_calls = 0
            self.code_calls = 0
        def tournament(self, state, parent_source, parent_telemetry=None, size=6,
                       deficiency=None):
            self.tournament_calls += 1
            self.tournaments_seen += 1
            broken = self.kind in ("no_arch", "no_sampling", "one_lane_x6", "too_few")
            mend = self.repair_on_retry and self.tournament_calls > 1
            cands = tourn_candidates(6) if (mend or not broken) else bad_set(self.kind)
            if deficiency is not None:
                self.saw_deficiency = deficiency
            return {"candidates": cands}, self._u("plan")
        def judge(self, candidates, state, parent_source, parent_telemetry=None,
                  problems=None):
            self.judge_calls += 1
            if problems is not None:
                self.saw_problems = problems
            mend = self.repair_on_retry and self.judge_calls > 1
            full = [{"id": c["id"], "rank": i + 1, "score_evidence": 1.0,
                     "score_expected_impact": 1.0, "score_novelty": 1.0,
                     "score_information_value": 1.0, "score_cpu_feasibility": 1.0,
                     "score_implementation_risk": 1.0}
                    for i, c in enumerate(candidates)]
            arch = [c["id"] for c in candidates if c["lane"] == "model_architecture"]
            verdicts = [{"candidate_id": i, "verdict": "selected" if i == "M1" else
                         "rejected", "reason": "scripted reason, long enough"}
                        for i in arch]
            sel = "M1" if any(c["id"] == "M1" for c in candidates) else candidates[0]["id"]
            out = {"selected_id": sel, "ranking": full, "why_selected": "scripted pick",
                   "architecture_verdicts": verdicts, "exploit_or_explore": "explore"}
            if not mend and self.kind == "bad_ranking":
                out["ranking"] = full[:2]                 # omits candidates
            if not mend and self.kind == "no_arch_verdict":
                out["architecture_verdicts"] = []          # unaddressed architecture
            return out, self._u("judge")
        def code(self, plan, parent_source, state, error=None, attempt=0):
            self.code_calls += 1
            return super().code(plan, parent_source, state, error, attempt)

    def run_adversarial(kind, repair, tag):
        ag = AdversarialAgent(kind, repair_on_retry=repair)
        bb = BUD.Budget(max_iterations=1, max_wall_seconds=1800)
        stt = ResearchState({"name": "kuairand_pure"}, {})
        rt = SCRATCH / f"adv_{tag}"
        shutil.rmtree(rt, ignore_errors=True)
        before = len([l for l in LEDGER_PATH.read_text().splitlines() if l.strip()])
        o = RD.research_loop(agent=ag, splits=splits, edges=edges, spec=spec, state=stt,
                             budget=bb, run_id=f"adv_{tag}", rec=rec, score=score,
                             run_root=rt, backend="none", max_seconds=900,
                             convergence="off", max_repairs=0, verbose=False,
                             seed_files=seed, policy="tournament", tournament_size=6,
                             workspace=WS, max_coverage_repairs=1 if repair else 0)
        ents = [json.loads(l) for l in LEDGER_PATH.read_text().splitlines()
                if l.strip()][before:]
        return ag, o, stt, ents, rt

    print("\n  -- adversarial cases, repair DISABLED (must abort before coding) --")
    for kind in ("no_arch", "no_sampling", "one_lane_x6", "too_few",
                 "bad_ranking", "no_arch_verdict"):
        ag, o, stt, ents, rt = run_adversarial(kind, False, kind)
        st2 = o["stats"]
        aborted = st2["coverage_aborts"] + st2["judge_aborts"] >= 1
        abandon = [e for e in ents
                   if (e["payload"].get("body") or {}).get("what") == "iteration_abandoned"]
        props_it1 = [e for e in ents if e["payload"].get("kind") == "proposal"
                     and e["payload"]["body"].get("iteration") == 1]
        check(f"[{kind}] the iteration was abandoned", aborted, str(st2))
        check(f"[{kind}] NO candidate was coded", ag.code_calls == 0,
              f"code() called {ag.code_calls}x")
        check(f"[{kind}] NO candidate was registered for the iteration",
              not props_it1, str(len(props_it1)))
        check(f"[{kind}] nothing was evaluated", st2["evaluated"] == 0, str(st2["evaluated"]))
        check(f"[{kind}] the abandonment is recorded with coded/fitted/scored false",
              abandon and abandon[0]["payload"]["body"]["coded"] is False
              and abandon[0]["payload"]["body"]["fitted"] is False
              and abandon[0]["payload"]["body"]["scored"] is False)
        check(f"[{kind}] the reason is recorded",
              abandon and len(str(abandon[0]["payload"]["body"].get("reason"))) > 10)
        # the specific defect must be named, not just "invalid"
        detail = json.dumps(abandon[0]["payload"]["body"]) if abandon else ""
        if kind in ("no_arch", "no_sampling", "one_lane_x6", "too_few"):
            check(f"[{kind}] the coverage problem is named", "problems" in detail
                  or "missing" in detail, detail[:80])
        else:
            check(f"[{kind}] the judge problem is named", "problems" in detail, detail[:80])

    print("\n  -- the same cases WITH one repair call (must proceed after mending) --")
    for kind in ("no_arch", "one_lane_x6", "bad_ranking", "no_arch_verdict"):
        ag, o, stt, ents, rt = run_adversarial(kind, True, f"{kind}_rep")
        st2 = o["stats"]
        check(f"[{kind}+repair] a repair call was made",
              (ag.tournament_calls > 1) or (ag.judge_calls > 1),
              f"tournament={ag.tournament_calls} judge={ag.judge_calls}")
        check(f"[{kind}+repair] the deficiency was fed back",
              bool(getattr(ag, "saw_deficiency", None))
              or bool(getattr(ag, "saw_problems", None)))
        check(f"[{kind}+repair] it then proceeded to code", ag.code_calls >= 1)
        check(f"[{kind}+repair] and evaluated exactly one candidate",
              st2["evaluated"] == 1, str(st2["evaluated"]))
        check(f"[{kind}+repair] it was not abandoned",
              st2["coverage_aborts"] + st2["judge_aborts"] == 0, str(st2))

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

    # 11 -- the chain still verifies ---------------------------------------------------
    v = verify(LEDGER_PATH, KEY.encode())
    check("the ledger verifies after all of it", v.ok, str(getattr(v, "reason", "")))
finally:
    proc.terminate()
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: proc.kill()
    shutil.rmtree(SCRATCH, ignore_errors=True)

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
