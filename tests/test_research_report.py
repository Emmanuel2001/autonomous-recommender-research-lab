"""Tests for the code-first run report.

WHY THIS SUITE IS SHAPED THIS WAY
---------------------------------
A report fails silently. On 2026-08-27 the official report's token card rendered an
em-dash on every real ledger ever produced, behind a green test, because the test asserted
a payload shape no producer writes. So this suite:

  * builds its entries in the SHAPE agent/research_driver.py actually records -- the field
    names here are the ones the driver writes, so a rename breaks this suite;
  * asserts on RENDERED VALUES, not on the presence of a section heading. A card whose
    number is missing renders as `<b>&ndash;</b>`, and that is asserted absent;
  * uses a scratch artifact store with one deliberately CORRUPTED tree, so the integrity
    section is checked in both directions -- a verifier that always says "all match" is
    not a verifier.
"""
import json, re, shutil, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import research_report as RR
from core import artifacts as AR

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

TMP = Path(tempfile.mkdtemp(prefix="track2_report_")); STORE = TMP / "artifacts"

BASE = {"config.json": '{\n  "loss": "pointwise",\n  "k": 16\n}\n',
        "features.py": "def encode(s, e, f):\n    return {}, 0\n",
        "model.py": "class FM:\n    pass\n",
        "train.py": "def fit_predict(data, params):\n    return {}\n"}
CHILD = {**BASE, "config.json": '{\n  "loss": "pairwise",\n  "k": 32\n}\n'}
a0 = AR.put(BASE, root=STORE)
a1 = AR.put(CHILD, root=STORE)

def prop(seq, run, it, tree, parent, diff, **kw):
    body = {"what": "candidate_proposal", "run_id": run, "iteration": it, "attempt": 0,
            "tree": tree, "parent_tree": parent, "artifact_dir": str(STORE / tree),
            "contract_sha256": "c" * 64, "tree_bytes": 500,
            "files": sorted(BASE), "diff_hash": AR.diff_hash(diff),
            "diff_bytes": len(diff.encode()), "diff": diff,
            "hypothesis": kw.get("hyp", "switch the loss to pairwise ranking"),
            "mechanism": "BPR optimises within-user order directly",
            "evidence": "the metric is a ranking metric", "stage": kw.get("stage", "training"),
            "expected_direction": "increase",
            "expected_magnitude": kw.get("pred", 0.003),
            "files_changed": kw.get("changed", ["config.json"]), "config": {"k": 32}}
    return {"seq": seq, "payload": {"kind": "proposal", "body": body}}

def score(seq, tree, primary, kind="score"):
    return {"seq": seq, "payload": {
        "kind": kind, "config_hash": tree, "split": "valid",
        "metrics": {"GAUC": 0.67, "nDCG@5": 0.53, "primary": primary},
        "delta_vs_official_baseline": round(primary - 0.6016, 6),
        "scores_sha256": "d" * 64, "split_content_sha256": "e" * 64}}

def ev(seq, run, what, **kw):
    return {"seq": seq, "payload": {"kind": "event",
                                    "body": {"what": what, "run_id": run, **kw}}}

DIFF = AR.tree_diff(BASE, CHILD)
ENTRIES = [
    # `task` on the GENESIS entry is what agent.official_report.is_official() reads, and a
    # real scorer started with --task official writes exactly that. Omitting it made this
    # fixture disagree with every real ledger and quietly disabled the routing check below.
    {"seq": 0, "payload": {"kind": "genesis", "task": "official",
                           "manifest_sha256": "f" * 64,
                           "manifest": {"task": "kuairand", "label": "long_view"}}},
    {"seq": 1, "payload": {"kind": "run_start", "body": {
        "run_id": "r1", "driver": "research_driver", "mode": "code_first",
        "backend": "gemini", "model": "flash-lite", "isolation": "seatbelt",
        "contract_sha256": "c" * 64, "max_iterations": 50, "max_wall_seconds": 21600,
        "seed_config_overrides": {"loss": "pairwise"},
        "starts_from_official_baseline": False,
        "convergence": {"epsilon": 0.002, "N": 3, "mode": "off"}}}},
    prop(2, "r1", 0, a0.tree, None, AR.tree_diff(None, BASE), changed=[]),
    score(3, a0.tree, 0.601696),
    ev(4, "r1", "iteration_result", iteration=0, tree=a0.tree, outcome="promoted",
       signed_primary=0.601696, train_primary=0.686861, dev_primary=0.601696,
       gap=0.085165, cost={"iteration": 0, "wall_seconds": 30.0, "stages": {}}),
    prop(5, "r1", 1, a1.tree, a0.tree, DIFF, pred=0.003),
    score(6, a1.tree, 0.604601),
    ev(7, "r1", "iteration_result", iteration=1, tree=a1.tree, outcome="promoted",
       signed_primary=0.604601, train_primary=0.665521, dev_primary=0.604601,
       gap=0.06092, expected_magnitude=0.003, delta_vs_champion=0.002905, repairs=1,
       reflection={"mechanism_supported": True, "why": "ranking loss"},
       cost={"iteration": 1, "wall_seconds": 41.0, "stages": {"fit": 30.0, "plan": 5.0}}),
    ev(8, "r1", "candidate_failed", iteration=2, attempt=0, tree="9" * 64,
       error_type="SyntaxError", error="name used prior to global declaration",
       error_frame="features.py:138", error_in="candidate", contract_violation=False),
    ev(9, "r1", "candidate_duplicate", iteration=3, tree=a1.tree,
       first_seen_iteration=1),
    ev(10, "r1", "confirmation_protocol", champion_tree=a1.tree,
       comparison_tree=a0.tree, comparison="seed", seeds=[0, 1, 2, 3, 4]),
    ev(11, "r1", "confirmation_result", champion_tree=a1.tree, comparison_tree=a0.tree,
       comparison="seed", protocol_seq=10, n=5, mean=0.00172, sd=0.000887,
       se=0.000397, t=4.34, wins=5, verdict="CONFIRMED",
       per_seed=[{"seed": i, "base": 0.6017 + i / 1e5, "champ": 0.6046 - i / 1e5,
                  "delta": 0.0029 - i / 5e4} for i in range(5)]),
    score(12, a1.tree, 0.6046, kind="confirmation"),
    {"seq": 13, "payload": {"kind": "run_end", "body": {
        "run_id": "r1", "termination": "budget",
        "stats": {"planned": 4, "evaluated": 2, "failed": 1, "promoted": 2},
        "budget": {"iterations": 4, "max_iterations": 50, "wall_hours": 1.13,
                   "max_wall_seconds": 21600, "tokens_total": 1661310,
                   "iteration_wall_seconds": [30.0, 41.0], "iteration_wall_mean": 35.5,
                   "iteration_wall_max": 41.0,
                   "stage_seconds": {"fit": 60.0, "plan": 10.0, "code": 8.0}},
        "champion": {"tree": a1.tree, "primary": 0.604601},
        "beliefs": {"pairwise ordering": {
            "mechanism": "pairwise ordering", "lane": "objective_loss",
            "belief": "ranking-aligned loss is useful", "llm_confidence": 0.8,
            "evidence_grade": "single_signed_result",
            "supporting_iterations": [1], "contradicting_iterations": [],
            "affected_files": ["config.json"], "effect_range": [0.002, 0.004],
            "unresolved": "architecture interaction", "retry_only_if": "new data",
            "last_updated_iteration": 1}},
        "manual_interventions": []}}},
]

H = RR.build(ENTRIES, artifact_root=STORE)
check("the report builds", len(H) > 5000, f"{len(H):,} bytes")
check("it is recognised as a research ledger", RR.is_research(ENTRIES))

# 1 -- headline cards carry real values, not em-dash placeholders ---------------
# Scoped to the HEADLINE grid only. Each run's Cost section also emits cards labelled
# "Tokens" and "Wall clock", so an unscoped dict keeps the LAST one and the headline
# assertions below would silently be testing a per-run cost card instead.
HEAD = H.split("<h2>")[0]
cards = dict((re.sub("<.*?>", "", l), v)
             for l, v in re.findall(r"<div class=card><span>(.*?)</span><b>(.*?)</b>", HEAD))
check("the headline grid is scoped correctly (6 cards, before the first section)",
      len(cards) == 6, str(sorted(cards)))
check("no card renders as a bare em-dash", "<b>&ndash;</b>" not in H)
check("the confirmed gain is the headline", cards.get("Best CONFIRMED gain") == "+0.001720",
      str(cards.get("Best CONFIRMED gain")))
check("epsilon comparison is computed, not assumed",
      cards.get("Clears organisers' &epsilon;") == "no", str(cards.get("Clears organisers' &epsilon;")))
check("tokens come from run_end.budget (the tokens-card bug)",
      cards.get("Tokens") == "1,661,310", str(cards.get("Tokens")))
check("wall clock is summed across runs", cards.get("Wall clock") == "1.13 h",
      str(cards.get("Wall clock")))
check("entry count is shown so truncation is visible",
      cards.get("Chain") == f"{len(ENTRIES)} entries", str(cards.get("Chain")))

# 2 -- integrity is RECOMPUTED, and can fail ------------------------------------
check("artifacts are re-hashed and match", "all match" in H)
check("pre-registration ordering holds", "ordering holds" in H)
(STORE / a1.tree / "source" / "config.json").write_text('{"loss": "tampered"}\n')
H_bad = RR.build(ENTRIES, artifact_root=STORE)
check("a CORRUPTED artifact is reported as a mismatch, not as 'all match'",
      "MISMATCH" in H_bad and "all match" not in H_bad)
(STORE / a1.tree / "source" / "config.json").write_text(CHILD["config.json"])
check("restoring the artifact restores the pass",
      "all match" in RR.build(ENTRIES, artifact_root=STORE))

late = [e for e in ENTRIES if not (e["payload"].get("kind") == "proposal"
                                  and e["payload"]["body"].get("iteration") == 1)]
H_late = RR.build(late, artifact_root=STORE)
check("a scored tree with NO proposal is reported as a violation",
      "VIOLATED" in H_late and "ordering holds" not in H_late)

# 3 -- the run-log requirement: hypothesis, diff, metrics, recovery -------------
check("the pre-registered hypothesis is rendered",
      "switch the loss to pairwise ranking" in H)
check("the applied diff is rendered as a diff", "<pre class=diff>" in H)
check("added and removed lines are distinguished",
      '<i class=a>' in H and '<i class=d>' in H)
check("the diff hash is shown beside the diff", AR.diff_hash(DIFF)[:16] in H)
check("predicted and actual sit side by side", "+0.00300" in H and "+0.00291" in H)
check("the train/validation gap is rendered", "+0.0852" in H or "0.085" in H)
check("a failure names the error type, frame and whose code it is",
      "SyntaxError" in H and "features.py:138" in H and "candidate" in H)
check("a duplicate refusal is shown", "candidate_duplicate" in H)
check("repair count is shown per iteration", ">1</td>" in H)

# 4 -- confirmation is presented as the only quoted finding ---------------------
blocks = H.split("Paired delta")[1:]
check("one confirmation block per confirmation event", len(blocks) == 1, str(len(blocks)))
# bounded to that table's own body: blocks[0] runs to the end of the document, so an
# unbounded count also picks up iteration and cost rows further down the page.
seed_body = blocks[0].split("</tbody>")[0]
n_seed_rows = len(re.findall(r"<tr><td class=n>\d</td>", seed_body))
check("every seed row is rendered, and only those", n_seed_rows == 5, str(n_seed_rows))
check("each seed row shows base, champion and a signed paired delta",
      len(re.findall(r"<td class='n (?:good|bad)'>[+-]", seed_body)) == 5,
      str(len(re.findall(r"<td class='n (?:good|bad)'>[+-]", seed_body))))
check("t and win count are rendered", "4.34" in H and "5/5" in H)
check("the verdict is rendered", "CONFIRMED" in H)
check("the protocol sequence is cited as pre-fit",
      "recorded at sequence 10" in H and "before any fit" in H)
check("a run with no confirmation says so, rather than implying one",
      "No paired confirmation" in RR.build(
          [e for e in ENTRIES if e["payload"].get("body", {}).get("what")
           != "confirmation_result"], artifact_root=STORE))

# 5 -- disclosures are not silently favourable ---------------------------------
check("a seed override is disclosed prominently",
      "OVERRIDDEN" in H and "loss" in H)
check("disabled convergence is disclosed", "Convergence DISABLED" in H)
check("an ABSENT override field is reported as 'not recorded', never as false",
      "not recorded" in RR.build(
          [({**e, "payload": {**e["payload"], "body": {
              k: v for k, v in e["payload"]["body"].items()
              if k not in ("seed_config_overrides", "starts_from_official_baseline")}}}
            if e["payload"].get("kind") == "run_start" else e) for e in ENTRIES],
          artifact_root=STORE))
check("manual interventions are counted", "Manual interventions:</b> 0" in H)
check("belief confidence is explicitly labelled as LLM-authored and uncalibrated",
      "LLM confidence" in H and "uncalibrated" in H)
check("categorical evidence is rendered separately from confidence",
      "single_signed_result" in H)

# 6 -- confirmation fits must not be counted as run iterations -----------------
d = RR.extract(ENTRIES)
check("confirmation scores are kept apart from iteration scores",
      len(d["confirm_scores"]) == 1 and a1.tree in d["scores"])
check("the iteration count comes from the budget, not from score entries",
      cards.get("Runs") == "1")

# 7 -- calibration and cost ----------------------------------------------------
check("calibration is rendered", "Over-predicted" in H)
check("stage shares are rendered", "Share" in H and "fit" in H)
check("GPU-hours are stated as zero", "GPU-hours" in H and ">0<" in H)

# 8 -- signatures are explicitly NOT claimed ----------------------------------
check("the page says it does not verify signatures",
      "signing key, which a reporting process must not hold" in H)
check("it gives the command that does", "core.scorer verify" in H)

# 9 -- the routing hazard, asserted rather than left as a comment -------------
# A research ledger ALSO satisfies agent.official_report.is_official(), because its scores
# carry `primary`. So agent/control.py's /report route must test is_research FIRST; testing
# officiality first would render code-first runs through a report that knows nothing about
# them, and the output would look plausible rather than wrong.
from agent.official_report import is_official
check("a research ledger also passes is_official (so the route's order matters)",
      RR.is_research(ENTRIES) and is_official(ENTRIES))
_route = (ROOT / "agent" / "control.py").read_text()
check("control.py's /report tests is_research BEFORE is_official",
      _route.index("if is_research(e):") < _route.index("if not is_official(e):"))

print("== D2: the report resolves compact tournament decisions")
from agent import tournament_view as TV

_full = [{"id": "O1", "lane": "objective_loss", "hypothesis": "within-user BPR pairwise",
          "mechanism": "aligns training with the ranking metric",
          "files_to_change": ["train.py"], "expected_magnitude": 0.004,
          "information_value": "settles whether the objective is the lever"},
         {"id": "M1", "lane": "model_architecture", "hypothesis": "LightGBM LambdaRank",
          "mechanism": "tree conjunctions over dense fields",
          "files_to_change": ["model.py"], "expected_magnitude": 0.006,
          "information_value": "settles the tree family"}]
_judge = {"selected_id": "O1", "why_selected": "the objective is the cheapest lever",
          "ranking": [{"id": "O1", "rank": 1}, {"id": "M1", "rank": 2}],
          "architecture_verdicts": [{"candidate_id": "M1", "verdict": "rejected",
                                     "reason": "no user-side features exist yet"}],
          "exploit_or_explore": "exploit"}
_proposal = {"_seq": 6, "iteration": 1, "what": "tournament_proposed",
             "candidates": _full, "coverage": {"lane_coverage_ok": True}}
_compact = {"_seq": 9, "iteration": 1, "what": "tournament", "proposal_seq": 6,
            "selected_id": "O1", "selection_how": "judge_selected_id",
            "candidates_considered": 2,
            "candidate_refs": [{"id": "O1", "lane": "objective_loss"},
                               {"id": "M1", "lane": "model_architecture"}],
            "coverage": {"lane_coverage_ok": True, "missing_required_lanes": []},
            "judge": _judge}

_html = RR._sec_tournaments({"tournaments": [_compact],
                             "tournament_proposals": [_proposal]})
check("a compact decision renders every hypothesis",
      "within-user BPR pairwise" in _html and "LightGBM LambdaRank" in _html)
check("a compact decision renders the judge ranking and rationale",
      "the objective is the cheapest lever" in _html
      and "no user-side features exist yet" in _html)
check("the rendered section names the record the candidates came from",
      "candidates from seq 6" in _html)
check("lane accounting survives compaction without hydration",
      TV.lane_counts(_compact) == {"objective_loss": 1, "model_architecture": 1},
      str(TV.lane_counts(_compact)))

_legacy = {"_seq": 9, "iteration": 1, "what": "tournament", "selected_id": "O1",
           "candidates": _full, "judge": _judge, "coverage": {}}
_legacy_html = RR._sec_tournaments({"tournaments": [_legacy],
                                    "tournament_proposals": []})
check("a historical full-candidate ledger still renders completely",
      "within-user BPR pairwise" in _legacy_html
      and "LightGBM LambdaRank" in _legacy_html)
check("historical lane accounting is unchanged",
      TV.lane_counts(_legacy) == {"objective_loss": 1, "model_architecture": 1})

_orphan_html = RR._sec_tournaments({"tournaments": [_compact],
                                    "tournament_proposals": []})
check("an unresolvable decision states the integrity gap in the report",
      "INTEGRITY" in _orphan_html and "not present in this ledger" in _orphan_html)
check("an unresolvable decision invents no hypothesis prose",
      "within-user BPR pairwise" not in _orphan_html)
check("an unresolvable decision still reports its lane accounting",
      TV.lane_counts(_compact)["model_architecture"] == 1)

_noseq = {k: v for k, v in _compact.items() if k != "proposal_seq"}
_noseq_html = RR._sec_tournaments({"tournaments": [_noseq],
                                   "tournament_proposals": []})
check("a decision naming no proposal reports unavailable data, not an integrity breach",
      "unavailable" in _noseq_html and "INTEGRITY" not in _noseq_html)


shutil.rmtree(TMP, ignore_errors=True)
print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
