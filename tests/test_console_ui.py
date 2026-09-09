"""Research Console presentation: the distinctions a judge must not have to infer.

Two of these exist because the numbers are easy to conflate and the conflation would
flatter the project:

  * "improvement over the run's seed" and "improvement over the task's declared baseline"
    are different quantities. The pilot is +0.141514 on the first and -0.042654 on the
    second, so a page that shows one and calls it progress is misleading.
  * when the provider returns no public thought summary, the panel must SAY so. A blank
    section reads as "nothing was recorded"; an invented one would be worse.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import console_ui as UI
from agent import tournament_view as TV          # noqa: E402
from agent.official_report import load as load_ledger   # noqa: E402

R = []


def check(name, cond, extra=""):
    R.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def text_of(h: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))


# ---------------------------------------------------------------- fixtures
PILOT = ROOT / "runs" / "ledgers" / "pilot_portable1.jsonl"
HAVE_PILOT = PILOT.is_file()

SEED, CHAMP, DECLARED = 0.315832, 0.457346, 0.5
GAIN_SEED, GAIN_DECL = 0.141514, -0.042654


def synthetic(with_summary=False, declared=DECLARED):
    """A minimal recorded-shape ledger. Built by hand so the metric assertions do not
    depend on a live run existing."""
    def ent(seq, kind, body=None, **payload):
        p = {"kind": kind}
        p.update(payload)
        if body is not None:
            p["body"] = body
        return {"seq": seq, "payload": p}
    ents = [
        ent(0, "genesis", {}),
        ent(1, "run_start", {"task": "shop_clicks_demo", "driver": "research_driver",
                             "model": "test-model", "isolation": "seatbelt",
                             "model_routing": {"plan": "test-model"},
                             "execution_profile": {"framework": "torch", "device": "mps",
                                                   "accelerator_policy": "prefer",
                                                   "allow_fallback": False},
                             "max_iterations": 1}),
        ent(3, "score", None, metrics={"primary": SEED}, split="valid", tree="a" * 64,
            scores_sha256="s" * 64),
        ent(4, "event", {"what": "seed_baseline_reproduction",
                         "seed_signed_primary": SEED,
                         "published_baseline_valid": declared}),
        ent(5, "event", {"what": "tournament_proposed", "iteration": 1,
                         "candidates": [{"id": "O1", "lane": "objective_loss",
                                         "hypothesis": "pairwise"},
                                        {"id": "M1", "lane": "model_architecture",
                                         "hypothesis": "dropout"}],
                         "coverage": {"lane_coverage_ok": True}}),
        ent(6, "event", {"what": "tournament", "iteration": 1,
                         "proposal_seq": 5,
                         "candidates_considered": 2,
                         "candidate_refs": [{"id": "O1", "lane": "objective_loss"},
                                            {"id": "M1", "lane": "model_architecture"}],
                         "judge": {"selected_id": "O1", "why_selected": "aligned with metric",
                                   "ranking": [{"id": "O1", "rank": 1,
                                                "score_expected_impact": 0.85,
                                                "score_evidence": 0.9,
                                                "score_implementation_risk": 0.1},
                                               {"id": "M1", "rank": 2}],
                                   "architecture_verdicts": [
                                       {"candidate_id": "M1", "verdict": "rejected",
                                        "reason": "sparse embeddings"}]}}),
        ent(7, "proposal", {"what": "candidate_proposal", "iteration": 1,
                            "tree": "b" * 64, "files_changed": ["config.json"],
                            "hypothesis": "pairwise", "expected_magnitude": 0.03,
                            "diff_bytes": 155, "diff_hash": "d" * 64}),
        ent(8, "score", None, metrics={"primary": CHAMP}, split="valid", tree="b" * 64,
            scores_sha256="t" * 64),
        ent(9, "event", {"what": "iteration_result", "iteration": 1,
                         "lane": "objective_loss", "outcome": "promoted",
                         "dev_primary": CHAMP, "delta_vs_champion": GAIN_SEED,
                         "gap": 0.0776,
                         "cost": {"wall_seconds": 19.7, "tokens_total": 28200},
                         "reflection": {"what_happened": "pairwise helped",
                                        "why": "aligns with ranking",
                                        "remaining_uncertainty": "batch size",
                                        "next_question": "more features?",
                                        "belief_updates": [{"belief": "pairwise wins",
                                                            "mechanism": "pairwise",
                                                            "supports": True,
                                                            "confidence": 0.9}]}}),
        ent(10, "run_end", {"champion": {"primary": CHAMP}, "termination": "budget",
                           "stats": {"evaluated": 1, "promoted": 1},
                           "budget": {"iterations": 1, "max_iterations": 1,
                                      "iteration_wall_seconds": [19.7]},
                           "usage_by_role": {"plan": {"calls": 1, "input_tokens": 100,
                                                      "output_tokens": 10,
                                                      "thinking_tokens": 0,
                                                      "total_tokens": 110,
                                                      "seconds": 1.0}}}),
    ]
    if with_summary:
        ents.insert(-1, ent(85, "event", {"what": "llm_thought_summary", "role": "plan",
                                          "summaries": ["Considered two loss families."]}))
    return ents


print("== the two improvement numbers are never conflated")
d = UI.digest(synthetic())
check("digest preserves the pre-judge tournament proposal",
      len(d["tournament_proposals"]) == 1, str(len(d["tournament_proposals"])))
check("digest computes improvement over SEED",
      abs(d["m"]["gain_vs_seed"] - GAIN_SEED) < 1e-9, str(d["m"]["gain_vs_seed"]))
check("digest computes improvement over DECLARED baseline separately",
      abs(d["m"]["gain_vs_declared"] - GAIN_DECL) < 1e-9, str(d["m"]["gain_vs_declared"]))
check("the two are different quantities in this fixture",
      d["m"]["gain_vs_seed"] > 0 > d["m"]["gain_vs_declared"])

ov = UI.view_overview(d, "observed external run", scorer_ready=False)
t = text_of(ov)
check("overview shows the seed score", f"{SEED:.6f}" in t)
check("overview shows the champion", f"{CHAMP:.6f}" in t)
check("overview shows the declared baseline", f"{DECLARED:.6f}" in t)
check("overview shows +improvement over seed", f"{GAIN_SEED:+.6f}" in t, )
check("overview shows the negative delta vs declared baseline", f"{GAIN_DECL:+.6f}" in t)
check("the seed card is labelled as the run's starting champion",
      "starting champion" in t.lower())
check("the declared-baseline card names the published task figure",
      "published with the task" in t.lower())
check("the improvement card explicitly disclaims being a baseline comparison",
      "not a comparison with the task baseline" in t.lower())
check("the outcome sentence states the baseline was NOT beaten",
      "not beaten" in t.lower())
check("the outcome sentence refuses to combine them",
      "never combined" in t.lower())

print("== a promotion is not reported as beating the task")
check("promoted and baseline-not-beaten appear together",
      "promoted within this run" in t.lower() and "not beaten" in t.lower())

print("== provider summary: absent means SAID, not blank")
live_none = UI.view_live(UI.digest(synthetic(with_summary=False)))
check("the exact required sentence is shown when no summary was returned",
      UI.NO_SUMMARY in text_of(live_none), UI.NO_SUMMARY[:44])
check("the reasoning panel is still populated without a summary",
      "pairwise helped" in text_of(live_none))
check("the console labels legacy belief confidence as uncalibrated",
      "LLM confidence=0.9 (uncalibrated)" in text_of(live_none))
check("the panel does not claim to expose chain-of-thought",
      "not hidden chain-of-thought" in text_of(live_none).lower())
live_some = UI.view_live(UI.digest(synthetic(with_summary=True)))
check("when a summary EXISTS the empty-state sentence is not shown",
      UI.NO_SUMMARY not in text_of(live_some))
check("the returned summary text is rendered",
      "Considered two loss families." in text_of(live_some))

print("== missing data is an em dash, never a fabricated zero")
bare = UI.digest([])
check("no seed is None rather than 0", bare["m"]["seed"] is None)
check("no gain is None rather than 0", bare["m"]["gain_vs_seed"] is None)
check("num() renders missing as an em dash", UI.num(None) == "&mdash;")
check("num() does not render missing as 0.000000", UI.num(None) != "0.000000")
check("dash() escapes the value and not the entity",
      UI.dash(None) == "&mdash;" and UI.dash("a<b") == "a&lt;b")
empty_live = UI.view_live(bare)
check("an empty ledger gets an explicit empty state, not a blank panel",
      "no research iteration" in text_of(empty_live).lower())
check("an empty ledger's experiment view says so",
      "no evaluated experiment" in text_of(UI.view_experiments(bare)).lower())

print("== charts refuse to draw from nothing")
one = UI.digest([])
check("the score chart states it needs two points",
      "at least two" in text_of(UI._chart_scores(one)).lower())
check("the budget chart states no usage was recorded",
      "no per-role token usage" in text_of(UI._chart_budget(one)).lower())

print("== requested and resolved devices are separate fields")
check("digest keeps requested device", d["exec"]["requested"] == "mps")
check("digest keeps resolved device as its own key (None when unrecorded)",
      "resolved" in d["exec"])
check("overview labels them as request and resolution",
      "device request" in t.lower() and "resolved target" in t.lower())
check("a resolved target without candidate evidence is explicitly unverified",
      "use unverified" in UI._device_evidence({"device": "mps"}))
check("matching candidate telemetry remains labelled as candidate-reported",
      UI._device_evidence({"device": "mps", "candidate_reported_device": "mps",
                           "candidate_report_status": "reported_match"}) ==
      "mps target; candidate reported mps")
check("a device mismatch is conspicuous",
      "mismatch" in UI._device_evidence(
          {"device": "mps", "candidate_reported_device": "cpu",
           "candidate_report_status": "reported_mismatch"}))

print("== raw JSON is a disclosure, not the primary UI")
check("the reflection JSON sits inside a details element",
      "<details class=adv>" in live_none)
check("raw JSON is not emitted outside a disclosure",
      live_none.index("<details class=adv>") < live_none.index("<pre>"))

print("== tournament candidates are ranked cards with the judge's reasoning")
lt = text_of(live_none)
check("each lane is shown", "objective_loss" in lt and "model_architecture" in lt)
check("the selected candidate is badged", "selected" in lt.lower())
check("an architecture rejection is badged distinctly", "arch rejected" in lt.lower())
check("the judge's explanation is rendered", "aligned with metric" in lt)
check("the architecture verdict reason is rendered", "sparse embeddings" in lt)
check("sub-scores are shown", "0.85" in lt and "0.90" in lt and "0.10" in lt)
check("lane coverage is disclosed as policy, not initiative",
      "policy working" in lt.lower())

print("== shell: five semantic tabs, no external dependency")
page = UI.page(token="tok", run_state="observed external run", scorer_ready=False, d=d,
               ledger_name="fixture.jsonl", verify_note="10 entries loaded")
check("the product title is used", "Autonomous Recommender Research Lab" in page)
# Count ELEMENTS, not substring hits: `role=tabpanel` also appears in the CSS selector
# `[role=tabpanel][hidden]`, so a naive count returns 6.
check("five tab panels exist", page.count('<section role=tabpanel') == 5,
      str(page.count('<section role=tabpanel')))
check("five tabs exist", page.count('<button role=tab') == 5,
      str(page.count('<button role=tab')))
for _v, _lbl in UI.TABS:
    check(f"panel {_v} is present and starts hidden",
          f'<section role=tabpanel id=panel-{_v}' in page and
          page.split(f'id=panel-{_v}')[1].split('>')[0].find('hidden') >= 0)
check("tabs are wired to panels by aria-controls", 'aria-controls=panel-overview' in page)
check("no external host is referenced",
      not re.search(r"https?://(?!127\.0\.0\.1)", page),
      str(re.findall(r"https?://[^\"' ]+", page)[:3]))
check("no CDN script tag", "src=" not in page.split("<script>")[0])
check("viewport meta present for small screens", "width=device-width" in page)
check("the page declares a language", "<html lang=en>" in page)
check("no literal double-escaped entity survives", "&amp;mdash;" not in page)

print("== architecture discovery is visible and cannot look official")
new_run = UI.view_newrun("tok")
check("new-run form exposes search mode", "name=search_mode" in new_run)
check("new-run form has an architecture discovery preset",
      "data-preset=discovery" in new_run and "Architecture discovery" in new_run)
check("discovery preset spends 12 iterations with convergence off",
      "discovery:{iterations:'12',convergence:'off'" in page)
disc_entries = synthetic()
disc_entries[1]["payload"]["body"]["search_mode"] = "discovery"
disc_entries[1]["payload"]["body"]["evidence_class"] = "diagnostic_discovery"
disc = text_of(UI.view_overview(UI.digest(disc_entries), "finished", False)).lower()
check("a discovery run is labelled diagnostic in the overview",
      "architecture discovery" in disc and "diagnostic evidence" in disc)
check("the discovery overview says official confirmation remains separate",
      "cannot replace a separate official confirmation run" in disc)

print("== the console never claims to hold a signing key")
check("the masthead states no signing key is held", "no signing key" in text_of(page).lower())
check("integrity view points at the external verify command",
      "core.scorer verify" in UI.view_integrity(d, "fixture.jsonl", "10 entries loaded"))

# ---------------------------------------------------------------- live pilot ledger
if HAVE_PILOT:
    print("== the preserved pilot ledger renders the required distinctions")
    pd = UI.digest(load_ledger(PILOT), run_name="pilot_portable1.jsonl")
    pt = text_of(UI.view_overview(pd, "observed external run", False))
    for label, want in (("seed", SEED), ("champion", CHAMP), ("declared", DECLARED)):
        got = pd["m"]["seed" if label == "seed" else
                      "champion" if label == "champion" else "declared"]
        check(f"pilot {label} is {want}", got is not None and abs(float(got) - want) < 1e-9,
              str(got))
    check("pilot improvement over seed is +0.141514",
          abs(pd["m"]["gain_vs_seed"] - GAIN_SEED) < 1e-6, str(pd["m"]["gain_vs_seed"]))
    check("pilot delta vs declared baseline is -0.042654",
          abs(pd["m"]["gain_vs_declared"] - GAIN_DECL) < 1e-6,
          str(pd["m"]["gain_vs_declared"]))
    check("pilot page says the declared baseline was not beaten", "not beaten" in pt.lower())
    check("pilot resolved device is recorded as mps",
          pd["exec"]["resolved"] == "mps", str(pd["exec"]))
    check("pilot provider returned no summary, and the page says so",
          not pd["summaries"] and UI.NO_SUMMARY in text_of(UI.view_live(pd)))
else:
    print("  [skip] runs/ledgers/pilot_portable1.jsonl absent; pilot assertions skipped")

print()
print("== D2: the board renders a compact decision, and says so when it cannot")
live = text_of(UI.view_live(d))
check("the board is populated from the compact decision's proposal",
      "pairwise" in live and "dropout" in live, live[:0] or "")
check("the judge ranking survives compaction", "#1" in live and "#2" in live)
check("the judge explanation is rendered", "aligned with metric" in live)
check("the selected candidate is still badged", "selected" in live.lower())

_hist = {**d, "tournaments": [{"seq": 6, "what": "tournament", "iteration": 1,
                               "candidates": [{"id": "O1", "lane": "objective_loss",
                                               "hypothesis": "legacy embedded prose"}],
                               "judge": {"selected_id": "O1", "ranking": [
                                   {"id": "O1", "rank": 1}]}}],
         "tournament_proposals": []}
check("a historical full-candidate ledger still renders",
      "legacy embedded prose" in text_of(UI.view_live(_hist)))

_gap = {**d, "tournaments": [{"seq": 6, "what": "tournament", "iteration": 1,
                              "proposal_seq": 999,
                              "candidate_refs": [{"id": "O1", "lane": "objective_loss"}],
                              "judge": {"selected_id": "O1", "ranking": []}}],
        "tournament_proposals": []}
_gap_text = text_of(UI.view_live(_gap))
check("an unresolvable decision shows an explicit integrity state, not blank content",
      "INTEGRITY" in _gap_text and "not present in this ledger" in _gap_text)
_gap_board = text_of(UI._cand_cards(
    TV.hydrate(_gap["tournaments"][0], _gap["tournament_proposals"])))
check("an unresolvable decision fabricates no hypothesis text on the board",
      "pairwise" not in _gap_board and "dropout" not in _gap_board
      and "legacy embedded" not in _gap_board, _gap_board[:120])

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
raise SystemExit(1 if bad else 0)
