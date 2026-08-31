"""Report tests. A report fails by rendering something false, so these check the
claims the page makes rather than that it produced bytes.

The synthetic ledgers below are hand-built so each assertion has one cause.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import official_report as R                                 # noqa: E402

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name}  {detail}")


def ent(seq, kind, **payload):
    return {"seq": seq, "payload": dict(kind=kind, **payload)}


def score(seq, cfg, primary, split="valid", delta=None, seed=0, gauc=0.66, ndcg=0.54):
    return ent(seq, "score", config_hash=cfg, split=split, seed=seed,
               metrics={"GAUC": gauc, "nDCG@5": ndcg, "primary": primary},
               delta_vs_official_baseline=(primary - 0.6016 if delta is None else delta),
               reference={"published_baseline": 0.6016, "published_random": 0.4834},
               n_users=22377, n_rows=124909, flags=[], scores_sha256="ab" * 32)


def curve(seq, cfg, hist, loss=None, best=None, name="pointwise", **kw):
    b = {"what": "train_curve", "config_hash": cfg, "loss": name,
         "valid_history": hist, "epochs_run": len(hist),
         "self_reported_best": max(hist) if best is None else best,
         "scorer_primary": max(hist) if best is None else best}
    if loss:
        b["train_loss_history"] = loss
    b.update(kw)
    return ent(seq, "event", body=b)


def proposal(seq, it, at, params, disp, cfg, parent, parent_p, mech="because", mag=0.005):
    return ent(seq, "proposal", body={
        "iteration": it, "action_type": at, "params": params, "disposition": disp,
        "candidate_config_hash": cfg, "parent_config_hash": parent,
        "parent_valid_primary": parent_p,
        "rationale": {"mechanism": mech, "expected_magnitude": mag,
                      "expected_direction": "increase"}})


GEN = ent(0, "genesis", task="official_within_user_ranking",
          manifest_sha256="cd" * 32,
          manifest={"task": "official_within_user_ranking", "label": "long_view",
                    "metric": "primary = mean(GAUC, nDCG@5)",
                    "official_evaluate_sha256": "ef" * 32,
                    "splits": {"train": {"rows": 1141112}, "valid": {"rows": 124909},
                               "test": {"rows": 170588}}})

FULL = [
    GEN,
    ent(1, "run_start", body={"run_id": "r1", "proposer": "gemini:x", "model": "x",
                             "seed": 0, "iterations_requested": 5, "start_loss": "pointwise",
                             "convergence_mode": "official", "epsilon": 0.002, "N": 3,
                             "epsilon_source": "organisers", "submittable": True,
                             "proposer_coverage": "random arm samples 15 of 16"}),
    score(2, "aaa", 0.601696),
    curve(3, "aaa", [0.58, 0.599, 0.601696], loss=[0.63, 0.54, 0.51], best=0.601696),
    proposal(4, 1, "SetLoss", {"loss": "pairwise"}, "evaluated", "bbb", "aaa", 0.601696),
    score(5, "bbb", 0.603570),
    curve(6, "bbb", [0.59, 0.6035, 0.60357], loss=[0.70, 0.65, 0.62], best=0.60357,
          name="pairwise"),
    proposal(7, 2, "SetK", {"k": 32}, "duplicate", "bbb", "aaa", 0.601696),
    ent(8, "event", body={"what": "inspect_result", "diagnostic": "item_coverage",
                          "result": {"unseen_pct": 0.014}}),
    ent(9, "run_end", body={"termination": "iteration limit (5)", "population_size": 2,
                           "best_config_hash": "bbb", "best_valid_primary": 0.60357,
                           "final_incumbent_config_hash": "bbb",
                           "tokens_in": 15399, "tokens_out": 1104,
                           "action_summary": {"SetLoss": {"tried": 1, "improved": 1,
                                                          "duplicate": 0},
                                              "SetK": {"tried": 1, "improved": 0,
                                                       "duplicate": 1}}}),
]

print("== dispatch")
check("official genesis detected", R.is_official(FULL))
check("legacy genesis rejected",
      not R.is_official([ent(0, "genesis", task="full_catalog_retrieval")]))
check("legacy score shape rejected",
      not R.is_official([ent(0, "score", metrics={"ndcg@10": 0.1, "recall@50": 0.2})]))
check("official score shape accepted without genesis",
      R.is_official([ent(0, "score", metrics={"primary": 0.6, "GAUC": 0.6})]))
check("empty ledger is not official", not R.is_official([]))

print("== load tolerance")
tmp = Path("/tmp/_rep_test"); tmp.mkdir(exist_ok=True)
p = tmp / "trunc.jsonl"
p.write_text(json.dumps(FULL[0]) + "\n" + json.dumps(FULL[1]) + "\n" + '{"seq":2,"pay')
e = R.load(p)
check("truncated tail stops cleanly, keeps the prefix", len(e) == 2, f"{len(e)}")
p.write_text("\n\n" + json.dumps(FULL[0]) + "\n\n")
check("blank lines skipped", len(R.load(p)) == 1)

print("== extract")
d = R.extract(FULL)
check("genesis captured", d["genesis"].get("manifest_sha256") == "cd" * 32)
check("scores captured", len(d["scores"]) == 2)
check("proposals captured", len(d["proposals"]) == 2)
check("curves keyed by config", set(d["curves"]) == {"aaa", "bbb"}, str(set(d["curves"])))
check("inspections captured", len(d["inspections"]) == 1)
check("by_cfg maps config -> score", d["by_cfg"]["bbb"]["metrics"]["primary"] == 0.60357)
check("run_start/run_end captured", len(d["run_start"]) == 1 and len(d["run_end"]) == 1)
check("failures empty when none", d["failures"] == [])
b, sd = R._reference(d)
check("baseline read from the ledger's own reference", b == 0.6016, str(b))
check("seed sd is the published 0.0008", sd == 0.0008)
check("reference falls back when absent",
      R._reference({"scores": []})[0] == R.FALLBACK_BASELINE)

print("== split must not bleed across the report (regression: mixed-split arithmetic)")
# The scorer writes published_baseline PER SPLIT: 0.6016 on valid, 0.5946 on test. A
# ledger whose FIRST score is a test row used to hand the TEST baseline to the whole
# report, overstating the gain 4.5x in the direction that flatters the result.
TEST_FIRST = [
    GEN,
    ent(1, "score", config_hash="zzz", split="test", seed=0,
        metrics={"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.594922},
        delta_vs_official_baseline=0.000322,
        reference={"published_baseline": 0.5946, "published_random": 0.4753},
        n_users=1, n_rows=1, flags=[], scores_sha256="cc" * 32),
    score(2, "bbb", 0.603570),
    proposal(3, 1, "SetLoss", {"loss": "pairwise"}, "evaluated", "bbb", "aaa", 0.601696),
]
d_tf = R.extract(TEST_FIRST)
b_tf, _sd = R._reference(d_tf)
check("baseline comes from the VALID row even when a test row is first",
      b_tf == 0.6016, f"got {b_tf} (0.5946 means the test baseline leaked)")
h_tf = R.build(TEST_FIRST)
check("the delta card is the true +0.00197, not the inflated +0.00897",
      "+0.00197" in h_tf and "+0.00897" not in h_tf)
check("a test row is not counted as the best valid score",
      h_tf.count("class=best") <= 1)
check("by_cfg holds only VALID scores",
      all(p.get("split") == "valid" for p in d_tf["by_cfg"].values()),
      str({k: v.get("split") for k, v in d_tf["by_cfg"].items()}))
# same config scored on BOTH splits, test first: `actual` must use the valid primary
BOTH = [
    GEN,
    ent(1, "score", config_hash="bbb", split="test", seed=0,
        metrics={"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.500000},
        delta_vs_official_baseline=-0.0946,
        reference={"published_baseline": 0.5946}, n_users=1, n_rows=1,
        flags=[], scores_sha256="dd" * 32),
    score(2, "bbb", 0.603570),
    proposal(3, 1, "SetLoss", {"loss": "pairwise"}, "evaluated", "bbb", "aaa", 0.601696),
]
d_b = R.extract(BOTH)
check("a config scored on both splits resolves to its VALID primary",
      d_b["by_cfg"]["bbb"]["metrics"]["primary"] == 0.603570,
      str(d_b["by_cfg"]["bbb"]["metrics"]["primary"]))
h_b = R.build(BOTH)
check("actual is +0.00187 (valid-vs-valid), not -0.10170 (test minus valid)",
      "+0.00187" in h_b and "-0.10170" not in h_b)
check("fallback still works when NO valid row exists",
      R._reference({"scores": [(1, TEST_FIRST[1]["payload"])]})[0] == 0.5946)

print("== calibration error must be direction-aware (adversarial review, bug 51)")
# err = abs(abs(exp) - abs(act)) scored an exactly BACKWARDS prediction as 0.00000, and that
# number is published as the page's "mean absolute calibration error". It could not tell a
# perfect prediction from a maximally inverted one.
def _cal(direction, magnitude, cand_primary):
    led = [GEN, score(1, "p0", 0.601696), score(2, "c0", cand_primary),
           ent(3, "proposal", body={
               "iteration": 1, "action_type": "SetLoss", "params": {"loss": "pairwise"},
               "disposition": "evaluated", "candidate_config_hash": "c0",
               "parent_config_hash": "p0", "parent_valid_primary": 0.601696,
               "rationale": {"mechanism": "m", "expected_magnitude": magnitude,
                             "expected_direction": direction}})]
    return R.build(led)

h_back = _cal("increase", 0.005, 0.596696)     # predicted +0.005, actual -0.005
check("an exactly BACKWARDS prediction no longer scores 0.00000",
      "0.01000" in h_back, "err is still direction-blind")
check("the backwards row shows the true actual", "-0.00500" in h_back)
h_right = _cal("increase", 0.005, 0.606696)    # predicted +0.005, actual +0.005
check("an exactly CORRECT prediction scores 0.00000", "0.00000" in h_right)
check("correct and backwards no longer give the same error",
      ("0.00000" in h_right) and ("0.00000" not in h_back.split("Pre-registered")[1]
                                  .split("</table>")[0]),
      "a perfect and an inverted prediction still look identical")
h_dec = _cal("decrease", 0.005, 0.596696)      # predicted -0.005, actual -0.005
check("a 'decrease' rationale is signed negative and scores 0.00000",
      "-0.00500" in h_dec and "0.00000" in h_dec)
check("a decrease prediction that goes UP is penalised",
      "0.01000" in _cal("decrease", 0.005, 0.606696))

print("== a multi-run ledger must not describe a run that never existed (bug 52)")
# Ledger order is COMPLETION order, so run_start[0] and run_end[0] can belong to different
# runs. Verified on the real runs/ledgers/review_probe.jsonl.
def _rs(seq, rid, seed, itreq, conv="official", sub=True):
    return ent(seq, "run_start", body={"run_id": rid, "seed": seed, "proposer": "random",
                                       "iterations_requested": itreq,
                                       "convergence_mode": conv, "submittable": sub})
def _re(seq, rid, term, best, cfg, ti=0, to=0):
    return ent(seq, "run_end", body={"run_id": rid, "termination": term,
                                     "best_valid_primary": best, "best_config_hash": cfg,
                                     "tokens_in": ti, "tokens_out": to})
TWO = [GEN,
       _rs(1, "runA", 0, 40, conv="off", sub=False),
       _rs(2, "runB", 1, 3),
       score(3, "bbb", 0.601696),
       _re(4, "runB", "converged", 0.601696, "bbb", ti=10, to=5),      # B finished FIRST
       score(5, "zzz", 0.612000),
       _re(6, "runA", "iteration limit (40)", 0.612000, "zzz", ti=900000, to=50000)]
d2 = R.extract(TWO)
rr = R._runs(d2)
check("runs are paired by run_id, not position",
      [(a.get("run_id"), b.get("run_id")) for a, b in rr]
      == [("runA", "runA"), ("runB", "runB")],
      str([(a.get("run_id"), b.get("run_id")) for a, b in rr]))
hs, he = R._headline_run(d2, "zzz")
check("the headline run is the one that produced the best score",
      hs.get("run_id") == "runA" and he.get("run_id") == "runA",
      f"{hs.get('run_id')}/{he.get('run_id')}")
h2 = R.build(TWO)
cfg_sec = h2.split("Run configuration")[1]
check("the config table describes runA, the headline owner",
      "runA" in cfg_sec and "iteration limit (40)" in cfg_sec, cfg_sec[:200])
check("runB's termination is NOT shown as the headline run's",
      "converged" not in cfg_sec)
check("the diagnostic run is flagged, not silently accepted",
      "not a submission" in h2 or "not submittable" in h2)
check("the multi-run banner states how many runs",
      "contains <b>2 runs</b>" in h2)
check("tokens are summed across ALL runs, not taken from run_end[0]",
      "950,015" in h2, "under-reported")
h1 = R.build([GEN, _rs(1, "runB", 1, 3), score(2, "bbb", 0.6017),
              _re(3, "runB", "converged", 0.6017, "bbb")])
check("a single-run ledger shows no multi-run banner", "runs</b>" not in h1)
check("a run_end with no matching run_start still renders",
      "<main>" in R.build([GEN, _re(1, "orphan", "x", 0.6, "aaa")]))
check("a run_start with no run_end still renders",
      "<main>" in R.build([GEN, _rs(1, "lonely", 0, 3), score(2, "aaa", 0.6017)]))

print("== the tokens card: nested under stats, and 0 is a measurement (bug 58)")
# run_end.body nests counters under "stats"; reading them at the top level made this card an
# em-dash on EVERY real ledger, and total tokens is a Feasibility rubric item. The old test
# passed because its fixture put them at the top level -- a shape no producer writes.
TOKS = [GEN, score(1, "aaa", 0.6017),
        ent(2, "run_end", body={"run_id": "r", "termination": "converged",
                                "best_config_hash": "aaa",
                                "stats": {"tokens_in": 38930, "tokens_out": 1990,
                                          "tokens_thinking": 0}})]
h_t = R.build(TOKS)
check("tokens are read from run_end.body['stats'], the real shape",
      "40,920" in h_t, "card is still empty on the real shape")
check("thinking tokens are included (the driver's summary counts them)",
      "40,920" in R.build([GEN, score(1, "aaa", 0.6017),
                           ent(2, "run_end", body={"stats": {"tokens_in": 30000,
                                                             "tokens_out": 1990,
                                                             "tokens_thinking": 8930}})]))
_zero = R.build([GEN, score(1, "aaa", 0.6017),
                 ent(2, "run_end", body={"stats": {"tokens_in": 0, "tokens_out": 0,
                                                   "tokens_thinking": 0}})])
check("ZERO tokens renders as 0, not as unknown (it is the control arm's whole point)",
      re.search(r"<span>tokens</span><b>0</b>", _zero) is not None,
      re.search(r"<span>tokens</span><b>([^<]*)</b>", _zero).group(1))
check("and says so", "zero-token control arm" in _zero)
_none = R.build([GEN, score(1, "aaa", 0.6017)])
check("NO run_end renders as unknown, which is different from zero",
      re.search(r"<span>tokens</span><b>&ndash;</b>", _none) is not None,
      re.search(r"<span>tokens</span><b>([^<]*)</b>", _none).group(1))
check("a top-level shape still works if one ever appears",
      "15" in R.build([GEN, score(1, "aaa", 0.6017),
                       ent(2, "run_end", body={"tokens_in": 10, "tokens_out": 5})]))
_multi = R.build([GEN, score(1, "aaa", 0.6017),
                  ent(2, "run_end", body={"stats": {"tokens_in": 100, "tokens_out": 0}}),
                  ent(3, "run_end", body={"stats": {"tokens_in": 200, "tokens_out": 50}})])
check("tokens sum across runs in the real nested shape", "350" in _multi)

print("== a self-contradicting chain is surfaced in Integrity, not buried in a cell")
CONTRA = [GEN, score(1, "ddd", 0.6035)]
CONTRA[1]["payload"]["flags"] = ["isolation_claimed_unavailable:docker"]
h_c = R.build(CONTRA)
check("the contradiction is called out", "contradicts itself" in h_c)
check("it is a warning, not a note", h_c.count("note warn") >= 1)
check("the flag itself is shown", "isolation_claimed_unavailable:docker" in h_c)
check("it says the SCORES are still real measurements",
      "real measurements of real" in h_c)
check("it explains why the field is called isolation_claimed",
      "isolation_claimed</span> for that reason" in h_c)
h_clean = R.build([GEN, score(1, "ddd", 0.6035)])
check("a clean chain says nothing about contradictions",
      "contradicts itself" not in h_clean)
check("unrelated flags do not trigger it",
      "contradicts itself" not in R.build(
          [GEN, dict(score(1, "eee", 0.6),
                     payload=dict(score(1, "eee", 0.6)["payload"],
                                  flags=["below_random_control"]))]))
multi = [GEN, score(1, "f1", 0.60), score(2, "f2", 0.601)]
for e in multi[1:]:
    e["payload"]["flags"] = ["isolation_claimed_unavailable:docker"]
# Assert on the SECTION, not the whole page: the flag also appears once per evaluation
# row's flags cell, which is correct. Counting occurrences page-wide measured the wrong
# thing and failed for the right reason.
_contra = R._sec_contradictions(R.extract(multi))
check("the contradictions table lists each flag once",
      _contra.count("isolation_claimed_unavailable:docker") == 1,
      str(_contra.count("isolation_claimed_unavailable:docker")))
check("and reports how many scores carried it",
      ">2<" in _contra, _contra[-160:])
check("the evaluations table still shows it per row (that is not duplication)",
      R.build(multi).count("isolation_claimed_unavailable:docker") == 3,
      str(R.build(multi).count("isolation_claimed_unavailable:docker")))

print("== curves are valid-split only (same defect class as by_cfg)")
# self_reported_best is a VALIDATION number (the ranker early-stops on valid). Pairing it
# with a test-split scorer_primary would print a huge spurious DISAGREES and tell the
# reader to distrust the whole page. And because curves are stored by plain assignment,
# a later test curve would OVERWRITE the valid one rather than be ignored.
CURVE_SPLITS = [
    GEN,
    score(1, "ccc", 0.603570),
    curve(2, "ccc", [0.59, 0.6035, 0.60357], best=0.60357, name="pairwise"),
]
CURVE_SPLITS[2]["payload"]["body"]["split"] = "valid"
d_cs = R.extract(CURVE_SPLITS)
check("a valid-split curve is kept", set(d_cs["curves"]) == {"ccc"}, str(set(d_cs["curves"])))
tc = json.loads(json.dumps(CURVE_SPLITS[2]))
tc["payload"]["body"]["split"] = "test"
tc["payload"]["body"]["scorer_primary"] = 0.500000
tc["seq"] = 3
d_cs2 = R.extract(CURVE_SPLITS + [tc])
check("a later TEST curve does not overwrite the valid one",
      d_cs2["curves"]["ccc"].get("scorer_primary") == 0.60357,
      str(d_cs2["curves"]["ccc"].get("scorer_primary")))
h_cs2 = R.build(CURVE_SPLITS + [tc])
check("no spurious DISAGREES from a cross-split pairing", "DISAGREES" not in h_cs2)
check("the cross-check still reports the real agreement", "agrees" in h_cs2)
only_test = json.loads(json.dumps(tc))
only_test["seq"] = 2
d_ot = R.extract([GEN, score(1, "ccc", 0.60357), only_test])
check("a test-only curve is dropped entirely", d_ot["curves"] == {}, str(d_ot["curves"]))
legacy = json.loads(json.dumps(CURVE_SPLITS[2]))
legacy["payload"]["body"].pop("split", None)
d_lg = R.extract([GEN, score(1, "ccc", 0.60357), legacy])
check("a curve predating the split field is still kept",
      set(d_lg["curves"]) == {"ccc"}, str(set(d_lg["curves"])))

print("== build: full ledger")
h = R.build(FULL)
for s in ["Search trajectory", "Training curves", "Self-reported vs scorer-measured",
          "Evaluations", "Pre-registered proposals", "Action space coverage",
          "Diagnostics the agent asked for", "Run configuration", "Integrity"]:
    check(f"section present: {s}", s in h)
# `>= 4` page-wide could not detect losing half the curves, because CH.trajectory alone
# contributes 2 polylines. Count inside the TRAINING CURVES section and require the exact
# number: 2 configs x 2 charts (validation primary, training loss) = 4.
_curve_sec = h.split("Training curves")[1].split("<h2")[0]
check("both configs appear in BOTH curve charts (exactly 4 curve polylines)",
      _curve_sec.count("<polyline") == 4, str(_curve_sec.count("<polyline")))
check("both curve charts are actually rendered", _curve_sec.count("<svg") == 2,
      str(_curve_sec.count("<svg")))
check("each config is named in the curve section",
      "aaa" in _curve_sec and "bbb" in _curve_sec)
check("best primary in a card", "0.60357" in h)
check("delta shown with sign", "+0.00197" in h)
check("delta expressed in seed sd", "seed sd" in h)
check("baseline cited", "0.6016" in h)
check("evaluate.py sha shown", "efefefefefefefef" in h)
check("split rows shown", "1,141,112" in h)
check("tokens summed", "16,503" in h)
check("proposer coverage surfaced", "samples 15 of 16" in h)
check("inspection finding rendered", "unseen_pct" in h)
check("duplicate disposition rendered", "duplicate" in h)
check("action bars drawn", "class='bar'" in h)
check("no external requests", "http://" not in h and "https://" not in h)
check("no script tag", "<script" not in h.lower())
check("single <main>", h.count("<main>") == 1 and h.count("</main>") == 1)
check("title present", "<title>" in h)
check("dark theme defined 3 ways",
      "prefers-color-scheme:dark" in h and "[data-theme=dark]" in h
      and "[data-theme=light]" in h)
check("body paints its own background", re.search(r"body\{[^}]*background:var\(--bg\)", h)
      is not None)

print("== crosscheck section")
check("agreement reported", "agrees" in h)
check("no false alarm on agreeing fits", "DISAGREES" not in h)
bad = [GEN, score(2, "aaa", 0.601696),
       curve(3, "aaa", [0.58, 0.61], best=0.610000)]
bad[2]["payload"]["body"]["scorer_primary"] = 0.601696
hb = R.build(bad)
check("real disagreement is flagged", "DISAGREES" in hb)
check("disagreement carries the warn class", "note warn" in hb)
# The REAL ensemble shape, taken from runs/ledgers/final_e2e.jsonl: strategy.ranker.run
# fits each seed, combines them, and never scores the combination, so the info dict has
# member_valid_primary and NO valid_primary_best -- hence no self_reported_best at all.
# The earlier fixture set best=0.6100, a value the real path never produces, and so
# validated a branch reality could not reach while every real ensemble was silently
# dropped from this table.
ens = [GEN, score(2, "eee", 0.602058),
       curve(3, "eee", [0.60, 0.6044], ensemble_seeds=8, ensemble_combine="rank",
             member_valid_primary=[0.601696, 0.602073, 0.602224, 0.601806,
                                   0.601761, 0.601696, 0.600994, 0.601968],
             member_valid_history=[[0.59, 0.601], [0.60, 0.6035]],
             member_train_loss_history=[[0.7, 0.65], [0.7, 0.64]])]
del ens[2]["payload"]["body"]["self_reported_best"]        # as the real path leaves it
ens[2]["payload"]["body"]["scorer_primary"] = 0.602058
he = R.build(ens)
check("an ensemble with NO self_reported_best still appears in the cross-check",
      "eee" in he.split("Self-reported vs scorer-measured")[1].split("<h2")[0])
check("ensemble mismatch is NOT called a bug", "DISAGREES" not in he)
check("ensemble explained instead", "no single self-reported best" in he)
check("the member spread is shown, since that is why ensembling helps",
      "8 seeds" in he and "0.600994" in he, "spread missing")
check("it notes the combination beat the mean of its members",
      "beats the mean of its members" in he)
check("member curves plotted", he.count("<polyline") >= 4, str(he.count("<polyline")))
check("an ensemble does NOT contribute to the single-fit gap verdict",
      "nothing to cross-check" in he, "ensemble leaked into the gap stats")
# a single fit in the same ledger must still be gap-checked
mix = ens + [score(4, "sss", 0.6035),
             curve(5, "sss", [0.60, 0.6035], best=0.6035, name="pairwise")]
mix[-1]["payload"]["body"]["scorer_primary"] = 0.6035
hm = R.build(mix)
check("a single fit alongside an ensemble is still gap-checked", "agrees" in hm)
check("and the ensemble is still explained, not flagged",
      "no single self-reported best" in hm and "DISAGREES" not in hm)

print("== build: degenerate ledgers")
h0 = R.build([GEN])
check("genesis-only does not crash", "<main>" in h0)
check("genesis-only says no evaluations", "no valid-split scores" in h0)
h1 = R.build([GEN, score(1, "a", 0.6)])
check("no curves -> explanatory note, not a blank chart", "No training curves" in h1)
check("no curves -> nothing to cross-check", "nothing to cross-check" in h1)
check("no proposals -> empty-table message", "no proposals recorded" in h1)
check("no inspections -> states it was a choice", "no diagnostics" in h1.lower())
h2 = R.build([GEN, ent(1, "run_end", body={"termination": "x"})])
check("run_end without scores does not crash", "<main>" in h2)

print("== escaping")
eh = R.build([GEN, score(1, "a", 0.6),
              proposal(2, 1, "SetLoss", {"loss": "<img src=x onerror=alert(1)>"},
                       "evaluated", "a", "a", 0.6,
                       mech="<script>alert('x')</script> & \"quoted\"")])
check("mechanism escaped", "<script>alert" not in eh)
check("params escaped", "onerror=alert" not in eh or "&lt;img" in eh)
check("ampersand escaped", "&amp;" in eh)
ih = R.build([GEN, ent(1, "event", body={"what": "inspect_result",
                                         "diagnostic": "<b>x</b>",
                                         "result": "<script>y</script>"})])
check("inspection content escaped", "<script>y" not in ih and "&lt;script&gt;" in ih)

print("== the signing key must never reach the page")
kp = Path(__file__).resolve().parents[1] / "secrets" / "scorer.key"
if kp.exists():
    key = kp.read_text().strip()
    check("key absent from a real rendered report", key and key not in h,
          "KEY LEAKED INTO THE REPORT")
    check("key is long enough for the test to be meaningful", len(key) >= 32, f"{len(key)}")
else:
    print("  skip secrets/scorer.key not present")

print("== failures section")
fh = R.build([GEN, score(1, "a", 0.6),
              ent(2, "event", body={"what": "patch_failed", "patch_hash": "dead" * 4,
                                    "error_type": "ContractViolation",
                                    "error": "returned 3 scores for 124909 rows",
                                    "isolation": "seatbelt"})])
check("failures section appears only when there are failures",
      "Refusals and failures" in fh and "Refusals and failures" not in h1)
check("failure detail rendered", "ContractViolation" in fh)

print("== convergence-off warning")
wh = R.build([GEN, ent(1, "run_start", body={"convergence_mode": "off",
                                             "proposer": "x"}),
              score(2, "a", 0.6)])
check("convergence off is flagged as not submittable", "not a submission" in wh)
check("convergence official is not flagged", "not a submission" not in h)

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
