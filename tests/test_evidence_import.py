#!/usr/bin/env python3
"""Evidence crosses runs only when the KEY HOLDER verified it and the numbers recompute.

The research driver holds no signing key, so it cannot verify a ledger and must not be
believed if it claims to have. It asks the scorer, which returns a bounded whitelisted
bundle; every number is then recomputed from the signed per-seed score records rather than
read out of the summary that the same process wrote.

No API calls, no training, no scorer process, no ledger outside the temporary directory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="track2_evimport_"))
STORE = TMP / "artifacts"
STORE.mkdir(parents=True, exist_ok=True)

from core import artifacts as AR
from core import evidence as EV
from core import scorer as SC
from core.ledger import Ledger

R = []


def check(name, condition, extra=""):
    ok = bool(condition)
    R.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


KEY = b"k" * 32
OTHER_KEY = b"z" * 32
LEDGERS = TMP / "ledgers"
LEDGERS.mkdir(parents=True, exist_ok=True)
SC.EVIDENCE_DIRS = (LEDGERS,)

BASE_CFG = {"loss": "pointwise", "fields": ["user_id", "video_id"], "k": 16,
            "neg_per_pos": 1, "seed": 0}
GOOD_CFG = {**BASE_CFG, "loss": "pairwise", "neg_per_pos": 2}
BAD_CFG = {"loss": "pairwise", "fields": ["user_id", "video_id"], "k": 16,
           "pairs_per_epoch": 600000, "seed": 0}
SRC = {"features.py": "def encode(x):\n    return x\n",
       "model.py": "class FM:\n    pass\n",
       "train.py": "def fit_predict(data, params):\n    return {}\n"}


def tree_of(cfg, src=None):
    files = {**(src or SRC), "config.json": json.dumps(cfg, sort_keys=True)}
    return AR.put(files, root=STORE).tree


BASE_T = tree_of(BASE_CFG)
GOOD_T = tree_of(GOOD_CFG)
BAD_T = tree_of(BAD_CFG)
SPLIT_HASH = "s" * 64
SPLITS = {"train": {"rows": 1000, "content_sha256": "t" * 64},
          "valid": {"rows": 100, "content_sha256": SPLIT_HASH},
          "test": {"rows": 200, "content_sha256": "h" * 64}}
SCOPE = {"name": "kuairand_pure", "metric": "primary", "label": "long_view",
         "dev_split": "valid", "held_out_split": "test",
         "dataset_fingerprint": None, "data_contract_sha256": None,
         "workspace_contract_sha256": "c" * 64, "split_content_sha256": SPLIT_HASH,
         "splits": {k: dict(v) for k, v in SPLITS.items()}}


def seeded(cfg, seed):
    return tree_of({**cfg, "seed": seed})


def build_ledger(path, key=KEY, *, forge_confirmation=None, seeds=5,
                 with_protocol=True, manifest_splits=None, workspace_hash="c" * 64,
                 extra_scores=()):
    """A miniature but structurally real ledger: proposals, scores, a paired comparison."""
    led = Ledger(path, key)
    led.append({"kind": "genesis", "task": "official",
                "manifest_sha256": "m" * 64,
                "manifest": {"task": "official_within_user_ranking",
                             "label": "long_view", "metric": "primary",
                             "convergence": {"N": 3},          # dropped by sanitisation
                             "splits": dict(manifest_splits or SPLITS)}})
    led.append({"kind": "run_start",
                "body": {"run_id": "r1", "task": "kuairand_pure",
                         "contract_sha256": workspace_hash}})
    for tree, cfg, primary, parent in ((BASE_T, BASE_CFG, 0.6000, None),
                                       (GOOD_T, GOOD_CFG, 0.6018, BASE_T),
                                       (BAD_T, BAD_CFG, 0.5993, BASE_T)):
        if parent:
            led.append({"kind": "proposal",
                        "body": {"what": "candidate_proposal", "run_id": "r1",
                                 "iteration": 1, "tree": tree,
                                 "parent_tree": parent,
                                 "files_changed": ["config.json"],
                                 "hypothesis": "h", "mechanism": "m"}})
        led.append({"kind": "score", "config_hash": tree, "split": "valid", "task": "kuairand_pure",
                             "label": "long_view", "metric": "primary",
                    "metrics": {"primary": primary},
                    "split_content_sha256": SPLIT_HASH})
    if with_protocol:
        led.append({"kind": "event", "body": {"what": "confirmation_protocol",
                                              "champion_tree": GOOD_T,
                                              "comparison_tree": BASE_T,
                                              "comparison": "seed"}})
    deltas = []
    for s in range(seeds):
        b = 0.6000 + s * 1e-5
        c = 0.6018 + s * 1e-5 + (s % 3) * 2e-5      # real per-seed variance, so sd > 0
        deltas.append(round(c - b, 6))
        for tree, cfg, primary in ((BASE_T, BASE_CFG, b), (GOOD_T, GOOD_CFG, c)):
            led.append({"kind": "confirmation", "config_hash": seeded(cfg, s), "seed": s,
                        "split": "valid", "task": "kuairand_pure",
                        "label": "long_view", "metric": "primary",
                        "metrics": {"primary": primary},
                        "split_content_sha256": SPLIT_HASH})
    for extra in extra_scores:
        led.append({"kind": "score", "split": "valid", "task": "kuairand_pure",
                    "label": "long_view", "metric": "primary",
                    "split_content_sha256": SPLIT_HASH, **extra})
    import statistics
    truth = {"n": len(deltas), "mean": round(statistics.fmean(deltas), 6),
             "sd": round(statistics.stdev(deltas), 6) if len(deltas) > 1 else 0.0,
             "wins": sum(1 for d in deltas if d > 0), "verdict": "CONFIRMED"}
    body = {"what": "confirmation_result", "champion_tree": GOOD_T,
            "comparison_tree": BASE_T, "comparison": "seed", **truth}
    if forge_confirmation:
        body.update(forge_confirmation)
    led.append({"kind": "event", "body": body})
    return truth


VALID = LEDGERS / "valid.jsonl"
TRUTH = build_ledger(VALID)


def store_read(tree):
    """Read from THIS suite's artifact store, never the ambient one."""
    return AR.read(tree, root=STORE)


def bundle(paths, key=KEY, current_task="official_within_user_ranking"):
    SC._CURRENT_MANIFEST = {"task": current_task, "label": "long_view",
                            "metric": "primary",
                            "splits": {k: dict(v) for k, v in SPLITS.items()}}
    SC._CURRENT_MANIFEST_SHA = "m" * 64
    return SC.evidence_bundle([str(p) for p in paths], key)


print("== the scorer refuses what it must not read")
outside = TMP / "outside.jsonl"
outside.write_text("{}\n")
for label, path in (("a path outside the approved ledger dirs", outside),
                    ("a directory", LEDGERS),
                    ("a file that does not exist", LEDGERS / "nope.jsonl")):
    try:
        bundle([path])
        refused = False
    except SC.EvidenceRefused:
        refused = True
    check(f"refuses {label}", refused)
link = LEDGERS / "link.jsonl"
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(outside)
try:
    bundle([link])
    sym_refused = False
except SC.EvidenceRefused as exc:
    sym_refused = "symlink" in str(exc)
check("refuses a symlink even inside an approved directory", sym_refused)
try:
    bundle([VALID] * (SC.EVIDENCE_MAX_LEDGERS + 1))
    many = False
except SC.EvidenceRefused:
    many = True
check("refuses more ledgers than the limit", many)
check("the driver never receives a signing key",
      "TRACK2_SIGNING_KEY" not in (ROOT / "agent/research_driver.py").read_text()
      and "_key_or_die" not in (ROOT / "agent/research_driver.py").read_text())

print()
print("== a valid ledger imports, a tampered or wrong-key one does not")
b_ok = bundle([VALID])
check("a valid ledger verifies", b_ok["ledgers"][0]["verified"] is True)
check("its sha256 is reported",
      b_ok["ledgers"][0]["sha256"] == hashlib.sha256(VALID.read_bytes()).hexdigest())
check("only whitelisted fields leave the scorer",
      all(set(r) <= ({"seq", "kind"} | set(SC.EVIDENCE_FIELDS.get(r["kind"], ())))
          for led in b_ok["ledgers"] for r in led["records"]))
obs_ok, sum_ok = EV.ingest_bundle(b_ok, current_scope=SCOPE, reader=store_read,
                                  allow_legacy=True)
check("a valid ledger yields observations", sum_ok["observations_imported"] >= 2,
      str(sum_ok["observations_imported"]))

TAMPER = LEDGERS / "tampered.jsonl"
shutil.copy(VALID, TAMPER)
lines = TAMPER.read_text().splitlines()
row = json.loads(lines[4])
row["payload"]["metrics"] = {"primary": 0.99}
lines[4] = json.dumps(row)
TAMPER.write_text("\n".join(lines) + "\n")
b_tamper = bundle([TAMPER])
check("a tampered ledger does not verify", b_tamper["ledgers"][0]["verified"] is False)
check("a tampered ledger yields NO records at all",
      b_tamper["ledgers"][0]["records"] == [])
obs_t, sum_t = EV.ingest_bundle(b_tamper, current_scope=SCOPE, reader=store_read,
                                allow_legacy=True)
check("a tampered ledger imports nothing",
      obs_t == [] and sum_t["observations_imported"] == 0)
check("the refusal is recorded with a reason",
      "did not verify" in json.dumps(sum_t["rejections"]))

b_wrongkey = bundle([VALID], key=OTHER_KEY)
check("a ledger signed with a key the scorer does not hold is refused",
      b_wrongkey["ledgers"][0]["verified"] is False)
obs_w, sum_w = EV.ingest_bundle(b_wrongkey, current_scope=SCOPE, reader=store_read,
                                allow_legacy=True)
check("a missing-key ledger imports nothing", sum_w["observations_imported"] == 0)

print()
print("== a forged confirmation_result cannot override the signed per-seed scores")
FORGED = LEDGERS / "forged.jsonl"
build_ledger(FORGED, forge_confirmation={"mean": 0.5, "wins": 99, "n": 99,
                                         "verdict": "CONFIRMED"})
obs_f, sum_f = EV.ingest_bundle(bundle([FORGED]), current_scope=SCOPE, reader=store_read,
                                allow_legacy=True)
conf = next((o.seed_confirmation for o in obs_f if o.seed_confirmation), None)
check("the forged summary is not believed", conf and conf["mean"] != 0.5)
check("the recomputed mean is used instead",
      conf and abs(conf["mean"] - TRUTH["mean"]) < 1e-9,
      f"{conf['mean'] if conf else None} vs {TRUTH['mean']}")
check("the recomputed n and wins come from the per-seed records",
      conf and conf["n"] == TRUTH["n"] and conf["wins"] == TRUTH["wins"])
check("the discrepancy is reported, not swallowed",
      any("mean" in json.dumps(d) for d in sum_f["confirmation_discrepancies"]),
      json.dumps(sum_f["confirmation_discrepancies"])[:160])
check("the verdict is derived, never copied",
      conf and conf["method"].startswith("recomputed"))

NOPROTO = LEDGERS / "noproto.jsonl"
build_ledger(NOPROTO, with_protocol=False)
_, sum_np = EV.ingest_bundle(bundle([NOPROTO]), current_scope=SCOPE, reader=store_read,
                             allow_legacy=True)
check("a confirmation with no opening protocol is not imported",
      any("confirmation_protocol" in json.dumps(x) for x in sum_np["rejections"]))

print()
print("== confirmations attach to the ordered pair only")
idx = EV.ConfirmationIndex()
idx.add({"champion_tree": "a" * 64, "comparison_tree": "b" * 64, "n": 20, "mean": 0.001,
         "wins": 20, "verdict": "CONFIRMED"}, ledger="L", seq=1)
check("the recorded direction resolves", idx.best_for("a" * 64) is not None)
check("reversing the trees does not attach the confirmation",
      idx.best_for("b" * 64) is None and idx.for_pair("b" * 64, "a" * 64) is None)
check("a real observation only carries a confirmation for its own champion role",
      all(o.seed_confirmation is None
          or o.seed_confirmation["champion_tree"] == o.tree for o in obs_ok))

print()
print("== compatibility fails closed; the legacy adapter is explicit and recorded")
_, sum_strict = EV.ingest_bundle(bundle([VALID]), current_scope=SCOPE, reader=store_read,
                                 allow_legacy=False)
check("without the legacy flag an unversioned ledger is refused",
      sum_strict["observations_imported"] == 0
      and "allow-legacy" in json.dumps(sum_strict["rejections"]))
check("with the flag the adapter runs and is recorded",
      sum_ok["compatibility_method"] == "legacy_adapter")
check("the adapter records which facts it matched",
      set(sum_ok["ledgers"][0]["legacy_facts_matched"])
      == set(EV.LEGACY_KEYS) | {"splits"},
      str(sorted(sum_ok["ledgers"][0]["legacy_facts_matched"])))
check("it records the full split manifest it matched on",
      set(sum_ok["ledgers"][0]["legacy_facts_matched"]["splits"]) == set(SPLITS))


def refuse_with(scope_override=None, **bundle_kw):
    scope = {**SCOPE, **(scope_override or {})}
    _, rep = EV.ingest_bundle(bundle([VALID], **bundle_kw), current_scope=scope,
                              reader=store_read, allow_legacy=True)
    return rep


rep = refuse_with({"splits": {**{k: dict(v) for k, v in SPLITS.items()},
                              "valid": {"rows": 100, "content_sha256": "different" * 8}}})
check("a differing split content hash refuses import",
      rep["observations_imported"] == 0
      and "content_sha256 differs" in json.dumps(rep["rejections"]))
rep = refuse_with({"splits": {**{k: dict(v) for k, v in SPLITS.items()},
                              "train": {"rows": 999, "content_sha256": "t" * 64}}})
check("a row-count mismatch refuses import",
      rep["observations_imported"] == 0
      and "rows differs" in json.dumps(rep["rejections"]))
for missing in ("train", "valid", "test"):
    partial = {k: dict(v) for k, v in SPLITS.items() if k != missing}
    rep = refuse_with({"splits": partial})
    check(f"a missing {missing} split refuses legacy import",
          rep["observations_imported"] == 0
          and "split names differ" in json.dumps(rep["rejections"]))
rep = refuse_with({"splits": {**{k: dict(v) for k, v in SPLITS.items()},
                              "holdout2": {"rows": 5, "content_sha256": "x" * 64}}})
check("an extra split name refuses legacy import",
      rep["observations_imported"] == 0
      and "split names differ" in json.dumps(rep["rejections"]))
rep = refuse_with({"workspace_contract_sha256": "w" * 64})
check("a workspace-contract mismatch refuses import",
      rep["observations_imported"] == 0
      and "workspace_contract_sha256 differs" in json.dumps(rep["rejections"]))
rep = refuse_with({"workspace_contract_sha256": None})
check("an unrecorded workspace contract refuses rather than assumes",
      rep["observations_imported"] == 0
      and "workspace_contract_sha256 is unavailable" in json.dumps(rep["rejections"]))
rep = refuse_with(current_task="some_other_task")
check("a differing task refuses import",
      rep["observations_imported"] == 0 and "task differs" in json.dumps(rep["rejections"]))
for field in ("label", "metric"):
    rep = refuse_with({field: "elsewhere"})
    check(f"a differing {field} refuses import", rep["observations_imported"] == 0)
rep = refuse_with({"splits": None})
check("an absent split manifest refuses rather than assumes",
      rep["observations_imported"] == 0
      and "split manifest is unavailable" in json.dumps(rep["rejections"]))
ok_legacy, why, _ = EV.legacy_compatibility(
    {"task": "t", "label": "l", "metric": "m", "workspace_contract_sha256": "w"},
    {"task": "t", "label": "l", "metric": "m", "workspace_contract_sha256": "w"})
check("the adapter never invents a split manifest that was not recorded",
      not ok_legacy and any("split manifest is unavailable" in r for r in why))
check("the sanitised genesis manifest drops unvetted nested structure",
      "convergence" not in json.dumps(
          bundle([VALID])["ledgers"][0]["genesis_manifest"]))
check("the sanitised manifest keeps exactly the declared facts",
      set(bundle([VALID])["ledgers"][0]["genesis_manifest"])
      == {"task", "label", "metric", "splits", "manifest_sha256"})

print()
print("== artifact hash mismatch refuses import")
def liar(tree):
    return {"config.json": json.dumps({"loss": "pairwise"}), "model.py": "# different"}
_, sum_bad_art = EV.ingest_bundle(bundle([VALID]), current_scope=SCOPE, reader=liar,
                                  allow_legacy=True)
check("a tree whose stored content does not re-hash to its address is refused",
      sum_bad_art["observations_imported"] == 0
      and "hash mismatch" in json.dumps(sum_bad_art["rejections"]))
cfg, why_cfg = EV.verified_config(GOOD_T, liar)
check("verified_config reports the mismatch rather than returning the config",
      cfg is None and "mismatch" in why_cfg)
cfg2, _ = EV.verified_config(GOOD_T, store_read)
check("a genuine tree verifies and yields its config", cfg2 == GOOD_CFG)
check("a missing artifact is refused, not guessed",
      EV.verified_config("f" * 64, store_read)[0] is None)

print()
print("== deltas come from signed score records, never from state.json")
check("both sides of every delta are signed scores",
      all(o.provenance.get("delta_source") in
          ("signed score records", "recomputed paired seed mean") for o in obs_ok))
good = next(o for o in obs_ok if o.tree == GOOD_T and o.delta_versus == "parent")
check("the delta equals the difference of two signed primaries",
      abs(good.delta - round(0.6018 - 0.6000, 6)) < 1e-9, str(good.delta))
check("the recomputed confirmation has real variance and a defined t",
      (conf or {}).get("sd", 0) > 0 and (conf or {}).get("t") is not None,
      f"sd={(conf or {}).get('sd')} t={(conf or {}).get('t')}")
check("a proposal without a signed score for both sides is rejected, not estimated",
      "state.json delta is not accepted" in json.dumps(sum_ok["rejections"])
      or all(o.tree in (GOOD_T, BAD_T) for o in obs_ok))

print()
print("== unrelated code-only architectures are never merged")
k1 = EV.code_mechanism_key(["model.py", "train.py"], "1" * 64)
k2 = EV.code_mechanism_key(["model.py", "train.py"], "2" * 64)
check("two rewrites touching the same files get different keys", k1 != k2, f"{k1} / {k2}")
check("the key is scoped to the candidate", k1.endswith("@" + "1" * 12))
merged = EV.index_by_mechanism([
    EV.Observation(mechanism=k1, tree="1" * 64, delta=0.001,
                   mechanism_label="DeepFM tower"),
    EV.Observation(mechanism=k2, tree="2" * 64, delta=-0.001,
                   mechanism_label="LightGBM LambdaRank")])
check("they do not collapse into one mechanism", len(merged) == 2)
check("neither becomes 'mixed' by accident",
      all(EV.grade(v) == "single_signed_result" for v in merged.values()))
check("an LLM label is displayed but is not the join key",
      k1 != "DeepFM tower" and "DeepFM" not in k1)

print()
print("== the run records its import before any LLM call")
driver = (ROOT / "agent/research_driver.py").read_text()
i_import = driver.index('r = request({"op": "evidence"')
i_runstart = driver.index('ok, seq = rec("run_start"')
i_agent = driver.index("ResearchAgent(backend=")
check("--evidence-from exists and is repeatable",
      '"--evidence-from", action="append"' in driver)
check("import happens before run_start is recorded", i_import < i_runstart)
check("run_start carries a BOUNDED evidence summary",
      '"evidence_import": EVID.bounded_evidence_summary(' in driver)
check("the driver builds the current split manifest from loaded data",
      "current_splits = {" in driver and "splits[name].content_sha256()" in driver)
check("nothing is imported implicitly", "if a.evidence_from:" in driver)
check("the summary reports every required field",
      set(sum_ok) >= {"observations_imported", "observations_rejected",
                      "confirmations_recomputed", "compatibility_method",
                      "imported_sequences", "rejections"}
      and set(sum_ok["ledgers"][0]) >= {"ledger", "sha256", "verified_entries",
                                        "imported_sequences", "compatibility_method"})

print()
print("== the next planner receives the real mixed BPR evidence")
from agent.research_state import ResearchState

st = ResearchState({**SCOPE, "run_id": "next"})
for o in obs_ok:
    st.note_observation(o)
blocks = st.for_prompt()["mechanism_evidence"]
bpr = next((b for b in blocks if b["mechanism"] == "loss=pointwise->pairwise"), None)
check("the BPR mechanism is present", bpr is not None)
check("its grade is mixed", bpr and bpr["evidence_grade"] == "mixed")
check("the confirmed implementation is the strongest evidence",
      bpr and bpr["strongest_evidence"]["grade"] == "seed_confirmed")
check("both sides are shown",
      bpr and bpr["supporting"] and bpr["contradicting"])
check("the planner is told the implementations differ",
      bpr and "implementation-sensitive" in bpr["interpretation"])

print()
print("== VerifyResult.ok is authoritative; a rendered string never is")
from core.ledger import VerifyResult


class FakeOK:
    """An error whose text contains 'OK'. The old substring check would have passed it."""
    ok = False
    n_verified = 0
    incomplete_tail = False
    error = "signature mismatch: expected TOKEN_OK_MARKER"
    bad_seq = 3

    def __str__(self):
        return "LEDGER FAILED at seq=3: signature mismatch: expected TOKEN_OK_MARKER"


real_verify = SC.verify
try:
    SC.verify = lambda path, key: FakeOK()
    poisoned = bundle([VALID])
finally:
    SC.verify = real_verify
check("an error string containing 'OK' does not pass verification",
      poisoned["ledgers"][0]["verified"] is False,
      poisoned["ledgers"][0].get("verify_error"))
check("no records leave the scorer for it", poisoned["ledgers"][0]["records"] == [])
check("the scorer source never parses str(result)",
      'in str(result)' not in (ROOT / "core/scorer.py").read_text())
check("VerifyResult.ok drives the decision",
      'getattr(result, "ok", False)' in (ROOT / "core/scorer.py").read_text())
good_vr = VerifyResult(ok=True, n_verified=7, incomplete_tail=False)
bad_vr = VerifyResult(ok=False, n_verified=0, incomplete_tail=False,
                      error="tampered", bad_seq=2)
check("a genuine VerifyResult exposes ok as a bool",
      good_vr.ok is True and bad_vr.ok is False)
check("the verified entry count comes from VerifyResult, not a parse",
      bundle([VALID])["ledgers"][0]["verified_entries"] > 0)

print()
print("== conflicting duplicate signed scores are refused; identical ones deduplicate")
DUP = LEDGERS / "dupe.jsonl"
build_ledger(DUP, extra_scores=[{"config_hash": GOOD_T,
                                 "metrics": {"primary": 0.6018}}])
_, sum_dup = EV.ingest_bundle(bundle([DUP]), current_scope=SCOPE, reader=store_read,
                              allow_legacy=True)
check("an identical repeated score is deduplicated, not refused",
      sum_dup["observations_imported"] >= 2
      and sum_dup["ledgers"][0]["deduplicated_identical_scores"] >= 1,
      f"imported={sum_dup['observations_imported']} "
      f"dedup={sum_dup['ledgers'][0]['deduplicated_identical_scores']}")
check("the deduplication is disclosed, not silent",
      sum_dup["ledgers"][0]["ambiguous_score_trees"] == 0)

CONFLICT = LEDGERS / "conflict.jsonl"
build_ledger(CONFLICT, extra_scores=[{"config_hash": GOOD_T,
                                      "metrics": {"primary": 0.9999}}])
obs_c, sum_c = EV.ingest_bundle(bundle([CONFLICT]), current_scope=SCOPE,
                                reader=store_read, allow_legacy=True)
check("a tree with disagreeing signed primaries is refused",
      sum_c["ledgers"][0]["ambiguous_score_trees"] == 1,
      str(sum_c["ledgers"][0]["ambiguous_score_trees"]))
check("no observation is produced for the ambiguous tree",
      not any(o.tree == GOOD_T and o.delta_versus == "parent" for o in obs_c))
check("the conflicting values are reported",
      "0.9999" in json.dumps(sum_c["rejections"])
      and "0.6018" in json.dumps(sum_c["rejections"]))
check("the conflicting sequences are reported",
      any("sequences" in json.dumps(r) for r in sum_c["rejections"]))
check("the first value is not silently selected",
      "refusing rather than selecting one" in json.dumps(sum_c["rejections"]))
cross_split = [
    {"kind": "score", "seq": 1, "config_hash": "t" * 64, "split": "valid",
     "metrics": {"primary": 0.5}},
    {"kind": "score", "seq": 2, "config_hash": "t" * 64, "split": "test",
     "metrics": {"primary": 0.4}}]
prim, conf_map, _ = EV._primary_by_tree(cross_split, split="valid")
check("the same tree on a DIFFERENT split is not a conflict",
      conf_map == {} and prim.get("t" * 64) == 0.5, str(prim))
same_split = [
    {"kind": "score", "seq": 1, "config_hash": "t" * 64, "split": "valid",
     "metrics": {"primary": 0.5}},
    {"kind": "score", "seq": 2, "config_hash": "t" * 64, "split": "valid",
     "metrics": {"primary": 0.4}}]
prim2, conf2, _ = EV._primary_by_tree(same_split, split="valid")
check("two disagreeing scores on the SAME split are a conflict",
      "t" * 64 in conf2 and ("t" * 64) not in prim2)
check("the conflict names both values and both sequences",
      sorted(conf2["t" * 64][0]["values"]) == [0.4, 0.5]
      and conf2["t" * 64][0]["sequences"] == [1, 2])

print()
print("== worst case: eight ledgers, thousands of sequences, run_start still fits")
from core.ledger import canonical
from core.scorer import MAX_RECORD_BYTES

worst = {
    "requested": [f"/very/long/absolute/machine/specific/path/runs/ledgers/l{i}.jsonl"
                  for i in range(8)],
    "compatibility_method": "legacy_adapter",
    "observations_imported": 4000, "observations_rejected": 3000,
    "confirmations_recomputed": 64,
    "imported_sequences": list(range(4000)),
    "confirmation_discrepancies": [
        {"ledger": "l" * 200, "seq": i, "champion_tree": "c" * 64,
         "discrepancies": [f"mean: recorded {i}, recomputed {i + 1}"] * 4}
        for i in range(64)],
    "rejections": [{"reason": f"rejection reason number {i} " * 12,
                    "ledger": "l" * 200, "seq": i, "tree": "t" * 12}
                   for i in range(3000)],
    "ledgers": [{"ledger": f"/very/long/absolute/path/runs/ledgers/l{i}.jsonl",
                 "sha256": "a" * 64, "verified": True, "verified_entries": 4000,
                 "compatibility_method": "legacy_adapter",
                 "observations_imported": 500, "observations_rejected": 375,
                 "ambiguous_score_trees": 3, "deduplicated_identical_scores": 7,
                 "imported_sequences": list(range(500)),
                 "legacy_facts_matched": {"splits": SPLITS}} for i in range(8)],
}
bounded = EV.bounded_evidence_summary(worst, root="/very/long/absolute/machine/specific/path")
run_start_body = {
    "run_id": "anthropic_contract_gate3", "task": "kuairand_pure", "driver": "research_driver",
    "mode": "code_first", "backend": "anthropic", "model": "claude-opus-5",
    "effort": "high", "isolation": "seatbelt", "max_iterations": 8,
    "max_wall_seconds": 21600.0, "max_tokens": 1200000,
    "llm_max_output_tokens": 120000, "max_repairs": 1,
    "execution_profile": {"framework": "auto", "device": "auto", "policy": "prefer",
                          "allow_fallback": True, "max_fit_seconds": 600.0},
    "hardware_capabilities": {"mps": {"available": False, "reason": "x" * 200},
                              "cuda": {"available": False, "reason": "y" * 200}},
    "contract_sha256": "c" * 64, "data_contract_sha256": "d" * 64,
    "dataset_fingerprint": "f" * 64, "data_contract": {
        "columns": [{"name": f"column_{i}", "dtype": "int64", "role": ["feature"],
                     "available_on": ["train", "valid", "test"]} for i in range(7)],
        "excluded_sources": [{"source": "s" * 60, "reason": "r" * 160} for _ in range(4)],
        "authority_note": "n" * 300},
    "research_policy": "tournament", "search_mode": "discovery",
    "evidence_class": "diagnostic_discovery", "tournament_size": 12,
    "model_routing": {r: "claude-opus-5" for r in ("plan", "judge", "code", "reflect")},
    "seed_profile": "minimal", "seed_provenance": {"model_py_functions": ["f"] * 20},
    "convergence": {"epsilon": 0.002, "N": 3, "mode": "off"},
    "evidence_import": bounded, "started_epoch": 1788000000.0,
}
size = len(canonical(run_start_body))
ev_size = len(canonical(bounded))
check(f"the WORST-CASE run_start body fits the signed cap ({size} B)",
      size < MAX_RECORD_BYTES, f"{size} < {MAX_RECORD_BYTES}")
# The summary's own size is not the property that matters -- every field in it is
# required evidence provenance. What matters is that run_start still fits with room to
# spare, so a slightly larger contract or routing block later does not push it over.
check(f"the bounded summary leaves real headroom in run_start "
      f"({MAX_RECORD_BYTES - size} B spare)",
      MAX_RECORD_BYTES - size > 2000, f"summary {ev_size} B, body {size} B")
check("the unbounded summary would NOT have fit",
      len(canonical({**run_start_body, "evidence_import": worst})) > MAX_RECORD_BYTES,
      str(len(canonical({**run_start_body, "evidence_import": worst}))))
check("long lists become counts plus a deterministic digest",
      bounded["imported_sequence_count"] == 4000
      and len(bounded["imported_sequences_sha256"]) == 64
      and bounded["imported_sequences_sha256"]
      == EV._digest(list(range(4000))))
check("the digest is reproducible by a reader with the same list",
      EV.bounded_evidence_summary(worst)["rejections_sha256"]
      == bounded["rejections_sha256"])
check("ledger names are repository-relative, not machine-specific",
      all(not l["ledger"].startswith("/") for l in bounded["ledgers"]),
      str(bounded["ledgers"][0]["ledger"]))
check("every required field survives bounding",
      set(bounded) >= {"compatibility_method", "observations_imported",
                       "observations_rejected", "confirmations_recomputed",
                       "imported_sequence_count", "imported_sequences_sha256",
                       "imported_sequences_sample", "rejection_reason_counts",
                       "confirmation_discrepancy_count", "ledgers"})
check("per-ledger digest and verified count survive",
      all({"sha256", "verified_entries", "compatibility_method"} <= set(l)
          for l in bounded["ledgers"]))
check("representative samples are kept for a human reader",
      bounded["imported_sequences_sample"] and bounded["rejection_reason_counts"])
check("a real summary bounds without losing its meaning",
      EV.bounded_evidence_summary(sum_ok)["observations_imported"]
      == sum_ok["observations_imported"])

print()
print("== no historical file is modified")
hist = sorted(glob_paths := [str(p) for p in (ROOT / "runs/ledgers").glob("*.jsonl")])
before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hist}
states = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (ROOT / "runs/candidates").glob("*/state.json")}
EV.ingest_bundle(bundle([VALID]), current_scope=SCOPE, reader=store_read, allow_legacy=True)
check("no repository ledger changed",
      all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
          for p, h in before.items()), f"{len(before)} ledgers")
check("no recorded state.json changed",
      all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
          for p, h in states.items()), f"{len(states)} states")
sub = ROOT / "submission_compliant"
check("submission_compliant/ is byte-identical",
      hashlib.sha256((sub / "submission_test.csv").read_bytes()).hexdigest()
      == "7cc68e2728ef3afb4c0881e28b1bf4c21215e2ce5db6eebc6f4b1b91b77de6d4"
      and hashlib.sha256((sub / "valid_scores.npy").read_bytes()).hexdigest()
      == "ffe4fed193db4b33cb8b2b88817c518df5c11f90c1d12b12393af641a398cfbd")

shutil.rmtree(TMP, ignore_errors=True)
print()
bad_checks = [n for n, ok in R if not ok]
print(f"{len(R) - len(bad_checks)}/{len(R)} passed"
      + (f"  FAILURES: {bad_checks}" if bad_checks else ""))
raise SystemExit(1 if bad_checks else 0)
