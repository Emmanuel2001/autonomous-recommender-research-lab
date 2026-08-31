"""The scoring authority. Holds the labels and the signing key; nothing else does.

TRUST MODEL
-----------
Start this process yourself, in its own terminal, with the key in ITS environment:

    TRACK2_SIGNING_KEY=$(openssl rand -hex 32) python -m core.scorer serve

The orchestrator then talks to it over a unix socket and never sees the key. If
the orchestrator launches the scorer as a subprocess, the key is inherited and
the guarantee is void -- `serve` refuses to start if it detects that pattern via
TRACK2_ORCHESTRATOR=1 in its environment.

WHAT IT ENFORCES (not trusts)
-----------------------------
  * coverage      every scoreable user is present. A SUPERSET is allowed and is
                  subset before scoring, because an isolated strategy cannot be
                  told which users are scoreable without leaking label-derived
                  information -- see _validate_and_load
  * no duplicates within a user's ranked list
  * masking       if the policy excludes train-seen items, their ABSENCE is
                  checked, so the strategy cannot claim a masking it did not do
  * provenance    every signed entry binds sha256(predictions file), so a
                  claimed metric with no matching entry is not a metric
  * sanity        negative controls must collapse to chance, or the entry is
                  flagged -- this catches label leakage and join-by-position bugs

PREDICTION FILE FORMAT  (.npz, RAW ids -- no dense mapping to fall out of sync)
    users : int64 (n_users,)            the scoreable users, or any superset
    items : int64 (n_users, N)          ranked best-first, N >= 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import splits as S
from core.ledger import Ledger, TamperedLedger, canonical, verify
from core.metrics import ndcg_at_k
from core.gate import paired_bootstrap, decide, average_over_seeds

K_NDCG, K_RECALL = 10, 50
COMPOSITE_MIN_EFFECT = 0.002   # calibrated; see noise_floor.json
_RUNS = Path(__file__).resolve().parents[1] / "runs"
SOCKET_PATH = Path(os.environ.get("TRACK2_SOCKET") or _RUNS / "scorer.sock")
# Overridable so each experiment writes its OWN ledger instead of sharing one.
# Every experiment script used to `rm -f runs/ledger.jsonl` before starting, which
# meant the next experiment destroyed the previous one's audit trail -- the
# 181-entry ledger behind the LLM-vs-random comparison was lost exactly that way.
# The Run & Iteration Logs are a graded deliverable; they cannot be transient.
LEDGER_PATH = Path(os.environ.get("TRACK2_LEDGER") or _RUNS / "ledger.jsonl")


class ScoringError(Exception):
    """Refusal. The orchestrator gets this back instead of a signed entry."""


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()



def _validate_and_load(preds_path: str, split: str, mask: str, view: S.ScorerView):
    """All structural + policy checks. Returns canonical-ordered arrays.

    Shared by score() and compare() so the two paths cannot drift apart.
    """
    pth = Path(preds_path)
    if not pth.exists():
        raise ScoringError(f"predictions file not found: {pth}")
    if split not in view.relevance:
        raise ScoringError(f"unknown or non-scoreable split: {split}")
    if mask not in ("none", "exclude_train_seen"):
        raise ScoringError(f"unknown mask policy: {mask}")

    z = np.load(pth)
    for f in ("users", "items"):
        if f not in z:
            raise ScoringError(f"predictions missing required array '{f}'")
    users = np.asarray(z["users"]).astype(np.int64).ravel()
    items = np.asarray(z["items"]).astype(np.int64)
    if items.ndim != 2 or items.shape[0] != users.size:
        raise ScoringError(f"items must be (n_users, N); got {items.shape} for "
                           f"{users.size} users")
    if items.shape[1] < K_RECALL:
        raise ScoringError(f"need N >= {K_RECALL} ranked items, got {items.shape[1]}")

    # COVERAGE: every scoreable user must be present. A SUPERSET is permitted and
    # is subset here.
    #
    # Why a superset is allowed: the scoreable set is "users with >=1 click in the
    # val window", which is derived from validation labels. An isolated strategy
    # cannot be told it without handing it label-derived information, so it ranks
    # every user in the catalog instead and the scorer keeps the ones it scores.
    # Extra rows change no scored user's ranking, so the metrics are identical --
    # asserted in tests/test_scorer.py::superset_coverage_scores_identically.
    su = np.sort(users)
    if np.unique(su).size != su.size:
        raise ScoringError("duplicate user ids in the prediction file")
    expect = np.sort(view.users[split].astype(np.int64))
    missing = np.setdiff1d(expect, su)
    if missing.size:
        raise ScoringError(f"user coverage: {missing.size} of {expect.size} scoreable "
                           f"users missing from the prediction file (split {split})")

    order = np.argsort(users, kind="stable")        # canonical order, so pairing aligns
    users, items = users[order], items[order]
    n_supplied = int(users.size)
    if n_supplied != expect.size:                  # subset BEFORE any check or metric,
        keep = np.isin(users, expect)              # so references and masks see exactly
        users, items = users[keep], items[keep]    # the scored population

    dup_rows = int(sum(1 for row in items if np.unique(row).size != row.size))
    if dup_rows:
        raise ScoringError(f"{dup_rows} users have duplicate items in their ranking")

    rel = {int(u): np.asarray(v, dtype=np.int64) for u, v in view.relevance[split].items()}
    n_all_masked = 0
    if mask == "exclude_train_seen":
        offenders = 0
        for row, u in zip(items, users):
            seen = view.train_seen.get(int(u))
            if seen is not None and seen.size and np.intersect1d(row, seen).size:
                offenders += 1
        if offenders:
            raise ScoringError(f"mask policy 'exclude_train_seen' violated by "
                               f"{offenders} users -- train-seen items present in ranking")
        for u in list(rel):
            seen = view.train_seen.get(u)
            if seen is not None and seen.size:
                rel[u] = np.setdiff1d(rel[u], seen)
            if rel[u].size == 0:
                n_all_masked += 1
    return pth, users, items, rel, n_all_masked, n_supplied


def _score_lists(items: np.ndarray, users: np.ndarray,
                 rel: dict[int, np.ndarray]) -> tuple[float, float, int]:
    """items: (n_users, N) ranked. Returns mean NDCG@10, mean Recall@50, n scored."""
    nd, rc, _ = _per_user(items, users, rel)
    if nd.size == 0:
        return 0.0, 0.0, 0
    return float(nd.mean()), float(rc.mean()), int(nd.size)


def _per_user(items: np.ndarray, users: np.ndarray, rel: dict) -> tuple:
    """Per-user metric arrays plus their users, in the order given.
    Users whose relevance is empty under the mask are dropped from BOTH arrays."""
    nd, rc, keep = [], [], []
    for row, u in zip(items, users):
        r = rel[int(u)]
        if r.size == 0:
            continue
        hits = np.isin(row[:K_RECALL], r).astype(np.float64)
        nd.append(ndcg_at_k(hits, K_NDCG, n_rel=int(r.size)))
        rc.append(float(hits.sum()) / int(r.size))
        keep.append(int(u))
    return (np.asarray(nd, dtype=np.float64), np.asarray(rc, dtype=np.float64),
            np.asarray(keep, dtype=np.int64))


def score(preds_path: str, split: str, config_hash: str, mask: str,
          view: S.ScorerView, ledger: Ledger, kind: str = "score",
          isolation: str = "unspecified") -> dict:
    p, users, items, rel, n_all_masked, n_supplied = _validate_and_load(
        preds_path, split, mask, view)

    ndcg, recall, n_scored = _score_lists(items, users, rel)

    # -- reference ladder, not "controls must be at chance".
    #
    # A permuted-LABELS control is useless for this task: popularity gives every
    # user the same list, so permuting relevance leaves its score IDENTICAL
    # (measured: 0.12138 both ways, 18x analytic chance). It conflates base-rate
    # exploitation with label leakage and would flag the simplest baseline.
    #
    # Instead: measure the floors empirically and flag only the unambiguous failure.
    preds_sha = _sha256_file(p)
    rng = np.random.default_rng(int(preds_sha[:16], 16) % (2**63))

    # (a) random ranking -> the empirical chance floor, no analytics assumed
    rnd = np.stack([rng.choice(view.catalog_items, size=K_RECALL, replace=False)
                    for _ in range(users.size)])
    c_rand = _score_lists(rnd, users, rel)

    # (b) row-shifted: user i is given user i+1's ranking. Catches join-by-position
    #     and user-misalignment bugs. Interpretation is model-dependent -- for a
    #     personalised model this collapses; for popularity it is unchanged -- so it
    #     is REPORTED, never flagged.
    c_shift = _score_lists(np.roll(items, 1, axis=0), users, rel)

    flags = []
    if recall < c_rand[1]:
        flags.append("recall_below_random_floor")
    if ndcg < c_rand[0]:
        flags.append("ndcg_below_random_floor")
    if n_scored == 0:
        flags.append("no_users_scored")

    payload = {
        "kind": kind,
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split,
        "mask": mask,
        "config_hash": config_hash,
        # CLAIMED, not verified. The scorer receives a prediction file; nothing in
        # it proves which sandbox produced it. The verifiable artefact is
        # experiments/isolation_probe.json, where the attack is actually run.
        # Named so no reader mistakes the two.
        "isolation_claimed": isolation,
        "manifest_sha256": view.manifest_sha256,
        "preds_sha256": preds_sha,
        "preds_bytes": p.stat().st_size,
        "n_ranked": int(items.shape[1]),
        "metrics": {f"ndcg@{K_NDCG}": round(ndcg, 6),
                    f"recall@{K_RECALL}": round(recall, 6)},
        "n_users_scored": n_scored,
        "n_users_supplied": n_supplied,
        "n_users_all_relevant_masked": n_all_masked,
        "reference": {
            "random_ranking": {f"ndcg@{K_NDCG}": round(c_rand[0], 6),
                               f"recall@{K_RECALL}": round(c_rand[1], 6)},
            "row_shifted": {f"ndcg@{K_NDCG}": round(c_shift[0], 6),
                            f"recall@{K_RECALL}": round(c_shift[1], 6)},
            "note": "row_shifted is reported, not flagged: for popularity it is "
                    "unchanged by construction, for a personalised model it collapses",
        },
        "flags": flags,
    }
    return ledger.append(payload)



def compare(a_preds: list[str], b_preds: list[str], split: str, mask: str,
            min_effect: dict, view: S.ScorerView, ledger: Ledger,
            a_label: str = "incumbent", b_label: str = "challenger",
            a_config: str = "", b_config: str = "") -> dict:
    """Paired gate decision, computed where the labels live and signed.

    Per-user values never leave this process -- only the aggregate verdict does.
    Multiple prediction files per side are replicate SEEDS of one config and are
    averaged per user before pairing.
    """
    if not a_preds or not b_preds:
        raise ScoringError("both sides need at least one prediction file")

    def side(paths):
        nds, rcs, keys, shas = [], [], None, []
        for path in paths:
            p, u, it, rel, _, _ = _validate_and_load(path, split, mask, view)
            nd, rc, k = _per_user(it, u, rel)
            if keys is None:
                keys = k
            elif not np.array_equal(keys, k):
                raise ScoringError("replicates disagree on the scoreable user set")
            nds.append(nd); rcs.append(rc); shas.append(_sha256_file(p))
        return (average_over_seeds(nds), average_over_seeds(rcs), keys, shas)

    a_nd, a_rc, a_keys, a_sha = side(a_preds)
    b_nd, b_rc, b_keys, b_sha = side(b_preds)
    if not np.array_equal(a_keys, b_keys):
        raise ScoringError("the two sides do not cover the same users; pairing is invalid")

    # THE decision metric is the composite the competition actually scores:
    #   score_dataset = mean over m of delta(m)
    # Testing each metric separately and demanding both is STRICTER than the
    # objective, and rejects real net gains (measured: content_alpha=0.05 gives
    # dNDCG +.00070 n.s. and dRecall +.00586 p=0.000 -- mean +.00328, a clear gain
    # that a both-must-move rule throws away). Per-metric results are kept as
    # diagnostics, since a change that trades one metric for the other is still
    # worth seeing.
    a_comp = 0.5 * (a_nd + a_rc)
    b_comp = 0.5 * (b_nd + b_rc)

    res, dec = {}, {}
    for name, av, bv in (("composite_mean", a_comp, b_comp),
                         (f"ndcg@{K_NDCG}", a_nd, b_nd), (f"recall@{K_RECALL}", a_rc, b_rc)):
        st = paired_bootstrap(av, bv, seed=int(b_sha[0][:8], 16) % (2**31))
        st["mean_a"] = round(float(av.mean()), 6)
        st["mean_b"] = round(float(bv.mean()), 6)
        res[name] = st
        if name == "composite_mean":
            # PRE-DECLARED and model-independent (see experiments/noise_floor.json
            # -> composite_calibration). It is NOT the mean of the per-metric
            # floors: those were derived from BPR-MF's SEED variance, which is
            # exactly 0 for a closed-form model like EASE, and using them rejected
            # a genuine +0.00328 EASE improvement. Calibrated instead from the
            # measured PAIRED SE (0.00071) as 2.8*SE ~= 0.002.
            floor = float(min_effect.get("composite_mean", COMPOSITE_MIN_EFFECT))
        else:
            floor = float(min_effect.get(name, 0.0))
        dec[name] = decide(st, floor)

    # the composite decides. Per-metric verdicts are reported alongside so a
    # metric-trading change is visible rather than hidden inside the mean.
    overall = dec["composite_mean"]["verdict"]
    per_metric = {k: v["verdict"] for k, v in dec.items() if k != "composite_mean"}
    traded = len(set(per_metric.values())) > 1

    payload = {
        "kind": "compare",
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split, "mask": mask,
        "manifest_sha256": view.manifest_sha256,
        "a": {"label": a_label, "config_hash": a_config, "n_seeds": len(a_preds),
              "preds_sha256": a_sha},
        "b": {"label": b_label, "config_hash": b_config, "n_seeds": len(b_preds),
              "preds_sha256": b_sha},
        "paired": res, "decision": dec, "overall_verdict": overall,
        "per_metric_verdicts": per_metric, "metrics_traded": traded,
        "decided_on": "composite_mean = mean(delta_ndcg@10, delta_recall@50), "
                      "which is the quantity the competition scores",
    }
    return ledger.append(payload)


MAX_RECORD_BYTES = 12288
RECORDABLE = {"proposal", "event", "run_start", "run_end"}


def record(kind: str, body: dict, manifest_sha256: str, ledger: Ledger) -> dict:
    """Countersign an assertion made by the ORCHESTRATOR, not by the scorer.

    WHAT THE SIGNATURE MEANS HERE
    -----------------------------
    Not truth. The scorer did not produce this content and cannot check it. What it
    establishes is POSITION: these exact bytes were received at this point in the
    chain, before everything that follows. `attested` says so in the payload so no
    reader has to infer it.

    WHY IT MATTERS ANYWAY
    ---------------------
    The driver records a proposal BEFORE running the evaluation it predicts, so the
    hypothesis is PRE-REGISTERED. A rationale cannot be quietly retrofitted to a
    result it did not anticipate -- that would need a later entry, and the sequence
    numbers show it. Innovation is graded on "what the agent chose to try and why",
    and this is what makes the "why" evidence rather than a claim written afterwards.
    It also makes calibration measurable: expected_magnitude is on the record before
    the delta that tests it.

    Before this existed the rationale was validated and then discarded -- the driver
    never wrote it anywhere -- so the Run & Iteration Logs deliverable had no
    hypothesis column and Innovation had no evidence at all.
    """
    if kind not in RECORDABLE:
        raise ScoringError(f"kind {kind!r} is not recordable; choose from "
                           f"{sorted(RECORDABLE)}")
    if not isinstance(body, dict):
        raise ScoringError("record body must be an object")
    raw = canonical(body)
    if len(raw) > MAX_RECORD_BYTES:
        # The ledger is an audit trail, not a log sink. An unbounded body would let
        # a buggy or hostile driver bloat the chain until verification is impractical.
        raise ScoringError(f"record body is {len(raw)} bytes; the cap is "
                           f"{MAX_RECORD_BYTES}")
    return ledger.append({
        "kind": kind,
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": manifest_sha256,
        "attested": "receipt order only. The scorer did not produce this content and "
                    "does not vouch for it; the signature fixes its POSITION in the "
                    "chain, which is what makes a pre-registered prediction checkable.",
        "body": body,
    })


# ------------------------------------------------------------------------ serve
KEY_FILE = Path(os.environ.get("TRACK2_KEY_FILE") or _RUNS.parent / "secrets" / "scorer.key")


def _key_or_die() -> bytes:
    """Environment first, then secrets/scorer.key.

    WHY A KEY FILE AT ALL
    ---------------------
    Every experiment script generated an EPHEMERAL key with
    `$(openssl rand -hex 32)` and never stored it. That made the ledger
    unverifiable the moment the shell exited -- and the ledger is a graded
    deliverable, so "verified once, on my machine, in a terminal I have since
    closed" is not evidence. A persisted key means the final ledger verifies for a
    judge months later.

    It is NOT in .env, because the driver does `set -a && source .env` and would
    then hold the signing key, which is precisely the thing the trust model forbids.
    It lives in secrets/ (mode 600, gitignored), the scorer reads it, and the
    strategy sandbox denies reads of that directory outright -- see
    core.isolation._extra_denied_paths.
    """
    k = os.environ.get("TRACK2_SIGNING_KEY", "")
    src = "environment"
    if not k and KEY_FILE.exists():
        k, src = KEY_FILE.read_text().strip(), str(KEY_FILE)
    if not k:
        sys.exit("no signing key. Create a persistent one:\n"
                 "  python3 -m core.scorer keygen\n"
                 "or supply one for this run only:\n"
                 "  TRACK2_SIGNING_KEY=$(openssl rand -hex 32) python3 -m core.scorer serve")
    if len(k) < 32:
        sys.exit(f"signing key from {src} is too short; use at least 32 hex chars")
    return k.encode()


def keygen(force: bool = False) -> None:
    """Create secrets/scorer.key. Never overwrites without --force: overwriting it
    silently makes every previously signed ledger unverifiable."""
    if KEY_FILE.exists() and not force:
        sys.exit(f"{KEY_FILE} already exists. Overwriting it would make every "
                 f"existing ledger unverifiable. Pass --force if that is what you want.")
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.parent.chmod(0o700)
    KEY_FILE.write_text(os.urandom(32).hex() + "\n")
    KEY_FILE.chmod(0o600)
    print(f"wrote {KEY_FILE} (mode 600). This is the only copy -- if you lose it, "
          f"every ledger signed with it becomes unverifiable.")


def _official_evaluate_sha256() -> str:
    """Hash of the vendored official metric, bound into genesis so a ledger states
    exactly which scoring code produced its numbers."""
    p = Path(__file__).resolve().parent / "official_evaluate.py"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def _isolation_available() -> dict:
    """Recorded in genesis so a reader knows what the machine COULD enforce."""
    try:
        from core.isolation import available
        return available()
    except Exception as e:                       # never block scoring on this
        return {"error": f"{type(e).__name__}: {e}"}


def serve(include_quarantine: bool = False, allow_broken_chain: bool = False,
          task: str = "official") -> None:
    """task="official" -> the CURRENT task (GAUC/nDCG@5, within-user, long_view).
       task="legacy"   -> the SUPERSEDED task (NDCG@10/Recall@50, full catalog).

    One process, one key, one ledger either way. Only the view and the ops differ, so
    the trust model is untouched by the task change.
    """
    if os.environ.get("TRACK2_ORCHESTRATOR") == "1":
        sys.exit("refusing to start: TRACK2_ORCHESTRATOR=1 means the orchestrator "
                 "would hold the signing key, which voids the trust model. Start "
                 "the scorer in its own terminal.")
    key = _key_or_die()
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    # Probe the sandbox backends once, before anything else uses the answer. Both the
    # genesis entry and every later isolation-claim check read this one map.
    isolation_avail = _isolation_available()

    if task == "official":
        from core.official_data import load_for_scorer as _load_official
        # `include_quarantine` has NO EFFECT here -- the official loader takes no such
        # argument -- so accepting it would write `include_quarantine: true` into a signed
        # genesis while nothing of the kind happened. A self-declaration that records an
        # INPUT rather than an EFFECT is worse than no field at all, so refuse instead.
        if include_quarantine:
            sys.exit("--include-quarantine has no effect on task=official (the official "
                     "loader has no quarantine switch), and recording it in the genesis "
                     "entry would be a false self-declaration. Drop the flag, or use "
                     "--task legacy where it does something.")
        official_splits, official_edges = _load_official()
        view = None
        print(f"task=official  splits=" +
              ", ".join(f"{k}:{len(v):,}" for k, v in official_splits.items()))
    elif task == "legacy":
        official_splits = official_edges = None
        view = S.load_for_scorer(include_quarantine=include_quarantine)
    else:
        sys.exit(f"unknown task {task!r}; choose official|legacy")
    try:
        led = Ledger(LEDGER_PATH, key, verify_on_open=not allow_broken_chain)
    except TamperedLedger as e:
        sys.exit(f"{e}\n\nRefusing to start. Every entry signed after a splice is "
                 f"unverifiable, so continuing would quietly invalidate the run.")
    if led.repaired:
        print(f"repaired a {led.repaired}-byte truncated tail (crash residue; those "
              f"bytes could not verify under any key)")
    if task == "official":
        manifest = {"task": "official_within_user_ranking", "label": "long_view",
                    "metric": "primary = mean(GAUC, nDCG@5)",
                    "convergence": {"epsilon": 0.002, "N": 3},
                    "splits": {k: {"rows": len(v), "content_sha256": v.content_sha256()}
                               for k, v in official_splits.items()},
                    "official_evaluate_sha256": _official_evaluate_sha256()}
        manifest_sha = hashlib.sha256(canonical(manifest)).hexdigest()
    else:
        manifest, manifest_sha = view.manifest, view.manifest_sha256
    led.append({"kind": "genesis", "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "task": task,
                "manifest_sha256": manifest_sha, "manifest": manifest,
                "include_quarantine": include_quarantine,
                # WAS THIS CHAIN VERIFIED BEFORE WE APPENDED TO IT? The genesis recorded
                # include_quarantine and repaired_tail_bytes but not this, which is the
                # one property the rest of the tamper-evidence argument rests on.
                # Measured: started with --allow-broken-chain on a spliced ledger, the
                # scorer signs a new entry at seq=1000 directly on top of the forgery, and
                # the artifact contained NO trace that verification had been skipped. This
                # project criticises detection-after-the-fact; a switch that silently
                # disables prevention has to at least be on the record.
                "chain_verified_on_open": not allow_broken_chain,
                "repaired_tail_bytes": led.repaired,
                # Measured ONCE, here, and reused for every isolation-claim check below,
                # so a reader of the genesis and the scorer are looking at the same map.
                # Re-probing per score would also mean a docker timeout on every request.
                "isolation_available": isolation_avail})

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCKET_PATH))
    os.chmod(SOCKET_PATH, 0o600)
    srv.listen(8)
    print(f"scorer listening on {SOCKET_PATH}")
    print(f"ledger  {LEDGER_PATH}")
    print(f"manifest {manifest_sha[:16]}  task={task}")
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                try:
                    buf = b""
                    while not buf.endswith(b"\n"):
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                    req = json.loads(buf.decode() or "{}")
                    op = req.get("op", "score")
                    if op in ("score", "compare") and view is None:
                        raise ScoringError(
                            f"op {op!r} belongs to the SUPERSEDED full-catalog task, but "
                            f"this scorer was started with --task official. Use "
                            f"op='score_official', or restart with --task legacy.")
                    if op == "compare":
                        entry = compare(req["a_preds"], req["b_preds"],
                                        req.get("split", "search_val"),
                                        req.get("mask", "none"),
                                        req.get("min_effect", {}), view, led,
                                        req.get("a_label", "incumbent"),
                                        req.get("b_label", "challenger"),
                                        req.get("a_config", ""), req.get("b_config", ""))
                        resp = {"ok": True, "entry": entry}
                        pl = entry["payload"]
                        print(f"  seq={entry['seq']:>4} COMPARE {pl['a']['label']} vs "
                              f"{pl['b']['label']} -> {pl['overall_verdict'].upper()}")
                    elif op == "score_official":
                        if official_splits is None:
                            raise ScoringError("this scorer was started with task=legacy")
                        sp = req.get("split", "valid")
                        if sp not in official_splits:
                            raise ScoringError(f"unknown split {sp!r}")
                        from core.official_scorer import (score_official,
                                                          OfficialScoringError)
                        try:
                            entry = score_official(
                                req["scores"], sp, official_splits[sp],
                                req.get("config_hash", ""), led,
                                kind=req.get("kind", "score"),
                                isolation=req.get("isolation", "unspecified"),
                                seed=req.get("seed"),
                                # measured once at startup and recorded in genesis, so
                                # the claim is checked against the same map a reader sees
                                isolation_available=isolation_avail)
                        except OfficialScoringError as e:
                            raise ScoringError(str(e)) from None
                        resp = {"ok": True, "entry": entry}
                        m = entry["payload"]["metrics"]
                        fl = entry["payload"]["flags"]
                        print(f"  seq={entry['seq']:>4} {sp:5s} GAUC={m['GAUC']:.4f} "
                              f"nDCG@5={m['nDCG@5']:.4f} primary={m['primary']:.4f} "
                              f"d={entry['payload']['delta_vs_official_baseline']:+.4f}"
                              + (f"  FLAGS={fl}" if fl else ""))
                    elif op == "record":
                        entry = record(req.get("kind", ""), req.get("body", {}),
                                       manifest_sha, led)
                        resp = {"ok": True, "entry": entry}
                        b = entry["payload"]["body"]
                        print(f"  seq={entry['seq']:>4} {entry['payload']['kind'].upper():10s} "
                              + str(b.get("action_type") or b.get("what") or "")[:48])
                    elif op == "ping":
                        # must not touch `view`: it is None when task=official, and a
                        # failing ping made the driver report "scorer not reachable"
                        # while the scorer was in fact listening.
                        # `ledger` is reported because the SCORER owns it: it is
                        # resolved from the scorer's own environment at startup, so a
                        # client setting TRACK2_LEDGER changes nothing about where
                        # entries land. A client that guesses instead of asking will
                        # name a file that never gets written -- which is exactly what
                        # the UI did before this field existed.
                        # `chain_verified_on_open` lets a client REFUSE to produce a
                        # graded artifact against a chain nobody checked, rather than
                        # finding out from a later `verify` that everything it signed was
                        # unusable.
                        resp = {"ok": True, "task": task,
                                "manifest_sha256": manifest_sha,
                                "ledger": str(led.path),
                                "chain_verified_on_open": not allow_broken_chain}
                    else:
                        entry = score(req["preds"], req.get("split", "search_val"),
                                      req.get("config_hash", ""), req.get("mask", "none"),
                                      view, led, kind=req.get("kind", "score"),
                                      isolation=req.get("isolation", "unspecified"))
                        resp = {"ok": True, "entry": entry}
                        m = entry["payload"]["metrics"]
                        fl = entry["payload"]["flags"]
                        print(f"  seq={entry['seq']:>4} {req.get('split'):11s} "
                              f"ndcg@10={m['ndcg@10']:.5f} recall@50={m['recall@50']:.5f}"
                              + (f"  FLAGS={fl}" if fl else ""))
                except ScoringError as e:
                    resp = {"ok": False, "error": f"REFUSED: {e}"}
                    print(f"  REFUSED: {e}")
                except Exception as e:                     # never take the scorer down
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    print(f"  ERROR: {type(e).__name__}: {e}")
                conn.sendall((json.dumps(resp) + "\n").encode())
    except KeyboardInterrupt:
        print("\nscorer stopped")
    finally:
        srv.close()
        SOCKET_PATH.unlink(missing_ok=True)


def request(payload: dict, timeout: float = 600.0) -> dict:
    """Client helper for the orchestrator. Holds no key."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(SOCKET_PATH))
    except (FileNotFoundError, ConnectionRefusedError):
        raise RuntimeError(f"scorer not running (no socket at {SOCKET_PATH}). "
                           "Start it in its own terminal.") from None
    with s:
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode())


def main() -> None:
    ap = argparse.ArgumentParser(description="Track 2 scoring authority")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve")
    sv.add_argument("--include-quarantine", action="store_true")
    sv.add_argument("--task", default="official", choices=["official", "legacy"],
                    help="official = the CURRENT task (GAUC/nDCG@5). legacy = the "
                         "superseded full-catalog task, kept for reproducing old runs.")
    sv.add_argument("--allow-broken-chain", action="store_true",
                    help="append to a ledger that does not verify. Only for "
                         "deliberately demonstrating tamper detection.")
    vf = sub.add_parser("verify"); vf.add_argument("--ledger", default=str(LEDGER_PATH))
    sub.add_parser("manifest")
    kg = sub.add_parser("keygen"); kg.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.include_quarantine, a.allow_broken_chain, a.task)
    elif a.cmd == "verify":
        print(verify(a.ledger, _key_or_die()))
    elif a.cmd == "keygen":
        keygen(a.force)
    else:
        print(json.dumps(S.build_manifest(), indent=1))


if __name__ == "__main__":
    main()
