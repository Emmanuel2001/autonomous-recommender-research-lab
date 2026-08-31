"""Build the official submission from a PROMOTED CANDIDATE ARTIFACT.

WHY THIS EXISTS SEPARATELY FROM make_official_submission.py
-----------------------------------------------------------
That script rebuilds a typed `strategy.ranker.Config` and refits it. There is no way to
express an agent-written candidate as a Config, so a code-first champion had no submission
path at all -- the loop could win and the win could not be shipped.

HOW THE SUBMISSION IS BOUND TO THE CHAMPION
-------------------------------------------
Not by refitting. Agent-written code is not guaranteed deterministic, so a refit could
produce different scores from the ones the scorer actually signed, and the submission would
no longer be the thing that was measured. Instead:

  1. the champion tree is VERIFIED against the content-addressed store -- the stored bytes
     must still hash to the address the ledger names;
  2. the dev scores saved during the scored run are hashed and compared to the
     `scores_sha256` the scorer signed. If they match, this directory is provably the run
     the signed number came from;
  3. only then are the held-out scores from that same directory written as the submission.

So the chain is: signed metric -> hash of the dev scores -> the directory holding them ->
the held-out scores beside them -> the CSV. Every link is checked here, and a break is a
refusal rather than a warning.

The held-out scores were produced inside the sandbox, in a process where the held-out
labels are absent by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from core import artifacts as AR
from core.metrics import OFFICIAL_TEST, OFFICIAL_VALID
from core.official_data import load_for_strategy
from make_official_submission import write_submission

#: The kit sits BESIDE the project, not inside it -- same path make_official_submission.py
#: uses. Getting this wrong reports "format unverified" instead of failing, which is a
#: silent downgrade of the one check the organisers actually run.
KIT = ROOT.parent / "kuairand-starter-kit"


def _rel(p) -> str:
    """Repository-relative, so a cloned submission does not point into someone's home
    directory. An absolute path in a shipped manifest is unusable to a reviewer and leaks
    the build machine's layout."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def find_signed(ledger: Path, tree: str) -> list[dict]:
    """Every signed score entry for this tree, oldest first."""
    out = []
    for line in Path(ledger).read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        pl = e.get("payload", {})
        if pl.get("config_hash") == tree and "metrics" in pl:
            out.append({"seq": e["seq"], **pl})
    return out


def find_run_dir(run_root: Path, tree: str) -> Path | None:
    """The candidate directory whose dev scores match this tree's signed hash.

    Directory names carry the tree prefix, but the NAME is not the evidence -- the hash
    comparison in main() is. This only narrows the search.
    """
    cands = sorted(p for p in Path(run_root).iterdir()
                   if p.is_dir() and tree[:12] in p.name)
    for d in cands:
        if (d / "dev_scores.npy").exists() and (d / "heldout_scores.npy").exists():
            return d
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True,
                    help="runs/candidates/<run_id>/state.json from a research run")
    ap.add_argument("--ledger", default=None,
                    help="ledger holding the signed score (default: runs/ledger.jsonl "
                         "or $TRACK2_LEDGER)")
    ap.add_argument("--outdir", default="submission_research")
    ap.add_argument("--allow-below-baseline", action="store_true",
                    help="write the submission even if the champion does not beat the "
                         "official baseline on validation")
    return ap


def main() -> None:
    a = build_parser().parse_args()
    state = json.loads(Path(a.state).read_text())
    champ = state.get("champion")
    if not champ:
        sys.exit(f"{a.state} records no champion; nothing to submit")
    tree = champ["tree"]
    run_root = Path(a.state).resolve().parent
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    import os
    ledger = Path(a.ledger or os.environ.get("TRACK2_LEDGER")
                  or (ROOT / "runs" / "ledger.jsonl"))
    if not ledger.exists():
        sys.exit(f"no ledger at {ledger}; pass --ledger")

    print(f"champion tree      {tree[:12]}  (iteration {champ.get('iteration')})")
    print(f"recorded dev prim  {champ.get('primary')}")

    # ---- 1. the artifact must still hash to the address the ledger names
    art = AR.verify(tree)
    print(f"artifact VERIFIED  {art.path}  ({art.bytes:,} bytes, {len(art.files)} files)")

    # ---- 2. the signed entry
    signed = find_signed(ledger, tree)
    if not signed:
        sys.exit(f"no signed score for tree {tree[:12]} in {ledger.name}. The submission "
                 f"must be bound to a signed measurement, so this is a refusal.")
    # Bind to the score that PROMOTED the champion (kind "score"), not to whichever
    # measurement happens to be most recent. A confirmation replicate carries the same
    # numbers, but the defensible provenance for a submission is the measurement the loop
    # actually acted on.
    promoting = [x for x in signed if x.get("kind") == "score"]
    sig = promoting[0] if promoting else signed[-1]
    corroborating = [x for x in signed
                     if x is not sig and x.get("scores_sha256") == sig.get("scores_sha256")]
    print(f"signed entry       seq {sig['seq']}  primary {sig['metrics']['primary']} "
          f"({sig.get('split')})  kind={sig.get('kind')}")
    if corroborating:
        print(f"corroborated       {len(corroborating)} further signed fit(s) of this tree "
              f"produced byte-identical scores (seq "
              f"{', '.join(str(x['seq']) for x in corroborating)}) -- the champion is "
              f"reproducible, not merely recorded")

    # ---- 3. the directory whose dev scores the scorer actually hashed
    d = find_run_dir(run_root, tree)
    if d is None:
        sys.exit(f"no candidate directory under {run_root} holds both dev and held-out "
                 f"scores for {tree[:12]}")
    got = sha256_file(d / "dev_scores.npy")
    want = sig.get("scores_sha256")
    if got != want:
        sys.exit(f"REFUSING: the dev scores in {d.name} hash to {got[:16]} but the scorer "
                 f"signed {str(want)[:16]}. This directory is not the run that produced "
                 f"the signed number, so its held-out scores cannot be trusted.")
    print(f"scores bound       {d.name}  dev sha256 {got[:16]} matches the signed value")

    # ---- 4. is it actually worth submitting?
    base = OFFICIAL_VALID.get("fm_official")
    delta = sig["metrics"]["primary"] - base
    print(f"vs official base   {delta:+.6f}  (baseline valid {base})")
    if delta <= 0 and not a.allow_below_baseline:
        sys.exit(f"REFUSING: the champion does not beat the official baseline on "
                 f"validation ({delta:+.6f}). Feasibility is only scored among entries "
                 f"that beat it. Pass --allow-below-baseline to override.")

    # ---- 5. write the submission from THAT directory's held-out scores
    splits, _edges = load_for_strategy()
    if splits["test"].label is not None:
        sys.exit("refusing: held-out labels reached this process")
    held = np.load(d / "heldout_scores.npy")
    if held.size != len(splits[ "test" ]):
        sys.exit(f"REFUSING: {held.size:,} held-out scores for "
                 f"{len(splits['test']):,} rows")
    sub = out / "submission_test.csv"
    write_submission(sub, splits["test"], held)
    print(f"wrote              {sub}  ({sub.stat().st_size/1e6:.1f} MB, "
          f"{len(splits['test']):,} rows)")

    dev_copy = out / "valid_scores.npy"
    dev_copy.write_bytes((d / "dev_scores.npy").read_bytes())

    # ---- 5b. the champion SOURCE, into the submission directory itself.
    # `.gitignore` excludes `runs/*`, so the content-addressed artifact never enters the
    # repository -- a judge cloning it could read the manifest's tree hash and find nothing
    # to verify it against. Exporting the exact verified tree here puts the four files a
    # reviewer needs beside the predictions they produced.
    csrc = out / "champion_source"
    csrc.mkdir(exist_ok=True)
    files = AR.read(tree)
    for rel, text in sorted(files.items()):
        (csrc / rel).write_text(text, encoding="utf-8")
    (csrc / "README.md").write_text(
        f"# Champion source — tree `{tree}`\n\n"
        f"The exact agent-written strategy this submission's predictions came from, exported\n"
        f"from the content-addressed store at `runs/artifacts/{tree}/`.\n\n"
        f"Verify it is the tree the ledger names:\n\n"
        f"```bash\n"
        f"python3 -c \"import sys; sys.path.insert(0,'.'); from core import artifacts as A; \\\n"
        f"  print(A.tree_hash({{p.name: p.read_text() for p in \\\n"
        f"  __import__('pathlib').Path('submission_research/champion_source').glob('*.py')}} | \\\n"
        f"  {{'config.json': open('submission_research/champion_source/config.json').read()}}))\"\n"
        f"```\n\n"
        f"which must print `{tree}`.\n\n"
        f"**Authorship, precisely:** `features.py` and `config.json` were written by the\n"
        f"agent. `model.py` and `train.py` are the seed implementations it inherited and\n"
        f"never edited — see `../../deliverables/AUTONOMY.md`.\n",
        encoding="utf-8")
    print(f"wrote              {csrc}/  ({len(files)} source files, tracked by git)")

    # ---- 6. the ORGANISERS' checker, not ours
    kit_ok = None
    if (KIT / "submit.py").exists():
        p = subprocess.run([sys.executable, "submit.py", "--check", "--split", "test",
                            "--data_dir", str(ROOT / "KuaiRand-Pure" / "data"),
                            str(sub.resolve())],
                           cwd=str(KIT), capture_output=True, text=True)
        kit_ok = p.returncode == 0
        print(f"kit submit.py      {'PASS' if kit_ok else 'FAIL'}")
        if not kit_ok:
            print((p.stdout + p.stderr)[-600:])
    else:
        print("kit submit.py      not found; format unverified")

    # ---- 7. a manifest that names the lineage, not just the number
    lineage = []
    for e in state.get("experiments", []):
        lineage.append({k: e.get(k) for k in
                        ("iteration", "tree", "parent", "outcome", "hypothesis",
                         "files_changed", "diff_hash", "expected_magnitude",
                         "dev_primary", "train_primary", "gap", "repairs")})
    manifest = {
        "task": "KuaiRand-Pure, within-user ranking of logged impressions",
        "label": "long_view", "metric": "primary = mean(GAUC, nDCG@5)",
        "produced_by": "agent/research_driver.py (code-first autonomous loop)",
        "champion": {
            "tree": tree, "artifact": _rel(art.path), "files": list(art.files),
            "tree_bytes": art.bytes, "iteration": champ.get("iteration"),
            "hypothesis": champ.get("hypothesis"),
            "contract_sha256": json.loads(
                (art.path / "meta.json").read_text()).get("meta", {})
                .get("contract_sha256"),
        },
        "signed_validation": {
            "seq": sig["seq"], "kind": sig.get("kind"), **sig["metrics"],
            "corroborating_identical_fits": [
                {"seq": x["seq"], "kind": x.get("kind"), "seed": x.get("seed")}
                for x in corroborating],
            "delta_vs_official_baseline": round(delta, 6),
            "scores_sha256": sig.get("scores_sha256"),
            "split_content_sha256": sig.get("split_content_sha256"),
        },
        "harness_measured": {
            "train_primary": champ.get("train_primary"),
            "dev_primary": champ.get("dev_primary"),
            "train_dev_gap": champ.get("gap"),
            "note": "computed by the harness with the official metric from the "
                    "candidate's returned score vectors; not self-reported",
        },
        "official_baseline": {"valid_primary": base,
                              "test_primary": OFFICIAL_TEST.get("fm_official")},
        "binding": {
            "held_out_scores_from": _rel(d),
            "dev_scores_sha256": got,
            "how": "the dev scores in this directory hash to the value the scorer signed, "
                   "so the held-out scores beside them are from the same fit. The "
                   "submission is NOT a refit.",
        },
        "kit_check_passed": kit_ok,
        "lineage": lineage,
        "manual_interventions": state.get("manual_interventions", []),
        "submission_file": "submission_test.csv",
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=False, allow_nan=False, default=str) + "\n")
    print(f"wrote              {out / 'MANIFEST.json'}")
    print(f"\nsubmission is bound to artifact {tree[:12]}, signed at seq {sig['seq']}.")


if __name__ == "__main__":
    main()
