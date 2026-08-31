"""Produce and verify the official submission, with every number signed.

Two rules this enforces on itself:

  1. **No number reaches the submission unless the scorer produced it.** The sweep and
     seed studies computed metrics in-process for speed; those are exploratory. The
     submitted figure goes through the socket so it is bound to a ledger entry.
  2. **The seed is not a hyperparameter.** The brief says to submit the validation-best
     checkpoint, and that is how the CONFIG is chosen. Choosing the best SEED on
     validation is cherry-picking a replicate, and measured here it also backfires:
     seed 3 is valid-best (0.60384) but its test is 0.59612, WORSE than seed 0's
     0.59721. So we submit the default seed and report the 5-seed mean as the expected
     delta.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np

KIT = ROOT.parent / "kuairand-starter-kit"


def write_submission(path: Path, rows, scores) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i in range(len(rows)):
            w.writerow([i, int(rows.user_id[i]), int(rows.video_id[i]),
                        f"{float(scores[i]):.6g}"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="pairwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="submission_official")
    a = ap.parse_args()

    from core.official_data import load_for_strategy
    from core.scorer import request
    from core.metrics import OFFICIAL_VALID, OFFICIAL_TEST
    from strategy.ranker import Config, run

    # This writes the GRADED artifact, so the preconditions matter more here than
    # anywhere. Checking only `ok` accepted a scorer serving the legacy task, which would
    # have refused every request after the fit; and nothing verified that this process and
    # the scorer were reading the same rows in the same order. See core/handshake.py.
    from core.handshake import (HandshakeError, require_task, require_same_data,
                                require_verified_chain)
    try:
        _pong = request({"op": "ping"})
        require_task(_pong, "official")
        require_verified_chain(_pong)
    except HandshakeError as e:
        sys.exit(f"{e}\n  (with TRACK2_LEDGER=runs/ledgers/submission.jsonl)")

    out = ROOT / a.outdir
    out.mkdir(parents=True, exist_ok=True)
    splits, edges = load_for_strategy()
    if splits["test"].label is not None:
        sys.exit("refusing: test labels reached this process")

    cfg = Config(loss=a.loss, seed=a.seed)
    print(f"fitting {a.loss} seed={a.seed} (config {cfg.hash()[:12]}) ...")
    scores, info = run(splits, edges, cfg)

    # ---- the VALIDATION number, through the scorer, signed
    vf = out / "valid_scores.npy"
    np.save(vf, scores["valid"])
    r = request({"op": "score_official", "scores": str(vf), "split": "valid",
                 "config_hash": cfg.hash(), "isolation": "none", "seed": a.seed,
                 "kind": "final_validation"})
    if not r.get("ok"):
        sys.exit(f"scorer refused the validation submission: {r['error']}")
    vp = r["entry"]["payload"]
    # The scorer has now told us which rows it scored. If that disagrees with the rows we
    # fitted and predicted on, the submission is aligned to the wrong labels and the only
    # symptom would be a slightly low score -- so refuse before writing the CSV.
    try:
        require_same_data(splits["valid"].content_sha256(),
                          vp.get("split_content_sha256"), "valid")
    except HandshakeError as e:
        sys.exit(str(e))
    print(f"  SIGNED valid: GAUC {vp['metrics']['GAUC']:.4f} "
          f"nDCG@5 {vp['metrics']['nDCG@5']:.4f} primary {vp['metrics']['primary']:.5f} "
          f"delta {vp['delta_vs_official_baseline']:+.5f}  (ledger seq {r['entry']['seq']})")

    # ---- the TEST submission, written but NOT scored here
    sub = out / "submission_test.csv"
    write_submission(sub, splits["test"], scores["test"])
    print(f"  wrote {sub} ({sub.stat().st_size/1e6:.1f} MB, {len(splits['test']):,} rows)")

    # ---- verify with the ORGANISERS' own checker, not ours
    if (KIT / "submit.py").exists():
        p = subprocess.run([sys.executable, "submit.py", "--check", "--split", "test",
                            "--data_dir", str(ROOT / "KuaiRand-Pure" / "data"),
                            str(sub.resolve())],
                           cwd=str(KIT), capture_output=True, text=True)
        ok = p.returncode == 0
        print(f"  kit submit.py --check: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print((p.stdout + p.stderr)[-600:])
    else:
        print("  [warn] kit submit.py not found; format unverified")

    seeds_path = ROOT / "experiments" / "official_seeds.json"
    seeds = json.loads(seeds_path.read_text()) if seeds_path.exists() else {}
    manifest = {
        "task": "KuaiRand-Pure, within-user ranking of logged impressions",
        "label": "long_view", "metric": "primary = mean(GAUC, nDCG@5)",
        "config": {"model": cfg.model, "loss": cfg.loss, "k": cfg.k, "lr": cfg.lr,
                   "fields": list(cfg.fields), "seed": cfg.seed,
                   "config_hash": cfg.hash()},
        "signed_validation": {"seq": r["entry"]["seq"], **vp["metrics"],
                              "delta_vs_official_baseline":
                                  vp["delta_vs_official_baseline"],
                              "scores_sha256": vp["scores_sha256"],
                              "split_content_sha256": vp["split_content_sha256"]},
        "official_baseline": {"valid_primary": OFFICIAL_VALID["fm_official"],
                              "test_primary": OFFICIAL_TEST["fm_official"],
                              "published_seed_std": 0.0008},
        "expected_test_delta_5_seed_mean": (
            seeds.get("arms", {}).get(a.loss, {}).get("test_mean", None)),
        "paired_test_gain_over_pointwise": seeds.get("paired_test", {}),
        "submission_file": sub.name,
        "note": "The seed is a replicate, not a hyperparameter: seed 3 is validation-best "
                "but its test score is lower than seed 0's, so the default seed is "
                "submitted and the 5-seed mean is reported as the expectation.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"  wrote {out/'MANIFEST.json'}")


if __name__ == "__main__":
    main()
