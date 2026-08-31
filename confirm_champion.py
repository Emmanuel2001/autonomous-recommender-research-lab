"""Paired multi-seed confirmation of a research run's champion.

WHY A SEPARATE STAGE
--------------------
The loop promotes on a single signed comparison at one seed. That is the right rule for
SEARCH -- it keeps exploration moving -- but it is not evidence. A live run promoted twice
on gains of +0.000178 and +0.000077, and a single measurement cannot distinguish either
from noise.

WHY NOT JUST A MAGNITUDE THRESHOLD
----------------------------------
Because "delta must exceed the published seed std (0.0008)" is the wrong statistic here.
A candidate and its parent are fitted at the SAME seed, so the comparison is PAIRED and
most of the seed-to-seed variance cancels. That is exactly how this project established
`hourmin_bucket` at +0.001192 with t=14.04 against sigma=0.0008 -- a magnitude gate at one
sigma would have thrown that real effect away. `core/task.py` says the same thing about
epsilon: it is a keep-searching threshold, not a significance test.

So the question is replication, and this is the test the project's two real findings already
used: fit both trees at k matched seeds, take the paired deltas, report mean / sd / se / t
and the win count.

GENERALITY
----------
Nothing here imports a per-dataset constant. The paired statistics are measured, so the
same stage works for any task; a task's declared `baseline["seed_std"]`, when it has one, is
printed only as a cross-check.

INTEGRITY
---------
The seed list is recorded in the ledger BEFORE any fit runs, so seeds cannot be chosen after
seeing which ones were kind. Every fit is scored by the same signing scorer as the run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import artifacts as AR
from agent import candidate as CAND


def with_seed(files: dict, s: int) -> dict:
    """The same tree with config.json's seed set to `s`.

    The seed lives inside the tree, so each replicate is a distinct candidate with its own
    content address -- which is why a deliberate seed replicate is not rejected as a
    duplicate. That was not true of the typed action space.
    """
    cfg = json.loads(files["config.json"])
    cfg["seed"] = int(s)
    return {**files, "config.json": json.dumps(cfg, indent=2) + "\n"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--seeds", type=int, default=5,
                    help="how many matched seeds to use, starting at --seed-start.")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed. Use this to confirm on GENUINELY UNSEEN seeds: if "
                         "0-19 have already been measured, --seed-start 20 --seeds 10 gives "
                         "a set that had no part in champion selection. Confirming on seeds "
                         "already measured is an expansion, not a replication, and must be "
                         "described as one.")
    ap.add_argument("--seed-list", default=None,
                    help="explicit comma-separated seeds, e.g. '20,21,22,23,24'. Overrides "
                         "--seeds/--seed-start. The exact list is recorded in the ledger "
                         "before any fit runs.")
    ap.add_argument("--against", default="seed", choices=["seed", "parent"],
                    help="'seed' compares the champion with the run's starting baseline "
                         "tree (the claim that matters for a submission); 'parent' "
                         "compares it with its immediate parent (attributes one step).")
    ap.add_argument("--baseline-tree", default=None,
                    help="explicit comparison tree (full sha256), overriding --against. "
                         "Use this to measure a run's champion against the ORIGINAL "
                         "official baseline when the run itself was seeded from something "
                         "else -- the total claim is against the baseline, not against a "
                         "seed that already contained a human-found improvement.")
    ap.add_argument("--isolation", default="seatbelt")
    ap.add_argument("--fit-seconds", type=float, default=1800.0)
    ap.add_argument("--out", default=None)
    return ap


def main() -> None:
    a = build_parser().parse_args()
    from core.scorer import request

    state = json.loads(Path(a.state).read_text())
    champ = state.get("champion")
    if not champ:
        sys.exit("that state records no champion")
    exps = state.get("experiments", [])
    if not exps:
        sys.exit("that state records no experiments")

    champ_tree = champ["tree"]
    if a.baseline_tree:
        base_tree = a.baseline_tree
        base_label = "explicit comparison tree"
        try:
            AR.verify(base_tree)
        except AR.ArtifactError as e:
            sys.exit(f"--baseline-tree does not verify: {e}")
    elif a.against == "seed":
        base_tree = exps[0]["tree"]
        base_label = "run seed"
    else:
        base_tree = next((e["parent"] for e in exps if e["tree"] == champ_tree), None)
        base_label = "immediate parent"
    if not base_tree:
        sys.exit("could not resolve the comparison tree")
    if base_tree == champ_tree:
        sys.exit("the champion IS the comparison tree; nothing to confirm")

    champ_files = AR.read(champ_tree)
    base_files = AR.read(base_tree)
    if a.seed_list:
        try:
            seeds = [int(x) for x in a.seed_list.replace(",", " ").split()]
        except ValueError:
            sys.exit(f"--seed-list must be integers; got {a.seed_list!r}")
        if not seeds:
            sys.exit("--seed-list is empty")
        if len(set(seeds)) != len(seeds):
            sys.exit(f"--seed-list repeats a seed: {a.seed_list!r}. A repeated seed is not "
                     f"an independent measurement.")
        if any(x < 0 for x in seeds):
            sys.exit("seeds must be non-negative")
        seeds = sorted(seeds)
    else:
        if a.seeds < 1:
            sys.exit("--seeds must be at least 1")
        seeds = list(range(a.seed_start, a.seed_start + a.seeds))

    from core.task import get as get_task
    spec = get_task("kuairand_pure")
    declared = (spec.baseline or {}).get("seed_std")

    print(f"champion   {champ_tree[:12]}  (iteration {champ.get('iteration')})")
    print(f"comparison {base_tree[:12]}  [{base_label}]")
    print(f"seeds      {seeds}   isolation {a.isolation}")
    print(f"           (source: {'--seed-list' if a.seed_list else f'--seed-start {a.seed_start} --seeds {a.seeds}'})")
    print(f"task's declared seed_std (cross-check only): {declared}\n")

    # Pre-register the protocol: the seeds are fixed before any of them is measured.
    r = request({"op": "record", "kind": "event", "body": {
        "what": "confirmation_protocol", "champion_tree": champ_tree,
        "comparison_tree": base_tree, "comparison": a.against, "seeds": seeds,
        "seed_selection": ("seed_list" if a.seed_list else "range"),
        "seed_start": (None if a.seed_list else a.seed_start),
        "seed_count": len(seeds),
        "method": "paired per-seed deltas, both trees fitted at each seed",
        "note": "recorded before any fit, so the seed set cannot be chosen after seeing "
                "which seeds were favourable"}})
    if not r.get("ok"):
        sys.exit(f"refusing to run unrecorded: {r.get('error')}")
    print(f"protocol recorded at ledger seq {r['entry']['seq']}\n")

    rows = []
    outroot = Path(a.state).resolve().parent / "confirm"
    for s in seeds:
        pair = {}
        for label, files in (("base", base_files), ("champ", champ_files)):
            tree_files = with_seed(files, s)
            d = outroot / f"s{s}_{label}"
            rep = CAND.run_candidate(tree_files, {}, run_dir=d, backend=a.isolation,
                                     max_seconds=a.fit_seconds)
            if not rep.get("ok"):
                sys.exit(f"seed {s} {label} failed to fit: "
                         f"{rep.get('error_type')}: {rep.get('error')}")
            sr = request({"op": "score_official", "scores": rep["dev_scores"],
                          "split": "valid", "config_hash": rep["tree"],
                          "isolation": a.isolation, "seed": s,
                          "kind": "confirmation"})
            if not sr.get("ok"):
                sys.exit(f"scorer refused seed {s} {label}: {sr.get('error')}")
            pair[label] = sr["entry"]["payload"]["metrics"]["primary"]
        delta = pair["champ"] - pair["base"]
        rows.append({"seed": s, "base": pair["base"], "champ": pair["champ"],
                     "delta": round(delta, 6)})
        print(f"  seed {s}: base {pair['base']:.6f}  champ {pair['champ']:.6f}  "
              f"delta {delta:+.6f}")

    deltas = [r_["delta"] for r_ in rows]
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 1 and sd > 0 else 0.0
    t = mean / se if se > 0 else None
    wins = sum(1 for d in deltas if d > 0)

    # A seed override that changes nothing produces identical replicates and an infinite
    # t. That means the candidate's own code ignores its seed, not that the effect is
    # certain -- so it is reported as a refusal to conclude.
    degenerate = sd == 0.0 and n > 1
    verdict = ("INCONCLUSIVE: every seed gave an identical result, so the candidate's code "
               "does not use config['seed']. Paired replication cannot say anything here."
               if degenerate else
               "CONFIRMED" if (wins == n and t is not None and t > 3) else
               "NOT CONFIRMED")

    print(f"\n  paired mean {mean:+.6f}   sd {sd:.6f}   se {se:.6f}")
    print(f"  t = {'n/a' if t is None else f'{t:.2f}'}   wins {wins}/{n}")
    print(f"\n  {verdict}")
    if declared and not degenerate:
        print(f"  (for scale: the task declares a seed_std of {declared}; a PAIRED test is "
              f"more sensitive than that, which is the point of pairing)")

    out = {"champion_tree": champ_tree, "comparison_tree": base_tree,
           "comparison": a.against, "protocol_seq": r["entry"]["seq"],
           "seeds": seeds,
           "seed_selection": ("seed_list" if a.seed_list else "range"),
           "seed_start": (None if a.seed_list else a.seed_start),
           "per_seed": rows, "n": n, "mean": round(mean, 6), "sd": round(sd, 6),
           "se": round(se, 6), "t": None if t is None else round(t, 2),
           "wins": wins, "verdict": verdict,
           "task_declared_seed_std": declared,
           "method": "paired per-seed deltas; both trees fitted at each matched seed"}
    request({"op": "record", "kind": "event",
             "body": {"what": "confirmation_result", **out}})
    dest = Path(a.out) if a.out else Path(a.state).resolve().parent / "confirmation.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
