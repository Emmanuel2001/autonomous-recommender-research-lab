"""Paired seed statistics derived from the SIGNED LEDGER, never from prose.

WHY THIS MODULE EXISTS
----------------------
The 20-seed figure quoted in the deliverables must be reproducible from the chain, and the
disclosure around it has to be exact:

  * seeds 0-4 were measured FIRST, as a 5-seed pre-registered confirmation;
  * the analysis was later EXPANDED to seeds 0-19;
  * seed 0 is also the seed the champion was selected on during the run.

So the 0-19 result is an **expanded paired robustness analysis**, not an independent
pre-registered 20-seed replication, and this module reports the newly observed subset
(seeds 5-19) separately so a reader can see the effect on seeds that played no part in
selection. Every number comes from `confirmation` score entries in the ledger.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def confirmation_scores(ledger: Path) -> list[dict]:
    """Every signed score entry of kind `confirmation`, with its seed and config hash."""
    out = []
    for line in Path(ledger).read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue                      # a truncated tail is expected after a crash
        pl = e.get("payload", {})
        if pl.get("kind") != "confirmation" or "metrics" not in pl:
            continue
        out.append({"seq": e.get("seq"), "tree": pl.get("config_hash"),
                    "seed": pl.get("seed"),
                    "primary": pl["metrics"]["primary"]})
    return out


def confirmation_results(ledger: Path) -> list[dict]:
    """Every `confirmation_result` event body, which carries the paired per-seed rows.

    Why not the raw score entries: a confirmation fits each tree at each seed, and the seed
    lives in `config.json`, which is part of the content hash -- so seed 3 of the champion is
    a DIFFERENT tree from the champion. The champion's own hash appears only at its own seed.
    The per-seed pairing is therefore recorded in the result event, and `cross_check()` below
    re-derives it against the raw signed scores.
    """
    out = []
    for line in Path(ledger).read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        pl = e.get("payload", {})
        if pl.get("kind") != "event":
            continue
        b = pl.get("body") or {}
        if b.get("what") == "confirmation_result":
            out.append({"seq": e.get("seq"), **b})
    return out


def _stats(per: list[dict]) -> dict:
    deltas = [p["delta"] for p in per]
    n = len(deltas)
    if n == 0:
        return {"n": 0, "seeds": [], "per_seed": []}
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 1 and sd > 0 else 0.0
    return {"n": n, "seeds": [p["seed"] for p in per], "per_seed": per,
            "mean": round(mean, 6), "sd": round(sd, 6), "se": round(se, 6),
            "t": (round(mean / se, 2) if se > 0 else None),
            "wins": sum(1 for d in deltas if d > 0),
            "min": round(min(deltas), 6), "max": round(max(deltas), 6)}


def paired(ledger: Path, champion: str, baseline: str,
           seeds: range | list | None = None) -> dict:
    """Paired per-seed deltas for one (champion, baseline) pair, derived from the chain.

    Uses the LARGEST confirmation recorded for that pair, then filters to `seeds`. Statistics
    are recomputed here from the per-seed rows rather than copied from the event's own
    summary, so a filtered view is arithmetic on ledger data and not a quoted constant.
    """
    cands = [c for c in confirmation_results(ledger)
             if c.get("champion_tree") == champion
             and c.get("comparison_tree") == baseline]
    if not cands:
        return {"n": 0, "seeds": [], "per_seed": []}
    best = max(cands, key=lambda c: len(c.get("per_seed") or []))
    want = None if seeds is None else set(seeds)
    per = [dict(r) for r in (best.get("per_seed") or [])
           if want is None or r.get("seed") in want]
    out = _stats(sorted(per, key=lambda r: r["seed"]))
    out["source_seq"] = best.get("seq")
    out["protocol_seq"] = best.get("protocol_seq")
    return out


def cross_check(ledger: Path, champion: str, baseline: str) -> dict:
    """Do the event's per-seed values agree with the raw SIGNED score entries?

    The result event is a convenience record; the scores are the signed measurements. Every
    base/champ value in the event must appear as a signed `confirmation` score at that seed,
    or the summary and the evidence disagree.
    """
    raw = confirmation_scores(ledger)
    have = {(r["seed"], round(r["primary"], 6)) for r in raw if r["seed"] is not None}
    full = paired(ledger, champion, baseline)
    missing = []
    for r in full.get("per_seed", []):
        for side in ("base", "champ"):
            if (r["seed"], round(r[side], 6)) not in have:
                missing.append({"seed": r["seed"], "side": side, "value": r[side]})
    return {"checked": 2 * full.get("n", 0), "unmatched": missing,
            "ok": not missing}


def champion_report(ledger: Path, champion: str, baseline: str) -> dict:
    """The three views the deliverables must state, all ledger-derived.

    `original_5` is the pre-registered confirmation. `expanded_20` is the later expansion.
    `newly_observed_5_19` is the subset that had no part in champion selection -- the one a
    sceptical reader should look at first.
    """
    full = paired(ledger, champion, baseline)
    return {
        "champion": champion, "baseline": baseline, "ledger": Path(ledger).name,
        "original_5": paired(ledger, champion, baseline, range(0, 5)),
        "expanded_20": paired(ledger, champion, baseline, range(0, 20)),
        "newly_observed_5_19": paired(ledger, champion, baseline, range(5, 20)),
        "all_seeds_present": full["seeds"],
        "disclosure": (
            "Seeds 0-4 were measured first as a pre-registered 5-seed confirmation; the "
            "analysis was later expanded to seeds 0-19. Seed 0 is also the seed the "
            "champion was selected on during the run. The 0-19 figure is therefore an "
            "EXPANDED paired robustness analysis, not an independent pre-registered 20-seed "
            "replication. Seeds 5-19 are reported separately as the subset that played no "
            "part in selection."),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", default=str(ROOT / "runs/ledgers/research_trials.jsonl"))
    ap.add_argument("--champion", required=True)
    ap.add_argument("--baseline", required=True)
    a = ap.parse_args()
    print(json.dumps(champion_report(Path(a.ledger), a.champion, a.baseline), indent=1))


if __name__ == "__main__":
    main()
