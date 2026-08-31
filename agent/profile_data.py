"""Deterministic profile of a data directory. No LLM, no network, no guessing.

This exists so the LLM's job in agent/draft_task.py is the part it is actually good at
-- reading a goal and choosing among plausible roles -- while the part that must be
correct (what columns exist, what their types are, which are binary, which look like
dates) is measured from the file. A model asked to invent column names invents column
names; a model handed a real header does not have to.

The profile is a SETUP artifact. It is not, and must never become, something an agent
reads during a run: it necessarily contains statistics over every row including the
held-out ones, since at drafting time no split exists yet to hold anything out. Keeping
it out of the loop is the whole reason profiling and proposing are different modules.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

SAMPLE_ROWS = 4000
MAX_DISTINCT = 4000
DATE_RE = re.compile(r"^(19|20)\d{6}$")            # 20220408
ID_HINTS = ("user", "uid", "member", "customer", "account")
ITEM_HINTS = ("video", "item", "photo", "product", "content", "sku", "movie", "song")
TIME_HINTS = ("date", "time", "day", "ts", "timestamp", "week", "month")
# Names that are labels in name only. `is_...` columns in KuaiRand ARE outcomes, but an
# `is_active` user attribute is not -- so naming is a hint that gets ranked, never a
# decision that gets made.
LABEL_HINTS = ("click", "like", "view", "watch", "follow", "purchase", "convert",
               "buy", "label", "target", "y", "rating", "relevant", "engage")


def _typeof(vals: list[str]) -> str:
    seen_float = False
    for v in vals:
        if v == "":
            continue
        try:
            int(v)
        except ValueError:
            try:
                float(v)
                seen_float = True
            except ValueError:
                return "str"
    return "float" if seen_float else "int"


def profile_csv(path: Path, sample: int = SAMPLE_ROWS) -> dict:
    """Header, an exact row count, and per-column facts from a bounded sample.

    The row count is exact (a full pass) because a wrong count makes a split-size check
    meaningless; the per-column stats are from a sample because reading 1.1M rows x 30
    columns to draft a spec is not a good trade.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.reader(fh)
        try:
            header = next(rd)
        except StopIteration:
            return {"file": path.name, "error": "empty file", "columns": []}
        # A file whose header row is present but EMPTY (a lone "\r\n") is not caught by
        # StopIteration -- csv.reader yields [] rather than stopping. Without this it was
        # profiled as a perfectly good zero-column file, and the failure surfaced much
        # later as a draft with `label: None` rejected by the validator, which reports the
        # symptom rather than the cause. Flag it where the cause is.
        if not any(h.strip() for h in header):
            return {"file": path.name, "error": "no column names in the header row",
                    "columns": []}
        cols = {h: [] for h in header}
        n = 0
        for row in rd:
            n += 1
            if n <= sample and len(row) == len(header):
                for h, v in zip(header, row):
                    cols[h].append(v)
    out = {"file": path.name, "rows": n, "n_columns": len(header), "columns": []}
    for h in header:
        vals = cols[h]
        nonnull = [v for v in vals if v != ""]
        c = Counter(nonnull)
        t = _typeof(nonnull)
        info = {"name": h, "type": t, "sampled": len(vals),
                "null_rate": round(1 - len(nonnull) / len(vals), 4) if vals else 1.0,
                "distinct_in_sample": len(c),
                "examples": [k for k, _ in c.most_common(3)]}
        if t in ("int", "float") and nonnull:
            nums = [float(v) for v in nonnull]
            info["min"], info["max"] = min(nums), max(nums)
            info["mean"] = round(sum(nums) / len(nums), 6)
            uniq = set(nums)
            info["binary"] = uniq <= {0.0, 1.0}
            if info["binary"]:
                info["positive_rate"] = round(sum(nums) / len(nums), 6)
            info["looks_like_date"] = all(DATE_RE.match(v) for v in nonnull[:200])
            info["integral"] = all(float(v).is_integer() for v in nonnull[:500])
        else:
            info["binary"] = False
        # a column whose sample is entirely distinct is a candidate key, not a feature
        info["high_cardinality"] = len(c) > max(50, 0.5 * max(len(nonnull), 1))
        out["columns"].append(info)
    return out


def _score_label(c: dict) -> float:
    """Rank a column's plausibility as a binary target. Named separately so the ordering
    is inspectable rather than buried in a sort key."""
    if not c.get("binary"):
        return -1.0
    p = c.get("positive_rate", 0.0)
    if not (0.001 <= p <= 0.9):                 # a constant column is not a target
        return -1.0
    s = 1.0
    if any(k in c["name"].lower() for k in LABEL_HINTS):
        s += 2.0
    if c["name"].lower().startswith("is_"):
        s += 0.5
    # a rate near 0.5 is the most informative; extremes are harder and rarer targets
    s += 1.0 - min(abs(p - 0.4) / 0.4, 1.0)
    return s


def _score_time(c: dict) -> float:
    if c.get("type") not in ("int", "float"):
        return -1.0
    s = 0.0
    if c.get("looks_like_date"):
        s += 3.0
    if any(k in c["name"].lower() for k in TIME_HINTS):
        s += 2.0
    if c.get("distinct_in_sample", 0) > 3:
        s += 0.5
    return s if s > 0 else -1.0


def _score_group(c: dict, hints) -> float:
    s = 0.0
    if any(k in c["name"].lower() for k in hints):
        s += 3.0
    if c.get("high_cardinality"):
        s += 1.5
    if c.get("distinct_in_sample", 0) > 20:
        s += 0.5
    return s if s > 0 else -1.0


def candidates(prof: dict) -> dict:
    """Ranked role candidates for one file. Ranked, not chosen -- the choice is the
    LLM's (or the operator's), and it needs to see the runners-up to make one."""
    cols = prof.get("columns", [])

    def rank(fn, *a):
        out = [(round(fn(c, *a), 3), c["name"]) for c in cols]
        return [{"column": n, "score": s} for s, n in
                sorted((x for x in out if x[0] > 0), reverse=True)[:6]]

    return {"label": rank(_score_label), "time": rank(_score_time),
            "group": rank(_score_group, ID_HINTS), "item": rank(_score_group, ITEM_HINTS)}


def profile_dir(data_dir: Path, sample: int = SAMPLE_ROWS,
                max_files: int = 12) -> dict:
    d = Path(data_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    files = sorted(d.glob("*.csv"), key=lambda p: -p.stat().st_size)[:max_files]
    if not files:
        raise FileNotFoundError(f"no .csv files in {d}")
    out = {"data_dir": str(d), "n_csv": len(files), "files": []}
    for p in files:
        pr = profile_csv(p, sample=sample)
        pr["bytes"] = p.stat().st_size
        pr["candidates"] = candidates(pr)
        out["files"].append(pr)
    # the largest file with a plausible label is the natural interaction log
    logs = [f for f in out["files"] if f.get("candidates", {}).get("label")]
    out["suggested_log"] = logs[0]["file"] if logs else out["files"][0]["file"]
    return out


def compact(prof: dict, max_cols: int = 26) -> dict:
    """A smaller profile for a prompt. Drops per-column examples, which are the bulk of
    the bytes and the least useful part for choosing a role -- and, for a label column,
    are literally target values, which have no business in a prompt."""
    out = {"data_dir": prof["data_dir"], "suggested_log": prof.get("suggested_log"),
           "files": []}
    for f in prof["files"]:
        cols = f.get("columns", [])[:max_cols]
        out["files"].append({
            "file": f["file"], "rows": f.get("rows"), "n_columns": f.get("n_columns"),
            "truncated_columns": max(0, len(f.get("columns", [])) - len(cols)),
            "columns": [{k: v for k, v in c.items()
                         if k in ("name", "type", "binary", "positive_rate",
                                  "distinct_in_sample", "min", "max",
                                  "looks_like_date", "high_cardinality")}
                        for c in cols],
            "candidates": f.get("candidates", {})})
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="profile a data directory (no LLM)")
    ap.add_argument("data_dir")
    ap.add_argument("--sample", type=int, default=SAMPLE_ROWS)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pr = profile_dir(Path(a.data_dir), sample=a.sample)
    if a.compact:
        pr = compact(pr)
    txt = json.dumps(pr, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
