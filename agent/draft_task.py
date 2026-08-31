"""Stage 1: draft a TaskSpec from a data directory plus a one-line goal.

    python3 -m agent.draft_task KuaiRand-Pure/data \
        --goal "rank each user's logged impressions by whether they long-viewed" \
        --proposer gemini --model gemini-3.5-flash-lite

The division of labour is the point:

    agent/profile_data.py   MEASURES the directory -- headers, types, which columns are
                            binary, which look like dates. Facts, deterministically.
    this module (LLM)       CHOOSES roles from those facts, given the goal. Which binary
                            column is the target, which files are the log, where the
                            split boundaries go. Judgement, and only judgement.
    validate()              CHECKS the choice against the real files before anyone acts
                            on it -- columns exist, the label really is binary, every
                            split really has rows.
    a human                 CONFIRMS once, at setup. Not per iteration, so it costs
                            nothing on autonomy, which counts interventions DURING a run.

The LLM cannot invent a metric. It picks from METRICS below, which are the ones that are
implemented. An agent that chooses its own metric chooses an easy one, and this is the
same failure one step earlier -- so the drafter gets the same constraint the loop does.

WHAT CONFIRMATION DOES: it emits the portable JSON contract consumed by core.task.load,
core.tabular_task and the separate scorer.  The supported family is deliberately bounded:
CSV logged-impression binary ranking/classification with a declared integer time split.
It is runnable on differently named columns; it is not a claim of arbitrary AutoML.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from agent.profile_data import compact, profile_dir

ROOT = Path(__file__).resolve().parents[1]
DRAFT_DIR = ROOT / "runs" / "drafts"

# Implemented metrics only. The drafter picks; it does not define.
METRICS = {
    "within_user_ranking_gauc_ndcg5": {
        "desc": "GAUC and nDCG@5 over each group's own logged impressions; primary is "
                "their mean. Needs a group column. This is the KuaiRand-Pure task.",
        "needs_group": True, "needs_binary_label": True,
        "impl": "core.metrics.primary",
    },
    "global_auc": {
        "desc": "One ROC AUC over every row, ungrouped. For plain binary classification "
                "where ranking within a group is not the question.",
        "needs_group": False, "needs_binary_label": True,
        "impl": "core.official_evaluate.auc",
    },
}

SPLIT_KEYS = ("train", "valid", "test")

#: how much it is worth that the operator's goal names a column by name.
#: Calibrated against profile_data's own scale, where a name hint is 2-3
#: and a structural signal (every value matches a date pattern) is 3.
GOAL_BONUS = 2.0

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["slug", "title", "log_files", "label", "time_col", "splits",
                 "metric", "rationale"],
    "properties": {
        "slug": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,40}$"},
        "title": {"type": "string", "maxLength": 120},
        "log_files": {"type": "array", "items": {"type": "string"},
                      "minItems": 1, "maxItems": 6},
        "label": {"type": "string"},
        "group_by": {"type": ["string", "null"]},
        "item_col": {"type": ["string", "null"]},
        "time_col": {"type": "string"},
        "splits": {
            "type": "object", "additionalProperties": False,
            "required": list(SPLIT_KEYS),
            "properties": {k: {"type": "array", "items": {"type": "integer"},
                               "minItems": 2, "maxItems": 2} for k in SPLIT_KEYS}},
        "metric": {"type": "string", "enum": sorted(METRICS)},
        "feature_cols": {"type": "array", "items": {"type": "string"}, "maxItems": 24},
        "rationale": {"type": "string", "maxLength": 1200},
    },
}


# ------------------------------------------------------------------ heuristic draft
def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _named_in(col: str, goal_tokens: list[str]) -> bool:
    """Does the goal name this column? Matched on TOKEN PREFIXES, because a goal is
    written in prose and a column is written in snake_case: the goal "whether they
    long-viewed" has to reach `long_view`, and a plain substring test does not
    (`long_view` is not inside `long-viewed`). Stop words are dropped so `is_click` does
    not match on `is`."""
    stop = {"is", "the", "a", "an", "of", "id", "num", "cnt", "col"}
    ct = [t for t in _tokens(col) if t not in stop]
    if not ct:
        return False
    return all(any(g.startswith(t) or t.startswith(g) for g in goal_tokens) for t in ct)


def _slug(data_dir: Path) -> str:
    """A schema-valid slug from a directory. The obvious `data_dir.parent.name` is not
    enough: a data dir under a one-character parent (a macOS temp dir is
    /var/folders/../T/xxx) produced the slug "t", which the schema's
    ^[a-z][a-z0-9_]{1,40}$ rejects -- so the drafter emitted a draft that could never
    validate. Try the parent, then the directory itself, then give up explicitly."""
    for cand in (data_dir.parent.name, data_dir.name):
        v = re.sub(r"[^a-z0-9_]+", "_", cand.lower()).strip("_")[:41]
        if re.match(r"^[a-z][a-z0-9_]{1,40}$", v):
            return v
    return "drafted_task"


def _overlaps(a, b, min_frac: float = 0.5) -> bool:
    """Do two [lo, hi] spans share at least `min_frac` of the shorter one?"""
    if not a or not b:
        return False
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if lo > hi:
        return False
    shorter = min(a[1] - a[0], b[1] - b[0]) + 1
    return (hi - lo + 1) / max(shorter, 1) >= min_frac


def _density(data_dir: Path, spans: dict) -> dict:
    """Rows per unit of the time column, per file. Exact row counts, because this number
    is the evidence a human confirms a log choice on."""
    out = {}
    for n, (a, b) in spans.items():
        days = b - a + 1
        rec = {"span": [a, b], "units": days}
        try:
            with open(data_dir / n, "rb") as fh:
                rec["rows"] = max(sum(1 for _ in fh) - 1, 0)
        except OSError:
            rec["rows"] = None
        if rec["rows"] and days:
            rec["rows_per_unit"] = round(rec["rows"] / days, 1)
        out[n] = rec
    return out


def _density_warning(dens: dict, pairs: list) -> str:
    """Density over an IDENTICAL period is the one signal that can separate two collection
    policies from the data alone: a policy showing several times the impressions to the
    same users over the same days is not the same policy. It cannot decide which log the
    task is about -- only documentation can -- so this is reported, never acted on."""
    for n1, n2 in pairs:
        r1 = dens.get(n1, {}).get("rows_per_unit")
        r2 = dens.get(n2, {}).get("rows_per_unit")
        if not r1 or not r2:
            continue
        hi, lo = (n1, n2) if r1 >= r2 else (n2, n1)
        rhi, rlo = max(r1, r2), min(r1, r2)
        if rhi / rlo >= 2.0:
            return (f"over the same period {hi} carries {rhi:,.0f} rows per time unit "
                    f"against {lo}'s {rlo:,.0f} ({rhi / rlo:.1f}x). That is a DIFFERENT "
                    f"exposure policy, not more of the same one -- decide which log the "
                    f"task is about before confirming.")
    return ""


def heuristic_draft(prof: dict, goal: str) -> dict:
    """A deterministic draft, always available. It is the fallback when there is no LLM
    and no quota, and it is also the control: if the LLM's draft is no better than this,
    the LLM added nothing at this stage, and the honest thing is to say so.

    It reads the goal shallowly -- token-prefix matching against candidate column names.
    Beating that reading is what the LLM step is for.
    """
    gtok = _tokens(goal)
    files = prof["files"]
    data_dir = Path(prof["data_dir"])
    logs = [f for f in files
            if f.get("candidates", {}).get("label") and f.get("candidates", {}).get("time")]
    if not logs:
        logs = files[:1]

    # Same schema first: concatenating files with different headers misaligns columns.
    # Compare ORDERED headers, exactly as validate() does. Comparing name SETS here while
    # the validator compares lists meant two logs with the same columns in a different order
    # were grouped as "same schema" by the drafter and then rejected as "different header" by
    # the validator -- so the drafter emitted a draft that could never validate, and the CLI
    # exited with a problem the drafter itself had created. `profile_csv` preserves real
    # header order, so the distinction is live, not theoretical.
    hdr = [c["name"] for c in logs[0]["columns"]]
    same = [f for f in logs if [c["name"] for c in f["columns"]] == hdr]

    def choose(candidates, role, default=None):
        """Rank by the profile's score PLUS a bonus when the goal names the column.

        The bonus is deliberately not an override. Overriding produced a real bug: the
        goal mentions "video", so `video_id` won the TIME role outright over `date`,
        which the profile had scored 5.5 against 0.5 because every one of its values
        matches a date pattern. A goal mention is good evidence, but weaker than a
        structural signal -- so it is worth about as much as a name hint (2.0), enough to
        separate two equally plausible candidates and not enough to beat a measured one.
        """
        opts = candidates.get(role) or []
        if not opts:
            return default
        best = max(opts, key=lambda o: (o["score"] + (GOAL_BONUS if
                                        _named_in(o["column"], gtok) else 0.0),
                                        -len(o["column"])))
        return best["column"]

    # The time column is a property of the shared schema, so it can be settled before the
    # file set is, and it is needed to settle the file set.
    time_col = choose(same[0]["candidates"], "time")

    # Among same-schema logs, keep a set whose periods do NOT overlap. Two logs over one
    # period are usually two collection policies rather than a longer timeline, and
    # concatenating those is a judgement no heuristic should make silently -- KuaiRand's
    # log_random is the organisers' unbiased validation log and shares log_standard's
    # dates exactly. Take the largest file, then only files that extend the timeline.
    spans = {}
    if time_col and len(same) > 1:
        for f in same:
            b = _time_bounds(data_dir, [f["file"]], time_col)
            if b:
                spans[f["file"]] = (b[0], b[-1])
        kept, cover, overlapped, untimed = [], [], [], []
        # Deterministic order: longest span first, then filename. Reproducible, which is
        # the most this can honestly claim -- see the note built below.
        for f in sorted(same, key=lambda x: (-(spans.get(x["file"], (0, 0))[1]
                                               - spans.get(x["file"], (0, 0))[0]),
                                             x["file"])):
            sp = spans.get(f["file"])
            if sp is None:
                untimed.append(f["file"])
                continue
            if any(not (sp[1] < a or sp[0] > b) for a, b in cover):
                overlapped.append(f["file"])
                continue
            kept.append(f)
            cover.append(sp)
        if kept:
            same = sorted(kept, key=lambda x: spans[x["file"]][0])
        excluded = {"overlapping": overlapped, "no_time_column": untimed}

    cand = same[0]["candidates"]
    label = choose(cand, "label")
    group = choose(cand, "group")
    item = choose(cand, "item")
    excluded = locals().get("excluded") or {"overlapping": [], "no_time_column": []}
    bounds = _time_bounds(data_dir, [f["file"] for f in same], time_col)
    note = ("HEURISTIC DRAFT (no LLM). Roles chosen by ranked name/type heuristics, "
            "preferring any column the goal names. The split is a 70/15/15 cut of the "
            "observed time range -- a convention, not a reading of the problem.")
    if excluded["no_time_column"]:
        note += (" Not treated as logs (no usable time column): "
                 + ", ".join(sorted(excluded["no_time_column"])) + ".")
    if excluded["overlapping"]:
        note += (" Excluded because their periods overlap a file already chosen: "
                 + ", ".join(sorted(excluded["overlapping"]))
                 + ". THIS CHOICE IS NOT DERIVABLE FROM THE DATA -- when two logs cover "
                   "one period, only documentation says which is primary. The selection "
                   "is deterministic (longest span, then filename), so it is reproducible, "
                   "not correct.")
        spans_all = dict(spans)
        for f_ in excluded["overlapping"]:
            spans_all.setdefault(f_, None)
        dw = _density_warning(_density(data_dir, {k: v for k, v in spans_all.items() if v}),
                              [(a_, b_) for a_ in spans_all for b_ in spans_all
                               if a_ < b_ and _overlaps(spans_all.get(a_), spans_all.get(b_))])
        if dw:
            note += " Evidence: " + dw
    return {
        "slug": _slug(data_dir),
        "title": f"{data_dir.parent.name} \u2014 drafted from a data profile",
        "log_files": [f["file"] for f in same],
        "label": label, "group_by": group, "item_col": item, "time_col": time_col,
        "splits": _three_way(bounds),
        "metric": ("within_user_ranking_gauc_ndcg5" if group else "global_auc"),
        "feature_cols": [c for c in (group, item) if c],
        "rationale": note,
    }


def _time_bounds(data_dir: Path, files: list[str], col: str) -> list[int]:
    """Every distinct value of the time column, sorted. A full pass, because a split
    boundary guessed from a sample can put rows on the wrong side of it."""
    vals = set()
    for f in files:
        p = data_dir / f
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.reader(fh)
            hdr = next(rd, None)
            if not hdr or col not in hdr:
                continue
            i = hdr.index(col)
            for row in rd:
                if len(row) > i and row[i]:
                    try:
                        vals.add(int(row[i]))
                    except ValueError:
                        pass
    return sorted(vals)


def _three_way(bounds: list[int]) -> dict:
    """70/15/15 over DISTINCT time values, so a split boundary never falls inside a day."""
    if len(bounds) < 3:
        lo = bounds[0] if bounds else 0
        hi = bounds[-1] if bounds else 0
        return {"train": [lo, lo], "valid": [lo, hi], "test": [hi, hi]}
    n = len(bounds)
    a, b = int(n * 0.70), int(n * 0.85)
    a, b = max(a, 1), max(b, 2)
    b = min(b, n - 1)
    a = min(a, b - 1) if b - 1 >= 1 else 1
    return {"train": [bounds[0], bounds[a - 1]],
            "valid": [bounds[a], bounds[b - 1]],
            "test": [bounds[b], bounds[-1]]}


# ------------------------------------------------------------------ LLM draft
def _prompt(prof_c: dict, goal: str, heur: dict) -> tuple[str, str]:
    metrics = "\n".join(f"  {k}\n      {v['desc']}" for k, v in sorted(METRICS.items()))
    system = f"""You are drafting the SETUP for an autonomous ML research loop: which
column is the target, which files are the interaction log, where the time-based split
boundaries go, and which implemented metric matches the goal.

You are NOT choosing a model, features to engineer, or a loss. Those belong to the agent
that will run afterwards. You are also not inventing a metric -- choose one of:
{metrics}

Rules that will make your draft REJECTED:
  * naming a column or file that is not in the profile
  * a label column the profile does not mark as binary
  * split ranges that overlap, are out of order, or leave a split with no rows
  * a test range that is not the LATEST period -- the held-out split must be the future,
    or the task is not a forecast and the score does not mean what it appears to
  * choosing a grouped metric with no group column

Prefer the label the GOAL describes over the one that merely looks most like a target:
several binary columns will be plausible and only one answers the stated question.
Include EVERY log file that shares the same schema and extends the timeline; taking only
the largest silently discards part of the period."""

    user = f"""GOAL (one line, from the operator):
  {goal}

DATA PROFILE (measured from the files -- treat as ground truth):
{json.dumps(prof_c, indent=1)}

A DETERMINISTIC HEURISTIC DRAFT, for reference. It ranks by column name and type and
splits the time range 70/15/15. Improve on it where the goal or the profile says it is
wrong, and say why in `rationale`:
{json.dumps(heur, indent=1)}

Return one JSON object matching the schema."""
    return system, user


def llm_draft(prof: dict, goal: str, kind: str, model: str | None,
              api_key: str = "") -> tuple[dict, dict]:
    from agent.backends import BackendError, make_backend
    heur = heuristic_draft(prof, goal)
    be = make_backend(kind, api_key=api_key, model=model)
    system, user = _prompt(compact(prof), goal, heur)
    try:
        resp, usage = be.complete_json(system, user, SCHEMA)
        text = be.text_of(resp)
    except BackendError as e:
        raise RuntimeError(f"{kind}:{model}: {e}") from None
    try:
        return json.loads(text), usage
    except json.JSONDecodeError as e:
        raise RuntimeError(f"drafter returned non-JSON despite a schema: {e}; "
                           f"first 200: {text[:200]!r}") from None


# ------------------------------------------------------------------ validation
def _header(p: Path) -> list[str]:
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        return next(csv.reader(fh), []) or []


def validate(draft: dict, data_dir: Path) -> dict:
    """Check a draft against the REAL files. Returns problems and the evidence gathered.

    This reads the data rather than the profile on purpose: the profile is a sample, and
    "every split has rows" is exactly the claim a sample cannot support.
    """
    probs, warns, ev = [], [], {}
    d = Path(data_dir)

    for k in SCHEMA["required"]:
        if k not in draft:
            probs.append(f"missing required field {k!r}")
    if probs:
        return {"ok": False, "problems": probs, "warnings": warns, "evidence": ev}

    files = draft["log_files"]
    if not isinstance(files, list) or not files:
        return {"ok": False, "problems": ["log_files must be a non-empty list"],
                "warnings": warns, "evidence": ev}
    paths = []
    for f in files:
        if "/" in f or "\\" in f or f.startswith("."):
            probs.append(f"log_files entry {f!r} must be a bare filename in the data dir")
            continue
        p = d / f
        if not p.is_file():
            probs.append(f"log file {f!r} does not exist in {d}")
        else:
            paths.append(p)
    if not paths:
        return {"ok": False, "problems": probs or ["no usable log files"],
                "warnings": warns, "evidence": ev}

    # Two log files covering the SAME period are almost never a continuous timeline;
    # they are usually two collection policies over one period, and concatenating them
    # conflates them. KuaiRand is exactly this: log_random and log_standard both cover
    # 4/22-5/08, and the random one is the organisers' UNBIASED VALIDATION log -- folding
    # it into training is a rules judgement, not a data-loading detail. This cannot be
    # detected from a filename in general, but overlapping ranges can be, so that is what
    # is checked.
    if len(paths) > 1 and draft.get("time_col"):
        spans = {}
        for p_ in paths:
            b = _time_bounds(d, [p_.name], draft["time_col"])
            if b:
                spans[p_.name] = (b[0], b[-1])
        names = sorted(spans)
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                a1, b1 = spans[n1]
                a2, b2 = spans[n2]
                lo, hi = max(a1, a2), min(b1, b2)
                if lo > hi:
                    continue
                shorter = min(b1 - a1, b2 - a2) + 1
                frac = (hi - lo + 1) / max(shorter, 1)
                if frac >= 0.5:
                    warns.append(
                        f"{n1} and {n2} both cover {lo}..{hi} ({frac:.0%} of the shorter "
                        f"file's span). Two logs over one period are usually two "
                        f"collection policies, not a continuous timeline -- check whether "
                        f"one is a designated unbiased/holdout log before training on "
                        f"both.")
        ev["log_density"] = _density(d, spans)
        pairs = [(n1, n2) for i, n1 in enumerate(names) for n2 in names[i + 1:]
                 if _overlaps(spans.get(n1), spans.get(n2))]
        w = _density_warning(ev["log_density"], pairs)
        if w:
            warns.append(w)
        ev["file_time_spans"] = spans

    heads = {p.name: _header(p) for p in paths}
    first = heads[paths[0].name]
    for n, h in heads.items():
        if h != first:
            probs.append(f"{n} has a different header from {paths[0].name}; "
                         f"concatenating them would misalign columns")
    hdr = set(first)
    ev["columns"] = len(first)

    for field in ("label", "time_col"):
        if draft[field] not in hdr:
            probs.append(f"{field} {draft[field]!r} is not a column of the log")
    for field in ("group_by", "item_col"):
        v = draft.get(field)
        if v and v not in hdr:
            probs.append(f"{field} {v!r} is not a column of the log")
    for c in draft.get("feature_cols") or []:
        if c not in hdr:
            probs.append(f"feature_cols entry {c!r} is not a column of the log")

    m = draft.get("metric")
    if m not in METRICS:
        probs.append(f"metric {m!r} is not implemented; choose from {sorted(METRICS)}")
    elif METRICS[m]["needs_group"] and not draft.get("group_by"):
        probs.append(f"metric {m!r} is grouped but no group_by column was chosen")

    sp = draft.get("splits") or {}
    if set(sp) != set(SPLIT_KEYS):
        probs.append(f"splits must have exactly {list(SPLIT_KEYS)}, got {sorted(sp)}")
    else:
        rng = {}
        for k in SPLIT_KEYS:
            v = sp[k]
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                probs.append(f"split {k} must be [lo, hi]")
                continue
            lo, hi = v
            if not all(isinstance(x, int) for x in (lo, hi)):
                probs.append(f"split {k} bounds must be integers, got {v!r}")
            elif lo > hi:
                probs.append(f"split {k} is inverted: {lo} > {hi}")
            else:
                rng[k] = (lo, hi)
        order = [rng[k] for k in SPLIT_KEYS if k in rng]
        for (a, b), (c, e2) in zip(order, order[1:]):
            if c <= b:
                probs.append(f"splits overlap or touch: [{a},{b}] then [{c},{e2}]")
        if len(rng) == 3 and rng["test"][0] <= rng["valid"][1]:
            probs.append("the test range must start after valid ends -- the held-out "
                         "split has to be the future")

        if not probs and draft["time_col"] in hdr and draft["label"] in hdr:
            ev.update(_count(paths, draft, rng))
            for k in SPLIT_KEYS:
                if ev["rows"].get(k, 0) == 0:
                    probs.append(f"split {k} matches ZERO rows in the real data "
                                 f"({rng[k][0]}..{rng[k][1]})")
            if ev.get("label_values") and not ev["label_values"] <= {0.0, 1.0}:
                probs.append(f"label {draft['label']!r} is not binary in the data; "
                             f"saw {sorted(ev['label_values'])[:6]}")
            pr = ev.get("positive_rate", {})
            for k in SPLIT_KEYS:
                if k in pr and pr[k] in (0.0, 1.0):
                    probs.append(f"label is constant in split {k} (rate {pr[k]}) -- "
                                 f"nothing to learn or to score")
            unmatched = ev.get("rows_outside_every_split", 0)
            if unmatched:
                warns.append(f"{unmatched:,} rows fall outside every split range and "
                             f"would be silently dropped")
            bad_t = ev.get("rows_unparseable_time", 0)
            if bad_t:
                frac = bad_t / max(ev.get("rows_total", 0) + bad_t, 1)
                msg = (f"{bad_t:,} rows have a {draft['time_col']!r} value that is not an "
                       f"integer and are dropped from every split")
                # A handful is a data blemish; a large share means the column is not the
                # time column at all, and a spec built on it would train on a fraction of
                # the data while reporting the split sizes as if nothing were missing.
                if frac >= 0.05:
                    probs.append(msg + f" -- that is {frac:.1%} of the file, so "
                                       f"{draft['time_col']!r} is probably not a usable "
                                       f"time column")
                else:
                    warns.append(msg)
            if ev.get("rows_short"):
                warns.append(f"{ev['rows_short']:,} rows have fewer columns than the header "
                             f"and were skipped entirely")
            if ev["rows"].get("train", 0) < ev["rows"].get("test", 0):
                warns.append("train has fewer rows than the held-out split, which is "
                             "unusual -- check the boundaries are the intended way round")

            # DOES group_by ACTUALLY GROUP? Only its existence was checked. Measured on the
            # real metric: with a per-row-unique group column GAUC is exactly 0.5000 (every
            # group has npos of 0 or npos == len, so all are excluded from the weighted
            # average) and nDCG@5 collapses to the label rate -- the primary score stops
            # measuring ranking at all, while looking like a perfectly ordinary number.
            # A per-row-unique id column is an easy thing to pick by mistake: profile_data
            # ranks HIGH-CARDINALITY columns highest for the group role.
            if m in METRICS and METRICS[m]["needs_group"] and draft.get("group_by"):
                for k in SPLIT_KEYS:
                    n_rows = ev["rows"].get(k, 0)
                    n_grp = (ev.get("groups") or {}).get(k, 0)
                    if not n_rows or not n_grp:
                        continue
                    # `groups` is capped at 200k distinct values per split, so a very large
                    # split can under-report. That only ever makes this check LENIENT, and a
                    # column unique per row in one split is unique in the others too, so a
                    # smaller split still catches it.
                    if n_grp >= 200_000:
                        continue
                    per = n_rows / n_grp
                    if per < 1.5:
                        probs.append(
                            f"group_by {draft['group_by']!r} does not group split {k}: "
                            f"{n_grp:,} distinct values over {n_rows:,} rows "
                            f"({per:.2f} rows per group). Metric {m!r} is grouped, and with "
                            f"one row per group GAUC is the no-information 0.5 and nDCG "
                            f"degenerates to the label rate -- the score would stop "
                            f"measuring ranking.")
                    elif per < 3.0:
                        warns.append(
                            f"group_by {draft['group_by']!r} gives only {per:.2f} rows per "
                            f"group in split {k}; a grouped metric has little to rank "
                            f"inside each group.")

    dev, held = draft.get("dev_split", "valid"), draft.get("held_out_split", "test")
    if dev == held:
        probs.append("dev_split and held_out_split must differ")
    if dev not in SPLIT_KEYS or held not in SPLIT_KEYS:
        probs.append(f"dev_split/held_out_split must be one of {list(SPLIT_KEYS)}")

    return {"ok": not probs, "problems": probs, "warnings": warns, "evidence": ev}


def _count(paths, draft, rng) -> dict:
    """One pass: rows per split, label values seen, positive rate per split."""
    rows = {k: 0 for k in SPLIT_KEYS}
    pos = {k: 0 for k in SPLIT_KEYS}
    groups = {k: set() for k in SPLIT_KEYS}
    seen_vals, outside, total = set(), 0, 0
    unparseable = short = 0
    for p in paths:
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            ti, li = hdr.index(draft["time_col"]), hdr.index(draft["label"])
            gi = hdr.index(draft["group_by"]) if draft.get("group_by") in hdr else None
            for row in rd:
                if len(row) <= max(ti, li):
                    short += 1
                    continue
                total += 1
                try:
                    t = int(row[ti])
                except ValueError:
                    # A row whose time value will not parse is dropped from EVERY split, so
                    # it must be counted. Previously it was silently `continue`d without
                    # touching `outside`, which left rows_outside_every_split at 0 -- so the
                    # "rows fall outside every split range and would be silently dropped"
                    # warning, whose entire job is to notice this, could never fire for it.
                    # Reachable with no LLM: `_typeof` skips empty strings when typing, so a
                    # `date` column with blanks is still typed int, scores 5.5, and the
                    # deterministic heuristic picks it.
                    unparseable += 1
                    continue
                hit = None
                for k in SPLIT_KEYS:
                    lo, hi = rng[k]
                    if lo <= t <= hi:
                        hit = k
                        break
                if hit is None:
                    outside += 1
                    continue
                rows[hit] += 1
                v = row[li]
                if len(seen_vals) < 12 and v != "":
                    try:
                        seen_vals.add(float(v))
                    except ValueError:
                        seen_vals.add(v)
                # Count positives NUMERICALLY, matching the float() normalisation two
                # lines above and profile_data's own binary/positive_rate logic. The
                # string test `v not in ("", "0")` counted every other spelling of zero
                # -- "0.0", "00", " 0", "0.000" -- as a POSITIVE, so a column that is zero
                # on every row reported positive_rate 1.0 and sailed past the
                # constant-label guard below, which only fires on 0.0 or 1.0 exactly...
                # and 1.0 is precisely what the bug produced. A float-spelled label column
                # is not exotic: profile_data detects binary-ness via float(), so the
                # deterministic heuristic will happily pick one.
                if v != "":
                    try:
                        if float(v) != 0.0:
                            pos[hit] += 1
                    except ValueError:
                        # a non-numeric label: any non-empty value is a positive, which is
                        # the old behaviour and the only sane reading of e.g. "yes"/""
                        pos[hit] += 1
                if gi is not None and len(groups[hit]) < 200_000:
                    groups[hit].add(row[gi])
    return {"rows": rows, "rows_total": total, "rows_outside_every_split": outside,
            # Counted separately so a reader can tell WHY a row vanished. Lumping them into
            # `outside` would say "wrong date range" about a row whose date is unreadable.
            "rows_unparseable_time": unparseable, "rows_short": short,
            "label_values": seen_vals,
            "positive_rate": {k: (round(pos[k] / rows[k], 5) if rows[k] else None)
                              for k in SPLIT_KEYS},
            # Distinct group values per split, so a caller can tell whether `group_by`
            # actually GROUPS anything -- a per-row-unique column makes GAUC collapse to a
            # constant 0.5, and nothing compared these counts to `rows` before.
            "groups": {k: len(groups[k]) for k in SPLIT_KEYS}}


# ------------------------------------------------------------------ render + cli
def render(draft: dict, v: dict) -> str:
    L = []
    L.append("=" * 78)
    L.append(f"DRAFT SPEC  {draft.get('slug','?')}")
    L.append("=" * 78)
    for k in ("title", "label", "group_by", "item_col", "time_col", "metric"):
        L.append(f"  {k:12s} {draft.get(k)}")
    L.append(f"  {'log_files':12s} {', '.join(draft.get('log_files') or [])}")
    for k in SPLIT_KEYS:
        b = (draft.get("splits") or {}).get(k)
        n = v["evidence"].get("rows", {}).get(k)
        g = v["evidence"].get("groups", {}).get(k)
        pr = (v["evidence"].get("positive_rate") or {}).get(k)
        L.append(f"  {('split ' + k):12s} {b}"
                 + (f"   rows {n:,}" if n is not None else "")
                 + (f"   groups {g:,}" if g else "")
                 + (f"   label rate {pr}" if pr is not None else ""))
    L.append(f"  {'features':12s} {', '.join(draft.get('feature_cols') or []) or '(none)'}")
    L.append("")
    L.append("  rationale:")
    for line in (draft.get("rationale") or "").split("\n"):
        L.append(f"    {line}")
    L.append("")
    if v["problems"]:
        L.append("  REJECTED:")
        for p in v["problems"]:
            L.append(f"    - {p}")
    else:
        L.append("  VALIDATED against the real files: every column exists, the label is "
                 "binary,")
        L.append("  and every split matches a non-zero number of rows.")
    for w in v["warnings"]:
        L.append(f"  warning: {w}")
    L.append("=" * 78)
    return "\n".join(L)


def portable_config(draft: dict, data_dir: Path, validation: dict, *, goal: str,
                    drafted_by: str, epsilon: float = 0.002, window: int = 3,
                    baseline_valid: float | None = None) -> dict:
    """Validated draft -> runnable, scorer-readable portable TaskSpec JSON."""
    if not validation.get("ok"):
        raise ValueError("cannot confirm a draft that did not validate")
    if epsilon <= 0 or window < 1:
        raise ValueError("epsilon must be positive and convergence window at least 1")
    features = list(dict.fromkeys(
        (draft.get("feature_cols") or [])
        + [x for x in (draft.get("group_by"), draft.get("item_col")) if x]))
    if not features:
        raise ValueError("a runnable task needs at least one declared feature column")
    cfg = {
        "slug": draft["slug"], "title": draft["title"],
        "data_dir": str(Path(data_dir).resolve()),
        "files": list(draft["log_files"]), "label": draft["label"],
        "time_col": draft["time_col"], "feature_cols": features,
        "group_by": draft.get("group_by"), "item_col": draft.get("item_col"),
        "outcome_columns": [draft["label"]],
        "splits": draft["splits"], "metric": draft["metric"],
        "convergence": {"epsilon": float(epsilon), "N": int(window),
                        "source": "human-confirmed task setup"},
        "baseline": ({"valid": float(baseline_valid)}
                     if baseline_valid is not None else {}),
        "setup": {"goal": goal, "drafted_by": drafted_by,
                  "rationale": draft.get("rationale"),
                  "validation_warnings": list(validation.get("warnings") or []),
                  "validated_rows": (validation.get("evidence") or {}).get("rows")},
    }
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="draft a TaskSpec from a data dir + a goal")
    ap.add_argument("data_dir")
    ap.add_argument("--goal", required=True, help="one line: what should be predicted")
    ap.add_argument("--proposer", default="heuristic",
                    choices=["heuristic", "gemini", "ollama", "groq"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--confirm", action="store_true",
                    help="write the spec. Without this, nothing is written -- a draft is "
                         "a proposal and a human confirms it exactly once.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epsilon", type=float, default=0.002,
                    help="confirmed convergence threshold for this task")
    ap.add_argument("--window", type=int, default=3,
                    help="confirmed convergence window N for this task")
    ap.add_argument("--baseline-valid", type=float, default=None,
                    help="optional declared validation baseline for signed deltas")
    a = ap.parse_args()

    d = Path(a.data_dir)
    print(f"[draft] profiling {d} ...")
    prof = profile_dir(d, sample=a.sample)
    print(f"[draft] {prof['n_csv']} csv files, "
          f"{sum(f.get('rows', 0) for f in prof['files']):,} rows total")

    usage = {}
    if a.proposer == "heuristic":
        draft = heuristic_draft(prof, a.goal)
        src = "heuristic"
    else:
        import os
        key = (os.environ.get("TRACK2_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or "")
        try:
            draft, usage = llm_draft(prof, a.goal, a.proposer, a.model, api_key=key)
            src = f"{a.proposer}:{a.model}"
        except RuntimeError as e:
            print(f"[draft] LLM unavailable ({e})")
            print("[draft] falling back to the deterministic heuristic draft")
            draft = heuristic_draft(prof, a.goal)
            src = "heuristic (llm unavailable)"

    v = validate(draft, d)
    print()
    print(render(draft, v))
    print(f"  drafted by: {src}"
          + (f"   tokens in/out {usage.get('input_tokens')}/{usage.get('output_tokens')}"
             if usage else ""))

    if not v["ok"]:
        raise SystemExit("draft rejected; nothing written")
    if not a.confirm:
        print("\n  Nothing written. Re-run with --confirm to accept this spec.")
        return
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else DRAFT_DIR / f"{draft['slug']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = portable_config(draft, d, v, goal=a.goal, drafted_by=src,
                          epsilon=a.epsilon, window=a.window,
                          baseline_valid=a.baseline_valid)
    out.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Refuse to print "confirmed" until the exact bytes just written instantiate the same
    # loader/scorer contract the research driver will use.
    from core.task import load as load_task
    confirmed = load_task(out)
    print(f"\n  confirmed -> {out}")
    print(f"  RUNNABLE: {confirmed.slug} loaded with "
          f"{confirmed.split.expect_rows}. Start scorer and driver with --task-spec.")


if __name__ == "__main__":
    main()
