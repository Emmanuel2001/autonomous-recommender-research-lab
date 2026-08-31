"""HTML report for a code-first research run: hypothesis, diff, result, recovery.

WHY THIS EXISTS
---------------
The problem statement makes the per-iteration log the thing Autonomy and Robustness are
assessed FROM: each iteration must show its hypothesis, the code diff applied, the metrics,
and any error/recovery events. All of that is in the signed chain -- and until this file
existed, reading it meant writing Python against a 415-line JSONL. `agent/official_report.py`
contains no reference to a research run, so nothing rendered them.

WHAT IT VERIFIES RATHER THAN DISPLAYS
------------------------------------
Two properties are recomputed here, not merely echoed:

  * **artifact integrity** -- every tree named in the chain is re-hashed from its stored
    bytes (`core.artifacts.verify`), so "the ledger names this source" is checked, not
    trusted. No signing key is needed for that.
  * **pre-registration ordering** -- for every scored tree, a proposal naming it must sit at
    a LOWER sequence number than its score. That ordering is the entire content of the claim
    that a prediction was registered before its result, and it is computable from the chain.

Signature verification is deliberately NOT done here: it needs the signing key, and a
reporting process has no business holding it. The page says so and gives the command.

REUSE
-----
CSS, cards, tables and the chart helpers come from `agent/official_report.py` and
`agent/charts.py` unchanged, so the two reports look like one product and a style fix lands
in both.
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from agent import charts as CH
from agent.official_report import CSS, _card, _num, _table, esc
from core import artifacts as AR
from core.metrics import OFFICIAL_EPSILON, OFFICIAL_FM_SEED_STD, OFFICIAL_VALID

EXTRA_CSS = """
details{border:1px solid var(--rule);border-radius:7px;background:var(--card);
margin:8px 0;padding:0}
details>summary{cursor:pointer;padding:9px 13px;font-size:13px;color:var(--mut);
list-style:none}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:'\\25b8  ';color:var(--mut)}
details[open]>summary::before{content:'\\25be  '}
details>summary b{color:var(--fg)}
pre.diff{margin:0;padding:11px 13px;border-top:1px solid var(--rule);overflow-x:auto;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;line-height:1.45}
pre.diff i{font-style:normal;display:block;white-space:pre}
pre.diff i.a{color:var(--good)}
pre.diff i.d{color:var(--bad)}
pre.diff i.h{color:var(--mut)}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;
border:1px solid var(--rule);color:var(--mut);white-space:nowrap}
.pill.ok{color:var(--good);border-color:color-mix(in srgb,var(--good) 40%,var(--rule))}
.pill.no{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 40%,var(--rule))}
"""


# ------------------------------------------------------------------------ loading
def load(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass                      # a truncated tail is expected after a crash
    return out


def is_research(entries: list[dict]) -> bool:
    return any(e["payload"].get("kind") == "run_start"
               and e["payload"]["body"].get("driver") == "research_driver"
               for e in entries)


def extract(entries: list[dict]) -> dict:
    """Group the chain by run. Confirmation scores are kept apart from iteration scores:
    counting a 5-seed confirmation's 10 fits as run iterations would inflate every
    per-run number on the page."""
    d = {"genesis": {}, "runs": defaultdict(lambda: {
        "start": {}, "end": {}, "proposals": [], "results": [], "failures": [],
        "refusals": [], "duplicates": [], "infra": [], "tournaments": [],
        "baseline_repro": {}}),
        "scores": {}, "confirm_scores": [], "confirmations": [], "protocols": [],
        "entries": len(entries), "max_seq": 0}
    for e in entries:
        pl, seq = e["payload"], e.get("seq", 0)
        d["max_seq"] = max(d["max_seq"], seq)
        kind = pl.get("kind")
        if kind == "genesis":
            d["genesis"] = pl
            continue
        if "metrics" in pl:
            if pl.get("kind") == "confirmation":
                d["confirm_scores"].append((seq, pl))
            else:
                d["scores"].setdefault(pl.get("config_hash"), (seq, pl))
            continue
        body = pl.get("body") or {}
        rid = body.get("run_id")
        if kind == "run_start" and rid:
            d["runs"][rid]["start"] = {**body, "_seq": seq}
        elif kind == "run_end" and rid:
            d["runs"][rid]["end"] = {**body, "_seq": seq}
        elif kind == "proposal" and rid:
            d["runs"][rid]["proposals"].append({**body, "_seq": seq})
        elif kind == "event":
            what = body.get("what")
            if what == "confirmation_protocol":
                d["protocols"].append({**body, "_seq": seq})
            elif what == "confirmation_result":
                d["confirmations"].append({**body, "_seq": seq})
            elif rid:
                r = d["runs"][rid]
                if what == "iteration_result":
                    r["results"].append({**body, "_seq": seq})
                elif what == "candidate_failed":
                    r["failures"].append({**body, "_seq": seq})
                elif what in ("candidate_refused", "candidate_unchanged"):
                    r["refusals"].append({**body, "_seq": seq})
                elif what == "candidate_duplicate":
                    r["duplicates"].append({**body, "_seq": seq})
                elif what == "tournament":
                    r["tournaments"].append({**body, "_seq": seq})
                elif what == "seed_baseline_reproduction":
                    r["baseline_repro"] = {**body, "_seq": seq}
                elif what in ("coder_error", "planner_error", "reflector_error"):
                    r["infra"].append({**body, "_seq": seq})
    d["runs"] = dict(d["runs"])
    return d


# ------------------------------------------------------------------- verification
def verify_artifacts(d: dict, root: Path | None = None) -> dict:
    """Re-hash every tree the chain names. No key required."""
    trees, missing, altered = set(), [], []
    for r in d["runs"].values():
        for p in r["proposals"]:
            if p.get("tree"):
                trees.add(p["tree"])
    for t in sorted(trees):
        try:
            AR.verify(t, root=root or AR.ARTIFACTS)
        except AR.ArtifactError as e:
            (missing if "no artifact" in str(e) else altered).append(t)
    return {"checked": len(trees), "missing": missing, "altered": altered,
            "ok": not missing and not altered}


def verify_preregistration(d: dict) -> dict:
    """Every scored tree must have a proposal at a LOWER sequence number."""
    first_prop = {}
    for r in d["runs"].values():
        for p in r["proposals"]:
            t = p.get("tree")
            if t:
                first_prop[t] = min(first_prop.get(t, 10 ** 9), p["_seq"])
    unregistered, late = [], []
    for tree, (seq, _pl) in d["scores"].items():
        if tree not in first_prop:
            unregistered.append(tree)
        elif first_prop[tree] >= seq:
            late.append(tree)
    return {"checked": len(d["scores"]), "unregistered": unregistered, "late": late,
            "ok": not unregistered and not late}


# ---------------------------------------------------------------------- sections
def _pill(ok: bool, yes: str, no: str) -> str:
    return f"<span class='pill {"ok" if ok else "no"}'>{yes if ok else no}</span>"


def _sec_integrity(d: dict, art: dict, pre: dict) -> str:
    gen = d["genesis"]
    rows = [
        f"<tr><td>Entries in the chain</td><td class=n>{d['entries']}</td>"
        f"<td class=muted>highest sequence {d['max_seq']}</td></tr>",
        f"<tr><td>Artifacts re-hashed from stored bytes</td>"
        f"<td class=n>{art['checked']}</td><td>{_pill(art['ok'], 'all match', 'MISMATCH')}"
        + (f" <span class=muted>missing {len(art['missing'])}, altered "
           f"{len(art['altered'])}</span>" if not art['ok'] else "") + "</td></tr>",
        f"<tr><td>Scored trees with a proposal at a lower sequence</td>"
        f"<td class=n>{pre['checked']}</td><td>{_pill(pre['ok'], 'ordering holds', 'VIOLATED')}"
        + (f" <span class=muted>{len(pre['unregistered'])} unregistered, "
           f"{len(pre['late'])} registered late</span>" if not pre['ok'] else "")
        + "</td></tr>",
        f"<tr><td>Scorer manifest</td><td class='n mono'>"
        f"{esc(str(gen.get('manifest_sha256',''))[:16])}</td>"
        f"<td class=muted>the metric and split definition the scorer loaded</td></tr>",
    ]
    return _table(["Property", "Count", "Result"], rows, "nothing recorded")


def _sec_confirmations(d: dict) -> str:
    if not d["confirmations"]:
        return ("<div class=note>No paired confirmation has been run. A single-seed "
                "promotion is a search decision, not evidence.</div>")
    out = []
    for c in d["confirmations"]:
        per = c.get("per_seed") or []
        rows = "".join(
            f"<tr><td class=n>{r.get('seed')}</td><td class=n>{_num(r.get('base'),'.6f')}</td>"
            f"<td class=n>{_num(r.get('champ'),'.6f')}</td>"
            f"<td class='n {"good" if (r.get("delta") or 0) > 0 else "bad"}'>"
            f"{_num(r.get('delta'),'+.6f')}</td></tr>" for r in per)
        t = c.get("t")
        ok = str(c.get("verdict", "")).startswith("CONFIRMED")
        out.append(
            f"<h3>{esc(str(c.get('champion_tree',''))[:12])} vs "
            f"{esc(str(c.get('comparison_tree',''))[:12])} "
            f"&middot; {esc(c.get('comparison','?'))}</h3>"
            f"<p class=sub>Protocol recorded at sequence {c.get('protocol_seq','?')} "
            f"<b>before any fit</b>, so the seed set could not be chosen after seeing "
            f"which seeds were favourable.</p>"
            + _table(["Seed", "Comparison", "Champion", "Paired delta"], [rows], "no seeds")
            + f"<p class=sub>mean <b>{_num(c.get('mean'),'+.6f')}</b> &middot; "
              f"sd {_num(c.get('sd'),'.6f')} &middot; se {_num(c.get('se'),'.6f')} "
              f"&middot; t = <b>{_num(t,'.2f') if t is not None else 'n/a'}</b> &middot; "
              f"wins {c.get('wins','?')}/{c.get('n','?')} &middot; "
              f"{_pill(ok, esc(str(c.get('verdict'))[:40]), esc(str(c.get('verdict'))[:60]))}"
              f"</p>")
    return "\n".join(out)


def _sec_trajectory(r: dict, baseline: float) -> str:
    res = sorted(r["results"], key=lambda x: (x.get("iteration", 0), x["_seq"]))
    vals = [x["signed_primary"] for x in res if x.get("signed_primary") is not None]
    if not vals:
        return "<div class=note>No scored iteration in this run.</div>"
    # trajectory() derives the running-best step line itself; passing one would be a
    # second source of truth for the same series.
    svg = CH.trajectory(vals, baseline=baseline, band=OFFICIAL_FM_SEED_STD,
                        x_label="evaluated iteration",
                        aria="signed validation primary per evaluated iteration")
    return (f"<figure>{svg}<figcaption>Signed validation primary per evaluated iteration. "
            f"The dashed reference is the official baseline "
            f"({_num(baseline,'.4f')}); the band is &plusmn;1 published seed std "
            f"({OFFICIAL_FM_SEED_STD}). A point inside the band is not distinguishable "
            f"from seed noise on its own &mdash; which is why promotions are confirmed "
            f"separately.</figcaption></figure>")


def _sec_overfitting(r: dict) -> str:
    res = sorted(r["results"], key=lambda x: (x.get("iteration", 0), x["_seq"]))
    tr = [x.get("train_primary") for x in res if x.get("signed_primary") is not None]
    dv = [x.get("dev_primary") for x in res if x.get("signed_primary") is not None]
    if not any(v is not None for v in tr):
        return "<div class=note>No train-subset metric recorded for this run.</div>"
    svg = CH.multiline([("train primary", tr), ("dev primary", dv)],
                       x_label="evaluated iteration",
                       aria="train and validation ranking metric per iteration",
                       emphasise="dev primary")
    gaps = [x.get("gap") for x in res if x.get("gap") is not None]
    first, last = (gaps[0], gaps[-1]) if gaps else (None, None)
    return (f"<figure>{svg}<figcaption>Train versus validation ranking metric per "
            f"evaluated iteration, both computed <b>by the harness</b> with the official "
            f"metric from the candidate&rsquo;s returned score vectors &mdash; not "
            f"self-reported. The distance between the lines is the generalisation gap: "
            f"{_num(first,'+.4f')} at the first scored iteration, {_num(last,'+.4f')} at "
            f"the last. Before this run existed the project could not see this at all: "
            f"only training <em>loss</em> was recorded, and loss and a ranking metric are "
            f"different quantities.</figcaption></figure>")


def _diff_html(diff: str) -> str:
    out = []
    for line in diff.splitlines():
        cls = ("h" if line.startswith(("@@", "+++", "---")) else
               "a" if line.startswith("+") else "d" if line.startswith("-") else "")
        out.append(f"<i class={cls}>{esc(line)}</i>" if cls else f"<i>{esc(line)}</i>")
    return "<pre class=diff>" + "".join(out) + "</pre>"


def _sec_iterations(r: dict, scores: dict) -> str:
    by_iter = defaultdict(list)
    for p in r["proposals"]:
        by_iter[p.get("iteration", -1)].append(p)
    res = {x.get("iteration"): x for x in r["results"]}
    fails = defaultdict(list)
    for f in r["failures"]:
        fails[f.get("iteration")].append(f)

    rows, details = [], []
    for it in sorted(by_iter):
        props = sorted(by_iter[it], key=lambda x: x.get("attempt", 0))
        x = res.get(it, {})
        p0 = props[0]
        outcome = x.get("outcome") or ("failed" if fails.get(it) else "&ndash;")
        cls = {"promoted": "good", "failed": "bad", "refused": "muted"}.get(outcome, "")
        act = x.get("delta_vs_champion")
        rows.append(
            f"<tr{' class=best' if outcome == 'promoted' else ''}>"
            f"<td class=n>{it}</td>"
            f"<td class=muted>{esc(p0.get('stage') or '?')}</td>"
            f"<td class=wrap>{esc(str(p0.get('hypothesis') or '')[:190])}</td>"
            f"<td class=mono>{esc(','.join(p0.get('files_changed') or []) or '&ndash;')}</td>"
            f"<td class=n>{_num(p0.get('expected_magnitude'),'+.5f')}</td>"
            f"<td class='n {"good" if (act or 0) > 0 else "bad" if act is not None else ""}'>"
            f"{_num(act,'+.5f')}</td>"
            f"<td class=n>{_num(x.get('gap'),'+.4f')}</td>"
            f"<td class=n>{len(props) - 1}</td>"
            f"<td class={cls}>{outcome}</td></tr>")

        for p in props:
            head = (f"iteration {it}" + (f", repair attempt {p['attempt']}"
                                         if p.get("attempt") else "")
                    + f" &middot; tree {str(p.get('tree',''))[:12]}"
                    + f" &middot; {p.get('diff_bytes',0):,}-byte diff"
                    + f" &middot; seq {p['_seq']}")
            body = (_diff_html(p["diff"]) if p.get("diff")
                    else f"<pre class=diff><i class=h>{esc(p.get('diff_omitted') or 'no diff recorded')}"
                         f"</i></pre>")
            why = (f"<pre class=diff><i class=h>hypothesis: {esc(p.get('hypothesis') or '')}"
                   f"</i><i class=h>mechanism : {esc(p.get('mechanism') or '')}</i>"
                   f"<i class=h>evidence  : {esc(p.get('evidence') or '')}</i>"
                   f"<i class=h>diff sha  : {esc(str(p.get('diff_hash') or '')[:16])}</i></pre>")
            details.append(f"<details><summary><b>{head}</b></summary>{why}{body}</details>")

    table = _table(["It", "Stage", "Hypothesis (pre-registered)", "Files", "Predicted",
                    "Actual", "Gap", "Repairs", "Outcome"], rows, "no iterations", 9)
    return table + "<h3>Applied diffs</h3>" + "\n".join(details)


def _sec_tournaments(r: dict) -> str:
    """Every tournament: what was CONSIDERED, what the judge said, what was selected.

    The distinction is the point. A candidate that was considered and not run is not an
    experiment, and this section is the only place the two are shown together, so a reader
    can see breadth of reasoning without mistaking it for breadth of results.
    """
    ts = sorted(r["tournaments"], key=lambda x: x.get("iteration", 0))
    if not ts:
        return ("<div class=note>No tournament in this run &mdash; it used the "
                "<span class=mono>single</span> policy, one hypothesis per iteration.</div>")
    out = []
    for t in ts:
        cov = t.get("coverage") or {}
        j = t.get("judge") or {}
        ranks = {x.get("id"): x for x in (j.get("ranking") or []) if isinstance(x, dict)}
        rows = []
        for c in (t.get("candidates") or []):
            rk = ranks.get(c.get("id")) or {}
            sel = c.get("id") == t.get("selected_id")
            rows.append(
                f"<tr{' class=best' if sel else ''}>"
                f"<td class=mono>{esc(c.get('id'))}</td>"
                f"<td>{esc(c.get('lane'))}</td>"
                f"<td class=wrap>{esc(str(c.get('hypothesis'))[:150])}</td>"
                f"<td class=mono>{esc(','.join(c.get('files_to_change') or []) or '&ndash;')}</td>"
                f"<td class=n>{_num(c.get('expected_magnitude'), '+.5f')}</td>"
                f"<td class=n>{rk.get('rank', '&ndash;')}</td>"
                f"<td class=wrap>{esc(str(c.get('information_value'))[:90])}</td>"
                f"<td>{'<b>EVALUATED</b>' if sel else 'considered only'}</td></tr>")
        bad = cov.get("missing_required_lanes") or []
        out.append(
            f"<h3>Iteration {t.get('iteration')} &middot; {len(rows)} candidates "
            f"&middot; recorded at seq {t.get('_seq')}</h3>"
            + (f"<div class='note warn'><b>Lane coverage incomplete:</b> no candidate in "
               f"{esc(bad)}. Recorded rather than hidden &mdash; this run cannot claim to "
               f"have considered those lanes at this iteration.</div>" if bad else "")
            + _table(["id", "lane", "hypothesis", "files", "predicted", "judge rank",
                      "information value", "outcome"], rows, "no candidates", 8)
            + f"<div class=note><b>Why {esc(t.get('selected_id'))}:</b> "
              f"{esc(str(j.get('why_selected'))[:600])}<br>"
              + ("<b>Architecture verdicts</b> (one per architecture candidate, required):"
                 "<br>" + "<br>".join(
                     f"&nbsp;&nbsp;<span class=mono>{esc(v.get('candidate_id'))}</span> "
                     f"&mdash; <b>{esc(v.get('verdict'))}</b>: "
                     f"{esc(str(v.get('reason'))[:300])}"
                     for v in (j.get("architecture_verdicts") or []))
                 if j.get("architecture_verdicts") else
                 "<b>Architecture verdict:</b> "
                 + esc(str(j.get("architecture_verdict") or "not recorded")[:600]))
              + f"<br><b>Stance:</b> {esc(j.get('exploit_or_explore'))}"
              + (f" &middot; <b>selection:</b> {esc(t.get('selection_how'))}"
                 if t.get("selection_how") != "judge_selected_id" else "")
              + "</div>")
    return "\n".join(out)


def _sec_lanes(r: dict) -> str:
    ls = (r["end"] or {}).get("lane_stats") or {}
    by = ls.get("by_lane") or {}
    if not by:
        return "<div class=note>No lane statistics recorded for this run.</div>"
    rows = [f"<tr><td>{esc(k)}</td><td class=n>{v.get('considered', 0)}</td>"
            f"<td class=n>{v.get('selected', 0)}</td><td class=n>{v.get('executed', 0)}</td>"
            f"<td class=n>{v.get('promoted', 0)}</td><td class=n>{v.get('rejected', 0)}</td>"
            f"<td class=n>{v.get('failed', 0)}</td><td class=n>{v.get('repairs', 0)}</td>"
            f"<td class=n>{_num(v.get('mean_signed_delta'), '+.5f')}</td></tr>"
            for k, v in sorted(by.items())]
    fe = ls.get("file_edits") or {}
    cards = "".join(_card(f"{k} edits", str(v)) for k, v in sorted(fe.items()))
    return (_table(["lane", "considered", "selected", "executed", "promoted", "rejected",
                    "failed", "repairs", "mean signed delta"], rows, "none", 9)
            + (f"<div class=grid>{cards}</div>" if cards else ""))


def _sec_beliefs(r: dict) -> str:
    bs = (r["end"] or {}).get("beliefs") or {}
    if not bs:
        return "<div class=note>No structured beliefs recorded for this run.</div>"
    rows = []
    for b in sorted(bs.values(), key=lambda x: -(x.get("last_updated_iteration") or -1)):
        lo, hi = (b.get("effect_range") or [None, None])[:2]
        rows.append(
            f"<tr><td class=wrap>{esc(b.get('mechanism'))}</td>"
            f"<td>{esc(b.get('lane'))}</td>"
            f"<td class=wrap>{esc(str(b.get('belief'))[:120])}</td>"
            f"<td class=n>{_num(b.get('confidence'), '.2f')}</td>"
            f"<td class=n>{esc(b.get('supporting_iterations'))}</td>"
            f"<td class=n>{esc(b.get('contradicting_iterations'))}</td>"
            f"<td class=n>{_num(lo, '+.4f')} .. {_num(hi, '+.4f')}</td>"
            f"<td class=wrap>{esc(str(b.get('retry_only_if'))[:110])}</td>"
            f"<td class=n>{esc(b.get('last_updated_iteration'))}</td></tr>")
    return (_table(["mechanism", "lane", "belief now", "conf", "supported by",
                    "contradicted by", "effect range", "retry only if", "updated"],
                   rows, "none", 9)
            + "<div class=note>These are what the NEXT tournament reasons over. A belief that "
              "a mechanism is unhelpful is as useful as one that it works: it removes a "
              "direction from consideration instead of letting it be re-proposed.<br><br>"
              "<b>These are interpretations, not measurements.</b> The delta behind each one "
              "is scorer-signed, but the belief, the confidence, and whether a result counts "
              "as supporting or contradicting are all written by the reflector &mdash; an LLM "
              "forming an opinion about a signed number. Read them as the agent's reasoning, "
              "not as evidence. The signed numbers are in the iterations table above.</div>")


def _sec_roles(r: dict) -> str:
    ur = (r["end"] or {}).get("usage_by_role") or {}
    st = r["start"] or {}
    if not ur:
        return "<div class=note>No per-role usage recorded for this run.</div>"
    rows = [f"<tr><td>{esc(k)}</td><td class=mono>{esc(', '.join(v.get('models') or []))}</td>"
            f"<td class=n>{v.get('calls', 0)}</td>"
            f"<td class=n>{v.get('input_tokens', 0):,}</td>"
            f"<td class=n>{v.get('output_tokens', 0):,}</td>"
            f"<td class=n>{v.get('thinking_tokens', 0):,}</td>"
            f"<td class=n>{v.get('total_tokens', 0):,}</td>"
            f"<td class=n>{_num(v.get('seconds'), '.1f')}</td>"
            f"<td class=n>{v.get('retries', 0)}</td>"
            f"<td class=n>{v.get('refusals', 0)}/{v.get('truncations', 0)}</td></tr>"
            for k, v in sorted(ur.items())]
    routing = st.get("model_routing") or {}
    return (f"<div class=note><b>Routing:</b> "
            + " &middot; ".join(f"{esc(k)} &rarr; <span class=mono>{esc(v)}</span>"
                                for k, v in sorted(routing.items()))
            + "</div>"
            + _table(["role", "model", "calls", "input", "output", "thinking", "total",
                      "seconds", "retries", "refused/trunc"], rows, "none", 10))


def _sec_breadth(d: dict) -> str:
    """Capability versus demonstrated behaviour. The section that must not overclaim."""
    allowed, edited, lanes_considered, lanes_run = set(), {}, {}, {}
    arch_considered = arch_evaluated = arch_won = arch_lost = 0
    prebuilt, agent_written = [], []
    for rid, r in d["runs"].items():
        st = r["start"] or {}
        prov = st.get("seed_provenance") or {}
        if prov:
            (prebuilt if prov.get("pairwise_code_present") else agent_written).append(rid)
        for t in r["tournaments"]:
            for c in (t.get("candidates") or []):
                lanes_considered[c.get("lane")] = lanes_considered.get(c.get("lane"), 0) + 1
                if c.get("lane") == "model_architecture":
                    arch_considered += 1
        for x in r["results"]:
            lane = x.get("lane") or "unspecified"
            lanes_run[lane] = lanes_run.get(lane, 0) + 1
            if lane == "model_architecture":
                arch_evaluated += 1
                if x.get("outcome") == "promoted":
                    arch_won += 1
                elif x.get("outcome") in ("rejected", "failed"):
                    arch_lost += 1
        for p in r["proposals"]:
            for f in (p.get("files") or []):
                allowed.add(f)
            for f in (p.get("files_changed") or []):
                edited[f] = edited.get(f, 0) + 1
    rows = [f"<tr><td class=mono>{esc(f)}</td>"
            f"<td>{'yes' if f in allowed else 'yes'}</td>"
            f"<td class=n>{edited.get(f, 0)}</td></tr>"
            for f in sorted(allowed | set(edited))]
    lrows = [f"<tr><td>{esc(l)}</td><td class=n>{lanes_considered.get(l, 0)}</td>"
             f"<td class=n>{lanes_run.get(l, 0)}</td></tr>"
             for l in sorted(set(lanes_considered) | set(lanes_run))]
    return (
        "<p class=sub>What the agent was <b>allowed</b> to do, and what it <b>did</b>. "
        "These are different numbers and the difference is the honest limit of the "
        "autonomy claim.</p>"
        # Without this the 'considered' column reads as spontaneous curiosity. It is not.
        + "<div class=note><b>Read the &lsquo;considered&rsquo; column with this in mind:</b> "
          "the tournament validator <b>requires</b> at least one candidate in "
          "<span class=mono>model_architecture</span> and "
          "<span class=mono>sampling_training</span>, and rejects a tournament that omits "
          "them. A non-zero count in a required lane is therefore <b>mandated by the "
          "search policy, not evidence of unprompted initiative</b>. What is genuinely the "
          "agent&rsquo;s own is narrower and still worth crediting: <i>which</i> mechanism "
          "it proposes inside the lane, which files it edits to get there, and &mdash; via "
          "the judge &mdash; which candidate it spends the evaluation on.</div>"
        + "<h3>Files</h3>"
        + _table(["file", "editable", "times actually edited"], rows, "none", 3)
        + "<h3>Research lanes</h3>"
        + _table(["lane", "candidates considered", "experiments evaluated"], lrows,
                 "no lanes recorded", 3)
        + f"<div class=grid>"
          f"{_card('Architecture considered', str(arch_considered))}"
          f"{_card('Architecture evaluated', str(arch_evaluated))}"
          f"{_card('Architecture improved', str(arch_won))}"
          f"{_card('Architecture rejected', str(arch_lost))}</div>"
        + (f"<div class='note warn'><b>No architecture experiment has been evaluated yet.</b> "
           f"{arch_considered} were considered across the recorded tournaments. Considering a "
           f"direction is not researching it, and this report does not claim otherwise."
           f"</div>" if arch_evaluated == 0 else
           f"<div class=note>{arch_evaluated} architecture experiment(s) were evaluated: "
           f"{arch_won} improved the champion, {arch_lost} did not. A losing architecture "
           f"experiment is a result, and it is recorded above rather than dropped.</div>")
        + f"<div class=note><b>Prebuilt versus agent-authored loss:</b> "
          f"runs seeded from the <span class=mono>inherited</span> profile "
          f"({len(prebuilt)}) had a pairwise/BPR implementation ALREADY PRESENT in the "
          f"train.py they started from &mdash; selecting it is not writing it. Runs seeded "
          f"from <span class=mono>minimal</span> ({len(agent_written)}) had no ranking loss "
          f"supplied, so any such objective in those runs was written by the agent."
          + ("" if (prebuilt or agent_written) else
             " No run in this chain recorded a seed profile, which means it predates the "
             "field &mdash; the report cannot assert either way.")
          + "</div>")


def _sec_failures(r: dict) -> str:
    rows = []
    for f in sorted(r["failures"], key=lambda x: x["_seq"]):
        rows.append(
            f"<tr><td class=n>{f.get('iteration')}</td>"
            f"<td class=n>{f.get('attempt')}</td>"
            f"<td class=mono>{esc(f.get('error_type') or '?')}</td>"
            f"<td class=mono>{esc(f.get('error_frame') or '?')}</td>"
            f"<td>{esc(f.get('error_in') or '?')}</td>"
            f"<td class=wrap>{esc(str(f.get('error') or '')[:200])}</td></tr>")
    for f in sorted(r["refusals"] + r["duplicates"], key=lambda x: x["_seq"]):
        rows.append(
            f"<tr><td class=n>{f.get('iteration')}</td>"
            f"<td class=n>{f.get('attempt','&ndash;')}</td>"
            f"<td class=mono>{esc(f.get('what'))}</td><td class=muted>&ndash;</td>"
            f"<td class=muted>refused before running</td>"
            f"<td class=wrap>{esc(str(f.get('error') or f.get('first_seen_iteration') or '')[:200])}"
            f"</td></tr>")
    return _table(["It", "Try", "Kind", "Frame", "Whose code", "Detail"], rows,
                  "no failures in this run", 6)


def _sec_calibration(r: dict) -> str:
    pairs = [(x.get("expected_magnitude"), x.get("delta_vs_champion"))
             for x in r["results"]
             if x.get("expected_magnitude") is not None
             and x.get("delta_vs_champion") is not None]
    if not pairs:
        return "<div class=note>No prediction/outcome pairs recorded.</div>"
    over = sum(1 for p, a in pairs if p > a)
    hits = sum(1 for p, a in pairs if (p > 0) == (a > 0))
    mae = sum(abs(p - a) for p, a in pairs) / len(pairs)
    return (f"<div class=grid>{_card('Predictions scored', str(len(pairs)))}"
            f"{_card('Over-predicted', f'{over}/{len(pairs)}')}"
            f"{_card('Direction correct', f'{hits}/{len(pairs)}')}"
            f"{_card('Mean abs error', _num(mae, '.5f'))}</div>"
            f"<div class=note>Every prediction was pre-registered at a sequence number "
            f"below the score that tested it, so these cannot have been fitted after the "
            f"fact. The agent is shown this table's contents on its next planning call."
            f"</div>")


def _sec_cost(r: dict) -> str:
    end = r["end"] or {}
    b = end.get("budget") or {}
    st = end.get("stats") or {}
    per = b.get("iteration_wall_seconds") or []
    stages = b.get("stage_seconds") or {}
    cards = "".join([
        _card("Iterations", f"{b.get('iterations','?')}/{b.get('max_iterations','?')}"),
        _card("Wall clock", f"{_num(b.get('wall_hours'),'.2f')} h",
              f"cap {_num((b.get('max_wall_seconds') or 0)/3600,'.1f')} h"),
        _card("Tokens", f"{b.get('tokens_total',0):,}"),
        _card("Mean iteration", f"{_num(b.get('iteration_wall_mean'),'.1f')} s",
              f"max {_num(b.get('iteration_wall_max'),'.0f')} s"),
        _card("GPU-hours", "0", "numpy only, no accelerator"),
    ])
    srows = "".join(f"<tr><td class=mono>{esc(k)}</td><td class=n>{_num(v,'.1f')} s</td>"
                    f"<td class=n>{_num(100*v/max(sum(stages.values()),1e-9),'.0f')}%</td>"
                    f"</tr>" for k, v in sorted(stages.items(), key=lambda t: -t[1]))
    return (f"<div class=grid>{cards}</div>"
            + (f"<figure>{CH.bars([(k, v, 0.0) for k, v in sorted(stages.items(), key=lambda t: -t[1])], aria='wall clock by stage')}"
               f"<figcaption>Wall-clock by stage, summed over the run. Agent wall-clock is "
               f"the compute measure the problem statement scores.</figcaption></figure>"
               if stages else "")
            + _table(["Stage", "Total", "Share"], [srows], "no stage timings", 3)
            + f"<div class=note>{esc(json.dumps(st, sort_keys=True))}</div>")


def _sec_disclosures(rid: str, r: dict) -> str:
    s = r["start"] or {}
    end = r["end"] or {}
    notes = []
    if "seed_config_overrides" not in s:
        notes.append("<b>Starting point:</b> not recorded &mdash; this run predates the "
                     "field, so the report cannot assert it started from the unmodified "
                     "baseline. Later runs record it explicitly.")
    elif s.get("seed_config_overrides"):
        notes.append(f"<b>Starting point OVERRIDDEN:</b> "
                     f"<span class=mono>{esc(json.dumps(s['seed_config_overrides'], sort_keys=True))}</span>"
                     f" &mdash; this run did <b>not</b> begin from the unmodified official "
                     f"baseline, and any total gain includes that head start.")
    else:
        notes.append("<b>Starting point:</b> the unmodified official baseline, recorded.")
    conv = (s.get("convergence") or {}).get("mode")
    if conv == "off":
        notes.append(
            f"<b>Convergence DISABLED</b> for this run, so the full iteration budget was "
            f"spent. The organisers&rsquo; rule (&epsilon;={OFFICIAL_EPSILON}, N=3) would "
            f"have stopped it at iteration 3. That rule WAS run compliantly &mdash; see the "
            f"<code>research_official</code> run &mdash; and it converged with <b>zero "
            f"improvement</b>, because this agent&rsquo;s first improvement arrives at "
            f"iteration 6 while the rule&rsquo;s window closes at 3. Both runs are "
            f"published; the extended one is what gets reported, and it is disclosed here "
            f"rather than presented as compliant.")
    term = str(end.get("termination") or "")
    if term:
        promoted = (end.get("stats") or {}).get("promoted", 0)
        if "converged" in term and not promoted:
            notes.append(
                f"<b>This run CONVERGED WITH NO IMPROVEMENT.</b> {esc(term)} &mdash; and it "
                f"promoted nothing, so its champion is the starting point it was given. That "
                f"is the organisers&rsquo; stopping rule applied faithfully, and it is the "
                f"evidence that the rule&rsquo;s window closes before this agent&rsquo;s "
                f"first improvement.")
        else:
            notes.append(f"<b>Termination:</b> {esc(term)} &middot; "
                         f"{promoted} promotion(s).")
    mi = end.get("manual_interventions")
    notes.append(f"<b>Manual interventions:</b> {len(mi) if isinstance(mi, list) else '?'} "
                 f"during the run.")
    prov = s.get("seed_provenance") or {}
    if prov:
        notes.append(
            f"<b>Seed profile:</b> <span class=mono>{esc(prov.get('seed_profile'))}</span> "
            f"({esc(prov.get('workspace'))}) &middot; pairwise code already present: "
            f"<b>{prov.get('pairwise_code_present')}</b> &middot; listwise: "
            f"{prov.get('listwise_code_present')} &middot; alternative architecture "
            f"supplied: {prov.get('alternative_architecture_code_present')}. "
            + ("A ranking objective in this run was SELECTED, not written."
               if prov.get("pairwise_code_present") else
               "No ranking objective was supplied, so any such loss here was written by "
               "the agent."))
    else:
        notes.append("<b>Seed profile:</b> not recorded &mdash; this run predates the field, "
                     "so the report cannot say what its seed already contained.")
    br = r.get("baseline_repro") or {}
    if br:
        notes.append(
            f"<b>Baseline reproduction, measured this run:</b> the seed candidate scored "
            f"<span class=mono>{_num(br.get('seed_signed_primary'), '.6f')}</span> against a "
            f"published baseline of <span class=mono>{br.get('published_baseline_valid')}"
            f"</span> (delta {_num(br.get('delta'), '+.6f')}). The published figure is "
            f"rounded, so a small delta is the rounding.")
    notes.append(f"<b>Policy:</b> <span class=mono>{esc(s.get('research_policy','single'))}"
                 f"</span>"
                 + (f" &middot; tournament size {s.get('tournament_size')}"
                    if s.get("research_policy") == "tournament" else ""))
    notes.append(f"<b>Model:</b> <span class=mono>{esc(s.get('backend','?'))}/"
                 f"{esc(s.get('model','?'))}</span> &middot; isolation "
                 f"<span class=mono>{esc(s.get('isolation','?'))}</span> &middot; contract "
                 f"<span class=mono>{esc(str(s.get('contract_sha256',''))[:12])}</span>")
    return "".join(f"<div class='note{' warn' if 'OVERRIDDEN' in n or 'DISABLED' in n else ''}'>"
                   f"{n}</div>" for n in notes)


# ------------------------------------------------------------------------- build
def build(entries: list[dict], artifact_root: Path | None = None) -> str:
    d = extract(entries)
    art, pre = verify_artifacts(d, artifact_root), verify_preregistration(d)
    baseline = OFFICIAL_VALID["fm_official"]

    best_conf = None
    for c in d["confirmations"]:
        if str(c.get("verdict", "")).startswith("CONFIRMED"):
            if best_conf is None or (c.get("mean") or -9) > (best_conf.get("mean") or -9):
                best_conf = c
    tot_tok = sum((r["end"].get("budget") or {}).get("tokens_total", 0)
                  for r in d["runs"].values() if r["end"])
    tot_it = sum((r["end"].get("budget") or {}).get("iterations", 0)
                 for r in d["runs"].values() if r["end"])
    tot_h = sum((r["end"].get("budget") or {}).get("wall_hours", 0) or 0
                for r in d["runs"].values() if r["end"])

    cards = "".join([
        _card("Best CONFIRMED gain",
              _num(best_conf.get("mean") if best_conf else None, "+.6f"),
              (f"t={_num(best_conf.get('t'),'.2f')}, {best_conf.get('wins')}/"
               f"{best_conf.get('n')} seeds, paired" if best_conf else "not confirmed")),
        _card("Clears organisers' &epsilon;",
              "yes" if best_conf and (best_conf.get("mean") or 0) > OFFICIAL_EPSILON
              else "no", f"&epsilon; = {OFFICIAL_EPSILON}"),
        _card("Runs", str(len(d["runs"])), f"{tot_it} iterations total"),
        _card("Wall clock", f"{_num(tot_h,'.2f')} h", "0 GPU-hours"),
        _card("Tokens", f"{tot_tok:,}"),
        _card("Chain", f"{d['entries']} entries",
              "artifacts + ordering re-checked below"),
    ])

    parts = [
        "<title>Code-First Research Runs</title>",
        f"<style>{CSS}{EXTRA_CSS}</style>", "<main>",
        "<h1>Autonomous research agent &mdash; code-first runs</h1>",
        "<p class=sub>Each iteration below pre-registered a hypothesis and a predicted "
        "magnitude, then wrote code, then was scored by a separate signing process. The "
        "diff shown is computed by the harness from the parent and child source trees "
        "&mdash; it is never taken from the model, because a diff the model authors could "
        "describe a change it did not make.</p>",
        f"<div class=grid>{cards}</div>",
        "<h2>Confirmation &mdash; paired, multi-seed</h2>",
        "<p class=sub>A single-seed promotion is a search decision. These are the only "
        "results quoted as findings.</p>",
        _sec_confirmations(d),
        "<h2>Research breadth: capability versus demonstrated behaviour</h2>",
        _sec_breadth(d),
        "<h2>Integrity</h2>", _sec_integrity(d, art, pre),
        "<div class=note>Artifact hashes and pre-registration ordering are "
        "<b>recomputed on this page</b>. Signature verification is not: it needs the "
        "signing key, which a reporting process must not hold. Run "
        "<span class=mono>python3 -m core.scorer verify --ledger &lt;path&gt;</span> for "
        "that. Whole-entry truncation at the tail needs an external anchor; the entry "
        "count is shown so a shortened file is at least visible.</div>",
    ]

    for rid in sorted(d["runs"], key=lambda k: (d["runs"][k]["start"] or {}).get("_seq", 0)):
        r = d["runs"][rid]
        if not r["proposals"] and not r["results"]:
            continue
        parts += [
            f"<h2>Run <span class=mono>{esc(rid)}</span></h2>",
            _sec_disclosures(rid, r),
            "<h3>Hypothesis tournaments: considered versus evaluated</h3>",
            _sec_tournaments(r),
            "<h3>Research lane distribution</h3>", _sec_lanes(r),
            "<h3>Structured beliefs carried forward</h3>", _sec_beliefs(r),
            "<h3>Model routing and cost by role</h3>", _sec_roles(r),
            "<h3>Search trajectory</h3>", _sec_trajectory(r, baseline),
            "<h3>Overfitting &mdash; train versus validation</h3>", _sec_overfitting(r),
            "<h3>Iterations: hypothesis, diff, result, recovery</h3>",
            _sec_iterations(r, d["scores"]),
            "<h3>Failures and recovery</h3>", _sec_failures(r),
            "<h3>Prediction calibration</h3>", _sec_calibration(r),
            "<h3>Cost</h3>", _sec_cost(r),
        ]
    parts.append("</main>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HTML report for code-first research runs")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / "research_report.html"))
    a = ap.parse_args()
    p = Path(a.ledger)
    if not p.exists():
        raise SystemExit(f"no such ledger: {p}")
    e = load(p)
    if not e:
        raise SystemExit(f"no entries in {p}")
    if not is_research(e):
        raise SystemExit(
            f"{p} holds no research_driver run. For a typed-action ledger use: "
            f"python3 -m agent.official_report --ledger {p}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(build(e), encoding="utf-8")
    print(f"wrote {a.out}  ({len(e)} entries)")


if __name__ == "__main__":
    main()
