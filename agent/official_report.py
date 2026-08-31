"""Self-contained HTML report for the OFFICIAL task, built from a ledger.

One file, no server, no JS. It renders on GitHub Pages, in a browser opened from a
finder window, and inside a Devpost iframe, and it cannot break in front of a judge.

Two things this report is careful about, because they are the difference between a
report and a brochure:

  1. It separates MEASURED from SELF-REPORTED. A `score` payload was produced by the
     out-of-process scorer against a held file hash; a training curve is the strategy's
     own arithmetic, recorded as a receipt-order `event`. They are shown in different
     sections with the distinction stated, and the two are cross-checked against each
     other where they should agree.
  2. It shows the seed-noise band. The official FM has a seed std of 0.0008 on this
     split, so a +0.0005 "win" is not one. Drawing the band makes that visible instead
     of leaving it to a reader who is scrolling.

It reads payloads only -- it does NOT verify signatures. Integrity is a separate
command that needs the key:  python3 -m core.scorer verify --ledger <path>
"""
from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
from pathlib import Path

from agent import charts as CH

ROOT = Path(__file__).resolve().parents[1]
LEDGER = pathlib.Path(os.environ.get("TRACK2_LEDGER") or ROOT / "runs" / "ledger.jsonl")

# Organisers' published numbers. Preferred from the ledger's own `reference` block when
# present, so a report of an old run shows what that run was compared against.
FALLBACK_BASELINE = 0.6016
FALLBACK_SEED_STD = 0.0008


def load(path: Path) -> list[dict]:
    """Tolerant read: stop at the first unparseable line rather than raise, so a run
    that was killed mid-write still produces a report of everything before the tear."""
    out = []
    for ln in path.read_text(encoding="utf-8").split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            break
    return out


def is_official(entries: list[dict]) -> bool:
    for e in entries:
        p = e.get("payload", {})
        if p.get("kind") == "genesis":
            return str(p.get("task", "")).startswith("official")
        if p.get("kind") == "score":
            return "primary" in (p.get("metrics") or {})
    return False


def esc(x) -> str:
    return html.escape(str(x))


def _num(v, spec=".5f", dash="&ndash;") -> str:
    try:
        if v is None:
            return dash
        return format(float(v), spec)
    except (TypeError, ValueError):
        return dash


def _card(label: str, value: str, sub: str = "") -> str:
    return (f"<div class=card><span>{label}</span><b>{value}</b>"
            + (f"<i>{sub}</i>" if sub else "") + "</div>")


def _table(headers: list[str], rows: list[str], empty: str, ncol: int | None = None) -> str:
    n = ncol or len(headers)
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(rows) or f"<tr><td colspan={n} class=muted>{empty}</td></tr>"
    return f"<div class=scroll><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


CSS = """
:root{--bg:#fcfcfd;--fg:#15181e;--mut:#69707c;--rule:#e2e5ea;--card:#fff;
--good:#0d6f4c;--bad:#a5301c;--accent:#1b48c4;--band:#1b48c4;--ink2:#8a919c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d0f13;
--fg:#e7eaf0;--mut:#8d949f;--rule:#242931;--card:#14171c;--good:#41c091;
--bad:#ff8f76;--accent:#82a6ff;--band:#82a6ff;--ink2:#6f7783}}
:root[data-theme=dark]{--bg:#0d0f13;--fg:#e7eaf0;--mut:#8d949f;--rule:#242931;
--card:#14171c;--good:#41c091;--bad:#ff8f76;--accent:#82a6ff;--band:#82a6ff;--ink2:#6f7783}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:40px 22px 90px;
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}
main{max-width:1040px;margin:0 auto}
h1{font-size:27px;margin:0 0 5px;letter-spacing:-.022em;text-wrap:balance}
h2{font-size:17px;margin:38px 0 8px;letter-spacing:-.012em;text-wrap:balance}
h3{font-size:13px;margin:20px 0 6px;color:var(--mut);text-transform:uppercase;
letter-spacing:.07em;font-weight:600}
p.sub{color:var(--mut);margin:0 0 6px;max-width:74ch}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
gap:10px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--rule);border-radius:7px;padding:11px 13px}
.card b{display:block;font-size:20px;font-variant-numeric:tabular-nums;
letter-spacing:-.02em;margin-top:2px}
.card span{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card i{display:block;font-style:normal;color:var(--mut);font-size:11.5px;margin-top:3px}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:7px;
background:var(--card);margin:10px 0}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;padding:8px 11px;color:var(--mut);font-size:10.5px;
text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--rule);
white-space:nowrap;background:var(--card);position:sticky;top:0}
td{padding:7px 11px;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:last-child td{border-bottom:none}
tr.best{background:color-mix(in srgb,var(--good) 10%,transparent)}
.n{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.strong{font-weight:600}.good{color:var(--good)}.bad{color:var(--bad)}.muted{color:var(--mut)}
.wrap{max-width:52ch;white-space:normal}
figure{margin:12px 0 0}
figcaption{color:var(--mut);font-size:12.5px;margin-top:7px;max-width:80ch}
svg{display:block;max-width:100%;height:auto;background:var(--card);
border:1px solid var(--rule);border-radius:7px;color:var(--accent)}
.ax{font-size:10px;fill:var(--mut)}
.ax-line{stroke:var(--rule);stroke-width:1}
.grid{stroke:var(--rule);stroke-width:.6;stroke-dasharray:2 3}
.band{fill:var(--band);opacity:.12}
.ref{stroke:var(--mut);stroke-width:1;stroke-dasharray:5 4}
.series{fill:none;stroke:currentColor;stroke-width:1.7}
.best-step{fill:none;stroke:var(--good);stroke-width:1.2;stroke-dasharray:3 3;opacity:.85}
.curve{fill:none;stroke:var(--ink2);stroke-width:1.1;opacity:.75}
.curve-em{fill:none;stroke:currentColor;stroke-width:2}
.pt{fill:currentColor;opacity:.85}
.pt-best{fill:var(--good)}
.bar{fill:var(--ink2);opacity:.45}
.bar-hi{fill:var(--good)}
.note{border-left:3px solid var(--rule);padding:9px 14px;color:var(--mut);
font-size:13.5px;margin:12px 0;max-width:84ch}
.note b{color:var(--fg)}
.warn{border-left-color:var(--bad)}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--mut);font-size:12px;margin:8px 0 0}
.legend span::before{content:'';display:inline-block;width:16px;height:0;
vertical-align:middle;margin-right:6px;border-top-width:2px;border-top-style:solid}
.legend .s::before{border-color:var(--accent)}
.legend .b::before{border-color:var(--good);border-top-style:dashed}
.legend .r::before{border-color:var(--mut);border-top-style:dashed}
"""


# --------------------------------------------------------------------- extract
def extract(entries: list[dict]) -> dict:
    """One pass over the ledger into the shapes the sections need.

    Keyed by config_hash throughout, because that is the only join key present in every
    payload: `evaluate` does not know the iteration number it is serving, so threading
    one through would have meant new state in the driver for the report's convenience.
    """
    pls = [(e.get("seq"), e.get("payload", {})) for e in entries]
    d = {"genesis": {}, "scores": [], "proposals": [], "events": [],
         "run_start": [], "run_end": []}
    for seq, p in pls:
        k = p.get("kind")
        if k == "genesis" and not d["genesis"]:
            d["genesis"] = p
        elif k == "score":
            d["scores"].append((seq, p))
        elif k == "proposal":
            d["proposals"].append((seq, p.get("body", {})))
        elif k == "event":
            d["events"].append((seq, p.get("body", p)))
        elif k in ("run_start", "run_end"):
            d[k].append(p.get("body", {}))

    # Training curves per config, VALID split only. Two reasons the filter is required,
    # both instances of the same mistake found in `by_cfg` below:
    #   * `self_reported_best` is by construction a VALIDATION number -- the ranker
    #     early-stops on valid -- so pairing it with a test-split `scorer_primary` in
    #     _sec_crosscheck() would report a large spurious DISAGREES and tell the reader to
    #     distrust every number on the page.
    #   * this is a plain assignment, not "first wins", so a later test-split curve for the
    #     same config would OVERWRITE the valid one rather than be ignored.
    # A curve with no `split` predates the field and is kept, since only the driver's
    # valid-scoring path has ever written one.
    d["curves"] = {}
    for _seq, b in d["events"]:
        if (b.get("what") == "train_curve" and b.get("config_hash")
                and b.get("split") in (None, "valid")):
            d["curves"][b["config_hash"]] = b
    d["inspections"] = [b for _s, b in d["events"]
                        if b.get("what") in ("inspect_result", "inspect_failed")]
    d["failures"] = [b for _s, b in d["events"]
                     if b.get("what") in ("patch_failed", "inspect_failed")]

    # Scored VALID primary per config; first score wins (a repeat of the same config is
    # the same fit, and the driver dedups before it ever gets here).
    #
    # The `split == "valid"` filter is load-bearing, not defensive. The only consumer is
    # _sec_iterations, which computes `actual = cand - parent_valid_primary`; without the
    # filter a config scored on TEST first put a test primary in `cand`, and the table
    # silently displayed a test-minus-validation difference as though it were the gain the
    # proposer had predicted.
    d["by_cfg"] = {}
    for _seq, p in d["scores"]:
        ch = p.get("config_hash")
        if ch and p.get("split") == "valid" and ch not in d["by_cfg"]:
            d["by_cfg"][ch] = p
    return d


def _reference(d: dict) -> tuple[float, float]:
    """The baseline this report compares against, taken from the VALID split.

    Split matters here and the first version ignored it. `core/official_scorer.py` writes
    `published_baseline` PER SPLIT -- 0.6016 on a valid row, 0.5946 on a test row -- so
    scanning ledger order for the first one found returns the TEST baseline whenever a
    test score happens to be recorded first. Everything downstream then compares a
    validation number against it: the trajectory chart draws a test baseline under a
    validation curve, and the delta card reads +0.00897 (+11.2 seed sd) instead of
    +0.00197 (+2.5). That error is in the direction that flatters the result, which is the
    one direction a report must never be wrong in.
    """
    for want in ("valid", None):                    # valid first, then any split
        for _seq, p in d["scores"]:
            if want is not None and p.get("split") != want:
                continue
            r = p.get("reference") or {}
            if r.get("published_baseline") is not None:
                return float(r["published_baseline"]), FALLBACK_SEED_STD
    return FALLBACK_BASELINE, FALLBACK_SEED_STD


# --------------------------------------------------------------------- sections
def _sec_trajectory(d: dict, baseline: float, sd: float) -> str:
    vals, labels = [], []
    for _seq, p in d["scores"]:
        if p.get("split") != "valid":
            continue
        vals.append(p["metrics"]["primary"])
        labels.append(f"{p['metrics']['primary']:.5f}  cfg {str(p.get('config_hash',''))[:10]}")
    if not vals:
        return "<p class=muted>no valid-split scores in this ledger</p>"
    fig = CH.trajectory(vals, baseline=baseline, band=sd, labels=labels,
                        aria=("validation primary per evaluation against the official "
                              "baseline and its seed-noise band"))
    inside = sum(1 for v in vals if abs(v - baseline) <= sd)
    return f"""<figure>{fig}
<div class=legend><span class=s>each evaluation</span><span class=b>running best</span>
<span class=r>official baseline</span><span>shaded: &plusmn;1 seed sd ({sd:g})</span></div>
<figcaption>Validation primary for every configuration the scorer measured, in ledger
order. <b>{inside} of {len(vals)}</b> evaluations land inside the seed-noise band, i.e.
are indistinguishable from the baseline whatever their sign. The band is why this chart
is drawn rather than a bare line: the official FM's own seed spread is {sd:g}, so a gain
smaller than that is not evidence of anything.</figcaption></figure>"""


def _sec_curves(d: dict, best_hash: str | None) -> str:
    """Two charts: validation primary per epoch, and training loss per epoch."""
    va, tl = [], []
    for ch, c in d["curves"].items():
        tag = f"{c.get('loss','?')} {str(ch)[:8]}"
        if c.get("valid_history"):
            va.append((tag, c["valid_history"]))
        if c.get("train_loss_history"):
            tl.append((tag, c["train_loss_history"]))
        for i, mv in enumerate(c.get("member_valid_history") or []):
            if mv:
                va.append((f"{tag} seed+{i}", mv))
        for i, mt in enumerate(c.get("member_train_loss_history") or []):
            if mt:
                tl.append((f"{tag} seed+{i}", mt))
    if not va and not tl:
        return ("<div class='note'>No training curves in this ledger. Curves are "
                "recorded as <span class=mono>train_curve</span> events, which were "
                "added after this run; the run itself is unaffected.</div>")
    em = None
    if best_hash and best_hash in d["curves"]:
        em = f"{d['curves'][best_hash].get('loss','?')} {str(best_hash)[:8]}"
    out = []
    if va:
        out.append("<h3>Validation primary per epoch</h3><figure>"
                   + CH.multiline(va, emphasise=em, x_label="epoch",
                                  aria="validation primary per epoch per configuration")
                   + "<figcaption>One line per fit; the emphasised line is the "
                     "configuration that ended up best. A line that turns over and then "
                     "stops is early stopping firing on patience, which is the intended "
                     "behaviour, not a truncated run.</figcaption></figure>")
    if tl:
        out.append("<h3>Training loss per epoch</h3><figure>"
                   + CH.multiline(tl, emphasise=em, x_label="epoch", invert_best=True,
                                  aria="mean minibatch training loss per epoch")
                   + "<figcaption>Mean minibatch loss. Losses are <b>not comparable "
                     "across the three objectives</b> &mdash; pointwise logloss, pairwise "
                     "BPR and listwise softmax are different quantities, so read each "
                     "line's shape, never two lines' heights against each "
                     "other.</figcaption></figure>")
    out.append("<div class='note'>These curves are <b>self-reported by the strategy</b>, "
               "not measured by the scorer: they are the model's own per-epoch arithmetic, "
               "recorded as receipt-order events. Only the numbers in the trajectory chart "
               "above and the evaluations table below carry a scorer signature over a "
               "prediction-file hash.</div>")
    return "".join(out)


def _sec_crosscheck(d: dict) -> str:
    """Self-reported best vs scorer-measured. For a single fit these must agree: the
    ranker restores its best epoch before predicting, so a gap means early stopping and
    scoring disagree about which checkpoint won -- a real bug, and one that would
    otherwise be invisible."""
    rows, gaps = [], []
    for ch, c in sorted(d["curves"].items()):
        sr, sc = c.get("self_reported_best"), c.get("scorer_primary")
        ens = c.get("ensemble_seeds")
        members = c.get("member_valid_primary") or []

        # An ENSEMBLE has no single self-reported best: `strategy.ranker.run` fits each
        # seed, combines them, and never scores the combination itself, so its info dict
        # carries `member_valid_primary` and no `valid_primary_best`.
        #
        # The first version skipped any row missing `self_reported_best`, which silently
        # dropped every ensemble from this table -- and the "not expected to match" branch
        # written for exactly that case became DEAD CODE. It passed its test only because
        # the synthetic fixture set a `best` value the real ensemble path never produces:
        # a test validating a branch reality cannot reach. Show the member spread instead,
        # which is the informative thing anyway -- it is the reason ensembling helps.
        if ens:
            lo = min(members) if members else None
            hi = max(members) if members else None
            span = ("&ndash;" if lo is None
                    else f"{lo:.6f} &ndash; {hi:.6f} ({len(members)} seeds)")
            beat = (sc is not None and members
                    and float(sc) > sum(members) / len(members))
            rows.append(
                f"<tr><td class=mono>{esc(str(ch)[:12])}</td>"
                f"<td>{esc(c.get('loss','?'))} &times;{ens} {esc(c.get('ensemble_combine',''))}</td>"
                f"<td class='n muted'>{span}</td>"
                f"<td class=n>{_num(sc, '.6f')}</td>"
                f"<td class='n muted'>&ndash;</td>"
                f"<td class='muted'>ensemble &mdash; members are combined after fitting, "
                f"so there is no single self-reported best to compare"
                f"{'; the combination beats the mean of its members' if beat else ''}"
                f"</td></tr>")
            continue

        if sr is None or sc is None:
            continue
        gap = abs(float(sr) - float(sc))
        gaps.append(gap)
        note = "agrees" if gap <= 1e-6 else "DISAGREES"
        cls = "good" if gap <= 1e-6 else "bad strong"
        rows.append(f"<tr><td class=mono>{esc(str(ch)[:12])}</td>"
                    f"<td>{esc(c.get('loss','?'))}</td>"
                    f"<td class=n>{_num(sr, '.6f')}</td>"
                    f"<td class=n>{_num(sc, '.6f')}</td>"
                    f"<td class='n {cls}'>{gap:.2e}</td>"
                    f"<td class='{cls}'>{note}</td></tr>")
    worst = max(gaps) if gaps else None
    if worst is None:
        verdict = ("<div class='note'>No single-fit evaluation carried both numbers, so "
                   "there is nothing to cross-check in this ledger.</div>")
    elif worst <= 1e-6:
        verdict = (f"<div class='note'>Largest disagreement across all single fits: "
                   f"<b>{worst:.2e}</b>. Early stopping and scoring pick the same "
                   f"checkpoint, to the precision the curve is stored at (6 dp).</div>")
    else:
        verdict = (f"<div class='note warn'>Largest disagreement: <b>{worst:.2e}</b>, "
                   f"above the 1e-6 the stored precision allows. Early stopping and "
                   f"scoring disagree about which epoch won &mdash; investigate before "
                   f"trusting any number on this page.</div>")
    return _table(["config", "loss", "self-reported best", "scorer measured",
                   "|gap|", "verdict"], rows,
                  "no train_curve events recorded") + verdict


def _sec_evaluations(d: dict, baseline: float, sd: float) -> str:
    rows = []
    vals = [p["metrics"]["primary"] for _s, p in d["scores"] if p.get("split") == "valid"]
    top = max(vals) if vals else None
    for i, (_seq, p) in enumerate(d["scores"]):
        m = p["metrics"]
        # portable tasks record delta_vs_declared_baseline; official records keep the
        # official key. Read both so one report renders either kind.
        dl = p.get("delta_vs_official_baseline", p.get("delta_vs_declared_baseline"))
        if dl is None:
            dl = m["primary"] - baseline
        n_sd = dl / sd if sd else 0.0
        is_best = top is not None and abs(m["primary"] - top) < 1e-12 and p.get("split") == "valid"
        flags = ", ".join(p.get("flags") or []) or "&ndash;"
        rows.append(
            f"<tr{' class=best' if is_best else ''}><td class=n>{i}</td>"
            f"<td>{esc(p.get('split','?'))}</td>"
            f"<td class=mono>{esc(str(p.get('config_hash',''))[:10])}</td>"
            f"<td class=n>{_num(m.get('GAUC'))}</td>"
            f"<td class=n>{_num(m.get('nDCG@5'))}</td>"
            f"<td class='n strong'>{_num(m.get('primary'))}</td>"
            f"<td class='n {'good' if dl > 0 else ('bad' if dl < 0 else '')}'>{dl:+.5f}</td>"
            f"<td class=n>{n_sd:+.1f}</td>"
            f"<td class=n>{p.get('seed','&ndash;')}</td>"
            f"<td class=n>{p.get('n_users','&ndash;')}</td>"
            f"<td class=mono>{esc(str(p.get('scores_sha256',''))[:10])}</td>"
            f"<td>{flags}</td></tr>")
    return _table(["#", "split", "config", "GAUC", "nDCG@5", "primary",
                   "&Delta; baseline", "in sd", "seed", "users", "scores sha", "flags"],
                  rows, "no scores")


def _sec_iterations(d: dict) -> str:
    """Pre-registration is the point of this table: each rationale was countersigned at
    a sequence number BELOW the score it predicts, so `predicted` cannot have been
    written after `actual` was known."""
    rows, errs = [], []
    for seq, b in d["proposals"]:
        rat = b.get("rationale") or {}
        exp = rat.get("expected_magnitude")
        # SIGN THE PREDICTION. `expected_magnitude` is a magnitude; the direction lives in
        # `expected_direction`, which this never read. With `err = abs(abs(exp) - abs(act))`
        # an exactly BACKWARDS prediction scored 0.00000 -- predicted +0.005, actual -0.005,
        # error zero -- and that number is published as the page's "mean absolute
        # calibration error". It could not distinguish a perfect prediction from a
        # maximally inverted one, which is the one thing a calibration figure exists to do.
        if exp is not None and rat.get("expected_direction") == "decrease":
            exp = -abs(exp)
        par = b.get("parent_valid_primary")
        cand = d["by_cfg"].get(b.get("candidate_config_hash"), {}).get("metrics", {}).get("primary")
        act = None if (par is None or cand is None) else cand - par
        err = None if (exp is None or act is None) else abs(exp - act)
        disp = b.get("disposition", "&ndash;")
        if disp == "evaluated" and err is not None:
            errs.append(err)
        cls = {"evaluated": "", "duplicate": "muted",
               "policy_refused": "bad", "schema_rejected": "bad"}.get(disp, "muted")
        rows.append(
            f"<tr><td class=n>{b.get('iteration','&ndash;')}</td><td class=n>{seq}</td>"
            f"<td>{esc(b.get('action_type',''))}</td>"
            f"<td class='mono wrap'>{esc(json.dumps(b.get('params', {})))[:90]}</td>"
            f"<td class='{cls}'>{esc(disp)}</td>"
            f"<td class=n>{'' if exp is None else f'{exp:+.5f}'}</td>"
            f"<td class='n {'good' if (act or 0) > 0 else ('bad' if (act or 0) < 0 else '')}'>"
            f"{'&ndash;' if act is None else f'{act:+.5f}'}</td>"
            f"<td class='n strong'>{'&ndash;' if err is None else f'{err:.5f}'}</td>"
            f"<td class=wrap>{esc(str(rat.get('mechanism',''))[:400])}</td></tr>")
    mean_err = sum(errs) / len(errs) if errs else None
    note = ("<div class='note'>Each rationale was countersigned by the scorer at the "
            "<span class=mono>seq</span> shown, which is <b>below</b> the score it "
            "predicts &mdash; so <span class=mono>predicted</span> could not have been "
            "written after the result was known. <span class=mono>actual</span> is the "
            "candidate&rsquo;s primary minus its <b>parent&rsquo;s</b>, which is the "
            "quantity the proposer was asked to predict."
            + (f" Mean absolute calibration error over {len(errs)} evaluated proposals: "
               f"<b>{mean_err:.5f}</b>." if mean_err is not None else "") + "</div>")
    return _table(["it", "seq", "action", "params", "disposition", "predicted",
                   "actual", "err", "mechanism"], rows,
                  "no proposals recorded") + note


def _sec_actions(d: dict) -> str:
    summ = {}
    for b in d["run_end"]:
        for k, v in (b.get("action_summary") or {}).items():
            cur = summ.setdefault(k, {"tried": 0, "improved": 0, "duplicate": 0})
            for f in cur:
                cur[f] += int(v.get(f, 0) or 0)
    if not summ:
        return "<p class=muted>no action summary recorded (the run did not reach run_end)</p>"
    items = sorted(((k, v["tried"], v["improved"]) for k, v in summ.items()),
                   key=lambda t: -t[1])
    rows = [f"<tr><td>{esc(k)}</td><td class=n>{v['tried']}</td>"
            f"<td class='n good'>{v['improved']}</td>"
            f"<td class='n muted'>{v['duplicate']}</td></tr>"
            for k, v in sorted(summ.items(), key=lambda t: -t[1]["tried"])]
    used = len(summ)
    return ("<figure>" + CH.bars(items, aria="proposals per action type, improvements "
                                             "highlighted")
            + f"<figcaption>Actions the agent actually reached for. It used "
              f"<b>{used}</b> distinct action types; the highlighted portion of each bar "
              f"is the proposals that improved the incumbent.</figcaption></figure>"
            + _table(["action", "tried", "improved", "duplicate"], rows, "none"))


def _sec_inspections(d: dict) -> str:
    if not d["inspections"]:
        return ("<div class='note'>The agent requested <b>no diagnostics</b> in this run. "
                "Inspect actions cost no evaluation and no fit, so this is a choice the "
                "proposer made, not a budget it ran out of &mdash; and on the evidence so "
                "far, the weaker model tier never makes it.</div>")
    rows = []
    for b in d["inspections"]:
        okk = b.get("what") == "inspect_result"
        res = b.get("result", b.get("error", ""))
        rows.append(f"<tr><td>{esc(b.get('diagnostic','?'))}</td>"
                    f"<td class='{'good' if okk else 'bad'}'>"
                    f"{'answered' if okk else 'failed'}</td>"
                    f"<td class='mono wrap'>{esc(json.dumps(res, default=str))[:600]}</td></tr>")
    return (_table(["diagnostic", "status", "finding"], rows, "none")
            + "<div class='note'>A diagnostic is a <b>query, not a step</b>: it changes no "
              "configuration, consumes no evaluation, and does not touch the convergence "
              "window &mdash; that window measures score progress, and an inspection "
              "produces no score. Findings are fed back into every subsequent prompt. No "
              "diagnostic can read the held-out split; that is asserted before each one "
              "runs, because a diagnostic that could would be a metric the agent could "
              "query directly.</div>")


def _sec_failures(d: dict) -> str:
    if not d["failures"]:
        return ""
    rows = []
    for b in d["failures"]:
        rows.append(f"<tr><td>{esc(b.get('what'))}</td>"
                    f"<td class=mono>{esc(str(b.get('patch_hash') or b.get('diagnostic') or ''))[:16]}</td>"
                    f"<td>{esc(b.get('error_type') or '')}</td>"
                    f"<td class='mono wrap'>{esc(str(b.get('error') or ''))[:300]}</td>"
                    f"<td>{esc(b.get('isolation') or '')}</td></tr>")
    return ("<h2>Refusals and failures</h2>"
            + _table(["what", "id", "type", "detail", "isolation"], rows, "none")
            + "<div class='note'>A rejected patch or a refused proposal is the harness "
              "working. Each one is recorded rather than retried silently, so the count "
              "of things the agent got wrong is auditable rather than lost.</div>")


def _runs(d: dict) -> list:
    """(run_start, run_end) pairs matched by RUN_ID, not by position.

    Ledger order is COMPLETION order, so position is not identity. Measured on a real
    two-run artifact (runs/ledgers/review_probe.jsonl): run_start[0] is seed 0 while
    run_end[0] belongs to seed 1, which finished first because it had fewer iterations. The
    old code paired [0] with [0] and printed seed 0's settings beside seed 1's termination
    and best config, while the headline card showed seed 0's best -- a page describing a
    run that never existed.
    """
    ends = {b.get("run_id"): b for b in d["run_end"] if b.get("run_id")}
    out = []
    unmatched = [b for b in d["run_end"] if not b.get("run_id")]
    for rs in d["run_start"]:
        rid = rs.get("run_id")
        out.append((rs, ends.get(rid) or (unmatched.pop(0) if unmatched and rid is None
                                          else {})))
    if not d["run_start"]:                 # run_end without a run_start (killed mid-start)
        out = [({}, b) for b in d["run_end"]]
    return out


def _headline_run(d: dict, best_hash: str | None) -> tuple:
    """The (start, end) pair that produced the score shown in the headline card, so the
    configuration table describes the run the reader is looking at."""
    rr = _runs(d)
    if not rr:
        return {}, {}
    if best_hash:
        for rs, re_ in rr:
            if re_.get("best_config_hash") == best_hash:
                return rs, re_
    return rr[-1]


def _sec_chain(gen: dict) -> str:
    """Did the scorer that wrote this ledger verify the chain before appending to it?

    Surfaced because it is the property every other number on this page depends on, and
    because a ledger written by a scorer started with `--allow-broken-chain` looks exactly
    like one that was not. A genesis predating the field is reported as UNKNOWN rather than
    as a pass -- "no answer" and "yes" are the two things this must never conflate.
    """
    v = gen.get("chain_verified_on_open")
    if v is True:
        return ("<div class='note'>The scorer <b>verified this chain before appending to "
                "it</b>, so nothing below was signed on top of a splice.</div>")
    if v is False:
        return ("<div class='note warn'>This ledger was written by a scorer started with "
                "<span class=mono>--allow-broken-chain</span>: it appended <b>without "
                "verifying the chain first</b>. Everything signed after a splice is "
                "unverifiable, so treat every number on this page as unevidenced until "
                "<span class=mono>core.scorer verify</span> says otherwise.</div>")
    return ("<div class='note'>This genesis predates the "
            "<span class=mono>chain_verified_on_open</span> field, so whether the chain "
            "was verified at open is <b>unknown</b> &mdash; not confirmed.</div>")


def _sec_contradictions(d: dict) -> str:
    """Claims in this chain that the chain itself contradicts.

    The evaluations table already shows flags in a cell, but a claim the scorer measured to
    be impossible is a question about whether the artifact can be read as evidence, and
    that is what a reader comes to the Integrity section for. Surfaced here rather than
    left to be noticed among eleven rows of numbers.
    """
    hits = {}
    for _seq, p in d["scores"]:
        for f in p.get("flags") or []:
            if f.startswith("isolation_claimed_"):
                hits.setdefault(f, 0)
                hits[f] += 1
    if not hits:
        return ""
    rows = "".join(
        f"<tr><td class=mono>{esc(f)}</td><td class=n>{n}</td></tr>"
        for f, n in sorted(hits.items(), key=lambda kv: -kv[1]))
    return ("<div class='note warn'><b>This chain contradicts itself.</b> One or more "
            "scores claim a sandbox that the genesis entry records as unavailable or "
            "unknown on this machine. The scores themselves are real measurements of real "
            "prediction files &mdash; what is not established is the isolation they were "
            "produced under, and the field is called "
            "<span class=mono>isolation_claimed</span> for that reason.</div>"
            + _table(["flag", "scores"], [rows], "none"))


def _sec_run(d: dict, rs: dict | None = None, re_: dict | None = None) -> str:
    """The settings of the run that produced the headline score.

    The pair is passed in (from `_headline_run`) rather than read as [0]/[0]: see `_runs`
    for the real two-run ledger where those two indices belong to different runs.
    """
    rr = _runs(d)
    if rs is None or re_ is None:
        rs, re_ = (rr[-1] if rr else ({}, {}))
    conv = rs.get("convergence_mode", "?")
    sub = rs.get("submittable")
    rows = []
    for k, v in [("proposer", rs.get("proposer")), ("model", rs.get("model")),
                 ("seed", rs.get("seed")), ("iterations requested", rs.get("iterations_requested")),
                 ("start loss", rs.get("start_loss")),
                 ("convergence mode", conv),
                 ("epsilon / N", f"{rs.get('epsilon')} / {rs.get('N')}"
                                 f" ({esc(rs.get('epsilon_source',''))})"),
                 ("patch isolation", rs.get("patch_isolation")),
                 ("proposer coverage", rs.get("proposer_coverage")),
                 ("random log allowed", rs.get("allow_random_log")),
                 ("termination", re_.get("termination") or re_.get("reason")),
                 ("population size", re_.get("population_size")),
                 ("best config", str(re_.get("best_config_hash") or "")[:16]),
                 ("final incumbent", str(re_.get("final_incumbent_config_hash") or "")[:16])]:
        if v is None or v == "":
            continue
        rows.append(f"<tr><td>{esc(k)}</td><td class=mono>{esc(v)}</td></tr>")
    warn = ""
    if conv == "off":
        warn = ("<div class='note warn'>Convergence was <b>disabled</b> for this run, so "
                "it is a diagnostic, not a submission. The organisers' stopping rule "
                "(&epsilon;=0.002 over N=3) is what a graded run must use; it is turned "
                "off here only to measure what the agent would have found with more "
                "budget.</div>")
    elif sub is False:
        warn = "<div class='note warn'>This run is marked not submittable.</div>"

    # A ledger can hold SEVERAL runs, and the charts, cards and action table aggregate all
    # of them while this table describes exactly one. Say which, and say it loudly if any
    # OTHER run in the same chain was a diagnostic -- otherwise a submittable-looking page
    # can be carrying a convergence-off run's headline number.
    if len(rr) > 1:
        others = [(a, b) for a, b in rr if a is not rs]
        diag = [a.get("run_id") for a, _b in others if a.get("convergence_mode") == "off"
                or a.get("submittable") is False]
        multi = (f"<div class='note{' warn' if diag else ''}'>This ledger contains "
                 f"<b>{len(rr)} runs</b>. The charts, cards and action table aggregate all "
                 f"of them; the table above describes only "
                 f"<span class=mono>{esc(rs.get('run_id') or 'the last run')}</span>, the "
                 f"one that produced the headline score. Ledger order is completion order, "
                 f"so a run's start and end entries are matched by "
                 f"<span class=mono>run_id</span>, not position.")
        if diag:
            multi += (" <b>Another run in this chain was a diagnostic</b> ("
                      + ", ".join(f"<span class=mono>{esc(str(x))}</span>" for x in diag)
                      + "), so numbers aggregated across the chain are not submittable.")
        warn += multi + "</div>"
    return _table(["setting", "value"], rows, "no run_start recorded") + warn


# --------------------------------------------------------------------- build
def build(entries: list[dict]) -> str:
    d = extract(entries)
    baseline, sd = _reference(d)
    gen = d["genesis"]
    man = gen.get("manifest") or {}
    valid = [p["metrics"]["primary"] for _s, p in d["scores"] if p.get("split") == "valid"]
    best = max(valid) if valid else None

    # The headline best is the max over EVERY valid score in the chain, so identify the run
    # that owns it from the scores rather than from run_end[0] -- which in a multi-run
    # ledger is simply whichever run finished first.
    best_hash = None
    if valid:
        for _s, p in d["scores"]:
            if p.get("split") == "valid" and abs(p["metrics"]["primary"] - best) < 1e-12:
                best_hash = p.get("config_hash")
                break
    if not best_hash:
        for b in d["run_end"]:
            if b.get("best_config_hash"):
                best_hash = b["best_config_hash"]
                break
    hl_start, re_ = _headline_run(d, best_hash)
    delta = None if best is None else best - baseline
    # Tokens live under run_end.body["stats"], NOT at the top level. Reading them at the
    # top level meant this card rendered an em-dash on every real ledger ever produced --
    # and total tokens is a Feasibility rubric item, so the one number the rubric asks for
    # was the one the report never showed. My own test "confirmed" the old code because its
    # fixture put tokens at the top level, a shape no producer writes: the same
    # unreachable-fixture mistake as bug 48, made again.
    # Thinking tokens are included because the driver's own summary line counts them, and
    # two different totals for one run is worse than either.
    def _tok(b: dict) -> int:
        src = b.get("stats") if isinstance(b.get("stats"), dict) else b
        return sum(int(src.get(k, 0) or 0)
                   for k in ("tokens_in", "tokens_out", "tokens_thinking"))

    # Distinguish "no run_end recorded" (genuinely unknown) from "zero tokens" (a real
    # measurement, and the zero-token control arm's whole point). Rendering a measured 0 as
    # an em-dash says "unknown" about the most quotable number a random-arm run produces.
    tok = sum(_tok(b) for b in d["run_end"]) if d["run_end"] else None
    dups = sum(1 for _s, b in d["proposals"] if b.get("disposition") == "duplicate")

    splits = man.get("splits") or {}
    split_line = " &middot; ".join(
        f"{k} {v.get('rows', 0):,}" for k, v in splits.items()) or "splits not recorded"

    cards = "".join([
        _card("evaluations", str(len(d["scores"])), "scorer-signed"),
        _card("best valid primary", _num(best), f"official {baseline:.4f}"),
        _card("&Delta; vs baseline",
              "&ndash;" if delta is None else f"{delta:+.5f}",
              "&ndash;" if delta is None else f"{delta / sd:+.1f} seed sd"),
        _card("proposals", str(len(d["proposals"])), f"{dups} duplicate"),
        _card("distinct actions used",
              str(len({b.get('action_type') for _s, b in d['proposals'] if b.get('action_type')}))),
        _card("diagnostics run", str(len(d["inspections"]))),
        _card("tokens", "&ndash;" if tok is None else f"{tok:,}",
              "in + out + thinking" if tok else "zero-token control arm" if tok == 0
              else "in + out + thinking"),
        _card("ledger entries", str(len(entries))),
    ])

    parts = [
        "<title>Official Loop Report</title>", f"<style>{CSS}</style>", "<main>",
        "<h1>Autonomous MLE loop &mdash; KuaiRand-Pure within-user ranking</h1>",
        f"<p class=sub>Task <span class=mono>{esc(man.get('task') or gen.get('task') or '?')}</span> "
        f"&middot; label <span class=mono>{esc(man.get('label') or '?')}</span> "
        f"&middot; {esc(man.get('metric') or 'primary')}</p>",
        f"<p class=sub>{split_line} &middot; manifest "
        f"<span class=mono>{esc(str(gen.get('manifest_sha256',''))[:16])}</span> "
        f"&middot; organisers&rsquo; <span class=mono>evaluate.py</span> sha256 "
        f"<span class=mono>{esc(str(man.get('official_evaluate_sha256',''))[:16])}</span>, "
        f"vendored byte-identical and never reimplemented.</p>",
        f"<div class=grid>{cards}</div>",
        "<h2>Search trajectory</h2>", _sec_trajectory(d, baseline, sd),
        "<h2>Training curves</h2>", _sec_curves(d, best_hash),
        "<h2>Self-reported vs scorer-measured</h2>", _sec_crosscheck(d),
        "<h2>Evaluations</h2>", _sec_evaluations(d, baseline, sd),
        "<h2>Pre-registered proposals</h2>", _sec_iterations(d),
        "<h2>Action space coverage</h2>", _sec_actions(d),
        "<h2>Diagnostics the agent asked for</h2>", _sec_inspections(d),
        _sec_failures(d),
        "<h2>Run configuration</h2>", _sec_run(d, hl_start, re_),
        "<h2>Integrity</h2>",
        _sec_chain(gen),
        _sec_contradictions(d),
        "<div class='note'>This page reads ledger payloads; it does <b>not</b> verify "
        "them. Run <span class=mono>python3 -m core.scorer verify --ledger "
        "&lt;path&gt;</span> to check every signature and every hash link. Whole-entry "
        "truncation at the tail is not detectable from the chain alone and needs an "
        "external anchor &mdash; the report states the entry count so a truncated file "
        "is at least visible.</div>",
        "</main>",
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="HTML report for an official-task ledger")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--out", default=str(ROOT / "runs" / "official_report.html"))
    a = ap.parse_args()
    p = Path(a.ledger)
    if not p.exists():
        raise SystemExit(f"no such ledger: {p}")
    e = load(p)
    if not e:
        raise SystemExit(f"no entries in {p}")
    if not is_official(e):
        raise SystemExit(f"{p} is not an official-task ledger; for the superseded "
                         f"full-catalog task use: python3 -m agent.report --ledger {p}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(build(e), encoding="utf-8")
    print(f"wrote {a.out}  ({len(e)} entries)")


if __name__ == "__main__":
    main()
