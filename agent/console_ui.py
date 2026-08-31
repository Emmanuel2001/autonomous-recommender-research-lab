"""Presentation layer for the Research Console.

WHY THIS IS A SEPARATE MODULE
-----------------------------
`agent/control.py` owns the things that must not move: loopback binding, the per-start
token, origin checks, the launch validator, and the rule that this process never holds a
signing key and never writes a ledger.  Redesigning the pages inside that file would mean
editing the same lines that carry those guarantees.  So the presentation lives here and
`control.py` calls into it.  Nothing in this module reads a key, writes a ledger, or runs a
subprocess -- it turns already-recorded entries into HTML.

WHAT IT MUST NEVER DO
---------------------
* conflate "improvement over the run's starting champion" with "improvement over the task's
  declared baseline".  They are different numbers with different meanings, and the pilot is
  the case that proves it: +0.141514 over its seed while still 0.042654 BELOW the declared
  baseline.  Both are shown, always, with their own labels.
* invent a value to fill a chart or a panel.  Every empty case has an explicit message.
* describe provider summaries as chain-of-thought, or imply reasoning was captured when the
  provider returned none.
"""
from __future__ import annotations

import html
import json
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def dash(x) -> str:
    """Escaped value, or a real em dash when missing.

    `dash(x)` escapes the ampersand of the fallback, so a missing value rendered
    as the literal text "&mdash;" on the page. Escape the value; never the entity.
    """
    return esc(x) if (x is not None and str(x) != "") else "&mdash;"


def num(x, places: int = 6, sign: bool = False) -> str:
    """Fixed-width number, or an em dash. Never returns '0.000000' for missing data."""
    if x is None or x == "":
        return "&mdash;"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    return f"{v:+.{places}f}" if sign else f"{v:.{places}f}"


# ===================================================================== design system
#: A restrained research-console palette. Status colours are reserved for outcomes;
#: navigation and structure use neutral graphite with one desaturated blue accent.
CSS = """
:root{
  --bg:#111315;--rail:#17191c;--surface:#1b1e21;--surface2:#202428;
  --surface3:#252a2f;--raise:#2b3036;--line:#30353a;--line2:#41474e;
  --text:#f1f1ee;--dim:#a9adaf;--faint:#777d82;
  --accent:#6f8fb8;--accent2:#8da6c4;--accent-dim:#252f3a;
  --proposed:#91979c;--selected:#c99a54;--running:#7e9fbb;
  --promoted:#70a382;--rejected:#b27b72;--failed:#c36f68;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --rail-w:252px;--content-w:1480px;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;max-width:100%;scroll-behavior:smooth}
body{margin:0;min-height:100vh;background:var(--bg);color:var(--text);font-family:var(--sans);
  font-size:15px;line-height:1.55;overflow-x:hidden}
a{color:#9db6d2;text-decoration-color:rgba(157,182,210,.45);text-underline-offset:3px}
a:hover{color:#c5d3e2}
a:focus-visible,button:focus-visible,summary:focus-visible,[role=tab]:focus-visible,
input:focus-visible,select:focus-visible{outline:2px solid var(--accent2);outline-offset:2px}
.mono{font-family:var(--mono);font-size:.92em}.dim{color:var(--dim)}.faint{color:var(--faint)}
h1,h2,h3{line-height:1.16;margin:0;letter-spacing:-.025em}
h1{font-size:clamp(1.55rem,2.5vw,2.2rem)}
h2{font-size:1.2rem;margin:30px 0 13px}h3{font-size:1rem;margin:18px 0 8px}
p{margin:0 0 10px;max-width:76ch}.sub{color:var(--dim);font-size:.9rem;max-width:76ch}
.eyebrow{margin:0 0 7px;color:var(--faint);font-size:.69rem;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase}

/* ---- product shell ---- */
.app-shell{display:grid;grid-template-columns:var(--rail-w) minmax(0,1fr);min-height:100vh}
.rail{position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:24px;
  padding:24px 17px 18px;background:var(--rail);
  border-right:1px solid var(--line);z-index:20}
.brand{display:grid;grid-template-columns:44px minmax(0,1fr);gap:12px;align-items:center}
.brandmark{width:44px;height:44px;border-radius:7px;display:grid;place-items:center;
  font:700 1rem var(--mono);letter-spacing:-.03em;color:var(--text);
  background:var(--surface2);border:1px solid var(--line2)}
.brandcopy b{display:block;font-size:.94rem;line-height:1.25;letter-spacing:-.02em}
.brandcopy span{display:block;color:var(--faint);font-size:.7rem;margin-top:3px}
.rail-label{font-size:.63rem;font-weight:750;letter-spacing:.14em;text-transform:uppercase;
  color:var(--faint);padding:0 10px;margin-bottom:6px}
.tabs{display:flex;flex-direction:column;gap:5px}
[role=tab]{width:100%;display:grid;grid-template-columns:28px minmax(0,1fr);gap:10px;
  align-items:center;text-align:left;background:transparent;border:1px solid transparent;
  color:var(--dim);font:inherit;padding:10px 11px;border-radius:6px;cursor:pointer;position:relative}
[role=tab] .navicon{height:27px;width:27px;display:grid;place-items:center;border-radius:8px;
  background:rgba(255,255,255,.035);font-family:var(--mono);font-size:.72rem;color:var(--faint)}
[role=tab] .navtext{min-width:0;font-size:.86rem;font-weight:650;white-space:nowrap}
[role=tab]:hover{background:rgba(255,255,255,.035);color:var(--text)}
[role=tab][aria-selected=true]{background:var(--surface2);color:var(--text);
  border-color:var(--line2);box-shadow:inset 2px 0 var(--accent)}
[role=tab][aria-selected=true] .navicon{background:var(--accent-dim);color:#b8c8da}
[role=tabpanel][hidden]{display:none}
.railtrust{margin-top:auto;padding:13px;border:1px solid var(--line);border-radius:7px;
  background:rgba(255,255,255,.025);font-size:.74rem;color:var(--dim)}
.railtrust .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--promoted);
  margin-right:7px}
.railtrust b{display:block;color:var(--text);font-size:.78rem;margin-bottom:5px}
.workspace{min-width:0}
.commandbar{min-height:84px;display:flex;justify-content:space-between;gap:24px;align-items:center;
  padding:20px clamp(20px,3vw,44px);border-bottom:1px solid var(--line);
  background:rgba(17,19,21,.96);position:sticky;top:0;z-index:15}
.commandbar .titleblock{min-width:0}.commandbar h1{font-size:1.38rem}
.commandbar .titleblock p{margin:4px 0 0;color:var(--dim);font-size:.82rem}
.commandbadges{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.trustpill{border:1px solid var(--line2);background:var(--surface);color:var(--dim);
  border-radius:5px;padding:5px 9px;font-size:.68rem;letter-spacing:.02em;white-space:nowrap}
.trustpill strong{color:var(--text)}
.wrap{max-width:var(--content-w);width:100%;margin:0 auto;padding:22px clamp(18px,3vw,44px) 72px}
.ledgerdock{padding:16px;background:var(--surface);border:1px solid var(--line);
  border-radius:8px;margin-bottom:28px}
.viewintro{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin:4px 0 18px}
.viewintro h2{font-size:clamp(1.45rem,2.2vw,2rem);margin:0}.viewintro .sub{margin:7px 0 0}

/* ---- run facts and ledger context ---- */
.strip{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);
  border-radius:7px;overflow:hidden;margin:0 0 10px;
  grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}
.strip .cell{background:var(--surface2);padding:12px 14px;min-width:0}
.strip .k{font-size:.63rem;font-weight:750;letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.strip .v{font-size:.91rem;margin-top:4px;overflow-wrap:anywhere}.strip .v .badge{white-space:normal}
.ledgerdock .strip{margin-bottom:12px}.ledgerdock .presets{margin-bottom:0;align-items:flex-end}

/* ---- hero score journey ---- */
.scorejourney{display:grid;grid-template-columns:1fr 54px 1fr 54px 1fr;gap:10px;
  align-items:stretch;margin:14px 0 18px}
.scorepoint{padding:18px;border:1px solid var(--line);border-radius:7px;background:var(--surface)}
.scorepoint.current{border-color:rgba(112,163,130,.58);box-shadow:inset 0 2px var(--promoted)}
.scorepoint.target{border-color:rgba(195,111,104,.44)}
.scorepoint .k{color:var(--faint);font-size:.65rem;font-weight:750;letter-spacing:.11em;text-transform:uppercase}
.scorepoint .v{font:700 clamp(1.45rem,3vw,2.25rem)/1 var(--mono);margin:11px 0 6px}
.scorepoint .n{color:var(--dim);font-size:.78rem}.scorearrow{display:grid;place-items:center;color:var(--faint);font-family:var(--mono)}

/* ---- metric cards ---- */
.metrics{display:grid;gap:12px;margin:14px 0 8px;grid-template-columns:repeat(4,minmax(0,1fr))}
.metric{position:relative;overflow:hidden;background:var(--surface);
  border:1px solid var(--line);border-radius:7px;padding:16px;min-width:0;min-height:132px}
.metric:before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:var(--line2)}
.metric.good:before{background:var(--promoted)}.metric.bad:before{background:var(--failed)}
.metric.ref:before{background:var(--accent)}.metric .k{font-size:.65rem;font-weight:750;
  letter-spacing:.09em;text-transform:uppercase;color:var(--faint)}
.metric .v{font:700 1.45rem/1.1 var(--mono);margin:13px 0 8px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.metric .v.pos{color:var(--promoted)}.metric .v.neg{color:var(--failed)}
.metric .n{font-size:.76rem;color:var(--dim);line-height:1.4}

/* ---- notes and badges ---- */
.note{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--line2);
  border-radius:6px;padding:12px 14px;margin:12px 0;font-size:.86rem}
.note.info{border-left-color:var(--accent)}.note.warn{border-left-color:var(--selected)}
.note.bad{border-left-color:var(--failed)}.note.ok{border-left-color:var(--promoted)}
.empty{background:var(--surface);border:1px dashed var(--line2);border-radius:7px;padding:24px;color:var(--dim)}
.badge{display:inline-block;font-size:.64rem;font-weight:750;letter-spacing:.07em;text-transform:uppercase;
  padding:2px 7px;border-radius:4px;border:1px solid currentColor;white-space:nowrap}
.b-proposed{color:var(--proposed)}.b-selected{color:var(--selected)}.b-running{color:var(--running)}
.b-promoted{color:var(--promoted)}.b-rejected{color:var(--rejected)}.b-failed{color:var(--failed)}
.b-signed{color:var(--promoted)}.b-llm{color:var(--accent)}

/* ---- research mission view ---- */
.flowwrap{overflow:visible;padding-bottom:5px}.flow{display:flex;min-width:0;gap:6px;align-items:center;flex-wrap:wrap;margin:10px 0 20px}
.flow .step{background:var(--surface2);border:1px solid var(--line);border-radius:5px;padding:6px 10px;
  font-size:.72rem;color:var(--dim);white-space:nowrap}.flow .step.on{color:var(--text);border-color:var(--line2)}
.flow .arrow{color:var(--faint);font-size:.72rem}.researchlayout{display:grid;grid-template-columns:minmax(0,1fr) minmax(330px,420px);gap:20px;align-items:start}
.researchmain{min-width:0}.brainpanel{position:sticky;top:108px;background:var(--surface);
  border:1px solid var(--line2);border-radius:8px;overflow:hidden}
.brainhead{padding:18px;border-bottom:1px solid var(--line);background:var(--surface2)}
.brainhead .recordmark{width:35px;height:35px;border-radius:6px;display:grid;place-items:center;margin-bottom:13px;
  color:#b8c8da;font:700 .7rem var(--mono);background:var(--accent-dim);border:1px solid #394b5e}
.brainhead h2{margin:0;font-size:1.15rem}.brainhead p{color:var(--dim);font-size:.78rem;margin:6px 0 0}
.brain{display:block}.brain .row{padding:12px 17px;border-bottom:1px solid rgba(52,68,95,.55)}
.brain .row:last-child{border-bottom:0}.brain .row .k{font-size:.61rem;font-weight:750;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-bottom:4px}
.brain .row .v{font-size:.84rem;overflow-wrap:anywhere}.cands{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr));align-items:start}
.cand{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:15px;min-width:0}
.cand.sel{grid-column:1/-1;border-color:rgba(201,154,84,.62);box-shadow:inset 3px 0 var(--selected)}
.cand.rej{border-color:rgba(178,123,114,.42)}.cand .hd{display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap}
.cand .rank{font-family:var(--mono);color:var(--faint);font-size:.76rem}.cand .lane{font-size:.64rem;font-weight:750;letter-spacing:.06em;text-transform:uppercase;color:#9db6d2;margin-top:11px}
.cand .hyp{margin:7px 0 10px;font-size:.86rem;line-height:1.5}.scores{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}
.scores div{background:rgba(255,255,255,.035);border:1px solid rgba(52,68,95,.55);border-radius:8px;padding:6px 7px;min-width:0}
.scores .k{font-size:.57rem;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)}.scores .v{font-family:var(--mono);font-size:.82rem}
.why{font-size:.78rem;color:var(--dim);margin-top:10px;border-top:1px solid var(--line);padding-top:9px}

/* ---- tables and charts ---- */
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:7px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.82rem}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:.62rem;font-weight:750;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);background:var(--surface2);white-space:nowrap}
tbody tr:hover{background:rgba(255,255,255,.025)}tbody tr:last-child td{border-bottom:none}td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.pos{color:var(--promoted)}td.neg{color:var(--failed)}td.wide{min-width:22ch}.charts{display:grid;gap:14px;grid-template-columns:repeat(2,minmax(0,1fr))}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:16px}
figure svg{display:block;width:100%;height:auto}figcaption{font-size:.76rem;color:var(--dim);margin-top:8px}

/* ---- disclosures and forms ---- */
details.adv{margin:14px 0;background:var(--surface);border:1px solid var(--line);border-radius:10px}
details.adv>summary{cursor:pointer;padding:10px 13px;font-size:.8rem;color:var(--dim)}details.adv[open]>summary{border-bottom:1px solid var(--line);color:var(--text)}
details.adv pre{margin:0;padding:13px;overflow-x:auto;font-family:var(--mono);font-size:.75rem;line-height:1.5;color:var(--dim)}
details.grp{background:var(--surface);border:1px solid var(--line);border-radius:12px;margin:0 0 12px}details.grp>summary{cursor:pointer;padding:12px 15px;font-weight:650;font-size:.88rem}
details.grp[open]>summary{border-bottom:1px solid var(--line)}.fields{display:grid;gap:12px;padding:15px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
label.f{display:flex;flex-direction:column;gap:5px;font-size:.65rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--faint);min-width:0}
label.f small{text-transform:none;letter-spacing:0;font-weight:400;font-size:.72rem;color:var(--dim)}input[type=text],input[type=number],select{background:#0b111c;color:var(--text);border:1px solid var(--line2);border-radius:8px;padding:8px 10px;font:inherit;font-size:.84rem;min-width:0;width:100%;max-width:100%}
select{text-overflow:ellipsis}.presets label.f{flex:1 1 240px}.presets{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}label.chk{display:flex;gap:8px;align-items:flex-start;font-size:.82rem;text-transform:none;letter-spacing:0;color:var(--text)}
.btn{background:#607fa6;color:white;border:1px solid #7693b5;border-radius:6px;padding:9px 15px;font:inherit;font-weight:700;cursor:pointer}
.btn.danger{background:var(--failed);color:#1b0707}.btn.ghost{background:var(--surface);color:var(--text);border:1px solid var(--line2);box-shadow:none}

@media (max-width:1150px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.researchlayout{grid-template-columns:1fr}.brainpanel{position:static}.cands{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:900px){.app-shell{display:block}.rail{position:static;height:auto;padding:14px 16px;gap:14px;border-right:0;border-bottom:1px solid var(--line)}
  .brand{grid-template-columns:38px minmax(0,1fr)}.brandmark{width:38px;height:38px}.rail-label,.railtrust{display:none}.tabs{display:grid;grid-template-columns:repeat(5,minmax(108px,1fr));overflow-x:auto;padding-bottom:3px}
  [role=tab]{grid-template-columns:22px minmax(0,1fr);padding:8px}[role=tab] .navicon{width:22px;height:22px}.commandbar{position:static;min-height:0;padding:16px 18px}.wrap{padding:18px 18px 60px}}
@media (max-width:640px){.commandbar{align-items:flex-start}.commandbadges{display:none}.ledgerdock{padding:12px;border-radius:8px}.strip,.metrics,.cands,.charts,.fields{grid-template-columns:1fr}
  .scorejourney{grid-template-columns:1fr}.scorearrow{height:22px;transform:rotate(90deg)}.cand.sel{grid-column:auto}.brain .row{padding:11px 14px}.viewintro{display:block}.wrap{padding:15px 14px 50px}
  .metric{min-height:112px}.presets label.f{flex:1 1 100%}.tablewrap{-webkit-overflow-scrolling:touch}
  .tabs{grid-template-columns:repeat(3,minmax(0,1fr));overflow:visible}.navtext{white-space:normal!important;font-size:.74rem;line-height:1.15}
  [role=tab]{grid-template-columns:22px minmax(0,1fr);gap:6px;min-height:42px}.ledgerdock .strip{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

TABS = [("overview", "Overview"), ("live", "Live Research"),
        ("experiments", "Experiments"), ("integrity", "Artifacts &amp; Integrity"),
        ("newrun", "New Run")]
NAV_ICONS = {"overview": "01", "live": "02", "experiments": "03",
             "integrity": "04", "newrun": "+"}

JS = """
(function(){
  var tabs=[].slice.call(document.querySelectorAll('[role=tab]'));
  function show(id){
    tabs.forEach(function(t){
      var on=t.dataset.view===id;
      t.setAttribute('aria-selected',on?'true':'false');
      t.tabIndex=on?0:-1;
      var p=document.getElementById('panel-'+t.dataset.view);
      if(p) p.hidden=!on;
    });
    try{location.hash='#'+id}catch(e){}
  }
  tabs.forEach(function(t,i){
    t.addEventListener('click',function(){show(t.dataset.view)});
    t.addEventListener('keydown',function(e){
      var d=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;
      if(!d)return; e.preventDefault();
      var n=tabs[(i+d+tabs.length)%tabs.length]; n.focus(); show(n.dataset.view);
    });
  });
  var want=(location.hash||'').replace('#','');
  show(tabs.some(function(t){return t.dataset.view===want})?want:tabs[0].dataset.view);
  // Presets fill the launch form; they never submit it.
  var P={
    smoke:{iterations:'1',convergence:'off',tournament_size:'6',max_tokens:'120000',
           max_wall_seconds:'900',research_policy:'tournament',framework:'auto',
           device:'auto',accelerator_policy:'prefer'},
    mps:{iterations:'6',convergence:'off',tournament_size:'6',max_tokens:'400000',
         max_wall_seconds:'3600',research_policy:'tournament',framework:'torch',
         device:'mps',accelerator_policy:'prefer',seed_profile:'accelerated'},
    official:{iterations:'50',convergence:'official',tournament_size:'6',
              max_wall_seconds:'21600',research_policy:'tournament',framework:'auto',
              device:'auto',accelerator_policy:'prefer',seed_profile:'inherited'}
  };
  [].slice.call(document.querySelectorAll('[data-preset]')).forEach(function(b){
    b.addEventListener('click',function(){
      var n0=document.getElementById('presetnote');
      var p=P[b.dataset.preset];
      if(!p){ if(n0) n0.textContent='Custom: fields left exactly as they are. '+
        'Open each section and set what you need.'; return; }
      Object.keys(p).forEach(function(k){
        var el=document.querySelector('[name='+k+']'); if(el) el.value=p[k];
      });
      var n=document.getElementById('presetnote');
      if(n) n.textContent='Preset applied: '+b.textContent.trim()+
        '. Review every section before launching \\u2014 nothing has been submitted.';
    });
  });
})();
"""


# ===================================================================== digest
def digest(entries: list[dict], run_name: str = "") -> dict:
    """Recorded entries -> one presentation model. Reads; never writes.

    Missing values stay None so the views can render an explicit empty state instead of a
    plausible-looking zero.
    """
    d: dict = {"name": run_name, "start": {}, "end": {}, "scores": [], "tournaments": [],
               "proposals": [], "iterations": [], "seed": {}, "entries": len(entries),
               "max_seq": None, "summaries": []}
    for e in entries:
        p = e.get("payload") or {}
        b = p.get("body") or {}
        d["max_seq"] = e.get("seq")
        kind, what = p.get("kind"), b.get("what")
        if kind == "run_start":
            d["start"] = b
        elif kind == "run_end":
            d["end"] = b
        elif kind == "score":
            d["scores"].append({"seq": e.get("seq"), "primary": (p.get("metrics") or {}).get("primary"),
                                "metrics": p.get("metrics") or {}, "tree": p.get("tree"),
                                "split": p.get("split"),
                                "scores_sha256": p.get("scores_sha256"),
                                "delta_declared": p.get("delta_vs_declared_baseline",
                                                        p.get("delta_vs_official_baseline")),
                                "flags": p.get("flags") or []})
        elif kind == "proposal":
            d["proposals"].append({"seq": e.get("seq"), **b})
        elif what == "tournament":
            d["tournaments"].append({"seq": e.get("seq"), **b})
        elif what == "iteration_result":
            d["iterations"].append({"seq": e.get("seq"), **b})
        elif what == "seed_baseline_reproduction":
            d["seed"] = b
        elif what == "llm_thought_summary":
            d["summaries"].append({"seq": e.get("seq"), **b})

    seed_primary = d["seed"].get("seed_signed_primary")
    if seed_primary is None and d["scores"]:
        seed_primary = d["scores"][0]["primary"]
    champ = (d["end"].get("champion") or {}).get("primary")
    if champ is None:
        promoted = [i for i in d["iterations"] if i.get("outcome") == "promoted"]
        champ = promoted[-1].get("dev_primary") if promoted else seed_primary
    declared = d["seed"].get("published_baseline_valid")
    if declared is None:
        declared = ((d["start"].get("baseline") or {}) or {}).get("valid")

    def sub(a, b_):
        return None if (a is None or b_ is None) else float(a) - float(b_)

    d["m"] = {"seed": seed_primary, "champion": champ,
              "gain_vs_seed": sub(champ, seed_primary),
              "declared": declared,
              "gain_vs_declared": sub(champ, declared),
              "evaluated": (d["end"].get("stats") or {}).get("evaluated"),
              "promoted": (d["end"].get("stats") or {}).get("promoted"),
              "termination": d["end"].get("termination"),
              "iterations": (d["end"].get("budget") or {}).get("iterations"),
              "max_iterations": (d["end"].get("budget") or {}).get("max_iterations")
                                or d["start"].get("max_iterations")}
    usage = d["end"].get("usage_by_role") or {}
    d["usage"] = usage
    tok = sum(int((v or {}).get("total_tokens") or 0) for v in usage.values())
    if not tok:
        tok = sum(int((i.get("cost") or {}).get("tokens_total") or 0) for i in d["iterations"])
    d["m"]["tokens"] = tok or None
    walls = (d["end"].get("budget") or {}).get("iteration_wall_seconds") or []
    if not walls:
        walls = [(i.get("cost") or {}).get("wall_seconds") for i in d["iterations"]]
    walls = [w for w in walls if w]
    d["m"]["wall"] = round(sum(walls), 2) if walls else None

    ep = d["start"].get("execution_profile") or {}
    d["exec"] = {"framework": ep.get("framework"), "requested": ep.get("device"),
                 "policy": ep.get("accelerator_policy"),
                 "fallback_allowed": ep.get("allow_fallback"),
                 "resolved": None, "resolved_source": None}
    # The RESOLVED device is harness-owned state, not a ledger field. Read it if it is on
    # disk and say where it came from; otherwise leave it as an explicit empty state rather
    # than echoing the request back as though it were the outcome.
    if run_name:
        sp = ROOT / "runs" / "candidates" / Path(run_name).stem.replace(".jsonl", "") / "state.json"
        try:
            st = json.loads(sp.read_text())
            ex = [x.get("execution") for x in (st.get("experiments") or []) if x.get("execution")]
            if ex:
                d["exec"]["resolved"] = ex[-1].get("device")
                d["exec"]["fallback_used"] = ex[-1].get("fallback_used")
                d["exec"]["resolved_source"] = str(sp.relative_to(ROOT))
        except (OSError, ValueError, KeyError):
            pass
    return d


def _fmt_secs(s) -> str:
    if s is None:
        return "&mdash;"
    s = float(s)
    return f"{s:.1f}s" if s < 90 else f"{s/60:.1f} min"


# ===================================================================== 1. overview
def view_overview(d: dict, run_state: str, scorer_ready: bool) -> str:
    m, ex = d["m"], d["exec"]
    routing = d["start"].get("model_routing") or {}
    models = sorted({v for v in routing.values() if v}) or [d["start"].get("model")]
    dev = esc(ex.get("resolved") or "not recorded")
    strip = [
        ("run status", run_state),
        ("task", dash(d["start"].get("task"))),
        ("llm / model", dash(", ".join(str(x) for x in models if x))),
        ("framework", dash(ex.get("framework"))),
        ("device req &rarr; resolved",
         f"{esc(ex.get('requested') or '?')} &rarr; <b>{dev}</b>"),
        ("iteration", f"{m['iterations'] if m['iterations'] is not None else '&mdash;'}"
                      f" / {m['max_iterations'] if m['max_iterations'] is not None else '&mdash;'}"),
        ("termination", dash(m["termination"])),
    ]
    strip_html = "".join(f"<div class=cell><div class=k>{k}</div><div class=v>{v}</div></div>"
                         for k, v in strip)

    gs, gd = m["gain_vs_seed"], m["gain_vs_declared"]
    cards = [
        ("good" if (gs or 0) > 0 else "bad" if (gs or 0) < 0 else "neutral",
         "improvement over seed", num(gs, sign=True),
         "champion minus seed. NOT a comparison with the task baseline.",
         "pos" if (gs or 0) > 0 else "neg" if (gs or 0) < 0 else ""),
        ("good" if (gd or 0) > 0 else "bad" if (gd or 0) < 0 else "neutral",
         "vs declared baseline", num(gd, sign=True),
         "champion minus declared baseline. This is the number that says whether the "
         "task was beaten.",
         "pos" if (gd or 0) > 0 else "neg" if (gd or 0) < 0 else ""),
        ("neutral", "evaluations used",
         f"{m['evaluated']}" if m["evaluated"] is not None else "&mdash;",
         "candidates actually fitted and scored", ""),
        ("neutral", "tokens", f"{m['tokens']:,}" if m["tokens"] else "&mdash;",
         "all roles combined", ""),
        ("neutral", "wall time", _fmt_secs(m["wall"]), "summed iteration wall clock", ""),
    ]
    cards_html = "".join(
        f"<div class='metric {cls}'><div class=k>{k}</div>"
        f"<div class='v {vc}'>{v}</div><div class=n>{n}</div></div>"
        for cls, k, v, n, vc in cards)

    verdict = ""
    if gs is not None and gd is not None:
        promoted = (m["promoted"] or 0) > 0
        verdict = (
            f"<div class='note {"warn" if gd < 0 else "ok"}'><b>Outcome.</b> "
            f"{'A candidate was promoted within this run' if promoted else 'No candidate was promoted'}"
            f" &mdash; the champion improved on its seed by "
            f"<span class=mono>{num(gs, sign=True)}</span>. "
            f"{'But the declared task baseline was NOT beaten' if gd < 0 else 'The declared task baseline was beaten'}"
            f": the champion is <span class=mono>{num(gd, sign=True)}</span> against a declared "
            f"<span class=mono>{num(m['declared'])}</span>. "
            f"These two numbers answer different questions and are never combined.</div>")

    scorer_note = (
        "<div class='note info'>A scorer is listening; it holds the signing key and this UI "
        "does not.</div>" if scorer_ready else
        "<div class='note warn'><b>No scorer is listening on the default socket.</b> This UI "
        "cannot start one &mdash; it deliberately holds no signing key. A completed run's "
        "recorded evidence below is unaffected.</div>")
    journey = (
        f"<div class=scorejourney>"
        f"<div class=scorepoint><div class=k>signed starting seed</div>"
        f"<div class=v>{num(m['seed'])}</div><div class=n>The run's starting champion, signed</div></div>"
        f"<div class=scorearrow>&rarr;</div>"
        f"<div class='scorepoint current'><div class=k>current run champion</div>"
        f"<div class=v>{num(m['champion'])}</div><div class=n>Best signed candidate in this search</div></div>"
        f"<div class=scorearrow>&rarr;</div>"
        f"<div class='scorepoint target'><div class=k>declared task baseline</div>"
        f"<div class=v>{num(m['declared'])}</div><div class=n>The figure published with the task</div></div>"
        f"</div>")
    return (f"<div class=viewintro><div><p class=eyebrow>Run overview</p>"
            f"<h2>One research run, completely traceable.</h2>"
            f"<p class=sub>Read the result first, then follow every decision back to signed evidence.</p>"
            f"</div></div>{journey}"
            f"<h2>Run facts</h2><div class=strip>{strip_html}</div>"
            f"{scorer_note}{verdict}"
            f"<h2>Research economy</h2><div class=metrics>{cards_html}</div>"
            f"<p class=sub>Every figure above is read from the signed ledger. "
            f"A single-seed promotion is a search decision, not a confirmed finding.</p>")


# ===================================================================== 2. live research
FLOW = ["Question", "6 hypotheses", "Judge ranking", "Selected experiment",
        "Files changed", "Training", "Signed result", "Promote / reject", "Reflection",
        "Next question"]

NO_SUMMARY = ("Provider returned no public thought summary for this call. The structured "
              "hypothesis, decision and reflection below remain available.")


def _cand_cards(t: dict) -> str:
    judge = t.get("judge") or t.get("verdict") or {}
    sel = judge.get("selected_id")
    ranks = {r.get("id"): r for r in (judge.get("ranking") or []) if isinstance(r, dict)}
    archv = {a.get("candidate_id"): a for a in (judge.get("architecture_verdicts") or [])}
    cands = t.get("candidates") or []
    if not cands:
        return "<div class=empty>No tournament candidates recorded for this iteration.</div>"
    ordered = sorted(cands, key=lambda c: (ranks.get(c.get("id"), {}).get("rank") or 99))
    out = []
    for c in ordered:
        cid = c.get("id")
        r = ranks.get(cid, {})
        arch = archv.get(cid)
        chosen = cid == sel
        cls = "sel" if chosen else ("rej" if arch and arch.get("verdict") == "rejected" else "")
        badge = ('<span class="badge b-selected">selected</span>' if chosen
                 else '<span class="badge b-rejected">arch rejected</span>' if cls == "rej"
                 else '<span class="badge b-proposed">considered only</span>')
        why = r.get("note") or ""
        if chosen and judge.get("why_selected"):
            why = judge["why_selected"]
        if arch and arch.get("reason"):
            why = (why + " " if why else "") + f"Architecture verdict: {arch['reason']}"
        out.append(
            f"<div class='cand {cls}'>"
            f"<div class=hd><span class=rank>#{r.get('rank') or '?'} &middot; "
            f"<span class=mono>{esc(cid)}</span></span>{badge}</div>"
            f"<div class=lane>{esc(c.get('lane'))}</div>"
            f"<div class=hyp>{esc(c.get('hypothesis'))}</div>"
            f"<div class=scores>"
            f"<div><div class=k>impact</div><div class=v>"
            f"{num(r.get('score_expected_impact'), 2)}</div></div>"
            f"<div><div class=k>evidence</div><div class=v>"
            f"{num(r.get('score_evidence'), 2)}</div></div>"
            f"<div><div class=k>risk</div><div class=v>"
            f"{num(r.get('score_implementation_risk'), 2)}</div></div>"
            f"</div>"
            + (f"<div class=why>{esc(why)}</div>" if why else "")
            + "</div>")
    return f"<div class=cands>{''.join(out)}</div>"


def view_live(d: dict) -> str:
    if not d["iterations"] and not d["tournaments"]:
        return ("<div class=empty>No research iteration has been recorded in this ledger "
                "yet. The timeline appears once a tournament is registered.</div>")
    it = d["iterations"][-1] if d["iterations"] else {}
    t = d["tournaments"][-1] if d["tournaments"] else {}
    prop = next((p for p in d["proposals"] if p.get("iteration") == it.get("iteration")), {})
    refl = it.get("reflection") or {}

    flow = []
    for i, s in enumerate(FLOW):
        flow.append(f"<span class='step on'>{esc(s)}</span>")
        if i < len(FLOW) - 1:
            flow.append("<span class=arrow>&rarr;</span>")
    outcome = it.get("outcome") or "?"
    ob = {"promoted": "b-promoted", "rejected": "b-rejected",
          "failed": "b-failed"}.get(outcome, "b-proposed")

    files = it.get("files_changed") or prop.get("files_changed") or []
    files_html = ", ".join(f"<span class=mono>{esc(f)}</span>" for f in files) or \
        "<span class=faint>none recorded</span>"

    summary_block = (
        "".join(f"<div class=row><div class=k>public summary (seq {esc(s['seq'])})</div>"
                f"<div class=v>{esc(' '.join(s.get('summaries') or []))}</div></div>"
                for s in d["summaries"])
        if d["summaries"] else
        f"<div class=row><div class=k>public thought summary</div>"
        f"<div class=v><span class=faint>{esc(NO_SUMMARY)}</span></div></div>")

    brain = [
        ("selected hypothesis", dash(prop.get("hypothesis") or it.get("hypothesis")
                                     or (t.get("judge") or {}).get("selected_id"))),
        ("why this one", dash((t.get("judge") or {}).get("why_selected"))),
        ("prediction", f"expected <span class=mono>"
                       f"{num(it.get('expected_magnitude') or prop.get('expected_magnitude'), 5, True)}"
                       f"</span> on the primary metric"),
        ("experiment plan", f"lane <b>{esc(it.get('lane'))}</b>, editing {files_html}"),
        ("observed result", f"signed <span class=mono>{num(it.get('dev_primary'))}</span> "
                            f"&middot; delta vs champion <span class=mono>"
                            f"{num(it.get('delta_vs_champion'), 6, True)}</span> "
                            f"&middot; train/dev gap <span class=mono>"
                            f"{num(it.get('gap'), 6)}</span>"),
        ("what happened", dash(refl.get("what_happened"))),
        ("why", dash(refl.get("why"))),
        ("belief update", "".join(
            f"<div>{esc(b.get('belief'))} <span class=faint>(mechanism: "
            f"{esc(b.get('mechanism'))}; supports={esc(b.get('supports'))}; "
            f"confidence={esc(b.get('confidence'))})</span></div>"
            for b in (refl.get("belief_updates") or [])) or "&mdash;"),
        ("remaining uncertainty", dash(refl.get("remaining_uncertainty"))),
        ("next question", dash(refl.get("next_question"))),
    ]
    brain_html = "".join(f"<div class=row><div class=k>{k}</div><div class=v>{v}</div></div>"
                         for k, v in brain)

    return (
        f"<div class=viewintro><div><p class=eyebrow>Live research</p>"
        f"<h2>From question to signed decision.</h2>"
        f"<p class=sub>Iteration {esc(it.get('iteration'))} &middot; lane "
        f"<b>{esc(it.get('lane'))}</b> &middot; outcome "
        f"<span class='badge {ob}'>{esc(outcome)}</span></p></div></div>"
        f"<div class=flowwrap><div class=flow>{''.join(flow)}</div></div>"
        f"<div class=researchlayout><div class=researchmain>"
        f"<h2>Tournament &mdash; {len(t.get('candidates') or [])} hypotheses, one funded</h2>"
        f"<p class=sub>Lane coverage is <b>required</b> by policy: every lane appearing is "
        f"the policy working, not evidence of unprompted initiative. The substantive agent "
        f"decision is the mechanism it proposes and the experiment the judge funds.</p>"
        f"{_cand_cards(t)}"
        f"<details class=adv><summary>Evidence drawer &mdash; raw reflection record</summary>"
        f"<pre>{esc(json.dumps(refl, indent=2, sort_keys=True))}</pre></details>"
        f"</div><aside class=brainpanel aria-label='Recorded research rationale'>"
        f"<div class=brainhead><div class=recordmark>DR</div><p class=eyebrow>Decision record</p>"
        f"<h2>Research rationale</h2>"
        f"<p>Recorded decisions and reflections only &mdash; this is not hidden chain-of-thought.</p>"
        f"</div><div class=brain>{summary_block}{brain_html}</div></aside></div>")


# ===================================================================== 3. experiments
def _chart_scores(d: dict) -> str:
    pts = []
    if d["m"]["seed"] is not None:
        pts.append((0, float(d["m"]["seed"])))
    for i in d["iterations"]:
        if i.get("dev_primary") is not None:
            pts.append((int(i.get("iteration") or len(pts)), float(i["dev_primary"])))
    if len(pts) < 2:
        return ("<figure><figcaption>Score over iterations &mdash; needs at least two "
                "recorded points. This run has "
                f"{len(pts)}.</figcaption></figure>")
    declared = d["m"]["declared"]
    ys = [p[1] for p in pts] + ([float(declared)] if declared is not None else [])
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.18 or 0.02
    lo, hi = lo - pad, hi + pad
    W, H, L, R, T, B = 460, 190, 46, 10, 12, 26
    def X(i): return L + (i / max(1, max(p[0] for p in pts))) * (W - L - R)
    def Y(v): return T + (1 - (v - lo) / (hi - lo)) * (H - T - B)
    line = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in pts)
    base = (f'<line x1="{L}" y1="{Y(float(declared)):.1f}" x2="{W-R}" '
            f'y2="{Y(float(declared)):.1f}" stroke="var(--accent)" stroke-width="1.5" '
            f'stroke-dasharray="4 4"/>'
            f'<text x="{W-R}" y="{Y(float(declared))-5:.1f}" text-anchor="end" '
            f'font-size="10" fill="var(--accent)">declared baseline '
            f'{float(declared):.4f}</text>') if declared is not None else ""
    dots = "".join(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" '
                   f'fill="var(--promoted)"/>' for i, v in pts)
    labs = "".join(f'<text x="{X(i):.1f}" y="{H-8}" text-anchor="middle" font-size="10" '
                   f'fill="var(--faint)">{i}</text>' for i, v in pts)
    return (f'<figure><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Signed primary score by iteration, against the declared task '
            f'baseline.">'
            f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="var(--line2)"/>'
            f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="var(--line2)"/>'
            f'<text x="4" y="{Y(hi-pad):.1f}" font-size="10" fill="var(--faint)">'
            f'{hi-pad:.3f}</text>'
            f'<text x="4" y="{Y(lo+pad):.1f}" font-size="10" fill="var(--faint)">'
            f'{lo+pad:.3f}</text>'
            f'{base}<polyline points="{line}" fill="none" stroke="var(--promoted)" '
            f'stroke-width="2"/>{dots}{labs}</svg>'
            f'<figcaption>Signed primary by iteration (0 = seed). Recorded values only.'
            f'</figcaption></figure>')


def _chart_budget(d: dict) -> str:
    usage = d["usage"] or {}
    rows = sorted(((r, int((v or {}).get("total_tokens") or 0)) for r, v in usage.items()),
                  key=lambda x: -x[1])
    if not rows or not any(v for _r, v in rows):
        return ("<figure><figcaption>Budget consumption &mdash; no per-role token usage "
                "was recorded in this ledger.</figcaption></figure>")
    W, H, L = 460, 26 * len(rows) + 30, 74
    mx = max(v for _r, v in rows) or 1
    bars = []
    for i, (r, v) in enumerate(rows):
        y = 12 + i * 26
        w = (v / mx) * (W - L - 62)
        bars.append(f'<text x="0" y="{y+12}" font-size="11" fill="var(--dim)">{esc(r)}</text>'
                    f'<rect x="{L}" y="{y}" width="{max(w,1):.1f}" height="15" rx="3" '
                    f'fill="var(--accent)" opacity="0.85"/>'
                    f'<text x="{L+max(w,1)+6:.1f}" y="{y+12}" font-size="10" '
                    f'fill="var(--faint)">{v:,}</text>')
    total = sum(v for _r, v in rows)
    return (f'<figure><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Tokens consumed per agent role.">{"".join(bars)}</svg>'
            f'<figcaption>Tokens by role &mdash; {total:,} total. Recorded per call, not '
            f'estimated.</figcaption></figure>')


def view_experiments(d: dict) -> str:
    if not d["iterations"]:
        return "<div class=empty>No evaluated experiment is recorded in this ledger.</div>"
    rows = []
    for i in d["iterations"]:
        prop = next((p for p in d["proposals"] if p.get("iteration") == i.get("iteration")), {})
        files = i.get("files_changed") or prop.get("files_changed") or []
        exp = i.get("expected_magnitude", prop.get("expected_magnitude"))
        act = i.get("delta_vs_champion")
        cost = i.get("cost") or {}
        out = i.get("outcome") or "?"
        ob = {"promoted": "b-promoted", "rejected": "b-rejected",
              "failed": "b-failed"}.get(out, "b-proposed")
        rows.append(
            f"<tr><td class=n>{esc(i.get('iteration'))}</td>"
            f"<td>{esc(i.get('lane'))}</td>"
            f"<td class=wide>{dash(prop.get('hypothesis'))}</td>"
            f"<td class=mono>{dash(', '.join(files))}</td>"
            f"<td class=n>{num(exp, 5, True)}</td>"
            f"<td class='n {'pos' if (act or 0) > 0 else 'neg' if (act or 0) < 0 else ''}'>"
            f"{num(act, 6, True)}</td>"
            f"<td class=n>{num(i.get('dev_primary'))}</td>"
            f"<td class=mono>{dash(d['exec'].get('resolved'))}</td>"
            f"<td class=n>{_fmt_secs(cost.get('wall_seconds'))}</td>"
            f"<td><span class='badge {ob}'>{esc(out)}</span></td></tr>")
    head = ("<tr><th>it</th><th>lane</th><th>hypothesis</th><th>files changed</th>"
            "<th>expected &Delta;</th><th>actual &Delta;</th><th>score</th><th>device</th>"
            "<th>runtime</th><th>outcome</th></tr>")
    return (f"<h2>Experiment comparison</h2>"
            f"<p class=sub>Expected against actual, per iteration. The agent's predicted "
            f"magnitude is shown next to what the scorer measured.</p>"
            f"<div class=tablewrap><table><thead>{head}</thead><tbody>"
            f"{''.join(rows)}</tbody></table></div>"
            f"<h2>Recorded charts</h2><div class=charts>"
            f"{_chart_scores(d)}{_chart_budget(d)}</div>")


# ===================================================================== 4. integrity
def view_integrity(d: dict, ledger_name: str, verify_note: str) -> str:
    sc = d["scores"]
    trees = []
    for p in d["proposals"]:
        trees.append((p.get("iteration"), p.get("tree"), p.get("files_changed") or [],
                      p.get("diff_bytes"), p.get("diff_hash")))
    rows = "".join(
        f"<tr><td class=n>{esc(i)}</td><td class=mono>{esc(str(t)[:16])}&hellip;</td>"
        f"<td class=mono>{dash(', '.join(f))}</td>"
        f"<td class=n>{esc(db)}</td>"
        f"<td class=mono>{esc(str(dh)[:12] if dh else '&mdash;')}</td></tr>"
        for i, t, f, db, dh in trees) or \
        "<tr><td colspan=5 class=dim>No candidate tree recorded.</td></tr>"

    signed = "".join(
        f"<tr><td class=n>{esc(s['seq'])}</td><td class=n>{num(s['primary'])}</td>"
        f"<td>{esc(s['split'])}</td>"
        f"<td class=mono>{esc(str(s['scores_sha256'])[:16])}&hellip;</td>"
        f"<td><span class='badge b-signed'>signed</span></td></tr>"
        for s in sc) or "<tr><td colspan=5 class=dim>No signed score recorded.</td></tr>"

    iso = d["start"].get("isolation") or "&mdash;"
    task_spec = d["start"].get("task_spec_path")
    return (
        f"<h2>Ledger verification</h2>"
        f"<div class='note {'ok' if 'OK' in verify_note else 'warn'}'>"
        f"<b>{esc(ledger_name)}</b> &mdash; {esc(verify_note)}</div>"
        f"<p class=sub>Signature checking needs the signing key, which this process must "
        f"not hold. Run <span class=mono>python3 -m core.scorer verify --ledger "
        f"runs/ledgers/{esc(ledger_name)}</span> to confirm independently.</p>"
        f"<h2>Signed scores</h2><div class=tablewrap><table><thead>"
        f"<tr><th>seq</th><th>primary</th><th>split</th><th>score vector sha256</th>"
        f"<th>state</th></tr></thead><tbody>{signed}</tbody></table></div>"
        f"<h2>Candidate trees and diffs</h2><div class=tablewrap><table><thead>"
        f"<tr><th>it</th><th>tree</th><th>files changed</th><th>diff bytes</th>"
        f"<th>diff hash</th></tr></thead><tbody>{rows}</tbody></table></div>"
        f"<h2>Boundaries</h2><div class=strip>"
        f"<div class=cell><div class=k>isolation</div><div class=v>{esc(iso)}</div></div>"
        f"<div class=cell><div class=k>held-out label access</div>"
        f"<div class=v>absent from the candidate process</div></div>"
        f"<div class=cell><div class=k>task specification</div>"
        f"<div class=v class=mono>{esc(task_spec or 'registered task')}</div></div>"
        f"<div class=cell><div class=k>entries in chain</div>"
        f"<div class=v>{d['entries']} (max seq {esc(d['max_seq'])})</div></div>"
        f"</div>"
        f"<p class=sub>Reproduction: the full HTML report and the event replay for this "
        f"ledger are linked at the top of this page. Candidate source is stored by tree "
        f"hash in the artifact store.</p>")


# ===================================================================== 5. new run
def _sel(name, options, cur=None, hint="") -> str:
    opts = "".join(f"<option value='{esc(o)}'{' selected' if o == cur else ''}>{esc(o)}</option>"
                   for o in options)
    return (f"<label class=f>{esc(name.replace('_', ' '))}"
            + (f"<small>{hint}</small>" if hint else "")
            + f"<select name={esc(name)}>{opts}</select></label>")


def _txt(name, value="", hint="", kind="text") -> str:
    return (f"<label class=f>{esc(name.replace('_', ' '))}"
            + (f"<small>{hint}</small>" if hint else "")
            + f"<input type={kind} name={esc(name)} value='{esc(value)}'></label>")


def _chk(name, label, checked=False) -> str:
    return (f"<label class='f chk'><input type=checkbox name={esc(name)} value=1"
            f"{' checked' if checked else ''}> <span>{esc(label)}</span></label>")


def view_newrun(token: str) -> str:
    groups = [
        ("Task", True, [
            _sel("driver", ["research", "official"], "research",
                 "research = code-first loop"),
            _txt("task_spec", "", "optional repo JSON for a portable task"),
            _sel("seed_profile", ["inherited", "minimal", "accelerated"], "inherited",
                 "what the starting code already contains"),
            _txt("seed_config", "", "JSON overrides for the seed config"),
        ]),
        ("Research strategy", True, [
            _sel("research_policy", ["tournament", "single"], "tournament"),
            _txt("tournament_size", "6", "minimum 6 &mdash; one per lane", "number"),
            _sel("convergence", ["official", "off"], "official",
                 "official = the organisers' stopping rule"),
            _txt("iterations", "50", "hard cap", "number"),
        ]),
        ("Models and roles", False, [
            _sel("backend", ["gemini", "ollama", "groq", "anthropic"], "gemini"),
            _txt("model", "gemini-3.5-flash-lite", "default for every role"),
            _txt("planner_model", "", "falls back to model"),
            _txt("judge_model", "", "falls back to model"),
            _txt("coder_model", "", "falls back to model"),
            _txt("reflector_model", "", "falls back to model"),
            _sel("thinking_level", ["", "low", "high"], ""),
            _chk("thought_summaries", "Request provider public thought summaries", True),
        ]),
        ("Training hardware", False, [
            _sel("framework", ["auto", "numpy", "torch"], "auto"),
            _sel("device", ["auto", "cpu", "mps", "cuda"], "auto"),
            _sel("accelerator_policy", ["prefer", "allow", "off", "require"], "prefer"),
            _chk("no_device_fallback", "Refuse the run if the requested device is unavailable"),
        ]),
        ("Budgets", False, [
            _txt("max_tokens", "", "hard token ceiling", "number"),
            _txt("max_wall_seconds", "21600", "wall-clock cap", "number"),
            _txt("max_seconds", "", "per-fit cap", "number"),
            _txt("max_repairs", "1", "code repair attempts", "number"),
        ]),
        ("Isolation", False, [
            _sel("patch_isolation", ["seatbelt", "subprocess", "docker", "none"], "seatbelt",
                 "seatbelt is the measured boundary on macOS"),
            _txt("patch_seconds", "", "candidate timeout", "number"),
        ]),
        ("Advanced settings", False, [
            _sel("proposer", ["random", "llm"], "random",
                 "typed-action arm only"),
            _txt("seed", "0", "SEARCH seed", "number"),
            _txt("model_seed", "0", "MODEL seed &mdash; changes candidate identity", "number"),
            _txt("max_coverage_repairs", "1", "lane-coverage repair calls", "number"),
            _txt("start_loss", "", "typed-action arm only"),
            _txt("tag", "", "label recorded with the run"),
            _chk("allow_random_log", "Include the quarantined random-log split (rarely correct)"),
        ]),
    ]
    body = []
    for title, open_, fields in groups:
        body.append(
            f"<details class=grp{' open' if open_ else ''}><summary>{esc(title)}</summary>"
            f"<div class=fields>{''.join(fields)}</div></details>")
    return (
        f"<h2>Start a run</h2>"
        f"<p class=sub>This form only builds a command line; the validator in "
        f"<span class=mono>agent/control_launch.py</span> refuses anything out of range, and "
        f"the scorer &mdash; not this UI &mdash; owns the signing key and the ledger.</p>"
        f"<div class=presets>"
        f"<button type=button class='btn ghost' data-preset=smoke>Quick smoke test</button>"
        f"<button type=button class='btn ghost' data-preset=mps>Local MPS research</button>"
        f"<button type=button class='btn ghost' data-preset=official>Official compliant run</button>"
        f"<button type=button class='btn ghost' data-preset=custom>Custom</button></div>"
        f"<p class=sub id=presetnote>Presets fill the fields below. Nothing is submitted "
        f"until you press Launch.</p>"
        f"<form method=post action='/launch?t={esc(token)}'>{''.join(body)}"
        f"<button class=btn>Launch run</button></form>")


# ===================================================================== shell
def page(*, token: str, run_state: str, scorer_ready: bool, d: dict,
         ledger_name: str, verify_note: str, banner: str = "",
         links: str = "", runs_table: str = "") -> str:
    tabs = "".join(
        f"<button role=tab data-view={v} id=tab-{v} aria-controls=panel-{v} "
        f"aria-selected=false tabindex=-1><span class=navicon aria-hidden=true>"
        f"{NAV_ICONS[v]}</span><span class=navtext>{label}</span></button>"
        for v, label in TABS)
    panels = [
        ("overview", view_overview(d, run_state, scorer_ready) + runs_table),
        ("live", view_live(d)),
        ("experiments", view_experiments(d)),
        ("integrity", view_integrity(d, ledger_name, verify_note)),
        ("newrun", view_newrun(token)),
    ]
    panel_html = "".join(
        f"<section role=tabpanel id=panel-{v} aria-labelledby=tab-{v} tabindex=0 hidden>"
        f"{h}</section>" for v, h in panels)
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Autonomous Recommender Research Lab</title>"
        f"<style>{CSS}</style></head><body><div class=app-shell>"
        f"<aside class=rail><div class=brand><div class=brandmark>R</div>"
        f"<div class=brandcopy><b>Research Console</b><span>Recommender experiments</span>"
        f"</div></div><div><div class=rail-label>Workspace</div>"
        f"<nav class=tabs role=tablist aria-label='Console views'>{tabs}</nav></div>"
        f"<div class=railtrust><b><span class=dot></span>Local evidence boundary</b>"
        f"Loopback only. This console never holds the scorer's signing key.</div></aside>"
        f"<main class=workspace><header class=commandbar><div class=titleblock>"
        f"<p class=eyebrow>Recommender systems research</p>"
        f"<h1>Research Console</h1>"
        f"<p>Recorded experiments, decisions and signed results.</p>"
        f"</div><div class=commandbadges>"
        f"<span class=trustpill><strong>LOCAL</strong> &middot; loopback</span>"
        f"<span class=trustpill><strong>SIGNED</strong> &middot; scorer-owned</span>"
        f"</div></header><div class=wrap>{banner}"
        f"<div class=ledgerdock>{links}</div>{panel_html}</div></main></div>"
        f"<script>{JS}</script></body></html>")
