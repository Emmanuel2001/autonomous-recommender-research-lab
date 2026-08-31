"""Local control plane: browse ledgers, read reports, launch and stop runs.

    python3 -m agent.control            # prints a URL with a token; open it

Stdlib only -- no Flask, no Streamlit, no npm. That is not minimalism for its own sake:
this has to still work on a fresh machine the night before a deadline, and every
dependency is a way for that not to happen.

WHAT THIS PROCESS CANNOT DO, and why it is built that way:

  * It holds NO SIGNING KEY. Only core.scorer does. So a UI-launched run cannot mint or
    alter a ledger entry -- it can only ask the scorer, over the socket, to sign one.
    The UI therefore cannot start a scorer either; it tells the operator to, which is
    the honest consequence of not being trusted with the key.
  * It NEVER OPENS A LEDGER FOR WRITING. The separately started scorer owns and reports
    the ledger path; there is no code path from an HTTP request to a chain's bytes.
  * It BINDS 127.0.0.1 ONLY, requires a per-start token on every request, and checks the
    Host and Origin headers. Localhost is not a security boundary on its own: any page in
    the operator's browser can POST to 127.0.0.1, so a bare local server that launches
    processes is reachable by any tab they have open.
  * It runs ONE CHILD AT A TIME and can only stop the child it started.

The launch rules themselves live in agent/control_launch.py, which has no networking in
it and is spec'd by tests/test_control.py.
"""
from __future__ import annotations

import argparse
import hmac
import html
import json
import os
import secrets
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import console_ui as UI
from agent import control_launch as CL
from agent.official_report import CSS, build as build_report, is_official, load as load_ledger

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 64 * 1024                      # a launch form is a few hundred bytes
LOG_TAIL = 16_000                         # bytes of run output shown in the browser

TOKEN = ""
RUNNER = CL.Runner()
_CAP_CACHE: tuple[float, dict] | None = None


def capabilities() -> dict:
    """Measured hardware for the setup page, cached because importing torch is expensive."""
    global _CAP_CACHE
    now = time.time()
    if _CAP_CACHE is None or now - _CAP_CACHE[0] > 60:
        from core.execution import public_capabilities
        _CAP_CACHE = (now, public_capabilities())
    return dict(_CAP_CACHE[1])


# ------------------------------------------------------------------ ledger index
def ledger_index() -> list[dict]:
    """Summarise every ledger without trusting any of them to be well-formed."""
    out = []
    for p in sorted(CL.LEDGER_DIR.glob("*.jsonl"), key=lambda q: -q.stat().st_mtime):
        row = {"name": p.name, "bytes": p.stat().st_size,
               "mtime": p.stat().st_mtime, "entries": 0, "official": False,
               "best": None, "proposer": None, "model": None, "termination": None,
               "arm": None, "research": False}
        try:
            e = load_ledger(p)
            row["entries"] = len(e)
            row["official"] = is_official(e)
            for x in e:
                b = (x.get("payload") or {}).get("body") or {}
                k = (x.get("payload") or {}).get("kind")
                if k == "run_start":
                    row["proposer"] = b.get("proposer") or b.get("backend")
                    row["model"] = b.get("model")
                    row["arm"] = ("code-first" if b.get("driver") == "research_driver"
                                  else "typed")
                    row["research"] = b.get("driver") == "research_driver"
                elif k == "run_end":
                    row["termination"] = b.get("termination") or b.get("reason")
                    if b.get("best_valid_primary") is not None:
                        row["best"] = b["best_valid_primary"]
                elif k == "score" and (x["payload"].get("split") == "valid"):
                    pm = (x["payload"].get("metrics") or {}).get("primary")
                    if pm is not None and (row["best"] is None or pm > row["best"]):
                        row["best"] = pm
        except Exception as exc:                                   # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
    return out


def resolve_ledger(name: str) -> Path:
    """Name -> path, by MEMBERSHIP in the real directory listing rather than by joining
    a string. `..` and absolute paths cannot survive a set-membership test, so traversal
    is impossible by construction instead of by sanitising."""
    known = {p.name: p for p in CL.LEDGER_DIR.glob("*.jsonl")}
    if name not in known:
        raise KeyError(name)
    return known[name]


def research_events(entries: list[dict], after: int = -1) -> list[dict]:
    """Ledger records -> safe, compact decision-trace events for live UI and replay.

    This is intentionally interpretation-light.  Signed scores are labelled signed;
    hypotheses, judge rationales and reflections are labelled LLM-authored.  The UI must
    not turn commentary into evidence merely by drawing it in the same colour.
    """
    out = []
    for x in entries:
        seq = x.get("seq")
        if not isinstance(seq, int) or seq <= after:
            continue
        payload = x.get("payload") or {}
        body = payload.get("body") or {}
        kind = payload.get("kind") or ""
        what = body.get("what") or kind
        event = {"seq": seq, "iteration": body.get("iteration"),
                 "kind": str(what or "event"), "signed": False,
                 "origin": "harness", "title": "Event", "summary": "", "detail": {}}
        if kind == "run_start":
            event.update(title="Run started", summary=(
                f"{body.get('research_policy', 'research')} policy · "
                f"{body.get('seed_profile', 'unknown')} seed · "
                f"{body.get('isolation', 'unknown')} isolation"))
            event["detail"] = {
                "model routing": body.get("model_routing"),
                "execution": body.get("execution_profile"),
                "hardware": body.get("hardware_capabilities")}
        elif what == "tournament":
            cs = body.get("candidates") or []
            chosen = body.get("selected_id")
            event.update(origin="LLM decision",
                         title=f"Hypothesis tournament · {len(cs)} considered",
                         summary=f"Judge selected {chosen}; one candidate will be evaluated")
            event["detail"] = {
                "candidates": [{k: c.get(k) for k in
                                ("id", "lane", "hypothesis", "files_to_change")}
                               for c in cs],
                "judge": body.get("judge"), "selected": chosen}
        elif kind == "proposal" and what == "candidate_proposal":
            event.update(origin="LLM hypothesis · harness-registered",
                         title="Candidate pre-registered",
                         summary=str(body.get("hypothesis") or "")[:400])
            event["detail"] = {"tree": body.get("tree"),
                               "files changed": body.get("files_changed"),
                               "expected magnitude": body.get("expected_magnitude"),
                               "diff bytes": body.get("diff_bytes"),
                               "execution request": (body.get("config") or {}).get("execution")}
        elif kind in ("score", "confirmation", "final_validation") and payload.get("metrics"):
            m = payload.get("metrics") or {}
            primary = m.get("primary")
            event.update(origin="scorer", title="Scorer-signed result", signed=True,
                         summary=(f"primary {float(primary):.6f}" if
                                  isinstance(primary, (int, float)) else str(m)))
            event["detail"] = {"metrics": m,
                               "delta vs baseline": (payload.get("delta_vs_official_baseline")
                                                    or payload.get("delta_vs_declared_baseline")),
                               "flags": payload.get("flags"),
                               "tree": payload.get("config_hash")}
        elif what in ("candidate_failed", "scorer_refused", "metric_disagreement"):
            event.update(title=str(what).replace("_", " ").title(),
                         summary=str(body.get("error") or body.get("error_type") or "")[:400])
            event["detail"] = {k: body.get(k) for k in
                               ("tree", "error_type", "error_frame", "execution", "isolation")}
        elif what == "llm_thought_summary":
            summaries = body.get("summaries") or []
            event.update(origin="provider public summary",
                         title=f"{str(body.get('role') or 'LLM').title()} thought summary",
                         summary=str(summaries[0] if summaries else "")[:600])
            event["detail"] = {"role": body.get("role"), "stage": body.get("stage"),
                               "model": body.get("model"),
                               "thinking tokens": body.get("thinking_tokens"),
                               "public summaries": summaries,
                               "disclosure": body.get("note")}
        elif what in ("iteration_result", "experiment_result", "reflection", "belief_update"):
            event.update(origin="LLM reflection · harness outcome",
                         title=str(what).replace("_", " ").title(),
                         summary=str(body.get("outcome") or body.get("why") or "")[:400])
            event["detail"] = body
        elif kind == "run_end":
            champ = body.get("champion") or {}
            event.update(title="Run complete",
                         summary=(f"{body.get('termination', 'complete')} · champion "
                                  f"{str(champ.get('tree') or '')[:12]} · "
                                  f"primary {champ.get('primary')}"))
            event["detail"] = {"stats": body.get("stats"),
                               "usage by role": body.get("usage_by_role"),
                               "beliefs": body.get("beliefs")}
        else:
            # Keep only events useful to understanding the research loop. Infrastructure
            # heartbeats remain in the full report and signed ledger.
            if not what:
                continue
            event.update(title=str(what).replace("_", " ").title(),
                         summary=str(body.get("reason") or body.get("error") or "")[:400])
            event["detail"] = {k: v for k, v in body.items()
                               if k not in ("diff", "source")}
        out.append(event)
    return out


def timeline_html(events: list[dict]) -> str:
    if not events:
        return "<div class='note muted'>No research events recorded yet.</div>"
    # A multi-seed confirmation appends two signed scores per seed. Forty nearly identical
    # cards pushed the actual tournament/judge/reflection off the live page—the exact
    # opposite of a decision trace. Collapse only consecutive CONFIRMATION entries; replay
    # and the ledger keep every signed record individually.
    compact = []
    for e in events:
        if e.get("kind") == "confirmation":
            if compact and compact[-1].get("kind") == "confirmation_batch":
                batch = compact[-1]
                batch["detail"]["count"] += 1
                batch["detail"]["last seq"] = e["seq"]
                batch["seq"] = e["seq"]
                batch["title"] = f"{batch['detail']['count']} signed confirmation scores"
                batch["summary"] = ("Collapsed in this live trace; replay and ledger retain "
                                    "every signed score")
            else:
                compact.append({"seq": e["seq"], "iteration": e.get("iteration"),
                                "kind": "confirmation_batch", "signed": True,
                                "origin": "scorer",
                                "title": "1 signed confirmation score",
                                "summary": ("Collapsed in this live trace; replay and ledger "
                                            "retain every signed score"),
                                "detail": {"count": 1, "first seq": e["seq"],
                                           "last seq": e["seq"]}})
        else:
            compact.append(e)
    cards = []
    for e in compact[-18:]:
        detail = json.dumps(e.get("detail") or {}, indent=1, default=str)[:2400]
        badge = "signed scorer evidence" if e.get("signed") else e.get("origin", "harness")
        cards.append(
            f"<article class='event {'signed' if e.get('signed') else ''}'>"
            f"<div class=eventmeta>seq {e['seq']} &middot; {html.escape(badge)}"
            + (f" &middot; iteration {e['iteration']}" if e.get("iteration") is not None else "")
            + f"</div><h3>{html.escape(e['title'])}</h3>"
              f"<p>{html.escape(e.get('summary') or '')}</p>"
              f"<details><summary>structured detail</summary><pre>"
              f"{html.escape(detail)}</pre></details></article>")
    return "<div class=timeline>" + "".join(cards) + "</div>"


def replay_page(name: str, events: list[dict]) -> str:
    data = json.dumps(events, ensure_ascii=False, default=str).replace("</", "<\\/")
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>Research replay · {html.escape(name)}</title><style>{CSS}
.event{{border-left:4px solid var(--accent);padding:12px 16px;margin:12px 0;background:var(--card)}}
.event.signed{{border-color:var(--good)}} .eventmeta{{color:var(--mut);font-size:12px}}
pre{{white-space:pre-wrap;overflow:auto}} .controls{{display:flex;gap:8px;align-items:center}}
</style></head><body><main><p><a href='/?t={TOKEN}'>&larr; dashboard</a></p>
<h1>Signed research replay</h1><p class=sub>{html.escape(name)} · recorded evidence, not a new run</p>
<div class=controls><button id=play>Replay</button><button id=all>Show all</button>
<span id=count></span></div><div id=events></div></main>
<script>const EVENTS={data}; const root=document.getElementById('events'); let timer=null;
function card(e){{const a=document.createElement('article');a.className='event'+(e.signed?' signed':'');
 const m=document.createElement('div');m.className='eventmeta';m.textContent=`seq ${{e.seq}} · ${{e.signed?'signed scorer evidence':(e.origin||'harness')}}${{e.iteration==null?'':` · iteration ${{e.iteration}}`}}`;
 const h=document.createElement('h3');h.textContent=e.title;const p=document.createElement('p');p.textContent=e.summary||'';
 const d=document.createElement('details');const s=document.createElement('summary');s.textContent='structured detail';
 const pre=document.createElement('pre');pre.textContent=JSON.stringify(e.detail||{{}},null,2);d.append(s,pre);a.append(m,h,p,d);root.append(a);}}
function showAll(){{clearInterval(timer);root.textContent='';EVENTS.forEach(card);document.getElementById('count').textContent=`${{EVENTS.length}} events`;}}
function replay(){{clearInterval(timer);root.textContent='';let i=0;document.getElementById('count').textContent='';timer=setInterval(()=>{{if(i>=EVENTS.length){{clearInterval(timer);return}}card(EVENTS[i++]);document.getElementById('count').textContent=`${{i}} / ${{EVENTS.length}}`;window.scrollTo(0,document.body.scrollHeight)}},650)}}
document.getElementById('play').onclick=replay;document.getElementById('all').onclick=showAll;showAll();</script>
</body></html>"""


# ------------------------------------------------------------------ page
def _opt(vals, cur) -> str:
    return "".join(f"<option value='{html.escape(v)}'"
                   f"{' selected' if v == cur else ''}>{html.escape(v)}</option>"
                   for v in vals)


def _run_state(st: dict, chosen: str | None, chosen_is_research: bool) -> str:
    """Honest run state.

    The old page said "No run has been started from this UI." whenever RUNNER was idle --
    which is true and useless: a run launched from a terminal, with its evidence sitting in
    a ledger this page is displaying, was reported as though nothing had happened. An
    externally launched run is OBSERVED, not absent.
    """
    if st["state"] == "running":
        return (f"<span class='badge b-running'>running</span> pid {st['pid']} &middot; "
                f"{st['seconds']:.0f}s")
    if st["state"] == "finished":
        return (f"<span class='badge b-promoted'>finished</span> exit "
                f"{st['returncode']} after {st['seconds']:.0f}s")
    if chosen and chosen_is_research:
        return ("<span class='badge b-promoted'>observed external run</span> "
                "launched outside this UI; recorded evidence below")
    if chosen:
        return "<span class='badge b-proposed'>observed ledger</span> not launched here"
    return "<span class='badge b-proposed'>no run started</span>"


def dashboard(msg: str = "", err: str = "", want: str = "") -> str:
    """The five-view console. Presentation lives in agent/console_ui.py; this function
    only decides WHICH recorded ledger to show and hands it over."""
    st = RUNNER.status()
    scorer = RUNNER.scorer_ready()
    index = ledger_index()

    # Prefer the ledger of a run started here; otherwise the most recent research ledger.
    chosen = None
    if want and want in {r["name"] for r in index}:
        chosen = want
    lv = st.get("ledger")
    if chosen is None and lv:
        cand = Path(str(lv)).name
        if cand in {r["name"] for r in index}:
            chosen = cand
    if want and want in {r["name"] for r in index}:
        chosen = want                       # explicit operator choice wins
    if chosen is None:
        chosen = next((r["name"] for r in index if r.get("research")), None)
    if chosen is None:
        chosen = next((r["name"] for r in index if r["entries"]), None)
    row = next((r for r in index if r["name"] == chosen), {})

    banner = ""
    if err:
        banner = f"<div class='note bad'><b>Refused.</b> {html.escape(err)}</div>"
    elif msg:
        banner = f"<div class='note info'>{html.escape(msg)}</div>"
    if st["state"] == "running":
        banner += (f"<form method=post action='/stop?t={TOKEN}'>"
                   f"<button class='btn danger'>Stop this run</button></form>")

    # Ledger switcher + report/replay. A ledger gets these links because it HAS ENTRIES --
    # not because it satisfies is_official(). The portable pilot is is_official=False and
    # is_research=True, and gating on `official` labelled it "legacy" and withheld both
    # links from a completed run.
    opts = []
    for r in index:
        kind = ("code-first research" if r.get("research")
                else "typed-action" if r["entries"] else "empty")
        sel = " selected" if r["name"] == chosen else ""
        short = "research" if r.get("research") else ("typed" if r["entries"] else "empty")
        opts.append(f"<option value='{html.escape(r['name'])}'{sel}>"
                    f"{html.escape(r['name'])} &middot; {r['entries']}e &middot; "
                    f"{short}</option>")
    links = ""
    if chosen:
        q = urllib.parse.quote(chosen)
        links = (f"<div class=strip>"
                 f"<div class=cell><div class=k>showing</div>"
                 f"<div class='v mono'>{html.escape(chosen)}</div></div>"
                 f"<div class=cell><div class=k>classification</div><div class=v>"
                 f"{'code-first research' if row.get('research') else 'typed-action'}"
                 f"</div></div>"
                 f"<div class=cell><div class=k>full report</div>"
                 f"<div class=v><a href='/report?ledger={q}&t={TOKEN}'>open report</a></div></div>"
                 f"<div class=cell><div class=k>event replay</div>"
                 f"<div class=v><a href='/replay?ledger={q}&t={TOKEN}'>open replay</a></div></div>"
                 f"</div>"
                 f"<form method=get action='/' class=presets>"
                 f"<input type=hidden name=t value='{TOKEN}'>"
                 f"<label class=f>switch ledger<select name=ledger>{''.join(opts)}</select>"
                 f"</label><button class='btn ghost'>Show</button></form>")

    # ---- runs on disk -------------------------------------------------------------
    # The redesign initially dropped this table, which also dropped the only place an
    # UNREADABLE ledger was surfaced -- a ledger that trips ledger_index's except used to be
    # rendered with whatever partial numbers had been gathered and no indication at all.
    # Restored with the same columns, and the same colspan arithmetic: the header has 9
    # columns, so an error row drawing 4 cells of its own spans the remaining 5.
    trows = []
    for r in index:
        age = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["mtime"]))
        q = urllib.parse.quote(r["name"])
        if r.get("error"):
            trows.append(
                f"<tr><td class=mono>{html.escape(r['name'])}</td>"
                f"<td class=n>!</td>"
                f"<td colspan=5>unreadable: {html.escape(str(r['error'])[:140])} &mdash; any "
                f"numbers for this ledger would be partial, so none are shown</td>"
                f"<td class=dim>{age}</td><td>&ndash;</td></tr>")
            continue
        try:
            best_txt = "&mdash;" if r["best"] is None else format(float(r["best"]), ".6f")
        except (TypeError, ValueError):
            best_txt = f"<span class=bad>?{html.escape(str(r['best'])[:12])}</span>"
        link = (f"<a href='/report?ledger={q}&t={TOKEN}'>report</a>"
                + (f" &middot; <a href='/replay?ledger={q}&t={TOKEN}'>replay</a>"
                   if r.get("research") else "")) if r["entries"] else \
               "<span class=dim>empty</span>"
        trows.append(
            f"<tr><td class=mono>{html.escape(r['name'])}</td>"
            f"<td class=n>{r['entries']}</td>"
            f"<td class=n>{best_txt}</td>"
            f"<td>{'code-first' if r.get('research') else html.escape(str(r.get('arm') or '&ndash;'))}</td>"
            f"<td>{html.escape(str(r['proposer'] or '&ndash;'))}</td>"
            f"<td class=mono>{html.escape(str(r['model'] or ''))}</td>"
            f"<td class=dim>{html.escape(str(r['termination'] or ''))}</td>"
            f"<td class=dim>{age}</td><td>{link}</td></tr>")
    runs_table = ("<h2>Runs on disk</h2><div class=tablewrap><table><thead><tr>"
                  "<th>ledger</th><th>entries</th><th>best signed</th><th>arm</th>"
                  "<th>proposer</th><th>model</th><th>termination</th><th>updated</th>"
                  "<th>links</th></tr></thead><tbody>"
                  + ("".join(trows) or "<tr><td colspan=9 class=dim>no ledgers yet</td></tr>")
                  + "</tbody></table></div>")

    if not chosen:
        return UI.page(token=TOKEN, run_state=_run_state(st, None, False),
                       scorer_ready=scorer, d=UI.digest([]), ledger_name="(none)",
                       verify_note="no ledger present", banner=banner, links=links,
                       runs_table=runs_table)

    entries = load_ledger(resolve_ledger(chosen))
    d = UI.digest(entries, run_name=chosen)
    verify_note = (f"{len(entries)} entries loaded; hash ordering and artifact hashes are "
                   f"recomputed in the full report. Signature checking needs the signing "
                   f"key, which this process must not hold.")
    return UI.page(token=TOKEN, run_state=_run_state(st, chosen, bool(row.get("research"))),
                   scorer_ready=scorer, d=d, ledger_name=chosen,
                   verify_note=verify_note, banner=banner, links=links,
                   runs_table=runs_table)


class Handler(BaseHTTPRequestHandler):
    server_version = "track2-control"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):                     # one tidy line, not two
        print(f"  [ui] {self.address_string()} {fmt % a}")

    # -- helpers
    def _send(self, code: int, body: bytes, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, code, s):
        self._send(code, s.encode("utf-8"))

    def _redirect(self, msg="", err=""):
        q = {"t": TOKEN}
        if msg:
            q["msg"] = msg
        if err:
            q["err"] = err
        self.send_response(303)
        self.send_header("Location", "/?" + urllib.parse.urlencode(q))
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _host_ok(self) -> bool:
        """DNS rebinding: a hostname that resolves to 127.0.0.1 lets a remote page talk
        to this server with the browser's cooperation. Pinning Host to a literal
        loopback address is the cheap, standard defence."""
        host = (self.headers.get("Host") or "").split(":")[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1")

    LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")

    def _origin_why(self) -> tuple[bool, str]:
        """Is this state-changing request same-origin? Returns (ok, reason).

        `Sec-Fetch-Site` is checked FIRST because it is the header that actually answers the
        question: browsers set it on every request, script cannot forge it, and
        `same-origin` is unambiguous. Origin alone was not enough -- the first version
        trusted `urlparse(Origin).hostname`, which refused a real form POST from the
        dashboard because a browser can send `Origin: null` (an opaque origin, whose
        hostname is None). My test only ever sent a synthetic `Origin: http://127.0.0.1:PORT`,
        a shape I invented, so it passed while the real browser was being turned away --
        the same unreachable-fixture mistake as bugs 48 and 58.

        The reason string is returned so the 403 page can say WHICH header refused it. A
        security check that fails without saying why gets disabled by whoever hits it.
        """
        sfs = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if sfs:
            # `none` is a user-initiated navigation (typing a URL, a bookmark); `same-origin`
            # is our own page. Anything else is another site talking to us.
            if sfs in ("same-origin", "none"):
                return True, f"Sec-Fetch-Site: {sfs}"
            return False, f"Sec-Fetch-Site: {sfs}"

        o = self.headers.get("Origin")
        if not o or o.strip().lower() == "null":
            # No Origin, or an opaque one, and no Sec-Fetch-Site to arbitrate. The token is
            # already required and unguessable, and Host is pinned to loopback, so this is
            # allowed rather than refusing a legitimate old-browser form post.
            return True, f"no Sec-Fetch-Site, Origin={o!r}"
        try:
            host = urllib.parse.urlparse(o).hostname
        except ValueError:
            return False, f"unparseable Origin {o!r}"
        if host in self.LOOPBACK:
            return True, f"Origin host {host}"
        return False, f"Origin host {host!r} is not loopback"

    def _origin_ok(self) -> bool:
        return self._origin_why()[0]

    def _auth(self, q: dict) -> bool:
        """Constant-time token check, on BYTES.

        `hmac.compare_digest` on `str` raises TypeError for any non-ASCII character, and
        both channels can carry one: `parse_qs("t=%C3%A9")` yields "é", and http.server
        decodes headers as latin-1. _auth is the FIRST place untrusted text is touched
        (only _host_ok runs before it), so a one-character token produced no 403, no 500,
        and no response at all -- the opposite of "a token is required on every request".
        Encoding first turns it into an ordinary mismatch.
        """
        got = (q.get("t", [""])[0] or self.headers.get("X-Track2-Token") or "")
        if not TOKEN:
            return False
        return hmac.compare_digest(got.encode("utf-8", "surrogateescape"),
                                   TOKEN.encode("utf-8"))

    def _read_form(self) -> dict:
        """Parse a form body, framing ONLY by Content-Length -- and saying so.

        BaseHTTPRequestHandler never decodes chunked transfer-encoding, so a chunked POST
        used to yield an empty form silently: MAX_BODY was not applied, the unread bytes
        stayed in the stream to be read as the next request, and /launch went on to start a
        run from the DEFAULT Spec. Refusing is the only honest option, since this server
        cannot dechunk.
        """
        te = (self.headers.get("Transfer-Encoding") or "").strip().lower()
        if te:
            raise ValueError(f"Transfer-Encoding {te!r} is not supported; send a body with "
                             f"Content-Length (this server does not dechunk, and silently "
                             f"reading nothing would launch a run with default settings)")
        rawlen = self.headers.get("Content-Length")
        try:
            n = int(rawlen) if rawlen is not None else 0
        except (TypeError, ValueError):
            raise ValueError(f"Content-Length {rawlen!r} is not an integer") from None
        if n < 0 or n > MAX_BODY:
            raise ValueError(f"body of {n} bytes refused (cap {MAX_BODY})")
        raw = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        return urllib.parse.parse_qs(raw, keep_blank_values=True)

    # -- verbs
    def do_GET(self):
        if not self._host_ok():
            return self._html(421, "<h1>421</h1><p>Bind is loopback-only.</p>")
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if not self._auth(q):
            return self._html(403, "<h1>403</h1><p>Missing or bad token. Use the URL "
                                   "printed in the terminal that started this UI.</p>")
        if u.path == "/":
            want = q.get("ledger", [""])[0]
            if want:
                try:
                    resolve_ledger(want)          # membership test, not a path join
                except KeyError:
                    return self._html(404, "<h1>404</h1><p>No such ledger.</p>")
            return self._html(200, dashboard(want=want,
                                             msg=(q.get("msg", [""])[0]),
                                             err=(q.get("err", [""])[0])))
        if u.path == "/status":
            body = json.dumps({"run": RUNNER.status(),
                               "scorer": RUNNER.scorer_ready(),
                               "capabilities": capabilities()}, indent=1).encode()
            return self._send(200, body, "application/json")
        if u.path == "/capabilities":
            return self._send(200, json.dumps(capabilities(), indent=1).encode(),
                              "application/json")
        if u.path == "/events":
            name = q.get("ledger", [""])[0]
            try:
                after = int(q.get("after", ["-1"])[0])
                p = resolve_ledger(name)
            except (KeyError, TypeError, ValueError):
                return self._html(404, "<h1>404</h1><p>No such ledger or bad cursor.</p>")
            ev = research_events(load_ledger(p), after=after)
            body = json.dumps({"ledger": name, "events": ev,
                               "last_seq": ev[-1]["seq"] if ev else after},
                              default=str).encode()
            return self._send(200, body, "application/json")
        if u.path == "/replay":
            name = q.get("ledger", [""])[0]
            try:
                p = resolve_ledger(name)
            except KeyError:
                return self._html(404, "<h1>404</h1><p>No such ledger.</p>")
            return self._html(200, replay_page(name, research_events(load_ledger(p))))
        if u.path == "/report":
            name = q.get("ledger", [""])[0]
            try:
                p = resolve_ledger(name)
            except KeyError:
                return self._html(404, "<h1>404</h1><p>No such ledger.</p>")
            e = load_ledger(p)
            if not e:
                return self._html(200, "<h1>Empty ledger</h1>")
            # A code-first ledger has a different run shape and its own renderer. It
            # is checked FIRST: a research ledger also satisfies is_official() (its
            # scores carry `primary`), so testing officiality first would render the
            # code-first runs through a report that knows nothing about them.
            from agent.research_report import build as build_research
            from agent.research_report import is_research
            if is_research(e):
                return self._html(200, build_research(e))
            if not is_official(e):
                return self._html(200, f"<h1>{html.escape(name)}</h1><p>This is a "
                                       f"superseded full-catalog ledger; render it with "
                                       f"<code>python3 -m agent.report</code>.</p>")
            return self._html(200, build_report(e))
        return self._html(404, "<h1>404</h1>")

    def do_POST(self):
        if not self._host_ok():
            return self._html(421, "<h1>421</h1>")
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if not self._auth(q):
            return self._html(403, "<h1>403</h1><p>Missing or bad token.</p>")
        ok, why = self._origin_why()
        if not ok:
            # Name the header and its value. The bare "Cross-origin POST refused." gave a
            # user no way to tell a real attack from a browser quirk in our own check.
            print(f"  [ui] REFUSED POST {u.path}: {why}  "
                  f"(Origin={self.headers.get('Origin')!r} "
                  f"Sec-Fetch-Site={self.headers.get('Sec-Fetch-Site')!r} "
                  f"Referer={self.headers.get('Referer')!r})")
            return self._html(403,
                "<h1>403</h1><p>This POST was refused as cross-origin.</p>"
                f"<p>Reason: <code>{html.escape(why)}</code></p>"
                "<p>If you clicked a button on this dashboard, that is a bug in the check, "
                "not an attack &mdash; the terminal running <code>agent.control</code> has "
                "logged every relevant header.</p>")
        try:
            form = self._read_form()
        except ValueError as e:
            return self._html(413, f"<h1>413</h1><p>{html.escape(str(e))}</p>")

        if u.path == "/launch":
            try:
                spec = CL.parse(form)
                st = RUNNER.launch(spec)
            except CL.LaunchRefused as e:
                return self._redirect(err=str(e))
            return self._redirect(msg=f"launched pid {st['pid']} -> {st['ledger']}")
        if u.path == "/stop":
            out = RUNNER.stop()
            return self._redirect(msg=f"stop: {out.get('state')} "
                                      f"{out.get('note', '')}".strip())
        return self._html(404, "<h1>404</h1>")


def main() -> None:
    global TOKEN
    ap = argparse.ArgumentParser(description="local control plane for the MLE loop")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback only. Anything else is refused: this process starts "
                         "training runs, so exposing it to a network is handing that "
                         "ability to the network.")
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"refusing to bind {a.host!r}: this UI launches processes and is "
                         f"loopback-only by design. Use an SSH tunnel if you need remote "
                         f"access, so the authentication is SSH's and not this server's.")
    TOKEN = secrets.token_urlsafe(32)
    if os.environ.get("TRACK2_SIGNING_KEY"):
        print("  note: TRACK2_SIGNING_KEY is set in this shell. The UI does not use it "
              "and strips it from every child it starts, but the cleanest setup is to "
              "start the UI from a shell that never had it.")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    url = f"http://{a.host}:{a.port}/?t={TOKEN}"
    print("=" * 78)
    print("Track 2 control plane")
    print(f"  {url}")
    print("  loopback only, per-start token, no signing key in this process.")
    print(f"  scorer listening: {RUNNER.scorer_ready()}")
    print("=" * 78)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        if RUNNER.alive():
            print(f"  a run is still in flight (pid {RUNNER.proc.pid}); leaving it "
                  f"running. Stop it with:  kill {RUNNER.proc.pid}")
        srv.server_close()


if __name__ == "__main__":
    main()
