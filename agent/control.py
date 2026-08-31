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
  * It NEVER OPENS A LEDGER FOR WRITING, and every launch is given a freshly generated
    ledger path. There is no code path from an HTTP request to an existing chain.
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

from agent import control_launch as CL
from agent.official_report import CSS, build as build_report, is_official, load as load_ledger

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 64 * 1024                      # a launch form is a few hundred bytes
LOG_TAIL = 16_000                         # bytes of run output shown in the browser

TOKEN = ""
RUNNER = CL.Runner()


# ------------------------------------------------------------------ ledger index
def ledger_index() -> list[dict]:
    """Summarise every ledger without trusting any of them to be well-formed."""
    out = []
    for p in sorted(CL.LEDGER_DIR.glob("*.jsonl"), key=lambda q: -q.stat().st_mtime):
        row = {"name": p.name, "bytes": p.stat().st_size,
               "mtime": p.stat().st_mtime, "entries": 0, "official": False,
               "best": None, "proposer": None, "model": None, "termination": None,
               "arm": None}
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


# ------------------------------------------------------------------ page
def _opt(vals, cur) -> str:
    return "".join(f"<option value='{html.escape(v)}'"
                   f"{' selected' if v == cur else ''}>{html.escape(v)}</option>"
                   for v in vals)


def dashboard(msg: str = "", err: str = "") -> str:
    st = RUNNER.status()
    live = st["state"] == "running"
    scorer = RUNNER.scorer_ready()
    rows = []
    for r in ledger_index():
        age = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["mtime"]))
        link = (f"<a href='/report?ledger={urllib.parse.quote(r['name'])}&t={TOKEN}'>"
                f"report</a>" if r["official"] and r["entries"] else
                ("<span class=muted>legacy</span>" if r["entries"]
                 else "<span class=muted>empty</span>"))
        # A ledger that tripped ledger_index's broad `except` used to be rendered with
        # whatever partial numbers had been gathered and NO indication -- the error was
        # stored in row["error"] and never read by anything. Say so instead.
        if r.get("error"):
            rows.append(
                f"<tr><td class=mono>{html.escape(r['name'])}</td>"
                f"<td class='n bad'>!</td>"
                # colspan is (total columns - the 4 cells this row draws itself). The
                # table has 9 columns, so 5. It was 5 against an EIGHT-column header
                # before, i.e. one too many -- an unreadable ledger's row shifted the
                # table by a column, which is only visible when a ledger fails to load.
                f"<td colspan=5 class=bad>unreadable: {html.escape(str(r['error'])[:160])} "
                f"&mdash; any numbers for this ledger would be partial, so none are shown</td>"
                f"<td class=muted>{age}</td><td>&ndash;</td></tr>")
            continue
        # `best` comes from ledger JSON with only an `is not None` check, and this format()
        # sits OUTSIDE ledger_index's try -- so one ledger carrying a string there killed the
        # whole dashboard with a ValueError inside do_GET.
        try:
            best_txt = "&ndash;" if r["best"] is None else format(float(r["best"]), ".5f")
        except (TypeError, ValueError):
            best_txt = f"<span class=bad>?{html.escape(str(r['best'])[:12])}</span>"
        rows.append(
            f"<tr><td class=mono>{html.escape(r['name'])}</td>"
            f"<td class=n>{r['entries']}</td>"
            f"<td class=n>{best_txt}</td>"
            f"<td>{'<b>code-first</b>' if r.get('arm') == 'code-first' else html.escape(str(r.get('arm') or '&ndash;'))}</td>"
            f"<td>{html.escape(str(r['proposer'] or '&ndash;'))}</td>"
            f"<td class='mono wrap'>{html.escape(str(r['model'] or ''))}</td>"
            f"<td class='muted wrap'>{html.escape(str(r['termination'] or ''))}</td>"
            f"<td class=muted>{age}</td><td>{link}</td></tr>")

    banner = ""
    if err:
        banner = f"<div class='note warn'><b>Refused.</b> {html.escape(err)}</div>"
    elif msg:
        banner = f"<div class='note'>{html.escape(msg)}</div>"

    if not scorer:
        banner += ("<div class='note warn'><b>No scorer is listening.</b> It holds the "
                   "signing key, and this UI deliberately does not &mdash; so it cannot "
                   "start one for you. In another terminal:<br>"
                   "<span class=mono>python3 -m core.scorer serve --task official</span>"
                   "</div>")
    else:
        led = CL.scorer_ledger()
        banner += ("<div class='note'>Scorer is listening and writing to "
                   f"<span class=mono>{html.escape(str(led or 'unknown'))}</span>. The "
                   "scorer chooses that file, not this UI &mdash; start it with a "
                   "different <span class=mono>TRACK2_LEDGER</span> to scope a run to its "
                   "own chain.</div>")

    if live:
        state = (f"<div class='note'><b>Running</b> &middot; pid {st['pid']} &middot; "
                 f"{html.escape(str(st['proposer']))}"
                 f"{' / ' + html.escape(str(st['model'])) if st.get('model') else ''} "
                 f"&middot; {st['seconds']:.0f}s &middot; ledger "
                 f"<span class=mono>{html.escape(str(st['ledger']))}</span></div>"
                 f"<form method=post action='/stop?t={TOKEN}'>"
                 f"<button class='btn danger'>Stop this run</button></form>")
    elif st["state"] == "finished":
        state = (f"<div class='note'>Last run exited with code "
                 f"<b>{st['returncode']}</b> after {st['seconds']:.0f}s &middot; ledger "
                 f"<span class=mono>{html.escape(str(st['ledger']))}</span></div>")
    else:
        state = "<div class='note muted'>No run has been started from this UI.</div>"

    log = ""
    if RUNNER.log and RUNNER.log.exists():
        try:
            with open(RUNNER.log, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - LOG_TAIL))
                tail = fh.read().decode("utf-8", "replace")
        except OSError as exc:
            tail = f"(log unreadable: {exc})"
        log = ("<h2>Run output</h2><pre class=log id=log>"
               + html.escape(tail) + "</pre>")

    d = CL.Spec()
    form = f"""<form method=post action='/launch?t={TOKEN}' class=form>
<div class=fields>
<label>loop<select name=driver>{_opt(CL.DRIVERS, d.driver)}</select></label>
<label>proposer <span class=muted>(typed)</span><select name=proposer>{_opt(CL.PROPOSERS, d.proposer)}</select></label>
<label>backend <span class=muted>(code-first)</span><select name=backend>{_opt(CL.BACKENDS, d.backend)}</select></label>
<label>model <input name=model placeholder='gemini-3.5-flash-lite' size=22></label>
<label>iterations<input name=iterations type=number min=1 max=200 value='{d.iterations}'></label>
<label>search seed <span class=muted>(typed; seeds the SEARCH, not the model)</span>
<input name=seed type=number min=0 value='{d.seed}'></label>
<label>model seed <span class=muted>(typed; reaches the fit's RNG)</span>
<input name=model_seed type=number min=0 value='{d.model_seed}'></label>
<label>convergence<select name=convergence>{_opt(CL.CONVERGENCE, d.convergence)}</select></label>
<label>start loss <span class=muted>(typed)</span><select name=start_loss>{_opt(CL.LOSSES, d.start_loss)}</select></label>
<label>sandbox<select name=patch_isolation>
{_opt([i for i in CL.ISOLATION if i != 'none'], d.patch_isolation)}</select></label>
<label>max seconds <span class=muted>(typed)</span><input name=max_seconds type=number min=10 max=86400 value='{d.max_seconds:.0f}'></label>
<label>wall cap h <span class=muted>(code-first)</span><input name=max_wall_seconds type=number min=60 max=21600 value='{d.max_wall_seconds:.0f}'></label>
<label>repairs <span class=muted>(code-first)</span><input name=max_repairs type=number min=0 max=5 value='{d.max_repairs}'></label>
<label>tag<input name=tag value='ui' size=10></label>
</div>
<label>seed config override <span class=muted>(code-first; leave EMPTY to start from the
unmodified official baseline &mdash; anything here is recorded in the ledger as a
disclosure)</span>
<input name=seed_config placeholder='{{"loss": "pairwise"}}' size=40></label>
<label class=chk><input type=checkbox name=allow_random_log value=1>
train on <span class=mono>log_random</span> (a rules judgement &mdash; refusals are recorded)</label>
<button class=btn {'disabled' if (live or not scorer) else ''}>Launch run</button>
{"<span class=muted>&nbsp; a run is already in flight</span>" if live else ""}
</form>"""

    extra = """
.btn{font:inherit;padding:8px 15px;border-radius:6px;border:1px solid var(--rule);
background:var(--accent);color:#fff;cursor:pointer;margin-top:12px}
.btn[disabled]{opacity:.45;cursor:not-allowed}
.btn.danger{background:var(--bad)}
.form{background:var(--card);border:1px solid var(--rule);border-radius:7px;padding:16px}
.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px}
label{display:flex;flex-direction:column;gap:4px;font-size:11.5px;color:var(--mut);
text-transform:uppercase;letter-spacing:.05em}
input,select{font:14px ui-sans-serif,system-ui,sans-serif;padding:6px 8px;
border:1px solid var(--rule);border-radius:5px;background:var(--bg);color:var(--fg)}
label.chk{flex-direction:row;align-items:center;gap:8px;text-transform:none;
letter-spacing:0;font-size:13px;margin-top:14px}
label.chk input{width:auto}
pre.log{background:var(--card);border:1px solid var(--rule);border-radius:7px;
padding:12px 14px;overflow:auto;max-height:340px;font:12px/1.5 ui-monospace,Menlo,monospace;
white-space:pre-wrap;word-break:break-word}
"""
    poll = ("<script>setTimeout(function(){location.reload()},4000)</script>"
            if live else "")
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Track 2 control plane</title>
<meta name=viewport content='width=device-width,initial-scale=1'>
<style>{CSS}{extra}</style></head><body><main>
<h1>Track 2 &mdash; control plane</h1>
<p class=sub>Launch and watch autonomous MLE runs, and read the report of any ledger.
This process holds <b>no signing key</b> and never writes a ledger: it asks the scorer,
over a local socket, to sign what a run produces.</p>
{banner}
<h2>Current run</h2>{state}
<h2>Launch</h2>{form}
{log}
<h2>Ledgers</h2>
<div class=scroll><table><thead><tr><th>ledger</th><th>entries</th><th>best valid</th>
<th>arm</th><th>proposer</th><th>model</th><th>termination</th><th>modified</th>
<th></th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan=8 class=muted>no ledgers yet</td></tr>'}
</tbody></table></div>
<h2>What this UI cannot do</h2>
<div class=note>It cannot sign a ledger entry (no key), cannot write or delete an
existing ledger (every launch gets a fresh path), cannot start the scorer, cannot run
more than one driver at once, and cannot pass a value the driver would reject &mdash;
every field is checked against the driver&rsquo;s own choices before a process starts.
Bound to <span class=mono>127.0.0.1</span> with a per-start token.</div>
</main>{poll}</body></html>"""


# ------------------------------------------------------------------ http
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
            return self._html(200, dashboard(msg=(q.get("msg", [""])[0]),
                                             err=(q.get("err", [""])[0])))
        if u.path == "/status":
            body = json.dumps({"run": RUNNER.status(),
                               "scorer": RUNNER.scorer_ready()}, indent=1).encode()
            return self._send(200, body, "application/json")
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
