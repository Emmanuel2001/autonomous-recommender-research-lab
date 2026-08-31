"""HTTP-layer tests for the control plane.

Runs a real server on an ephemeral loopback port. TRACK2_SOCKET is pointed at a path
that cannot exist FIRST, before anything imports the control module, so `scorer_ready()`
is false for the whole file and a POST /launch exercises the refusal path instead of
starting a 30-second training run from a test.
"""
import http.client
import json
import os
import re
import sys
import threading
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TRACK2_SOCKET"] = "/nonexistent/track2-test/scorer.sock"

from http.server import ThreadingHTTPServer                            # noqa: E402
from agent import control as C                                         # noqa: E402
from agent import control_launch as CL                                 # noqa: E402

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name}  {detail}")


C.TOKEN = "test-token-" + "z" * 20
GOOD = C.TOKEN
srv = ThreadingHTTPServer(("127.0.0.1", 0), C.Handler)
srv.daemon_threads = True
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()


def req(method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=15)
    h = {"Host": host or f"127.0.0.1:{PORT}"}
    if body is not None:
        h["Content-Type"] = "application/x-www-form-urlencoded"
    h.update(headers or {})
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read().decode("utf-8", "replace")
    st, hdr = r.status, dict(r.getheaders())
    c.close()
    return st, data, hdr


print("== the scorer must be absent for this file to be safe")
check("scorer_ready is False (so no test can start a driver)",
      not CL.Runner().scorer_ready())
check("no run in flight at start", not C.RUNNER.alive())

print("== token is required on every route")
for path in ["/", "/status", "/report?ledger=x"]:
    st, _b, _h = req("GET", path)
    check(f"GET {path} without a token -> 403", st == 403, str(st))
st, b, _h = req("GET", f"/?t={GOOD[:-1]}x")
check("a wrong token -> 403", st == 403, str(st))
st, b, _h = req("GET", "/?t=")
check("an empty token -> 403", st == 403, str(st))
st, b, _h = req("GET", f"/?t={GOOD}")
check("the right token -> 200", st == 200, str(st))
check("dashboard renders the launch form", "Launch run" in b)
st, b, _h = req("GET", "/", headers={"X-Track2-Token": GOOD})
check("token accepted in a header too", st == 200, str(st))

print("== a blank TOKEN must lock the server, not open it")
saved, C.TOKEN = C.TOKEN, ""
st, _b, _h = req("GET", "/?t=")
check("empty server token refuses rather than matching", st == 403, str(st))
C.TOKEN = saved

print("== Host is pinned to loopback (DNS rebinding)")
for bad in ["evil.example.com", "attacker.test:1234", "169.254.169.254"]:
    st, _b, _h = req("GET", f"/?t={GOOD}", host=bad)
    check(f"Host {bad!r} -> 421", st == 421, str(st))
for good in [f"127.0.0.1:{PORT}", f"localhost:{PORT}"]:
    st, _b, _h = req("GET", f"/?t={GOOD}", host=good)
    check(f"Host {good!r} accepted", st == 200, str(st))
st, _b, _h = req("POST", f"/stop?t={GOOD}", body="", host="evil.example.com")
check("POST with a foreign Host -> 421", st == 421, str(st))

print("== cross-site POST is refused (Origin)")
st, _b, _h = req("POST", f"/stop?t={GOOD}", body="",
                 headers={"Origin": "https://evil.example.com"})
check("foreign Origin -> 403", st == 403, str(st))
st, _b, h = req("POST", f"/stop?t={GOOD}", body="",
                headers={"Origin": f"http://127.0.0.1:{PORT}"})
check("own Origin allowed", st == 303, str(st))
st, _b, _h = req("POST", f"/stop?t={GOOD}", body="")
check("no Origin (a plain same-origin form post) allowed", st == 303, str(st))
st, _b, _h = req("POST", f"/stop?t=nope", body="",
                 headers={"Origin": f"http://127.0.0.1:{PORT}"})
check("token still checked before Origin", st == 403, str(st))

print("== REAL browser header sets must be able to POST (the 403 a user actually hit)")
# The first version keyed only on urlparse(Origin).hostname, and my test only ever sent a
# synthetic Origin I made up -- so it passed while a real dashboard form POST was refused.
# These are the header combinations browsers actually send.
BROWSER = [
    ("chrome/safari same-origin form post",
     {"Origin": f"http://127.0.0.1:{PORT}", "Sec-Fetch-Site": "same-origin"}, 303),
    ("opaque origin with same-origin fetch metadata",
     {"Origin": "null", "Sec-Fetch-Site": "same-origin"}, 303),
    ("opaque origin, no fetch metadata (older browser)",
     {"Origin": "null"}, 303),
    ("no Origin, same-origin fetch metadata",
     {"Sec-Fetch-Site": "same-origin"}, 303),
    ("localhost rather than the literal IP",
     {"Origin": f"http://localhost:{PORT}", "Sec-Fetch-Site": "same-origin"}, 303),
    ("user-initiated navigation (typed URL, bookmark)",
     {"Sec-Fetch-Site": "none"}, 303),
    ("no Origin and no fetch metadata at all", {}, 303),
]
for name, hdrs, want in BROWSER:
    st, _b, _h = req("POST", f"/stop?t={GOOD}", body="", headers=hdrs)
    check(f"{name} -> {want}", st == want, f"got {st}")

print("== a real cross-site attack is still refused, with or without fetch metadata")
ATTACK = [
    ("evil.com with cross-site metadata",
     {"Origin": "https://evil.com", "Sec-Fetch-Site": "cross-site"}),
    ("evil.com without metadata", {"Origin": "https://evil.com"}),
    ("same-site but not same-origin", {"Sec-Fetch-Site": "same-site"}),
    ("a host that merely starts with the loopback IP",
     {"Origin": "http://127.0.0.1.evil.com"}),
    ("metadata beats a spoofed loopback Origin",
     {"Origin": f"http://127.0.0.1:{PORT}", "Sec-Fetch-Site": "cross-site"}),
]
for name, hdrs in ATTACK:
    st, b, _h = req("POST", f"/stop?t={GOOD}", body="", headers=hdrs)
    check(f"{name} -> 403", st == 403, f"got {st}")
    if st == 403:
        check(f"  ...and the 403 names the header that refused it",
              "Reason:" in b and ("Sec-Fetch-Site" in b or "Origin" in b), b[:120])
check("Sec-Fetch-Site is preferred over Origin when both are present",
      "Sec-Fetch-Site" in __import__("inspect").getsource(C.Handler._origin_why).split("Origin")[0])
check("the token is still checked BEFORE the origin", 
      req("POST", "/stop?t=wrong", body="",
          headers={"Origin": "https://evil.com"})[0] == 403)

print("== body size cap")
st, _b, _h = req("POST", f"/launch?t={GOOD}", body="x=" + "a" * (C.MAX_BODY + 100))
check("oversized body -> 413", st == 413, str(st))
check("cap is small (a form, not an upload)", C.MAX_BODY <= 128 * 1024, str(C.MAX_BODY))

print("== ledger names resolve by directory membership, not string joining")
for bad in ["../../etc/passwd", "..%2f..%2fetc%2fpasswd", "/etc/passwd",
            "nope.jsonl", "", ".", "..",
            "../ledgers/official_run1.jsonl"]:
    st, _b, _h = req("GET", f"/report?ledger={urllib.parse.quote(bad, safe='')}&t={GOOD}")
    check(f"ledger {bad[:26]!r} -> 404", st == 404, str(st))
real = sorted(p.name for p in CL.LEDGER_DIR.glob("*.jsonl"))
if real:
    st, b, _h = req("GET", f"/report?ledger={urllib.parse.quote(real[0])}&t={GOOD}")
    check(f"a real ledger renders ({real[0]})", st == 200, str(st))
else:
    # Say so. A check that quietly does not run reads exactly like a check that passed,
    # and the count at the bottom hides the difference.
    print("  skip no ledgers in runs/ledgers, so the positive path is untested here")
try:
    C.resolve_ledger("../../etc/passwd")
    check("resolve_ledger refuses traversal", False, "IT RESOLVED")
except KeyError:
    check("resolve_ledger refuses traversal", True)

print("== status endpoint")
st, b, h = req("GET", f"/status?t={GOOD}")
check("status is 200 JSON", st == 200 and "application/json" in h.get("Content-Type", ""))
j = json.loads(b)
check("status reports idle", j["run"]["state"] == "idle", str(j))
check("status reports the scorer as absent", j["scorer"] is False, str(j))
check("status leaks no token", GOOD not in b)

print("== launching without a scorer is refused, and starts nothing")
st, _b, h = req("POST", f"/launch?t={GOOD}",
                body=urllib.parse.urlencode({"proposer": "random", "tag": "httptest"}))
check("launch redirects (303)", st == 303, str(st))
loc = h.get("Location", "")
check("redirect carries the refusal", "err=" in loc, loc[:120])
check("refusal names the missing scorer", "scorer" in urllib.parse.unquote_plus(loc))
check("NO child process was started", not C.RUNNER.alive())
check("no ledger was created for it",
      not list(CL.LEDGER_DIR.glob("ui_httptest_*.jsonl")))

print("== a refused FORM value never reaches a process either")
for form, why in [({"proposer": "gemini"}, "model"),
                  ({"proposer": "random", "iterations": "9999"}, "outside"),
                  ({"proposer": "random", "tag": "../esc"}, "tag"),
                  ({"proposer": "bogus"}, "not one of"),
                  ({"proposer": "random", "patch_isolation": "none"}, "boundary")]:
    st, _b, h = req("POST", f"/launch?t={GOOD}", body=urllib.parse.urlencode(form))
    loc = urllib.parse.unquote_plus(h.get("Location", ""))
    check(f"{list(form.values())} refused ({why})",
          st == 303 and "err=" in h.get("Location", "") and why in loc, loc[:130])
check("still nothing running after every refusal", not C.RUNNER.alive())

print("== unknown routes and methods")
for m, p, exp in [("GET", "/nope", 404), ("POST", "/nope", 404),
                  ("GET", "/../agent/control.py", 404),
                  ("GET", "/runs/ledger.jsonl", 404),
                  ("GET", "/secrets/scorer.key", 404)]:
    st, _b, _h = req(m, f"{p}?t={GOOD}", body="" if m == "POST" else None)
    check(f"{m} {p} -> {exp}", st == exp, str(st))
st, _b, _h = req("GET", "/favicon.ico")
check("favicon needs no token and returns no content", st == 204, str(st))

print("== response headers")
st, _b, h = req("GET", f"/?t={GOOD}")
check("no-store set", "no-store" in h.get("Cache-Control", ""))
check("nosniff set", h.get("X-Content-Type-Options") == "nosniff")
check("no-referrer set", h.get("Referrer-Policy") == "no-referrer")
check("content-length present", "Content-Length" in h)

print("== the bind guard is loopback-only")
src = (Path(__file__).resolve().parents[1] / "agent" / "control.py").read_text()
check("main refuses a non-loopback host", "refusing to bind" in src)
check("default host is 127.0.0.1", 'default="127.0.0.1"' in src)
check("no 0.0.0.0 anywhere in the module", "0.0.0.0" not in src)

print("== the UI process holds no key")
# This is the single most important assertion in the file, and it had ZERO discriminating
# power. It read:
#     "TRACK2_SIGNING_KEY" not in src.replace('os.environ.get("TRACK2_SIGNING_KEY")', "")
#     or "does not use it" in src
# control.py mentions the name on two lines, so .replace() removed only the first and the
# left clause was ALREADY False on the unmodified file -- the whole check was carried by a
# grep for a phrase in control.py's own printed note. An adversarial reviewer sabotaged
# control.py to read the key and exfiltrate it, and this still passed.
# Replaced with an exact whitelist of the two benign lines: anything else that mentions the
# key fails, including a line that merely reads it into a variable.
_key_lines = [ln.strip() for ln in src.splitlines() if "TRACK2_SIGNING_KEY" in ln]
check("control.py mentions the signing key on exactly two known lines",
      len(_key_lines) == 2, f"{len(_key_lines)} lines: {_key_lines}")
check("the first is only a presence check, never a read into a value",
      _key_lines and _key_lines[0] == 'if os.environ.get("TRACK2_SIGNING_KEY"):',
      _key_lines[0] if _key_lines else "(none)")
check("the second is only the operator note",
      len(_key_lines) > 1 and _key_lines[1].startswith('print("  note: TRACK2_SIGNING_KEY'),
      _key_lines[1] if len(_key_lines) > 1 else "(none)")
check("the key is never bound to a name in this module",
      not re.search(r"=\s*os\.environ(?:\.get)?\(?\s*['\"]TRACK2_SIGNING_KEY", src),
      "control.py assigns the signing key to a variable")
check("and never handed to a subprocess or a socket from here",
      not re.search(r"TRACK2_SIGNING_KEY[^\n]*(?:env|send|write|post|Popen)", src))
kp = Path(__file__).resolve().parents[1] / "secrets" / "scorer.key"
if kp.exists():
    key = kp.read_text().strip()
    st, b, _h = req("GET", f"/?t={GOOD}")
    check("the key never appears in a served page", key and key not in b)
    st, b, _h = req("GET", f"/status?t={GOOD}")
    check("nor in the status JSON", key not in b)

srv.shutdown()
srv.server_close()
print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
