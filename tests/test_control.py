"""The UI's security spec. A web UI that starts a training run is remote code execution
by design, so these tests are the boundary, not a formality.

Nothing here starts a real driver: Runner.launch takes an injectable `popen`, so process
control is tested without a 30-second fit or a scorer.
"""
import os
import re
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import control_launch as CL                                 # noqa: E402
from agent.official_driver import build_parser                         # noqa: E402

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name}  {detail}")


def refused(form, needle=None):
    try:
        CL.parse(form)
        return False
    except CL.LaunchRefused as e:
        return needle is None or needle in str(e)


BASE = {"proposer": "random"}

print("== the ledger table's columns must line up")
import re as _re
_src = Path(__file__).resolve().parents[1].joinpath("agent/control.py").read_text()
_i = _src.index("<th>arm</th>")
_head = _src[_src.rindex("<thead>", 0, _i):_src.index("</tr></thead>", _i)]
_ncol = _head.count("<th>")
_k = _src.index('best_txt = f"<span class=bad>')
_blk = _src[_src.index("rows.append(", _k):]
_data = len(_re.findall(r"<td", _blk[:_blk.index("</tr>")]))
check("the data row has one cell per header column", _ncol == _data,
      f"header={_ncol} row={_data}")
_e = _src.index("colspan=")
_span = int(_re.search(r"colspan=(\d+)", _src[_e:]).group(1))
_erow = _src[_src.rindex("rows.append(", 0, _e):_src.index("</tr>", _e)]
_ecells = _erow.count("<td") - 1
check("the unreadable-ledger row spans exactly the table width",
      _ecells + _span == _ncol, f"{_ecells} cells + colspan {_span} vs {_ncol} columns")

print("== the allowlist must match the driver's real choices (drift guard)")
acts = {a.dest: a for a in build_parser()._actions}
pairs = [("proposer", CL.PROPOSERS), ("start_loss", CL.LOSSES),
         ("convergence", CL.CONVERGENCE), ("patch_isolation", CL.ISOLATION)]
for dest, mine in pairs:
    real = tuple(acts[dest].choices or ())
    check(f"{dest} choices agree with the driver", tuple(mine) == real,
          f"ui={mine} driver={real}")
check("every UI field is a real driver argument",
      all(d in acts for d, _ in pairs) and "allow_random_log" in acts)

# The seed field in this UI only ever seeded the SEARCH, so every model was fitted at
# seed 0 whatever was entered. Both are now separate fields and both must reach the driver.
check("the driver accepts --model-seed", "model_seed" in acts)
check("the UI sends --model-seed", "--model-seed" in CL.parse({**BASE}).argv())
_sp = CL.parse({**BASE, "seed": "7", "model_seed": "3"})
_av = _sp.argv()
check("search seed and model seed are sent independently",
      _av[_av.index("--seed") + 1] == "7"
      and _av[_av.index("--model-seed") + 1] == "3", " ".join(_av))
check("the model seed defaults to 0, preserving every earlier run's behaviour",
      CL.Spec().model_seed == 0)

# The same guard for the CODE-FIRST driver. Without this, adding a backend to one side
# and not the other is a runtime failure at iteration 1 instead of a test failure.
from agent.research_driver import build_parser as rbuild
racts = {a.dest: a for a in rbuild()._actions}
check("research backend choices agree with the research driver",
      tuple(CL.BACKENDS) == tuple(racts["backend"].choices or ()),
      f"ui={CL.BACKENDS} driver={tuple(racts['backend'].choices or ())}")
check("research isolation choices agree with the research driver",
      tuple(CL.ISOLATION) == tuple(racts["isolation"].choices or ()),
      f"ui={CL.ISOLATION} driver={tuple(racts['isolation'].choices or ())}")
check("research convergence choices agree with the research driver",
      tuple(CL.CONVERGENCE) == tuple(racts["convergence"].choices or ()))
for flag in ("iterations", "max_wall_seconds", "fit_seconds", "max_repairs",
             "max_tokens", "model", "thinking_level", "thought_summaries",
             "task_spec", "seed_config"):
    check(f"the research driver really accepts --{flag.replace('_','-')}", flag in racts)
check("both drivers are offered by the UI", tuple(CL.DRIVERS) == ("official", "research"))

# the argv the UI would actually send must be accepted by the driver's own parser
sp = CL.parse({"driver": "research", "backend": "gemini", "model": "gemini-3.5-flash-lite",
               "iterations": "50", "convergence": "off",
               "seed_config": '{"loss": "pairwise"}'})
av = sp.argv()
check("the research argv targets the research driver",
      av[:5] == ["python3", "-u", "-m", "agent.research_driver", "--backend"], str(av[:5]))
try:
    rbuild().parse_args(av[4:])
    ok_parse = True
except SystemExit:
    ok_parse = False
check("the research driver parses the exact argv the UI builds", ok_parse, " ".join(av[4:]))

portable = Path(__file__).resolve().parent / "fixtures" / "portable_recommender" / "task.json"
sp_portable = CL.parse({"driver": "research", "backend": "gemini", "model": "m",
                        "task_spec": str(portable), "thought_summaries": "1"})
check("portable task path reaches the research driver",
      sp_portable.argv()[sp_portable.argv().index("--task-spec") + 1] == str(portable))
check("public thought summaries reach the research driver",
      "--thought-summaries" in sp_portable.argv())
check("task specs outside the repository are refused",
      refused({"driver": "research", "backend": "gemini", "model": "m",
               "task_spec": "/tmp/outside.json"}, "inside this repository"))
check("task spec is research-only",
      refused({"proposer": "random", "task_spec": str(portable)}, "code-first"))

print("== enums reject anything not on the list")
check("unknown proposer refused", refused({"proposer": "gpt5"}, "not one of"))
check("unknown loss refused", refused({**BASE, "start_loss": "hinge"}))
check("unknown convergence refused", refused({**BASE, "convergence": "maybe"}))
check("unknown isolation refused", refused({**BASE, "patch_isolation": "chroot"}))
# An EMPTY form is now a refusal, not "all defaults": a body that failed to arrive used to
# launch a real 12-iteration run with settings nobody chose. Omitted-but-not-empty still
# defaults, which is what this needs to prove.
check("an EMPTY form is refused, not treated as all-defaults",
      refused({}, "settings nobody chose"))
check("omitted fields still fall back to their defaults",
      CL.parse({"proposer": "random"}).convergence == "official"
      and CL.parse({"proposer": "random"}).iterations == 12
      and CL.parse({"proposer": "random"}).patch_isolation == "seatbelt")
check("case is not silently normalised", refused({"proposer": "RANDOM"}))
check("whitespace-padded value still accepted",
      CL.parse({"proposer": " random "}).proposer == "random")

print("== isolation 'none' is refused outright")
check("none refused even though the driver accepts it",
      refused({**BASE, "patch_isolation": "none"}, "no boundary"))
check("the driver itself does still offer it (so this is the UI's own policy)",
      "none" in (acts["patch_isolation"].choices or ()))

print("== numeric bounds are cost controls")
for k, bad in [("iterations", "0"), ("iterations", "201"), ("iterations", "-5"),
               ("max_tokens", "999"), ("max_tokens", "20000001"),
               ("max_seconds", "9"), ("max_seconds", "86401"),
               ("patch_seconds", "3601"), ("epsilon_greedy", "1.1"),
               ("epsilon_greedy", "-0.1"), ("seed", "-1")]:
    check(f"{k}={bad} refused", refused({**BASE, k: bad}, "outside"), )
for k, bad in [("iterations", "abc"), ("max_seconds", ""), ("seed", "1e")]:
    if bad == "":
        check(f"{k} empty -> default", CL.parse({**BASE, k: bad}).max_seconds == 5400.0)
    else:
        check(f"{k}={bad!r} refused as non-numeric", refused({**BASE, k: bad}))
check("nan refused", refused({**BASE, "max_seconds": "nan"}))
check("inf refused", refused({**BASE, "max_seconds": "inf"}))
check("1e9 seconds refused", refused({**BASE, "max_seconds": "1e9"}))
check("in-range value accepted",
      CL.parse({**BASE, "iterations": "40"}).iterations == 40)
check("float given for an int field is truncated, not crashed",
      CL.parse({**BASE, "iterations": "12.9"}).iterations == 12)

print("== no exception may escape parse() except LaunchRefused (bug 53)")
# `cast(float(raw))` ran the int cast BEFORE the finiteness check, so iterations=inf raised
# OverflowError -- past parse()'s contract and past control.py's `except LaunchRefused`.
for field in ("iterations", "seed", "max_tokens", "max_seconds", "patch_seconds",
              "epsilon_greedy"):
    for raw in ("inf", "-inf", "1e400", "-1e400", "nan", "abc", "", "1e", "0x10",
                "1_000", "+5", " 7 ", "\u0665"):
        try:
            CL.parse({"proposer": "random", field: raw})
            outcome = "accepted"
        except CL.LaunchRefused:
            outcome = "refused"
        except Exception as e:                                        # noqa: BLE001
            outcome = f"ESCAPED {type(e).__name__}"
        check(f"{field}={raw!r} -> {outcome}", not outcome.startswith("ESCAPED"), outcome)
check("inf on an INT field is refused as non-finite",
      refused({**BASE, "iterations": "inf"}, "not finite"))
check("1e400 on an int field is refused as non-finite",
      refused({**BASE, "iterations": "1e400"}, "not finite"))
check("a normal int still parses", CL.parse({**BASE, "iterations": "7"}).iterations == 7)
check("truncation of a float is preserved",
      CL.parse({**BASE, "iterations": "12.9"}).iterations == 12)

print("== launch() is atomic under real concurrency (bug 54)")
import threading as _th

class SlowProc:
    """popen that blocks briefly, widening any check-and-act window."""
    def __init__(self, argv, **kw):
        self.argv, self.kw, self.pid, self._rc = argv, kw, 9001, None
        time.sleep(0.05)
    def poll(self): return self._rc
    def wait(self, timeout=None): self._rc = -15; return self._rc
    def terminate(self): self._rc = -15
    def kill(self): self._rc = -9

import time
rc = CL.Runner()
rc.scorer_ready = lambda socket_path=None: True
_prev_sl, CL.scorer_ledger = CL.scorer_ledger, (lambda: "/tmp/fake-race-ledger.jsonl")
_spec = CL.parse({"proposer": "random", "tag": "raceprobe"})
started, refusals, errors = [], [], []
_gate = _th.Barrier(8)

def _try_launch():
    _gate.wait()
    try:
        started.append(rc.launch(_spec, popen=SlowProc)["pid"])
    except CL.LaunchRefused:
        refusals.append(1)
    except Exception as e:                                            # noqa: BLE001
        errors.append(f"{type(e).__name__}: {e}")

ths = [_th.Thread(target=_try_launch) for _ in range(8)]
for t in ths: t.start()
for t in ths: t.join()
check("exactly ONE of 8 concurrent launches succeeded",
      len(started) == 1, f"started={len(started)} refused={len(refusals)}")
check("the other 7 were cleanly refused, not crashed",
      len(refusals) == 7 and not errors, f"refusals={len(refusals)} errors={errors}")
check("only one log file was claimed",
      len({h['log'] for h in rc.history}) == len(rc.history) == 1, str(rc.history))
CL.scorer_ledger = _prev_sl
check("the Runner has a lock", hasattr(rc, "_lock"))
check("stop() is locked too",
      "with self._lock" in __import__("inspect").getsource(CL.Runner.stop))
for h in rc.history:
    (CL.LOG_DIR / h["log"]).unlink(missing_ok=True)

print("== duplicated form fields cannot bypass validation")
check("two proposers refused", refused({"proposer": ["random", "gemini"]}, "expected one"))
check("single-element list accepted",
      CL.parse({"proposer": ["random"]}).proposer == "random")

print("== model identifier")
for good in ["gemini-3.5-flash-lite", "claude-opus-5", "llama3.1:8b",
             "org/model-v2", "gpt_oss_20b", "a"]:
    try:
        CL.parse({"proposer": "gemini", "model": good})
        check(f"accepts {good!r}", True)
    except CL.LaunchRefused as e:
        check(f"accepts {good!r}", False, str(e))
for bad in ["a; rm -rf /", "a && b", "a|b", "$(id)", "`id`", "a b", "../../etc/passwd",
            "a\nb", "a\x00b", "-flag", "x" * 200, "модель"]:
    check(f"rejects {bad[:22]!r}", refused({"proposer": "gemini", "model": bad}),
          "ACCEPTED A HOSTILE MODEL STRING")
check("empty model with an LLM proposer is refused",
      refused({"proposer": "gemini"}, "needs an explicit"))
check("random proposer needs no model", CL.parse({"proposer": "random"}).model is None)

print("== thinking level")
check("accepts a word", CL.parse({"proposer": "gemini", "model": "m",
                                  "thinking_level": "high"}).thinking_level == "high")
for bad in ["HIGH", "high; id", "1", "a" * 20, "-x"]:
    check(f"rejects thinking_level {bad[:12]!r}",
          refused({"proposer": "gemini", "model": "m", "thinking_level": bad}))

print("== tag reaches a FILENAME, so it is the traversal-sensitive field")
check("plain tag accepted", CL.parse({**BASE, "tag": "sweep-3"}).tag == "sweep-3")
for bad in ["../escape", "a/b", "a\\b", "..", ".", "a b", "A", "a" * 60,
            "a;b", "a\x00", "-a", "$x"]:
    check(f"rejects tag {bad[:14]!r}", refused({**BASE, "tag": bad}),
          "TAG COULD ESCAPE THE LEDGER DIRECTORY")
check("empty tag defaults", CL.parse({**BASE, "tag": ""}).tag == "ui")

print("== allow_random_log is opt-in and only by an explicit truthy value")
for v in ["1", "on", "true", "yes", "TRUE"]:
    check(f"{v!r} enables it", CL.parse({**BASE, "allow_random_log": v}).allow_random_log)
for v in ["", "0", "off", "false", "no", "maybe"]:
    check(f"{v!r} leaves it off",
          not CL.parse({**BASE, "allow_random_log": v}).allow_random_log)
check("absent leaves it off", not CL.parse(BASE).allow_random_log)

print("== the UI owns its LOG path; the ledger is the scorer's and it must not guess")
s = CL.parse({**BASE, "tag": "abc", "seed": "7"})
p1 = CL.log_path(s, 1_700_000_000.0, existing=set())
check("named by tag/driver/proposer/seed",
      "ui_abc_official_random_s7_" in p1.name and p1.name.endswith(".log"), p1.name)
# The driver is in the name so a log can be attributed to the loop that wrote it; with two
# loops sharing this directory, a name that omits it is ambiguous.
p1r = CL.log_path(CL.parse({"driver": "research", "backend": "gemini", "model": "m"}),
                  1_700_000_000.0, existing=set())
check("a code-first log names its loop and backend",
      "_research_gemini_" in p1r.name, p1r.name)
check("the two loops cannot collide on a log name", p1.name != p1r.name)
check("lands in runs/ui_logs",
      p1.parent.resolve() == CL.LOG_DIR.resolve(), str(p1.parent))
check("no traversal in the resolved path",
      CL.LOG_DIR.resolve() in p1.resolve().parents)
p2 = CL.log_path(s, 1_700_000_000.0, existing={p1.name})
check("collision produces a NEW name, never a reuse", p2.name != p1.name, p2.name)
p3 = CL.log_path(s, 1_700_000_000.0, existing={p1.name, p2.name})
check("second collision also avoided", p3.name not in {p1.name, p2.name})
check("same inputs are deterministic",
      CL.log_path(s, 1_700_000_000.0, existing=set()).name == p1.name)
check("the module no longer invents a ledger path at all",
      not hasattr(CL, "ledger_path"))
# The real property is that child_env never ADDS or REWRITES a ledger path -- not that
# the child's env lacks one. Inheriting the operator's TRACK2_LEDGER is normal and
# harmless (nothing in the child reads it any more); INVENTING one was bug 33. Asserting
# a bare absence passed only when the ambient shell happened not to set it, and failed
# under run_all.sh, which sets it deliberately.
check("child_env does not ADD a ledger when there is none",
      "TRACK2_LEDGER" not in CL.child_env(s, base={"PATH": "/usr/bin"}))
check("child_env passes an inherited ledger through UNCHANGED",
      CL.child_env(s, base={"TRACK2_LEDGER": "/x/amb.jsonl"})["TRACK2_LEDGER"]
      == "/x/amb.jsonl")
# Must not assume the ambient scorer state: this file is run both with a live scorer and
# without one. Assert the contract in whichever case applies.
_sl = CL.scorer_ledger()
if CL.Runner().scorer_ready():
    check("scorer_ledger returns a path when a scorer IS reachable",
          isinstance(_sl, str) and _sl.endswith(".jsonl"), repr(_sl))
else:
    check("scorer_ledger returns None when the scorer is unreachable", _sl is None,
          repr(_sl))

print("== child environment: the UI cannot mint a ledger entry")
base = {"TRACK2_SIGNING_KEY": "deadbeef" * 8, "TRACK2_KEY_FILE": "/x/k",
        "GEMINI_API_KEY": "gk", "TRACK2_LLM_API_KEY": "lk",
        "OPENAI_API_KEY": "ok", "GITHUB_TOKEN": "gh",
        "AWS_SECRET_ACCESS_KEY": "aws", "PATH": "/usr/bin"}
env_r = CL.child_env(CL.parse({"proposer": "random"}), base=base)
check("signing key stripped", "TRACK2_SIGNING_KEY" not in env_r)
check("key FILE pointer stripped too", "TRACK2_KEY_FILE" not in env_r)
check("no key material anywhere in the child env",
      not any("deadbeef" in str(v) for v in env_r.values()))
check("unrelated secrets stripped",
      not any(k in env_r for k in ("OPENAI_API_KEY", "GITHUB_TOKEN",
                                   "AWS_SECRET_ACCESS_KEY")))
check("random arm gets NO llm key",
      not any(k in env_r for k in CL.LLM_KEYS))
check("PATH preserved", env_r.get("PATH") == "/usr/bin")
check("ledger NOT pinned in the child env (the scorer decides)",
      "TRACK2_LEDGER" not in env_r)
env_l = CL.child_env(CL.parse({"proposer": "gemini", "model": "m"}), base=base)
check("llm arm DOES get its key back", env_l.get("GEMINI_API_KEY") == "gk")
check("llm arm still never gets the signing key", "TRACK2_SIGNING_KEY" not in env_l)

print("== argv: a list, no shell, and accepted by the driver's own parser")
av = CL.parse({"proposer": "gemini", "model": "gemini-3.5-flash",
               "iterations": "5", "convergence": "off",
               "allow_random_log": "1"}).argv()
check("argv is a list of str", isinstance(av, list) and all(isinstance(x, str) for x in av))
check("no shell metacharacters anywhere",
      not any(re.search(r"[;&|`$><\n]", x) for x in av), str(av))
check("no ledger and no log path on the command line",
      not any(".jsonl" in x or ".log" in x for x in av), str(av))
parsed = build_parser().parse_args(av[4:])
check("the driver accepts the UI's argv", parsed.proposer == "gemini")
check("values survive the round trip",
      parsed.model == "gemini-3.5-flash" and parsed.iterations == 5
      and parsed.convergence == "off" and parsed.allow_random_log is True)
av2 = CL.parse({"proposer": "random"}).argv()
p2_ = build_parser().parse_args(av2[4:])
check("defaults round trip too",
      p2_.proposer == "random" and p2_.convergence == "official"
      and p2_.start_loss == "pointwise" and p2_.allow_random_log is False)
check("no --model flag when there is no model", "--model" not in av2)

print("== Runner: one at a time, and it needs a scorer it does not own")
class FakeProc:
    def __init__(self, argv, **kw):
        self.argv, self.kw, self.pid, self._rc = argv, kw, 4242, None
    def poll(self): return self._rc
    def wait(self, timeout=None): self._rc = -15; return self._rc
    def terminate(self): self._rc = -15
    def kill(self): self._rc = -9

# launch() now demands BOTH that the gate opens and that the scorer names its ledger, so a
# fake run needs both stubbed. `_real_scorer_ledger` is restored at the end of the block.
_real_scorer_ledger = CL.scorer_ledger
CL.scorer_ledger = lambda: "/tmp/fake-test-ledger.jsonl"

r = CL.Runner()
check("idle before anything", r.status()["state"] == "idle" and not r.alive())
spec = CL.parse({"proposer": "random", "tag": "unittest"})
r.scorer_ready = lambda socket_path=None: False
try:
    r.launch(spec, popen=FakeProc)
    check("refuses to launch with no scorer", False, "IT LAUNCHED")
except CL.LaunchRefused as e:
    check("refuses to launch with no scorer", "no scorer is listening" in str(e))
    check("and says the UI does not hold the key", "does not" in str(e))

r.scorer_ready = lambda socket_path=None: True
st = r.launch(spec, now=1_700_000_000.0, popen=FakeProc)
check("launch reports running", st["state"] == "running" and st["pid"] == 4242)
check("child was not handed a UI-invented ledger",
      r.proc.kw["env"].get("TRACK2_LEDGER") == os.environ.get("TRACK2_LEDGER"),
      f"{r.proc.kw['env'].get('TRACK2_LEDGER')!r}")
check("child got no signing key", "TRACK2_SIGNING_KEY" not in r.proc.kw["env"])
check("child in its own session (so stop kills the whole group)",
      r.proc.kw.get("start_new_session") is True)
check("log written under runs/ui_logs", r.log.parent.name == "ui_logs")
check("log recorded in history", r.history[-1]["log"] == r.log.name)
try:
    r.launch(spec, popen=FakeProc)
    check("refuses a second concurrent launch", False, "IT LAUNCHED TWICE")
except CL.LaunchRefused as e:
    check("refuses a second concurrent launch", "already in flight" in str(e))
    check("explains why (one ledger, one chain)", "interleave" in str(e))
r.proc._rc = 0
check("finished state reported once the child exits",
      r.status()["state"] == "finished" and r.status()["returncode"] == 0)
st = r.launch(spec, now=1_700_000_001.0, popen=FakeProc)
check("a new launch is allowed after the first finished", st["state"] == "running")
# DO NOT let a fake pid reach the OS. FakeProc reports pid 4242 and reports itself alive,
# so the unpatched `out = r.stop()` called os.killpg(os.getpgid(4242), SIGTERM) -- sending
# SIGTERM to the process group of whatever real, same-uid process happened to hold pid 4242
# on the host. An adversarial reviewer demonstrated the kill by cycling ~100k forks until
# the kernel handed it that pid. The old assertion (`state == "stopped"`) passed either way,
# so the test both endangered the machine and proved nothing about the signal.
# Patching also makes it STRONGER: assert WHICH signal goes to WHICH group.
sent = []
_killpg, _getpgid = CL.os.killpg, CL.os.getpgid
CL.os.getpgid = lambda p: p            # never ask the OS about a pid we do not own
CL.os.killpg = lambda pg, sig: sent.append((pg, sig))
try:
    out = r.stop()
finally:
    CL.os.killpg, CL.os.getpgid = _killpg, _getpgid
check("stop reports stopped", out["state"] == "stopped")
check("stop signalled the child's process GROUP, not just the pid",
      sent == [(4242, signal.SIGTERM)], str(sent))
check("stop sent exactly one signal to a cleanly-terminating child",
      len(sent) == 1, str(sent))
check("no real process was signalled (the pid never reached the OS)",
      CL.os.killpg is _killpg and CL.os.getpgid is _getpgid)
check("stop on an idle runner is a no-op",
      CL.Runner().stop()["note"] == "nothing running")
_before = len(sent)
CL.os.getpgid = lambda p: p
CL.os.killpg = lambda pg, sig: sent.append((pg, sig))
try:
    CL.Runner().stop()
finally:
    CL.os.killpg, CL.os.getpgid = _killpg, _getpgid
check("stop on an idle runner signals nothing at all", len(sent) == _before,
      f"{len(sent) - _before} stray signal(s)")

print("== a scorer that vanishes between the check and the launch is refused")
CL.scorer_ledger = lambda: None            # gate open, scorer gone
rv = CL.Runner()
rv.scorer_ready = lambda socket_path=None: True
try:
    rv.launch(CL.parse({"proposer": "random", "tag": "vanish"}), popen=FakeProc)
    check("a vanished scorer is refused", False, "IT LAUNCHED against nothing")
except CL.LaunchRefused as e:
    check("a vanished scorer is refused", "stopped answering" in str(e))
    check("and says nothing was started", "Nothing was started" in str(e))
check("no child was started for it", not rv.alive())
check("scorer_ready itself now PINGS, not just stats a socket node",
      "scorer_ledger() is not None" in __import__("inspect").getsource(CL.Runner.scorer_ready))
CL.scorer_ledger = _real_scorer_ledger

print("== the module holds no networking (so this spec is testable without a socket)")
src = (Path(__file__).resolve().parents[1] / "agent" / "control_launch.py").read_text()
for bad in ("http.server", "socketserver", "BaseHTTPRequestHandler", "flask"):
    check(f"no {bad}", bad not in src)

# clean up the two log files the fake launches created
for h in r.history:
    (CL.LOG_DIR / h["log"]).unlink(missing_ok=True)

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
