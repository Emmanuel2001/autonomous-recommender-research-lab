"""Validation and process control for the UI. Deliberately separate from the HTTP layer.

A web UI that starts a training run is remote code execution by design, so the rules
about WHAT may be launched are kept here, in a module with no networking in it, where
they can be unit-tested without a socket. tests/test_control.py is the real spec.

Four invariants, each for a concrete reason:

  1. NO SIGNING KEY. The UI launches the driver; the driver never reads the signing key
     (only core.scorer does), so the UI can run without it and therefore cannot mint a
     ledger entry. `child_env` strips it, along with every other secret, so this holds
     even if the operator exported one into the shell that started the UI.

  2. NEVER TOUCH A LEDGER AT ALL. The UI does not choose, open, or name the ledger --
     the SCORER owns it, resolving it from its own environment at startup. This was
     originally written the other way round (a fresh path per launch, passed to the child
     via TRACK2_LEDGER) and that was simply wrong: the child asks the scorer to record,
     and the scorer writes where IT was told to, so the generated path was never created
     and the UI displayed a filename that did not exist. The UI now ASKS the scorer where
     it is writing, which is the only correct source. Not being able to redirect the chain
     is the property that matters, and it is stronger this way.

  3. ALLOWLIST, NOT PASSTHROUGH. Every field is checked against the driver's own choices
     and numeric bounds. argv is a list, so there is no shell to inject into -- the risk
     being managed here is cost and resource exhaustion (a 10-million-token run), not
     quoting.

  4. ONE RUN AT A TIME, AND ACTUALLY SO. The scorer socket and its ledger are single
     global resources; two concurrent drivers interleave proposals into one chain and make
     both runs unreadable. A launch is refused while a child is alive, rather than queued
     and silently reordered.
     The first version checked `alive()` and then called `popen()` with nothing in between
     holding the two together, while `control.py` drives a single module-level Runner from
     a ThreadingHTTPServer. An adversarial reviewer released two pre-connected clients
     through a barrier and got two live drivers on 3 of 3 attempts, sharing one log path.
     The invariant was documented and untrue. `launch()` and `stop()` now hold a lock
     across the whole check-and-act, which is the only thing that makes the sentence above
     a fact rather than an intention.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "runs" / "ledgers"
LOG_DIR = ROOT / "runs" / "ui_logs"

# Mirrors agent/official_driver.py's argparse choices. Duplicated on purpose: the UI
# must not be able to send a value the driver would reject, and tests assert the two
# lists agree, so a drift is a test failure rather than a runtime error at iteration 1.
PROPOSERS = ("random", "gemini", "ollama", "groq", "anthropic")
LOSSES = ("pointwise", "pairwise", "listwise")
CONVERGENCE = ("official", "off")
ISOLATION = ("seatbelt", "docker", "subprocess", "none")
#: Which loop to run. "official" chooses from a typed action space; "research" writes the
#: strategy source. Same mirroring rule: the drift guard asserts these against BOTH
#: drivers' argparse, so adding a choice to one and not the other is a test failure.
DRIVERS = ("official", "research")
BACKENDS = ("gemini", "ollama", "groq", "anthropic")
RESEARCH_POLICIES = ("tournament", "single")
SEARCH_MODES = ("standard", "discovery")
SEED_PROFILES = ("inherited", "minimal", "accelerated")
FRAMEWORKS = ("auto", "numpy", "torch")
DEVICES = ("auto", "cpu", "mps", "cuda")
ACCELERATOR_POLICIES = ("off", "allow", "prefer", "require")

# Bounds are cost controls, not style. An LLM arm at 400 iterations is real money, and a
# --max-seconds of a week is a machine nobody gets back.
BOUNDS = {
    "iterations": (1, 200, int),
    "seed": (0, 2**31 - 1, int),
    "model_seed": (0, 2**31 - 1, int),
    "max_seconds": (10.0, 86_400.0, float),
    "max_tokens": (1_000, 20_000_000, int),
    "llm_max_output_tokens": (1_024, 128_000, int),
    "patch_seconds": (10.0, 3_600.0, float),
    # the organisers' cap is 6 h; the upper bound here is that, not a round number
    "max_wall_seconds": (60.0, 21_600.0, float),
    "max_repairs": (0, 5, int),
    "tournament_size": (6, 12, int),
    "max_coverage_repairs": (0, 3, int),
    "epsilon_greedy": (0.0, 1.0, float),
}

# A model identifier reaches an HTTP backend, never a shell. Still constrained, because
# an unconstrained string in a launch request is how a field becomes a foothold later.
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-/]{0,80}$")
THINKING_RE = re.compile(r"^[a-z]{1,16}$")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")

# Secrets that must not cross into a child the UI starts. TRACK2_LLM_API_KEY and
# GEMINI_API_KEY are the exception: an LLM arm cannot work without one, so they pass
# ONLY when the operator chose an LLM proposer, and never otherwise.
NEVER_INHERIT = ("TRACK2_SIGNING_KEY", "TRACK2_KEY_FILE", "AWS_SECRET_ACCESS_KEY",
                 "AWS_SESSION_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY")
LLM_KEYS = ("TRACK2_LLM_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY")


class LaunchRefused(ValueError):
    """A refusal is a normal outcome, not an error: it is the allowlist working."""


@dataclass
class Spec:
    proposer: str = "random"
    model: str | None = None
    thinking_level: str | None = None
    thought_summaries: bool = False
    iterations: int = 12
    #: SEARCH seed -- the driver's RNG and the proposer's sampling.
    seed: int = 0
    #: MODEL seed -- reaches Config.seed and therefore the fit's RNG. Separate because the
    #: single field labelled "seed" in this UI only ever seeded the SEARCH, so every model
    #: was fitted at seed 0 whatever was entered. Default 0 keeps that behaviour.
    model_seed: int = 0
    convergence: str = "official"
    start_loss: str = "pointwise"
    patch_isolation: str = "seatbelt"
    max_seconds: float = 5400.0
    max_tokens: int = 5_000_000
    #: Per LLM call, distinct from the total run budget above.
    llm_max_output_tokens: int = 64_000
    patch_seconds: float = 900.0
    epsilon_greedy: float = 0.25
    allow_random_log: bool = False
    tag: str = "ui"
    # --- code-first loop only
    driver: str = "official"
    backend: str = "gemini"
    max_wall_seconds: float = 21_600.0
    max_repairs: int = 3
    research_policy: str = "tournament"
    search_mode: str = "standard"
    tournament_size: int = 6
    max_coverage_repairs: int = 1
    seed_profile: str = "inherited"
    planner_model: str | None = None
    judge_model: str | None = None
    coder_model: str | None = None
    reflector_model: str | None = None
    framework: str = "auto"
    device: str = "auto"
    accelerator_policy: str = "prefer"
    no_device_fallback: bool = False
    task_spec: str = ""
    #: JSON object merged into the seed tree's config.json. Empty means start from the
    #: unmodified official baseline, which is what an unqualified autonomy claim requires.
    seed_config: str = ""

    def argv(self) -> list[str]:
        """The child's command line. Takes no ledger: the ledger is never passed on the
        command line, because it is not the caller's to choose."""
        if self.driver == "research":
            av = ["python3", "-u", "-m", "agent.research_driver",
                  "--backend", self.backend,
                  "--iterations", str(self.iterations),
                  "--convergence", self.convergence,
                  "--isolation", self.patch_isolation,
                  "--max-wall-seconds", str(self.max_wall_seconds),
                  "--fit-seconds", str(self.patch_seconds),
                  "--max-repairs", str(self.max_repairs),
                  "--max-tokens", str(self.max_tokens),
                  "--llm-max-output-tokens", str(self.llm_max_output_tokens),
                  "--research-policy", self.research_policy,
                  "--search-mode", self.search_mode,
                  "--tournament-size", str(self.tournament_size),
                  "--max-coverage-repairs", str(self.max_coverage_repairs),
                  "--seed-profile", self.seed_profile,
                  "--framework", self.framework,
                  "--device", self.device,
                  "--accelerator-policy", self.accelerator_policy]
            if self.model:
                av += ["--model", self.model]
            if self.thinking_level:
                av += ["--thinking-level", self.thinking_level]
            if self.thought_summaries:
                av += ["--thought-summaries"]
            for flag, value in (("--planner-model", self.planner_model),
                                ("--judge-model", self.judge_model),
                                ("--coder-model", self.coder_model),
                                ("--reflector-model", self.reflector_model)):
                if value:
                    av += [flag, value]
            if self.no_device_fallback:
                av += ["--no-device-fallback"]
            if self.task_spec:
                av += ["--task-spec", self.task_spec]
            if self.seed_config:
                av += ["--seed-config", self.seed_config]
            return av
        av = ["python3", "-u", "-m", "agent.official_driver",
              "--proposer", self.proposer,
              "--iterations", str(self.iterations),
              "--seed", str(self.seed),
              "--model-seed", str(self.model_seed),
              "--convergence", self.convergence,
              "--start-loss", self.start_loss,
              "--patch-isolation", self.patch_isolation,
              "--max-seconds", str(self.max_seconds),
              "--max-tokens", str(self.max_tokens),
              "--patch-seconds", str(self.patch_seconds),
              "--epsilon-greedy", str(self.epsilon_greedy)]
        if self.model:
            av += ["--model", self.model]
        if self.thinking_level:
            av += ["--thinking-level", self.thinking_level]
        if self.allow_random_log:
            av += ["--allow-random-log"]
        return av


def _one(form: dict, key: str) -> str:
    """Take a single value. A duplicated field in a form arrives as a list, and quietly
    using the first is how a validated value gets bypassed by appending a second."""
    v = form.get(key)
    if isinstance(v, (list, tuple)):
        if len(v) != 1:
            raise LaunchRefused(f"{key}: expected one value, got {len(v)}")
        v = v[0]
    return "" if v is None else str(v)


def _enum(form: dict, key: str, allowed: tuple, default: str) -> str:
    raw = _one(form, key).strip()
    if not raw:
        return default
    if raw not in allowed:
        raise LaunchRefused(f"{key}: {raw!r} is not one of {list(allowed)}")
    return raw


def _num(form: dict, key: str, default):
    raw = _one(form, key).strip()
    lo, hi, cast = BOUNDS[key]
    if not raw:
        return default
    # Parse, then check FINITENESS, then cast. The order matters: the old code did
    # `cast(float(raw))` first, so `iterations=inf` raised OverflowError from int(inf) --
    # not in the except clause, not a LaunchRefused, and therefore straight past
    # control.py's `except LaunchRefused` to a dead connection instead of a clean refusal.
    # The finiteness guard below was unreachable for every int field.
    try:
        f = float(raw)
    except (TypeError, ValueError):
        raise LaunchRefused(f"{key}: {raw!r} is not a number") from None
    if f != f or f in (float("inf"), float("-inf")):
        raise LaunchRefused(f"{key}: {raw!r} is not finite")
    v = cast(f)
    if not (lo <= v <= hi):
        raise LaunchRefused(f"{key}: {v} is outside [{lo}, {hi}]")
    return v


def parse(form: dict) -> Spec:
    """Untrusted form -> validated Spec, or LaunchRefused. No side effects.

    An EMPTY form is refused rather than treated as "all defaults". Every field defaults, so
    `parse({})` used to return a perfectly launchable Spec -- which meant any request whose
    body failed to arrive (a chunked POST this server cannot dechunk, a proxy that dropped
    it) started a real 12-iteration run with settings nobody chose. A launch has to be asked
    for explicitly.
    """
    if not form:
        raise LaunchRefused(
            "empty request: no fields were received, and every field has a default, so "
            "continuing would start a run with settings nobody chose. If you meant the "
            "defaults, send them explicitly (at minimum proposer=random).")
    s = Spec()
    s.driver = _enum(form, "driver", DRIVERS, s.driver)
    s.proposer = _enum(form, "proposer", PROPOSERS, s.proposer)
    s.convergence = _enum(form, "convergence", CONVERGENCE, s.convergence)
    s.start_loss = _enum(form, "start_loss", LOSSES, s.start_loss)
    s.patch_isolation = _enum(form, "patch_isolation", ISOLATION, s.patch_isolation)
    s.backend = _enum(form, "backend", BACKENDS, s.backend)
    s.research_policy = _enum(
        form, "research_policy", RESEARCH_POLICIES, s.research_policy)
    s.search_mode = _enum(form, "search_mode", SEARCH_MODES, s.search_mode)
    s.seed_profile = _enum(form, "seed_profile", SEED_PROFILES, s.seed_profile)
    s.framework = _enum(form, "framework", FRAMEWORKS, s.framework)
    s.device = _enum(form, "device", DEVICES, s.device)
    s.accelerator_policy = _enum(
        form, "accelerator_policy", ACCELERATOR_POLICIES, s.accelerator_policy)
    for k in ("iterations", "seed", "model_seed", "max_seconds", "max_tokens",
              "llm_max_output_tokens",
              "patch_seconds", "epsilon_greedy", "max_wall_seconds", "max_repairs",
              "tournament_size", "max_coverage_repairs"):
        setattr(s, k, _num(form, k, getattr(s, k)))

    m = _one(form, "model").strip()
    if m:
        if not MODEL_RE.match(m):
            raise LaunchRefused(f"model: {m!r} is not an acceptable identifier")
        s.model = m
    for attr in ("planner_model", "judge_model", "coder_model", "reflector_model"):
        value = _one(form, attr).strip()
        if value:
            if not MODEL_RE.match(value):
                raise LaunchRefused(f"{attr}: {value!r} is not an acceptable identifier")
            setattr(s, attr, value)
    t = _one(form, "thinking_level").strip()
    if t:
        if not THINKING_RE.match(t):
            raise LaunchRefused(f"thinking_level: {t!r} is not an acceptable value")
        s.thinking_level = t
    tag = _one(form, "tag").strip() or "ui"
    if not TAG_RE.match(tag):
        raise LaunchRefused(f"tag: {tag!r} must be lowercase letters, digits, - or _")
    s.tag = tag

    s.allow_random_log = _one(form, "allow_random_log").strip().lower() in ("1", "on", "true", "yes")
    s.no_device_fallback = _one(form, "no_device_fallback").strip().lower() in (
        "1", "on", "true", "yes")
    s.thought_summaries = _one(form, "thought_summaries").strip().lower() in (
        "1", "on", "true", "yes")

    task_spec = _one(form, "task_spec").strip()
    if task_spec:
        if s.driver != "research":
            raise LaunchRefused("task_spec applies only to the code-first research loop")
        p = Path(task_spec)
        p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
        try:
            p.relative_to(ROOT)
        except ValueError:
            raise LaunchRefused("task_spec must be a JSON file inside this repository")
        if p.suffix.lower() != ".json" or not p.is_file():
            raise LaunchRefused(f"task_spec is not an existing JSON file: {p}")
        s.task_spec = str(p)

    sc = _one(form, "seed_config").strip()
    if sc:
        if s.driver != "research":
            raise LaunchRefused("seed_config applies only to the code-first loop")
        # Validated HERE as well as in the driver: the UI's job is to be unable to send a
        # value the driver would reject, and a bad override should cost a form redisplay
        # rather than a launched process that exits.
        import json as _json
        try:
            ov = _json.loads(sc)
        except ValueError as e:
            raise LaunchRefused(f"seed_config is not valid JSON: {e}") from None
        if not isinstance(ov, dict) or not ov:
            raise LaunchRefused("seed_config must be a non-empty JSON object")
        if len(sc) > 400:
            raise LaunchRefused("seed_config is too long to be a config override")
        s.seed_config = _json.dumps(ov, sort_keys=True)

    if s.driver == "research":
        if s.backend != "ollama" and not s.model:
            raise LaunchRefused(f"backend {s.backend!r} needs an explicit --model")
        if s.search_mode == "discovery":
            if s.research_policy != "tournament":
                raise LaunchRefused("discovery search requires research_policy=tournament")
            if s.convergence != "off":
                raise LaunchRefused("discovery search requires convergence=off; it is "
                                    "diagnostic evidence, not an official run")
            if s.iterations < 3:
                raise LaunchRefused("discovery search requires at least 3 iterations for "
                                    "the architecture funding floor")
    elif s.search_mode != "standard":
        raise LaunchRefused("search_mode applies only to the code-first research loop")
    elif s.proposer != "random" and not s.model:
        raise LaunchRefused(f"proposer {s.proposer!r} needs an explicit --model")
    if s.patch_isolation == "none":
        raise LaunchRefused("patch_isolation 'none' runs generated code with no boundary; "
                            "the UI will not offer it. Use the CLI deliberately if you "
                            "really mean it.")
    return s


def log_path(spec: Spec, now: float, existing=None) -> Path:
    """A FRESH path for this run's captured stdout -- the one file the UI does own.

    `existing` is injectable so the collision branch is proved by a test rather than
    trusted. Note what this is NOT: it is not the ledger. The UI has no say in that.
    """
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now))
    who = spec.backend if spec.driver == "research" else spec.proposer
    base = f"ui_{spec.tag}_{spec.driver}_{who}_s{spec.seed}_{stamp}"
    seen = existing if existing is not None else {p.name for p in LOG_DIR.glob("*.log")}
    name, n = f"{base}.log", 1
    while name in seen:
        name, n = f"{base}_{n}.log", n + 1
    return LOG_DIR / name


def child_env(spec: Spec, base: dict | None = None) -> dict:
    """The child's environment. Strips every secret, then adds back only an LLM key, and
    only for an LLM arm. The signing key is never added back at all -- the driver has no
    use for it, so a UI-started run structurally cannot sign a ledger entry itself.

    TRACK2_LEDGER is deliberately NOT set. Setting it here looked like it scoped the run
    to its own chain and did nothing of the kind: the scorer resolves the ledger from its
    own environment, and a child with a different value only made the two processes
    disagree about which file the ledger was. Leaving it unset means the child inherits
    whatever the operator's shell has, and the driver asks the scorer regardless.
    """
    env = dict(os.environ if base is None else base)
    for k in NEVER_INHERIT:
        env.pop(k, None)
    needs_llm = spec.driver == "research" or spec.proposer != "random"
    if not needs_llm:
        for k in LLM_KEYS:
            env.pop(k, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def scorer_ledger() -> str | None:
    """Ask the scorer which ledger it is writing. None when it is unreachable."""
    try:
        from core.scorer import request
        r = request({"op": "ping"})
        return r.get("ledger") if r.get("ok") else None
    except Exception:                                              # noqa: BLE001
        return None


@dataclass
class Runner:
    """Owns at most one child. Holds no key and never opens a ledger.

    Thread-safe by necessity, not by taste: control.py serves this from a
    ThreadingHTTPServer with one module-level instance, so every mutating method is a
    critical section.
    """
    #: guards the whole check-and-act in launch() and stop(). Excluded from equality and
    #: repr so the dataclass stays comparable and printable.
    _lock: object = field(default_factory=threading.Lock, repr=False, compare=False)
    proc: subprocess.Popen | None = None
    spec: Spec | None = None
    ledger: Path | None = None
    log: Path | None = None
    started: float = 0.0
    history: list = field(default_factory=list)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> dict:
        if self.proc is None:
            return {"state": "idle"}
        rc = self.proc.poll()
        return {"state": "running" if rc is None else "finished",
                "returncode": rc, "pid": self.proc.pid,
                "proposer": self.spec.proposer if self.spec else None,
                "model": self.spec.model if self.spec else None,
                "ledger": self.ledger.name if self.ledger else None,
                "log": self.log.name if self.log else None,
                "seconds": round(time.time() - self.started, 1)}

    def scorer_ready(self, socket_path: Path | None = None) -> bool:
        """Is a scorer actually ANSWERING? Not merely: does a socket node exist.

        The scorer unlinks its socket only in serve()'s `finally`, so SIGKILL, an OOM, or a
        reboot leaves the node behind. A bare `is_socket()` therefore opened the launch gate
        for a scorer that had been dead for days -- and two lines into `launch()` the real
        ping in `scorer_ledger()` returned None, which was silently discarded and displayed
        as `ledger: None` for a run that had just been started against nothing.
        Stat first (cheap, and avoids a connect on the common absent case), then ping.
        """
        p = socket_path or Path(os.environ.get("TRACK2_SOCKET")
                                or ROOT / "runs" / "scorer.sock")
        if not p.is_socket():
            return False
        return scorer_ledger() is not None

    def launch(self, spec: Spec, now: float | None = None,
               popen=subprocess.Popen) -> dict:
        # The lock must span the alive() check, log_path(), the open(), AND the popen():
        # anything less leaves a window where two requests both see "idle". It also must
        # cover history.append so the recorded order matches the launch order.
        with self._lock:
            return self._launch_locked(spec, now, popen)

    def _launch_locked(self, spec: Spec, now: float | None, popen) -> dict:
        if self.alive():
            raise LaunchRefused(
                f"a run is already in flight (pid {self.proc.pid}). The scorer and its "
                f"ledger are single resources; two drivers would interleave into one "
                f"chain and make both runs unreadable.")
        if not self.scorer_ready():
            raise LaunchRefused(
                "no scorer is listening. Start it in its own terminal -- it holds the "
                "signing key, and this UI deliberately does not:\n"
                "  python3 -m core.scorer serve --task official")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time() if now is None else now
        log = log_path(spec, now)
        # The ledger is the scorer's; ask, never assume. Recorded at launch so the status
        # line points at the file that will actually contain this run.
        led = scorer_ledger()
        if led is None:
            # The gate passed and the scorer is gone: it died between the two calls, or the
            # socket node is stale. Refusing beats starting a driver whose every evaluation
            # will be refused, and beats displaying `ledger: None` as though that were a
            # ledger.
            raise LaunchRefused(
                "the scorer stopped answering between the readiness check and the launch. "
                "Nothing was started. Check it is still running:\n"
                "  python3 -m core.scorer serve --task official")
        fh = open(log, "wb", buffering=0)
        self.proc = popen(spec.argv(), cwd=str(ROOT), stdout=fh, stderr=fh,
                          env=child_env(spec), start_new_session=True)
        self.spec, self.log, self.started = spec, log, now
        self.ledger = Path(led) if led else None
        self.history.append({"ledger": (Path(led).name if led else None),
                             "log": log.name, "proposer": spec.proposer,
                             "model": spec.model, "started": now})
        return self.status()

    def stop(self) -> dict:
        """Terminate only the child this Runner started, by process group, so the
        driver's own children (a sandboxed patch) go with it rather than surviving.

        Locked against launch(): otherwise a stop can read `self.proc`, a launch can
        replace it, and the stop then signals the wrong child -- or the new one.
        """
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> dict:
        if not self.alive():
            return {"state": "idle", "note": "nothing running"}
        pid = self.proc.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()
        return {"state": "stopped", "pid": pid, "returncode": self.proc.poll()}
