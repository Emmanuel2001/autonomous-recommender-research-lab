"""Hard budget enforcement for a research run: iterations, wall-clock, tokens.

WHY THIS EXISTS
---------------
The problem statement caps a run at 50 iterations and 6 hours of agent wall-clock, and
wall-clock has replaced GPU-hours as the compute measure that gets scored. Neither cap
existed anywhere in this repo: the driver took `--iterations` (default 12) and a
`--max-seconds` soft budget, and per-iteration timing was recorded for sandboxed patches
only -- an in-process typed fit recorded none at all. So the one number now being scored
was the one number not being measured.

TWO CLOCKS, DELIBERATELY
------------------------
Elapsed time is measured with `time.monotonic()`, which cannot go backwards when the
system clock is adjusted mid-run -- a 6-hour cap enforced against a wall clock that
jumps is not a cap. The epoch timestamps are recorded separately, from `time.time()`,
because a run log needs to say *when* something happened and monotonic values are
meaningless outside the process that produced them.

SPENDING IS THE POINT
---------------------
Feasibility is scored in three coarse tiers, and only among entries that beat the
baseline. Finishing early buys nothing. So this module reports headroom as prominently as
it reports exhaustion: `remaining()` is what the loop should be looking at.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

#: The organisers' caps.
OFFICIAL_MAX_ITERATIONS = 50
OFFICIAL_MAX_WALL_SECONDS = 6 * 60 * 60

# Output allocations are deliberately role-specific. Planning and judging return compact
# structured decisions; code may legitimately return several complete source files.
ROLE_OUTPUT_CAPS = {"plan": 20_000, "judge": 12_000,
                    "code": 128_000, "reflect": 12_000}
ROLE_MINIMUM_OUTPUT = {"plan": 4_000, "judge": 3_000,
                       "code": 16_000, "reflect": 3_000}
ROLE_RESERVE_AFTER = {
    "plan": ROLE_MINIMUM_OUTPUT["judge"] + ROLE_MINIMUM_OUTPUT["code"]
            + ROLE_MINIMUM_OUTPUT["reflect"],
    "judge": ROLE_MINIMUM_OUTPUT["code"] + ROLE_MINIMUM_OUTPUT["reflect"],
    "code": ROLE_MINIMUM_OUTPUT["reflect"],
    "reflect": 0,
}


class BudgetExceeded(RuntimeError):
    """Raised only when a caller asks to start work the budget cannot cover."""


@dataclass
class IterationCost:
    """What one iteration actually cost. Every field is measured, none estimated."""
    iteration: int
    wall_seconds: float
    stages: dict = field(default_factory=dict)     # stage name -> seconds
    tokens_in: int = 0
    tokens_out: int = 0
    #: reasoning/thinking tokens. These are GENERATED tokens and are billed, so a budget that
    #: omitted them under-reported the quantity being capped -- a run could exhaust its real
    #: allowance while the counter said it had headroom.
    tokens_thinking: int = 0
    repairs: int = 0
    started_epoch: float = 0.0

    def to_record(self) -> dict:
        return {"iteration": self.iteration,
                "wall_seconds": round(self.wall_seconds, 3),
                "stages": {k: round(v, 3) for k, v in sorted(self.stages.items())},
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "tokens_thinking": self.tokens_thinking,
                "tokens_total": self.tokens_in + self.tokens_out + self.tokens_thinking,
                "repairs": self.repairs,
                "started_epoch": round(self.started_epoch, 3)}


class Budget:
    """Iteration, wall-clock and token caps, with per-iteration and per-stage timing.

    Usage:

        b = Budget()
        while not b.stop_reason():
            with b.iteration() as it:
                with it.stage("plan"):  ...
                with it.stage("fit"):   ...
                it.add_tokens(in_=1200, out_=300)
    """

    def __init__(self, max_iterations: int = OFFICIAL_MAX_ITERATIONS,
                 max_wall_seconds: float = OFFICIAL_MAX_WALL_SECONDS,
                 max_tokens: int | None = None, mode: str | None = None):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")
        self.max_iterations = int(max_iterations)
        self.max_wall_seconds = float(max_wall_seconds)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.mode = mode or ("fixed" if self.max_tokens is not None else "open")
        if self.mode not in ("fixed", "maximize", "open"):
            raise ValueError("budget mode must be fixed, maximize, or open")
        if self.mode in ("fixed", "maximize") and self.max_tokens is None:
            raise ValueError(f"budget mode {self.mode!r} requires max_tokens")
        if self.mode == "open" and self.max_tokens is not None:
            raise ValueError("open budget mode cannot also set max_tokens")
        self._t0 = time.monotonic()
        self.started_epoch = time.time()
        self.costs: list[IterationCost] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_thinking = 0
        self.call_preflights = 0
        self.call_preflight_refusals = 0
        self._call_stop_reason: str | None = None

    # ------------------------------------------------------------------ reading
    @property
    def iterations_done(self) -> int:
        return len(self.costs)

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def tokens(self) -> int:
        """Everything that counts against `max_tokens`, INCLUDING thinking tokens."""
        return self.tokens_in + self.tokens_out + self.tokens_thinking

    def remaining(self) -> dict:
        """Headroom. Negative values are clamped to zero -- a caller asking "how much is
        left" wants a budget, not an overdraft."""
        r = {"iterations": max(0, self.max_iterations - self.iterations_done),
             "wall_seconds": max(0.0, self.max_wall_seconds - self.elapsed())}
        if self.max_tokens is not None:
            r["tokens"] = max(0, self.max_tokens - self.tokens())
        return r

    def stop_reason(self) -> str | None:
        """The reason to stop, or None. Checked BEFORE starting an iteration."""
        if self.iterations_done >= self.max_iterations:
            return f"iteration_cap:{self.max_iterations}"
        if self.elapsed() >= self.max_wall_seconds:
            return (f"wall_clock:{self.max_wall_seconds:.0f}s "
                    f"(elapsed {self.elapsed():.0f}s)")
        if self.max_tokens is not None and self.tokens() >= self.max_tokens:
            return f"token_cap:{self.max_tokens}"
        if self._call_stop_reason:
            return self._call_stop_reason
        return None

    def reserve_call(self, role: str, requested_output: int,
                     prompt_bytes: int) -> int:
        """Return a safe output allowance or refuse before contacting a provider.

        ``prompt_bytes`` is a conservative upper bound on tokenizer pieces, unlike a
        chars/4 estimate that can undercount code and Unicode. The provider reports the
        exact input count afterward; this preflight exists to prevent a known-insufficient
        call from consuming money and leaving no room for coding or reflection.
        """
        self.call_preflights += 1
        requested = max(1, int(requested_output))
        role_cap = ROLE_OUTPUT_CAPS.get(role, requested)
        requested = min(requested, role_cap)
        if self.max_tokens is None:
            return requested
        remaining = self.max_tokens - self.tokens()
        reserve = ROLE_RESERVE_AFTER.get(role, 0)
        available = remaining - max(0, int(prompt_bytes)) - reserve
        minimum = min(requested, ROLE_MINIMUM_OUTPUT.get(role, 1_024))
        if available < minimum:
            self.call_preflight_refusals += 1
            self._call_stop_reason = (
                f"token_reserve:{role} needs at least {minimum} output tokens plus "
                f"{reserve} reserved downstream; {max(0, available)} available")
            raise BudgetExceeded(self._call_stop_reason)
        return min(requested, available)

    def would_exceed_wall(self, estimated_seconds: float) -> bool:
        """Whether starting work of this estimated length would run past the cap.

        Used to decline a full fit near the deadline rather than be killed halfway
        through one and record neither a result nor a clean stop.
        """
        return self.elapsed() + max(0.0, float(estimated_seconds)) > self.max_wall_seconds

    def summary(self) -> dict:
        """What the run log and the report both need. All measured."""
        per = [c.wall_seconds for c in self.costs]
        stages: dict[str, float] = {}
        for c in self.costs:
            for k, v in c.stages.items():
                stages[k] = stages.get(k, 0.0) + v
        return {"iterations": self.iterations_done,
                "max_iterations": self.max_iterations,
                "wall_seconds": round(self.elapsed(), 2),
                "max_wall_seconds": self.max_wall_seconds,
                "wall_hours": round(self.elapsed() / 3600.0, 3),
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "tokens_thinking": self.tokens_thinking,
                "tokens_total": self.tokens(),
                # Reconciliation: the run total must equal the sum over iterations. If these
                # ever disagree, one of the two is not counting a channel.
                "tokens_total_from_iterations": sum(
                    c.tokens_in + c.tokens_out + c.tokens_thinking for c in self.costs),
                "max_tokens": self.max_tokens,
                "mode": self.mode,
                "call_preflights": self.call_preflights,
                "call_preflight_refusals": self.call_preflight_refusals,
                "iteration_wall_seconds": [round(x, 2) for x in per],
                "iteration_wall_mean": round(sum(per) / len(per), 2) if per else None,
                "iteration_wall_max": round(max(per), 2) if per else None,
                "stage_seconds": {k: round(v, 2) for k, v in sorted(stages.items())},
                "repairs": sum(c.repairs for c in self.costs),
                "started_epoch": round(self.started_epoch, 3),
                "remaining": {k: (round(v, 2) if isinstance(v, float) else v)
                              for k, v in self.remaining().items()}}

    # ------------------------------------------------------------------ writing
    def iteration(self, *, strict: bool = True) -> "_Iteration":
        """Begin one iteration. With strict=True (the default) an exhausted budget raises
        rather than silently allowing a 51st iteration."""
        if strict:
            why = self.stop_reason()
            if why is not None:
                raise BudgetExceeded(why)
        return _Iteration(self, self.iterations_done + 1)

    def add_tokens(self, in_: int = 0, out_: int = 0, thinking: int = 0) -> None:
        self.tokens_in += int(in_)
        self.tokens_out += int(out_)
        self.tokens_thinking += int(thinking)


class _Iteration:
    """Context manager recording one iteration's wall-clock, stages and tokens.

    Cost is recorded in `__exit__` whether or not the body raised: an iteration that
    crashed still consumed wall-clock, and hiding that would flatter the compute report.
    """

    def __init__(self, budget: Budget, number: int):
        self.budget = budget
        self.cost = IterationCost(iteration=number, wall_seconds=0.0,
                                  started_epoch=time.time())
        self._t0 = 0.0

    def __enter__(self) -> "_Iteration":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cost.wall_seconds = time.monotonic() - self._t0
        self.budget.costs.append(self.cost)
        return False                                  # never swallow an exception

    def elapsed(self) -> float:
        """Wall-clock for this iteration SO FAR, from the live monotonic clock."""
        return time.monotonic() - self._t0

    def record_now(self) -> dict:
        """`to_record()` with `wall_seconds` taken from the live clock.

        WHY THIS EXISTS. `__exit__` is what normally fills `wall_seconds`, so anything that
        serialises the cost from INSIDE the `with` block reads 0.0. The driver records its
        per-iteration ledger entry inside the block -- deliberately, so the entry lands in
        sequence with that iteration's other records -- and therefore wrote
        `wall_seconds: 0.0` while the stages beneath it totalled ~54 s. The run summary was
        right and the per-iteration record was wrong, which is the worse way round: the
        summary is a derived number and the record is the evidence.

        The value here is elapsed-at-record-time, a hair smaller than the final value
        `__exit__` stores, because the record is written near the end of the iteration.
        Both are correct for their moment; neither is 0.
        """
        self.cost.wall_seconds = self.elapsed()
        return self.cost.to_record()

    def stage(self, name: str) -> "_Stage":
        return _Stage(self, name)

    def add_tokens(self, in_: int = 0, out_: int = 0, thinking: int = 0) -> None:
        self.cost.tokens_in += int(in_)
        self.cost.tokens_out += int(out_)
        self.cost.tokens_thinking += int(thinking)
        self.budget.add_tokens(in_, out_, thinking)

    def note_repair(self) -> None:
        self.cost.repairs += 1


class _Stage:
    """Times one stage. Repeated entries with the same name ACCUMULATE, so a repair loop
    that re-enters `code` three times reports the total spent coding, not the last try."""

    def __init__(self, it: _Iteration, name: str):
        self.it, self.name, self._t0 = it, name, 0.0

    def __enter__(self) -> "_Stage":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        d = time.monotonic() - self._t0
        self.it.cost.stages[self.name] = self.it.cost.stages.get(self.name, 0.0) + d
        return False
