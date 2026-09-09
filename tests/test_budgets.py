"""Tests for the hard iteration / wall-clock / token caps.

Wall-clock is now the scored compute measure, so the two things tested hardest are the
boundary (a cap of N must stop at N, not N+1) and the clock choice (elapsed must not move
when the system clock does -- a 6-hour cap enforced against an adjustable clock is not a
cap).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import budgets as B

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

def raises(exc, fn, *a, **k):
    try: fn(*a, **k)
    except exc as e: return str(e) or "raised"
    except Exception as e: return f"WRONG-TYPE:{type(e).__name__}"
    return ""

# 1 -- the organisers' numbers, in one place -------------------------------------
check("iteration cap is the organisers' 50", B.OFFICIAL_MAX_ITERATIONS == 50)
check("wall cap is the organisers' 6 hours", B.OFFICIAL_MAX_WALL_SECONDS == 21600)

# 2 -- the iteration boundary ---------------------------------------------------
b = B.Budget(max_iterations=3, max_wall_seconds=600)
check("a fresh budget has no stop reason", b.stop_reason() is None)
for i in range(3):
    with b.iteration() as it:
        pass
check("three iterations were recorded", b.iterations_done == 3)
check("the cap fires AT the limit, not after it",
      (b.stop_reason() or "").startswith("iteration_cap:3"), b.stop_reason())
check("remaining iterations is 0, never negative", b.remaining()["iterations"] == 0)
check("a 4th strict iteration is refused",
      raises(B.BudgetExceeded, b.iteration) != "")
check("strict=False still permits an overrun deliberately",
      b.iteration(strict=False) is not None)

# 3 -- wall clock ---------------------------------------------------------------
w = B.Budget(max_iterations=99, max_wall_seconds=0.05)
check("wall budget starts unexhausted", w.stop_reason() is None)
time.sleep(0.08)
check("the wall cap fires once elapsed", (w.stop_reason() or "").startswith("wall_clock:"),
      w.stop_reason())
check("remaining wall_seconds clamps at 0", w.remaining()["wall_seconds"] == 0.0)
check("an iteration is refused past the wall cap",
      raises(B.BudgetExceeded, w.iteration) != "")

fut = B.Budget(max_wall_seconds=100)
check("would_exceed_wall is False well inside the budget", not fut.would_exceed_wall(1))
check("would_exceed_wall is True for work longer than the budget",
      fut.would_exceed_wall(200))
check("would_exceed_wall treats a negative estimate as zero",
      not fut.would_exceed_wall(-5))

# 4 -- elapsed() must not follow the system clock -------------------------------
mono = B.Budget(max_wall_seconds=600)
real_time = time.time
try:
    time.time = lambda: real_time() + 10_000_000      # system clock jumps forward
    jumped = mono.elapsed()
finally:
    time.time = real_time
check("elapsed() ignores a system-clock jump (monotonic, not wall)", jumped < 5,
      f"elapsed={jumped:.3f}s")
check("started_epoch is still a real timestamp (for the run log)",
      mono.started_epoch > 1_700_000_000)

# 5 -- tokens -------------------------------------------------------------------
t = B.Budget(max_iterations=99, max_wall_seconds=600, max_tokens=100)
with t.iteration() as it:
    it.add_tokens(in_=40, out_=10)
check("tokens accumulate on the budget", t.tokens() == 50)
check("tokens split in/out", (t.tokens_in, t.tokens_out) == (40, 10))
check("token headroom is reported", t.remaining()["tokens"] == 50)
with t.iteration() as it:
    it.add_tokens(in_=50, out_=0)
check("the token cap fires at the limit", (t.stop_reason() or "").startswith("token_cap:"),
      t.stop_reason())
check("no token key is reported when there is no token cap",
      "tokens" not in B.Budget(max_wall_seconds=60).remaining())

# 5b -- THINKING tokens count against the cap, and everything reconciles --------
# Thinking tokens are generated tokens and are billed. Omitting them from the cap meant a run
# could exhaust its real allowance while the counter reported headroom.
th = B.Budget(max_iterations=99, max_wall_seconds=600, max_tokens=100)
with th.iteration() as it:
    it.add_tokens(in_=0, out_=0, thinking=100)
check("thinking tokens alone can exhaust the budget",
      (th.stop_reason() or "").startswith("token_cap:"), str(th.stop_reason()))
check("thinking tokens are counted in the total", th.tokens() == 100)
check("they are tracked in their own channel",
      th.tokens_thinking == 100 and th.tokens_in == 0 and th.tokens_out == 0)
check("a further strict iteration is refused",
      raises(B.BudgetExceeded, th.iteration) != "")

rec3 = B.Budget(max_iterations=9, max_wall_seconds=600, max_tokens=10_000)
for tri in ((10, 5, 3), (7, 2, 11), (0, 0, 9)):
    with rec3.iteration() as it:
        it.add_tokens(in_=tri[0], out_=tri[1], thinking=tri[2])
want = sum(sum(t) for t in ((10, 5, 3), (7, 2, 11), (0, 0, 9)))
sm3 = rec3.summary()
check("the run total equals input+output+thinking", rec3.tokens() == want, str(rec3.tokens()))
check("the summary total agrees with the run total", sm3["tokens_total"] == want)
check("the summary reconciles against the per-iteration records",
      sm3["tokens_total_from_iterations"] == want,
      f"{sm3['tokens_total_from_iterations']} vs {want}")
check("each iteration record carries its own reconciled total",
      all(c.to_record()["tokens_total"]
          == c.tokens_in + c.tokens_out + c.tokens_thinking for c in rec3.costs))
check("the summary breaks the total into three channels",
      sm3["tokens_in"] + sm3["tokens_out"] + sm3["tokens_thinking"] == want)
check("thinking is not silently folded into output",
      sm3["tokens_thinking"] == 23 and sm3["tokens_out"] == 7,
      f"thinking={sm3['tokens_thinking']} out={sm3['tokens_out']}")

# 5c -- provider calls are preflighted and role capacity is reserved -----------
pre = B.Budget(max_iterations=9, max_wall_seconds=600, max_tokens=100_000)
plan_allowance = pre.reserve_call("plan", 64_000, prompt_bytes=20_000)
check("planning uses a role-specific output cap",
      plan_allowance == B.ROLE_OUTPUT_CAPS["plan"], str(plan_allowance))
check("planning reserves later judge, code and reflection capacity",
      plan_allowance + 20_000 + B.ROLE_RESERVE_AFTER["plan"] <= 100_000)
pre.add_tokens(in_=20_000, out_=plan_allowance)
code_allowance = pre.reserve_call("code", 64_000, prompt_bytes=10_000)
check("coder receives the largest safe remainder",
      code_allowance >= B.ROLE_MINIMUM_OUTPUT["code"], str(code_allowance))

too_small = B.Budget(max_iterations=9, max_wall_seconds=600, max_tokens=25_000)
refusal = raises(B.BudgetExceeded, too_small.reserve_call,
                 "plan", 64_000, 10_000)
check("a known-insufficient call is refused before provider contact",
      "token_reserve:plan" in refusal, refusal)
check("preflight refusal becomes the run stop reason",
      (too_small.stop_reason() or "").startswith("token_reserve:plan"),
      str(too_small.stop_reason()))
check("preflight refusal spends no tokens", too_small.tokens() == 0)
check("preflight activity is reported",
      too_small.summary()["call_preflight_refusals"] == 1)

opened = B.Budget(max_iterations=2, max_wall_seconds=60, mode="open")
check("open mode has no token limit",
      opened.reserve_call("code", 128_000, 9_999_999) == 128_000)
maximized = B.Budget(max_iterations=2, max_wall_seconds=60,
                     max_tokens=200_000, mode="maximize")
check("maximize mode remains bounded by the declared allowance",
      maximized.mode == "maximize" and maximized.max_tokens == 200_000)

# 6 -- stages accumulate, and a crashed iteration still costs -------------------
s = B.Budget(max_iterations=9, max_wall_seconds=600)
with s.iteration() as it:
    for _ in range(3):
        with it.stage("code"):
            time.sleep(0.01)
    with it.stage("fit"):
        time.sleep(0.01)
    it.note_repair(); it.note_repair()
cost = s.costs[-1]
check("a repeated stage accumulates rather than overwriting",
      cost.stages["code"] > cost.stages["fit"] * 2,
      f"code={cost.stages['code']:.3f} fit={cost.stages['fit']:.3f}")
check("repairs are counted", cost.repairs == 2)
check("iteration wall covers all its stages",
      cost.wall_seconds >= cost.stages["code"] + cost.stages["fit"])

# A record serialised from INSIDE the iteration context must still carry a real wall time.
# The old assertion checked budget.costs AFTER the loop, where __exit__ had already filled
# the field -- so it passed while the ledger was being written with wall_seconds: 0.0.
inner = B.Budget(max_iterations=3, max_wall_seconds=600)
with inner.iteration() as it:
    with it.stage("fit"):
        time.sleep(0.03)
    with it.stage("score"):
        time.sleep(0.01)
    rec_inside = it.record_now()
    # record_now() rounds to 3dp on the way into the record, so keep the unrounded value
    # for the monotonicity check below -- otherwise a snapshot that rounds UP (0.0476 ->
    # 0.048) is compared against a later, larger, unrounded value and loses.
    snap_unrounded = float(it.cost.wall_seconds)
    raw_inside = it.cost.to_record()
    stage_sum_inside = sum(it.cost.stages.values())
check("record_now() inside the context reports a NON-ZERO wall time",
      rec_inside["wall_seconds"] > 0, str(rec_inside["wall_seconds"]))
# `to_record()` rounds the total to 3dp and each stage to 3dp independently, so a sum of
# rounded parts can exceed a rounded whole by up to half a millisecond per stage. The
# invariant is about elapsed time, so it is asserted on the UNROUNDED values, with the
# rounded form allowed exactly that much slack.
_tol = 0.0005 * (len(it.cost.stages) + 1)
check("wall time >= the sum of its stages (unrounded)",
      it.elapsed() >= stage_sum_inside,
      f"{it.elapsed():.6f} vs {stage_sum_inside:.6f}")
check("the RECORDED wall_seconds >= the recorded stage sum, within rounding",
      rec_inside["wall_seconds"] + _tol
      >= sum(rec_inside["stages"].values()),
      f"{rec_inside['wall_seconds']} + {_tol} vs "
      f"{sum(rec_inside['stages'].values())}")
check("elapsed() is live inside the context", it.elapsed() > 0)
check("the raw record, once record_now has run, is no longer zero",
      raw_inside["wall_seconds"] > 0)
check("the final value from __exit__ is at least the in-context snapshot",
      inner.costs[-1].wall_seconds >= snap_unrounded,
      f"{inner.costs[-1].wall_seconds} vs {snap_unrounded}")
check("the summary still reconciles after an in-context snapshot",
      len(inner.summary()["iteration_wall_seconds"]) == 1
      and inner.summary()["iteration_wall_seconds"][0] > 0)

try:
    with s.iteration() as it:
        with it.stage("fit"):
            time.sleep(0.01)
        raise ValueError("candidate blew up")
except ValueError:
    crashed = True
else:
    crashed = False
check("an exception is not swallowed by the iteration context", crashed)
check("a crashed iteration is still recorded as spent", s.iterations_done == 2)
check("a crashed iteration's wall-clock is recorded", s.costs[-1].wall_seconds > 0)
check("a crashed iteration's stages are recorded", "fit" in s.costs[-1].stages)

# 7 -- the summary the run log and report both consume --------------------------
sm = s.summary()
for key in ("iterations", "max_iterations", "wall_seconds", "max_wall_seconds",
            "wall_hours", "tokens_total", "iteration_wall_seconds",
            "iteration_wall_mean", "iteration_wall_max", "stage_seconds", "repairs",
            "started_epoch", "remaining"):
    check(f"summary reports {key}", key in sm)
check("per-iteration wall-clock is a list, one entry per iteration",
      len(sm["iteration_wall_seconds"]) == s.iterations_done)
check("stage seconds are totalled across iterations",
      abs(sm["stage_seconds"]["fit"]
          - sum(c.stages.get("fit", 0.0) for c in s.costs)) < 0.01)
check("an empty budget summarises without dividing by zero",
      B.Budget(max_wall_seconds=60).summary()["iteration_wall_mean"] is None)
check("IterationCost.to_record is JSON-shaped",
      set(cost.to_record()) == {"iteration", "wall_seconds", "stages", "tokens_in",
                                "tokens_out", "tokens_thinking", "tokens_total",
                                "repairs", "started_epoch"},
      str(sorted(cost.to_record())))

# 8 -- refuse a nonsensical budget rather than silently accepting it ------------
check("max_iterations must be >= 1", raises(ValueError, B.Budget, max_iterations=0) != "")
check("max_wall_seconds must be positive",
      raises(ValueError, B.Budget, max_wall_seconds=0) != "")
check("fixed mode requires a token ceiling",
      raises(ValueError, B.Budget, mode="fixed") != "")
check("open mode rejects a contradictory token ceiling",
      raises(ValueError, B.Budget, mode="open", max_tokens=100) != "")
check("token ceilings must be positive",
      raises(ValueError, B.Budget, max_tokens=0) != "")

print()
bad_l = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad_l)}/{len(R)} passed" + (f"  FAILURES: {bad_l}" if bad_l else ""))
sys.exit(1 if bad_l else 0)
