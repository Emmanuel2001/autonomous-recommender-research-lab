"""Tests for the parent-selection policies -- the archive-vs-chain ablation axis.

The degenerate cases are the point. On this task the archive's composites differ in
the third decimal, so a fixed softmax temperature would either flatten to uniform or
collapse onto the argmax depending on the run; and early on every member scores
identically, which is a division by the standard deviation of zero.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from agent.driver import SELECTIONS, select_parent, population_digest
from strategy.mle_strategy import Config

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")


def pop_of(comps):
    return {f"h{i}": {"cfg": Config(model="ease"),
                      "pl": {"metrics": {"ndcg@10": c / 2, "recall@50": c / 2}}}
            for i, c in enumerate(comps)}


rng = np.random.default_rng(0)
spread = pop_of([0.150, 0.155, 0.158, 0.160, 0.161])

# 1 -- chain never leaves the incumbent, whatever the archive holds
ch = {}
picks = [select_parent(spread, "h4", ch, "chain", 0.25, rng) for _ in range(40)]
check("chain always returns the incumbent", {p[0] for p in picks} == {"h4"})
check("chain labels itself 'chain'", {p[1] for p in picks} == {"chain"})

# 2 -- epsilon_greedy branches at roughly epsilon
ch = {}
tags = [select_parent(spread, "h4", ch, "epsilon_greedy", 0.25, rng)[1] for _ in range(2000)]
frac = tags.count("restart") / len(tags)
check("epsilon_greedy branches at ~epsilon", 0.20 < frac < 0.30, f"{frac:.3f} vs 0.25")

# 3 -- epsilon=0 makes epsilon_greedy identical to a chain
ch = {}
z = {select_parent(spread, "h4", ch, "epsilon_greedy", 0.0, rng)[0] for _ in range(50)}
check("epsilon=0 degenerates to a chain", z == {"h4"})

# 4 -- archive actually spreads, and prefers the better members
ch = {}
seen = []
for _ in range(400):
    h, _t = select_parent(spread, "h4", ch, "archive", 0.25, rng)
    ch[h] = ch.get(h, 0) + 1
    seen.append(h)
check("archive uses every member", len(set(seen)) == len(spread), f"{len(set(seen))}/{len(spread)}")
top2 = ch.get("h4", 0) + ch.get("h3", 0)
bot2 = ch.get("h0", 0) + ch.get("h1", 0)
check("archive prefers higher-scoring members", top2 > bot2, f"top2={top2} bot2={bot2}")
check("archive still explores the worst members", ch.get("h0", 0) > 0, f"h0 used {ch.get('h0',0)}x")

# 5 -- THE DEGENERATE CASES
flat = pop_of([0.15] * 4)                      # sd == 0: naive code divides by zero
ch = {}
try:
    picks = [select_parent(flat, "h0", ch, "archive", 0.25, rng)[0] for _ in range(60)]
    check("archive survives an all-identical archive (sd=0)", len(set(picks)) == 4,
          f"{len(set(picks))} distinct")
except Exception as e:
    check("archive survives an all-identical archive (sd=0)", False, f"{type(e).__name__}: {e}")

single = pop_of([0.15])
for pol in SELECTIONS:
    h, t = select_parent(single, "h0", {}, pol, 0.25, rng)
    want = "chain" if pol == "chain" else "mutate"
    check(f"{pol} handles a single-member archive", h == "h0" and t == want,
          f"tag={t!r}")
# The tag names the ACTION, not the policy: a one-member archive under an archive
# policy has not "become a chain", it just has nothing to branch to yet.
check("a one-member archive does not mislabel itself as 'chain'",
      select_parent(single, "h0", {}, "archive", 0.25, rng)[1] == "mutate")

try:
    select_parent(spread, "h4", {}, "not_a_policy", 0.25, rng)
    check("an unknown policy raises rather than silently defaulting", False)
except ValueError:
    check("an unknown policy raises rather than silently defaulting", True)

# 6 -- the under-exploration bonus must actually bite
ch = {"h4": 50}                                 # incumbent already hammered
later = [select_parent(spread, "h4", ch, "archive", 0.25, rng)[0] for _ in range(200)]
check("a heavily-used parent is damped", later.count("h4") < 60,
      f"h4 chosen {later.count('h4')}/200 after 50 prior uses")

# 7 -- the digest the proposer sees stays bounded as the archive grows
big = pop_of([0.15 + i * 1e-4 for i in range(200)])
d = population_digest(big, "h199", keep=12)
check("population_digest is bounded in archive size", d["shown"] <= 12,
      f"shown={d['shown']} of total={d['total_evaluated']}")
check("population_digest reports the true total", d["total_evaluated"] == 200)
check("population_digest marks the incumbent",
      any("CURRENT INCUMBENT" in c for c in d["configs"]))

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
