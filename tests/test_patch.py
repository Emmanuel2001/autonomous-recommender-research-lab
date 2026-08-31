"""ProposePatch: the escape hatch that lets the agent write the model.

The static checks tested here are for FAST FEEDBACK, not safety -- the sandbox is the
boundary, and that is covered by test_isolation.py. What matters here is that a bad patch
is refused cheaply and specifically, and that a good one honours the contract.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.patch import (validate, normalise, patch_hash, PatchError,
                         MAX_PATCH_BYTES, DISCOURAGED_IMPORTS)
from agent import official_actions as A
from strategy.ranker import Config

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

GOOD = '''
import numpy as np
def fit_predict(train, dev, heldout, params):
    n = len(dev.user_id); m = len(heldout.user_id)
    return {"dev": np.arange(n, dtype=float), "heldout": np.arange(m, dtype=float)}
'''
def refused(src):
    try:
        validate(src); return None
    except PatchError as e:
        return str(e)

# 1 -- contract enforcement -------------------------------------------------------
meta = validate(GOOD)
check("a contract-honouring patch validates", meta["functions"] == ["fit_predict"])
check("validate reports the imports it saw", meta["imports"] == ["numpy"])
check("empty patch refused", refused(""))
check("missing fit_predict refused", refused("def other(): pass"))
check("wrong arity refused", refused("def fit_predict(a, b, c): pass"))
check("syntax error refused with a line number",
      (refused("def fit_predict(a,b,c,d)\n pass") or "").count("line") == 1)
check("oversized patch refused",
      refused(GOOD + "\n# " + "x" * MAX_PATCH_BYTES))

# 2 -- fast-feedback refusals, each with a reason ---------------------------------
for mod in ("socket", "requests", "subprocess", "pickle", "ctypes"):
    m = refused(f"import {mod}\n" + GOOD)
    check(f"import {mod} refused with a reason", m and mod in m and len(m) > len(mod) + 20)
check("the reason for banning network imports mentions the sandbox",
      "sandbox" in (refused("import socket\n" + GOOD) or ""))

# These must appear as CODE, not comments. Comments are prose and are deliberately
# ignored -- the minimal example we show the model documents that heldout.label is None,
# and was refused by its own comment until strip_comments existed.
for frag, why, code in (
    ("log_standard_4_22_to_5_08", "eval-window labels",
     'open("log_standard_4_22_to_5_08_pure.csv")'),
    ("secrets/scorer.key", "signing key", 'open("secrets/scorer.key")'),
    ("ledger", "hill-climb", 'open("runs/ledger.jsonl")'),
    ("core.metrics", "grading its own homework", 'from core.metrics import primary'),
):
    m = refused(f"{code}\n" + GOOD)
    check(f"referencing {frag!r} in CODE is refused", m is not None)
    if m:
        check(f"  ...and the refusal explains why ({why})", why.split()[0] in m)

m = refused(GOOD.replace('n = len(dev.user_id)', 'n = len(heldout.label)'))
check("touching heldout.label in CODE is refused", m is not None)
check("the same reference in a COMMENT is ignored (prose, not use)",
      refused(GOOD + "\n# heldout.label is None by construction\n") is None)
from agent import official_actions as _OA
check("the minimal example shown to the model passes its own validator",
      refused(_OA.MINIMAL_PATCH_EXAMPLE) is None)
check("strip_comments preserves code layout so patterns still match",
      "heldout.label" in __import__("agent.patch", fromlist=["strip_comments"])
      .strip_comments("x = heldout.label  # note\n"))

# 3 -- hashing gives free dedup ---------------------------------------------------
check("hash ignores trailing whitespace and blank lines",
      patch_hash(GOOD) == patch_hash(GOOD.replace("\n", "  \n") + "\n\n\n"))
check("hash changes when the code changes",
      patch_hash(GOOD) != patch_hash(GOOD.replace("np.arange", "np.zeros")))
check("normalise drops blank lines", "\n\n" not in normalise("a\n\n\nb"))

# 4 -- the action integrates with the typed space ---------------------------------
RAT = {"mechanism": "Write the ranker directly so the objective matches the metric "
                    "rather than tuning a pointwise model that does not.",
       "expected_direction": "increase", "expected_magnitude": 0.004}
env = {"action_type": "ProposePatch",
       "params": {"code": GOOD, "params": {"k": 8}}, "rationale": dict(RAT)}
act = A.validate(env, 0.002, Config())
cfg = A.apply(Config(), act)
check("ProposePatch is in the action space", "ProposePatch" in A.ACTION_TYPES)
check("applying it switches the model to 'patch'", cfg.model == "patch")
check("the patch source is carried on the config", cfg.patch_code.strip() == GOOD.strip())
check("patch params are carried", cfg.patch_params == {"k": 8})
check("config hash identifies the patch, so a repeat dedups for free",
      cfg.hash() == A.apply(Config(), A.validate(env, 0.002, Config())).hash())
check("a different patch gets a different hash",
      cfg.hash() != A.apply(Config(), A.validate(
          {**env, "params": {"code": GOOD.replace("arange", "zeros"),
                             "params": {"k": 8}}}, 0.002, Config())).hash())
bad = {**env, "params": {"code": "def nope(): pass", "params": {}}}
try:
    A.validate(bad, 0.002, Config()); check("an invalid patch is refused by the action", False)
except A.ActionError as e:
    check("an invalid patch is refused by the action", "patch rejected" in str(e))
check("the json schema exposes a code field the model can fill",
      A.json_schema()["properties"]["params"]["properties"]["code"]["type"] == "string")

# 5 -- typed and patch configs coexist -------------------------------------------
check("typed configs still hash on their own fields",
      Config(loss="pairwise").hash() != Config(loss="pointwise").hash())
check("a typed config and a patch config never collide",
      Config(loss="pairwise").hash() != cfg.hash())
check("patch_code is inert for a typed model (not in its relevant fields)",
      Config(loss="pairwise").hash() ==
      Config(loss="pairwise", patch_code="irrelevant").hash())

print()
bad_l = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad_l)}/{len(R)} passed" + (f"  FAILURES: {bad_l}" if bad_l else ""))
sys.exit(1 if bad_l else 0)
