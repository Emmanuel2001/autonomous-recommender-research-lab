"""The current task's action space, and the refusals that make it worth having."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import official_actions as A
from strategy.ranker import Config, LOSSES

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

RAT = {"mechanism": "Pointwise logloss is misaligned with the ranking metric; pairing "
                    "within a user optimises the ordering the metric measures.",
       "expected_direction": "increase", "expected_magnitude": 0.003,
       "symptoms_addressed": ["loss_metric_mismatch"]}
def env(at, params, rat=None):
    return {"action_type": at, "params": params, "rationale": dict(rat or RAT)}
def refused(at, params, rat=None):
    try:
        A.validate(env(at, params, rat), 0.002, Config()); return None
    except A.ActionError as e:
        return str(e)

# 1 -- the happy path, every action reachable and applying -----------------------
GOOD = {"SetLoss": {"loss": "pairwise"}, "SetNegPerPos": {"neg_per_pos": 4},
        "SetListwiseTemp": {"listwise_temp": 2.0},
        "SetAuxWeights": {"aux_weights": {"is_click": 0.3}},
        # a SUBSET, not the baseline five -- applying the default fields is a
        # legitimate no-op and would fail the "changes the config" check below
        "SetFields": {"fields": ["user_id", "video_id", "tab"]},
        "ChangeLr": {"lr": 0.003}, "ChangeL2": {"l2": 1e-5},
        "ChangeBatch": {"batch": 4096}, "ChangeEpochs": {"epochs": 25},
        "ChangePatience": {"patience": 6}, "ChangeK": {"k": 24},
        "SwapModel": {"model": "popularity"}, "SetUseRandomLog": {"use_random_log": True},
        "SetEnsemble": {"n_seeds": 5, "combine": "rank"},
        "ProposePatch": {"code": A.MINIMAL_PATCH_EXAMPLE, "params": {}},
        "Inspect": {"diagnostic": "label_rates"}}
check("every action type has a happy-path fixture", set(GOOD) == set(A.ACTION_TYPES),
      f"missing {set(A.ACTION_TYPES) - set(GOOD)}")
# Inspect is a QUERY: it must leave the config untouched. Everything else must change it.
CHANGERS = {k: v for k, v in GOOD.items() if k != "Inspect"}
applied = 0
for at, p in CHANGERS.items():
    a = A.validate(env(at, p), 0.002, Config())
    if A.apply(Config(), a) != Config():
        applied += 1
check("every config-changing action changes the config", applied == len(CHANGERS),
      f"{applied}/{len(CHANGERS)}")
_ins = A.validate(env("Inspect", {"diagnostic": "label_rates"},
                      {"mechanism": "Check whether dev shares train's base rate before "
                                    "spending another fit on tuning."}), 0.002, Config())
check("Inspect leaves the config untouched (it is a query, not a change)",
      A.apply(Config(loss="pairwise"), _ins) == Config(loss="pairwise"))
check("Inspect needs no expected_magnitude", _ins["rationale"]["expected_magnitude"] is None)
check("but Inspect still needs a real mechanism",
      refused("Inspect", {"diagnostic": "label_rates"}, {"mechanism": "look"}) is not None)
check("re-applying the DEFAULT fields is correctly a no-op (dedup will catch it)",
      A.apply(Config(), A.validate(env("SetFields", {"fields": list(A.BASELINE_FIELDS)}),
                                   0.002, Config())) == Config())
check("apply() does not mutate its input",
      A.apply(Config(), A.validate(env("ChangeK", {"k": 40}), 0.002, Config())) is not None
      and Config().k == 16)

# 2 -- THE PUBLISHED DEAD ENDS, refused WITH the citation ------------------------
m = refused("ChangeK", {"k": 16})
check("k in {8,16,32} refused as a measured dead end", m is not None)
check("the capacity refusal cites their numbers", m and ".5895" in m and ".5902" in m)
for k in (8, 32):
    check(f"k={k} also refused", refused("ChangeK", {"k": k}) is not None)
check("a k they did NOT measure is allowed", refused("ChangeK", {"k": 24}) is None)

m = refused("SetFields", {"fields": list(A.BASELINE_FIELDS) + ["music_id", "video_type"]})
check("piling on raw categorical fields refused", m is not None)
# These fields were removed from the pool once measured, so the refusal now comes from
# REMOVED_FIELDS -- which is the stronger message: it quotes each field's OWN number rather
# than only the aggregate. Both citations must survive.
check("the refusal quotes each removed field's measured gain",
      m and "+0.00028" in m and "+0.00016" in m, m)
check("and still cites the aggregate the organisers measured",
      m and ".5940" in m and ".5950" in m, m)
check("and states the rule, not just the number",
      m and "VARIES within that user" in m, m)
check("a genuinely unknown field gets the plain message, not a fabricated measurement",
      (refused("SetFields", {"fields": ["user_id", "not_a_column"]}) or "").startswith(
          "fields has unknown fields"))

print("== hourmin_bucket is a MEASURED WIN and must not be refused as piling-on")
check("baseline + hourmin_bucket is allowed",
      refused("SetFields",
              {"fields": list(A.BASELINE_FIELDS) + ["hourmin_bucket"]}) is None)
check("hourmin_bucket alone with a couple of ids is allowed",
      refused("SetFields", {"fields": ["user_id", "video_id", "hourmin_bucket"]}) is None)
check("it is declared as an encouraged extra, not special-cased inline",
      "hourmin_bucket" in A.ENCOURAGED_EXTRA)
check("the exemption is justified by a measurement in the source",
      "t=14.04" in (Path(__file__).resolve().parents[1]
                    / "agent" / "official_actions.py").read_text())
check("every field the pool offers can actually be encoded",
      all(f in ("user_id", "video_id", "author_id", "tab", "dur_bucket", "hourmin_bucket")
          for f in A.KNOWN_FIELDS), str(A.KNOWN_FIELDS))
check("the baseline 5 fields are allowed",
      refused("SetFields", {"fields": list(A.BASELINE_FIELDS)}) is None)
check("a SUBSET of the baseline fields is allowed (not a dead end)",
      refused("SetFields", {"fields": ["user_id", "video_id"]}) is None)

# user_active_degree has been removed from the pool, so it now refuses with its measurement.
# `user_id` alone still exercises the PROVABLY-INERT arithmetic branch, which is the point.
m = refused("SetFields", {"fields": ["user_id"]})
check("user-side-only fields refused as PROVABLY inert", m is not None)
m2 = refused("SetFields", {"fields": ["user_id", "user_active_degree"]})
check("a removed user-side field cites its 0.0% within-user variation",
      m2 and "0.0% within-user variation" in m2, m2)
check("the inert refusal explains the arithmetic, not just the measurement",
      m and "within-user" in m and "constant" in m)
m = refused("SetFields", {"fields": ["user_id", "author_id", "tab"]})
check("dropping video_id refused (nothing item-side to cross)", m is not None)

# 3 -- rationale gating ----------------------------------------------------------
small = dict(RAT); small["expected_magnitude"] = 0.0005
m = refused("SetLoss", {"loss": "listwise"}, small)
check("expected_magnitude below the OFFICIAL epsilon is refused", m is not None)
check("the epsilon refusal names the organisers' threshold", m and "0.002" in m)
short = dict(RAT); short["mechanism"] = "try it"
check("a one-liner mechanism is refused", refused("SetLoss", {"loss": "listwise"}, short))
nosym = dict(RAT); nosym["symptoms_addressed"] = ["not_a_symptom"]
check("unknown symptoms are refused", refused("SetLoss", {"loss": "listwise"}, nosym))

# 4 -- schema hygiene ------------------------------------------------------------
check("unknown action_type refused", refused("MakeItBetter", {}))
check("wrong params for an action refused", refused("SetLoss", {"k": 8}))
check("missing params refused", refused("SetLoss", {}))
check("out-of-bounds refused", refused("ChangeLr", {"lr": 10.0}))
check("bool is not accepted as a number", refused("ChangeK", {"k": True}))
sch = A.json_schema()
check("json schema constrains params to typed named fields",
      sch["properties"]["params"]["additionalProperties"] is False)
check("json schema enum matches the real action list",
      sch["properties"]["action_type"]["enum"] == list(A.ACTION_TYPES))
check("loss enum in the schema matches the ranker's LOSSES",
      set(sch["properties"]["params"]["properties"]["loss"]["enum"]) == set(LOSSES))

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
