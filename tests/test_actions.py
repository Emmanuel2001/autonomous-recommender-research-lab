"""The action space is the agent's whole surface. Every rejection path matters."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import actions as A
from agent.actions import ActionError
from strategy.mle_strategy import Config

ME = {"ndcg@10": 0.002285, "recall@50": 0.004468}
R = []
def check(n, c, extra=""): R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")
def rejects(n, env, want=""):
    try:
        A.validate(env, ME); check(n, False, "accepted when it should not")
    except ActionError as e:
        check(n, want in str(e) if want else True, f"({str(e)[:70]})")

good_rat = {"mechanism": "EASE lambda controls the ridge penalty on the item-item weights",
            "target_metric": "recall@50", "expected_direction": "increase",
            "expected_magnitude": 0.01, "symptoms_addressed": ["lambda_plateau"]}

# happy path
a = A.validate({"action_type": "ChangeEaseLambda", "params": {"value": 700},
                "rationale": good_rat}, ME)
check("valid action accepted", a["action_type"] == "ChangeEaseLambda" and a["params"]["value"] == 700.0)
check("params coerced to the declared type", isinstance(a["params"]["value"], float))

# structural rejections
rejects("unknown action_type", {"action_type": "Nope", "params": {}, "rationale": good_rat}, "unknown")
rejects("missing param", {"action_type": "ChangeEaseLambda", "params": {}, "rationale": good_rat}, "missing param")
rejects("extra param", {"action_type": "ChangeEaseLambda",
                        "params": {"value": 1, "bogus": 2}, "rationale": good_rat}, "unexpected param")
rejects("extra envelope field", {"action_type": "SetMask", "params": {"mask": "none"},
                                 "rationale": good_rat, "sneak": 1}, "unexpected field")
rejects("value below range", {"action_type": "ChangeEaseLambda", "params": {"value": 0.0},
                              "rationale": good_rat}, "outside")
rejects("value above range", {"action_type": "ChangeDim", "params": {"dim": 99999},
                              "rationale": good_rat}, "outside")
rejects("non-numeric value", {"action_type": "ChangeLr", "params": {"lr": "fast"},
                              "rationale": good_rat}, "not a float")
rejects("bad enum", {"action_type": "SwapModel", "params": {"model": "transformer"},
                     "rationale": good_rat}, "not in")
rejects("not an object", "[1,2,3]", "must be a JSON object")
rejects("malformed JSON", "{not json", "not valid JSON")

# signal-weight rules
rejects("weights without is_click", {"action_type": "SetSignalWeights",
        "params": {"weights": {"long_view": 1.0}}, "rationale": good_rat}, "is_click must be present")
rejects("unknown signal", {"action_type": "SetSignalWeights",
        "params": {"weights": {"is_click": 1.0, "is_vibe": 1.0}}, "rationale": good_rat}, "unknown signal")
rejects("weight out of range", {"action_type": "SetSignalWeights",
        "params": {"weights": {"is_click": 1.0, "long_view": 99}}, "rationale": good_rat}, "outside")
ok = A.validate({"action_type": "SetSignalWeights",
                 "params": {"weights": {"is_click": 1.0, "long_view": 0.5}},
                 "rationale": good_rat}, ME)
check("valid weights accepted", ok["params"]["weights"] == {"is_click": 1.0, "long_view": 0.5})

# rationale rules
for field in ("mechanism", "target_metric", "expected_direction", "expected_magnitude"):
    r = {k: v for k, v in good_rat.items() if k != field}
    rejects(f"rationale missing {field}", {"action_type": "SetMask",
            "params": {"mask": "none"}, "rationale": r}, "missing required field")
rejects("mechanism too short", {"action_type": "SetMask", "params": {"mask": "none"},
        "rationale": {**good_rat, "mechanism": "try it"}}, "real sentence")
rejects("unknown symptom", {"action_type": "SetMask", "params": {"mask": "none"},
        "rationale": {**good_rat, "symptoms_addressed": ["vibes"]}}, "unknown symptom")

# THE noise-floor gate on proposals
rejects("expected_magnitude below the measured MDE",
        {"action_type": "ChangeEaseLambda", "params": {"value": 600},
         "rationale": {**good_rat, "expected_magnitude": 0.0001}}, "below the measured minimum")
okb = A.validate({"action_type": "ChangeEaseLambda", "params": {"value": 600},
                  "rationale": {**good_rat, "target_metric": "both",
                                "expected_magnitude": 0.0023}}, ME)
check("target_metric 'both' uses the SMALLER floor", okb["rationale"]["target_metric"] == "both")

# apply() purity and replay
base = Config(model="ease", ease_lambda=500)
c1 = A.apply(base, A.validate({"action_type": "ChangeEaseLambda", "params": {"value": 900},
                               "rationale": good_rat}, ME))
check("apply mutates only the named field", c1.ease_lambda == 900.0 and c1.model == "ease")
check("apply does not mutate the input", base.ease_lambda == 500.0)
seq = [{"action_type": "SwapModel", "params": {"model": "itemknn"}, "rationale": good_rat},
       {"action_type": "ChangeKnnK", "params": {"k": 300}, "rationale": good_rat},
       {"action_type": "SetContentAlpha", "params": {"alpha": 0.2}, "rationale": good_rat}]
def replay(start):
    c = start
    for e in seq:
        c = A.apply(c, A.validate(e, ME))
    return c
check("replaying an action list is deterministic", replay(base).hash() == replay(base).hash())
check("replay produces the expected config",
      replay(base).model == "itemknn" and replay(base).knn_k == 300 and replay(base).content_alpha == 0.2)
check("every action_type has an apply implementation", set(A.SPEC) == set(A._APPLY))
sch = A.json_schema()
check("json_schema enumerates exactly the implemented actions",
      set(sch["properties"]["action_type"]["enum"]) == set(A.ACTION_TYPES))

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
