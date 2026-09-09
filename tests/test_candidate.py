"""Tests for candidate-tree validation and the failure modes generated code actually hits.

Two groups. The first is static and free: every refusal path, each asserted to fire for
its own reason rather than "something raised". The second RUNS hostile and broken
candidates and checks that each failure comes back as usable data -- with a control arm
wherever isolation is claimed, because a probe that fails everywhere proves nothing.

One deliberate divergence from agent/patch.py is asserted here: a candidate MAY compute
the official metric. patch.py refuses source mentioning GAUC, which also made early
stopping on the public validation split impossible. The labels that decide the competition
are not in the process, and the number that counts is the scorer's signed one.
"""
import json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from agent import candidate as C
from core import artifacts as AR

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

def refusal(files):
    """The refusal message, or '' if it was accepted."""
    try:
        C.validate_tree(files)
    except (C.CandidateError, AR.ArtifactError) as e:
        return str(e) or "raised"
    except Exception as e:
        return f"WRONG-TYPE:{type(e).__name__}"
    return ""

SEED = C.seed_tree()
TMP = Path(ROOT / "runs" / "t" / "candidate"); shutil.rmtree(TMP, ignore_errors=True)

# 1 -- the seed is valid, and its identity is stable ----------------------------
m = C.validate_tree(SEED)
check("the seed tree validates", m["tree"] == AR.tree_hash(SEED))
check("the seed declares exactly the contract's mutable files",
      m["files"] == sorted(C._contract_files()))
check("the contract hash is a full sha256", len(m["contract_sha256"]) == 64)
check("the contract hash matches the file on disk",
      m["contract_sha256"] == C.contract_sha256())
check("the seed's config is parsed and returned", m["config"]["loss"] == "pointwise")
check("re-validating gives the same tree hash", C.validate_tree(SEED)["tree"] == m["tree"])

# 2 -- refusal paths, each for its own reason ----------------------------------
cases = [
    ("a missing required file", {k: v for k, v in SEED.items() if k != "model.py"},
     "missing required files"),
    ("a tree that ships its own contract.py",
     {**SEED, "contract.py": "MUTABLE_FILES = ()\n"}, "immutable"),
    ("a subdirectory", {**SEED, "sub/extra.py": "x = 1\n"}, "flat"),
    ("a non-code file", {**SEED, "notes.txt": "hello\n"}, "only .py and .json"),
    ("a module shadowing the stdlib", {**SEED, "json.py": "x = 1\n"}, "would shadow"),
    ("a module shadowing a repo package", {**SEED, "core.py": "x = 1\n"}, "would shadow"),
    ("config.json that is not JSON", {**SEED, "config.json": "{nope}"}, "not valid JSON"),
    ("config.json that is not an object", {**SEED, "config.json": "[1, 2]"},
     "must be an object"),
    ("a syntax error", {**SEED, "model.py": "def f(:\n"}, "syntax error"),
    ("a missing entry point",
     {**SEED, "train.py": "def other(a, b):\n    return a\n"}, "must define fit_predict"),
    ("the wrong entry-point arity",
     {**SEED, "train.py": SEED["train.py"].replace(
         "def fit_predict(data, params: dict) -> dict:",
         "def fit_predict(data, params, extra) -> dict:")}, "exactly 2 arguments"),
    ("a discouraged import", {**SEED, "model.py": "import socket\n" + SEED["model.py"]},
     "do not import"),
    ("naming the eval-window label file",
     {**SEED, "features.py": SEED["features.py"]
      + "\nPATH = 'log_standard_4_22_to_5_08_pure.csv'\n"}, "eval-window labels"),
    ("naming the signing key",
     {**SEED, "features.py": SEED["features.py"] + "\nK = 'secrets/scorer.key'\n"},
     "signing key"),
    ("reading the ledger",
     {**SEED, "features.py": SEED["features.py"] + "\nL = open('runs/x.jsonl')\n"},
     "ledger"),
    ("using held-out labels",
     {**SEED, "train.py": SEED["train.py"].replace(
         "    cfg = load_config(params)",
         "    y = heldout.label\n    cfg = load_config(params)")}, "held-out labels"),
    ("a path traversal", {**SEED, "../escape.py": "x = 1\n"}, "contains '..'"),
    ("an absolute path", {**SEED, "/etc/passwd": "x = 1\n"}, "must be relative"),
]
for name, files, expect in cases:
    msg = refusal(files)
    check(f"refused: {name}", expect.lower() in msg.lower(), msg[:90])

heldout_msg = refusal({**SEED, "train.py": SEED["train.py"].replace(
    "    cfg = load_config(params)",
    "    y = heldout.label\n    cfg = load_config(params)")})
check("a held-out-label refusal names the offending file and line",
      "train.py line " in heldout_msg, heldout_msg[:160])
check("a held-out-label refusal gives the grouped-ranker repair",
      "group sizes without a label" in heldout_msg
      and "train and dev" in heldout_msg, heldout_msg[:220])

# 2b -- the model seed is part of candidate identity ---------------------------
# The typed path excluded `seed` from every dedup tuple, so the agent could not evaluate
# one configuration at a second seed: it came back as a duplicate. Here the seed lives in
# config.json, which is inside the tree, so a replicate is a distinct candidate.
def with_cfg(**kw):
    c = json.loads(SEED["config.json"]); c.update(kw)
    return {**SEED, "config.json": json.dumps(c, indent=2) + "\n"}

check("two candidates differing only in seed are DIFFERENT candidates",
      C.validate_tree(with_cfg(seed=0))["tree"]
      != C.validate_tree(with_cfg(seed=1))["tree"])
check("a deliberate seed replicate is therefore not refused as a duplicate",
      len({C.validate_tree(with_cfg(seed=i))["tree"] for i in range(5)}) == 5)
check("trailing whitespace in config.json does not change the candidate",
      C.validate_tree(SEED)["tree"]
      == C.validate_tree({**SEED,
                          "config.json": SEED["config.json"].rstrip() + "   \n\n"})["tree"])

# 3 -- what is deliberately ALLOWED --------------------------------------------
check("a candidate MAY compute the official metric through the contract",
      refusal({**SEED, "train.py": SEED["train.py"]
               + "\ndef extra(d):\n    return primary(d.dev.user_id, d.dev.label, [0.0])\n"})
      == "")
check("a comment mentioning held-out labels is prose, not a use",
      refusal({**SEED, "train.py": SEED["train.py"]
               + "\n# heldout.label is None by construction; we never touch it\n"}) == "")
check("an extra module with a safe name is allowed",
      refusal({**SEED, "losses.py": "def bpr(a, b):\n    return a - b\n"}) == "")
check("an allowlisted dependency declaration is accepted",
      refusal({**SEED, "model.py": "# requires: torch\n" + SEED["model.py"]}) == "")

# 4 -- broken returns come back as DATA, not as a crash ------------------------
def trivial_train(body: str) -> dict:
    """A candidate that skips training entirely, so a contract check costs seconds."""
    return {**SEED, "train.py":
            "from contract import train_subset\nimport numpy as np\n\n\n"
            "def fit_predict(data, params):\n" + body + "\n"}

BAD = {
    "wrong dev length": trivial_train(
        "    idx, _ = train_subset(data)\n"
        "    return {'dev': np.arange(10.0),\n"
        "            'heldout': np.arange(float(len(data.heldout))),\n"
        "            'train_subset': np.arange(float(idx.size))}"),
    "constant dev scores": trivial_train(
        "    idx, _ = train_subset(data)\n"
        "    return {'dev': np.zeros(len(data.dev)),\n"
        "            'heldout': np.arange(float(len(data.heldout))),\n"
        "            'train_subset': np.arange(float(idx.size))}"),
    "a NaN in dev scores": trivial_train(
        "    idx, _ = train_subset(data)\n"
        "    d = np.arange(float(len(data.dev))); d[7] = np.nan\n"
        "    return {'dev': d, 'heldout': np.arange(float(len(data.heldout))),\n"
        "            'train_subset': np.arange(float(idx.size))}"),
    "a missing train_subset": trivial_train(
        "    return {'dev': np.arange(float(len(data.dev))),\n"
        "            'heldout': np.arange(float(len(data.heldout)))}"),
    "a non-dict return": trivial_train("    return 42"),
    "an exception inside the candidate": trivial_train(
        "    raise ZeroDivisionError('deliberate')"),
}
EXPECT = {"wrong dev length": "One score per row",
          "constant dev scores": "every score is identical",
          "a NaN in dev scores": "non-finite",
          "a missing train_subset": "no scores for 'train_subset'",
          "a non-dict return": "must return a dict",
          "an exception inside the candidate": "deliberate"}
for name, files in BAD.items():
    d = TMP / name.replace(" ", "_")
    rep = C.run_candidate(files, {}, run_dir=d, backend="none", max_seconds=300)
    msg = str(rep.get("error") or "")
    check(f"failure returned as data: {name}",
          rep.get("ok") is False and EXPECT[name].lower() in msg.lower(), msg[:80])
    check(f"  ... and it names where it happened: {name}",
          bool(rep.get("error_frame")) and rep.get("error_in") in
          ("candidate", "harness_check"),
          f"{rep.get('error_in')} {rep.get('error_frame')}")
check("an exception in the candidate is attributed to the CANDIDATE, not the harness",
      C.run_candidate(BAD["an exception inside the candidate"], {},
                      run_dir=TMP / "attr", backend="none",
                      max_seconds=300).get("error_in") == "candidate")

# 5 -- one candidate cannot write into another's directory ---------------------
def attack_tree(target: Path) -> dict:
    return trivial_train(
        f"    import pathlib\n"
        f"    t = pathlib.Path({str(target)!r})\n"
        f"    try:\n"
        f"        t.write_bytes(b'OVERWRITTEN')\n"
        f"        raise RuntimeError('ATTACK-SUCCEEDED')\n"
        f"    except PermissionError as e:\n"
        f"        raise RuntimeError('ATTACK-BLOCKED: ' + str(e))")

from core import isolation as I
results = {}
for backend in ("none", "seatbelt"):
    if backend != "none" and I.available().get(backend, "x") != "":
        print(f"  [skip] {backend} unavailable: {I.available()[backend]}")
        continue
    victim = (TMP / f"victim_{backend}").resolve()
    victim.mkdir(parents=True, exist_ok=True)
    tgt = victim / "dev_scores.npy"
    tgt.write_bytes(b"ORIGINAL")
    rep = C.run_candidate(attack_tree(tgt), {}, run_dir=TMP / f"attacker_{backend}",
                          backend=backend, max_seconds=300)
    err = str(rep.get("error") or "")
    results[backend] = ("SUCCEEDED" if "ATTACK-SUCCEEDED" in err
                        else "BLOCKED" if "ATTACK-BLOCKED" in err else f"?{err[:60]}")
    results[backend + "_intact"] = tgt.read_bytes() == b"ORIGINAL"

check("CONTROL: with no isolation the cross-candidate write SUCCEEDS "
      "(so the probe is real)", results.get("none") == "SUCCEEDED", str(results.get("none")))
check("CONTROL: the victim's bytes really were overwritten",
      results.get("none_intact") is False)
if "seatbelt" in results:
    check("seatbelt BLOCKS a write into a sibling candidate's directory",
          results["seatbelt"] == "BLOCKED", str(results["seatbelt"]))
    check("the victim's scores survive the seatbelt attempt",
          results["seatbelt_intact"] is True)

# 5a -- an unhonourable metric cutoff is refused, not silently substituted ------
# candidate_runner rebinds contract.primary to the active TaskSpec's metric. That wrapper
# accepted k and dropped it, so a candidate early-stopping on nDCG@10 silently received the
# nDCG@5 number (0.5429 vs 0.4463 on a synthetic split -- and on the official task, since the
# wrapper is installed for every task). No seed passes k, so nothing caught it.
_k_tree = dict(C.seed_tree(C.workspace_for("minimal")))
_k_tree["train.py"] = _k_tree["train.py"].replace(
    "dev_m = primary(udev, ydev, m.predict(Xdev))",
    "dev_m = primary(udev, ydev, m.predict(Xdev), k=10)")
check("the k-probe fixture actually asks for a different cutoff",
      "k=10" in _k_tree["train.py"])
_k_rep = C.run_candidate(_k_tree, {}, run_dir=TMP / "kprobe", backend="none",
                         max_seconds=300)
check("a cutoff the task metric cannot honour is REFUSED, not silently substituted",
      _k_rep.get("ok") is False and "cannot honour k=10" in str(_k_rep.get("error")),
      f"ok={_k_rep.get('ok')} err={str(_k_rep.get('error'))[:80]}")
check("the refusal is a contract violation the coder can act on",
      _k_rep.get("error_type") == "ContractError", str(_k_rep.get("error_type")))
# CONTROL: the same seed WITHOUT the k argument still runs, so the probe is specific.
_ctl_rep = C.run_candidate(dict(C.seed_tree(C.workspace_for("minimal"))), {},
                           run_dir=TMP / "kprobe_ctl", backend="none", max_seconds=600)
check("CONTROL: the unmodified seed still runs (the refusal is not blanket)",
      _ctl_rep.get("ok") is True, str(_ctl_rep.get("error"))[:80])

# 5b -- a candidate cannot edit its OWN content-addressed source ---------------
# agent/candidate.py executes the artifact source directly and says "a candidate cannot
# edit its own source". That invariant had no test: a second copy under run_dir once made
# it false, because run_dir is deliberately the sandbox's write allowance. Deliberately
# NOT run under backend="none" -- an unsandboxed self-edit would corrupt the artifact
# store it is asserting about.
def self_edit_tree(rel: str) -> dict:
    return trivial_train(
        f"    import pathlib\n"
        f"    t = pathlib.Path(__file__).resolve().parent / {rel!r}\n"
        f"    try:\n"
        f"        t.write_text('SELF-EDITED')\n"
        f"        raise RuntimeError('ATTACK-SUCCEEDED')\n"
        f"    except (PermissionError, OSError) as e:\n"
        f"        raise RuntimeError('ATTACK-BLOCKED: ' + type(e).__name__)")


def tmp_write_tree() -> dict:
    return trivial_train(
        "    import pathlib, tempfile\n"
        "    t = pathlib.Path(tempfile.gettempdir()) / 'track2_probe_control.txt'\n"
        "    try:\n"
        "        t.write_text('ok')\n"
        "        raise RuntimeError('ATTACK-SUCCEEDED')\n"
        "    except (PermissionError, OSError) as e:\n"
        "        raise RuntimeError('ATTACK-BLOCKED: ' + type(e).__name__)")


if I.available().get("seatbelt", "x") == "":
    rep_self = C.run_candidate(self_edit_tree("model.py"), {},
                               run_dir=TMP / "selfedit", backend="seatbelt",
                               max_seconds=300)
    err_self = str(rep_self.get("error") or "")
    tree_self = rep_self.get("tree") or ""
    check("seatbelt BLOCKS a candidate writing over its own source",
          "ATTACK-BLOCKED" in err_self, err_self[:90])
    still_ok = False
    try:
        AR.verify(tree_self)
        still_ok = True
    except Exception as e:
        still_ok = f"{type(e).__name__}: {e}"
    check("the artifact tree still verifies after the self-edit attempt",
          still_ok is True, str(still_ok))
    # CONTROL: the same write pattern must SUCCEED somewhere permitted, otherwise the
    # block above could just be "writes never work in this sandbox".
    rep_ctl = C.run_candidate(tmp_write_tree(), {}, run_dir=TMP / "selfedit_ctl",
                              backend="seatbelt", max_seconds=300)
    check("CONTROL: the same candidate CAN write to a permitted path under seatbelt",
          "ATTACK-SUCCEEDED" in str(rep_ctl.get("error") or ""),
          str(rep_ctl.get("error"))[:90])
else:
    print(f"  [skip] seatbelt unavailable: {I.available().get('seatbelt')}")

# 6 -- a relative sandbox path is refused, not silently granted ----------------
try:
    I.seatbelt_profile([], [], [Path("runs/relative")])
    rel = ""
except I.IsolationError as e:
    rel = str(e)
check("a relative write_allow path is refused rather than emitting a dead rule",
      "must be absolute" in rel, rel[:80])

shutil.rmtree(TMP, ignore_errors=True)
print()
bad_l = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad_l)}/{len(R)} passed" + (f"  FAILURES: {bad_l}" if bad_l else ""))
sys.exit(1 if bad_l else 0)
