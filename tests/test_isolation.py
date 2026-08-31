"""Tests for the OS-enforced quarantine.

The first block is not a unit test in the usual sense: it RUNS a hostile strategy
and records what the kernel refused. The control arm (`none`) must SUCCEED at the
attack -- a probe that fails everywhere proves nothing, so that is asserted too.
"""
import json, os, subprocess, sys, tempfile
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core import isolation as I
from core import splits as S
from core.ledger import Ledger, TamperedLedger, verify

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

AV = I.available()
print(f"  [env] backends: " + ", ".join(f"{k}={'ok' if not v else v}" for k, v in AV.items()))

def usable(b: str) -> bool:
    """available() returns "" for a usable backend and a REASON otherwise, so a bare
    `if AV[b]` reads backwards. It did, and it silently skipped every seatbelt
    assertion in this file while the header printed `seatbelt=ok`."""
    return AV.get(b, "unknown") == ""

# 1 -- the attack, run for real -------------------------------------------------
ctrl = I.probe("none")
check("CONTROL: the attack succeeds with no isolation (else the probe is broken)",
      len(ctrl["succeeded"]) >= 5, f"{len(ctrl['succeeded'])}/7 succeeded")
check("CONTROL: eval labels are readable in-process",
      ctrl["attempts"]["read_eval_labels"].startswith("SUCCEEDED"))
# The key is deliberately NOT in this checkout (see CLEANUP_AUDIT.md), so there may be no
# target for this particular attack. A missing target is not a blocked attack and must not be
# reported as one: when the key is absent the control arm is stated as NOT EXERCISED, and the
# seatbelt assertion below still runs on every other attack. When a key IS present the full
# control assertion applies, unchanged.
_KEYFILE = Path(__file__).resolve().parents[1] / "secrets" / "scorer.key"
_kmsg = ctrl["attempts"]["read_signing_key"]
if _KEYFILE.is_file():
    check("CONTROL: the signing key is readable in-process",
          _kmsg.startswith("SUCCEEDED") or "no key file" in _kmsg, _kmsg)
else:
    print(f"  [n/a]  CONTROL read_signing_key NOT EXERCISED: no key at {_KEYFILE.name} "
          f"in this checkout -- the probe reported {_kmsg!r}")
    check("the signing key is genuinely absent from this checkout (not merely unreadable)",
          not _KEYFILE.exists())

if not usable("seatbelt"):
    print(f"  [skip] seatbelt unavailable: {AV['seatbelt']}")
else:
    sbp = I.probe("seatbelt")
    for k in sorted(I.EXPECTED_BLOCKED["seatbelt"]):
        check(f"seatbelt blocks {k}", sbp["attempts"][k].startswith("blocked"),
              sbp["attempts"][k])
    check("seatbelt blocks everything it claims to", sbp["ok"],
          f"unexpectedly allowed: {sbp['unexpectedly_allowed']}")
    check("the signing key is not in the sandboxed environment",
          not sbp["signing_key_in_env"])
    check("the LLM api key is not in the sandboxed environment",
          not sbp["llm_key_in_env"])

# `subprocess` must not be mistaken for enforcement. Asserting that it blocks
# NOTHING keeps the docs honest: if a future change makes it enforce something,
# this fails and the claim gets updated deliberately.
sp = I.probe("subprocess")
check("subprocess enforces nothing (documented, not a regression)",
      sp["attempts"]["read_eval_labels"].startswith("SUCCEEDED"))
check("detect_backend never silently returns 'none'", I.detect_backend() != "none")

# 2 -- the prepared data root ---------------------------------------------------
dr = I.prepare_data_root(allow_log_random=False)
names = {p.name for p in dr.path.iterdir()}
check("prepared root contains the train log", "log_standard_4_08_to_4_21_pure.csv" in names)
check("prepared root does NOT contain the eval-window file",
      "log_standard_4_22_to_5_08_pure.csv" not in names)
check("prepared root does NOT contain log_random when it is not opted in",
      "log_random_4_22_to_5_08_pure.csv" not in names)
check("prepared root withholds video_features_statistic (unverified leak risk)",
      "video_features_statistic_pure.csv" not in names)
dr2 = I.prepare_data_root(allow_log_random=True)
check("opting in adds log_random and says so",
      (dr2.path / "log_random_4_22_to_5_08_pure.csv").exists()
      and "caveat" in dr2.describe())
I.prepare_data_root(allow_log_random=False)      # leave it in the default state

check("the environment handed to the strategy is scrubbed of every secret",
      not ({"TRACK2_SIGNING_KEY", "TRACK2_LLM_API_KEY", "GEMINI_API_KEY"}
           & set(I.child_env(dr.path))))
check("the strategy is told it is sandboxed", I.child_env(dr.path)["TRACK2_SANDBOX"] == "1")

# 3 -- the loaders --------------------------------------------------------------
# load_train_only must work under a read-deny; load_for_strategy must NOT, because
# it reads the eval csv via build_manifest. That asymmetry IS the bug this found.
if usable("seatbelt"):
    sb = I.make_sandbox("seatbelt")
    def under_sandbox(code: str):
        return subprocess.run(sb.argv(["-c", code]), capture_output=True, text=True,
                              cwd=str(I.ROOT), env=I.child_env(sb.data.path), timeout=600)
    r = under_sandbox("import sys; sys.path.insert(0,'.');"
                      "from core import splits as S; t,m = S.load_train_only();"
                      "print('OK', len(t), m['n_items_catalog'])")
    check("load_train_only works behind a kernel read-deny", r.returncode == 0,
          (r.stdout or r.stderr).strip()[:90])
    # The eval file is now present in the root as a REDACTED copy (the current task
    # ranks eval rows, so it needs their features). The assertion that matters is no
    # longer "the loader fails" but "the labels it can see are gone".
    r2 = under_sandbox(
        "import sys, csv, collections; sys.path.insert(0,'.');"
        "from core.official_data import DATA;"
        "c=collections.Counter();"
        "f=open(DATA/'log_standard_4_22_to_5_08_pure.csv', newline='');"
        "[c.update({('held' if 20220429<=int(r['date'])<=20220508 else 'dev',"
        " r['long_view']!='0'): 1}) for r in csv.DictReader(f)];"
        "print(c[('held',True)], c[('dev',True)])")
    if r2.returncode == 0:
        held_pos, dev_pos = (int(x) for x in r2.stdout.split())
        check("held-out labels are REDACTED in the sandbox copy (0 positives)",
              held_pos == 0, f"{held_pos} held-out positives visible")
        check("dev labels survive redaction (public validation feedback)",
              dev_pos > 30000, f"{dev_pos:,} dev positives visible")
    else:
        check("held-out labels are REDACTED in the sandbox copy (0 positives)", False,
              (r2.stderr or "")[-160:])
        check("dev labels survive redaction (public validation feedback)", False)
    r3 = under_sandbox("import sys; sys.path.insert(0,'.');"
                       "from core.official_data import load_for_strategy as L;"
                       "sp,_=L(); print(sp['test'].label is None, len(sp['valid']))")
    check("the official loader works behind the deny and still quarantines test",
          r3.returncode == 0 and r3.stdout.startswith("True"),
          (r3.stdout or r3.stderr or "").strip()[:90])

# the guard: with TRACK2_SANDBOX=1 and a data root that DOES expose the eval file,
# load_train_only must refuse rather than produce a number with a false claim.
env = dict(os.environ); env["TRACK2_SANDBOX"] = "1"
env["TRACK2_DATA_ROOT"] = str(I.REAL_DATA)      # the real dir: eval file reachable
g = subprocess.run([sys.executable, "-B", "-c",
                    "import sys; sys.path.insert(0,'.');"
                    "from core import splits as S; S.load_train_only()"],
                   capture_output=True, text=True, cwd=str(I.ROOT), env=env, timeout=300)
check("load_train_only refuses when a 'sandbox' still exposes the eval file",
      g.returncode != 0 and "sandbox misconfigured" in g.stderr)

# 4 -- config survives the process boundary ------------------------------------
from strategy.mle_strategy import Config
from strategy.run_iteration import config_from_json, _TUPLE_FIELDS
cfg = Config(model="ease", ease_lambda=500.0, content_alpha=0.05,
             signal_weights={"is_click": 1.0, "long_view": 0.3}, content_topk=40)
rt = config_from_json(json.dumps(asdict(cfg)))
check("config hash survives the JSON round-trip", rt.hash() == cfg.hash())
check("every config field survives the round-trip", asdict(rt) == asdict(cfg))
check("tuple fields come back as tuples, not lists",
      all(isinstance(getattr(rt, f), tuple) for f in _TUPLE_FIELDS))
try:
    config_from_json(json.dumps({**asdict(cfg), "not_a_field": 1}))
    check("unknown config fields are rejected", False)
except SystemExit:
    check("unknown config fields are rejected", True)

# 5 -- the ledger refuses to extend a broken chain -----------------------------
d = Path(tempfile.mkdtemp()); f = d / "l.jsonl"; k = b"k" * 32
L = Ledger(f, k); L.append({"kind": "genesis"}); L.append({"kind": "score", "v": 1})
check("a fresh ledger verifies", verify(f, k).ok)
with f.open("ab") as fh:
    fh.write(b'{"seq":2,"prev":"aa')                       # crash mid-write
L2 = Ledger(f, k)
check("a truncated tail is repaired on open", L2.repaired == 19, f"{L2.repaired} bytes")
L2.append({"kind": "score", "v": 2})
check("the ledger still verifies after repair + append", verify(f, k).ok)
with f.open("ab") as fh:
    fh.write(json.dumps({"seq": 999, "prev": "0" * 64,
                         "payload": {"ndcg@10": 0.99}, "sig": "forged"}).encode() + b"\n")
try:
    Ledger(f, k); check("the ledger REFUSES to extend a forged chain", False)
except TamperedLedger:
    check("the ledger REFUSES to extend a forged chain", True)
check("verify still reports the forgery",
      not verify(f, k).ok and verify(f, k).bad_seq == 3)
check("verify_on_open=False is available for a deliberate tamper demo",
      Ledger(f, k, verify_on_open=False) is not None)

print()
print(f"  [diagnostic] seatbelt profile: {I.PROFILE_PATH}")
print(f"  [diagnostic] probe results:    experiments/isolation_probe.json")
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
