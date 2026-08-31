"""Declared dependencies for generated patches.

The sandbox has no network, so a patch can never install anything itself -- resolution
happens in the harness, before the sandbox is entered. What is tested here is that the
allowlist cannot be talked around, and that a resolved library actually reaches the child.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import deps as D
from agent.patch import PatchError, run_sandboxed
from core import isolation as I

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

# 1 -- parsing the declaration ----------------------------------------------------
check("parses one name", D.parse_requires("# requires: numpy") == ["numpy"])
check("parses several on one line",
      D.parse_requires("# requires: numpy, scipy") == ["numpy", "scipy"])
check("parses across lines and preserves order",
      D.parse_requires("# requires: torch\n# requires: numpy") == ["torch", "numpy"])
check("deduplicates", D.parse_requires("# requires: numpy\n# requires: numpy") == ["numpy"])
check("case-insensitive directive", D.parse_requires("# REQUIRES: numpy") == ["numpy"])
check("no declaration means no dependency", D.parse_requires("import numpy") == [])
check("an undeclared import is NOT auto-resolved",
      D.parse_requires("import lightgbm\nlightgbm.train()") == [],
      "a patch must SAY what it needs, so the ledger records an intention")

# 2 -- the allowlist cannot be talked around --------------------------------------
def refused(names):
    try:
        D.validate_declarations(names); return None
    except D.DependencyError as e:
        return str(e)

for bad, why in (("lightgbm==4.0", "version specifier"),
                 ("lightgbm>=1", "version range"),
                 ("https://example.invalid/x.whl", "URL"),
                 ("git+ssh://example.invalid/x", "VCS ref"),
                 ("numpy[all]", "extras"),
                 ("-r reqs.txt", "pip flag"),
                 ("../../etc/passwd", "path traversal"),
                 ("reqeusts", "typosquat-shaped name not on the list"),
                 ("evilpkg", "simply not on the list")):
    m = refused([bad])
    check(f"refuses {why}", m is not None, f"{bad!r}")
check("the refusal explains WHY an allowlist exists",
      "language model" in (refused(["evilpkg"]) or ""))
check("every allowlisted name passes", refused(sorted(D.ALLOWLIST)[:4]) is None)
check("more than the cap is refused",
      refused(["numpy", "scipy", "torch", "pandas", "lightgbm"]) is not None)
check("empty declaration list is fine", refused([]) is None)

# 3 -- resolution reports honestly ------------------------------------------------
rep = D.resolve(["numpy"], install=False)
check("an already-importable dependency is reported present, not installed",
      [x["name"] for x in rep["already_present"]] == ["numpy"] and not rep["installed"])
check("resolve records the version it found",
      rep["already_present"][0]["version"] is not None)
absent = [n for n, (m, _) in D.ALLOWLIST.items() if not D._importable(m)]
if absent:
    rep2 = D.resolve([absent[0]], install=False)
    check("a missing dependency with install disabled is reported FAILED, not raised",
          rep2["failed"] and rep2["failed"][0]["name"] == absent[0],
          f"{absent[0]!r} is absent here")
else:
    print("  [skip] every allowlisted package is installed; cannot test the absent path")

# 4 -- a refused declaration stops the patch BEFORE any compute -------------------
SRC = ("import numpy as np\ndef fit_predict(train, dev, heldout, params):\n"
       "    return {'dev': np.zeros(len(dev.user_id)),"
       " 'heldout': np.zeros(len(heldout.user_id))}\n")
try:
    run_sandboxed("# requires: evilpkg\n" + SRC, {}, backend="seatbelt")
    check("a non-allowlisted declaration refuses the patch", False)
except PatchError as e:
    check("a non-allowlisted declaration refuses the patch",
          "not on the allowlist" in str(e))

# 5 -- THE MECHANISM: does a resolved library reach the sandboxed child -----------
if not I.available().get("seatbelt", "x") == "":
    print("  [skip] seatbelt unavailable; cannot test PYTHONPATH threading")
else:
    D.TARGET.mkdir(parents=True, exist_ok=True)
    pkg = D.TARGET / "track2_dep_probe"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("MARK = 'reached'\n")
    try:
        sb = I.make_sandbox("seatbelt")
        extra = {"PYTHONPATH": str(D.TARGET)}
        ok = subprocess.run(sb.argv(["-c", "import track2_dep_probe as t; print(t.MARK)"]),
                            capture_output=True, text=True, cwd=str(I.ROOT),
                            env=I.child_env(sb.data.path, extra), timeout=300)
        check("a library in patch_libs IS importable in the sandbox with PYTHONPATH",
              ok.returncode == 0 and "reached" in ok.stdout,
              (ok.stdout or ok.stderr).strip()[:60])
        no = subprocess.run(sb.argv(["-c", "import track2_dep_probe"]),
                            capture_output=True, text=True, cwd=str(I.ROOT),
                            env=I.child_env(sb.data.path), timeout=300)
        check("and is NOT importable without it (so the path is what carries it)",
              no.returncode != 0)
    finally:
        shutil.rmtree(pkg, ignore_errors=True)

# 6 -- env_extra cannot reintroduce a secret --------------------------------------
e = I.child_env(D.TARGET, {"TRACK2_SIGNING_KEY": "x", "TRACK2_LLM_API_KEY": "y",
                           "PYTHONPATH": "/tmp/p"})
check("env_extra is re-scrubbed of the signing key", "TRACK2_SIGNING_KEY" not in e)
check("env_extra is re-scrubbed of the LLM key", "TRACK2_LLM_API_KEY" not in e)
check("but legitimate extras survive", e.get("PYTHONPATH") == "/tmp/p")
check("the sandbox flag is still set", e.get("TRACK2_SANDBOX") == "1")

print()
bad_l = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad_l)}/{len(R)} passed" + (f"  FAILURES: {bad_l}" if bad_l else ""))
sys.exit(1 if bad_l else 0)
