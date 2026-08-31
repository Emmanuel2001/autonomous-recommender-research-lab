"""The ledger's own claims must not contradict each other.

A signed chain is evidence only if its statements are mutually consistent. These test two
places where they were not:

  * a score payload could claim `isolation_claimed: "docker"` while the SAME chain's
    genesis said *"docker daemon not responding"*;
  * genesis recorded `include_quarantine` as given on the command line, even on the
    official task where the loader has no such switch and the flag does nothing.

Neither is about whether a number is right. Both are about whether the artifact can be
read as evidence, which is the whole point of signing it.
"""
import ast
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import scorer as SC                                          # noqa: E402
from core.ledger import Ledger                                         # noqa: E402
from core.official_scorer import score_official                        # noqa: E402

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name}  {detail}")


TMP = Path(tempfile.mkdtemp(prefix="track2_claims_"))
KEY = b"k" * 32
AVAIL = {"none": "", "subprocess": "", "seatbelt": "", "docker": "daemon not responding"}


class FakeRows:
    """Minimal stand-in: score_official needs labels, user_id and a content hash. Using a
    fake keeps this suite off the 1.1M-row loader, so it runs in milliseconds."""
    split = "valid"

    def __init__(self, n=400):
        rng = np.random.default_rng(7)
        self.user_id = np.repeat(np.arange(n // 4), 4)
        self.label = (rng.random(n) < 0.3).astype(np.float64)
        self._n = n

    def __len__(self):
        return self._n

    def content_sha256(self):
        return "f" * 64


ROWS = FakeRows()
SF = TMP / "s.npy"
np.save(SF, np.random.default_rng(1).random(len(ROWS)))


def flags_for(claim, avail=AVAIL, i=[0]):
    i[0] += 1
    L = Ledger(TMP / f"l{i[0]}.jsonl", KEY)
    e = score_official(SF, "valid", ROWS, "cfg", L, isolation=claim,
                       reference=False, isolation_available=avail)
    return [f for f in e["payload"]["flags"] if f.startswith("isolation_")], e["payload"]


print("== an isolation claim is cross-checked against what the scorer measured")
f, pl = flags_for("seatbelt")
check("an AVAILABLE backend is not flagged", f == [], str(f))
check("the claim is still recorded verbatim", pl["isolation_claimed"] == "seatbelt")
f, pl = flags_for("docker")
check("an UNAVAILABLE backend is flagged",
      f == ["isolation_claimed_unavailable:docker"], str(f))
check("the score is still signed, not discarded",
      pl["metrics"]["primary"] is not None and pl["isolation_claimed"] == "docker")
f, _ = flags_for("kubernetes")
check("an UNKNOWN backend is flagged distinctly",
      f == ["isolation_claimed_unknown_backend:kubernetes"], str(f))
check("the two flag kinds are distinguishable",
      flags_for("docker")[0] != flags_for("kubernetes")[0])
for neutral in ("unspecified", "none"):
    f, _ = flags_for(neutral)
    check(f"{neutral!r} is not flagged", f == [], str(f))
f, _ = flags_for(None)
check("None is not flagged", f == [], str(f))
f, _ = flags_for("docker", avail=None)
check("with no availability map, behaviour is unchanged (older callers)", f == [], str(f))
f, _ = flags_for("docker", avail={})
check("an empty map flags the claim as an unknown backend",
      f == ["isolation_claimed_unavailable:docker"] or
      f == ["isolation_claimed_unknown_backend:docker"], str(f))
check("a flag names the backend, so a reader need not guess",
      "docker" in flags_for("docker")[0][0])

print("== the scorer measures availability ONCE and shares that measurement")
src = inspect.getsource(SC.serve)
check("availability is probed into a variable, not inline per use",
      "isolation_avail = _isolation_available()" in src)
check("genesis records that same variable",
      '"isolation_available": isolation_avail' in src)
check("the score path is handed that same variable",
      "isolation_available=isolation_avail" in src)
check("_isolation_available is called exactly once in serve",
      src.count("_isolation_available()") == 1, str(src.count("_isolation_available()")))

print("== include_quarantine must not be recorded as done when it does nothing")
check("serve refuses --include-quarantine on the official task",
      "has no effect on task=official" in src)
check("the refusal explains it would be a false self-declaration",
      "false self-declaration" in src)
check("it still works on the legacy task, where it has an effect",
      "include_quarantine=include_quarantine" in src)
# the refusal must sit BEFORE the loader and before any ledger write
i_refuse = src.index("has no effect on task=official")
check("refused before the official loader runs",
      i_refuse < src.index("_load_official()"), "would load 1.1M rows first")
check("refused before anything is appended to the ledger",
      i_refuse < src.index('led.append({"kind": "genesis"'),
      "would write a genesis entry before refusing")

print("== no unused imports left behind in the modules touched")
for mod in ("core/scorer.py", "core/official_scorer.py", "core/handshake.py"):
    p = Path(__file__).resolve().parents[1] / mod
    tree = ast.parse(p.read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.value.id for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    imported = [(a.asname or a.name.split(".")[0])
                for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    dead = [m for m in imported if m not in used]
    check(f"{mod} has no unused plain imports", not dead, str(dead))

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
