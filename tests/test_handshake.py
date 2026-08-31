"""Client-side preconditions on the scorer. Both were no-ops that were reached in practice.

Deterministic: no socket, no key, no data load. That is the point of factoring them out --
the earlier attempt to verify them by running against whatever scorer happened to be
listening is exactly the ambient-state mistake this suite has already been bitten by.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.handshake import (HandshakeError, require_same_data,        # noqa: E402
                            require_task, require_verified_chain)

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name}  {detail}")


def raises(fn, *a, needle=None):
    try:
        fn(*a)
        return False
    except HandshakeError as e:
        return needle is None or needle in str(e)


print("== require_task")
check("a matching task passes",
      require_task({"ok": True, "task": "official"}, "official") is None)
check("the legacy task is refused (the case actually hit)",
      raises(require_task, {"ok": True, "task": "legacy"}, "official", needle="legacy"))
check("the refusal names the task the client speaks",
      raises(require_task, {"ok": True, "task": "legacy"}, "official", needle="'official'"))
check("the refusal gives the exact command to fix it",
      raises(require_task, {"ok": True, "task": "legacy"}, "official",
             needle="serve --task official"))
check("a MISSING task field is refused, not assumed",
      raises(require_task, {"ok": True}, "official"))
check("task=None is refused", raises(require_task, {"ok": True, "task": None}, "official"))
check("ok=False is refused as unreachable",
      raises(require_task, {"ok": False}, "official", needle="not reachable"))
check("an empty response is refused", raises(require_task, {}, "official"))
check("a non-dict response is refused", raises(require_task, None, "official"))
check("a string response is refused", raises(require_task, "ok", "official"))
check("the want argument is honoured, not hardcoded",
      require_task({"ok": True, "task": "legacy"}, "legacy") is None
      and raises(require_task, {"ok": True, "task": "official"}, "legacy"))
check("case is not silently normalised",
      raises(require_task, {"ok": True, "task": "Official"}, "official"))

print("== require_same_data")
H = "a" * 64
check("identical hashes pass", require_same_data(H, H) is None)
check("different hashes refused", raises(require_same_data, H, "b" * 64))
check("the refusal shows BOTH hashes",
      raises(require_same_data, H, "b" * 64, needle="a" * 64)
      and raises(require_same_data, H, "b" * 64, needle="b" * 64))
check("the refusal explains the silent symptom",
      raises(require_same_data, H, "b" * 64, needle="only symptom"))
check("a one-character difference is caught",
      raises(require_same_data, H, "a" * 63 + "b"))
check("None from the scorer is refused, NOT treated as agreement",
      raises(require_same_data, H, None, needle="Refusing rather than assuming"))
check("an empty local hash is refused", raises(require_same_data, "", H))
check("a None local hash is refused", raises(require_same_data, None, H))
check("both empty is refused, not vacuously equal", raises(require_same_data, "", ""))
check("the split name appears in the message",
      raises(require_same_data, H, "b" * 64, "test", needle="test"))
check("the split name defaults to valid",
      raises(require_same_data, H, "b" * 64, needle="valid"))

print("== require_verified_chain")
check("a verified chain passes",
      require_verified_chain({"ok": True, "chain_verified_on_open": True}) is None)
check("--allow-broken-chain is refused",
      raises(require_verified_chain, {"ok": True, "chain_verified_on_open": False},
             needle="allow-broken-chain"))
check("the refusal explains WHY it matters",
      raises(require_verified_chain, {"ok": True, "chain_verified_on_open": False},
             needle="unverifiable"))
check("the refusal says the flag is for the tamper demo only",
      raises(require_verified_chain, {"ok": True, "chain_verified_on_open": False},
             needle="tamper demonstration"))
check("a MISSING field is refused, not assumed safe",
      raises(require_verified_chain, {"ok": True}, needle="Refusing rather than assuming"))
check("an empty ping is refused", raises(require_verified_chain, {}))
check("a None ping is refused", raises(require_verified_chain, None))
check("a truthy non-True value is NOT accepted as verified",
      raises(require_verified_chain, {"ok": True, "chain_verified_on_open": 1})
      and raises(require_verified_chain, {"ok": True, "chain_verified_on_open": "yes"}))

print("== the scorer records and reports it")
import inspect
from core import scorer as SC
src = inspect.getsource(SC.serve)


def _genesis_dict(source: str) -> str:
    """The genesis dict literal, with COMMENTS REMOVED.

    The first version of this took the 900 characters after the append marker, and the
    comment explaining the new field is itself ~600 characters, so the check reported the
    field missing while it sat just past the window. That is bug 30 exactly -- a broken
    diagnostic that looks like a real defect -- and it is why the code was inspected
    before being 'fixed'. Slice on comment-stripped source and count braces instead.
    """
    body = source.split('led.append({"kind": "genesis"')[1]
    lines = [ln for ln in body.splitlines() if not ln.strip().startswith("#")]
    out, depth = [], 1
    for ln in lines:
        out.append(ln)
        depth += ln.count("{") - ln.count("}")
        if depth <= 0:
            break
    return "\n".join(out)


gen = _genesis_dict(src)
check("the genesis-dict extractor terminates at the dict, not a byte count",
      len(gen) < len(src) and "isolation_available" in gen, f"{len(gen)} chars")
check("genesis records chain_verified_on_open", "chain_verified_on_open" in gen, gen[:120])
check("genesis derives it from allow_broken_chain",
      "not allow_broken_chain" in gen)
check("genesis still records the fields it always did",
      all(k in gen for k in ("include_quarantine", "repaired_tail_bytes",
                             "manifest_sha256", "task")))
check("ping reports chain_verified_on_open",
      '"chain_verified_on_open": not allow_broken_chain' in src)
check("the Ledger is still opened with verify_on_open by default",
      "verify_on_open=not allow_broken_chain" in src)

print("== the report surfaces all three states")
from agent.official_report import _sec_chain
check("verified -> no warning",
      "note warn" not in _sec_chain({"chain_verified_on_open": True}))
check("unverified -> WARNING",
      "note warn" in _sec_chain({"chain_verified_on_open": False}))
check("unverified names the flag",
      "allow-broken-chain" in _sec_chain({"chain_verified_on_open": False}))
check("missing -> reported as unknown, not as a pass",
      "unknown" in _sec_chain({}).lower()
      and "note warn" not in _sec_chain({}))
check("missing does NOT claim the chain was verified",
      "verified this chain" not in _sec_chain({}))

print("== both clients actually call them")
root = Path(__file__).resolve().parents[1]
drv = (root / "agent" / "official_driver.py").read_text()
sub = (root / "make_official_submission.py").read_text()
for name, src in [("official_driver", drv), ("make_official_submission", sub)]:
    check(f"{name} calls require_task", "require_task(" in src)
    check(f"{name} calls require_same_data", "require_same_data(" in src)
    check(f"{name} calls require_verified_chain", "require_verified_chain(" in src)
    check(f"{name} no longer accepts a scorer on ok alone",
          'request({"op": "ping"}).get("ok")' not in src
          and 'if not ping.get("ok")' not in src, "still checks only ok")

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
