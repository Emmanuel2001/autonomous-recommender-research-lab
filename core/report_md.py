"""Task-agnostic helpers for generating the markdown deliverables.

PROVENANCE — read this before editing
-------------------------------------
These functions were lifted VERBATIM out of the original repository's `make_submission.py`
(`_key`, `load`, `_fmt`, `_md_table`, `build_verification`, `build_isolation`) and
`experiments/rationale_quality.py` (`segments`).

`make_submission.py` itself is NOT carried into this repository. It was the packager for the
SUPERSEDED task definition: its `pair()` and `composite_index()` compute
`0.5 * (ndcg@10 + recall@50)`, so it raises `KeyError: 'ndcg@10'` on any current-task ledger
and cannot run at all. Only the parts above are task-agnostic, so only those were kept, and
they live here rather than in a module whose name implies it still builds a submission.

`segments()` splits a ledger into runs by `run_start`/`run_end`. It is shape-agnostic and
works for both the typed and the code-first run shapes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from core.ledger import verify


def _key() -> bytes | None:
    k = os.environ.get("TRACK2_SIGNING_KEY", "")
    if not k:
        f = ROOT / "secrets" / "scorer.key"
        k = f.read_text().strip() if f.exists() else ""
    return k.encode() if k else None

def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def _fmt(v, nd=5) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else f"{v:,}")

def _md_table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)

def build_verification(ledgers: list[Path], key: bytes | None) -> tuple[str, dict]:
    rows, status = [], {}
    for p in sorted(ledgers):
        if key is None:
            v, note = None, "NO KEY AVAILABLE"
        else:
            v = verify(p, key)
            note = str(v)
        status[p.name] = {"ok": bool(v and v.ok), "note": note,
                          "entries": (v.n_verified if v else 0)}
        rows.append([f"`{p.name}`", status[p.name]["entries"],
                     "OK" if status[p.name]["ok"] else "FAILED", note])
    body = f"""# Ledger verification

Every number in this submission comes from a signed entry in one of these ledgers.
Re-check them yourself:

```bash
python3 -m core.scorer verify --ledger runs/ledgers/<name>.jsonl
```

The key is `secrets/scorer.key`, which is **not** in the repository -- publishing
it would let anyone forge an entry, and then the signatures would mean nothing. To
verify these files you need the key from us; the hash chain alone still proves that
no entry was reordered, altered or removed from the middle, without any key at all.

{_md_table(["ledger", "entries", "status", "detail"], rows)}

## What the signature does and does not prove

* **Does prove:** the entry was produced by the process holding the key -- the
  scorer -- and has not been altered since; and its POSITION in the chain, which is
  what makes a pre-registered prediction checkable.
* **Does not prove:** that a `proposal` body is true. The scorer countersigns
  orchestrator assertions for ordering only, and says so in every such payload's
  `attested` field.
* **Does not prevent:** truncation of the whole tail. Removing the last N entries
  leaves a shorter chain that still verifies. Detecting that needs an external
  anchor (a published head hash), which we do not have. Stated here rather than
  discovered by a judge.

## Known caveat

`tamper_demo_*.jsonl` is expected to FAIL and is kept deliberately: it contains a
hand-injected forgery written to test detection, plus the entries the scorer
chained on top of it before `Ledger` learned to refuse. Its key was ephemeral and
is gone -- which is why `secrets/scorer.key` is now persisted.
"""
    return body, status

def build_isolation(probe_path: Path) -> str:
    if not probe_path.exists():
        return "# Isolation\n\n_probe results missing; run `python3 -m core.isolation probe`_\n"
    res = json.loads(probe_path.read_text())
    attempts = sorted({k for r in res for k in r["attempts"]})
    head = ["attack"] + [r["backend"] for r in res]
    rows = []
    for k in attempts:
        row = [f"`{k}`"]
        for r in res:
            v = r["attempts"].get(k, "-")
            row.append("**blocked**" if v.startswith("blocked")
                       else ("allowed" if v.startswith("SUCCEEDED") else v[:20]))
        rows.append(row)
    return f"""# Isolation: what is enforced, and by what

Our stated improvement on the Darwin Godel Machine is that its mitigation for
reward hacking was **observational** -- a transparent archive you can read
afterwards -- whereas ours is **architectural**. That claim is only worth making if
the boundary actually holds, so it is measured: `strategy/attack_probe.py` is a
deliberately hostile strategy, it is executed for real under each backend, and the
kernel's answer is what appears below.

`none` is the control. If the attack did not succeed there, the probe would be
testing nothing -- so that column is expected to be all "allowed".

{_md_table(head, rows)}

## What each backend is worth

{chr(10).join(f"* **{r['backend']}** -- {r['sandbox']['enforced_by']}" for r in res)}

`subprocess` is listed honestly: it prepares a data root containing only permitted
files, which stops an accidental read, but the real path still resolves, so it
enforces nothing. It is never presented as a boundary.

**`docker` is implemented but does not appear in the table above, because the daemon
was not running on the machine these measurements were taken on.** It is the only
backend that survives a sandbox escape -- the eval file is not in the container
filesystem at all rather than present-and-denied -- so it is the right choice on a
machine that has it. Untested is untested, and it is not claimed as measured:
`python3 -m core.isolation build-image` then `--isolation docker`.

## Why each attack is in the list

| attack | the leak it represents |
|---|---|
| `read_eval_labels` | the quarantine breach: open the eval-window csv by absolute path |
| `read_ledger` | the ledger holds the validation metric of every config evaluated, so reading it allows hill-climbing on validation without ever facing the gate. Easy to miss, and closed here. |
| `read_signing_key` | with the key, the strategy signs its own numbers and the out-of-process design buys nothing |
| `write_ledger` | append a forged entry |
| `overwrite_scorer` | rewrite the code that grades it -- the Sakana failure mode |
| `network` | fetch a label set, or exfiltrate |
| `escape_via_child` | a sandbox not inherited across fork/exec is not a sandbox |

## Limits, stated rather than found

* A seatbelt escape defeats it. `docker` answers that; `seatbelt` answers a
  rewritten strategy, which is the threat we actually have.
* The scorer cannot verify which backend produced a prediction file. That claim is
  stored as `isolation_claimed` -- named so no reader mistakes it for a checked
  fact. The probe is the checked part.
* With `--allow-log-random`, log_random is inside the sandbox and carries
  eval-window clicks (67 of 144,042 quarantined test pairs). The read-deny is
  weaker by exactly that much.
* Isolation costs about 48% more wall clock per evaluation (28.8s vs 19.5s for the
  same config), from re-loading data per iteration and ranking every user instead
  of the scoreable subset. Metrics are identical to five decimals -- asserted in
  `tests/test_scorer.py`.
"""

def segments(entries: list[dict]) -> list[dict]:
    """Split one ledger into runs, and pair each proposal with what happened next.

    Pairing is positional: a proposal is followed by the score of the config it
    named and then the gate comparison. Positional is safe because the driver is
    single-threaded and the scorer serialises the socket -- and the config hash is
    checked, so a mispairing is caught rather than silently averaged in.
    """
    runs, cur = [], None
    for e in entries:
        pl = e.get("payload", {})
        k = pl.get("kind")
        if k == "run_start":
            cur = {"start": pl["body"], "proposals": [], "scores": [], "compares": [],
                   "events": [], "end": None}
            runs.append(cur)
        elif cur is None:
            continue                          # pre-run entries (genesis, warm-up score)
        elif k == "proposal":
            cur["proposals"].append({"seq": e["seq"], **pl["body"]})
        elif k == "score":
            cur["scores"].append({"seq": e["seq"], **pl})
        elif k == "compare":
            cur["compares"].append({"seq": e["seq"], **pl})
        elif k == "event":
            cur["events"].append({"seq": e["seq"], **pl["body"]})
        elif k == "run_end":
            cur["end"] = pl["body"]
            # Close the segment. Without this, `cur` stays open and the NEXT run's
            # warm-up incumbent score -- which lands between run_end and the next
            # run_start -- is appended to the previous run's scores. Pairing also
            # matches on config hash so nothing mispaired in practice, but the run
            # boundary is a fact about the ledger and should not depend on that.
            cur = None
    return runs
