"""Append-only, hash-chained, HMAC-signed ledger.

GUARANTEE AND ITS LIMIT
-----------------------
This provides tamper-EVIDENCE, not tamper-PREVENTION. Any process with write
access to the file can append or truncate it. What it cannot do without the
signing key is produce an entry that VERIFIES. So:

  * the scorer process holds the key and is the only thing that can sign;
  * the orchestrator can append garbage, but `verify()` will reject it;
  * a metric that has no verifying entry is not a metric.

The key must be supplied to the scorer out of band (started by a human, key in
its environment only) so that the orchestrator never holds it. If the
orchestrator can read the key, this whole module is decoration.

CHAIN
-----
entry n is  {"seq": n, "prev": <sha256 of entry n-1's canonical bytes>,
             "payload": {...}, "sig": <hmac-sha256 over seq|prev|payload>}
Entry 0 is a genesis entry whose `prev` is GENESIS_PREV.

The signature covers `prev`, so entries cannot be reordered or spliced.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path

GENESIS_PREV = "0" * 64
_SIG_FIELD = "sig"


def canonical(obj) -> bytes:
    """Deterministic serialisation. sort_keys + no whitespace.

    CPython's float repr is round-trippable, so json.dumps(json.loads(x)) == x
    for any float we wrote ourselves. test_ledger asserts this.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _entry_hash(entry: dict) -> str:
    """Hash of a COMPLETE entry (signature included) -> the next entry's prev."""
    return hashlib.sha256(canonical(entry)).hexdigest()


def _sign(key: bytes, seq: int, prev: str, payload: dict) -> str:
    msg = canonical({"seq": seq, "prev": prev, "payload": payload})
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    n_verified: int
    incomplete_tail: bool          # final line truncated: normal after a crash
    error: str | None = None
    bad_seq: int | None = None

    def __str__(self) -> str:
        if self.ok:
            s = f"LEDGER OK  {self.n_verified} entries verified"
            return s + ("  (+1 incomplete trailing line, ignored)" if self.incomplete_tail else "")
        return f"LEDGER FAILED at seq={self.bad_seq}: {self.error}"


class TamperedLedger(RuntimeError):
    """The existing chain does not verify. Refuse to extend it."""


class Ledger:
    """Writer. Only the process holding the key should construct this.

    VERIFY ON OPEN
    --------------
    Opening an existing ledger re-verifies it before appending, and refuses if it
    does not verify. Without this the writer happily chains onto forged bytes:
    measured on a real scratch ledger, someone (me) appended a hand-written
    {"seq":999,"metrics":{"ndcg@10":0.99}} to test detection, and the next scorer
    start resumed from it as seq=1000 and signed a genesis entry on top. `verify`
    would still have caught it afterwards -- but only afterwards, and everything
    signed after the splice was unusable. Detection after the fact is what we
    criticise DGM's archive for; a writer that refuses is prevention.

    REPAIRING A TRUNCATED TAIL
    --------------------------
    A crash mid-append leaves partial bytes with no newline. The old writer then
    appended a fresh entry directly after those bytes, which turned a recoverable
    crash into a permanently unverifiable ledger: the partial line was no longer
    final, so verify() called it an "unparseable interior line" forever. Those
    bytes cannot verify under any key, so they are truncated away on open and the
    fact is returned in `self.repaired` for the caller to record.
    """

    def __init__(self, path: str | Path, key: bytes, verify_on_open: bool = True):
        if not key:
            raise ValueError("refusing to sign with an empty key")
        self.path = Path(path)
        self._key = key
        self.repaired = 0                      # bytes of unverifiable tail dropped
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if verify_on_open and self.path.exists():
            v = verify(self.path, key)
            if not v.ok:
                raise TamperedLedger(
                    f"refusing to append to {self.path}: {v}. The chain up to this "
                    f"point cannot be extended without making every later entry "
                    f"unverifiable too. Move the file aside and start a new ledger, "
                    f"or pass verify_on_open=False if you are deliberately "
                    f"demonstrating tamper detection.")
        self._seq, self._prev = self._tail()

    def _tail(self) -> tuple[int, str]:
        """Resume from the last COMPLETE entry, truncating an unverifiable tail."""
        if not self.path.exists():
            return 0, GENESIS_PREV
        data = self.path.read_bytes()
        parts = data.split(b"\n")
        terminated, trailing = parts[:-1], parts[-1]

        seq, prev, good_end, clean = 0, GENESIS_PREV, 0, True
        for ln in terminated:
            good_end += len(ln) + 1
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                clean = False
                break
            if not isinstance(e, dict) or "seq" not in e:
                clean = False
                break
            seq, prev = int(e["seq"]) + 1, _entry_hash(e)

        # Only ever drop the un-terminated trailing bytes, and only when every
        # terminated line parsed. Anything else is a job for a human, not for a
        # writer that is about to append.
        if clean and trailing:
            self.repaired = len(trailing)
            with self.path.open("r+b") as fh:
                fh.truncate(good_end)
                fh.flush()
                os.fsync(fh.fileno())
        return seq, prev

    def append(self, payload: dict) -> dict:
        """Sign and durably append one entry. Returns the entry."""
        entry = {
            "seq": self._seq,
            "prev": self._prev,
            "payload": payload,
            _SIG_FIELD: _sign(self._key, self._seq, self._prev, payload),
        }
        line = canonical(entry) + b"\n"
        # Durability: the entry must be on disk before we advance our state, so
        # a crash truncates the tail rather than skipping a link.
        with self.path.open("ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._seq += 1
        self._prev = _entry_hash(entry)
        return entry


def verify(path: str | Path, key: bytes) -> VerifyResult:
    """Re-check every link and every signature.

    Distinguishes an incomplete FINAL line (expected after a crash or a
    disk-full event) from a broken interior link or a bad signature (tampering).
    Without that distinction your own integrity tool accuses you of fraud at 3am.
    """
    path = Path(path)
    if not path.exists():
        return VerifyResult(False, 0, False, "ledger file does not exist")

    raw = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]
    n_ok, prev, incomplete = 0, GENESIS_PREV, False

    for idx, line in enumerate(raw):
        is_last = idx == len(raw) - 1
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            if is_last:
                incomplete = True
                break
            return VerifyResult(False, n_ok, False, "unparseable interior line", n_ok)

        if not isinstance(e, dict) or set(e) != {"seq", "prev", "payload", _SIG_FIELD}:
            if is_last:
                incomplete = True
                break
            return VerifyResult(False, n_ok, False, "malformed interior entry", n_ok)

        if e["seq"] != n_ok:
            return VerifyResult(False, n_ok, False,
                                f"sequence gap: expected {n_ok}, found {e['seq']}", n_ok)
        if e["prev"] != prev:
            return VerifyResult(False, n_ok, False, "broken hash chain", n_ok)

        expect = _sign(key, e["seq"], e["prev"], e["payload"])
        if not hmac.compare_digest(expect, e[_SIG_FIELD]):
            return VerifyResult(False, n_ok, False, "bad signature", n_ok)

        # Canonicalisation must be stable, or verification is luck.
        if canonical(e) != line.encode("utf-8"):
            return VerifyResult(False, n_ok, False, "non-canonical serialisation", n_ok)

        prev = _entry_hash(e)
        n_ok += 1

    return VerifyResult(True, n_ok, incomplete)


def read_payloads(path: str | Path) -> list[dict]:
    """Payloads of complete entries. Does NOT verify -- call verify() first."""
    out = []
    for ln in Path(path).read_text(encoding="utf-8").split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln)["payload"])
        except Exception:
            break
    return out
