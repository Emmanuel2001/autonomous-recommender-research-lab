"""Client-side preconditions on the scorer. Two questions, asked before any work.

These are separate from core/scorer.py on purpose: they are what a CLIENT must check
before trusting a scorer it did not start, and keeping them here makes them unit-testable
without a socket, a key, or a 20-second data load.

Both existed as no-ops for a while, and both were reached in practice:

  1. IS IT SERVING MY TASK?  `ping` has always returned `task`, and both clients checked
     only `ok`. Pointing the official driver at a scorer started with `--task legacy`
     therefore passed startup, spent ~20s loading 1.1M rows, and then failed on EVERY
     evaluation with "REFUSED: this scorer was started with task=legacy". A
     misconfiguration detectable in a millisecond, reported as a run-long stream of
     crashes. Observed for real.

  2. ARE WE LOOKING AT THE SAME DATA?  `read_scores` catches a row-COUNT mismatch, so the
     loud case was covered. The quiet case was not: the same count in a different ORDER
     misaligns every prediction with a label belonging to another row, and the only symptom
     is a mediocre score -- indistinguishable from a mediocre model. `Rows.content_sha256()`
     binds order, and the scorer returns its own view of it in every score payload, so this
     costs one comparison of a string the client is handed anyway.

`make_official_submission.py` needed both more than the driver did: it writes the graded
artifact, and a silent misalignment there is the entry.
"""
from __future__ import annotations


class HandshakeError(RuntimeError):
    """A precondition failed. The message is written to be actionable at a terminal."""


def require_task(ping: dict, want: str) -> None:
    """The scorer must be serving the task this client speaks."""
    if not isinstance(ping, dict) or not ping.get("ok"):
        raise HandshakeError(
            "scorer not reachable. Start it in its own terminal -- it holds the signing "
            "key:\n  python3 -m core.scorer serve --task " + want)
    got = ping.get("task")
    if got != want:
        raise HandshakeError(
            f"scorer is serving task={got!r}, but this client speaks {want!r}. Every "
            f"scoring request would be refused. Restart it:\n"
            f"  python3 -m core.scorer serve --task {want}")


def require_verified_chain(ping: dict) -> None:
    """The scorer must have VERIFIED its ledger before appending to it.

    `core.scorer serve --allow-broken-chain` skips verify-on-open. Measured: pointed at a
    spliced ledger it starts silently and signs a new entry at seq=1000 directly on top of
    the forgery. Everything signed after a splice is unverifiable, so a client that
    produces a graded artifact against such a scorer produces a worthless one -- and used
    to have no way to know, because neither `ping` nor the genesis entry said so.

    A scorer too old to report the field is refused rather than assumed safe, exactly as
    in `require_same_data`: one rule, applied the same way, since "no answer" and "yes"
    are the two things a precondition must never conflate.
    """
    v = (ping or {}).get("chain_verified_on_open")
    if v is True:
        return
    if v is None:
        raise HandshakeError(
            "the scorer did not report whether it verified its ledger chain on open. "
            "Refusing rather than assuming it did -- restart it from the current code:\n"
            "  python3 -m core.scorer serve --task official")
    raise HandshakeError(
        "REFUSING: this scorer was started with --allow-broken-chain, so it appended to "
        "its ledger WITHOUT verifying the chain first. Every entry signed after a splice "
        "is unverifiable, which makes any artifact produced now worthless as evidence.\n"
        "That flag exists only for the deliberate tamper demonstration. For real work, "
        "restart without it:\n"
        "  python3 -m core.scorer serve --task official")


def require_same_data(mine: str, theirs: str | None, split: str = "valid") -> None:
    """The client's view of a split must be the scorer's view of it, ORDER included.

    `theirs` may be None on an older payload that predates the field; that is treated as
    "cannot check" rather than "agrees", and says so, because silently passing an
    unverifiable case is how a check becomes decoration.
    """
    if not mine:
        raise HandshakeError(f"cannot verify data agreement: no local {split} content hash")
    if theirs is None:
        raise HandshakeError(
            f"cannot verify data agreement: the scorer returned no "
            f"split_content_sha256 for {split}. Refusing rather than assuming; a scorer "
            f"this old should be restarted from the current code.")
    if theirs != mine:
        raise HandshakeError(
            f"REFUSING TO RUN: this process and the scorer are looking at different data.\n"
            f"  mine   ({split} content sha256): {mine}\n"
            f"  scorer ({split} content sha256): {theirs}\n"
            f"Same row count, different content or order -- every prediction would be "
            f"scored against a label belonging to a different row, and the only symptom "
            f"would be a low score. Check that both processes read the same data "
            f"directory and that no CSV changed between starting them.")
