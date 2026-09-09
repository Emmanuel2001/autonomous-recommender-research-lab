"""Resolve a compact signed tournament DECISION back into a readable view.

WHY THIS EXISTS
---------------
Two signed records describe one tournament:

    tournament_proposed   the candidate set, bounded, signed BEFORE the judge is called
    tournament            the judge's decision, signed after -- and deliberately COMPACT

The decision names its proposal by `proposal_seq` and carries only candidate id and lane.
Duplicating the eight candidate descriptions cost ~8.8 kB of the scorer's 12,288-byte body
cap and, on the first real eight-candidate Opus tournament, pushed the decision to 13,004
bytes: the record was refused and a fully judged iteration was abandoned after both LLM
calls had been paid for.

So the RENDERED view hydrates; the SIGNED record stays compact. Every reader -- the HTML
report, the Research Console, the control timeline -- goes through `hydrate()` so there is
one place where "which record owns the prose" is decided.

WHAT THIS MUST NEVER DO
-----------------------
Invent a candidate. A decision whose proposal cannot be found is an evidence gap, and a
reader that filled it with plausible text would be manufacturing the exact thing the signed
chain exists to prevent. `hydrate()` reports the gap in `_hydration` and leaves the
candidates empty; the renderers say so on the page.

THE FOUR STATES, KEPT SEPARATE ON PURPOSE
-----------------------------------------
    embedded     a historical record that still carries full candidate descriptions.
                 Pre-compaction ledgers are evidence and must keep rendering.
    resolved     compact, and its proposal was found in this ledger. The normal case.
    missing_seq  compact, and it names no proposal at all. Older or hand-written record;
                 unavailable data, not a broken chain.
    unresolved   compact, names a proposal seq, and no such record is in this ledger.
                 THIS one is an integrity signal: a decision is pointing at evidence that
                 should exist and does not.
"""
from __future__ import annotations

#: Fields that only ever appear on a FULL candidate description. A compact `candidate_refs`
#: entry carries id and lane and nothing else, so the presence of any of these identifies a
#: historical embedded record without depending on a version number nobody wrote down.
_FULL_MARKERS = ("hypothesis", "mechanism", "information_value", "files_to_change")


def _seq_of(record: dict):
    """A record's own sequence number.

    The HTML report stores it as `_seq`, the console as `seq`, and a raw ledger entry has
    it beside the payload. Accepting all three keeps the hydrator usable from every reader
    rather than forcing one of them to rewrite its loader.
    """
    for key in ("_seq", "seq"):
        if record.get(key) is not None:
            return record[key]
    return None


def _same_seq(a, b) -> bool:
    """Compare sequence numbers without caring whether one side arrived as a string."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return False


def is_compact(decision: dict) -> bool:
    """True when the decision defers its candidate prose to a proposal record."""
    for c in (decision.get("candidates") or []):
        if isinstance(c, dict) and any(c.get(k) for k in _FULL_MARKERS):
            return False
    return True


def find_proposal(decision: dict, proposals) -> dict | None:
    """The `tournament_proposed` record this decision names, or None.

    Matching is by `proposal_seq` first, because that is the binding the driver signed.
    Iteration is a fallback for records written before `proposal_seq` existed -- it is a
    weaker key (a repaired iteration can produce more than one proposal), so the newest
    match wins and the caller still reports the state as `missing_seq`.
    """
    want = decision.get("proposal_seq")
    if want is not None:
        for p in proposals or ():
            if _same_seq(_seq_of(p), want):
                return p
        return None
    it = decision.get("iteration")
    if it is None:
        return None
    matches = [p for p in (proposals or ()) if p.get("iteration") == it]
    return matches[-1] if matches else None


def hydrate(decision: dict, proposals=None) -> dict:
    """Return a RENDERED copy of `decision` with full candidate descriptions resolved.

    The input is never mutated: the signed record stays exactly as the scorer holds it.
    The copy gains `candidates` (full descriptions when they could be resolved) and
    `_hydration` describing how -- see the module docstring for the four states.
    """
    view = dict(decision or {})
    embedded = view.get("candidates") or []

    if embedded and not is_compact(view):
        view["_hydration"] = {"state": "embedded", "resolved": len(embedded),
                              "proposal_seq": view.get("proposal_seq"),
                              "note": "historical record; candidates are stored inline"}
        return view

    refs = [c for c in (view.get("candidate_refs") or []) if isinstance(c, dict)]
    proposal = find_proposal(view, proposals or ())
    full = [c for c in ((proposal or {}).get("candidates") or []) if isinstance(c, dict)]

    if full:
        by_id = {str(c.get("id")): c for c in full}
        # Order follows the decision's own refs when it has them, so the reader sees the
        # set the judge ranked. A candidate present in the proposal but not referenced is
        # still shown rather than silently dropped.
        ordered, seen = [], set()
        for r in refs:
            c = by_id.get(str(r.get("id")))
            if c is not None:
                ordered.append(c)
                seen.add(str(r.get("id")))
        ordered.extend(c for c in full if str(c.get("id")) not in seen)
        view["candidates"] = ordered
        view["_hydration"] = {
            "state": "resolved", "resolved": len(ordered),
            "referenced": len(refs),
            "proposal_seq": _seq_of(proposal),
            "unresolved_ids": sorted(str(r.get("id")) for r in refs
                                     if str(r.get("id")) not in by_id),
        }
        return view

    # Nothing to hydrate from. Say which kind of nothing: a decision that names no
    # proposal is incomplete data; one that names a proposal this ledger does not hold is
    # a chain that does not resolve.
    view["candidates"] = []
    if view.get("proposal_seq") is None:
        view["_hydration"] = {
            "state": "missing_seq", "resolved": 0, "referenced": len(refs),
            "proposal_seq": None,
            "note": ("this decision names no tournament_proposed record, so the candidate "
                     "descriptions cannot be resolved from this ledger")}
    else:
        view["_hydration"] = {
            "state": "unresolved", "resolved": 0, "referenced": len(refs),
            "proposal_seq": view.get("proposal_seq"),
            "note": (f"this decision names tournament_proposed seq "
                     f"{view.get('proposal_seq')}, which is not present in this ledger")}
    return view


def hydrate_all(decisions, proposals=None) -> list[dict]:
    return [hydrate(d, proposals) for d in (decisions or ())]


#: Message shown wherever a reader cannot resolve a decision's candidates. Kept here so the
#: report, the console and the timeline all say the same thing about the same evidence gap.
UNAVAILABLE_NOTE = {
    "missing_seq": ("Candidate descriptions unavailable: this tournament decision names no "
                    "tournament_proposed record. Nothing is shown rather than reconstructed."),
    "unresolved": ("INTEGRITY: this tournament decision names a tournament_proposed record "
                   "that is not present in this ledger. The candidate set it was judged "
                   "against cannot be verified from this evidence."),
}


def unavailable_note(view: dict) -> str:
    """The note for a view that could not be hydrated, or '' when it was fine."""
    return UNAVAILABLE_NOTE.get((view.get("_hydration") or {}).get("state"), "")


def lane_counts(view: dict) -> dict[str, int]:
    """Candidates per lane for a decision, hydrated or not.

    Lane accounting must survive compaction: `candidate_refs` carries lane precisely so
    the report's "architecture considered" figures keep working when the prose does not
    resolve. Reading only `candidates` made those counters silently report zero.
    """
    source = (view.get("candidates")
              or view.get("candidate_refs") or [])
    out: dict[str, int] = {}
    for c in source:
        if isinstance(c, dict):
            lane = c.get("lane") or "unspecified"
            out[lane] = out.get(lane, 0) + 1
    return out
