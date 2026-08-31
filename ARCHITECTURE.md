# Architecture

## The trust boundary is the point

Three processes, and the separation is what makes any number here evidence rather than a
claim.

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  agent/research_driver.py            the loop. Holds NO signing key.     │
  │                                                                          │
  │  tournament ──▶ judge ──▶ code ──▶ diff ──▶ REGISTER ──▶ fit ──▶ score   │
  │      │            │         │        │          │          │       │     │
  │  agent/research_agent.py (LLM: plan · judge · code · reflect)      │     │
  │      │                              │          │          │       │     │
  │      │                    core/artifacts.py    │          │       │     │
  │      │                    content-addressed    │          │       │     │
  │      │                    trees; the harness   │          │       │     │
  │      │                    computes the diff    │          │       │     │
  └──────┼──────────────────────────────┼──────────┼──────────┼───────┼─────┘
         │                              │          │          │       │
         │                              │          │          │       ▼
         │                              │          │          │  ┌──────────────┐
         │                              │          │          │  │ core/scorer  │
         │                              │          │          │  │ SEPARATE     │
         │                              │          │          │  │ PROCESS.     │
         │                              │          │          │  │ Holds the    │
         │                              │          │          │  │ labels AND   │
         │                              │          │          │  │ the signing  │
         │                              │          │          │  │ key. The     │
         │                              │          │          │  │ only thing   │
         │                              │          │          │  │ that makes   │
         │                              │          │          │  │ a number.    │
         │                              │          │          │  └──────────────┘
         │                              │          ▼                    │
         │                              │   signed hash-chained ledger ◀─┘
         │                              │   nothing runs before it is recorded
         │                              ▼
         │                   ┌────────────────────────────────────┐
         │                   │ strategy/candidate_runner.py       │
         │                   │ inside core/isolation.py (seatbelt)│
         │                   │  · no network                      │
         │                   │  · cannot read held-out labels     │
         │                   │  · cannot read the ledger or key   │
         │                   │  · writes only its own directory   │
         │                   │  7/7 attacks blocked, measured     │
         │                   │  with a control arm that succeeds  │
         │                   └────────────────────────────────────┘
         ▼
   research_workspace{,_minimal}/  the agent-editable strategy:
     contract.py   IMMUTABLE — the interface, the metric helper, the train subsample
     features.py   agent-editable
     model.py      agent-editable   ← architecture lives here
     train.py      agent-editable   ← objective and sampling live here
     config.json   agent-editable
```

## Why each boundary exists

| boundary | what it prevents |
|---|---|
| the scorer is a **separate process** holding the key | the driver cannot mint a score. A claimed metric with no matching signed entry is not a metric |
| **register before execute** | a hypothesis cannot be written after its result is known. A refused record abandons the candidate unrun |
| the **harness computes the diff** from parent and child trees | a model-authored diff could describe a change it did not make, and the run log is what a reader trusts |
| **content-addressed** trees | two identical candidates are one artifact, lineage is a fact about the data, and `verify()` re-hashes stored bytes so a ledger can carry a hash instead of source |
| the **kernel sandbox** | generated code runs against a dataset we are scored on. This is the precondition, not insurance |
| held-out labels **absent from the process**, asserted on every load | the one invariant that cannot be relaxed without invalidating the submission |

## Where the research decisions live

| file | responsibility |
|---|---|
| `agent/research_agent.py` | the six research lanes, the tournament and judge schemas, prompts, per-role model routing, per-call cost accounting |
| `agent/research_driver.py` | the loop, the policy branch, pre-execution recording, repair bounds, promotion on the signed number only |
| `agent/research_state.py` | champion, experiment history, **structured beliefs** keyed by mechanism, lane statistics, prediction calibration |
| `core/artifacts.py` | content addressing, harness-computed diffs, `verify()` |
| `core/budgets.py` | the 50-iteration and 6-hour caps, per-stage wall-clock |
| `core/seed_stats.py` | the seed analyses, derived from the ledger with a cross-check against raw signed scores |
| `core/report_md.py` | task-agnostic markdown helpers (see `CLEANUP_AUDIT.md` for their provenance) |
| `agent/research_report.py` | the HTML report, including *capability versus demonstrated behaviour* |

## The typed control arm

`agent/official_driver.py` remains. It chooses from a fixed action space rather than writing
code, and it produced the reproduction baseline. It is kept as a **control arm**: the
comparison between choosing from a menu and writing the source is itself a result. It is not
the entry point. See `CLEANUP_AUDIT.md` for why its dependencies (`agent/actions.py`,
`agent/proposer.py`) are still shipped.

## What the agent cannot touch

The metric, the splits, the convergence rule, the scorer, the ledger, `contract.py`, and the
harness itself. It edits four files inside a sandbox and is judged by a process it cannot
reach. Everything it *can* do is enumerated in `research_workspace*/contract.py`, and
`agent/candidate.py` refuses 14 distinct classes of tree before any compute is spent.
