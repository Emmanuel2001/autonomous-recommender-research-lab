# Architecture

## The trust boundary is the point

Three processes, and the separation is what makes any number here evidence rather than a
claim.

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  agent/research_driver.py            the loop. Holds NO signing key.     │
  │  TaskSpec fixes data / split / label / metric / stopping rule.           │
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
         │                   │  · immutable source mounted/read   │
         │                   │    outside its writable output     │
         │                   │  · CPU / MPS / CUDA plan recorded  │
         │                   │  7/7 attacks blocked, measured     │
         │                   │  with a control arm that succeeds  │
         │                   └────────────────────────────────────┘
         ▼
   research_workspace{,_minimal,_accelerated}/  the agent-editable strategy:
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
| `TaskSpec` owns data/split/metric/stopping | a new dataset can change the problem contract without allowing the LLM to reward-hack it |
| `ExperimentSpec` resolves to measured CPU/MPS/CUDA | GPU use and fallback are evidence in the run, not prose or a machine-name assumption |

## Runtime portability

The official frozen candidate remains NumPy/CPU. New experiments may use the accelerated
PyTorch profile:

```
candidate config/source ──▶ ExperimentSpec ──▶ capability probe
                                                   │
                         strict refusal ◀──────────┼── unavailable required device
                                                   │
                                                   ▼
                                  CPU | Apple MPS | NVIDIA CUDA
                                                   │
                                                   ▼
                              exact device + fallback in state and ledger
```

Native subprocess/seatbelt execution uses the host PyTorch runtime. Docker receives
read-only candidate/workspace/task mounts; CUDA uses an operator-supplied CUDA image and
`--gpus all`. MPS is refused under Docker rather than silently becoming CPU.

This section describes the resolution *mechanism*, which is not a claim that every branch has
run. CPU and Apple MPS have actually executed here; **CUDA and Docker have not** — only their
argv, mounts and refusal paths are asserted. README §10A carries the per-device table.

## Task portability

`agent/draft_task.py` profiles CSV files and proposes a setup. A human confirms the immutable
problem definition once. `core/tabular_task.py` then supplies both views of the same rows:

* strategy loader: train/dev labels present, held-out label absent;
* scorer loader: held-out labels present in the separate signing process;
* sandbox root: held-out outcomes physically redacted; under seatbelt the original root is
  kernel-denied, and under Docker it is not mounted.

The included `shop_clicks_demo` exercises different user/item/label column names through the
same NumPy candidate, PyTorch candidate and signed scorer. The supported family is explicit;
adding a metric or data modality requires another reviewed adapter.

## Research Console

`agent/control.py` is a loopback-only operator console. It can launch one allowlisted run,
display measured hardware and replay the signed research sequence. It separates:

* LLM-authored hypotheses, judge verdicts, reflections and optional public thought summaries;
* harness-derived diffs, execution telemetry and failures;
* scorer-signed metrics, rendered with a separate evidence badge.

It does not hold the signing key and does not expose hidden chain-of-thought.

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
| `core/execution.py` | capability detection plus strict/fallback CPU/MPS/CUDA resolution |
| `core/task.py`, `core/tabular_task.py` | registered and portable task contracts/loaders |
| `core/task_scorer.py` | task-owned positional scoring in the separate signing process |
| `agent/control.py` | live decision trace, replay, launch controls and hardware status |

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
