# Cleanup audit — how this repository was derived from the original

The original lives at `../Track2` and was treated as **read-only**: every file here was
produced by reading it and writing a copy, never by moving, renaming or editing anything in
it. A sha256 snapshot of 3,451 original files was taken before any work began and re-checked
afterwards; the result is in the final section.

Nothing was copied with `cp -R`. The build ran from an explicit allowlist, and the copier
refuses to finish if a secret or dataset path reaches the output.

## Method: dependency analysis, not filename guessing

Two closures were computed by parsing `import` statements transitively:

| entry points | reachable local modules |
|---|---|
| the code-first research path (`research_driver`, `research_report`, `candidate`, `candidate_runner`, `confirm_champion`, `make_research_submission`) | 35 |
| `make_deliverables` on top of that | +3 |
| the control-plane UI | +2 |
| the typed control arm (`official_driver`) | +1 |
| every test suite | +5 |

Import analysis alone is not sufficient, so **dynamic** dependencies were found separately by
searching for module-name strings and dynamic loaders. Four matter, and none of them appears
in any `import`:

| dynamically loaded | by | verdict |
|---|---|---|
| `strategy/candidate_runner.py` | `sb.run(["-m", "strategy.candidate_runner", ...])` in `agent/candidate.py` | **KEEP** |
| `strategy/patch_runner.py` | `sb.run(["-m", "strategy.patch_runner", ...])` in `agent/patch.py` | **KEEP** |
| `strategy/attack_probe.py` | `core/isolation.py` runs it as the hostile strategy in `probe()` | **KEEP** — it is the security measurement |
| `research_workspace*/{features,model,train}.py` | `importlib.import_module("train")` inside the sandbox | **KEEP** — the editable strategy itself |

`strategy/run_iteration.py` is also dynamically launched, but only by `agent/driver.py`, the
superseded driver.

## Classification of every top-level path

### KEEP_CURRENT — code the current entry points need

| path | note |
|---|---|
| `agent/` (21 files) | includes `official_driver.py` (typed control arm) and, as **non-entry dependencies**, `driver.py`, `actions.py`, `proposer.py`, `draft_task.py`, `profile_data.py`. The brief lists `driver.py` and `actions.py` as superseded *"as an entry point"* — they are not entry points here, and they are genuinely imported (`proposer.py` defaults to `agent.actions`), so removing them would break `official_driver` and three test suites. |
| `core/` (14 files + 2 new) | plus `core/report_md.py` and `core/seed_stats.py`, both added here — see below |
| `strategy/` (6 files) | `ranker.py` (baseline reference), `candidate_runner.py`, `patch_runner.py`, `attack_probe.py`, `mle_strategy.py`, `run_iteration.py`. The last two are kept only because `tests/test_isolation.py` — the suite that measures the sandbox — imports them. |
| `research_workspace/` | the `inherited` seed profile; required to reproduce the recorded runs |
| `tests/` (28 files) | every suite, plus two new ones |
| `requirements.txt`, `.gitignore`, `.env.example` | |
| `confirm_champion.py`, `make_research_submission.py`, `make_official_submission.py`, `make_deliverables.py` | `make_official_submission` is imported by `make_research_submission` for `write_submission` |

### GENERATED_EVIDENCE

| path | note |
|---|---|
| `deliverables/` (34 files) | the graded markdown package plus its ledger copies |
| `runs/research_report.html` | regenerable from the ledgers |

### REQUIRED_PROVENANCE

| path | note |
|---|---|
| `submission_compliant/` (8 files) | **the frozen entry.** Byte-identical to the original; hashes in the final section |
| `runs/ledgers/research_trials.jsonl` | the submitted run (`trial_4`), the four trials, and all confirmations |
| `runs/ledgers/research_official.jsonl` | the compliant-protocol run that converged with no improvement |
| `runs/ledgers/research_run1.jsonl` | the two extended runs, retained as non-entry supporting evidence |
| `runs/artifacts/` (1,030 files, 5.1 MB) | the whole content-addressed store. It is small, and the report re-hashes every tree the ledgers name, so shipping all of it keeps that verification green |
| `runs/candidates/trial_4/*.json`, `runs/candidates/research_official/state.json` | run state and the confirmation records; the per-candidate fit outputs are not shipped |
| `experiments/isolation_probe.json` | **data**, read by `build_isolation()` to generate `ISOLATION.md` |

### OMIT_SUPERSEDED

| path | why |
|---|---|
| `submission/` | generated for the **previous** task definition: NDCG@10 / Recall@50, composite ≈0.157, an ALS+KNN action space, every rationale marked "SYNTHETIC — uniform random sampling". Four of its six files mention the current task zero times |
| `make_submission.py` | the packager for that task. `pair()` and `composite_index()` compute `0.5*(ndcg@10 + recall@50)`, so it raises `KeyError: 'ndcg@10'` on any current-task ledger — **it cannot run at all**. Its six task-agnostic helpers were lifted verbatim into `core/report_md.py`, which is why `make_deliverables.py` no longer imports it |
| `experiments/rationale_quality.py` | only `segments()` was needed; also lifted into `core/report_md.py` |
| `agent/report.py` | superseded reporter, unreachable from any current entry point |
| `core/watchdog.py` | unreachable |
| `docs/architecture.html` | describes the typed arm only |
| `src/`, `baseline_results.json` | superseded baseline scripts and their output |
| `BASELINE_REPRODUCTION.md` | 23 mentions of CWM; a superseded-task document. The current baseline-reproduction claim is proved instead by a test (`test_frozen_and_profiles.py`) and re-measured on every run (the seed candidate is scored at iteration 0) |
| `PLAN.md`, `SPEC_ASSUMPTIONS.md`, `AUTONOMOUS_AGENT_REBUILD_PLAN.md`, `CROSS_REVIEW_PROMPT.md`, `STATE.md` | working log and planning archaeology. `STATE.md` is 1,500+ lines and contains a section explicitly marked SUPERSEDED; publishing it would mislead a reader looking for current state |

### OMIT_EXPLORATORY

`experiments/*.py` (16 scripts: `field_headroom`, `hourmin_paired`, `measure_noise_floor`,
`official_sweep`, `ui_integration`, …). Real measurements, none reachable from a current
entry point, and their conclusions already live in the deliverables. Only the probe **JSON**
is retained, because a generator reads it.

### OMIT — regenerable run scratch

`runs/candidates/**` fit outputs (1.0 GB), `runs/iso/` (109 MB), `runs/official/` (59 MB),
`runs/t/` (29 MB), `runs/strategy_data/` (21 MB), `runs/*.npz`, and 30 superseded-task
ledgers. All reproducible; none is evidence for the current entry.

### SECRET_OR_DATA — never copied

`.env`, `secrets/` (the scorer signing key), `KuaiRand-Pure/` and `KuaiRand-Pure.tar.gz`
(CC BY-SA 4.0 — cite, do not redistribute), `__pycache__/`, `.DS_Store`, `.git/`, unix
sockets. The signing key is *read* at verification time from the original path and is never
copied; `KuaiRand-Pure` here is a **symlink**, gitignored, for local test runs only.

### UNKNOWN — resolved

Three paths were initially unclassifiable and were resolved by inspection rather than
assumption:

| path | resolution |
|---|---|
| `agent/actions.py` | **KEEP.** `agent/proposer.py` imports it as its default action space, so it is a live dependency even though it is not an entry point |
| `strategy/mle_strategy.py`, `strategy/run_iteration.py` | **KEEP.** `tests/test_isolation.py` imports both; that suite measures the kernel sandbox and is the last thing to drop |
| `experiments/isolation_probe.json` | **KEEP as data.** `build_isolation()` reads it. It was also stale in the original until the probe was made to write it by default |

## New files, not present in the original

| file | why |
|---|---|
| `core/report_md.py` | the six task-agnostic helpers extracted from `make_submission.py` plus `segments()`, so the superseded packager could be omitted without breaking `make_deliverables.py` |
| `core/seed_stats.py` | derives the 5-seed, expanded-20-seed and newly-observed-5–19 statistics **from the signed ledger**, with a cross-check against the raw signed score entries. The brief requires these not be handwritten constants |
| `research_workspace_minimal/` | the `minimal` seed profile: pointwise-only, no pairwise or listwise implementation |
| `tests/test_tournament.py`, `tests/test_frozen_and_profiles.py` | the new behaviour and the freeze |
| `README.md`, `RUN.md`, `ARCHITECTURE.md`, `CLEANUP_AUDIT.md` | documentation |

## Verification

| gate | result |
|---|---|
| original project content unchanged | **yes** — 0 of 3,451 files differ except `.DS_Store` (macOS Finder metadata, gitignored, written by the OS, not by any command run here). 0 changes across 1,368 `.py`, 73 ledgers, 36 submission files, 1,030 artifacts, 27 markdown files |
| frozen submission byte-identical | **yes** — 8 of 8 files |
| champion tree hash | `fe985741a836…` re-derived from the shipped `champion_source/` and re-verified in the artifact store |
| shipped ledgers verify | 3 of 3 (`research_trials` 117, `research_run1` 415, `research_official` 14 entries) |
| secret scan | clean |
