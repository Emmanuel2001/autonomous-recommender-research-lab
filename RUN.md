# RUN.md — exact commands

No command here contains a secret. Two things must be supplied from your environment:

| | how |
|---|---|
| the scorer's signing key | `secrets/scorer.key` (mode 600, never committed). Generate with `python3 -m core.scorer keygen` |
| the LLM key | `TRACK2_LLM_API_KEY`, a **project-scoped** key with a console spend limit. Put it in `.env` (gitignored); see `.env.example` |

## 1. Install

```bash
python3 -m pip install -r requirements.txt      # numpy + pandas only

# Optional research runtime for PyTorch candidates on CPU / Apple MPS
python3 -m pip install -r requirements-accelerated.txt

# On a CUDA worker, install the PyTorch build matched to that worker's CUDA runtime.
# Do not assume the generic wheel has CUDA merely because the machine has a GPU.
```

The dataset is **not** in this repository (CC BY-SA 4.0 — cite, do not redistribute). Place
KuaiRand-Pure so that `KuaiRand-Pure/data/` resolves from the repository root; a symlink is
fine and is gitignored.

## 2. Run the tests

```bash
bash tests/run_all.sh                    # every suite
python3 tests/test_tournament.py         # the hypothesis tournament, fake backend, no credits
python3 tests/test_frozen_and_profiles.py # the frozen entry, seed profiles, claim scan
```

## 3. Start the protected scorer

It holds the labels **and** the signing key, in its own process. Start it in its own terminal:
the driver must never be its parent, or the key is inherited and the trust model is void.

```bash
TRACK2_SIGNING_KEY="$(cat secrets/scorer.key)" \
TRACK2_LEDGER="runs/ledgers/my_run.jsonl" \
TRACK2_SOCKET="runs/my_run.sock" \
python3 -u -m core.scorer serve --task official
```

## 4. Verify the frozen entry

```bash
# the champion source re-hashes to the tree the ledger names
python3 tests/test_frozen_and_profiles.py

# the signed chains
TRACK2_SIGNING_KEY="$(cat secrets/scorer.key)" \
  python3 -m core.scorer verify --ledger runs/ledgers/research_trials.jsonl

# the three seed analyses, derived from the ledger rather than quoted
python3 -m core.seed_stats \
  --ledger runs/ledgers/research_trials.jsonl \
  --champion fe985741a8368e082be4c71e0d9e3be06ca333f2aac1b26216203cbf2283c3b5 \
  --baseline e1d994dedd110658b8c1c130486fb980b98df84e6fabafcad664b94d38ed693c
```

## 5. Reproduce the baseline

```bash
# the minimal seed profile against strategy/ranker.py at Config() defaults
python3 - <<'PY'
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'research_workspace_minimal')
import numpy as np
from core.official_data import load_for_strategy
from core.task import get
from strategy.ranker import Config, run as ranker
from contract import Data, Spec
import train as T
sp, ed = load_for_strategy(); spec = get("kuairand_pure")
d = Data(train=sp["train"], dev=sp["valid"], heldout=sp["test"],
         aux=dict(sp["train"].aux or {}), edges=ed,
         spec=Spec("kuairand_pure","valid","test",spec.label))
r = T.fit_predict(d, {}); s, _ = ranker(sp, ed, Config())
print("byte-identical:", np.array_equal(np.asarray(r["dev"]), np.asarray(s["valid"])))
PY
```

## 6. Legacy policy — one hypothesis per iteration

Reproduces the recorded runs.

```bash
set -a; . ./.env; set +a
TRACK2_SOCKET="runs/my_run.sock" TRACK2_LEDGER="runs/ledgers/my_run.jsonl" \
python3 -u -m agent.research_driver \
  --research-policy single --seed-profile inherited \
  --backend gemini --model gemini-3.5-flash-lite \
  --iterations 50 --convergence official --run-name legacy_single
```

## 7. The new tournament policy

```bash
set -a; . ./.env; set +a
TRACK2_SOCKET="runs/my_run.sock" TRACK2_LEDGER="runs/ledgers/my_run.jsonl" \
python3 -u -m agent.research_driver \
  --research-policy tournament --tournament-size 6 \
  --seed-profile minimal \
  --backend gemini --model gemini-3.5-flash-lite \
  --iterations 50 --convergence official --run-name tournament_minimal
```

### 7A. Architecture discovery (diagnostic, never the official submission run)

Discovery is opt-in. It removes the incumbent-FM preference from judging, tells the planner
which approved libraries actually exist, rejects config-only changes masquerading as model
architecture, and requires one genuine `model.py` architecture selection by the third
decision. The first two selections remain the judge's choice. A failed candidate falls back
to the unchanged champion on the next iteration and remains in the signed evidence.

Use a **separate scorer process, socket and ledger** from an official run. Start that scorer
as in section 3, with `runs/ledgers/architecture_discovery.jsonl` and
`runs/architecture_discovery.sock`, then run:

```bash
set -a; . ./.env; set +a
TRACK2_SOCKET="runs/architecture_discovery.sock" \
python3 -u -m agent.research_driver \
  --search-mode discovery \
  --research-policy tournament --tournament-size 8 \
  --seed-profile minimal \
  --backend gemini \
  --model gemini-3.5-flash-lite \
  --planner-model gemini-3.5-flash \
  --judge-model gemini-3.5-flash \
  --coder-model gemini-3.5-flash-lite \
  --reflector-model gemini-3.5-flash-lite \
  --framework auto --device auto --accelerator-policy prefer \
  --iterations 12 --convergence off \
  --budget-mode fixed --max-tokens 2500000 \
  --llm-max-output-tokens 128000 \
  --run-name architecture_discovery1
```

The two token settings control different things:

- `--max-tokens` is the total research-run budget across planning, judging, coding and
  reflection calls. Before each call, the harness uses the prompt's UTF-8 byte length as a
  conservative input bound, applies a role-specific output cap and reserves minimum output
  capacity for the roles still needed in that iteration. A known-insufficient call is
  refused before contacting the provider. Use the provider console's spend limit as the
  independent hard monetary guard.
- `--llm-max-output-tokens` is the maximum output for one role call. It is 64,000 by
  default and may be raised to 128,000 for Opus 5 architecture rewrites. Thinking tokens
  count inside this per-call allowance. Long Anthropic responses use SDK streaming and
  are assembled before the JSON is validated.

Budget policy is explicit:

- `--budget-mode fixed --max-tokens N` treats `N` as the hard local run allowance.
- `--budget-mode maximize --max-tokens N --convergence off` is a diagnostic policy that
  keeps researching until an iteration, time or token limit ends the run. It does not mean
  that one call receives all remaining tokens.
- `--budget-mode open` has no local token ceiling, but still obeys iteration and wall-clock
  limits. This is intentionally incompatible with `--max-tokens`; always keep a provider
  spend limit when using it.

If a code response still reaches its per-call ceiling, the driver records and charges the
failed call, then uses the normal bounded repair allowance to retry at medium effort with a
concise, changed-files-only instruction. It does not silently train partial source code.

Static candidate refusals use the same bounded repair loop. In particular, a grouped ranker
that references `heldout.label` is never executed: the validator records the exact file and
line, and the coder must rebuild held-out features and group sizes without labels before the
candidate can be registered or trained. Keep `--max-repairs` above zero for discovery runs;
the Research Console discovery preset uses three.

This command can consider LightGBM, XGBoost, torch neural interaction models, multitask
models and ensembles because those dependencies are visible to the planner and enforced by
the harness allowlist. They are examples, not a hard-coded menu. The ledger marks this run
`diagnostic_discovery` and `submittable: false`. Any winner still needs a new standard run
with `--convergence official` and multi-seed confirmation before it becomes a submission
candidate. No discovery run is launched by setup or by selecting the console preset.

### 7B. Data contract seen by the researcher

Before the first LLM call, the driver measures the task loader and records a versioned data
contract. It contains loader-visible column names, dtypes, roles, split row counts and
boundaries, label visibility, metric direction, personalization requirements and declared
excluded sources. It contains no raw feature values and no held-out labels.

For KuaiRand-Pure, the current loader exposes seven fields: `user_id`, `video_id`,
`author_id`, `tab`, `duration_ms`, `date` and `hourmin`. This is narrower than the raw
dataset directory. `user_features_pure.csv` and additional basic-video fields are present
but are not joined. `video_features_statistic_pure.csv` contains full-window aggregates, so
a naive join can leak held-out-period information. The planner sees both the loaded surface
and these exclusions. It may propose a new leakage-safe loader in future work, but it may
not invent a raw column and silently train against it.

Every tournament hypothesis declares `data_columns`, computed `derived_features` and its
personalization pathway. Unknown loader columns are refused before coding. After coding,
the harness checks that every declared file actually changed and that a
`model_architecture` experiment really changed `model.py`.

### 7C. Cross-run memory and belief evidence

Runs start with `--memory fresh` by default. Carrying conclusions is always explicit:

```bash
# Import compatible beliefs and reflections, but start a separate search lineage.
python3 -u -m agent.research_driver \
  --memory import \
  --memory-from runs/candidates/<prior_run>/state.json \
  <the rest of the new run arguments>

# Continue the exact prior lineage and dedup history from its champion source.
python3 -u -m agent.research_driver \
  --memory continue \
  --memory-from runs/candidates/<prior_run>/state.json \
  --seed-tree <the prior state's full champion tree hash> \
  <the rest of the continuation arguments>
```

Memory is refused unless task, metric, label, split identity, dataset fingerprint, data
contract hash and workspace contract hash match. Legacy beliefs with no contract scope are
skipped, not silently trusted. Imported beliefs keep their source-run provenance and remain
revisable by new results. Historical ledgers and state files are not rewritten.

`llm_confidence` is the reflector's subjective and uncalibrated opinion. It is not an
accuracy percentage. `evidence_grade` is separate and categorical:
`unreplicated`, `single_signed_result`, `replicated`, `contradicted` or `seed_confirmed`.
The harness computes that grade only from recorded supporting or contradicting experiments
and seed confirmation evidence, never from the LLM's confidence number.

## 8. Seed profiles

```bash
--seed-profile inherited   # research_workspace/  — pairwise switch present
--seed-profile minimal     # research_workspace_minimal/ — pointwise only
--seed-profile accelerated # research_workspace_accelerated/ — PyTorch CPU/MPS/CUDA
```

Both are recorded in `run_start` with `seed_provenance`, which states by *parsed function
name* whether pairwise, listwise or an alternative architecture was already present. That
field is how an autonomy claim gets checked rather than believed.

## 8A. Run accelerator-aware research

First inspect what this exact Python process can use:

```bash
python3 -m core.execution
```

Prefer MPS/CUDA but record a CPU fallback:

```bash
python3 -u -m agent.research_driver \
  --research-policy tournament --seed-profile accelerated \
  --framework torch --device auto --accelerator-policy prefer \
  --backend gemini --model gemini-3.5-flash-lite \
  --iterations 6 --convergence official --run-name accelerated_auto
```

Require CUDA and refuse to pretend a fallback is equivalent:

```bash
python3 -u -m agent.research_driver \
  --research-policy tournament --seed-profile accelerated \
  --framework torch --device cuda --accelerator-policy require --no-device-fallback \
  --isolation subprocess \
  --backend gemini --model gemini-3.5-flash-lite \
  --iterations 6 --convergence official --run-name cuda_required
```

For Docker CUDA isolation, provide a CUDA-capable PyTorch image and set
`TRACK2_DOCKER_CUDA_IMAGE`; the harness adds `--gpus all`. Apple MPS is a host runtime and
is deliberately refused under this Docker backend—use seatbelt/subprocess on the Mac.

## 9. Role-specific models

Each role falls back to `--model`, then to the backend default.

```bash
python3 -u -m agent.research_driver \
  --research-policy tournament --seed-profile minimal \
  --backend gemini \
  --model         gemini-3.5-flash-lite \
  --planner-model gemini-3.5-flash \
  --judge-model   gemini-3.5-flash \
  --coder-model   gemini-3.5-flash-lite \
  --reflector-model gemini-3.5-flash-lite \
  --thought-summaries \
  --iterations 50 --convergence official --run-name tournament_routed
```

`--thought-summaries` requests the provider's public summary when supported. The ledger and
console label it as a provider summary, never private chain-of-thought. The hypothesis,
judge ranking, code diff, signed metric and reflection remain visible for every backend.

## 9A. Use the Research Console

Start the scorer first, then in another terminal:

```bash
TRACK2_SOCKET="runs/my_run.sock" python3 -m agent.control --port 8787
```

Open the tokenized loopback URL printed in the terminal. The console can select the seed
profile, task spec, role-specific LLM models, framework/device policy and budgets. It shows
measured CPU/MPS/CUDA availability, live output, an auditable decision timeline, and a replay
of a completed signed ledger. The **Architecture discovery** preset fills the diagnostic
settings above but does not launch until you press Launch; use the strongest available model
for planner/judge and a cheaper model for coder/reflector. The console labels discovery runs
as non-official evidence. It holds no signing key and can run only one driver at a time.

## 9B. Confirm and run a second recommender task

The portable family currently supports CSV logged-impression binary ranking or global AUC
with an integer time split. Drafting measures the files, proposes roles and validates them;
`--confirm` writes a runnable JSON contract:

```bash
python3 -m agent.draft_task /path/to/data \
  --goal "rank products within each account by conversion" \
  --proposer gemini --model gemini-3.5-flash-lite \
  --epsilon 0.002 --window 3 --confirm --out tasks/shop.json
```

Start a task-specific scorer in its own terminal:

```bash
TRACK2_SIGNING_KEY="$(cat secrets/scorer.key)" \
TRACK2_LEDGER="runs/ledgers/shop.jsonl" TRACK2_SOCKET="runs/shop.sock" \
python3 -u -m core.scorer serve --task-spec tasks/shop.json
```

Then point the same research loop at it:

```bash
TRACK2_SOCKET="runs/shop.sock" python3 -u -m agent.research_driver \
  --task-spec tasks/shop.json --research-policy tournament \
  --seed-profile accelerated --framework torch --device auto \
  --backend gemini --model gemini-3.5-flash-lite \
  --iterations 20 --convergence official --run-name shop_research
```

The included `tests/fixtures/portable_recommender/task.json` is the executable second-task
proof: NumPy training, PyTorch training, physical held-out-label redaction and signed scoring
are all exercised. A new metric family or non-CSV modality still requires a new audited
loader/metric adapter; the platform does not claim arbitrary data support.

## 10. Generate the HTML report

```bash
python3 -m agent.research_report --ledger runs/ledgers/my_run.jsonl \
  --out runs/my_report.html
open runs/my_report.html
```

## 11. Confirm a champion on GENUINELY UNSEEN seeds

The exact seed list is recorded in the ledger **before any fit**, so seeds cannot be chosen
after seeing which were favourable.

Seeds **0–19 are already measured** for the shipped champion, and **seed 0 was used during
champion selection**. Re-running `--seeds 20` therefore *expands* the existing analysis; it
does not replicate it independently. For genuinely unseen seeds, start past what has been
measured:

```bash
# seeds 20-29: none has been measured, none played any part in selection
TRACK2_SOCKET="runs/my_run.sock" TRACK2_LEDGER="runs/ledgers/my_run.jsonl" \
python3 -u confirm_champion.py \
  --state runs/candidates/trial_4/state.json \
  --seed-start 20 --seeds 10 \
  --baseline-tree e1d994dedd110658b8c1c130486fb980b98df84e6fabafcad664b94d38ed693c \
  --out runs/candidates/trial_4/confirmation_unseen_20_29.json
```

Or name them explicitly:

```bash
  --seed-list 20,21,22,23,24,25,26,27,28,29
```

`--seed-list` refuses a repeated seed, because measuring the same seed twice is not two
measurements. Both forms record `seed_selection`, `seed_start` and the resolved list in the
protocol entry, so a reader can tell an expansion from a replication without taking anyone's
word for it.

To reproduce the three analyses already published (original 5, expanded 20, and the
newly-observed 5–19 subset) **without refitting**, read them from the ledger instead:

```bash
python3 -m core.seed_stats \
  --ledger runs/ledgers/research_trials.jsonl \
  --champion fe985741a8368e082be4c71e0d9e3be06ca333f2aac1b26216203cbf2283c3b5 \
  --baseline e1d994dedd110658b8c1c130486fb980b98df84e6fabafcad664b94d38ed693c
```

## 12. Package a successful run

Refuses unless the champion beats the baseline, and binds the submission to the score that
**promoted** it rather than to a refit.

```bash
python3 -u make_research_submission.py \
  --state  runs/candidates/<run_name>/state.json \
  --ledger runs/ledgers/my_run.jsonl \
  --outdir submission_new

TRACK2_SIGNING_KEY="$(cat secrets/scorer.key)" python3 make_deliverables.py
```

`submission_compliant/` is the frozen entry and must not be overwritten — a test asserts its
champion tree still re-hashes correctly. Package into a **new** directory.
