# RUN.md — exact commands

No command here contains a secret. Two things must be supplied from your environment:

| | how |
|---|---|
| the scorer's signing key | `secrets/scorer.key` (mode 600, never committed). Generate with `python3 -m core.scorer keygen` |
| the LLM key | `TRACK2_LLM_API_KEY`, a **project-scoped** key with a console spend limit. Put it in `.env` (gitignored); see `.env.example` |

## 1. Install

```bash
python3 -m pip install -r requirements.txt      # numpy + pandas only
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

## 8. Seed profiles

```bash
--seed-profile inherited   # research_workspace/  — pairwise switch present
--seed-profile minimal     # research_workspace_minimal/ — pointwise only
```

Both are recorded in `run_start` with `seed_provenance`, which states by *parsed function
name* whether pairwise, listwise or an alternative architecture was already present. That
field is how an autonomy claim gets checked rather than believed.

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
  --iterations 50 --convergence official --run-name tournament_routed
```

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
