# Autonomous ML Research Agent — TikTok TechJam 2026, Track 2

An LLM agent that **writes the strategy code** for a recommender-system experiment, runs it
behind a kernel sandbox, and is scored by a separate signing process it cannot reach.

**Benchmark** KuaiRand-Pure · **task** within-user ranking of each user's own logged
impressions (*not* full-catalogue retrieval) · **label** `long_view` ·
**metric** `primary = mean(GAUC, nDCG@5)`.

## 1. The task

Each user was shown a handful of videos — about 5.6 in this data — and the label says whether
they watched one to completion. The job is to **order each user's own impressions**. Nothing
is retrieved and nothing is recalled from a catalogue: the candidates are given, and only the
ordering within a user is scored. That places this at the **ranking stage** of an industrial
recommender funnel, which is also where the organisers placed it.

## 2. Two different programs, and it matters which is which

| | what it is | who wrote it |
|---|---|---|
| **the submitted recommender** | a NumPy Factorization Machine: embeddings, Adam, a loss, early stopping. It is intentionally CPU-reproducible. | four files are agent-editable; **the submitted champion changed two of them** — see §8 |
| **future candidates** | the same contract can also run agent-written PyTorch models on measured CPU, Apple MPS or NVIDIA CUDA devices. | agent-editable; this capability does not retroactively change the frozen result |
| **the researcher** | the LLM. It never scores anything and never sees held-out labels. It reads results, forms hypotheses, and writes the recommender's source. | us, as a harness |

The LLM is not the model. It is the thing that decides what the model should be.

## 3. The loop

```
plan / tournament → judge → code → diff → REGISTER → train → score → reflect → beliefs → promote
```

* **tournament** — several competing hypotheses, spanning six research lanes, of which at
  least one must be a `model_architecture` candidate and one a `sampling_training` candidate.
* **judge** — ranks them on evidence, expected impact, novelty, information value, CPU
  resource feasibility and implementation risk, picks **one**, and must state in writing why it passed
  over each architecture candidate.
* **register** — the tournament, the judge's reasoning and the chosen hypothesis go into a
  signed hash-chained ledger **before any candidate code exists or runs**. If that record is
  refused, the candidate is abandoned unrun.
* **score** — a separate process holding the labels and the signing key produces the only
  number that decides anything.
* **beliefs** — the reflector updates structured beliefs keyed by *mechanism*, and the next
  tournament reasons over those instead of re-reading prose.

Only the selected hypothesis consumes an evaluation. The others are recorded as
**considered**, and the report never presents them as experiments.

## 4. The result: `trial_4`

A compliant run — the organisers' convergence rule left **on**, started from the
**unmodified** official baseline, no human configuration supplied. It ran 5 iterations,
promoted twice, and converged on the organisers' own rule.

The agent set `loss: pairwise` with `neg_per_pos: 2` and added a `user_hist_len` feature.

## 5–7. The evidence, and exactly what kind it is

| analysis | seeds | n | paired mean | sd | se | t | wins |
|---|---|---|---|---|---|---|---|
| original 5-seed (pre-registered) | 0-4 | 5 | **+0.001591** | 0.000634 | 0.000284 | 5.61 | 5/5 |
| expanded 20-seed | 0-19 | 20 | **+0.001888** | 0.000734 | 0.000164 | 11.5 | 20/20 |
| newly observed 5-19 | 5-19 | 15 | **+0.001987** | 0.000758 | 0.000196 | 10.15 | 15/15 |

Every figure is derived from the signed ledger by `core/seed_stats.py`, and each per-seed
value is cross-checked against a raw signed score entry (40 of 40 matched). None is a
constant typed into a document.

**The disclosure that matters.** Seeds 0–4 were measured **first**, as a pre-registered
5-seed confirmation. The analysis was **later expanded** to 0–19. Seed 0 is also the seed the
champion was selected on during the run. So the 0–19 figure is an **expanded paired
robustness analysis — not an independent pre-registered 20-seed replication**, and this
repository never calls it one.

Because that expansion could have flattered the result, seeds **5–19** are reported
separately: they played no part in selection, and on them the effect is **+0.001987 with
15 of 15 wins** — slightly larger than on the original five, not smaller.

**On the size of it.** +0.001888 is 2.4× the organisers' published seed std (0.0008) and
0.94× their epsilon (0.002). It is also small in absolute terms — roughly 0.7% of the
headroom between the FM baseline and the published oracle ceiling. Both are true: the
evidence is strong, the effect is modest.

## 8. Inherited versus agent-authored

| file | in the submitted champion | authorship |
|---|---|---|
| `config.json` | **changed** | the agent |
| `features.py` | **changed** | the agent |
| `model.py` | **inherited unchanged** | the seed profile |
| `train.py` | **inherited unchanged** | the seed profile |

The pairwise/BPR implementation **already existed** in the `train.py` the run started from,
and that file was never edited.

So: the agent **identified the training objective as the lever and activated it** — which no
other compliant run did. It **did not implement BPR**. That distinction is kept everywhere in
this repository, because the two are not the same achievement.

Across the wider collection of runs, `model.py` was edited **once** and `train.py`
**never** — so this repository does not claim `model.py` was never edited, only that the
submitted champion inherited it.

## 9. Why the champion kept a fixed FM

Because nothing better was demonstrated. The agent's improvement came from the objective and
one feature; no architecture change was ever evaluated under the old single-hypothesis
policy, so a fixed second-order FM remained the best measured option. That is a description
of what happened, not a claim that architecture does not matter.

## 10. The new capability

Three seed profiles now exist:

| profile | contains | use |
|---|---|---|
| `inherited` (`research_workspace/`) | the prepared pairwise switch | reproducing the recorded runs |
| `minimal` (`research_workspace_minimal/`) | **pointwise only** — no pairwise, no listwise, no alternative architecture | any ranking objective must be **written** by the agent |
| `accelerated` (`research_workspace_accelerated/`) | PyTorch FM with pointwise/pairwise training and declared device execution | CPU/MPS/CUDA architecture and scale experiments; **not** the official NumPy baseline |

`minimal` reproduces the official pointwise baseline **byte-identically** (verified: identical
dev and held-out score vectors against `strategy/ranker.py` at `Config()` defaults), so it is
a like-for-like starting point and not a weaker one.

Model routing is per role — `--planner-model`, `--judge-model`, `--coder-model`,
`--reflector-model`, each falling back to `--model` and then the backend default — so a
stronger model can carry the research decisions while a cheaper one writes routine code.

## 10A. The platform is no longer tied to this laptop or this dataset

**The submitted entry is, and remains, the NumPy CPU implementation.** Nothing below changes
it: no accelerator, no PyTorch and no portable task participates in the frozen submission,
which still reproduces exactly on CPU with NumPy alone. The platform around it grew; the
entry did not move.

* `core/execution.py` measures CPU, MPS and CUDA availability and resolves every candidate's
  `ExperimentSpec` to one actual device. A fallback is recorded; strict mode refuses it. A
  candidate may narrow an operator constraint in `config.json` but never widen one, because
  that file is written by the LLM.
* the accelerated PyTorch profile has completed a real fit end to end **on Apple MPS**: 11
  epochs, `dev_primary_best` 0.602171, on `torch 2.11.0` under macOS 15.7.3 (arm64). That is
  a portability check, not a result — it is a single seed and it is not compared against the
  champion.
* a confirmed portable JSON `TaskSpec` owns column names, CSV files, time splits, label,
  metric, convergence and baseline. The same candidate and separate signed scorer have run
  end to end on the included `shop_clicks_demo`, whose columns are `account_id`,
  `product_id`, `context` and `converted` rather than KuaiRand's schema. This proves
  **bounded CSV-recommender portability — not arbitrary AutoML**: the portable family is
  limited to CSV logged-impression data with a time-ordered split and either within-user
  ranking or global AUC. A task outside that shape needs new code, not just new JSON.
* the local Research Console exposes hardware, task spec, role models, tournament policy,
  budgets and accelerator policy. Its decision trace shows hypotheses, judge verdicts,
  pre-registration, signed results, reflections and optional provider-generated public
  thought summaries. It does **not** claim to reveal private chain-of-thought.

### Which devices were actually exercised, and which were not

Stated per device, because "supports CUDA" and "ran on CUDA" are different claims:

| device / backend | status on this machine |
|---|---|
| **CPU (NumPy)** | **executed** — the reference path, the frozen entry, and every suite |
| **CPU (PyTorch)** | **executed** — forward pass matches the NumPy FM to 2e-7 |
| **Apple MPS** | **executed** — real forward pass, an optimiser loop that lowers the loss, and a full 11-epoch candidate fit |
| **NVIDIA CUDA** | **never executed.** No CUDA hardware is present (`torch.cuda.is_available()` is false here). Only the *plan* is tested: `--gpus all` in the container argv, device forwarding, and the refusal raised when the CUDA image is absent |
| **Docker** | **never executed.** The daemon does not respond on this machine. Only argv and mount construction are asserted — read-only candidate/workspace/task mounts and the container output path |

These are platform capability tests, not new leaderboard evidence. **This engineering pass
established no new leaderboard improvement.** The submitted champion, prediction file and
signed historical ledgers are not rebuilt by it.

### Architecture discovery is now a separate, opt-in policy

The official path remains conservative and reproducible. A second search mode now exists for
the broader question the historical runs did not answer:

* `--search-mode discovery` is accepted only with the tournament policy, at least three
  iterations and `--convergence off`; its ledger is marked `diagnostic_discovery` and
  `submittable: false`;
* the planner sees the real approved dependency list (`lightgbm`, `xgboost`, `torch`,
  `implicit`, scikit-learn and the baseline numerical libraries) instead of merely being
  told that an unnamed allowlist exists;
* a discovery architecture candidate must name its model family and structural change and
  must edit `model.py`. Changing only embedding width or another config constant no longer
  counts as architecture;
* the first two discovery selections remain open. If neither funds architecture, the third
  judge decision must do so. This floor and any corrected judge verdict are recorded rather
  than silently overridden;
* the incumbent remains the safe fallback after a failed fit. The failure is retained as
  evidence, and any discovery winner must return to a separate standard run with official
  convergence and multi-seed confirmation.

The Research Console exposes this as an **Architecture discovery** preset: 12 evaluations,
eight hypotheses per tournament, convergence disabled and the minimal seed profile. Role
routing remains explicit so a stronger planner/judge can be paired with cheaper coder and
reflector calls. Selecting the preset only fills the form; it does not spend API credits.

This is implemented and tested policy, not a new result. **No real LLM has run discovery mode
yet, and no boosted-tree, DeepFM, DCN or ensemble candidate has been shown to beat the shipped
champion.** Those families are now reachable directions, not claims.

## 11. What real LLM runs have done, and what they have not

Three real runs now exist. Their ledgers are separate on purpose: a **diagnostic** run that
switches the stopping rule off is not evidence about the official protocol, and must never
share a ledger with a compliant one.

| run | ledger | profile | iterations | stopping rule | tokens |
|---|---|---|---|---|---|
| `arch_smoke1` | `runs/ledgers/arch_smoke1.jsonl` | `minimal` | 1 | capped at 1 | 30,069 |
| `breadth_diag1` | `runs/ledger.jsonl` | `minimal` | 6 | **convergence off — diagnostic** | 217,666 |
| `research_submission1` | `runs/ledgers/research_submission.jsonl` | `inherited` | 3 | **official** (ε=0.002, N=3) | 102,210 |
| `architecture_discovery2` | `runs/ledgers/architecture_discovery2.jsonl` | `minimal` | 7 | **discovery; convergence off — diagnostic** | 301,168 |

What they establish:

* **The tournament runs under the official protocol.** `research_submission1` converged on the
  real rule — *"the validation best improved by +0.001905 over 3 iterations, which does not
  exceed epsilon=0.002"* — with 0 failures, 0 repairs, 0 lane-coverage violations and 0 judge
  violations. Its champion is confirmed at **+0.001845, t=11.12, 20 of 20 paired seeds**.
* **The agent now edits training source, and a source edit was promoted.** Iteration 3 changed
  `config.json` *and* `train.py`, adding a cosine learning-rate decay it wrote itself
  (`m.lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * (ep - 1) / epochs))`). The superseded
  typed-action policy could never edit source at all. The gain attributable to that iteration
  is **+0.000029 and unconfirmed** — the edit is the autonomy evidence, not the improvement.
* **A stronger-tier discovery run selected and executed a real architecture.** In
  `architecture_discovery2`, Gemini 3.5 Flash planned and judged 32 recorded candidates; the
  judge funded a PyTorch DeepFM change spanning `model.py`, `train.py` and `config.json`.
  The candidate ran successfully but was rejected at **0.572274 (−0.029422)** while its
  train/dev gap widened from 0.0852 to 0.1498. The exact 12,561-byte source diff is retained.
* **Hardware resolution is not hardware attestation.** The harness resolved that Torch
  candidate to Apple MPS, but the generated source ignored `TRACK2_DEVICE` and created its
  model and tensors on CPU. The score is valid; an “MPS executed” claim for that candidate is
  not. The console now displays the resolved target separately from candidate-reported device
  evidence, and future coder prompts explicitly require device propagation.

What they do **not** establish, stated precisely because the distinction is easy to blur:

* **`research_submission1` is not an independent implementation of BPR.** It ran on the
  `inherited` profile, where `step_pairwise` already existed and the ledger records
  `pairwise_code_present: true`. At iteration 1 the agent observed the existing implementation
  and changed **only `config.json` (a 202-byte diff)**. That run is an *independent
  tournament-driven search route to essentially the same performance* — not an independent
  authorship of the objective.
* **The one genuine minimal-profile BPR implementation failed.** In `breadth_diag1` the agent
  wrote `step_bpr` plus per-user negative sampling across `model.py` and `train.py` from a
  pointwise-only start, and it was **rejected at −0.206874**: the gradient signs handed to the
  optimiser were inverted, so training performed ascent on the BPR objective. The training loss
  rose monotonically 0.738 → 11.165 over 30 epochs, and both splits fell below the 0.5 random
  floor. The tree and its source are retained.
* **The submitted prediction file does not come from either new run.** `submission_compliant/`
  is unchanged and still packages the earlier champion. A direct paired comparison of the two
  champions over the same 20 seeds gives **−0.0000433 (t = −0.31, existing wins 13/20)** — they
  are statistically indistinguishable, so the frozen champion stands.

So the repository still separates a **capability** from the **behaviour** observed, and the
`Research breadth: capability versus demonstrated behaviour` report section keeps them apart.

## 12. Honest limitations

* A stronger planner/judge has now driven one diagnostic run: Gemini 3.5 Flash for planning and
  judging, with Flash Lite for coding and reflection. It produced broader architecture choices,
  but **did not produce a better recommender**. Three of seven iterations ended at the provider
  layer (timeout, incomplete response, and the free-tier 20-request quota), so LLM reliability
  and quota were material limits even though only 301,168 of 1.8M configured tokens were used.
* The research policy is **high-variance**: across the seven official-convergence runs in
  these ledgers (excluding the one-iteration smoke test), **4 beat the baseline and 3 did
  not** — `research_live1`, `trial_2`, `trial_4` and `research_submission1` improved on it;
  `research_official`, `trial_1` and `trial_3` converged without a promotion. The submitted
  champion is the best of them, and the distribution is published so the best run is not
  mistaken for the expected one. Exactly **one** of those runs has had an `objective_loss`
  candidate *evaluated*; under the tournament policy every run *proposes* one by requirement,
  which is why "proposed" is not the interesting number.
* **Breadth is partly compelled, and the report says so.** The tournament validator *requires*
  a `model_architecture` and a `sampling_training` candidate, so the agent proposing one is
  policy working, not initiative. The agent's own contribution within that constraint is the
  mechanism it picks, the files it edits, and which candidate the judge funds. Read the
  *considered* column of the breadth section as coverage, never as curiosity.
* **One mechanism is now supported by the belief system; the rest are not.**
  `research_submission1` recorded *"Pairwise BPR loss improves within-user ranking metrics over
  pointwise logloss"* with `supporting_iterations: [1]`, `contradicting_iterations: []`,
  confidence 0.8. Every other belief recorded so far carries contradicting evidence — including
  the smoke run's sole belief (`supporting: []`, `contradicting: [1]`), where the narrow
  observation that held was that stronger L2 shrank the train/dev gap (0.085 → 0.033) while
  costing validation primary (−0.0009). Negative results are retained, never overwritten.
* **Pairwise ranking is the dominant *observed* lever — and the secondary effects are
  unresolved, not disproved.** Two materially different surrounding configurations built on the
  same pairwise implementation reached nearly identical 20-seed improvements (+0.001888 and
  +0.001845). That is strong evidence about where the measured gain lives. It is *not* evidence
  that the surrounding choices are worthless: the negative-sampling change lost only on a single
  search iteration rather than a 20-seed component ablation, `user_hist_len` helped in the
  original seed-0 lineage, and the L2-plus-cosine step was never confirmed against its immediate
  parent across seeds. These are **small or unresolved secondary effects**, and no component
  ablation has been run.
* The agent over-predicted its own gains on essentially every scored iteration. The run log
  shows predicted and actual side by side rather than hiding it.
* The train/dev gap is a **diagnostic**: the harness verifies the *size* of the returned
  train-subset vector but cannot verify which rows produced it.
* Docker command construction (including read-only source mounts and CUDA passthrough) is
  tested, but a real Docker daemon and CUDA device were unavailable on this Mac. The controlled
  accelerated profile has exercised MPS; arbitrary agent-written Torch code can still ignore
  its resolved target, so its device use remains **unverified unless separately reported and
  inspected**. The discovery DeepFM exposed exactly that distinction.
* The history is **local only** — no remote and nothing pushed, by instruction.
  The dataset, credentials, fitted models and prediction vectors are excluded; the signed
  ledgers, the artifact store and the submitted prediction file are committed, because a
  reader who cannot check a hash against an artifact cannot check anything.
