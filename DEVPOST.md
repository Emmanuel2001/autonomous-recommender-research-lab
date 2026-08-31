# An autonomous ML research agent that writes the code — and reports what failed

**Track 2 — Autonomous ML Research Agent for Recommender Systems**
Task: KuaiRand-Pure, within-user ranking of logged impressions. Label `long_view`.
Metric: `primary = mean(GAUC, nDCG@5)`.

---

## Tagline

A research loop whose actuator is **an edit to a source tree**, not a choice from a menu of
hyperparameters — with a signed, pre-registered ledger that makes every claim in this write-up
checkable, including the ones that make the agent look bad.

---

## What it does

Each iteration, the agent:

1. **proposes six competing hypotheses**, one per research lane — architecture, objective,
   sampling/training, features, optimisation/regularisation, ensembling — and a judge ranks them
   and funds exactly one;
2. **writes the code** for the funded hypothesis, editing any of `config.json`, `features.py`,
   `model.py`, `train.py` in a mutable workspace;
3. **pre-registers** the hypothesis, the resulting tree hash and the harness-computed diff in a
   signed ledger *before the candidate is allowed to run*;
4. **fits and scores** it in a sandbox that cannot read held-out labels;
5. **reflects**, updating a structured belief keyed by *mechanism*, recording whether the
   mechanism was supported or contradicted and under what condition it should be retried.

The champion is immutable and advances only on a signed validation improvement, so a bad run can
spend budget but cannot damage the submission.

## How I built it

**The actuator is a diff, not a menu.** The predecessor exposed 16 typed actions, 14 of which
were hyperparameters — the definition of a naive baseline tweak. Here the agent returns source
files; the harness computes the parent→child diff itself and never trusts a diff supplied by the
model.

**Content-addressed artifacts.** Every candidate tree is stored under its own length-prefixed
hash, so `runs/artifacts/<sha256>/source/` holds the exact code that produced every number in
this document. The full source deliberately does *not* go in the ledger: the signed record cap
(12,288 bytes) is smaller than the legal patch cap (24,000), so a large proposal could not record
itself and the failure would be silent.

**Pre-registration, then execution.** Entries are HMAC-signed and hash-chained. The ordering is
enforced, not conventional: a candidate that fails to register aborts instead of running, which
is what stops a result from being written after the fact.

**Two seed profiles, to keep an honest distinction.** `inherited` ships with a working pairwise
implementation; `minimal` is pointwise-only and reproduces the official baseline
*byte-identically*. On `minimal`, any ranking objective has to be **written** by the agent rather
than switched on — which is how I found out what the agent can actually implement (see below).

**Multi-seed confirmation, with the seed list pre-registered.** A single-seed promotion is a
search decision. Only paired multi-seed results are quoted as findings, and the exact seed list
is recorded before the first fit so it cannot be chosen after seeing the outcome.

## Results

The submitted champion, from a compliant run under the official stopping rule:

| | |
|---|---|
| signed validation primary | **0.604074** (GAUC 0.670569, nDCG@5 0.53758) |
| vs published baseline (0.6016) | **+0.002474** at the run's seed |
| paired 20-seed confirmation | **+0.001888, t = 11.50, wins 20/20** |
| train/dev gap | 0.044325 |
| manual interventions | 0 |

For scale: the seed standard deviation on this task is **0.0008**, and a per-user oracle ceiling
is **0.8645** — so real gains here are small, and a headline number without a paired multi-seed
test behind it is not a result.

**An independent run reached the same place by a different route.** A later compliant run
converged in 3 iterations on a structurally different configuration — no `user_hist_len`,
`neg_per_pos: 1`, `l2: 1e-05`, plus a cosine decay schedule it wrote into `train.py` itself — and
confirmed at **+0.001845, t = 11.12, 20/20**. Compared directly against the submitted champion
over the same 20 seeds: **−0.0000433, t = −0.31, 13/20 to 7/20.** Statistically
indistinguishable, so the frozen champion stands and the rerun is reported as corroboration.

That convergence is the most useful scientific finding here: **pairwise ranking is the dominant
observed lever**, and two materially different surrounding configurations built on the same
pairwise implementation land within 0.00005 of each other.

## Challenges — the parts that failed

This is the section I would most want a judge to read, because the failures are better evidence
that the loop is real than the +0.0019 is.

**The agent's own BPR implementation was broken, and the harness caught it.** On the `minimal`
profile the agent wrote `step_bpr` plus per-user negative sampling across `model.py` and
`train.py` — a genuine multi-file implementation from a pointwise-only start. It scored
**−0.206874**, far below the 0.5 random floor, with train (0.3156) *worse* than validation
(0.3948). The defect: the optimiser expects `dLoss/dz` and descends, but the BPR step passed
`+σ(−diff)` to positives and `−σ(−diff)` to negatives, where the true gradients are the
opposite sign. It therefore performed **gradient ascent** — training the model to rank negatives
above positives. The per-epoch curve confirms it: training loss rose monotonically
**0.738 → 11.165** across 30 epochs. The tree and its source are retained in the repository.

**The reflector diagnosed it vaguely.** It recorded "improper gradient alignment or sampling
mismatch" at confidence 0.95 — directionally near, but it did not identify the sign inversion,
and nothing aborted a run whose loss had been climbing since epoch 2.

**One iteration was silently inert.** The agent implemented feature-level dropout properly — a
constructor parameter, an inverted-scaling binomial mask, `training=True` correctly threaded into
the training path — but never passed the config value into the constructor, so the default `0.0`
stood and `"dropout": 0.1` was dead code. The result was **bit-identical to its parent to 15
significant figures**, and deduplication could not see it because the tree hash genuinely
differed. 38,113 tokens re-derived the champion.

**Architecture was considered nine times and promoted zero times.** Lane coverage is
*required* by the tournament validator, so a `model_architecture` candidate appearing is the
policy working — not the agent showing initiative. The report says this in its own words rather
than letting a "considered" column imply curiosity.

**The agent over-predicted its own gain on essentially every scored iteration.** Predicted and
actual are printed side by side rather than hidden.

## What I learned

**A test that cannot fail is worse than no test.** Of 38 defects found in one review sweep, 12
were in tests and **5 could not fail** — including the two guarding the invariants I would have
defended most confidently. The habit that found them: after fixing a defect, sweep for its
*class* across every module instead of re-reading the file it was found in.

**A measurement that confirms what you expect should be verified before it is acted on.** Nine
times in this project the measurement was wrong rather than the code — a filter that could not
display the outcome I was not expecting, a `timeout` binary that does not exist on macOS and made
28 passing suites report as failures, a single successful API probe read as "the quota reset".

**Never let a test-set number justify a protocol decision.** I once used a test-derived delta to
argue about the convergence threshold. The correct figures were validation-only and smaller. The
episode is recorded in the repository rather than quietly deleted.

**A wall-clock field that reads 0.0 will be believed.** Per-iteration records were written
*inside* the timing context, where the duration had not yet been assigned, so every iteration
logged `wall_seconds: 0.0` against stage sums of 11–18 seconds. The fix ships with a
mutation-tested invariant — every ledger record's wall time must be at least its own stage sum —
which I confirmed fails when the fix is reverted.

## Accomplishments I'm proud of

- **Every number here is checkable.** `python3 -m core.scorer verify --ledger <path>` re-verifies
  the signed chain; artifact hashes and pre-registration ordering are recomputed by the report
  itself.
- **Diagnostic and compliant runs live in separate ledgers.** A run with the stopping rule
  switched off is not evidence about the official protocol and never shares a ledger with a run
  that is.
- **The submission is bound to a promoted artifact**, not reconstructed from a config after the
  fact.
- **28 test suites**, including adversarial tournament cases, a sandbox escape attempt via a
  symlinked dataset, and a scanner that fails the build on stale claims in the documentation.

## Limitations

- Only the cheapest model tier (`gemini-3.5-flash-lite`) has ever driven this loop. Whether a
  stronger tier reaches for architecture is untested.
- No `model_architecture` candidate has been selected and successfully promoted.
- The secondary effects around the pairwise objective are **small or unresolved, not disproved**:
  the negative-sampling change lost on a single search iteration rather than a 20-seed ablation,
  `user_hist_len` helped in the original lineage, and the L2-plus-cosine step was never confirmed
  against its immediate parent across seeds. No component ablation has been run.
- The train/dev gap is a diagnostic: the harness verifies the size of the returned train-subset
  vector but cannot verify which rows produced it.
- Docker isolation is implemented but unexercised; the measured boundary is the macOS seatbelt
  sandbox.

## What's next

1. **A divergence guard.** A monotonically rising training loss is an unambiguous abort signal
   from epoch 2, and would have saved ~150 s on the BPR failure while handing the coder a far
   sharper repair signal than "primary was 0.39".
2. **An inert-change detector.** A delta of exactly zero *with* an identical train metric means
   semantically inert — cheap to flag, and it recovers an evaluation slot that hash-based
   deduplication structurally cannot see.
3. **Component ablations** over 20 seeds, to turn the unresolved secondary effects into
   measurements.
4. **A stronger tier on the architecture lane**, which is the one open question the budget never
   let me answer.

## How to verify

```bash
python3 -m core.scorer verify --ledger runs/ledgers/research_submission.jsonl   # signed chain
python3 -m agent.research_report --ledger runs/ledgers/research_submission.jsonl # report
for f in tests/test_*.py; do python3 "$f"; done                                  # 28 suites
```

`submission_compliant/MANIFEST.json` binds the submitted predictions to the champion tree hash,
the signing sequence number, and the score vector's own SHA-256.
