# Future Work: Faithful DGM Extension

## Decision to revisit

The current project is a DGM-inspired autonomous ML research agent. It evolves recommender candidates, but it does not recursively evolve the research agent itself.

This was a deliberate hackathon scope decision, not a claim that the current implementation is a faithful reproduction of Darwin Godel Machine.

## Current boundary

The research agent can propose and evaluate changes to recommender components such as:

- model architecture;
- features and embeddings;
- loss functions;
- sampling and training procedures;
- optimisation settings and configuration.

The protected research machinery remains fixed, including:

- planner, judge, coder and reflector behaviour;
- prompts and role-routing policy;
- archive and parent-selection policy;
- memory and belief representation;
- patching and tool-use policy;
- scorer, labels, sandbox and integrity controls.

## What a faithful DGM extension would add

Add a second, outer optimisation loop:

1. Select a version of the research agent from an archive.
2. Let that agent perform several bounded recommender-research episodes.
3. Evaluate the agent across multiple tasks, datasets and seeds.
4. Permit controlled modifications to mutable research-agent code.
5. Store the resulting agent version with its lineage and evaluation evidence.
6. Retain successful and useful unsuccessful variants as future parents.

The scorer, hidden labels, safety policy and outer evaluator must remain immutable.

## Prerequisites

- A research-agent fitness function, separate from recommender score.
- Multiple evaluation environments, including Pure, 1K and 27K where practical.
- Held-out research tasks to detect meta-overfitting.
- Fixed LLM-token, training, wall-time and intervention budgets.
- Evaluation of improvement rate, reliability, cost and safety violations.
- Versioned mutable agent code with an immutable outer sandbox.
- Enough API and compute budget to evaluate several agent variants fairly.

## Important distinction

Improving a recommender does not prove that the researcher itself improved. A faithful DGM experiment must compare research-agent versions on repeated, controlled research episodes.

## Suggested future milestone

Build a small bounded prototype with two or three agent variants, three research episodes per variant and an immutable outer evaluator. Demonstrate that an agent-authored change to planning, memory or experiment selection improves research performance on a held-out episode without weakening safety or increasing the allowed budget.

Do not describe this extension as implemented until that outer loop has actually run and produced verified lineage and evaluation records.
