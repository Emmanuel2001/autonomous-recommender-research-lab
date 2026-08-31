#!/bin/bash
# Run every suite. Exit non-zero if any fails.
#
# ISOLATION FROM LIVE RUNS: tests/test_scorer.py deliberately unlinks the scorer socket
# and the ledger to test a cold start. Both paths are read from the environment, and
# both default INSIDE runs/ -- so with the defaults, running this suite while an
# experiment is in flight silently kills that experiment and can delete its hash chain.
# That is not hypothetical: it happened, twice, and the second time cost a live run.
# Pinning both to a scratch directory here makes the suite safe to run at any time, and
# keeps the guarantee in one place rather than in every test file.
cd "$(dirname "$0")/.." || exit 1

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/track2_tests_XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
export TRACK2_SOCKET="$SCRATCH/scorer.sock"
export TRACK2_LEDGER="$SCRATCH/ledger.jsonl"
echo "test scratch: $SCRATCH  (live runs/ is untouched)"

# A live scorer must not be inherited either: a suite that silently talks to whatever
# is already listening is testing the wrong process.
if [ -S "runs/scorer.sock" ]; then
  echo "note: a scorer is listening on runs/scorer.sock -- the suite ignores it and"
  echo "      uses $TRACK2_SOCKET instead."
fi

fail=0
failed=""
for t in test_metrics test_official_metric test_task test_actions test_official_actions \
         test_artifacts test_budgets test_candidate test_research_loop test_research_report \
         test_tournament test_frozen_and_profiles \
         test_ledger test_handshake test_claims test_splits test_official_data test_patch test_deps test_inspect \
         test_charts test_report test_draft test_control test_control_http test_scorer test_isolation test_selection; do
  echo "=========================================================== $t"
  if ! python3 "tests/$t.py"; then
    fail=1
    failed="$failed $t"
  fi
done
echo
if [ $fail -eq 0 ]; then
  echo "ALL SUITES PASSED"
else
  echo "SOME SUITES FAILED:$failed"
fi
exit $fail
