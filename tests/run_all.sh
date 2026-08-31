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

# Removed on ANY exit: success, a failing suite, or an interrupt. The signal traps exit
# explicitly, because a bash trap on INT/TERM otherwise runs the handler and then carries on
# to the next suite -- which is how a Ctrl-C mid-run used to leave the scratch tree behind.
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 129' HUP

export TRACK2_SOCKET="$SCRATCH/scorer.sock"
export TRACK2_LEDGER="$SCRATCH/ledger.jsonl"
# THE ARTIFACT STORE IS TRACKED. runs/artifacts/ holds the trees the ledger names by hash, so
# a reader can verify them -- but the suite writes real trees on every run, which left
# untracked artifacts in the repository and turned `git status` into a commit hazard. Point
# the store at the scratch tree; core/artifacts.py reads TRACK2_ARTIFACTS once at import and
# requires it to be absolute, which mktemp -d guarantees.
export TRACK2_ARTIFACTS="$SCRATCH/artifacts"
mkdir -p "$TRACK2_ARTIFACTS"
echo "test scratch: $SCRATCH  (live runs/ and runs/artifacts/ are untouched)"

# A live scorer must not be inherited either: a suite that silently talks to whatever
# is already listening is testing the wrong process.
if [ -S "runs/scorer.sock" ]; then
  echo "note: a scorer is listening on runs/scorer.sock -- the suite ignores it and"
  echo "      uses $TRACK2_SOCKET instead."
fi

fail=0
failed=""
# The suite list is DISCOVERED, not hard-coded. It used to be a fixed list of names, and
# five new suites (execution, accelerated, portable task, portable scorer, thought
# summaries) were added without being appended -- so `run_all.sh` reported success while
# never running them. A glob cannot rot that way. ORDER puts the fast, foundational suites
# first so a basic breakage is reported before the slow end-to-end ones.
ORDER="test_metrics test_official_metric test_task test_execution test_actions \
       test_official_actions test_artifacts test_budgets test_portable_task \
       test_portable_scorer test_thought_summaries test_accelerated test_candidate \
       test_research_loop test_research_report test_tournament test_frozen_and_profiles \
       test_ledger test_handshake test_claims test_splits test_official_data test_patch \
       test_deps test_inspect test_charts test_report test_draft test_control \
       test_control_http test_scorer test_isolation test_selection"

suites=""
for t in $ORDER; do
  [ -f "tests/$t.py" ] && suites="$suites $t"
done
# Anything on disk that ORDER does not mention still runs, at the end.
for f in tests/test_*.py; do
  t="$(basename "$f" .py)"
  case " $suites " in *" $t "*) ;; *) suites="$suites $t"; echo "note: $t was not in ORDER; running it last" ;; esac
done

total=0
for t in $suites; do total=$((total+1)); done
echo "discovered $total suites ($(ls tests/test_*.py | wc -l | tr -d ' ') files on disk)"

for t in $suites; do
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
