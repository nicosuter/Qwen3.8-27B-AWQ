#!/usr/bin/env bash
# Check that a run directory's comparison step reports what it actually did.
#
# Two jobs finished, printed `paired-eval=complete` naming a comparison file,
# and wrote no such file. The comparator had aborted on LiveCodeBench rows that
# are deferred by design -- their verdict comes later, from a CPU job -- and the
# `|| true` that keeps a failing near-lossless gate from failing the job
# swallowed the abort along with it. The path was then echoed regardless, so the
# only way to find out there was no verdict was to open the directory.
#
# A failing gate and an abort are different outcomes and have to be told apart:
# the gate is the answer, the abort is the absence of one.
set -uo pipefail

SCRIPT="${1:?usage: test_compare_run_dir.sh <path to compare_run_dir.sh>}"
ROOT="$(cd "$(dirname "$SCRIPT")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PASS=0
FAIL=0

check() { # check <desc> <expected> <actual>
    if [[ "$2" == "$3" ]]; then
        echo "  ok   $1"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $1: expected [$2] got [$3]"
        FAIL=$((FAIL + 1))
    fi
}

contains() { # contains <desc> <needle> <haystack>
    if [[ "$3" == *"$2"* ]]; then
        echo "  ok   $1"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $1: [$2] not found in [$3]"
        FAIL=$((FAIL + 1))
    fi
}

absent() { # absent <desc> <needle> <haystack>
    if [[ "$3" != *"$2"* ]]; then
        echo "  ok   $1"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $1: [$2] should not appear in [$3]"
        FAIL=$((FAIL + 1))
    fi
}

# A run directory with one scored suite on each side. The rows only have to be
# enough for the stub comparator to be handed something.
mkdir -p "$WORK/run/raw/baseline" "$WORK/run/raw/candidate"
echo '{"suite":"ruler","id":"a","score":1}' > "$WORK/run/raw/baseline/ruler-r0.jsonl"
echo '{"suite":"ruler","id":"a","score":1}' > "$WORK/run/raw/candidate/ruler-r0.jsonl"

# The script is run from a tree where the comparator and the suite lister are
# stubs, so the outcomes below are chosen rather than provoked.
mkdir -p "$WORK/eval/scripts"
cp "$SCRIPT" "$WORK/eval/scripts/compare_run_dir.sh"
chmod +x "$WORK/eval/scripts/compare_run_dir.sh"
cat > "$WORK/eval/scripts/eval_suite.py" <<'STUB'
import sys
if "--names" in sys.argv:
    print("ruler")
STUB

# COMPARE_MODE picks the outcome: a written verdict, a written verdict with a
# failing gate, or an abort that writes nothing. All three exit non-zero except
# the first, which is exactly why exit status alone cannot distinguish them.
cat > "$WORK/eval/scripts/compare_eval_results.py" <<'STUB'
import os, sys
out = sys.argv[sys.argv.index("--output") + 1]
mode = os.environ.get("COMPARE_MODE", "verdict")
if mode == "abort":
    print("00_a is deferred and has not been scored", file=sys.stderr)
    sys.exit(2)
open(out, "w").write('{"macro": {"recovery": 0.99}}\n')
sys.exit(1 if mode == "gate-fails" else 0)
STUB

run() { # run <mode>  -> sets $out $rc
    out="$(cd "$WORK" && \
        COMPARE_MODE="$1" RUN_DIR="$WORK/run" PYTHON=python3 \
        EVAL_SUITE_VERSION=v2 RUN_STAMP=20260820T000000Z COMPARISON_TAG=test \
        ./eval/scripts/compare_run_dir.sh 2>&1)"
    rc=$?
}

reset() { rm -rf "$WORK/run/comparisons" "$WORK/run/comparison.json"; }

echo "a verdict is reported and the path is real"
reset; run verdict
check "exits zero" 0 "$rc"
contains "names the comparison" "comparison=$WORK/run/comparisons/" "$out"
named="$(sed -n 's/^comparison=//p' <<<"$out")"
check "the named file exists" "yes" "$([[ -s "$named" ]] && echo yes || echo no)"
check "comparison.json points at it" "yes" \
    "$([[ -L "$WORK/run/comparison.json" ]] && echo yes || echo no)"

echo "a failing gate is a verdict, not a job error"
reset; run gate-fails
check "exits zero" 0 "$rc"
named="$(sed -n 's/^comparison=//p' <<<"$out")"
check "the named file exists" "yes" "$([[ -s "$named" ]] && echo yes || echo no)"

echo "an abort names nothing and says so"
reset; run abort
check "exits non-zero" "yes" "$([[ $rc -ne 0 ]] && echo yes || echo no)"
absent "no comparison= line" "comparison=" "$out"
contains "reports why" "deferred" "$out"
check "no comparison.json symlink" "no" \
    "$([[ -e "$WORK/run/comparison.json" ]] && echo yes || echo no)"

# The callers are the ones that print the line a human reads, so an accurate
# exit status here only helps if they consult it. Both invoke this script the
# same way and both used to announce completion unconditionally.
echo "both callers report the absence of a verdict"
for caller in eval/slurm/paired-suite-eval.sbatch eval/slurm/score-deferred.sbatch; do
    src="$(cat "$ROOT/$caller")"
    tail_line="$(grep -n 'compare_run_dir.sh' "$ROOT/$caller" | head -1)"
    check "$caller calls compare_run_dir.sh" "yes" \
        "$([[ -n "$tail_line" ]] && echo yes || echo no)"
    # An unguarded `=complete` on the same line as the comparison path is the
    # bug: it asserts a verdict without having looked for one.
    check "$caller does not announce completion unconditionally" "no" \
        "$(grep -qE '^echo "(paired-eval|score-deferred)=complete .*comparison=\$COMPARISON"' \
            "$ROOT/$caller" && echo yes || echo no)"
    contains "$caller branches on an empty comparison" 'if [[ -n "$COMPARISON" ]]' "$src"
done

echo ""
echo "passed $PASS, failed $FAIL"
[[ $FAIL -eq 0 ]]
