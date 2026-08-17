#!/usr/bin/env bash
#SBATCH --job-name=qwen38-prepare-eval
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/%x-%j.out

# Materialize every suite in the protocol and freeze its task order. No GPU: this
# downloads datasets, writes images, resolves pins and shuffles orders, none of
# which needs one, and asking for four would idle them for an hour and take a
# node from somebody who needs it.
#
# It exists because paired-suite-eval.sbatch does not prepare. It asserts that
# orders/<suite>.json is already there and stops if it is not, which is correct
# for a scoring job but leaves nothing that produces them.
#
# Running this first is also what turns a bad pin or a gated dataset into a
# two-minute failure instead of one discovered four hours into an allocation.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./common/scripts/load_env.sh

RUN_DIR="${PAIRED_RUN_DIR:?set PAIRED_RUN_DIR to the run directory to prepare}"
EVAL_SUITE_VERSION="${PAIRED_EVAL_SUITE:-v2}"
CONFIG="${PAIRED_CONFIG:-$PROJECT_DIR/eval/eval-suite-$EVAL_SUITE_VERSION.json}"
BATCH="${PAIRED_BATCH:-prepare}"
# Same convention as paired-suite-eval.sbatch: EVAL_VENV names the environment,
# and EVAL_PYTHON is derived from it. Two entry points that disagreed about
# which variable chooses the interpreter is how a chain died at the first line
# that used one.
PYTHON="${EVAL_VENV:+$EVAL_VENV/bin/python}"
PYTHON="${PYTHON:-${EVAL_PYTHON:-python3}}"

# The same deployment-specific paths paired-suite-eval.sbatch exports. Suite
# commands name them rather than spelling out one cluster's layout, and the
# runner fails closed on an unresolved one.
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_BASE/v2/model}"
export EVAL_PYTHON="$PYTHON"
export EVAL_CORPUS="${PAIRED_CORPUS:-$RUN_BASE/eval/haystack.txt}"
export EVAL_TOKENIZER="${PAIRED_TOKENIZER:-$OUTPUT_DIR}"

SUITES="$("$PYTHON" eval/scripts/eval_suite.py --version "$EVAL_SUITE_VERSION" --batch "$BATCH")"
echo "eval-suite=$EVAL_SUITE_VERSION batch=$BATCH suites=$SUITES"
echo "run-dir=$RUN_DIR config=$CONFIG"

mkdir -p "$RUN_DIR"/{orders,materialized,metadata} "$PROJECT_DIR/slurm-logs"

# Prepared once per run directory and reused by both arms: the frozen order is
# what makes the comparison paired, so regenerating it per variant would quietly
# unpair the run.
FAILED=()
for suite in $SUITES; do
    if [[ -s "$RUN_DIR/orders/$suite.json" ]]; then
        echo "prepare: $suite already frozen, keeping it"
        continue
    fi
    echo "=== prepare $suite"
    if ! "$PYTHON" eval/scripts/run_adapter_suite.py \
        --config "$CONFIG" --suite "$suite" --run-dir "$RUN_DIR" --phase prepare; then
        # One suite's dataset being unreachable should not hide whether the
        # other six are fine; the exit status still reports the failure.
        echo "prepare: $suite FAILED" >&2
        FAILED+=("$suite")
    fi
done

echo "=== frozen orders"
for suite in $SUITES; do
    if [[ -s "$RUN_DIR/orders/$suite.json" ]]; then
        printf '%-24s %s items\n' "$suite" \
            "$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" \
               "$RUN_DIR/orders/$suite.json")"
    else
        printf '%-24s %s\n' "$suite" "MISSING"
    fi
done

if (( ${#FAILED[@]} )); then
    echo "prepare-eval=failed suites=${FAILED[*]}" >&2
    exit 1
fi
echo "prepare-eval=complete run-dir=$RUN_DIR"
