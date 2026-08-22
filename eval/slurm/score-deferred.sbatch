#!/usr/bin/env bash
#SBATCH --job-name=eval-qwen38-27b-score-deferred
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/%x-%j.out

# Execute LiveCodeBench's deferred generations, score them, and re-run the
# comparison over the run directory.
#
# No --gres on purpose. The suite runs with --defer-execution because executing
# model-written code on the node serving the model is the one thing that clearly
# does not belong there, so the run writes generations and a metadata sidecar
# marked deferred, and stops. Nothing was running the other half: the comparator
# refuses a deferred row rather than counting it as zero, so every comparison
# this protocol has produced has failed on the first LiveCodeBench item.
#
# Safe to re-run. A suite already executed is skipped unless PAIRED_FORCE=1, and
# the comparison is rebuilt from whatever is on disk either way -- which is also
# how a run directory scored before the comparator reported generation lengths
# gets one that does.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./common/scripts/load_env.sh

RUN_DIR="${PAIRED_RUN_DIR:?set PAIRED_RUN_DIR to the run directory to score}"
EVAL_SUITE_VERSION="${PAIRED_EVAL_SUITE:-v2}"
SUITE="${PAIRED_DEFERRED_SUITE:-livecodebench_v6}"
REPLICATES="${PAIRED_REPLICATES:-1}"
# Same convention as the other entry points: EVAL_VENV names the environment and
# the interpreter is derived from it.
PYTHON="${EVAL_VENV:+$EVAL_VENV/bin/python}"
PYTHON="${PYTHON:-${EVAL_PYTHON:-python3}}"

# Executing untrusted generated code is the whole job, so the limits are not
# incidental. They are the adapter's defaults unless a deployment says otherwise.
EXEC_TIMEOUT="${PAIRED_EXEC_TIMEOUT:-}"
EXEC_MEMORY_MB="${PAIRED_EXEC_MEMORY_MB:-}"
ITEM_BUDGET="${PAIRED_ITEM_BUDGET:-}"
CONCURRENCY="${PAIRED_EXEC_CONCURRENCY:-${SLURM_CPUS_PER_TASK:-8}}"

KEY="$RUN_DIR/materialized/$SUITE.key.json"
test -s "$KEY" || { echo "no materialized key at $KEY" >&2; exit 1; }

echo "run-dir=$RUN_DIR suite=$SUITE replicates=$REPLICATES concurrency=$CONCURRENCY"

scored=0
skipped=0
for variant in baseline candidate; do
    for (( rep = 0; rep < REPLICATES; rep++ )); do
        generations="$RUN_DIR/generations/$SUITE-$variant-r$rep.jsonl"
        results="$RUN_DIR/raw/$variant/$SUITE-r$rep.jsonl"
        metadata="$RUN_DIR/metadata/$SUITE-$variant-r$rep.json"
        if [[ ! -s "$generations" ]]; then
            echo "=== $SUITE r$rep ($variant): nothing generated, skipping"
            continue
        fi
        # `deferred` in the sidecar is the record of whether the verdict is in.
        # Reading it rather than the rows means a partially written result file
        # is not mistaken for a finished one.
        if [[ "${PAIRED_FORCE:-0}" != "1" ]] && [[ -s "$metadata" ]] && \
           ! "$PYTHON" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('deferred') else 1)" \
                "$metadata"; then
            echo "=== $SUITE r$rep ($variant): already executed, keeping it"
            skipped=$(( skipped + 1 ))
            continue
        fi
        echo "=== $SUITE r$rep ($variant): executing $(wc -l < "$generations") generations"
        "$PYTHON" eval/scripts/adapters/$SUITE.py score \
            --generations "$generations" \
            --key "$KEY" \
            --results "$results" \
            --metadata "$metadata" \
            --concurrency "$CONCURRENCY" \
            ${EXEC_TIMEOUT:+--exec-timeout "$EXEC_TIMEOUT"} \
            ${EXEC_MEMORY_MB:+--exec-memory-mb "$EXEC_MEMORY_MB"} \
            ${ITEM_BUDGET:+--item-budget "$ITEM_BUDGET"}
        scored=$(( scored + 1 ))
    done
done

echo "scored=$scored skipped=$skipped"

# Rebuilt over the whole run directory, not just this suite: the comparator
# needs both arms complete, and the point of executing here is to let the
# suites that were already scored finally produce a verdict alongside this one.
#
# tee -a, not tee: under Slurm /dev/stderr resolves to the job log, and tee
# without -a opens it with O_TRUNC. Every completed run lost its whole log at
# this last step, leaving only the lines the comparator printed after it.
COMPARISON="$(
    RUN_DIR="$RUN_DIR" PYTHON="$PYTHON" \
    EVAL_SUITE_VERSION="$EVAL_SUITE_VERSION" \
    COMPARISON_TAG="${SLURM_JOB_ID:-manual}" \
    ALLOW_PARTIAL="${PAIRED_ALLOW_PARTIAL:-0}" \
    ./eval/scripts/compare_run_dir.sh | tee -a /dev/stderr | sed -n 's/^comparison=//p' || true
)"

# This job exists to produce the verdict, so failing to produce one is its
# failure and not a footnote -- a missing arm, an unscored suite, anything the
# comparator stops on.
if [[ -n "$COMPARISON" ]]; then
    echo "score-deferred=complete results=$RUN_DIR comparison=$COMPARISON"
else
    echo "score-deferred=failed results=$RUN_DIR comparison=none" \
         "(executed $scored, skipped $skipped; the comparator produced no verdict)" >&2
    exit 1
fi
