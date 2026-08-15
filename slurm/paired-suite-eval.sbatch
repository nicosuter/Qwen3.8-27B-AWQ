#!/usr/bin/env bash
#SBATCH --job-name=qwen38-paired-suites
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=qwen38-paired-suites-%j.out

# First paired FP8-versus-AWQ measurement. Serves the pinned baseline, scores
# the ready suites, serves the candidate, scores the same frozen items, then
# compares them.
#
# This is not the EVAL.md protocol: one replicate, no checkpoint alternation
# across seeds, no overlap audit, and only the suites that have met a live
# server. It exists to exercise the paired loop and produce a first delta.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./scripts/load_env.sh
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_BASE/v2/model}"
: "${EVAL_APPTAINER_IMAGE:?set EVAL_APPTAINER_IMAGE to the pinned vLLM .sif}"

# At data-parallel size 8, eight workers share one squashfuse mount and it has
# dropped out mid-startup, surfacing as ENOTCONN on libc reads and a confusing
# "NVCC compilation failed" from the DeepGEMM warmup. An unpacked sandbox takes
# FUSE out of the path. It is derived from the pinned image, so record which
# image it came from: the sandbox itself carries no hash of its own.
IMAGE="$EVAL_APPTAINER_IMAGE"
SANDBOX="${EVAL_APPTAINER_SANDBOX:-${EVAL_APPTAINER_IMAGE%.sif}-sandbox}"
if [[ -d "$SANDBOX" ]]; then
    IMAGE="$SANDBOX"
    echo "container=sandbox path=$SANDBOX derived_from=$EVAL_APPTAINER_IMAGE"
    sha256sum "$EVAL_APPTAINER_IMAGE" || true
else
    echo "container=sif path=$EVAL_APPTAINER_IMAGE"
fi

RUN_DIR="${PAIRED_RUN_DIR:-$RUN_BASE/v2/paired-1}"
CONFIG="${PAIRED_CONFIG:-$RUN_BASE/eval/paired-1.json}"
SUITES="${PAIRED_SUITES:-gpqa_diamond ruler}"
DP="${PAIRED_DP:-8}"
CONCURRENCY="${PAIRED_CONCURRENCY:-}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
PORT="$(( 8000 + (SLURM_JOB_ID % 1000) ))"
SERVED_NAME=qwen38-eval
BASE_URL="http://127.0.0.1:$PORT/v1"
LOG_DIR="$RUN_BASE/v2/logs"
# The baseline lives in a Hugging Face cache, whose snapshot directory is a farm
# of symlinks into ../../blobs. Binding the snapshot alone leaves every one of
# them dangling inside the container, so bind the repository root and address
# the snapshot through it.
BASELINE_REPO="${PAIRED_BASELINE_REPO:?set PAIRED_BASELINE_REPO to the models--... directory}"
BASELINE_REVISION="${PAIRED_BASELINE_REVISION:?set PAIRED_BASELINE_REVISION to the pinned commit}"
BASELINE_DIR="$BASELINE_REPO/snapshots/$BASELINE_REVISION"
mkdir -p "$LOG_DIR"

test -s "$CONFIG"
test -s "$OUTPUT_DIR/config.json"
test -s "$BASELINE_DIR/config.json"
for suite in $SUITES; do
    test -s "$RUN_DIR/orders/$suite.json"
done

SERVER_PID=""
stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT

start_server() {
    local variant="$1" bind_src="$2" model_path="$3"
    local log="$LOG_DIR/paired-${variant}-${SLURM_JOB_ID:-manual}.log"
    # Identical flags and served name for both checkpoints, so no harness
    # behavior can branch on which model is loaded.
    apptainer run --nv \
        --bind "$bind_src:/mnt/model:ro" \
        "$IMAGE" \
        "$model_path" \
        --served-model-name "$SERVED_NAME" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --data-parallel-size "$DP" \
        --max-model-len 262144 \
        --kv-cache-dtype auto \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        > "$log" 2>&1 &
    SERVER_PID=$!
    echo "waiting for vllm variant=$variant (log: $log)"
    local deadline=$(( SECONDS + 2400 ))
    until curl -sf "$BASE_URL/models" > /dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            sleep 1
            echo "vllm exited before becoming ready; see $log" >&2
            exit 1
        fi
        if (( SECONDS > deadline )); then
            echo "vllm did not become ready within 2400s; see $log" >&2
            exit 1
        fi
        sleep 5
    done
    echo "server-ready=ok variant=$variant"
}

variant_is_scored() {
    local variant="$1"
    for suite in $SUITES; do
        [[ -s "$RUN_DIR/raw/$variant/$suite-r0.jsonl" ]] || return 1
    done
    return 0
}

score_variant() {
    local variant="$1"
    echo "=== generation policy probe ($variant) ==="
    "$PYTHON" scripts/adapters/gpqa_diamond.py probe --base-url "$BASE_URL" --model "$SERVED_NAME"
    for suite in $SUITES; do
        echo "=== $suite ($variant) ==="
        "$PYTHON" scripts/run_adapter_suite.py \
            --config "$CONFIG" --suite "$suite" --run-dir "$RUN_DIR" \
            --phase run --variant "$variant" --base-url "$BASE_URL" \
            ${CONCURRENCY:+--concurrency "$CONCURRENCY"}
    done
}

# A cancelled run leaves one side already scored. Reusing it costs nothing in
# validity as long as the flags and image are the same, and it keeps the node
# for half as long.
if [[ "${PAIRED_FORCE:-0}" != "1" ]] && variant_is_scored baseline; then
    echo "baseline already scored in $RUN_DIR/raw/baseline; skipping"
else
    start_server baseline "$BASELINE_REPO" "/mnt/model/snapshots/$BASELINE_REVISION"
    score_variant baseline
    stop_server
fi

if [[ "${PAIRED_FORCE:-0}" != "1" ]] && variant_is_scored candidate; then
    echo "candidate already scored in $RUN_DIR/raw/candidate; skipping"
else
    start_server candidate "$OUTPUT_DIR" "/mnt/model"
    score_variant candidate
    stop_server
fi

# One file per checkpoint so the comparator sees every suite at once; it
# requires identical keys on both sides and will say so if a suite is missing.
for variant in baseline candidate; do
    cat "$RUN_DIR"/raw/"$variant"/*.jsonl > "$RUN_DIR/$variant-all.jsonl"
done

echo "=== paired comparison ==="
"$PYTHON" scripts/compare_eval_results.py \
    --baseline "$RUN_DIR/baseline-all.jsonl" \
    --candidate "$RUN_DIR/candidate-all.jsonl" \
    --output "$RUN_DIR/comparison.json" \
    --baseline-floor gpqa_diamond="${GPQA_FLOOR:-0.75}" \
    || true   # a failing gate is a result, not a job error

echo "paired-eval=complete results=$RUN_DIR"
