#!/usr/bin/env bash
#SBATCH --job-name=qwen38-adapter-dry-run
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=qwen38-adapter-dry-run-%j.out

# First scored inference through the real adapters: serve the AWQ checkpoint and
# run the RULER and GPQA suites against it. This is a harness dry run, not the
# EVAL.md protocol: one checkpoint, no replicates, no paired comparison, no
# overlap audit, and GPQA is truncated. It answers "does the harness produce
# valid scored rows against a live server", nothing about release quality.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./scripts/load_env.sh
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_BASE/v2/model}"
: "${EVAL_APPTAINER_IMAGE:?set EVAL_APPTAINER_IMAGE to the pinned vLLM .sif}"

DRY_RUN_DIR="${DRY_RUN_DIR:-$RUN_BASE/v2/dry-run}"
CONFIG="${DRY_RUN_CONFIG:-$RUN_BASE/eval/dry-run.json}"
GPQA_LIMIT="${GPQA_LIMIT:-24}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
PORT="$(( 8000 + (SLURM_JOB_ID % 1000) ))"
SERVED_NAME=qwen38-awq-dry
BASE_URL="http://127.0.0.1:$PORT/v1"
LOG_DIR="$RUN_BASE/v2/logs"
mkdir -p "$LOG_DIR"

test -s "$OUTPUT_DIR/config.json"
test -s "$CONFIG"
test -s "$DRY_RUN_DIR/orders/gpqa_diamond.json"
test -s "$DRY_RUN_DIR/orders/ruler.json"

cleanup() {
    if [[ -n "${TAIL_PID:-}" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

SERVER_LOG="$LOG_DIR/adapter-dry-run-${SLURM_JOB_ID:-manual}.log"
: > "$SERVER_LOG"

# %runscript is `exec vllm serve "$@"`, so the first argument is the model.
# Context and parsers match the paired protocol. One GPU on purpose: this box
# is shared, and a 45-item dry run does not justify crowding anyone out.
apptainer run --nv \
    --bind "$OUTPUT_DIR:/model:ro" \
    "$EVAL_APPTAINER_IMAGE" \
    /model \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 262144 \
    --kv-cache-dtype auto \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

tail -n +1 -f "$SERVER_LOG" | sed -u "s/^/[vllm] /" &
TAIL_PID=$!

echo "waiting for vllm port=$PORT (log: $SERVER_LOG)"
DEADLINE=$(( SECONDS + 1800 ))
until curl -sf "$BASE_URL/models" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        sleep 1
        echo "vllm exited before becoming ready; full log: $SERVER_LOG" >&2
        exit 1
    fi
    if (( SECONDS > DEADLINE )); then
        echo "vllm did not become ready within 1800s; full log: $SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done
echo "server-ready=ok port=$PORT"

# Fail before scoring if the server rejects or ignores the generation policy.
# A rejected reasoning_effort would abort every suite one request in; silently
# dropped thinking would score a different model than the protocol describes.
echo "=== generation policy probe ==="
"$PYTHON" scripts/adapters/gpqa_diamond.py probe --base-url "$BASE_URL" --model "$SERVED_NAME"

# RULER first: it is the suite with no upstream anchor and the one whose
# synthesis has never met a real server.
echo "=== ruler (21 items, 4k/32k/128k) ==="
"$PYTHON" scripts/run_adapter_suite.py \
    --config "$CONFIG" --suite ruler --run-dir "$DRY_RUN_DIR" \
    --phase run --variant candidate --base-url "$BASE_URL"

echo "=== gpqa_diamond (first $GPQA_LIMIT of the frozen order) ==="
"$PYTHON" scripts/run_adapter_suite.py \
    --config "$CONFIG" --suite gpqa_diamond --run-dir "$DRY_RUN_DIR" \
    --phase run --variant candidate --base-url "$BASE_URL" --limit "$GPQA_LIMIT"

echo "dry-run=complete results=$DRY_RUN_DIR/raw/candidate"
