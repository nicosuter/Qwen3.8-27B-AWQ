#!/usr/bin/env bash
#SBATCH --job-name=evalscope-smoke
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#
# Serve one checkpoint and score a handful of items through EvalScope.
#
# Everything about the port has been verified against `mock_llm`, which never
# opens a socket. This is the part that cannot be checked that way: whether our
# generation parameters actually reach vLLM, what the served reply looks like
# coming back through EvalScope's OpenAI-compatible client, and whether a lane
# writes the two files the comparison step expects.
#
# It is deliberately not a paired run. One variant, one suite, a few items.
set -euo pipefail

RUN_BASE="${RUN_BASE:?set RUN_BASE}"
SUITE="${SMOKE_SUITE:-mmlu_pro}"
LIMIT="${SMOKE_LIMIT:-20}"
MAX_TOKENS="${SMOKE_MAX_TOKENS:-4096}"
VENV="${EVAL_VENV:?set EVAL_VENV to the evalscope venv}"
PYTHON="$VENV/bin/python"
IMAGE="${EVAL_APPTAINER_IMAGE:?set EVAL_APPTAINER_IMAGE}"
MODEL_DIR="${SMOKE_MODEL_DIR:?set SMOKE_MODEL_DIR to the checkpoint to serve}"
OUT="${SMOKE_OUT:-$RUN_BASE/v2/evalscope-smoke}"

PORT="$(( 8000 + (SLURM_JOB_ID % 1000) ))"
BASE_URL="http://127.0.0.1:$PORT/v1"
SERVED_NAME=qwen38-eval
mkdir -p "$OUT"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
echo "gpu=$GPU_NAME suite=$SUITE limit=$LIMIT max_tokens=$MAX_TOKENS"

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

apptainer run --nv --bind "$MODEL_DIR:/mnt/model:ro" "$IMAGE" /mnt/model \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 --data-parallel-size 4 \
    --max-model-len 262144 --kv-cache-dtype auto \
    --reasoning-parser qwen3 \
    > "$OUT/server.log" 2>&1 &
SERVER_PID=$!

echo "waiting for vllm"
deadline=$(( SECONDS + 2400 ))
until curl -sf "$BASE_URL/models" > /dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "vllm exited; see $OUT/server.log" >&2; exit 1; }
    (( SECONDS > deadline )) && { echo "vllm never became ready" >&2; exit 1; }
    sleep 5
done
echo "server-ready=ok"

"$PYTHON" scripts/evalscope_bridge.py run \
    --suite "$SUITE" --variant baseline --replicate 0 \
    --model "$SERVED_NAME" --api-url "$BASE_URL" \
    --work-dir "$OUT/work" \
    --results "$OUT/raw/baseline/$SUITE-r0.jsonl" \
    --metadata "$OUT/metadata/$SUITE-baseline-r0.json" \
    --max-tokens "$MAX_TOKENS" --limit "$LIMIT"

echo "=== rows ==="
head -2 "$OUT/raw/baseline/$SUITE-r0.jsonl"
echo "=== metadata ==="
cat "$OUT/metadata/$SUITE-baseline-r0.json"
echo "evalscope-smoke=complete"
