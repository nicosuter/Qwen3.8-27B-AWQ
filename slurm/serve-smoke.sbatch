#!/usr/bin/env bash
#SBATCH --job-name=qwen38-serve-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=qwen38-serve-smoke-%j.out

# Smallest possible end-to-end check: serve the local checkpoint with the pinned
# vLLM image and ask it one arithmetic question. This is deliberately not the
# EVAL.md protocol -- it answers "does the checkpoint load and generate at all",
# which is the question worth answering before an upload.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./scripts/load_env.sh
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_BASE/v2/model}"
: "${EVAL_APPTAINER_IMAGE:?set EVAL_APPTAINER_IMAGE to the pinned vLLM .sif}"

test -s "$OUTPUT_DIR/config.json"
test -s "$OUTPUT_DIR/model.safetensors.index.json"

# Refuse to smoke an uncompressed checkpoint: a save path that silently drops
# the packed weights produces a loadable BF16 model that passes every generation
# check while being the wrong artifact entirely.
PACKED="$(python3 -c "
import json, sys
index = json.load(open('$OUTPUT_DIR/model.safetensors.index.json'))
print(sum('weight_packed' in name for name in index['weight_map']))
")"
if [[ "$PACKED" -eq 0 ]]; then
    echo "checkpoint contains no weight_packed tensors; it is not compressed" >&2
    exit 1
fi
echo "checkpoint-compressed=ok packed_tensors=$PACKED"
du -sh "$OUTPUT_DIR"

PORT="$(( 8000 + (SLURM_JOB_ID % 1000) ))"
SERVED_NAME=qwen38-awq-smoke
SERVER_LOG="$(mktemp -t serve-smoke-XXXXXX.log)"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# %runscript is `exec vllm serve "$@"`, so the first argument is the model path.
apptainer run --nv \
    --bind "$OUTPUT_DIR:/model:ro" \
    "$EVAL_APPTAINER_IMAGE" \
    /model \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 4096 \
    --enforce-eager \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "waiting for vllm on port $PORT (log: $SERVER_LOG)"
DEADLINE=$(( SECONDS + 900 ))
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vllm exited before becoming ready; last 40 log lines:" >&2
        tail -40 "$SERVER_LOG" >&2
        exit 1
    fi
    if (( SECONDS > DEADLINE )); then
        echo "vllm did not become ready within 900s; last 40 log lines:" >&2
        tail -40 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done
echo "server-ready=ok port=$PORT"

RESPONSE="$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",
         \"messages\":[{\"role\":\"user\",\"content\":\"What is 1+1? Reply with only the number.\"}],
         \"max_tokens\":16,
         \"temperature\":0}")"

echo "$RESPONSE"
ANSWER="$(printf '%s' "$RESPONSE" | python3 -c "
import json, sys
print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())
")"
echo "answer=$ANSWER"

if [[ "$ANSWER" != *2* ]]; then
    echo "model did not answer 2; treat the checkpoint as suspect" >&2
    exit 1
fi
echo "serve-smoke=ok"
