#!/usr/bin/env bash
#SBATCH --job-name=qwen38-policy-probe
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=qwen38-policy-probe-%j.out

# One GPU, one request: does the served checkpoint honor the model-card
# generation policy? Cheap enough to iterate on, so the expensive dry run never
# discovers a serving-flag problem two hours in.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./common/scripts/load_env.sh
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_BASE/v2/model}"
: "${EVAL_APPTAINER_IMAGE:?set EVAL_APPTAINER_IMAGE to the pinned vLLM .sif}"

PYTHON="$PROJECT_DIR/.venv/bin/python"
PORT="$(( 8000 + (SLURM_JOB_ID % 1000) ))"
SERVED_NAME=qwen38-awq-probe
BASE_URL="http://127.0.0.1:$PORT/v1"
LOG_DIR="$RUN_BASE/v2/logs"
# Reasoning parser under test; qwen3 is what the protocol pins.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
mkdir -p "$LOG_DIR"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

SERVER_LOG="$LOG_DIR/policy-probe-${SLURM_JOB_ID:-manual}.log"
apptainer run --nv \
    --bind "$OUTPUT_DIR:/model:ro" \
    "$EVAL_APPTAINER_IMAGE" \
    /model \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 40960 \
    --reasoning-parser "$REASONING_PARSER" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "waiting for vllm port=$PORT parser=$REASONING_PARSER (log: $SERVER_LOG)"
DEADLINE=$(( SECONDS + 1200 ))
until curl -sf "$BASE_URL/models" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        sleep 1
        echo "vllm exited before becoming ready; full log: $SERVER_LOG" >&2
        exit 1
    fi
    if (( SECONDS > DEADLINE )); then
        echo "vllm did not become ready within 1200s" >&2
        exit 1
    fi
    sleep 5
done
echo "server-ready=ok"

echo "=== raw response, so the reasoning field placement is visible ==="
curl -sf "$BASE_URL/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$SERVED_NAME\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Is 2903 prime? Work it out.\"}],
    \"chat_template_kwargs\": {\"enable_thinking\": true, \"reasoning_effort\": \"xhigh\"},
    \"max_tokens\": 16384, \"temperature\": 1.0, \"top_p\": 0.95}" \
    | "$PYTHON" -c "
import json, sys
data = json.load(sys.stdin)
message = data['choices'][0]['message']
content = message.get('content') or ''
reasoning = message.get('reasoning_content') or ''
print('finish_reason      :', data['choices'][0].get('finish_reason'))
print('reasoning_content  :', len(reasoning), 'chars')
print('content            :', len(content), 'chars')
print('closing tag in body:', '</think>' in content)
print('completion_tokens  :', data.get('usage', {}).get('completion_tokens'))
print('usage              :', json.dumps(data.get('usage', {})))
print('--- reasoning head ---'); print(reasoning[:300])
print('--- content head ---'); print(content[:300])
"

echo "=== adapter probe ==="
"$PYTHON" eval/scripts/adapters/gpqa_diamond.py probe --base-url "$BASE_URL" --model "$SERVED_NAME"
echo "policy-probe=ok"
