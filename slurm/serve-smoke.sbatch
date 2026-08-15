#!/usr/bin/env bash
#SBATCH --job-name=qwen38-serve-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=qwen38-serve-smoke-%j.out

# Small end-to-end gate: serve the local checkpoint with the pinned vLLM image
# first without speculation and then with native MTP. This is deliberately not
# the scored EVAL.md protocol.

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
read -r PACKED MTP IGNORED <<<"$(python3 -c "
from pathlib import Path
from scripts.preserve_mtp import QWEN38_MTP_KEYS, QWEN38_MTP_LINEAR_MODULES, QWEN38_MTP_SHAPES, validate_mtp_artifact
result = validate_mtp_artifact(
    Path('$OUTPUT_DIR'),
    expected_keys=QWEN38_MTP_KEYS,
    expected_modules=QWEN38_MTP_LINEAR_MODULES,
    expected_shapes=QWEN38_MTP_SHAPES,
)
print(result['packed_weights'], result['mtp_parameters'], len(result['mtp_ignored_modules']))
")"
if [[ "$PACKED" -eq 0 ]]; then
    echo "checkpoint contains no weight_packed tensors; it is not compressed" >&2
    exit 1
fi
if [[ "$MTP" -ne 15 ]]; then
    echo "checkpoint contains $MTP MTP tensors; expected 15 from the pinned source" >&2
    exit 1
fi
if [[ "$IGNORED" -ne 8 ]]; then
    echo "checkpoint has $IGNORED ignored MTP projections; expected 8" >&2
    exit 1
fi
echo "checkpoint-structure=ok packed_tensors=$PACKED mtp_tensors=$MTP mtp_ignored=$IGNORED"
du -sh "$OUTPUT_DIR"

PORT="$(( 8000 + (SLURM_JOB_ID % 1000) ))"
SERVED_NAME=qwen38-awq-smoke
MTP_MIN_ACCEPTANCE="${MTP_MIN_ACCEPTANCE:-0.40}"
# Keep the server log next to the run, not in a node-local /tmp that disappears
# with the allocation.
LOG_DIR="${RUN_ROOT:-$RUN_BASE/v2}/logs"
mkdir -p "$LOG_DIR"

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

start_server() {
    local mode="$1"
    shift
    SERVER_LOG="$LOG_DIR/serve-smoke-${mode}-${SLURM_JOB_ID:-manual}.log"
    : > "$SERVER_LOG"

    # %runscript is `exec vllm serve "$@"`, so the first argument is the model.
    apptainer run --nv \
        --bind "$OUTPUT_DIR:/model:ro" \
        "$EVAL_APPTAINER_IMAGE" \
        /model \
        --served-model-name "$SERVED_NAME" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --max-model-len 4096 \
        --enforce-eager \
        "$@" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    # Mirror loader failures into Slurm output and retain a per-mode full log.
    tail -n +1 -f "$SERVER_LOG" | sed -u "s/^/[vllm-$mode] /" &
    TAIL_PID=$!

    echo "waiting for vllm mode=$mode port=$PORT (log: $SERVER_LOG)"
    DEADLINE=$(( SECONDS + 900 ))
    until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            sleep 1
            echo "vllm mode=$mode exited before becoming ready; full log: $SERVER_LOG" >&2
            exit 1
        fi
        if (( SECONDS > DEADLINE )); then
            echo "vllm mode=$mode did not become ready within 900s; full log: $SERVER_LOG" >&2
            exit 1
        fi
        sleep 5
    done
    echo "server-ready=ok mode=$mode port=$PORT"
}

stop_server() {
    if [[ -n "${TAIL_PID:-}" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    unset TAIL_PID SERVER_PID
}

run_arithmetic_smoke() {
    local mode="$1"
    local response finish answer
    response="$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_NAME\",
             \"messages\":[{\"role\":\"user\",\"content\":\"What is 1+1? Reply with only the number.\"}],
             \"chat_template_kwargs\":{\"enable_thinking\":false},
             \"max_tokens\":512,
             \"temperature\":0}")"

    read -r finish answer <<<"$(printf '%s' "$response" | python3 -c "
import json, sys
choice = json.load(sys.stdin)['choices'][0]
message = choice['message']
# A reasoning answer may arrive split across reasoning_content and content.
text = ' '.join(
    str(part).replace('\n', ' ')
    for part in (message.get('reasoning_content'), message.get('content'))
    if part
)
print(choice.get('finish_reason'), text.strip())
")"
    echo "arithmetic mode=$mode finish_reason=$finish answer=$answer"
    if [[ "$answer" != *2* ]]; then
        echo "model mode=$mode did not answer 2" >&2
        exit 1
    fi
    echo "arithmetic-smoke=ok mode=$mode"
}

start_server baseline
run_arithmetic_smoke baseline
stop_server

start_server mtp --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
run_arithmetic_smoke mtp

# Measure a counter delta around a longer request. The arithmetic request above
# is intentionally excluded because one or two drafted tokens are too noisy for
# an acceptance gate.
MTP_METRICS_BEFORE="$LOG_DIR/serve-smoke-mtp-metrics-before-${SLURM_JOB_ID:-manual}.txt"
MTP_METRICS_AFTER="$LOG_DIR/serve-smoke-mtp-metrics-after-${SLURM_JOB_ID:-manual}.txt"
curl -sf "http://127.0.0.1:$PORT/metrics" > "$MTP_METRICS_BEFORE"

MTP_RESPONSE="$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",
         \"messages\":[{\"role\":\"user\",\"content\":\"List the integers from 1 through 128 in order, separated by commas.\"}],
         \"chat_template_kwargs\":{\"enable_thinking\":false},
         \"max_tokens\":512,
         \"temperature\":0}")"
MTP_COMPLETION_TOKENS="$(printf '%s' "$MTP_RESPONSE" | python3 -c "
import json, sys
response = json.load(sys.stdin)
print(int(response['usage']['completion_tokens']))
")"
if [[ "$MTP_COMPLETION_TOKENS" -lt 64 ]]; then
    echo "MTP probe generated only $MTP_COMPLETION_TOKENS tokens; need at least 64" >&2
    exit 1
fi

curl -sf "http://127.0.0.1:$PORT/metrics" > "$MTP_METRICS_AFTER"
python3 scripts/check_mtp_metrics.py \
    --before "$MTP_METRICS_BEFORE" \
    --after "$MTP_METRICS_AFTER" \
    --min-acceptance "$MTP_MIN_ACCEPTANCE"
echo "serve-smoke=ok modes=baseline,mtp mtp_completion_tokens=$MTP_COMPLETION_TOKENS"
