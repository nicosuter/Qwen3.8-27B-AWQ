#!/usr/bin/env bash
#SBATCH --job-name=qwen38-serve-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=qwen38-serve-probe-%j.out

# Can this GPU serve this checkpoint at all? Starts vLLM, asks one question,
# stops. Deliberately makes no claim about quality.
#
# It exists for two questions the rest of the harness cannot answer:
#   - block-wise FP8 needs SM89+, so an FP8 checkpoint may simply not load on
#     Ampere, which decides whether a cluster can host a paired baseline
#   - a mixed W4A16 + FP8-block checkpoint is a combination this vLLM image has
#     never been asked to load
#
# A load failure is the result, not an error: it prints the reason and exits 0
# so the outcome is legible rather than a bare job failure.
#
#   PROBE_BIND=/path/to/dir PROBE_MODEL=/mnt/model[/subpath] sbatch slurm/serve-probe.sbatch
#
# PROBE_BIND is the host directory to mount. Point it at a Hugging Face
# repository root rather than a snapshot: a snapshot is a farm of symlinks into
# ../../blobs, and binding it alone leaves every one of them dangling.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./common/scripts/load_env.sh
: "${EVAL_APPTAINER_IMAGE:?set EVAL_APPTAINER_IMAGE to the pinned vLLM .sif}"
: "${PROBE_BIND:?set PROBE_BIND to the host directory to mount}"
: "${PROBE_MODEL:?set PROBE_MODEL to the model path as seen inside the container}"

# See paired-suite-eval.sbatch: a leaked host CC breaks Triton's startup JIT.
unset CC CXX

IMAGE="$EVAL_APPTAINER_IMAGE"
SANDBOX="${EVAL_APPTAINER_SANDBOX:-${EVAL_APPTAINER_IMAGE%.sif}-sandbox}"
[[ -d "$SANDBOX" ]] && IMAGE="$SANDBOX"

MAX_LEN="${PROBE_MAX_LEN:-8192}"
PORT="$(( 8000 + (SLURM_JOB_ID % 1000) ))"
BASE_URL="http://127.0.0.1:$PORT/v1"
LOG="${PROBE_LOG:-$RUN_BASE/v2/logs/probe-${SLURM_JOB_ID:-manual}.log}"
mkdir -p "$(dirname "$LOG")"

echo "gpu      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "bind     : $PROBE_BIND"
echo "model    : $PROBE_MODEL"
echo "image    : $IMAGE"

apptainer run --nv \
    --bind "$PROBE_BIND:/mnt/model:ro" \
    "$IMAGE" \
    "$PROBE_MODEL" \
    --served-model-name probe \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "${PROBE_TP:-1}" \
    --max-model-len "$MAX_LEN" \
    --kv-cache-dtype auto \
    > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

deadline=$(( SECONDS + ${PROBE_TIMEOUT:-1500} ))
until curl -sf "$BASE_URL/models" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "probe=load-failed"
        # The actionable line is usually a capability assertion rather than the
        # final traceback frame, so surface both.
        grep -iE "not supported|requires|capability|compute cap|sm[0-9]{2}|no kernel|unsupported" \
            "$LOG" | tail -15 || true
        echo "--- tail ---"
        tail -25 "$LOG"
        exit 0
    fi
    if (( SECONDS > deadline )); then
        echo "probe=timeout after ${PROBE_TIMEOUT:-1500}s"
        tail -25 "$LOG"
        exit 0
    fi
    sleep 5
done

echo "probe=served"
curl -sf "$BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"probe","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":2048,"temperature":0}' \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
choice = d['choices'][0]
print('finish_reason :', choice['finish_reason'])
print('content       :', repr(choice['message']['content'])[:200])
print('completion_tok:', d['usage']['completion_tokens'])
"
echo "probe=ok"
