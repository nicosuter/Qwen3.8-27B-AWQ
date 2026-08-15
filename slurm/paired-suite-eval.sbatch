#!/usr/bin/env bash
#SBATCH --job-name=qwen38-paired-suites
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:4
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

# Apptainer passes the host environment through, and one cluster's login profile
# exports CC/CXX pointing at a gcc that exists on neither the host nor the
# container. Triton honours CC when it JIT-compiles at startup, fails, and vLLM
# reports it as "Model architectures [...] failed to be inspected", which says
# nothing about the real cause. Let Triton find the container's own compiler.
unset CC CXX

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
# Four, not eight: the node is shared and the rest is somebody else's.
DP="${PAIRED_DP:-4}"
# The header's --gres and this DP are set independently, so they can disagree.
# vLLM would then either idle GPUs or fail deep into startup, after the image
# and weights are already loaded. Check against what the cgroup actually grants.
VISIBLE_GPUS="$(nvidia-smi -L | wc -l)"
if (( DP > VISIBLE_GPUS )); then
    echo "data-parallel size $DP exceeds the $VISIBLE_GPUS GPUs this job was granted" >&2
    echo "raise --gres in the sbatch header, or lower PAIRED_DP" >&2
    exit 1
fi
echo "gpus=$VISIBLE_GPUS data-parallel=$DP"

# Request timeouts in the config are written for the reference card (H200). A
# slower one needs proportionally longer, or the token cap stops being what
# bounds a reply and the clock starts scoring items zero: on A100 five matharena
# items stopped at exactly 4200s having produced no output at all, which
# measures the card and not the checkpoint. Measured per-stream decode was
# 51.8 tok/s on H200 against 25.3 on A100, so H200's headroom only survives at
# roughly twice the timeout. An over-generous timeout costs nothing unless a
# request genuinely hangs, so these round upwards.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "$GPU_NAME" in
    *H200*|*B200*|*GB200*) TIMEOUT_SCALE_DEFAULT=1.0 ;;
    *H100*)                TIMEOUT_SCALE_DEFAULT=1.5 ;;
    *A100*)                TIMEOUT_SCALE_DEFAULT=2.5 ;;
    *)                     TIMEOUT_SCALE_DEFAULT=2.5 ;;   # unknown card, assume slow
esac
TIMEOUT_SCALE="${PAIRED_TIMEOUT_SCALE:-$TIMEOUT_SCALE_DEFAULT}"
echo "gpu=${GPU_NAME:-unknown} timeout-scale=$TIMEOUT_SCALE"
# Every suite is one wave of requests trailed by a long tail: gpqa averaged 37 of
# 192 streams busy because a single 100k-token reply outlived the other 197, and
# multimodal spent 485 of its 502 seconds on one runaway item. Scoring jobs
# therefore run concurrently, so one job's tail is filled by another's body.
# Replicates fill it best -- the comparator already averages them per item, and
# they are what the recovery interval actually needs.
REPLICATES="${PAIRED_REPLICATES:-1}"
PARALLEL="${PAIRED_PARALLEL:-0}"   # 0 means "as many lanes as there are jobs"
CONCURRENCY="${PAIRED_CONCURRENCY:-}"
# Default to scaling each suite's configured concurrency by how much of the
# protocol's eight-replica server this job actually has. One absolute number
# across suites is wrong: RULER holds about six percent of the KV pool per
# request where a short prompt holds one.
CONCURRENCY_SCALE="${PAIRED_CONCURRENCY_SCALE:-$(python3 -c "print($DP/8)")}"
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

# Several checkpoints sit side by side under v2/ and OUTPUT_DIR has a default, so
# a submit that forgets to set it evaluates the wrong one and says nothing. Name
# what is actually about to be served.
echo "candidate=$OUTPUT_DIR"
python3 -c "
import json, sys
q = json.load(open(sys.argv[1] + '/config.json')).get('quantization_config') or {}
print('candidate-quant format=%s groups=%d ignored=%d' % (
    q.get('format'), len(q.get('config_groups') or {}), len(q.get('ignore') or [])))
" "$OUTPUT_DIR"

# Scoring somebody else's quantization is the same paired measurement with a
# different candidate, so the baseline half is identical work. Seed it from a
# finished run rather than buying it again. The inherited baseline must have been
# scored on THIS cluster: A100 dequantizes the FP8 baseline where H200 runs it
# natively, and a comparison across that boundary is not paired.
if [[ -n "${PAIRED_INHERIT_BASELINE_FROM:-}" ]]; then
    INHERIT="$PAIRED_INHERIT_BASELINE_FROM"
    test -d "$INHERIT" || { echo "PAIRED_INHERIT_BASELINE_FROM=$INHERIT not found" >&2; exit 1; }
    mkdir -p "$RUN_DIR"/{orders,materialized,metadata,raw/baseline}
    # -n so a rerun never clobbers results this job already produced.
    cp -n "$INHERIT"/orders/*.json          "$RUN_DIR/orders/"        2>/dev/null || true
    cp -n "$INHERIT"/materialized/*         "$RUN_DIR/materialized/"  2>/dev/null || true
    cp -n "$INHERIT"/raw/baseline/*.jsonl   "$RUN_DIR/raw/baseline/"  2>/dev/null || true
    cp -n "$INHERIT"/metadata/*-baseline-r*.json "$RUN_DIR/metadata/" 2>/dev/null || true
    echo "inherited baseline and frozen order from $INHERIT"
fi

for suite in $SUITES; do
    test -s "$RUN_DIR/orders/$suite.json"
done

SERVER_PID=""
SAMPLER_PID=""
stop_sampler() {
    if [[ -n "${SAMPLER_PID:-}" ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
    fi
    SAMPLER_PID=""
}
stop_server() {
    stop_sampler
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
    # Record what the scheduler was actually doing, rather than inferring
    # occupancy afterwards from how long each request took.
    "$PYTHON" scripts/sample_telemetry.py \
        --metrics-url "http://127.0.0.1:$PORT/metrics" \
        --output "$RUN_DIR/telemetry/$variant-${SLURM_JOB_ID:-manual}.jsonl" \
        --interval 10 &
    SAMPLER_PID=$!
}

variant_is_scored() {
    local variant="$1" rep
    for suite in $SUITES; do
        for (( rep = 0; rep < REPLICATES; rep++ )); do
            suite_is_current "$variant" "$suite" "$rep" || return 1
        done
    done
    return 0
}

# Results already scored under the same generation policy are reusable: pairing
# is per item, not per process, so a baseline measured in an earlier job is as
# valid as one measured here. What is NOT reusable is a result taken under a
# different cap or timeout, which changes what was measured -- reusing a
# 16384-cap run against a 65536 config would invent an improvement. Concurrency
# is deliberately not compared: it changes scheduling, not the distribution.
suite_is_current() {
    local variant="$1" suite="$2" rep="${3:-0}"
    local out="$RUN_DIR/raw/$variant/$suite-r$rep.jsonl"
    local meta="$RUN_DIR/metadata/$suite-$variant-r$rep.json"
    [[ -s "$out" && -s "$meta" ]] || return 1
    "$PYTHON" - "$CONFIG" "$suite" "$meta" "$TIMEOUT_SCALE" <<'PY'
import json, sys

config_path, suite, meta_path, timeout_scale = sys.argv[1:5]
run = next(s["run"] for s in json.load(open(config_path))["suites"] if s["name"] == suite)
recorded = json.load(open(meta_path))

# The cap decides what was measured, so it has to match exactly.
if "--max-tokens" in run:
    want = float(run[run.index("--max-tokens") + 1])
    have = recorded.get("max_tokens")
    if have is None or float(have) != want:
        print(f"{suite}: --max-tokens config {want}, recorded {have}", file=sys.stderr)
        sys.exit(1)

# The timeout only decides what was measured when it actually fired. A run under
# a shorter timeout that never hit it produced exactly the results a longer one
# would have, so it stays reusable; a run that timed out did not.
if "--request-timeout" in run:
    want = float(run[run.index("--request-timeout") + 1]) * float(timeout_scale)
    have = recorded.get("request_timeout_seconds")
    fired = recorded.get("timeouts")
    if have is None or (float(have) < want and fired != 0):
        print(
            f"{suite}: --request-timeout wants {want:.0f}s, recorded {have}"
            f" with timeouts={fired}",
            file=sys.stderr,
        )
        sys.exit(1)
sys.exit(0)
PY
}

# Lanes are counted by pid rather than with `jobs -r`, because the telemetry
# sampler is also a background job and would otherwise occupy a lane forever.
count_alive() {
    local pid alive=0
    for pid in "$@"; do
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=$(( alive + 1 ))
    done
    echo "$alive"
}

score_variant() {
    local variant="$1" suite rep job label log rc failed=0
    echo "=== generation policy probe ($variant) ==="
    "$PYTHON" scripts/adapters/gpqa_diamond.py probe --base-url "$BASE_URL" --model "$SERVED_NAME"

    local pending=() lane_pids=()
    for suite in $SUITES; do
        for (( rep = 0; rep < REPLICATES; rep++ )); do
            if [[ "${PAIRED_FORCE:-0}" != "1" ]] && suite_is_current "$variant" "$suite" "$rep"; then
                echo "=== $suite r$rep ($variant) already scored under this policy; reusing ==="
                continue
            fi
            pending+=("$suite:$rep")
        done
    done
    (( ${#pending[@]} > 0 )) || return 0

    local lanes=${#pending[@]}
    if (( PARALLEL > 0 && PARALLEL < lanes )); then
        lanes=$PARALLEL
    fi
    # Divide the request budget across lanes, so the server holds the same number
    # in flight as a sequential run would. The win comes from item supply keeping
    # the batch full, not from asking the server to carry more at once.
    local lane_conc="" lane_scale=""
    if [[ -n "$CONCURRENCY" ]]; then
        lane_conc=$(( CONCURRENCY / lanes ))
        (( lane_conc > 0 )) || lane_conc=1
    elif [[ -n "$CONCURRENCY_SCALE" ]]; then
        lane_scale="$(python3 -c "print($CONCURRENCY_SCALE/$lanes)")"
    fi

    local lane_dir="$RUN_DIR/logs/lanes-$variant"
    mkdir -p "$lane_dir"
    echo "scoring ${#pending[@]} job(s) for $variant across $lanes lane(s); live logs in $lane_dir"

    for job in "${pending[@]}"; do
        suite="${job%%:*}"
        rep="${job##*:}"
        label="$suite-r$rep"
        rm -f "$lane_dir/$label.rc"
        # Hold at `lanes` running jobs. `wait -n` returns the exit status of
        # whichever finished, which is why each job records its own into a file.
        while (( $(count_alive "${lane_pids[@]}") >= lanes )); do wait -n || true; done
        (
            "$PYTHON" scripts/run_adapter_suite.py \
                --config "$CONFIG" --suite "$suite" --run-dir "$RUN_DIR" \
                --phase run --variant "$variant" --replicate "$rep" \
                --base-url "$BASE_URL" \
                --request-timeout-scale "$TIMEOUT_SCALE" \
                ${lane_conc:+--concurrency "$lane_conc"} \
                ${lane_conc:-${lane_scale:+--concurrency-scale "$lane_scale"}} \
                > "$lane_dir/$label.log" 2>&1
            echo $? > "$lane_dir/$label.rc"
        ) &
        lane_pids+=("$!")
    done
    # Named explicitly: a bare `wait` would also wait on the sampler, which
    # never exits.
    (( ${#lane_pids[@]} == 0 )) || wait "${lane_pids[@]}" 2>/dev/null || true

    # Replay the lanes in a fixed order; concurrent stdout would be unreadable.
    for job in "${pending[@]}"; do
        suite="${job%%:*}"
        rep="${job##*:}"
        label="$suite-r$rep"
        echo "=== $label ($variant) ==="
        cat "$lane_dir/$label.log" 2>/dev/null || true
        rc="$(cat "$lane_dir/$label.rc" 2>/dev/null || echo 1)"
        if [[ "$rc" != "0" ]]; then
            echo "$label ($variant) failed with exit $rc" >&2
            failed=1
        fi
    done
    return "$failed"
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
