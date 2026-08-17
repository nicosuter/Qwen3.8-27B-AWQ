#!/usr/bin/env bash
#SBATCH --job-name=qwen38-paired-suites
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
# %x is the job name, so pass --job-name=<what is being scored>: several
# checkpoints share this script and "qwen38-paired-suites-4619.out" says nothing
# about which one it measured. The directory must already exist or slurm drops
# the job at submit time.
#SBATCH --output=slurm-logs/%x-%j.out

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
CONFIG="${PAIRED_CONFIG:-$PROJECT_DIR/eval/eval-suite-v1.json}"
# A batch selects from the protocol; scripts/eval_suite.py refuses a name the
# definition does not contain. Empty means every suite in it. Resolving here
# rather than trusting the string is the point: a typo, or a suite the protocol
# never contained, used to reach the comparator and enter the macro silently.
SUITES="${PAIRED_SUITES:-}"
EVAL_SUITE_VERSION="${PAIRED_EVAL_SUITE:-v1}"
# PAIRED_BATCH names a plan entry from eval/batches.json, which is the recorded
# way to run part of the protocol. PAIRED_SUITES stays for one-off lists.
BATCH="${PAIRED_BATCH:-}"
if [[ -n "$BATCH" ]]; then
    if [[ -n "$SUITES" ]]; then
        echo "set PAIRED_BATCH or PAIRED_SUITES, not both" >&2
        exit 1
    fi
    SUITES="$("${EVAL_PYTHON:-python3}" scripts/eval_suite.py --version "$EVAL_SUITE_VERSION" --batch "$BATCH")"
    # A batch carries more than a suite list. The shakedown is fifty items of one
    # arm, and a plan that recorded that without the runner acting on it would
    # have quietly scored all 1730.
    BATCH_LIMIT="$("${EVAL_PYTHON:-python3}" scripts/eval_suite.py --version "$EVAL_SUITE_VERSION" --batch-field "$BATCH" limit)"
    BATCH_VARIANTS="$("${EVAL_PYTHON:-python3}" scripts/eval_suite.py --version "$EVAL_SUITE_VERSION" --batch-field "$BATCH" variants)"
    [[ -n "$BATCH_LIMIT" && -z "${PAIRED_LIMIT:-}" ]] && PAIRED_LIMIT="$BATCH_LIMIT"
    [[ -n "$BATCH_VARIANTS" && -z "${PAIRED_VARIANTS:-}" ]] && PAIRED_VARIANTS="$BATCH_VARIANTS"
    echo "batch=$BATCH limit=${PAIRED_LIMIT:-none} variants=${PAIRED_VARIANTS:-both}"
fi
if [[ -z "$SUITES" ]]; then
    SUITES="$("${EVAL_PYTHON:-python3}" scripts/eval_suite.py --version "$EVAL_SUITE_VERSION" \
        | awk '/^  /{print $1}' | tr '\n' ' ')"
else
    SUITES="$("${EVAL_PYTHON:-python3}" scripts/eval_suite.py --version "$EVAL_SUITE_VERSION" --select $SUITES)"
fi
echo "eval-suite=$EVAL_SUITE_VERSION suites=$SUITES"
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
# An unrecognised card is refused rather than guessed at. Slurm here advertises
# every node as plain "gpu:4" with no type and no feature to select on, so a job
# asking for four GPUs can land on V100s: sm_70 has no Marlin kernels for W4A16,
# no bf16, and 32GB per card against a 27.7GB checkpoint leaves nothing for KV.
# That produces either a crash or numbers not comparable to the A100 runs, and
# the second is worse.
case "$GPU_NAME" in
    *H200*|*B200*|*GB200*) TIMEOUT_SCALE_DEFAULT=1.0 ;;
    *H100*)                TIMEOUT_SCALE_DEFAULT=1.5 ;;
    *A100*)                TIMEOUT_SCALE_DEFAULT=2.5 ;;
    *)
        if [[ "${PAIRED_ALLOW_ANY_GPU:-0}" != "1" ]]; then
            echo "unvalidated GPU: ${GPU_NAME:-none detected}" >&2
            echo "known good: H200, H100, A100, B200, GB200. Pin the node with" >&2
            echo "--nodelist, or set PAIRED_ALLOW_ANY_GPU=1 to score on it anyway" >&2
            exit 1
        fi
        TIMEOUT_SCALE_DEFAULT=4.0
        ;;
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
# Both arms in one job serialises two server loads onto one allocation. Naming a
# single variant lets them run on separate nodes against the same run directory,
# which is safe because the two write disjoint paths: raw/$variant and
# metadata/*-$variant-r*. Whichever finishes second produces the comparison.
VARIANTS="${PAIRED_VARIANTS:-baseline candidate}"
for want in $VARIANTS; do
    case "$want" in
        baseline|candidate) ;;
        *) echo "PAIRED_VARIANTS must be 'baseline', 'candidate' or both; got $want" >&2
           exit 1 ;;
    esac
done
variant_requested() {
    local want="$1" v
    for v in $VARIANTS; do [[ "$v" == "$want" ]] && return 0; done
    return 1
}
# Which harness scores a lane. `adapter` is ours; `evalscope` runs the upstream
# task definitions against the same pinned data and converts its per-item output
# into the same rows. The server lifecycle, lane fan-out, reuse check and
# comparison are identical either way.
RUNNER="${PAIRED_RUNNER:-adapter}"
case "$RUNNER" in
    adapter|evalscope) ;;
    *) echo "PAIRED_RUNNER must be 'adapter' or 'evalscope'; got $RUNNER" >&2; exit 1 ;;
esac
ES_SUITES="${PAIRED_ES_SUITES:-eval/evalscope-suites.json}"
# Read from the config so both runners derive replicate seeds from the same
# number, and a config edit cannot leave them disagreeing.
ORDER_SEED="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['order_seed'])" "$CONFIG")"
# One materialized copy for every lane and both variants. Downloading per lane
# would fetch the same pinned dataset dozens of times.
ES_DATASETS="${PAIRED_ES_DATASETS:-$RUN_BASE/eval-materialized/evalscope}"
ES_MAX_TOKENS="${PAIRED_ES_MAX_TOKENS:-131072}"
ES_REQUEST_TIMEOUT="${PAIRED_ES_REQUEST_TIMEOUT:-5400}"
# EvalScope's eval_batch_size defaults to 1, one request in flight. Divided
# across lanes below, exactly as the adapter path divides its own concurrency.
ES_CONCURRENCY="${PAIRED_ES_CONCURRENCY:-256}"
PARALLEL="${PAIRED_PARALLEL:-0}"   # 0 means "as many lanes as there are jobs"
CONCURRENCY="${PAIRED_CONCURRENCY:-}"
# Default to scaling each suite's configured concurrency by how much of the
# protocol's eight-replica server this job actually has. One absolute number
# across suites is wrong: RULER holds about six percent of the KV pool per
# request where a short prompt holds one.
CONCURRENCY_SCALE="${PAIRED_CONCURRENCY_SCALE:-$(python3 -c "print($DP/8)")}"
PYTHON="${EVAL_VENV:-$PROJECT_DIR/.venv}/bin/python"
# One stamp for the whole job, so everything it supersedes or emits sorts
# together and can be attributed to a single run.
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
# Suite commands in the eval config name their deployment-specific paths as
# ${EVAL_*} rather than spelling them out, so the config is a tracked, portable
# artifact instead of a hand-edited file that exists only on one cluster. The
# runner expands these fail-closed, so an unset one stops the run.
export EVAL_PYTHON="$PYTHON"
export EVAL_CORPUS="${PAIRED_CORPUS:-$RUN_BASE/eval/haystack.txt}"
export EVAL_TOKENIZER="${PAIRED_TOKENIZER:-$OUTPUT_DIR}"
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

# Both variants are served under one --served-model-name so no harness behavior
# can branch on which model is loaded, which also means nothing in a result says
# which weights produced it. Several checkpoints sit side by side under v2/ and
# OUTPUT_DIR has a default, so a submit that forgets to set it silently scores
# the wrong one. Fingerprint both now, print them, and keep them with the run.
BASELINE_INFO="$("$PYTHON" scripts/checkpoint_fingerprint.py "$BASELINE_DIR" --label baseline)"
CANDIDATE_INFO="$("$PYTHON" scripts/checkpoint_fingerprint.py "$OUTPUT_DIR" --label candidate)"
mkdir -p "$RUN_DIR"
printf '%s\n%s\n' "$BASELINE_INFO" "$CANDIDATE_INFO" > "$RUN_DIR/checkpoints.jsonl"
echo "baseline-checkpoint=$BASELINE_INFO"
echo "candidate-checkpoint=$CANDIDATE_INFO"

# The checkpoint is fingerprinted but the code that scored it was not, and this
# harness decides lane structure, timeouts, reuse and the gate arithmetic. A
# result identified by weights alone is only half identified. submit-paired.sh
# checks out a commit and exports these; a hand-run job records what it can.
CODE_COMMIT="${EVAL_CODE_COMMIT:-$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)}"
CODE_DIRTY="${EVAL_CODE_DIRTY:-unknown}"
if [[ "$CODE_DIRTY" == "unknown" ]] && git -C "$PROJECT_DIR" rev-parse HEAD >/dev/null 2>&1; then
    if git -C "$PROJECT_DIR" diff --quiet HEAD 2>/dev/null; then CODE_DIRTY=false; else CODE_DIRTY=true; fi
fi
CODE_INFO="$("$PYTHON" - "$CODE_COMMIT" "$CODE_DIRTY" "$PROJECT_DIR" "${EVAL_CODE_REMOTE:-}" <<'PY'
import json, sys
commit, dirty, project, remote = sys.argv[1:5]
print(json.dumps({
    "commit": commit,
    "dirty": {"true": True, "false": False}.get(dirty),
    "remote": remote or None,
    "checkout": project,
}))
PY
)"
printf '%s\n' "$CODE_INFO" > "$RUN_DIR/code.json"
echo "code=$CODE_INFO"
if [[ "$CODE_COMMIT" == "unknown" || "$CODE_DIRTY" == "true" ]]; then
    echo "warning: results will not be attributable to a published commit" >&2
    if [[ "${PAIRED_REQUIRE_PINNED_CODE:-0}" == "1" ]]; then
        echo "PAIRED_REQUIRE_PINNED_CODE=1 and the checkout is not pinned" >&2
        exit 1
    fi
fi
BASELINE_FP="$(printf '%s' "$BASELINE_INFO" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["fingerprint"])')"
CANDIDATE_FP="$(printf '%s' "$CANDIDATE_INFO" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["fingerprint"])')"

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
    local want_fp
    if [[ "$variant" == "baseline" ]]; then want_fp="$BASELINE_FP"; else want_fp="$CANDIDATE_FP"; fi
    "$PYTHON" - "$CONFIG" "$suite" "$meta" "$TIMEOUT_SCALE" "$want_fp" <<'PY'
import json, sys

config_path, suite, meta_path, timeout_scale, want_fingerprint = sys.argv[1:6]
run = next(s["run"] for s in json.load(open(config_path))["suites"] if s["name"] == suite)
recorded = json.load(open(meta_path))

# Reuse across runs is the point of this check, and it is also how results from
# one checkpoint end up standing in for another: the run directory does not
# change when OUTPUT_DIR does, and an inherited baseline may have come from
# another cluster entirely. Results whose provenance cannot be established are
# not reusable, which includes anything scored before this was recorded.
have_fingerprint = (recorded.get("checkpoint") or {}).get("fingerprint")
if have_fingerprint != want_fingerprint:
    print(
        f"{suite}: scored by checkpoint {have_fingerprint}, serving {want_fingerprint}",
        file=sys.stderr,
    )
    sys.exit(1)

# The cap decides what was measured, so it has to match exactly.
#
# It is tempting to relax this the way the timeout check below is relaxed -- a
# cap no item reached did not bind, so why rescore? Two reasons. Removing
# max_tokens changes admission control and KV preallocation, so it changes batch
# composition and therefore the order of floating-point reductions; the results
# are the same in distribution but not identical, and the timeout argument is an
# identity argument. And "no item reached it" is a property of one draw rather
# than of the distribution: MathArena's p90 output length sat exactly at the
# 131072 cap, so keeping the draws that happened to fit and rescoring the ones
# that did not would select on the outcome, biasing what is kept toward shorter
# reasoning -- the axis the two arms actually differ on.
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

# A rescored suite replaces a result that is still evidence. Two draws of the
# same comparison differing by more than their interval is the most direct
# measure of reproducibility this protocol produces, and it only exists if the
# earlier draw survives. Move it aside instead of letting the adapter write over
# it; raw and metadata travel together so the pair stays interpretable.
archive_existing() {
    local variant="$1" suite="$2" rep="$3"
    local out="$RUN_DIR/raw/$variant/$suite-r$rep.jsonl"
    local meta="$RUN_DIR/metadata/$suite-$variant-r$rep.json"
    [[ -s "$out" || -s "$meta" ]] || return 0
    local dest="$RUN_DIR/superseded/$RUN_STAMP-job${SLURM_JOB_ID:-manual}"
    mkdir -p "$dest/raw/$variant" "$dest/metadata"
    [[ -s "$out" ]] && mv "$out" "$dest/raw/$variant/"
    [[ -s "$meta" ]] && mv "$meta" "$dest/metadata/"
    echo "superseded $suite r$rep ($variant); previous result kept in $dest"
    return 0
}

# Suites are independent measurements, so one that fails is dropped from the
# comparison rather than taking the whole run with it -- the GPU hours the others
# already spent are worth keeping. A dropped suite is removed from BOTH variants:
# the comparator pairs per item and would otherwise weigh a suite that only one
# side scored. If nothing survives there is nothing to compare, and that is fatal.
FAILURE_LOG="$RUN_DIR/logs/suite-failures.tsv"
FAILED_SUITES=()

suite_failed() {
    local suite="$1" name
    for name in ${FAILED_SUITES[@]+"${FAILED_SUITES[@]}"}; do
        [[ "$name" == "$suite" ]] && return 0
    done
    return 1
}

record_failure() {
    local variant="$1" suite="$2" rep="$3" rc="$4" log="$5"
    suite_failed "$suite" || FAILED_SUITES+=("$suite")
    mkdir -p "$(dirname "$FAILURE_LOG")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Iseconds)" "$variant" "$suite" "$rep" "$rc" "$log" >> "$FAILURE_LOG"
}

usable_suites() {
    local suite out=()
    for suite in $SUITES; do
        suite_failed "$suite" || out+=("$suite")
    done
    echo "${out[@]+${out[@]}}"
}

report_failures() {
    (( ${#FAILED_SUITES[@]} > 0 )) || return 0
    echo "=== suite failures ===" >&2
    cat "$FAILURE_LOG" >&2
    echo "dropped from the comparison: ${FAILED_SUITES[*]}" >&2
    echo "kept: $(usable_suites)" >&2
}

# Called after each variant: once every suite is dead the remaining work is pure
# waste, and serving the second checkpoint would burn hours to produce nothing.
require_usable_suites() {
    if [[ -z "$(usable_suites)" ]]; then
        echo "every suite failed; nothing left to compare" >&2
        report_failures
        exit 1
    fi
}

score_variant() {
    local variant="$1" suite rep job label log rc
    echo "=== generation policy probe ($variant) ==="
    "$PYTHON" scripts/adapters/gpqa_diamond.py probe --base-url "$BASE_URL" --model "$SERVED_NAME"
    # Travels with every result this variant produces, so the checkpoint is
    # identifiable from the run rather than from a job log that gets overwritten.
    if [[ "$variant" == "baseline" ]]; then
        export EVAL_CHECKPOINT_JSON="$BASELINE_INFO"
    else
        export EVAL_CHECKPOINT_JSON="$CANDIDATE_INFO"
    fi

    local pending=() lane_pids=()
    for suite in $SUITES; do
        if suite_failed "$suite"; then
            echo "=== $suite ($variant) skipped; it already failed on the other variant ==="
            continue
        fi
        for (( rep = 0; rep < REPLICATES; rep++ )); do
            if [[ "${PAIRED_FORCE:-0}" != "1" ]] && suite_is_current "$variant" "$suite" "$rep"; then
                echo "=== $suite r$rep ($variant) already scored under this policy; reusing ==="
                continue
            fi
            archive_existing "$variant" "$suite" "$rep"
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

    # Same budget split as the adapter lanes: the win comes from item supply
    # keeping the batch full, not from asking the server to carry more at once.
    local es_lane_conc=$(( ${ES_CONCURRENCY:-256} / lanes ))
    (( es_lane_conc > 0 )) || es_lane_conc=1

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
            if [[ "${RUNNER:-adapter}" == "evalscope" ]]; then
                # EvalScope owns the task definitions; we keep the server
                # lifecycle, the lane fan-out and the comparison. The lane
                # writes the same two files an adapter lane does, so nothing
                # downstream of here knows which runner produced them.
                "$PYTHON" scripts/evalscope_bridge.py run \
                    --suites "${ES_SUITES:-eval/evalscope-suites.json}" --suite "$suite" \
                    --variant "$variant" --replicate "$rep" \
                    --model "$SERVED_NAME" --api-url "$BASE_URL" \
                    --work-dir "$RUN_DIR/evalscope/$variant/$suite-r$rep" \
                    --datasets-root "${ES_DATASETS:-${RUN_BASE:-.}/eval-materialized/evalscope}" \
                    --results "$RUN_DIR/raw/$variant/$suite-r$rep.jsonl" \
                    --metadata "$RUN_DIR/metadata/$suite-$variant-r$rep.json" \
                    --max-tokens "${ES_MAX_TOKENS:-131072}" \
                    --concurrency "$es_lane_conc" \
                    --request-timeout "$(python3 -c "print(${ES_REQUEST_TIMEOUT:-5400}*$TIMEOUT_SCALE)")" \
                    --seed "$(( ${ORDER_SEED:-0} + rep ))" \
                    > "$lane_dir/$label.log" 2>&1 && rc=0 || rc=$?
            else
            "$PYTHON" scripts/run_adapter_suite.py \
                --config "$CONFIG" --suite "$suite" --run-dir "$RUN_DIR" \
                --phase run --variant "$variant" --replicate "$rep" \
                --base-url "$BASE_URL" \
                --request-timeout-scale "$TIMEOUT_SCALE" \
                ${PAIRED_LIMIT:+--limit "$PAIRED_LIMIT"} \
                ${lane_conc:+--concurrency "$lane_conc"} \
                ${lane_conc:-${lane_scale:+--concurrency-scale "$lane_scale"}} \
                > "$lane_dir/$label.log" 2>&1 && rc=0 || rc=$?
            fi
            echo "$rc" > "$lane_dir/$label.rc"
            # Reported twice on purpose: here so the failure is visible the
            # moment it happens, and again in the ordered replay below, because
            # concurrent lanes finish in an order nobody can read.
            (( rc == 0 )) || echo "[$(date +%H:%M:%S)] $label ($variant) exited $rc;" \
                "see $lane_dir/$label.log" >&2
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
            echo "$label ($variant) failed with exit $rc; $suite is dropped from the comparison" >&2
            record_failure "$variant" "$suite" "$rep" "$rc" "$lane_dir/$label.log"
        fi
    done
    # Always 0: which suites survived is decided once, after both variants, so a
    # failure here cannot abort the run through `set -e`.
    return 0
}

# A cancelled run leaves one side already scored. Reusing it costs nothing in
# validity as long as the flags and image are the same, and it keeps the node
# for half as long.
if ! variant_requested baseline; then
    echo "baseline not requested by PAIRED_VARIANTS=$VARIANTS; skipping"
elif [[ "${PAIRED_FORCE:-0}" != "1" ]] && variant_is_scored baseline; then
    echo "baseline already scored in $RUN_DIR/raw/baseline; skipping"
else
    start_server baseline "$BASELINE_REPO" "/mnt/model/snapshots/$BASELINE_REVISION"
    score_variant baseline
    stop_server
    require_usable_suites
fi

# Somebody else's quantization arrives through the Hugging Face cache, and a
# snapshot directory there is a farm of relative symlinks into ../../blobs.
# Binding the snapshot alone leaves every one of them dangling inside the
# container, config.json included, which vLLM reports as "Invalid repository ID
# or local directory" rather than as a broken mount -- an error that sends you
# looking at the pin instead of the bind. The baseline has always been bound
# through its repository root for this reason; a candidate needs the same
# treatment. A checkpoint we produced ourselves is a real directory and is bound
# as it is.
candidate_bind() {
    CANDIDATE_BIND="$1"
    CANDIDATE_MOUNT="/mnt/model"
    case "$1" in
        */snapshots/*)
            CANDIDATE_BIND="${1%/snapshots/*}"
            CANDIDATE_MOUNT="/mnt/model/snapshots/${1##*/snapshots/}"
            ;;
    esac
}
candidate_bind "$OUTPUT_DIR"
[[ "$CANDIDATE_MOUNT" == "/mnt/model" ]] ||
    echo "candidate is a cache snapshot; binding $CANDIDATE_BIND at /mnt/model"

if ! variant_requested candidate; then
    echo "candidate not requested by PAIRED_VARIANTS=$VARIANTS; skipping"
elif [[ "${PAIRED_FORCE:-0}" != "1" ]] && variant_is_scored candidate; then
    echo "candidate already scored in $RUN_DIR/raw/candidate; skipping"
else
    start_server candidate "$CANDIDATE_BIND" "$CANDIDATE_MOUNT"
    score_variant candidate
    stop_server
    require_usable_suites
fi

report_failures

# The concatenation and the comparator call live in scripts/compare_run_dir.sh
# so that a job with no GPU can also produce a comparison: LiveCodeBench defers
# execution, and its verdict is produced later on a CPU node. Two copies of the
# call would mean two copies of the baseline floors.
COMPARISON="$(
    RUN_DIR="$RUN_DIR" PYTHON="$PYTHON" \
    EVAL_SUITE_VERSION="$EVAL_SUITE_VERSION" RUN_STAMP="$RUN_STAMP" \
    COMPARISON_TAG="${SLURM_JOB_ID:-manual}" \
    EXCLUDE_SUITES="${FAILED_SUITES[*]-}" \
    ALLOW_PARTIAL="$( [[ -n "${PAIRED_SUITES:-}${PAIRED_BATCH:-}" ]] && echo 1 || echo 0 )" \
    ./scripts/compare_run_dir.sh | tee /dev/stderr | sed -n 's/^comparison=//p'
)"

echo "paired-eval=complete results=$RUN_DIR comparison=$COMPARISON"
