#!/usr/bin/env bash
# Submit a quantization run.
#
#   bash quant/scripts/submit_quantize.sh 8
#   bash quant/scripts/submit_quantize.sh 8 --fp8-gdn=true
#   bash quant/scripts/submit_quantize.sh 8 --fp8-gdn=true --algorithm=awq+gptq
#
# --algorithm chooses how weights are rounded once AWQ has scaled activations.
# awq rounds each weight to the nearest representable value on its own;
# awq+gptq pushes each column's rounding error into the columns not yet
# quantized, correcting the layer output instead. GPTQ costs roughly two to
# three times the runtime and writes to its own directory so both survive.
#
# --fp8-gdn selects what happens to the Gated DeltaNet input projections,
# in_proj_qkv and in_proj_z, which are 4.0B parameters and most of the
# checkpoint's size. false keeps them in source precision; true quantizes them
# with FP8_BLOCK, the scheme Qwen's own FP8 release applies to those same
# layers. Everything else is W4A16 either way.
#
# The two variants write to different directories so both can exist while both
# are evaluated.
set -euo pipefail

usage() {
    echo "usage: bash quant/scripts/submit_quantize.sh NGPUS [--fp8-gdn=true|false] [--output-dir=PATH]" >&2
    echo "                                      [--algorithm=awq|awq+gptq]" >&2
    echo "                                      [--dependency=SPEC] [--time=HH:MM:SS]" >&2
    exit 2
}

NGPUS="${1:-}"
[[ -n "$NGPUS" ]] || usage
shift
case "$NGPUS" in
    *[!0-9]* | 0) echo "NGPUS must be a positive integer, got: $NGPUS" >&2; exit 2 ;;
esac

FP8_GDN=false
ALGORITHM=awq
OUTPUT_DIR=""
DEPENDENCY=""
# The job header reserves 24h for a full prepare-and-quantize. A requant is
# well under an hour, and on a shared node the reservation is what other people
# see, so allow a shorter one.
TIME_LIMIT=""
for arg in "$@"; do
    case "$arg" in
        --dependency=*) DEPENDENCY="${arg#*=}" ;;
        --time=*) TIME_LIMIT="${arg#*=}" ;;
        --fp8-gdn=*)
            FP8_GDN="${arg#*=}"
            case "$FP8_GDN" in
                true | false) ;;
                *) echo "--fp8-gdn must be true or false, got: $FP8_GDN" >&2; exit 2 ;;
            esac
            ;;
        --algorithm=*)
            ALGORITHM="${arg#*=}"
            case "$ALGORITHM" in
                awq | awq+gptq) ;;
                *) echo "--algorithm must be awq or awq+gptq, got: $ALGORITHM" >&2; exit 2 ;;
            esac
            ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
        -h | --help) usage ;;
        *) echo "unknown argument: $arg" >&2; usage ;;
    esac
done

if [[ "$FP8_GDN" == "true" ]]; then
    GDN_PRECISION=fp8
    DEFAULT_SUFFIX="-fp8gdn"
else
    GDN_PRECISION=source
    DEFAULT_SUFFIX=""
fi
# The algorithm leaves config.json untouched, so without a distinct directory an
# AWQ+GPTQ build and an AWQ one differ only in their weights.
[[ "$ALGORITHM" == "awq+gptq" ]] && DEFAULT_SUFFIX="$DEFAULT_SUFFIX-gptq"

# Resolve the default output directory the same way the job would, so the two
# variants cannot overwrite each other by accident.
if [[ -z "$OUTPUT_DIR" ]]; then
    RUN_BASE_GUESS="${RUN_BASE:-/scratch/$USER/qwen38-27b-awq}"
    OUTPUT_DIR="$RUN_BASE_GUESS/v2/model$DEFAULT_SUFFIX"
fi

if [[ -e "$OUTPUT_DIR/config.json" ]]; then
    echo "refusing to overwrite an existing checkpoint at $OUTPUT_DIR" >&2
    echo "pass --output-dir=PATH to write elsewhere, or move it aside" >&2
    exit 3
fi

echo "gpus           : $NGPUS"
echo "gdn projections: $GDN_PRECISION"
echo "algorithm      : $ALGORITHM"
echo "output         : $OUTPUT_DIR"
[[ -n "$DEPENDENCY" ]] && echo "dependency     : $DEPENDENCY"
[[ -n "$TIME_LIMIT" ]] && echo "time limit     : $TIME_LIMIT"

exec sbatch \
    --gres="gpu:$NGPUS" \
    ${DEPENDENCY:+--dependency="$DEPENDENCY"} \
    ${TIME_LIMIT:+--time="$TIME_LIMIT"} \
    --export="ALL,GDN_PRECISION=$GDN_PRECISION,QUANT_ALGORITHM=$ALGORITHM,OUTPUT_DIR=$OUTPUT_DIR" \
    quant/slurm/quantize.sbatch
