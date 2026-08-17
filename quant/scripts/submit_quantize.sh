#!/usr/bin/env bash
# Submit a quantization run.
#
#   bash quant/scripts/submit_quantize.sh 8
#   bash quant/scripts/submit_quantize.sh 8 --gdn=int4
#   bash quant/scripts/submit_quantize.sh 8 --gdn=fp8 --algorithm=awq+gptq
#
# --algorithm chooses how weights are rounded once AWQ has scaled activations.
# awq rounds each weight to the nearest representable value on its own;
# awq+gptq pushes each column's rounding error into the columns not yet
# quantized, correcting the layer output instead. GPTQ costs roughly two to
# three times the runtime and writes to its own directory so both survive.
#
# --gdn selects what happens to the Gated DeltaNet input projections,
# in_proj_qkv and in_proj_z, which are 4.0B parameters and most of the
# checkpoint's size. Everything else is W4A16 in every mode. It used to be a
# --fp8-gdn boolean, which could only express the two ends of what turned out to
# be a four-way choice: the useful modes are int4, which is what the third-party
# releases that quantize this path at all use, and fp8-a16, which is what the
# serving hardware executes when handed the fp8 build. See gdn_precision.py.
#
# Every build writes to its own directory so they can all exist while they are
# being compared, which is the only way to attribute a difference to one of
# them.
set -euo pipefail

# The mode list, its validation and the directory a mode writes to all come from
# the module the job itself imports, so the submitter and the job cannot
# disagree about what a mode means or where it lands. A shell script repeating
# the list is one that accepts a mode the job then rejects, an hour into a
# reservation.
GDN_MODULE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gdn_precision.py"
gdn() { python3 "$GDN_MODULE" "$@"; }

usage() {
    echo "usage: bash quant/scripts/submit_quantize.sh NGPUS [--gdn=MODE] [--output-dir=PATH]" >&2
    echo "                                      [--algorithm=awq|awq+gptq]" >&2
    echo "  --gdn  what the Gated DeltaNet input projections are built at:" >&2
    echo "         $(gdn --modes) (default source)" >&2
    echo "         quant/scripts/gdn_precision.py says what each one means" >&2
    echo "                                      [--dependency=SPEC] [--time=HH:MM:SS]" >&2
    exit 2
}

NGPUS="${1:-}"
[[ -n "$NGPUS" ]] || usage
shift
case "$NGPUS" in
    *[!0-9]* | 0) echo "NGPUS must be a positive integer, got: $NGPUS" >&2; exit 2 ;;
esac

GDN_PRECISION=""
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
        --gdn=*) GDN_PRECISION="${arg#*=}" ;;
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

# Rejected here rather than in the job: an invalid mode should cost a shell
# prompt, not a queued reservation. The suffix comes from the same call, so the
# algorithm and the mode cannot both claim the same directory.
GDN_PRECISION="$(gdn "$GDN_PRECISION")" || exit 2
DEFAULT_SUFFIX="$(gdn "$GDN_PRECISION" --algorithm "$ALGORITHM" --suffix)"

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
