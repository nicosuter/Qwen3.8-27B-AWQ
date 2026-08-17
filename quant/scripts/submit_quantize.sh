#!/usr/bin/env bash
# Submit a quantization run.
#
#   bash quant/scripts/submit_quantize.sh 8
#   bash quant/scripts/submit_quantize.sh 8 --gdn-in-proj=source
#   bash quant/scripts/submit_quantize.sh 4 --gdn-in-proj=int4 --algorithm=awq+gptq
#
# --algorithm chooses how weights are rounded once AWQ has scaled activations.
# awq rounds each weight to the nearest representable value on its own;
# awq+gptq pushes each column's rounding error into the columns not yet
# quantized, correcting the layer output instead. GPTQ costs roughly two to
# three times the runtime and writes to its own directory so both survive.
#
# --gdn-in-proj selects what happens to linear_attn.in_proj_qkv and in_proj_z,
# two of the eight tensor families in a Gated DeltaNet block and 4.0B parameters
# between them. The rest of the block -- in_proj_a, in_proj_b, the conv1d, the
# norm and the decay parameters -- is in source precision in every mode, and
# out_proj is four-bit in every mode, so this flag is about the input to the
# state update and not about the block. It used to be a
# --fp8-gdn boolean, which could only express the two ends of what turned out to
# be a choice between four, eight and source precision, all weight-only. It
# defaults to int8:
# the AWQ mappings do not reach this path, so four bits here would be bare
# round-to-nearest where every other four-bit tensor gets activation-aware
# rescaling. Activations are BF16 in every mode and are not a flag. See
# gdn_in_proj.py.
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
GDN_MODULE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gdn_in_proj.py"
gdn() { python3 "$GDN_MODULE" "$@"; }

usage() {
    echo "usage: bash quant/scripts/submit_quantize.sh NGPUS [--gdn-in-proj=MODE] [--output-dir=PATH]" >&2
    echo "                                      [--algorithm=awq|awq+gptq]" >&2
    echo "  --gdn-in-proj  what linear_attn.in_proj_qkv and in_proj_z are built at:" >&2
    echo "         $(gdn --modes) (default $(gdn))" >&2
    echo "         quant/scripts/gdn_in_proj.py says what each one means" >&2
    echo "  --cpus-per-gpu N  cores per GPU (16); the total scales with NGPUS" >&2
    echo "                                      [--dependency=SPEC] [--time=HH:MM:SS]" >&2
    exit 2
}

NGPUS="${1:-}"
[[ -n "$NGPUS" ]] || usage
shift
case "$NGPUS" in
    *[!0-9]* | 0) echo "NGPUS must be a positive integer, got: $NGPUS" >&2; exit 2 ;;
esac

GDN_IN_PROJ_PRECISION=""
ALGORITHM=awq
# The job header asks for 64 cores because it was written for a whole-node,
# eight-GPU run. Asking for the same 64 while using two GPUs is half a shared
# node's CPU for a quarter of its GPUs, and CPU is a consumable resource where
# host memory here is not -- so this is the request that can actually keep
# somebody else's job pending. Scaled with the GPU count instead.
CPUS_PER_GPU=16
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
        --gdn-in-proj=*) GDN_IN_PROJ_PRECISION="${arg#*=}" ;;
        --algorithm=*)
            ALGORITHM="${arg#*=}"
            case "$ALGORITHM" in
                awq | awq+gptq) ;;
                *) echo "--algorithm must be awq or awq+gptq, got: $ALGORITHM" >&2; exit 2 ;;
            esac
            ;;
        --cpus-per-gpu=*) CPUS_PER_GPU="${arg#*=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
        -h | --help) usage ;;
        *) echo "unknown argument: $arg" >&2; usage ;;
    esac
done

# Rejected here rather than in the job: an invalid mode should cost a shell
# prompt, not a queued reservation. The suffix comes from the same call, so the
# algorithm and the mode cannot both claim the same directory.
GDN_IN_PROJ_PRECISION="$(gdn "$GDN_IN_PROJ_PRECISION")" || exit 2
DEFAULT_SUFFIX="$(gdn "$GDN_IN_PROJ_PRECISION" --algorithm "$ALGORITHM" --suffix)"

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

is_positive() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_positive "$CPUS_PER_GPU" || { echo "--cpus-per-gpu must be a positive integer" >&2; exit 2; }
CPUS=$(( NGPUS * CPUS_PER_GPU ))

echo "gpus           : $NGPUS"
echo "cpus           : $CPUS ($CPUS_PER_GPU per gpu)"
echo "gdn in_proj    : $GDN_IN_PROJ_PRECISION"
echo "algorithm      : $ALGORITHM"
echo "output         : $OUTPUT_DIR"
[[ -n "$DEPENDENCY" ]] && echo "dependency     : $DEPENDENCY"
[[ -n "$TIME_LIMIT" ]] && echo "time limit     : $TIME_LIMIT"

exec sbatch \
    --gres="gpu:$NGPUS" \
    --cpus-per-task="$CPUS" \
    ${DEPENDENCY:+--dependency="$DEPENDENCY"} \
    ${TIME_LIMIT:+--time="$TIME_LIMIT"} \
    --export="ALL,GDN_IN_PROJ_PRECISION=$GDN_IN_PROJ_PRECISION,QUANT_ALGORITHM=$ALGORITHM,OUTPUT_DIR=$OUTPUT_DIR" \
    quant/slurm/quantize.sbatch
