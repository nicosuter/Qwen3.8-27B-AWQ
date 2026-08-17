#!/usr/bin/env bash
# Submit a paired evaluation campaign: an FP8 baseline against one or more
# quantized candidates, one lane per candidate per batch.
#
# Two failures this file exists to stop repeating.
#
# The lanes used to be submitted by hand, one ssh line each, with the entire
# configuration carried in the --export list. Slurm will not show a pending
# job's environment, so once the terminal scrolled away there was no way to read
# back what a queued job was going to do; recovering it meant cancelling the
# queue and reconstructing it from the logs of the one job that had already
# started. What a campaign runs is a decision, and decisions belong in the
# repository.
#
# Then there was one of these per cluster, which is the same mistake a level up.
# The two files differed in the checkpoints they named, the walltimes they gave
# and how much of the allocation they assumed -- all runtime facts -- and they
# drifted in the machinery, which is not. So there is one campaign, and what
# differs between two runs of it is written on the command line:
#
#   eval/slurm/campaign.sh --candidates cyankiwi,soyrsoyr --arch a100 \
#       --gpu-quota 8 --gpus-per-lane 4 --lane full=16:00:00
#
#   eval/slurm/campaign.sh --candidates philbert,barry,bf16gdn --arch h200 \
#       --gpu-quota 4 --gpus-per-lane 4 \
#       --baseline-from "$RUN_BASE/v2/eval-suite-v1"
#
# The first scores the baseline itself, in the first candidate's run directory,
# because nothing has scored it there yet. The second inherits a baseline that
# already exists: scoring somebody else's quantization is the same paired
# measurement with a different candidate, so the baseline half is identical work
# and is seeded from a finished run rather than bought again. That inherited
# baseline must have been scored on this cluster -- A100 dequantizes the FP8
# baseline where H200 runs it natively, and a comparison across that boundary is
# not paired.
#
# Where the checkpoints live is in eval/checkpoints.json, and how much of the
# cluster is ours is on the command line. Which nodes to keep off is a property
# of one deployment and comes from PAIRED_EXCLUDE in the environment, never from
# this file: the partition advertises the same GRES on hardware that is not the
# same part, and some of it cannot hold a 27B checkpoint at any data-parallel
# size. paired-suite-eval.sbatch refuses to score on an unvalidated device
# anyway; that guard is the backstop, not the plan.
set -euo pipefail

CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=eval/slurm/campaign-lib.sh
source "$CAMPAIGN_ROOT/eval/slurm/campaign-lib.sh"

CHECKPOINTS="$CAMPAIGN_ROOT/eval/scripts/checkpoints.py"
resolve() { # resolve <candidate>  -> CHECKPOINT_DIR, CHECKPOINT_RUN
    local line
    line="$("$PYTHON" "$CHECKPOINTS" --candidate "$1" \
        --suite-version "$CAMPAIGN_SUITE_VERSION")" || exit 1
    IFS=$'\t' read -r CHECKPOINT_DIR CHECKPOINT_RUN <<< "$line"
}

line="$("$PYTHON" "$CHECKPOINTS" --baseline)" || exit 1
IFS=$'\t' read -r BASELINE_REPO BASELINE_REVISION <<< "$line"
BASELINE=(
    "PAIRED_BASELINE_REPO=$BASELINE_REPO"
    "PAIRED_BASELINE_REVISION=$BASELINE_REVISION"
)

declare -a NAMES=()
for name in ${CANDIDATES//,/ }; do NAMES+=("$name"); done

# One run directory per candidate: a run directory holds exactly one
# raw/candidate tree. The baseline arm is scored once, into the first
# candidate's directory, and every other candidate inherits it rather than
# buying the same GPU hours again.
HOST=""
if [[ -z "$BASELINE_FROM" ]]; then
    HOST="${NAMES[0]}"
    resolve "$HOST"
    BASELINE_FROM="$CHECKPOINT_RUN"
    HOST_CHECKPOINT="$CHECKPOINT_DIR"
    for entry in "${LANE_PLAN[@]}"; do
        lane fp8 "${entry%%=*}" "${entry#*=}" "" \
            "${BASELINE[@]}" "PAIRED_RUN_DIR=$BASELINE_FROM" \
            "OUTPUT_DIR=$HOST_CHECKPOINT" \
            "PAIRED_BATCH=${entry%%=*}" "PAIRED_VARIANTS=baseline"
    done
fi

for name in "${NAMES[@]}"; do
    resolve "$name"
    for entry in "${LANE_PLAN[@]}"; do
        batch="${entry%%=*}"
        # The host candidate's two arms are separate lanes on purpose: they
        # write disjoint paths under one run directory, so they can hold a node
        # each, and whichever finishes second produces the comparison. Every
        # other candidate copies the baseline in at job start, so it waits for
        # the lane that produced that half -- and for nothing else.
        declare -a inherit=()
        dep=""
        if [[ "$name" != "$HOST" ]]; then
            inherit=("PAIRED_INHERIT_BASELINE_FROM=$BASELINE_FROM")
            dep="afterok:@fp8-$batch"
        fi
        lane "$name" "$batch" "${entry#*=}" "$dep" \
            "${BASELINE[@]}" "PAIRED_RUN_DIR=$CHECKPOINT_RUN" \
            "OUTPUT_DIR=$CHECKPOINT_DIR" ${inherit[@]+"${inherit[@]}"} \
            "PAIRED_BATCH=$batch" "PAIRED_VARIANTS=candidate"
    done
done

campaign_summary
