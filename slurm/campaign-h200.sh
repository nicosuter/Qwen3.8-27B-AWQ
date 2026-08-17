#!/usr/bin/env bash
# The other half of the v1 campaign: the FP8 baseline against our own
# W4A16+FP8-GDN checkpoint and then the published AWQ quantizations.
#
# The baseline and our candidate are already scored in v2/eval-suite-v1, so the
# lanes here are the ones that inherit that baseline rather than rebuy it. Order
# is deliberate and is the order they were asked for: ours, then philbert, then
# barry, then the bf16-GDN variant last.
#
# Serialisation is the whole difficulty here. Our share of the allocation is
# smaller than what the scheduler will hand out, and slurm will not enforce the
# difference -- left alone it would happily start three of these at once on
# capacity that is not ours. The previous answer was to chain every lane on
# afterok, which serialises correctly and then propagates: one lane that
# overruns its limit turns every lane behind it into DependencyNeverSatisfied.
# That is what happened on the other cluster, with four dead jobs and an idle
# allocation.
#
# --dependency=singleton is the right tool and was the missing one. Slurm runs
# at most one job per (user, job name) at a time, and it keys on nothing else,
# so lanes serialise without any of them depending on another one succeeding. A
# lane that fails costs that lane. The per-lane job names existed to keep the
# logs readable, so the lanes share a name now and the output file carries the
# lane instead.
#
# The one real dependency left is the long-context baseline: a candidate RULER
# lane copies the baseline RULER results in at startup, so it waits for the job
# that produces them. That job is passed in as DEP_BASELINE_RULER because it was
# submitted before this file existed.
set -euo pipefail

CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# All lanes share this so --dependency=singleton can serialise them. Do not
# change it while lanes are queued under the old name: singleton keys on the
# name alone, so a rename mid-queue lets an old lane and a new one run at
# once. Rename the queued jobs with scontrol at the same time, or wait.
CAMPAIGN_JOB_NAME="${PAIRED_CAMPAIGN_NAME:-eval-qwen38-27b-v1-h200}"
CAMPAIGN_JOB_PREFIX="${PAIRED_JOB_PREFIX:-eval-qwen38-27b}"
CAMPAIGN_ARCH="${PAIRED_ARCH:-h200}"
CAMPAIGN_EXCLUDE="${PAIRED_EXCLUDE:-}"
SHORT_TIME="${PAIRED_SHORT_TIME:-12:00:00}"
LONG_TIME="${PAIRED_LONG_TIME:-10:00:00}"
# shellcheck source=slurm/campaign-lib.sh
source "$CAMPAIGN_ROOT/slurm/campaign-lib.sh"

HUB="$RUN_BASE/huggingface/hub"
BASELINE=(
    "PAIRED_BASELINE_REPO=$HUB/models--Qwen--Qwen3.8-27B-FP8"
    "PAIRED_BASELINE_REVISION=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
)

# Where the baseline lives, and what every candidate here inherits from.
RUN_BASE_DIR="$RUN_BASE/v2/eval-suite-v1"

# The published quantizations arrive through the Hugging Face cache; ours are
# directories we wrote. paired-suite-eval.sbatch binds the two differently and
# works that out from the path, so both are named the same way here.
PHILBERT="$HUB/models--philbert440--Qwen3.8-27B-W4A16-AWQ/snapshots/7908d42a71077a5e4dc458f273682b12dfe384a0"
BARRY="$HUB/models--barrydeen--Qwen3.8-27B-AWQ-4bit/snapshots/e6b4b8b025f85b3e18f13281db576e0b7d5fe314"
# v2/model is the bf16-GDN build: the same W4A16 AWQ weights as v2/model-fp8gdn
# with the 96 linear-attention projections left unquantized rather than FP8.
BF16GDN="$RUN_BASE/v2/model"

# Every lane waits on the running RULER job as well as on the singleton, because
# singleton only serialises against lanes sharing this name and that job does
# not. Without it the first lane would start beside it and double our share.
GATE="afterok:@fp8-ruler,singleton"

candidate_lanes() { # candidate_lanes <name> <checkpoint>
    local name="$1" checkpoint="$2" run_dir="$RUN_BASE/v2/eval-suite-v1-$1"
    lane "$name" short "$SHORT_TIME" "$GATE" \
        "${BASELINE[@]}" "PAIRED_RUN_DIR=$run_dir" "OUTPUT_DIR=$checkpoint" \
        "PAIRED_INHERIT_BASELINE_FROM=$RUN_BASE_DIR" \
        "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=candidate"
    lane "$name" ruler "$LONG_TIME" "$GATE" \
        "${BASELINE[@]}" "PAIRED_RUN_DIR=$run_dir" "OUTPUT_DIR=$checkpoint" \
        "PAIRED_INHERIT_BASELINE_FROM=$RUN_BASE_DIR" \
        "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=candidate"
}

candidate_lanes philbert "$PHILBERT"
candidate_lanes barry "$BARRY"
candidate_lanes bf16gdn "$BF16GDN"

campaign_summary
