#!/usr/bin/env bash
# One half of the v1 campaign: the FP8 baseline against two externally
# published AWQ checkpoints.
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
# The lanes were also chained: five jobs on afterok in a single line. Only the
# job that overruns is killed, but every job behind it becomes
# DependencyNeverSatisfied and never runs, which is how one timeout turned into
# four dead jobs and an idle cluster. The dependencies below are the real ones
# and nothing more -- a candidate that inherits a baseline waits for that
# baseline, and everything else waits for nobody. That is also what lets both
# allocation work in parallel instead of one lane at a time.
#
# On node selection: the partition advertises the same GRES on hardware that is
# not the same part, and some of it cannot hold a 27B checkpoint at any
# data-parallel size. sinfo's GRES column does not name the device and there is
# no node feature to select on, so the unusable nodes are excluded by name
# below. paired-suite-eval.sbatch refuses to score on an unvalidated device
# anyway; that guard is the backstop, not the plan.
set -euo pipefail

CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Which nodes to keep off is a property of one deployment, not of the protocol,
# so it comes from the environment the way RUN_BASE does rather than being
# written down here. Unset is safe but wasteful: the sbatch refuses to score on
# an unvalidated device, so a bad landing costs a scheduling round-trip.
CAMPAIGN_EXCLUDE="${PAIRED_EXCLUDE:-}"
CAMPAIGN_JOB_PREFIX="${PAIRED_JOB_PREFIX:-eval-qwen38-27b}"
CAMPAIGN_ARCH="${PAIRED_ARCH:-a100}"
# Six suites against a dequantized FP8 baseline took a measured 6h+ here and
# was killed at a six-hour limit; RULER is one suite but a long-context one.
# Both sit well above the measurement, because an overrun costs the whole lane
# and a generous limit costs only scheduling priority.
SHORT_TIME="${PAIRED_SHORT_TIME:-16:00:00}"
LONG_TIME="${PAIRED_LONG_TIME:-10:00:00}"
# shellcheck source=slurm/campaign-lib.sh
source "$CAMPAIGN_ROOT/slurm/campaign-lib.sh"

HUB="$RUN_BASE/huggingface/hub"
BASELINE=(
    "PAIRED_BASELINE_REPO=$HUB/models--Qwen--Qwen3.8-27B-FP8"
    "PAIRED_BASELINE_REVISION=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
)
CYAN="$HUB/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea"
SOY="$HUB/models--soyrsoyr--Qwen3.8-27B-W4A16-AWQ-GPTQ/snapshots/15a8ec529ecb0c5d3437c3f853ce6df871f40028"

# One run directory per candidate: a run directory holds exactly one
# raw/candidate tree. The baseline arm is scored once, in the cyan directory,
# and soy inherits it rather than buying the same GPU hours twice.
RUN_CYAN="$RUN_BASE/v2/eval-suite-v1-cyan"
RUN_SOY="$RUN_BASE/v2/eval-suite-v1-soy"

# The two arms of the cyan comparison are separate lanes on purpose: they write
# disjoint paths under one run directory, so they can hold a node each, and
# whichever finishes second is the one that produces the comparison.
lane fp8 short "$SHORT_TIME" "" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=baseline"

lane cyankiwi short "$SHORT_TIME" "" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=candidate"

lane fp8 ruler "$LONG_TIME" "" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=baseline"

lane cyankiwi ruler "$LONG_TIME" "" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=candidate"

# soy inherits the baseline instead of rescoring it, which means each soy lane
# waits on the lane that produces the half it needs -- and on nothing else. The
# inherit copies orders, materialized items and the baseline results across at
# job start, so the source has to be finished, not merely running.
lane soyrsoyr short "$SHORT_TIME" "afterok:@fp8-short" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_SOY" "OUTPUT_DIR=$SOY" \
    "PAIRED_INHERIT_BASELINE_FROM=$RUN_CYAN" \
    "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=candidate"

lane soyrsoyr ruler "$LONG_TIME" "afterok:@fp8-ruler" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_SOY" "OUTPUT_DIR=$SOY" \
    "PAIRED_INHERIT_BASELINE_FROM=$RUN_CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=candidate"

campaign_summary
