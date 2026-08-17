#!/usr/bin/env bash
# The piora half of the v1 campaign: the FP8 baseline against cyankiwi's and
# soyrsoyr's AWQ checkpoints, on A100.
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
# A100 nodes work at once instead of one.
#
# On node selection: piora advertises gpu:4 on six nodes, but only piora1 and
# piora2 carry A100s. piora6-8 are V100, which cannot hold a 27B checkpoint at
# any data-parallel size, and piora5 is usually down. sinfo's GRES column does
# not name the part and there is no node feature to select on, so they are
# excluded by name. paired-suite-eval.sbatch would refuse to score on a V100
# anyway; that guard is the backstop, not the plan.
set -euo pipefail

CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN_EXCLUDE="${PAIRED_EXCLUDE:-piora5,piora6,piora7,piora8}"
# Six suites against a dequantized FP8 baseline took a measured 6h+ on A100 and
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
lane baseline-short "$SHORT_TIME" "" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=baseline"

lane cyan-short "$SHORT_TIME" "" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=candidate"

lane baseline-ruler "$LONG_TIME" "" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=baseline"

lane cyan-ruler "$LONG_TIME" "" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=candidate"

# soy inherits the baseline instead of rescoring it, which means each soy lane
# waits on the lane that produces the half it needs -- and on nothing else. The
# inherit copies orders, materialized items and the baseline results across at
# job start, so the source has to be finished, not merely running.
lane soy-short "$SHORT_TIME" "afterok:@baseline-short" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_SOY" "OUTPUT_DIR=$SOY" \
    "PAIRED_INHERIT_BASELINE_FROM=$RUN_CYAN" \
    "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=candidate"

lane soy-ruler "$LONG_TIME" "afterok:@baseline-ruler" \
    "${BASELINE[@]}" "PAIRED_RUN_DIR=$RUN_SOY" "OUTPUT_DIR=$SOY" \
    "PAIRED_INHERIT_BASELINE_FROM=$RUN_CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=candidate"

campaign_summary
