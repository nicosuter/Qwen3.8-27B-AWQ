#!/usr/bin/env bash
# The piora half of the v1 campaign: the FP8 baseline against cyankiwi's and
# soyrsoyr's AWQ checkpoints, on A100.
#
# Two failures this file exists to stop repeating.
#
# The lanes used to be submitted by hand, one ssh line each, with the entire
# configuration carried in the --export list. Slurm will not show a pending
# job's environment, so once the terminal scrolled away there was no way to read
# back what a queued job was going to do; recovering it meant cancelling and
# reconstructing it from the logs of the one that had already started. What a
# campaign runs is a decision, and decisions belong in the repository.
#
# The lanes were also chained: five jobs on afterok in a single line. Only the
# job that overruns is killed, but every job behind it becomes
# DependencyNeverSatisfied and never runs, which is how the first attempt turned
# one timeout into four dead jobs and an idle cluster. The dependencies below are
# the real ones and nothing more -- a candidate that inherits a baseline waits
# for that baseline, and everything else waits for nobody. That is also what lets
# both A100 nodes work at once instead of one.
#
# On node selection: piora advertises gpu:4 on six nodes, but only piora1 and
# piora2 carry A100s. piora6-8 are V100, which cannot hold a 27B checkpoint at
# any data-parallel size, and piora5 is usually down. sinfo's GRES column does
# not name the part and there is no node feature to select on, so they are
# excluded by name. paired-suite-eval.sbatch would refuse to score on a V100
# anyway; that guard is the backstop, not the plan.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BASE="${RUN_BASE:-/scratch/$USER/qwen38-27b-awq}"
export RUN_BASE

# The commit every lane runs, which is the commit this campaign file came from.
# Naming a fixed sha here reads as safer and is not: the pin goes stale the
# moment the harness is fixed, and a campaign that keeps launching the version
# with the bug in it is worse than one that moves. Override to re-run an old
# campaign against the code that produced it.
COMMIT="${PAIRED_COMMIT:-$(git -C "$HERE/.." rev-parse HEAD)}"

HUB="$RUN_BASE/huggingface/hub"
BASELINE_REPO="$HUB/models--Qwen--Qwen3.8-27B-FP8"
BASELINE_REVISION="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
CYAN="$HUB/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea"
SOY="$HUB/models--soyrsoyr--Qwen3.8-27B-W4A16-AWQ-GPTQ/snapshots/15a8ec529ecb0c5d3437c3f853ce6df871f40028"

# One run directory per candidate: a run directory holds exactly one raw/candidate
# tree. The baseline arm is scored once, in the cyan directory, and soy inherits
# it rather than buying the same GPU hours twice.
RUN_CYAN="$RUN_BASE/v2/eval-suite-v1-cyan"
RUN_SOY="$RUN_BASE/v2/eval-suite-v1-soy"

EXCLUDE="${PAIRED_EXCLUDE:-piora5,piora6,piora7,piora8}"
# Six suites against a dequantized FP8 baseline took a measured 6h+ on A100 and
# was killed at the limit; RULER is one suite but a long-context one. Both are
# set well above the measurement because the cost of an overrun is the whole
# lane, and the cost of a generous limit is only scheduling priority.
SHORT_TIME="${PAIRED_SHORT_TIME:-16:00:00}"
LONG_TIME="${PAIRED_LONG_TIME:-10:00:00}"

# The already-running baseline short-context job, if there is one. soy's
# short-context lane copies that baseline in at startup, so it has to wait for
# it; leave this empty once the baseline is on disk and the lane submits free.
BASELINE_SHORT_JOB="${PAIRED_BASELINE_SHORT_JOB:-}"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# submit-paired.sh prints its checkout bookkeeping on stdout and then sbatch's
# output, so with --parsable the job id is the last all-digit line. Everything
# else is relayed to stderr, where it is readable but not captured.
submit() {
    local name="$1" walltime="$2" dep="$3"
    shift 3
    local exports
    exports="$(IFS=,; echo "$*")"

    local args=(--commit "$COMMIT" --export "$exports"
                --job-name "$name" --time "$walltime" --exclude "$EXCLUDE"
                --parsable)
    [[ -n "$dep" ]] && args+=(--dependency "afterok:$dep")

    # A dry run goes through submit-paired.sh too rather than printing what this
    # script believes it would do. The interesting mistakes are in the checkout,
    # the venv and the --export list it assembles, and a rehearsal that skips
    # those rehearses nothing.
    (( DRY )) && args+=(--dry-run)

    local out
    out="$("$HERE/submit-paired.sh" "${args[@]}")"
    printf '%s\n' "$out" >&2
    if (( DRY )); then
        printf '%s' "would-be-$name"
        return 0
    fi
    printf '%s' "$(printf '%s\n' "$out" | grep -xE '[0-9]+' | tail -n1)"
}

common=(
    "PAIRED_BASELINE_REPO=$BASELINE_REPO"
    "PAIRED_BASELINE_REVISION=$BASELINE_REVISION"
)

# --- cyan, in the run directory that also holds the shared baseline -----------
# Both arms of the cyan comparison are separate jobs on purpose: they write
# disjoint paths under one run directory, so they can hold a node each, and
# whichever finishes second is the one that produces the comparison.
CYAN_SHORT="$(submit v1-a100-cyan-short "$SHORT_TIME" "" \
    "${common[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=candidate")"

BASE_RULER="$(submit v1-a100-baseline-ruler "$LONG_TIME" "" \
    "${common[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=baseline")"

CYAN_RULER="$(submit v1-a100-cyan-ruler "$LONG_TIME" "" \
    "${common[@]}" "PAIRED_RUN_DIR=$RUN_CYAN" "OUTPUT_DIR=$CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=candidate")"

# --- soy, inheriting the baseline rather than rescoring it --------------------
# The inherit copies orders, materialized items and the baseline results out of
# the cyan directory, so each soy lane waits on the baseline lane that produces
# the half it needs -- and on nothing else.
SOY_SHORT="$(submit v1-a100-soy-short "$SHORT_TIME" "$BASELINE_SHORT_JOB" \
    "${common[@]}" "PAIRED_RUN_DIR=$RUN_SOY" "OUTPUT_DIR=$SOY" \
    "PAIRED_INHERIT_BASELINE_FROM=$RUN_CYAN" \
    "PAIRED_BATCH=short-context" "PAIRED_VARIANTS=candidate")"

SOY_RULER="$(submit v1-a100-soy-ruler "$LONG_TIME" "$BASE_RULER" \
    "${common[@]}" "PAIRED_RUN_DIR=$RUN_SOY" "OUTPUT_DIR=$SOY" \
    "PAIRED_INHERIT_BASELINE_FROM=$RUN_CYAN" \
    "PAIRED_BATCH=long-context" "PAIRED_VARIANTS=candidate")"

cat >&2 <<SUMMARY

submitted, commit $COMMIT, excluding $EXCLUDE
  cyan   short  candidate  $CYAN_SHORT   (no dependency)
  base   ruler  baseline   $BASE_RULER   (no dependency)
  cyan   ruler  candidate  $CYAN_RULER   (no dependency)
  soy    short  candidate  $SOY_SHORT   afterok:${BASELINE_SHORT_JOB:-none}
  soy    ruler  candidate  $SOY_RULER   afterok:$BASE_RULER
SUMMARY
