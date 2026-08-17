#!/usr/bin/env bash
# Shared submit machinery for the per-cluster campaign files. Sourced, not run.
#
# A campaign file declares its lanes and nothing else; how a lane becomes a
# queued job -- the checkout, the export list, the node exclusions, the job-id
# bookkeeping -- lives here so the two clusters cannot drift apart in the
# mechanics while differing, as they should, in what they measure.
#
# The one piece of policy that is here rather than in the campaign files is that
# a lane is skippable. Re-running a campaign to pick up a harness fix used to
# resubmit every lane, including the ones already in flight, which put two jobs
# on the same run directory writing the same file. `--only` names the lanes to
# submit; the rest are reported as skipped rather than silently omitted, and a
# lane whose dependency was skipped has to be given that job id explicitly.

RUN_BASE="${RUN_BASE:-/scratch/$USER/qwen38-27b-awq}"
export RUN_BASE

# The commit every lane runs, which is the commit the campaign file came from.
# Naming a fixed sha reads as safer and is not: it goes stale the moment the
# harness is fixed, and a campaign that keeps launching the version with the bug
# in it is worse than one that moves. Override to re-run an old campaign against
# the code that produced it.
COMMIT="${PAIRED_COMMIT:-$(git -C "$CAMPAIGN_ROOT" rev-parse HEAD)}"

DRY=0
ONLY=""
while (( $# )); do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --only)    ONLY="$2"; shift 2 ;;
        --only=*)  ONLY="${1#*=}"; shift ;;
        *) echo "usage: $(basename "$0") [--dry-run] [--only lane,lane]" >&2; exit 2 ;;
    esac
done

# Lane ids are held in LANE_ID_<lane> rather than an associative array, and the
# variable names are built with tr rather than ${x^^}: both of those are bash 4,
# and the shell a developer runs the tests under is not the shell the cluster
# runs the campaign under. macOS is still on 3.2.
declare -a LANE_ORDER=()

lane_var() { # lane_var <lane> [prefix]  -> the variable holding its job id
    printf '%s%s' "${2:-LANE_ID_}" "$(printf '%s' "$1" | tr 'a-z-' 'A-Z_')"
}

lane_selected() {
    [[ -z "$ONLY" ]] && return 0
    local want
    for want in ${ONLY//,/ }; do [[ "$want" == "$1" ]] && return 0; done
    return 1
}

# The job id of a lane this run submitted. A lane that was not selected has no
# id, and guessing one is how a candidate ends up waiting on the wrong job, so
# this fails loudly and says which variable would supply it.
lane_id() {
    local name="$1" own dep
    own="$(lane_var "$name")"
    if [[ -n "${!own:-}" ]]; then
        printf '%s' "${!own}"
        return 0
    fi
    dep="$(lane_var "$name" DEP_)"
    if [[ -n "${!dep:-}" ]]; then
        printf '%s' "${!dep}"
        return 0
    fi
    echo "lane $name was not submitted and \$$dep is unset, so nothing can depend on it" >&2
    return 1
}

# A dependency names lanes rather than job ids: "afterok:@baseline-ruler" or
# "afterok:@baseline-ruler,singleton". Resolving it here, after the lane is
# known to be selected, is what lets --only name a lane whose siblings are
# already in flight. Resolving eagerly in the campaign file instead meant
# --only cyan-short died on a dependency belonging to a lane it was skipping.
resolve_dep() {
    local spec="$1" name id
    while [[ "$spec" =~ @([A-Za-z0-9_-]+) ]]; do
        name="${BASH_REMATCH[1]}"
        id="$(lane_id "$name")" || return 1
        spec="${spec//@$name/$id}"
    done
    printf '%s' "$spec"
}

# lane <name> <walltime> <dependency|""> <KEY=VALUE>...
#
# CAMPAIGN_JOB_NAME, if set, makes every lane share one slurm job name so that
# --dependency=singleton can serialise them. Slurm keys singleton on the name,
# and the per-lane names exist to keep the logs readable, so the output file is
# named for the lane instead.
lane() {
    local name="$1" walltime="$2" dep="$3"
    shift 3
    if ! lane_selected "$name"; then
        printf 'lane %-22s skipped (not in --only)\n' "$name" >&2
        return 0
    fi
    dep="$(resolve_dep "$dep")" || exit 1
    LANE_ORDER+=("$name")

    local exports
    exports="$(IFS=,; echo "$*")"
    local args=(--commit "$COMMIT" --export "$exports"
                --time "$walltime" --parsable)
    if [[ -n "${CAMPAIGN_JOB_NAME:-}" ]]; then
        args+=(--job-name "$CAMPAIGN_JOB_NAME" --output "slurm-logs/$name-%j.out")
    else
        args+=(--job-name "$name")
    fi
    [[ -n "${CAMPAIGN_EXCLUDE:-}" ]] && args+=(--exclude "$CAMPAIGN_EXCLUDE")
    [[ -n "$dep" ]] && args+=(--dependency "$dep")

    # A dry run goes through submit-paired.sh too rather than printing what this
    # script believes it would do. The interesting mistakes are in the checkout,
    # the venv and the --export list it assembles, and a rehearsal that skips
    # those rehearses nothing.
    (( DRY )) && args+=(--dry-run)

    local out
    out="$("$CAMPAIGN_ROOT/slurm/submit-paired.sh" "${args[@]}")"
    printf '%s\n' "$out" >&2
    local id
    if (( DRY )); then
        id="would-be-$name"
    else
        id="$(printf '%s\n' "$out" | grep -xE '[0-9]+' | tail -n1)"
    fi
    printf -v "$(lane_var "$name")" '%s' "$id"
    printf 'lane %-22s %s  %s  dep=%s\n' "$name" "$id" "$walltime" "${dep:-none}" >&2
}

campaign_summary() {
    local name var
    echo >&2
    echo "submitted at commit $COMMIT" >&2
    for name in ${LANE_ORDER[@]+"${LANE_ORDER[@]}"}; do
        var="$(lane_var "$name")"
        printf '  %-22s %s\n' "$name" "${!var}" >&2
    done
}
