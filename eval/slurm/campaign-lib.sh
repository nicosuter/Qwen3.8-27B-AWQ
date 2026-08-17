#!/usr/bin/env bash
# Submit machinery for campaign.sh. Sourced, not run.
#
# The campaign file declares lanes; this turns a lane into a queued job -- the
# checkout, the export list, the GPU request, the serialisation, the job-id
# bookkeeping. The split exists so that what a campaign measures stays readable
# next to how it reaches the queue.
#
# Three pieces of policy live here rather than in the campaign file.
#
# A lane is skippable. Re-running a campaign to pick up a harness fix used to
# resubmit every lane, including the ones already in flight, which put two jobs
# on the same run directory writing the same file. `--only` names the lanes to
# submit; the rest are reported as skipped rather than silently omitted, and a
# lane whose dependency was skipped has to be given that job id explicitly.
#
# A lane's share of the allocation is declared, not assumed. `--gpus-per-lane`
# sets both the GRES the job asks slurm for and how vLLM is started across them,
# so the two cannot disagree; the sbatch checks them against the cgroup anyway
# and refuses a job that got fewer. By default those GPUs go into one
# tensor-parallel replica rather than that many data-parallel ones, because the
# wall clock is a tail and tensor parallelism is what shortens a tail;
# `--tp`/`--dp` override, and their product has to be the lane's GPU count.
#
# And the campaign holds no more of the cluster than it was told it may.
# `--gpu-quota` divided by `--gpus-per-lane` is how many lanes may run at once;
# the lanes are dealt round-robin into that many slots, and each lane waits on
# afterany of the previous lane in its slot. So the quota holds without any lane
# depending on another one succeeding, and every job keeps a name that says what
# it measures.
#
# Two earlier answers, and why neither survived. Chaining on afterok serialises
# correctly and then propagates: one lane that overruns turns every lane behind
# it into DependencyNeverSatisfied, which is how one timeout once cost four jobs
# and left the allocation idle. Naming each slot and carrying
# --dependency=singleton fixes that -- slurm runs one job per (user, job name)
# regardless of outcome -- but singleton keys on the name and nothing else, so
# the name has to be the slot, and a queue full of eval-...-a100-s0 says nothing
# about what any of it is measuring. afterany gets the outcome-independence of
# singleton while leaving the name free.

RUN_BASE="${RUN_BASE:-/scratch/$USER/qwen38-27b-awq}"
export RUN_BASE
PYTHON="${EVAL_PYTHON:-python3}"

# The commit every lane runs, which is the commit the campaign file came from.
# Naming a fixed sha reads as safer and is not: it goes stale the moment the
# harness is fixed, and a campaign that keeps launching the version with the bug
# in it is worse than one that moves. Override to re-run an old campaign against
# the code that produced it.
COMMIT="${PAIRED_COMMIT:-$(git -C "$CAMPAIGN_ROOT" rev-parse HEAD)}"

# Everything below is a runtime decision. Anything that describes one deployment
# rather than the measurement -- which nodes to keep off, how much of the cluster
# is ours -- comes from the environment and is never written down in the
# repository.
DRY=0
ONLY=""
CANDIDATES="${PAIRED_CANDIDATES:-}"
BASELINE_FROM="${PAIRED_BASELINE_FROM:-}"
GPU_QUOTA="${PAIRED_GPU_QUOTA:-}"
GPUS_PER_LANE="${PAIRED_GPUS_PER_LANE:-4}"
LANE_TP="${PAIRED_TP:-}"
LANE_DP="${PAIRED_DP:-}"
CAMPAIGN_ARCH="${PAIRED_ARCH:-}"
# Derived, never written down twice. A job named v1 while scoring v2 is the
# same drift the suite file exists to stop, and it is worse in a job name
# than in a config, because the name is what somebody reads off squeue
# months later to decide what a result was.
CAMPAIGN_SUITE_VERSION="${CAMPAIGN_SUITE_VERSION:-${PAIRED_SUITE_VERSION:-$(
    python3 "$CAMPAIGN_ROOT/eval/scripts/eval_suite.py" --default-version
)}}"
CAMPAIGN_JOB_PREFIX="${PAIRED_JOB_PREFIX:-eval-qwen38-27b}"
CAMPAIGN_EXCLUDE="${PAIRED_EXCLUDE:-}"
declare -a LANE_PLAN=()

campaign_usage() {
    cat >&2 <<USAGE
usage: $(basename "$0") --candidates <name,...> --arch <name>
                    --gpu-quota <n> [--gpus-per-lane <n>] [options]

  --candidates a,b     which checkpoints to score, by name from the registry
                       ($("$PYTHON" "$CAMPAIGN_ROOT/eval/scripts/checkpoints.py" --names 2>/dev/null || echo "eval/checkpoints.json"))
  --arch NAME          what the lanes run on; names the job and its log only
  --gpu-quota N        GPUs this campaign may hold at once, in total
  --gpus-per-lane N    GPUs one lane asks for, and how vLLM spans them (4)
  --tp N / --dp N      how those GPUs are split; the product must equal
                       --gpus-per-lane (default: all tensor-parallel)
  --lane BATCH=TIME    a lane per candidate, repeatable; a batch from
                       eval/batches.json and its walltime
                       (full=16:00:00)
  --baseline-from DIR  inherit an already-scored baseline from a run directory
                       instead of scoring one in the first candidate's
  --suite-version V    which suite version the job names carry (defaults to
                       the current one, from eval/scripts/eval_suite.py)
  --only a,b           submit just these lanes; the rest are reported skipped
  --dry-run            prepare the checkout and print the sbatch, submit nothing

Deployment details stay in the environment, never in the repository:
PAIRED_EXCLUDE for nodes to keep off, RUN_BASE for where a run lives.
A lane whose dependency is not submitted in this run needs its job id in
DEP_<LANE>, or DEP_<LANE>=done if it has already finished.
USAGE
    exit 2
}

while (( $# )); do
    case "$1" in
        --dry-run)          DRY=1; shift ;;
        --only)             ONLY="$2"; shift 2 ;;
        --only=*)           ONLY="${1#*=}"; shift ;;
        --candidates)       CANDIDATES="$2"; shift 2 ;;
        --candidates=*)     CANDIDATES="${1#*=}"; shift ;;
        --arch)             CAMPAIGN_ARCH="$2"; shift 2 ;;
        --arch=*)           CAMPAIGN_ARCH="${1#*=}"; shift ;;
        --gpu-quota)        GPU_QUOTA="$2"; shift 2 ;;
        --gpu-quota=*)      GPU_QUOTA="${1#*=}"; shift ;;
        --gpus-per-lane)    GPUS_PER_LANE="$2"; shift 2 ;;
        --gpus-per-lane=*)  GPUS_PER_LANE="${1#*=}"; shift ;;
        --tp)               LANE_TP="$2"; shift 2 ;;
        --tp=*)             LANE_TP="${1#*=}"; shift ;;
        --dp)               LANE_DP="$2"; shift 2 ;;
        --dp=*)             LANE_DP="${1#*=}"; shift ;;
        --lane)             LANE_PLAN+=("$2"); shift 2 ;;
        --lane=*)           LANE_PLAN+=("${1#*=}"); shift ;;
        --baseline-from)    BASELINE_FROM="$2"; shift 2 ;;
        --baseline-from=*)  BASELINE_FROM="${1#*=}"; shift ;;
        --suite-version)    CAMPAIGN_SUITE_VERSION="$2"; shift 2 ;;
        --suite-version=*)  CAMPAIGN_SUITE_VERSION="${1#*=}"; shift ;;
        -h|--help)          campaign_usage ;;
        *) echo "unknown argument: $1" >&2; campaign_usage ;;
    esac
done

is_positive() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

[[ -n "$CANDIDATES" ]] || { echo "--candidates is required" >&2; campaign_usage; }
[[ -n "$CAMPAIGN_ARCH" ]] || { echo "--arch is required" >&2; campaign_usage; }
is_positive "${GPU_QUOTA:-}" || { echo "--gpu-quota must be a positive integer" >&2; campaign_usage; }
is_positive "$GPUS_PER_LANE" || { echo "--gpus-per-lane must be a positive integer" >&2; campaign_usage; }
# One tensor-parallel replica over the lane's GPUs unless told otherwise.
LANE_TP="${LANE_TP:-$GPUS_PER_LANE}"
LANE_DP="${LANE_DP:-1}"
is_positive "$LANE_TP" || { echo "--tp must be a positive integer" >&2; campaign_usage; }
is_positive "$LANE_DP" || { echo "--dp must be a positive integer" >&2; campaign_usage; }
# Refused here rather than after the job starts: a product that does not match
# leaves vLLM either idling GPUs the campaign paid for or failing deep into
# startup, and the sbatch's own check cannot suggest the campaign flag to fix.
if (( LANE_TP * LANE_DP != GPUS_PER_LANE )); then
    echo "--tp $LANE_TP x --dp $LANE_DP is $(( LANE_TP * LANE_DP )) GPUs," \
         "but --gpus-per-lane is $GPUS_PER_LANE" >&2
    exit 2
fi

# How many lanes may be in flight. A quota smaller than one lane is a mistake
# worth failing on: it would either hold more than was granted or nothing at all.
SLOTS=$(( GPU_QUOTA / GPUS_PER_LANE ))
if (( SLOTS < 1 )); then
    echo "a quota of $GPU_QUOTA GPUs cannot run a lane of $GPUS_PER_LANE" >&2
    exit 2
fi

# One lane for the whole protocol, at a walltime well above the measurement. The
# split runs it replaces measured 4.0h and 4.9h for the two arms of the short
# suites plus 1.0h each for RULER, so about 11h of work that the colocated job
# overlaps down to roughly 9.5h. An overrun costs the whole lane where a generous
# limit costs scheduling priority, and this lane now carries every scored suite,
# so it is the one place where an overrun costs everything.
if (( ${#LANE_PLAN[@]} == 0 )); then
    LANE_PLAN=("full=16:00:00")
fi

# Lane ids are held in LANE_ID_<lane> rather than an associative array, and the
# variable names are built with tr rather than ${x^^}: both of those are bash 4,
# and the shell a developer runs the tests under is not the shell the cluster
# runs the campaign under. macOS is still on 3.2.
declare -a LANE_ORDER=()
SUBMITTED=0

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

# A dependency names lanes rather than job ids: "afterok:@fp8-full".
# Resolving it here, after the lane is known to be selected, is what lets --only
# name a lane whose siblings are already in flight. Resolving eagerly in the
# campaign file instead meant --only cyankiwi-full died on a dependency
# belonging to a lane it was skipping.
resolve_dep() {
    local spec="$1" name id term out=() terms=()
    IFS=',' read -ra terms <<< "$spec"
    for term in ${terms[@]+"${terms[@]}"}; do
        if [[ "$term" =~ @([A-Za-z0-9_-]+) ]]; then
            name="${BASH_REMATCH[1]}"
            id="$(lane_id "$name")" || return 1
            # DEP_<LANE>=done says that lane has already finished, which is the
            # normal case when a campaign is resumed part-way. Slurm rejects a
            # dependency on a job it has already purged -- "Job dependency
            # problem" -- so the term is dropped rather than resolved.
            [[ "$id" == "done" ]] && continue
            term="${term//@$name/$id}"
        fi
        out+=("$term")
    done
    printf '%s' "$(IFS=,; echo "${out[*]-}")"
}

# lane <quant> <batch|""> <walltime> <dependency|""> <KEY=VALUE>...
#
# The lane key -- what --only and dependencies refer to -- is "<quant>-<batch>".
# The slurm job name says what the job measures:
#   <prefix>-<quant>-<suite version>-<arch>[-<batch>]
# which is the same string its log file carries.
lane() {
    local quant="$1" batch="$2" walltime="$3" dep="$4"
    shift 4
    local key="$quant${batch:+-$batch}"
    if ! lane_selected "$key"; then
        printf 'lane %-28s skipped (not in --only)\n' "$key" >&2
        return 0
    fi
    dep="$(resolve_dep "$dep")" || exit 1
    LANE_ORDER+=("$key")

    local scheme="$CAMPAIGN_JOB_PREFIX-$quant-$CAMPAIGN_SUITE_VERSION-$CAMPAIGN_ARCH"
    scheme="$scheme${batch:+-$batch}"
    # The quota is held by chaining each slot's lanes on afterany, not by giving
    # them a shared name and a singleton. Both serialise; only one of them lets
    # the job say what it measures, and a queue of identically named jobs is how
    # a campaign gets lost track of.
    #
    # afterany rather than afterok, and the difference is the whole reason the
    # slot names existed. Chaining on afterok propagates: one lane that overruns
    # turns every lane behind it into DependencyNeverSatisfied, which once cost
    # four jobs and left the allocation idle. afterany fires when the previous
    # lane stops for any reason, so a failure costs that lane and nothing else.
    # It is sound here because lanes in a slot share only the GPUs -- a lane that
    # consumes another's results says so in its own afterok dependency, which is
    # resolved separately above and survives this.
    local slot_var slot_prev
    slot_var="SLOT_LAST_$(( SUBMITTED % SLOTS ))"
    SUBMITTED=$(( SUBMITTED + 1 ))
    slot_prev="${!slot_var:-}"
    [[ -n "$slot_prev" ]] && dep="${dep:+$dep,}afterany:$slot_prev"

    local exports
    exports="$(IFS=,; echo "$*"),PAIRED_TP=$LANE_TP,PAIRED_DP=$LANE_DP"
    local args=(--commit "$COMMIT" --export "$exports"
                --time "$walltime" --parsable
                --output "slurm-logs/$scheme-%j.out" --comment "$key"
                --job-name "$scheme" --gres "gpu:$GPUS_PER_LANE")
    [[ -n "$dep" ]] && args+=(--dependency "$dep")
    [[ -n "$CAMPAIGN_EXCLUDE" ]] && args+=(--exclude "$CAMPAIGN_EXCLUDE")

    # A dry run goes through submit-paired.sh too rather than printing what this
    # script believes it would do. The interesting mistakes are in the checkout,
    # the venv and the --export list it assembles, and a rehearsal that skips
    # those rehearses nothing.
    (( DRY )) && args+=(--dry-run)

    local out
    out="$("$CAMPAIGN_ROOT/eval/slurm/submit-paired.sh" "${args[@]}")"
    printf '%s\n' "$out" >&2
    local id
    if (( DRY )); then
        id="would-be-$key"
    else
        id="$(printf '%s\n' "$out" | grep -xE '[0-9]+' | tail -n1)"
    fi
    printf -v "$(lane_var "$key")" '%s' "$id"
    # The next lane dealt to this slot waits on this one, which is what holds
    # the campaign inside its share of the GPUs.
    printf -v "$slot_var" '%s' "$id"
    printf 'lane %-28s %s  %-9s %s dep=%s\n' \
        "$key" "$id" "$walltime" "$scheme" "${dep:-none}" >&2
}

campaign_summary() {
    local name var
    echo >&2
    echo "submitted at commit $COMMIT" >&2
    echo "$SUBMITTED lanes over $SLOTS slot(s) of $GPUS_PER_LANE GPUs" >&2
    echo "  tensor-parallel $LANE_TP, data-parallel $LANE_DP" >&2
    for name in ${LANE_ORDER[@]+"${LANE_ORDER[@]}"}; do
        var="$(lane_var "$name")"
        printf '  %-28s %s\n' "$name" "${!var}" >&2
    done
}
