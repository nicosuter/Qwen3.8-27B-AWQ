#!/usr/bin/env bash
# Submit a paired evaluation from a checkout of a published commit.
#
# The harness decides lane structure, timeouts, reuse and the gate arithmetic,
# and the eval configs decide what is measured. Both are in this repository, so
# a run is only identified once the commit that produced it is. Deployments used
# to be an rsync of whatever the working tree happened to hold, which meant a
# result could correspond to no commit that existed anywhere.
#
# Each commit gets its own checkout under $RUN_BASE/code/<sha>. They are cheap
# (the repository is under a megabyte) and disposable: gc-code-checkouts.sh
# removes the ones no job is using, and any commit can be fetched again.
#
# The virtualenv deliberately lives outside the checkout and is shared, because
# it is several gigabytes and has nothing to do with which commit is running.
set -euo pipefail

REMOTE="${PAIRED_REMOTE:-https://github.com/nicosuter/Qwen3.8-27B-AWQ.git}"
RUN_BASE="${RUN_BASE:-/scratch/$USER/qwen38-27b-awq}"
VENV="${EVAL_VENV:-$RUN_BASE/venv}"
COMMIT=""
DRY=0
declare -a SBATCH_ARGS=()
EXPORTS=""

usage() {
    cat >&2 <<'USAGE'
usage: submit-paired.sh --commit <sha> [--export K=V,...] [--dry-run] [sbatch args...]

  --commit <sha>   full or short commit to run; must exist on the remote
  --export         appended to the job's --export=ALL,... list
  --dry-run        prepare the checkout and print the sbatch command, submit nothing

Anything else is passed through to sbatch, so --nodelist, --time and
--job-name work as usual.
USAGE
    exit 2
}

while (( $# )); do
    case "$1" in
        --commit)   COMMIT="$2"; shift 2 ;;
        --export)   EXPORTS="$2"; shift 2 ;;
        --dry-run)  DRY=1; shift ;;
        -h|--help)  usage ;;
        *)          SBATCH_ARGS+=("$1"); shift ;;
    esac
done
[[ -n "$COMMIT" ]] || usage

test -x "$VENV/bin/python" || {
    echo "no interpreter at $VENV/bin/python; set EVAL_VENV" >&2
    exit 1
}

# A checkout is only useful if the commit can be fetched again later, which is
# what makes discarding old ones safe. An unpushed commit would strand the run.
mkdir -p "$RUN_BASE/code"
CODE_DIR="$RUN_BASE/code/$COMMIT"
if [[ ! -d "$CODE_DIR/.git" ]]; then
    echo "cloning $REMOTE at $COMMIT"
    rm -rf "$CODE_DIR"
    git clone --quiet --no-checkout "$REMOTE" "$CODE_DIR"
    git -C "$CODE_DIR" checkout --quiet --detach "$COMMIT" || {
        echo "commit $COMMIT is not on $REMOTE; push it first" >&2
        rm -rf "$CODE_DIR"
        exit 1
    }
else
    echo "reusing checkout $CODE_DIR"
fi

FULL_SHA="$(git -C "$CODE_DIR" rev-parse HEAD)"
# Detached at the requested commit and never written to, so this is belt and
# braces rather than a real possibility.
if git -C "$CODE_DIR" diff --quiet HEAD; then DIRTY=false; else DIRTY=true; fi

echo "commit   $FULL_SHA (dirty=$DIRTY)"
echo "checkout $CODE_DIR"
echo "venv     $VENV"

EXPORT_LIST="ALL,EVAL_VENV=$VENV,EVAL_CODE_COMMIT=$FULL_SHA,EVAL_CODE_DIRTY=$DIRTY"
EXPORT_LIST="$EXPORT_LIST,EVAL_CODE_REMOTE=$REMOTE,PAIRED_REQUIRE_PINNED_CODE=1"
[[ -n "$EXPORTS" ]] && EXPORT_LIST="$EXPORT_LIST,$EXPORTS"

# The sbatch takes its project directory from SLURM_SUBMIT_DIR, which is where
# sbatch is invoked rather than what --chdir says, so submit from the checkout.
cd "$CODE_DIR"
set -- sbatch --chdir="$CODE_DIR" --export="$EXPORT_LIST" \
    ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} "$CODE_DIR/slurm/paired-suite-eval.sbatch"

if (( DRY )); then
    printf 'would run:'; printf ' %q' "$@"; printf '\n'
    exit 0
fi
"$@"
