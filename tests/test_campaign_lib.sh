#!/usr/bin/env bash
# Exercise the campaign lane machinery with submit-paired.sh stubbed out, so
# lane selection and dependency resolution are checked without a cluster.
#
# These are the two places where an operational mistake is expensive rather than
# merely annoying. Re-running a campaign to pick up a harness fix resubmitted
# every lane including the ones already running, which put two jobs on one run
# directory writing the same file; and a dependency that silently resolves to
# nothing produces a candidate that inherits a baseline which does not exist yet
# and only says so hours later.
set -uo pipefail

LIB="${1:?usage: test_campaign_lib.sh <path to campaign-lib.sh>}"
ROOT="$(cd "$(dirname "$LIB")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PASS=0
FAIL=0

check() { # check <desc> <expected> <actual>
    if [[ "$2" == "$3" ]]; then
        echo "  ok   $1"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $1: expected [$2] got [$3]"
        FAIL=$((FAIL + 1))
    fi
}

# A stub standing in for submit-paired.sh: records the arguments it was handed
# and prints an id the way sbatch --parsable does.
mkdir -p "$WORK/slurm"
cat > "$WORK/slurm/submit-paired.sh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${SUBMIT_LOG:?}"
n="$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
echo "$(( 1000 + n ))"
STUB
chmod +x "$WORK/slurm/submit-paired.sh"
cp "$LIB" "$WORK/slurm/campaign-lib.sh"
# The lib takes the commit from the checkout it came from, so it needs one.
git -C "$WORK" init --quiet
git -C "$WORK" -c user.email=t@t -c user.name=t commit --quiet --allow-empty -m t

# A campaign is a script that sources the lib, so the tests are too. Written to
# a file rather than sourced inline because the lib parses "$@" at source time.
campaign() { # campaign <args...>  -> sets $out $rc, writes $SUBMIT_LOG
    SUBMIT_LOG="$WORK/submitted.txt"
    : > "$SUBMIT_LOG"
    export SUBMIT_LOG
    out="$(RUN_BASE=/scratch/test bash "$WORK/campaign.sh" "$@" 2>&1)"
    rc=$?
}

cat > "$WORK/campaign.sh" <<'CAMPAIGN'
set -euo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_JOB_PREFIX=eval-qwen38-27b
CAMPAIGN_ARCH=testarch
source "$CAMPAIGN_ROOT/slurm/campaign-lib.sh"
lane alpha "" 01:00:00 "" "K=1"
lane bravo "" 02:00:00 "" "K=2"
lane charlie "" 03:00:00 "afterok:@alpha" "K=3"
campaign_summary
CAMPAIGN

echo "== case 1: every lane submits when nothing is selected =="
campaign
check "exit 0" 0 "$rc"
check "three lanes submitted" 3 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
check "charlie waits on alpha's id" 1 "$(grep -c -- '--dependency afterok:1001' "$SUBMIT_LOG")"
check "alpha carries its own walltime" 1 "$(grep -c -- '--time 01:00:00' "$SUBMIT_LOG")"
# One glance at squeue has to say which checkpoint is being scored and on what.
check "the job name follows the scheme" 1 \
    "$(grep -c -- '--job-name eval-qwen38-27b-alpha-v1-testarch' "$SUBMIT_LOG")"

echo "== case 2: --only submits just the named lanes =="
campaign --only bravo
check "exit 0" 0 "$rc"
check "one lane submitted" 1 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
check "it was bravo" 1 "$(grep -c -- 'K=2' "$SUBMIT_LOG")"
check "the others are reported, not hidden" 1 "$(grep -c 'lane alpha .* skipped' <<<"$out")"
# The point of resolving dependencies lazily: charlie is skipped too, so the
# dependency it would have had is never looked up and cannot abort the run.
check "a skipped lane's dependency is not resolved" 0 "$(grep -c 'DEP_ALPHA' <<<"$out")"

echo "== case 3: a dependency on a skipped lane stops the campaign =="
# The failure this replaces submitted charlie with no dependency at all, because
# `exit` inside a command substitution leaves only the subshell.
campaign --only charlie
check "refused" 1 "$rc"
check "said which variable would supply it" 1 "$(grep -c 'DEP_ALPHA is unset' <<<"$out")"
check "and submitted nothing" 0 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"

echo "== case 4: the id can be supplied for a lane submitted earlier =="
campaign_with_dep() {
    SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
    out="$(RUN_BASE=/scratch/test DEP_ALPHA=98765 bash "$WORK/campaign.sh" --only charlie 2>&1)"
    rc=$?
}
campaign_with_dep
check "exit 0" 0 "$rc"
check "used the supplied id" 1 "$(grep -c -- '--dependency afterok:98765' "$SUBMIT_LOG")"

echo "== case 5: a shared job name serialises lanes and renames the logs =="
cat > "$WORK/campaign.sh" <<'CAMPAIGN'
set -euo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_JOB_PREFIX=eval-qwen38-27b
CAMPAIGN_ARCH=testarch
CAMPAIGN_JOB_NAME=shared
source "$CAMPAIGN_ROOT/slurm/campaign-lib.sh"
lane alpha "" 01:00:00 singleton "K=1"
CAMPAIGN
campaign
check "exit 0" 0 "$rc"
check "one slurm name for the campaign" 1 "$(grep -c -- '--job-name shared' "$SUBMIT_LOG")"
check "output still follows the scheme" 1 \
    "$(grep -c -- '--output slurm-logs/eval-qwen38-27b-alpha-v1-testarch-%j.out' "$SUBMIT_LOG")"
check "and the comment carries the lane" 1 "$(grep -c -- '--comment alpha' "$SUBMIT_LOG")"
check "singleton passed through" 1 "$(grep -c -- '--dependency singleton' "$SUBMIT_LOG")"

echo "== case 6: node exclusions reach sbatch only when set =="
cat > "$WORK/campaign.sh" <<'CAMPAIGN'
set -euo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_JOB_PREFIX=eval-qwen38-27b
CAMPAIGN_ARCH=testarch
CAMPAIGN_EXCLUDE=nodeX,nodeY
source "$CAMPAIGN_ROOT/slurm/campaign-lib.sh"
lane alpha "" 01:00:00 "" "K=1"
CAMPAIGN
campaign
check "excluded nodes passed through" 1 "$(grep -c -- '--exclude nodeX,nodeY' "$SUBMIT_LOG")"

echo "== case 7: exports are one comma-joined list, as sbatch wants =="
check "joined" 1 "$(grep -c -- '--export K=1' "$SUBMIT_LOG")"

echo "== case 7b: a dependency already finished is dropped, not resolved =="
# Slurm rejects afterok on a job it has purged, which is what a resumed campaign
# hands it. DEP_<LANE>=done says so explicitly.
cat > "$WORK/campaign.sh" <<'CAMPAIGN'
set -euo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_JOB_PREFIX=eval-qwen38-27b
CAMPAIGN_ARCH=testarch
source "$CAMPAIGN_ROOT/slurm/campaign-lib.sh"
lane solo "" 01:00:00 "afterok:@earlier,singleton" "K=1"
CAMPAIGN
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE=/scratch/test DEP_EARLIER=done bash "$WORK/campaign.sh" 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "the finished term is gone" 0 "$(grep -c 'afterok' "$SUBMIT_LOG")"
check "the rest of the dependency survives" 1 "$(grep -c -- '--dependency singleton' "$SUBMIT_LOG")"

echo "== case 8: the suffix lands in the job name and in the lane key =="
cat > "$WORK/campaign.sh" <<'CAMPAIGN'
set -euo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_JOB_PREFIX=eval-qwen38-27b
CAMPAIGN_ARCH=testarch
source "$CAMPAIGN_ROOT/slurm/campaign-lib.sh"
lane cyankiwi short 01:00:00 "" "K=1"
lane cyankiwi ruler 02:00:00 "afterok:@cyankiwi-short" "K=2"
CAMPAIGN
campaign
check "exit 0" 0 "$rc"
check "quant and suffix both in the name" 1 \
    "$(grep -c -- '--job-name eval-qwen38-27b-cyankiwi-v1-testarch-short' "$SUBMIT_LOG")"
check "the ruler lane too" 1 \
    "$(grep -c -- '--job-name eval-qwen38-27b-cyankiwi-v1-testarch-ruler' "$SUBMIT_LOG")"
# Dependencies refer to the short key, never to the slurm name.
check "dependency resolved through the key" 1 "$(grep -c -- '--dependency afterok:1001' "$SUBMIT_LOG")"

echo
echo "passed $PASS, failed $FAIL"
[[ "$FAIL" -eq 0 ]]
