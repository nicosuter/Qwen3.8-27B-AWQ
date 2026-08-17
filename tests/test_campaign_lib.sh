#!/usr/bin/env bash
# Exercise the campaign lane machinery with submit-paired.sh stubbed out, so
# lane selection, dependency resolution and the quota arithmetic are checked
# without a cluster.
#
# These are the places where an operational mistake is expensive rather than
# merely annoying. Re-running a campaign to pick up a harness fix resubmitted
# every lane including the ones already running, which put two jobs on one run
# directory writing the same file; a dependency that silently resolves to
# nothing produces a candidate that inherits a baseline which does not exist yet
# and only says so hours later; and a campaign that holds more of the cluster
# than it was granted is somebody else's job not starting.
set -uo pipefail

LIB="${1:?usage: test_campaign_lib.sh <path to campaign-lib.sh>}"
ROOT="$(cd "$(dirname "$LIB")/../.." && pwd)"
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
mkdir -p "$WORK/eval/slurm"
cat > "$WORK/eval/slurm/submit-paired.sh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${SUBMIT_LOG:?}"
n="$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
echo "$(( 1000 + n ))"
STUB
chmod +x "$WORK/eval/slurm/submit-paired.sh"
cp "$LIB" "$WORK/eval/slurm/campaign-lib.sh"
# The lib takes the commit from the checkout it came from, so it needs one.
git -C "$WORK" init --quiet
git -C "$WORK" -c user.email=t@t -c user.name=t commit --quiet --allow-empty -m t

# Every campaign has to say what it is scoring, on what, and how much of the
# cluster it may hold. The lane declarations are what differs between cases.
BASE_ARGS=(--candidates alpha --arch testarch --gpu-quota 8 --gpus-per-lane 4)

# A campaign is a script that sources the lib, so the tests are too. Written to
# a file rather than sourced inline because the lib parses "$@" at source time.
campaign() { # campaign <args...>  -> sets $out $rc, writes $SUBMIT_LOG
    SUBMIT_LOG="$WORK/submitted.txt"
    : > "$SUBMIT_LOG"
    export SUBMIT_LOG
    out="$(RUN_BASE=/scratch/test bash "$WORK/campaign.sh" "${BASE_ARGS[@]}" "$@" 2>&1)"
    rc=$?
}

cat > "$WORK/campaign.sh" <<'CAMPAIGN'
set -euo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CAMPAIGN_ROOT/eval/slurm/campaign-lib.sh"
lane alpha "" 01:00:00 "" "K=1"
lane bravo "" 02:00:00 "" "K=2"
lane charlie "" 03:00:00 "afterok:@alpha" "K=3"
campaign_summary
CAMPAIGN

echo "== case 1: every lane submits when nothing is selected =="
campaign
check "exit 0" 0 "$rc"
check "three lanes submitted" 3 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
check "charlie waits on alpha's id" 1 "$(grep -c -- '--dependency afterok:1001,singleton' "$SUBMIT_LOG")"
check "alpha carries its own walltime" 1 "$(grep -c -- '--time 01:00:00' "$SUBMIT_LOG")"
# What a job is measuring is carried by its log name and its comment, because
# the slurm name is a quota slot shared with whatever lane comes next.
check "the output follows the scheme" 1 \
    "$(grep -c -- '--output slurm-logs/eval-qwen38-27b-alpha-v1-testarch-%j.out' "$SUBMIT_LOG")"
check "and the comment carries the lane" 1 "$(grep -c -- '--comment alpha' "$SUBMIT_LOG")"

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
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE=/scratch/test DEP_ALPHA=98765 bash "$WORK/campaign.sh" \
    "${BASE_ARGS[@]}" --only charlie 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "used the supplied id" 1 "$(grep -c -- '--dependency afterok:98765,singleton' "$SUBMIT_LOG")"

echo "== case 5: the quota decides how many lanes may run at once =="
# Two lanes' worth of GPUs, three lanes: two slot names, dealt round-robin, and
# every lane on singleton so slurm runs one job per name.
campaign
check "three lanes, two slots" 2 \
    "$(grep -o -- '--job-name [^ ]*' "$SUBMIT_LOG" | sort -u | wc -l | tr -d ' ')"
check "slot 0 twice" 2 "$(grep -c -- '--job-name eval-qwen38-27b-testarch-s0' "$SUBMIT_LOG")"
check "slot 1 once" 1 "$(grep -c -- '--job-name eval-qwen38-27b-testarch-s1' "$SUBMIT_LOG")"
check "every lane is a singleton" 3 "$(grep -c -- 'singleton' "$SUBMIT_LOG")"

echo "== case 5b: one lane's worth of quota serialises the campaign =="
campaign --gpu-quota 4
check "one slot for everything" 1 \
    "$(grep -o -- '--job-name [^ ]*' "$SUBMIT_LOG" | sort -u | wc -l | tr -d ' ')"

echo "== case 5c: a quota that cannot run a single lane is refused =="
campaign --gpu-quota 2
check "refused" 2 "$rc"
check "said why" 1 "$(grep -c 'cannot run a lane' <<<"$out")"
check "and submitted nothing" 0 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"

echo "== case 5d: the GPUs a lane asks slurm for are the ones vLLM is given =="
# The sbatch header's --gres and PAIRED_DP are set independently and can
# disagree, which vLLM discovers after the image and the weights are loaded.
campaign --gpus-per-lane 2
check "gres requested" 3 "$(grep -c -- '--gres gpu:2' "$SUBMIT_LOG")"
check "and the same data-parallel size exported" 3 "$(grep -c -- 'PAIRED_DP=2' "$SUBMIT_LOG")"

echo "== case 6: node exclusions come from the environment, never the file =="
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE=/scratch/test PAIRED_EXCLUDE=nodeX,nodeY bash "$WORK/campaign.sh" \
    "${BASE_ARGS[@]}" --only alpha 2>&1)"; rc=$?
check "excluded nodes passed through" 1 "$(grep -c -- '--exclude nodeX,nodeY' "$SUBMIT_LOG")"
campaign --only alpha
check "and are absent when unset" 0 "$(grep -c -- '--exclude' "$SUBMIT_LOG")"

echo "== case 7: exports are one comma-joined list, as sbatch wants =="
check "joined" 1 "$(grep -c -- '--export K=1,PAIRED_DP=4' "$SUBMIT_LOG")"

echo "== case 7b: a dependency already finished is dropped, not resolved =="
# Slurm rejects afterok on a job it has purged, which is what a resumed campaign
# hands it. DEP_<LANE>=done says so explicitly.
cat > "$WORK/campaign.sh" <<'CAMPAIGN'
set -euo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CAMPAIGN_ROOT/eval/slurm/campaign-lib.sh"
lane solo "" 01:00:00 "afterok:@earlier" "K=1"
CAMPAIGN
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE=/scratch/test DEP_EARLIER=done bash "$WORK/campaign.sh" \
    "${BASE_ARGS[@]}" 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "the finished term is gone" 0 "$(grep -c 'afterok' "$SUBMIT_LOG")"
check "the singleton survives" 1 "$(grep -c -- '--dependency singleton' "$SUBMIT_LOG")"

echo "== case 8: the batch lands in the log name and in the lane key =="
cat > "$WORK/campaign.sh" <<'CAMPAIGN'
set -euo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CAMPAIGN_ROOT/eval/slurm/campaign-lib.sh"
lane cyankiwi short-context 01:00:00 "" "K=1"
lane cyankiwi long-context 02:00:00 "afterok:@cyankiwi-short-context" "K=2"
CAMPAIGN
campaign
check "exit 0" 0 "$rc"
check "quant and batch both in the log name" 1 \
    "$(grep -c -- '--output slurm-logs/eval-qwen38-27b-cyankiwi-v1-testarch-short-context-%j.out' "$SUBMIT_LOG")"
check "the long-context lane too" 1 \
    "$(grep -c -- '--output slurm-logs/eval-qwen38-27b-cyankiwi-v1-testarch-long-context-%j.out' "$SUBMIT_LOG")"
# Dependencies refer to the lane key, never to the slurm name.
check "dependency resolved through the key" 1 "$(grep -c -- '--dependency afterok:1001,singleton' "$SUBMIT_LOG")"

echo "== case 9: a campaign that says nothing about itself is refused =="
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE=/scratch/test bash "$WORK/campaign.sh" --arch testarch --gpu-quota 8 2>&1)"; rc=$?
check "no candidates refused" 2 "$rc"
check "said so" 1 "$(grep -c -- '--candidates is required' <<<"$out")"
out="$(RUN_BASE=/scratch/test bash "$WORK/campaign.sh" --candidates alpha --gpu-quota 8 2>&1)"; rc=$?
check "no arch refused" 2 "$rc"
out="$(RUN_BASE=/scratch/test bash "$WORK/campaign.sh" --candidates alpha --arch testarch 2>&1)"; rc=$?
check "no quota refused" 2 "$rc"

echo
echo "passed $PASS, failed $FAIL"
[[ "$FAIL" -eq 0 ]]
