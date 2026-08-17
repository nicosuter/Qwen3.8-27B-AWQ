#!/usr/bin/env bash
# Check what campaign.sh actually queues, with submit-paired.sh stubbed out.
#
# The lane machinery is tested next door; this is about the campaign's own two
# decisions. Which arm each lane scores and where it writes -- get that wrong and
# two lanes share a run directory, or a candidate is compared against a baseline
# that was never scored. And whether the baseline is bought or inherited, which
# is the difference between a campaign that costs one set of GPU hours and one
# that costs two.
set -uo pipefail

CAMPAIGN="${1:?usage: test_campaign.sh <path to campaign.sh>}"
ROOT="$(cd "$(dirname "$CAMPAIGN")/../.." && pwd)"
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

mkdir -p "$WORK/eval/slurm" "$WORK/eval/scripts"
cp "$ROOT/eval/slurm/campaign.sh" "$ROOT/eval/slurm/campaign-lib.sh" "$WORK/eval/slurm/"
cp "$ROOT/eval/scripts/checkpoints.py" "$WORK/eval/scripts/"
# The library asks eval_suite.py which suite version is current rather than
# carrying a literal, so the fixture needs it too.
cp "$ROOT/eval/scripts/eval_suite.py" "$WORK/eval/scripts/"
cp "$ROOT/eval/checkpoints.json" "$WORK/eval/"
cat > "$WORK/eval/slurm/submit-paired.sh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${SUBMIT_LOG:?}"
n="$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
echo "$(( 1000 + n ))"
STUB
chmod +x "$WORK/eval/slurm/submit-paired.sh"
git -C "$WORK" init --quiet
git -C "$WORK" -c user.email=t@t -c user.name=t commit --quiet --allow-empty -m t

RUN=/scratch/test
HUB="$RUN/huggingface/hub"

campaign() { # campaign <args...>  -> sets $out $rc, writes $SUBMIT_LOG
    SUBMIT_LOG="$WORK/submitted.txt"
    : > "$SUBMIT_LOG"
    export SUBMIT_LOG
    out="$(RUN_BASE="$RUN" EVAL_PYTHON=python3 bash "$WORK/eval/slurm/campaign.sh" \
        --arch testarch --gpu-quota 8 --gpus-per-lane 4 --suite-version v1 "$@" 2>&1)"
    rc=$?
}

echo "== case 1: with nothing to inherit, the first candidate hosts the baseline =="
campaign --candidates cyankiwi,soyrsoyr
check "exit 0" 0 "$rc"
# One batch by default -- every scored suite colocates on one server -- so two
# candidates plus the baseline's own lane.
check "three lanes" 3 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
check "the baseline is scored once" 1 \
    "$(grep -c -- 'PAIRED_VARIANTS=baseline' "$SUBMIT_LOG")"
check "into the first candidate's run directory, which is cyankiwi's" 2 \
    "$(grep -c -- "PAIRED_RUN_DIR=$RUN/v2/eval-suite-v1-cyan," "$SUBMIT_LOG")"
# The host's own arm shares that directory and needs nothing copied into it.
check "the host inherits nothing" 0 \
    "$(grep -- "PAIRED_RUN_DIR=$RUN/v2/eval-suite-v1-cyan," "$SUBMIT_LOG" \
        | grep -c 'PAIRED_INHERIT_BASELINE_FROM')"
check "and waits on nothing but its slot" 0 \
    "$(grep -- 'comment cyankiwi' "$SUBMIT_LOG" | grep -c 'afterok')"

echo "== case 2: every other candidate inherits that baseline and waits for it =="
check "soyrsoyr inherits" 1 \
    "$(grep -c -- "PAIRED_INHERIT_BASELINE_FROM=$RUN/v2/eval-suite-v1-cyan" "$SUBMIT_LOG")"
check "into its own run directory" 1 \
    "$(grep -c -- "PAIRED_RUN_DIR=$RUN/v2/eval-suite-v1-soy," "$SUBMIT_LOG")"
# The inherit copies the baseline results at job start, so the lane that
# produces them has to be finished, not merely running.
check "the candidate waits on the baseline lane" 1 \
    "$(grep -- 'comment soyrsoyr-full' "$SUBMIT_LOG" | grep -c -- '--dependency afterok:1001')"

echo "== case 3: the checkpoint paths come from the registry =="
# Two: the candidate's lane, and the baseline lane that shares the host's run
# directory and so is told the same candidate.
check "a published quantization is addressed through its snapshot" 2 \
    "$(grep -c -- "OUTPUT_DIR=$HUB/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea" "$SUBMIT_LOG")"
# The baseline is bound through its repository root, not its snapshot: a
# snapshot is a farm of symlinks into ../../blobs.
check "the baseline through its repository root" 3 \
    "$(grep -c -- "PAIRED_BASELINE_REPO=$HUB/models--Qwen--Qwen3.8-27B-FP8," "$SUBMIT_LOG")"
check "with its revision beside it" 3 \
    "$(grep -c -- 'PAIRED_BASELINE_REVISION=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a' "$SUBMIT_LOG")"
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE="$RUN" EVAL_PYTHON=python3 DEP_FP8_FULL=done bash "$WORK/eval/slurm/campaign.sh" \
    --arch testarch --gpu-quota 8 --candidates bf16gdn \
    --baseline-from "$RUN/v2/eval-suite-v1" --only bf16gdn-full 2>&1)"; rc=$?
check "one of ours is served as the directory we wrote" 1 \
    "$(grep -c -- "OUTPUT_DIR=$RUN/v2/model," "$SUBMIT_LOG")"

echo "== case 4: an inherited baseline is not scored again =="
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE="$RUN" EVAL_PYTHON=python3 DEP_FP8_FULL=done \
    bash "$WORK/eval/slurm/campaign.sh" --arch testarch --gpu-quota 8 \
    --candidates philbert,barry --baseline-from "$RUN/v2/eval-suite-v1" 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "two lanes, neither of them a baseline" 2 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
check "no baseline arm submitted" 0 "$(grep -c -- 'PAIRED_VARIANTS=baseline' "$SUBMIT_LOG")"
check "both inherit" 2 \
    "$(grep -c -- "PAIRED_INHERIT_BASELINE_FROM=$RUN/v2/eval-suite-v1" "$SUBMIT_LOG")"
# A baseline lane that has already finished cannot be depended on -- slurm
# rejects a dependency on a job it has purged.
check "the finished baseline is depended on by nobody" 0 \
    "$(grep -- 'comment philbert-full' "$SUBMIT_LOG" | grep -c 'afterok')"

# The other half of that rule, which needs its own submission because a lane is
# either finished or running and one campaign cannot show both of one lane.
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE="$RUN" EVAL_PYTHON=python3 DEP_FP8_FULL=4242 \
    bash "$WORK/eval/slurm/campaign.sh" --arch testarch --gpu-quota 8 \
    --candidates philbert --baseline-from "$RUN/v2/eval-suite-v1" 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "a baseline still running gates its lane" 1 \
    "$(grep -- 'comment philbert-full' "$SUBMIT_LOG" | grep -c -- '--dependency afterok:4242')"

echo "== case 5: a baseline that is neither scored nor named is refused =="
SUBMIT_LOG="$WORK/submitted.txt"; : > "$SUBMIT_LOG"; export SUBMIT_LOG
out="$(RUN_BASE="$RUN" EVAL_PYTHON=python3 bash "$WORK/eval/slurm/campaign.sh" --arch testarch \
    --gpu-quota 8 --candidates philbert --baseline-from "$RUN/v2/eval-suite-v1" 2>&1)"; rc=$?
check "refused" 1 "$rc"
check "and said which variable would supply it" 1 \
    "$(grep -c 'DEP_FP8_FULL is unset' <<<"$out")"

echo "== case 6: lanes and their walltimes are runtime decisions =="
campaign --candidates cyankiwi --lane full=16:00:00
check "one batch, one candidate, plus its baseline" 2 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"
check "at the walltime asked for" 2 "$(grep -c -- '--time 16:00:00' "$SUBMIT_LOG")"
check "and the batch reaches the job" 2 "$(grep -c -- 'PAIRED_BATCH=full' "$SUBMIT_LOG")"

echo "== case 7: an unknown checkpoint is refused before anything is queued =="
campaign --candidates nosuchthing
check "refused" 1 "$rc"
check "named the registry's contents" 1 "$(grep -c 'no checkpoint named' <<<"$out")"
check "and submitted nothing" 0 "$(wc -l < "$SUBMIT_LOG" | tr -d ' ')"

echo
echo "passed $PASS, failed $FAIL"
[[ "$FAIL" -eq 0 ]]
