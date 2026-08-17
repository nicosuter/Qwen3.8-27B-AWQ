#!/usr/bin/env bash
# Concatenate a run directory's per-suite results and run the comparator over it.
#
# This used to be the tail of paired-suite-eval.sbatch, which was fine while a
# GPU job was the only thing that could produce a comparison. It is not: the
# LiveCodeBench lane defers execution deliberately -- running model-written code
# on the node serving the model is the one thing that clearly does not belong
# there -- so its verdict is produced later, by a job with no GPU at all. Two
# copies of the comparator invocation would have meant two copies of the floors,
# and a floor that exists twice is a floor that will disagree with itself.
#
# Inputs are environment variables rather than flags because both callers are
# sbatch scripts that already have them in scope:
#
#   RUN_DIR          the run directory to compare (required)
#   PYTHON           interpreter (default python3)
#   EVAL_SUITE_VERSION  which pre-registered suite set (default v1)
#   EXCLUDE_SUITES   space-separated suites to leave out, e.g. ones that failed
#   ALLOW_PARTIAL    1 when the caller scored a subset and knows it
#   COMPARISON_TAG   what to name the comparison, normally the job id
set -euo pipefail

RUN_DIR="${RUN_DIR:?set RUN_DIR to the run directory to compare}"
PYTHON="${PYTHON:-python3}"
EVAL_SUITE_VERSION="${EVAL_SUITE_VERSION:-v2}"
EXCLUDE_SUITES="${EXCLUDE_SUITES:-}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
COMPARISON_TAG="${COMPARISON_TAG:-manual}"

# Suites the eval suite does not define. A run directory outlives the protocol
# version that filled it -- dropping a suite leaves its results on disk -- and
# the comparator refuses a set it was not calibrated against rather than
# averaging one in. Excluding them here keeps the rows as evidence without
# letting them reach the macro.
DEFINED="$("$PYTHON" scripts/eval_suite.py --version "$EVAL_SUITE_VERSION" --names)"

excluded() {
    local name found=0
    for name in $EXCLUDE_SUITES; do [[ "$name" == "$1" ]] && return 0; done
    for name in $DEFINED; do [[ "$name" == "$1" ]] && found=1; done
    if (( ! found )); then
        echo "excluding $1: eval suite $EVAL_SUITE_VERSION does not define it" >&2
        return 0
    fi
    return 1
}

# One file per checkpoint so the comparator sees every suite at once; it
# requires identical keys on both sides and will say so if a suite is missing.
# Everything on disk is swept, not just the suites a given job scored: a chained
# job scores one suite and reuses the rest, and narrowing here would silently
# drop them from the macro. Failed suites are the one exclusion.
concat_variant() {
    local variant="$1" path suite found=0
    # Built to a temp and moved into place. Truncating first blanked the other
    # arm's file whenever a job scored only one variant, and left a reader
    # holding a half-written file even when it scored both.
    local scratch="$RUN_DIR/.$variant-all.jsonl.$$"
    : > "$scratch"
    for path in "$RUN_DIR"/raw/"$variant"/*.jsonl; do
        [[ -e "$path" ]] || continue
        suite="$(basename "$path")"
        suite="${suite%-r*}"
        if excluded "$suite"; then
            echo "excluding $(basename "$path") from the comparison" >&2
            continue
        fi
        cat "$path" >> "$scratch"
        found=1
    done
    if (( found )); then
        mv -f "$scratch" "$RUN_DIR/$variant-all.jsonl"
    else
        rm -f "$scratch"
        echo "no scored results for $variant; leaving $variant-all.jsonl as it was" >&2
    fi
}

for variant in baseline candidate; do
    concat_variant "$variant"
done

echo "=== paired comparison ==="
# Every comparison is kept. A run directory is shared across jobs on purpose --
# that is what makes an earlier baseline reusable -- but it also meant the last
# job to finish silently replaced the verdict, and a chained job that rescores
# two suites produces a genuinely different number. comparison.json stays as the
# name everything else reads, now pointing at the newest.
#
# Floors sit at 95% of each suite's measured FP8 baseline, from v2/paired-2 and
# v2/paired-fp8gdn. They alert rather than gate: a baseline that far below where
# it has always landed means the harness broke for both arms, which is a reason
# to go and look at the run and not a reason to fail a checkpoint. The suites
# with no paired baseline yet get floors once they have one worth taking 95% of;
# declaring one now would be inventing the number it is supposed to check.
mkdir -p "$RUN_DIR/comparisons"
COMPARISON="$RUN_DIR/comparisons/$RUN_STAMP-job$COMPARISON_TAG.json"
"$PYTHON" scripts/compare_eval_results.py \
    --baseline "$RUN_DIR/baseline-all.jsonl" \
    --candidate "$RUN_DIR/candidate-all.jsonl" \
    --output "$COMPARISON" \
    --baseline-floor bfcl_v4="${BFCL_FLOOR:-0.830}" \
    --baseline-floor gpqa_diamond="${GPQA_FLOOR:-0.844}" \
    --baseline-floor livecodebench_v6="${LCB_FLOOR:-0.830}" \
    --baseline-floor multimodal="${MULTIMODAL_FLOOR:-0.824}" \
    --baseline-floor ruler="${RULER_FLOOR:-0.760}" \
    --eval-suite "$EVAL_SUITE_VERSION" \
    $( (( ALLOW_PARTIAL )) && echo --allow-partial ) \
    || true   # a failing gate is a result, not a job error
if [[ -s "$COMPARISON" ]]; then
    ln -sfn "comparisons/$(basename "$COMPARISON")" "$RUN_DIR/comparison.json"
fi
echo "comparison=$COMPARISON"
