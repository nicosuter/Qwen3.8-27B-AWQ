#!/usr/bin/env bash
# Exercise score_variant()/suite_is_current() from the real sbatch with a stubbed
# python, so lane fan-out, concurrency splitting, reuse and failure propagation
# are checked without a GPU.
set -uo pipefail

SBATCH="${1:?usage: test_lanes.sh <path to paired-suite-eval.sbatch>}"
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

# Pull the two functions verbatim out of the sbatch.
awk '/^suite_is_current\(\) \{/,/^\}/' "$SBATCH" > "$WORK/fns.sh"
awk '/^count_alive\(\) \{/,/^\}/'     "$SBATCH" >> "$WORK/fns.sh"
for fn in suite_failed record_failure usable_suites report_failures require_usable_suites \
          variant_requested candidate_bind stale_suites capability_of \
          require_materialized; do
    awk -v f="^${fn}\\\\(\\\\) \\\\{" '$0 ~ f, /^\}/' "$SBATCH" >> "$WORK/fns.sh"
done
awk '/^score_variant\(\) \{/,/^\}/'   "$SBATCH" >> "$WORK/fns.sh"
# concat_variant and its exclusion test live in the shared comparison script now,
# so that a CPU-only job can rebuild a comparison too.
ROOT="$(cd "$(dirname "$SBATCH")/../.." && pwd)"
for fn in excluded concat_variant; do
    awk -v f="^${fn}\\\\(\\\\) \\\\{" '$0 ~ f, /^\}/' "$ROOT/eval/scripts/compare_run_dir.sh" >> "$WORK/fns.sh"
done
# Checked per source file: an awk pattern that silently matched nothing let the
# concat tests pass against an empty extraction.
grep -q "score_variant"  "$WORK/fns.sh" || { echo "could not extract from the sbatch"; exit 1; }
grep -q "score_variant"  "$WORK/fns.sh" || { echo "could not extract score_variant"; exit 1; }
grep -q "concat_variant" "$WORK/fns.sh" || { echo "could not extract from compare_run_dir.sh"; exit 1; }

cat > "$WORK/stub" <<'STUB'
#!/usr/bin/env bash
# arg1 "-" is suite_is_current's inline python; run it for real.
if [[ "${1:-}" == "-" ]]; then exec python3 "$@"; fi
case "${1:-}" in
    *gpqa_diamond.py) echo "probe ok"; exit 0 ;;
esac
# Which script a lane invoked is the only way to tell the runners apart.
echo "script=${1##*/}"
suite=""; rep=""; conc=""; scale=""; variant=""; tscale=""
while (( $# )); do
    case "$1" in
        --suite) suite="$2"; shift 2 ;;
        --replicate) rep="$2"; shift 2 ;;
        --concurrency) conc="$2"; shift 2 ;;
        --concurrency-scale) scale="$2"; shift 2 ;;
        --request-timeout-scale) tscale="$2"; shift 2 ;;
        --variant) variant="$2"; shift 2 ;;
        *) shift ;;
    esac
done
mkdir -p "$LANE_TRACE"
python3 -c "import time; print(time.time())" > "$LANE_TRACE/$suite-r$rep.start"
echo "suite=$suite rep=$rep variant=$variant conc=$conc scale=$scale tscale=$tscale"
sleep 0.4
python3 -c "import time; print(time.time())" > "$LANE_TRACE/$suite-r$rep.end"
[[ "$suite" == "failsuite" ]] && exit 3
exit 0
STUB
chmod +x "$WORK/stub"

setup() {
    RUN_DIR="$WORK/run"; rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR/logs"
    LANE_TRACE="$WORK/trace"; rm -rf "$LANE_TRACE"; mkdir -p "$LANE_TRACE"
    export LANE_TRACE
    CONFIG="$WORK/config.json"
    cat > "$CONFIG" <<'JSON'
{"suites":[
  {"name":"alpha","run":["x","run","--max-tokens","1000","--request-timeout","60"]},
  {"name":"beta","run":["x","run","--max-tokens","2000"]},
  {"name":"failsuite","run":["x","run"]}]}
JSON
    # Built exactly as the sbatch builds it, from whatever config the case wrote.
    KV_CLASSES="$(python3 - "$CONFIG" <<'PY'
import json, sys
for suite in json.load(open(sys.argv[1]))["suites"]:
    print(f"{suite['name']}\t{suite.get('kv_class', 'short')}")
PY
)"
    PYTHON="$WORK/stub"; BASE_URL="http://x"; SERVED_NAME="m"
    CONCURRENCY=""; CONCURRENCY_SCALE="0.5"; PAIRED_FORCE=0; TIMEOUT_SCALE="1.0"
    BASELINE_FP="sha256:aaa"; CANDIDATE_FP="sha256:bbb"
    BASELINE_INFO='{"label":"baseline","fingerprint":"sha256:aaa"}'
    CANDIDATE_INFO='{"label":"candidate","fingerprint":"sha256:bbb"}'
    FAILURE_LOG="$RUN_DIR/logs/suite-failures.tsv"
    FAILED_SUITES=()
    EXCLUDE_SUITES=""
    # concat_variant drops anything the eval suite does not define.
    DEFINED="alpha beta failsuite bfcl_v4"
    EVAL_SUITE_VERSION="vtest"
}

max_overlap() {
    python3 - "$LANE_TRACE" <<'PY'
import glob, os, sys
d = sys.argv[1]
ev = []
for s in glob.glob(os.path.join(d, "*.start")):
    e = s[:-6] + ".end"
    if not os.path.exists(e):
        continue
    ev.append((float(open(s).read()), 1))
    ev.append((float(open(e).read()), -1))
cur = best = 0
for _, delta in sorted(ev):
    cur += delta
    best = max(best, cur)
print(best)
PY
}

# shellcheck disable=SC1090
source "$WORK/fns.sh"

echo "== case 1: 2 suites x 2 replicates, unlimited lanes =="
setup; SUITES="alpha beta"; REPLICATES=2; PARALLEL=0
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "four lanes announced" 1 "$(grep -c 'across 4 lane' <<<"$out")"
check "every lane offers the full scale" 4 "$(grep -c 'scale=0.5' <<<"$out")"
check "replicate 0 and 1 both run" 2 "$(grep -c 'rep=1' <<<"$out")"
check "all four ran concurrently" 4 "$(max_overlap)"

echo "== case 2: PARALLEL caps the lanes =="
setup; SUITES="alpha beta"; REPLICATES=2; PARALLEL=2
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "two lanes announced" 1 "$(grep -c 'across 2 lane' <<<"$out")"
check "capping lanes does not narrow them" 4 "$(grep -c 'scale=0.5' <<<"$out")"
overlap="$(max_overlap)"
check "never exceeded 2 concurrent" "yes" "$([[ "$overlap" -le 2 ]] && echo yes || echo "no($overlap)")"

# Lanes do not ration the server between themselves. Two rounds of divisor --
# by the job, then by a declared kv_class -- both starved it, because the client
# cannot know what the server can hold and does not need to: vLLM admits what
# fits and queues the rest. A lane offers its suite's whole configured width and
# the scheduler decides. The fixture still declares classes so the tests can say
# the field no longer narrows anything.
mixed_classes() { # replace the fixture config with one that spans both classes
    cat > "$CONFIG" <<'JSON'
{"suites":[
  {"name":"alpha","kv_class":"short","run":["x","run"]},
  {"name":"beta","kv_class":"short","run":["x","run"]},
  {"name":"longsuite","kv_class":"long","run":["x","run"]}]}
JSON
}

# An inherited baseline is copied into a fresh run directory, and the copy used
# not to recurse, so the two image suites arrived without their images and failed
# on the first missing PNG -- per item, inside a lane, with the GPUs already
# allocated. The run then reported a macro over the four suites that survived.
echo "== case 2d: a run directory missing its datasets is refused up front =="
setup
mkdir -p "$RUN_DIR/materialized" "$RUN_DIR/orders"
for s in alpha mmmu_pro multimodal; do
    echo '{"id": 1}' > "$RUN_DIR/materialized/$s.jsonl"
    echo '[1]'       > "$RUN_DIR/orders/$s.json"
done
out="$(require_materialized alpha 2>&1)"; rc=$?
check "a suite with no images is fine" 0 "$rc"
out="$(require_materialized mmmu_pro 2>&1)"; rc=$?
check "an image suite with no image directory is refused" 1 "$rc"
check "and says how to fix it" 1 "$(grep -c 'run the prepare batch' <<<"$out")"
mkdir -p "$RUN_DIR/materialized/mmmu_pro-images"
out="$(require_materialized mmmu_pro 2>&1)"; rc=$?
check "an empty image directory is still refused" 1 "$rc"
: > "$RUN_DIR/materialized/mmmu_pro-images/a.png"
out="$(require_materialized mmmu_pro multimodal 2>&1)"; rc=$?
check "one image suite ready is not enough for the other" 1 "$rc"
check "and the other one is named" 1 "$(grep -c 'multimodal' <<<"$out")"
mkdir -p "$RUN_DIR/materialized/multimodal-images"
: > "$RUN_DIR/materialized/multimodal-images/b.png"
out="$(require_materialized alpha mmmu_pro multimodal 2>&1)"; rc=$?
check "all three ready passes" 0 "$rc"
rm -f "$RUN_DIR/materialized/alpha.jsonl"
out="$(require_materialized alpha 2>&1)"; rc=$?
check "a missing manifest is refused too" 1 "$rc"

# An inherited baseline names the card that produced it and nothing else, so the
# capability has to come from the name. Guessing is the failure that matters:
# treating an unknown card as equivalent to the one serving now is what lets an
# A100-measured baseline pair against an H200 candidate without complaint.
echo "== case 2e: a gpu name maps to a capability, or to nothing =="
check "A100 is 8.0, below the 8.9 fp8 threshold" "8.0" "$(capability_of 'NVIDIA A100 80GB PCIe')"
check "H200 is 9.0" "9.0" "$(capability_of 'NVIDIA H200 NVL')"
check "H100 is 9.0" "9.0" "$(capability_of 'NVIDIA H100 80GB HBM3')"
check "B200 is 10.0" "10.0" "$(capability_of 'NVIDIA B200')"
check "an unvalidated card is unknown, not assumed" "" "$(capability_of 'Tesla V100-SXM2-32GB')"
check "an empty name is unknown too" "" "$(capability_of '')"

echo "== case 2b: no lane is narrowed, by class or by the job =="
setup; mixed_classes
SUITES="alpha beta longsuite"; REPLICATES=1; PARALLEL=0
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "three lanes announced" 1 "$(grep -c 'across 3 lane' <<<"$out")"
check "all three offer the full scale" 3 "$(grep -c 'scale=0.5' <<<"$out")"
check "not divided by the class" 0 "$(grep -c 'scale=0.25' <<<"$out")"
check "not divided by the job" 0 "$(grep -c 'scale=0.16' <<<"$out")"

echo "== case 2c: an absolute concurrency is not split either =="
setup; mixed_classes; CONCURRENCY=120
SUITES="alpha beta longsuite"; REPLICATES=1; PARALLEL=0
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "every lane gets all of it" 3 "$(grep -c 'conc=120' <<<"$out")"
check "nothing was halved" 0 "$(grep -c 'conc=60' <<<"$out")"

# The sbatch calls score_variant directly, so the failure ledger it builds has
# to survive into the caller. Capturing through $(...) would run it in a subshell
# and hide exactly the bug that matters here.
run_variant() { # run_variant <variant>  -> sets $out, $rc
    score_variant "$1" > "$WORK/variant.out" 2>&1
    rc=$?
    out="$(cat "$WORK/variant.out")"
}

echo "== case 3: a failing suite is dropped, its siblings carry on =="
setup; SUITES="alpha failsuite"; REPLICATES=1; PARALLEL=0
run_variant candidate
check "exit 0" 0 "$rc"
check "failure named with its code" 1 "$(grep -c 'failsuite-r0 (candidate) failed with exit 3' <<<"$out")"
check "reported the drop" 1 "$(grep -c 'failsuite is dropped from the comparison' <<<"$out")"
check "reported at the moment it happened" 1 "$(grep -c 'failsuite-r0 (candidate) exited 3' <<<"$out")"
check "sibling log still printed" 1 "$(grep -c 'suite=alpha rep=0' <<<"$out")"
check "only the failure is dropped" "alpha" "$(usable_suites)"
check "ledger written" 1 "$(grep -c 'failsuite' "$RUN_DIR/logs/suite-failures.tsv")"

echo "== case 3b: a suite that failed on one variant is not rerun on the other =="
SUITES="alpha failsuite"; REPLICATES=1; PARALLEL=0
run_variant baseline
check "exit 0" 0 "$rc"
check "skipped on the second variant" 1 "$(grep -c 'failsuite (baseline) skipped' <<<"$out")"
check "no lane spent on it" 0 "$(grep -c 'suite=failsuite' <<<"$out")"
check "sibling still scored" 1 "$(grep -c 'suite=alpha rep=0 variant=baseline' <<<"$out")"

echo "== case 3c: the only suite failing leaves nothing to compare =="
setup; SUITES="failsuite"; REPLICATES=1; PARALLEL=0
run_variant candidate
check "score_variant still returns 0" 0 "$rc"
check "nothing usable" "" "$(usable_suites)"
out="$( (require_usable_suites) 2>&1 )"; rc=$?
check "the run is aborted" 1 "$rc"
check "said why" 1 "$(grep -c 'every suite failed; nothing left to compare' <<<"$out")"

echo "== case 3d: one failed replicate drops the whole suite =="
setup; SUITES="alpha failsuite"; REPLICATES=3; PARALLEL=0
run_variant candidate
check "exit 0" 0 "$rc"
check "every replicate is recorded" 3 "$(grep -c 'failsuite' "$RUN_DIR/logs/suite-failures.tsv")"
check "the suite is named once" 1 "${#FAILED_SUITES[@]}"
check "alpha survives whole" "alpha" "$(usable_suites)"

echo "== case 4: an already-scored replicate is reused, not rerun =="
setup; SUITES="alpha"; REPLICATES=2; PARALLEL=0
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":60,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "r0 reused" 1 "$(grep -c 'alpha r0 (candidate) already scored' <<<"$out")"
check "r1 still scored" 1 "$(grep -c 'rep=1' <<<"$out")"
check "only one lane left to run" 1 "$(grep -c 'across 1 lane' <<<"$out")"

echo "== case 5: stale cap invalidates the reuse =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":512,"request_timeout_seconds":60,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "not reused under a changed cap" 0 "$(grep -c 'already scored' <<<"$out")"
check "rescored" 1 "$(grep -c 'suite=alpha rep=0' <<<"$out")"

echo "== case 6: the hardware timeout scale reaches the scoring job =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0; TIMEOUT_SCALE="2.5"
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "scale forwarded" 1 "$(grep -c 'tscale=2.5' <<<"$out")"

echo "== case 7: a shorter timeout that never fired is still reusable =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0; TIMEOUT_SCALE="2.5"
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
# config asks 60 * 2.5 = 150s; this run had 60s but nothing timed out.
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":60,"timeouts":0,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "reused despite the shorter timeout" 1 "$(grep -c 'already scored' <<<"$out")"

# Only two adapters write request_timeout_seconds. Reading its absence as "not
# scored under this config" made the other four permanently unreusable: rescored
# on every job, and -- since stale_suites feeds the same predicate into the
# comparison's exclusions -- dropped from every macro ever produced. Both
# clusters had been reporting a two-suite macro without saying so.
echo "== case 7b: an unrecorded timeout that never fired is still reusable =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0; TIMEOUT_SCALE="2.5"
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"timeouts":0,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "reused with no recorded timeout" 1 "$(grep -c 'already scored' <<<"$out")"

echo "== case 7c: an unrecorded timeout that DID fire is not reusable =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0; TIMEOUT_SCALE="2.5"
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"timeouts":38,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "rescored, the threshold it hit is unknown" 0 "$(grep -c 'already scored' <<<"$out")"

echo "== case 7d: no timeout count at all establishes nothing =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0; TIMEOUT_SCALE="2.5"
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "rescored, nothing recorded either way" 0 "$(grep -c 'already scored' <<<"$out")"

echo "== case 8: a shorter timeout that DID fire is not reusable =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0; TIMEOUT_SCALE="2.5"
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":60,"timeouts":5,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "not reused when the clock scored the items" 0 "$(grep -c 'already scored' <<<"$out")"
check "rescored under the scaled timeout" 1 "$(grep -c 'suite=alpha rep=0' <<<"$out")"

echo "== case 9: a generous recorded timeout is reusable regardless =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0; TIMEOUT_SCALE="1.0"
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":9999,"timeouts":3,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "reused" 1 "$(grep -c 'already scored' <<<"$out")"

echo "== case 10: the telemetry sampler neither occupies a lane nor blocks the wait =="
setup; SUITES="alpha beta"; REPLICATES=1; PARALLEL=2
sleep 300 &                      # stands in for sample_telemetry.py
SAMPLER_PID=$!
start=$(python3 -c "import time; print(time.time())")
out="$(score_variant candidate 2>&1)"; rc=$?
took=$(python3 -c "import time; print(time.time() - $start)")
kill "$SAMPLER_PID" 2>/dev/null || true
check "exit 0" 0 "$rc"
check "returned promptly, not blocked on the sampler" "yes" \
    "$(python3 -c "print('yes' if $took < 30 else 'no($took)')")"
check "both lanes still ran" 2 "$(grep -cE 'suite=(alpha|beta) rep=0' <<<"$out")"
check "sampler did not steal a lane" 2 "$(max_overlap)"

echo "== case 11: results from a different checkpoint are not reused =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":60,"checkpoint":{"fingerprint":"sha256:SOMEONE_ELSE"}}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "not reused across checkpoints" 0 "$(grep -c 'already scored' <<<"$out")"
check "rescored" 1 "$(grep -c 'suite=alpha rep=0' <<<"$out")"

echo "== case 12: results with no recorded provenance are not reused =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0
mkdir -p "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":60}
JSON
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "unprovenanced results rejected" 0 "$(grep -c 'already scored' <<<"$out")"

echo "== case 13: the baseline is checked against the baseline checkpoint =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0
mkdir -p "$RUN_DIR/raw/baseline" "$RUN_DIR/metadata"
echo '{}' > "$RUN_DIR/raw/baseline/alpha-r0.jsonl"
cat > "$RUN_DIR/metadata/alpha-baseline-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":60,"checkpoint":{"fingerprint":"sha256:aaa"}}
JSON
out="$(score_variant baseline 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "baseline reused on its own fingerprint" 1 "$(grep -c 'already scored' <<<"$out")"

echo "== case 14: PAIRED_VARIANTS selects which arms a job scores =="
VARIANTS="baseline"
variant_requested baseline; check "baseline requested" 0 "$?"
variant_requested candidate; check "candidate not requested" 1 "$?"
VARIANTS="baseline candidate"
variant_requested candidate; check "both requested" 0 "$?"

echo "== case 15: scoring one arm does not blank the other arm's file =="
setup
mkdir -p "$RUN_DIR/raw/baseline"
echo '{"id":"a"}' > "$RUN_DIR/raw/baseline/alpha-r0.jsonl"
# Left by an earlier job that scored the candidate on another node.
echo '{"id":"prior"}' > "$RUN_DIR/candidate-all.jsonl"
concat_variant baseline > /dev/null 2>&1
concat_variant candidate > /dev/null 2>&1
check "baseline rebuilt" '{"id":"a"}' "$(cat "$RUN_DIR/baseline-all.jsonl")"
check "candidate left intact" '{"id":"prior"}' "$(cat "$RUN_DIR/candidate-all.jsonl")"
check "no temp files left behind" 0 "$(find "$RUN_DIR" -maxdepth 1 -name '.*-all.jsonl.*' | wc -l | tr -d ' ')"

echo "== case 16: a failed suite is still excluded when concatenating =="
setup
mkdir -p "$RUN_DIR/raw/baseline"
echo '{"id":"good"}' > "$RUN_DIR/raw/baseline/alpha-r0.jsonl"
echo '{"id":"bad"}'  > "$RUN_DIR/raw/baseline/failsuite-r0.jsonl"
EXCLUDE_SUITES=failsuite
concat_variant baseline > /dev/null 2>&1
check "failed suite excluded" '{"id":"good"}' "$(cat "$RUN_DIR/baseline-all.jsonl")"
EXCLUDE_SUITES=""

echo "== case 17: PAIRED_RUNNER picks which harness scores a lane =="
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0
RUNNER=evalscope
ES_SUITES=eval/evalscope-suites.json; ES_MAX_TOKENS=131072
ES_REQUEST_TIMEOUT=5400; ORDER_SEED=38027; SERVED_NAME=served
out="$(score_variant baseline 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "the evalscope bridge ran" 1 "$(grep -c 'script=evalscope_bridge.py' <<<"$out")"
check "run_adapter_suite did not" 0 "$(grep -c 'script=run_adapter_suite.py' <<<"$out")"
RUNNER=adapter
setup; SUITES="alpha"; REPLICATES=1; PARALLEL=0
out="$(score_variant baseline 2>&1)"; rc=$?
check "adapter runner still used by default" 1 "$(grep -c 'script=run_adapter_suite.py' <<<"$out")"
unset RUNNER

echo "== case 17b: a suite the eval suite no longer defines never reaches the macro =="
# A run directory outlives the protocol version that filled it. The comparator
# refuses a suite set it was not calibrated against, so these have to be dropped
# when the file is assembled, not left for it to reject.
setup
mkdir -p "$RUN_DIR/raw/baseline"
echo '{"id":"kept"}'    > "$RUN_DIR/raw/baseline/bfcl_v4-r0.jsonl"
echo '{"id":"dropped"}' > "$RUN_DIR/raw/baseline/matharena_2026_06-r0.jsonl"
DEFINED="bfcl_v4"   # matharena is not in it, standing in for a retired suite
out="$(concat_variant baseline 2>&1)"
check "in-protocol suite kept" '{"id":"kept"}' "$(cat "$RUN_DIR/baseline-all.jsonl")"
check "retired suite named" 1 "$(grep -c 'does not define it' <<<"$out")"

echo "== case 18: a candidate in the HF cache is bound through its repository root =="
# The snapshot directory is relative symlinks into ../../blobs. Bound on its own
# every file in it dangles, config.json included, and vLLM calls that an invalid
# model directory rather than a broken mount -- which sends you to check the pin.
HUB=/scratch/u/hf/hub/models--vendor--Qwen3.8-27B-AWQ
candidate_bind "$HUB/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea"
check "binds the repository root" "$HUB" "$CANDIDATE_BIND"
check "addresses the snapshot through it" \
    "/mnt/model/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea" "$CANDIDATE_MOUNT"

# The checkpoints we quantize ourselves are real directories, and rewriting
# those would break every run that works today.
candidate_bind /scratch/u/qwen38/v2/model-fp8gdn
check "a plain directory is bound as it is" "/scratch/u/qwen38/v2/model-fp8gdn" "$CANDIDATE_BIND"
check "and mounted at /mnt/model" "/mnt/model" "$CANDIDATE_MOUNT"

# Same shape as the baseline, which has always been bound this way. Reading the
# revision off the end rather than assuming one path depth is what keeps the two
# in agreement.
candidate_bind /a/models--x--y/snapshots/deadbeef/
check "a trailing slash does not change the root" "/a/models--x--y" "$CANDIDATE_BIND"
check "nor the revision" "/mnt/model/snapshots/deadbeef/" "$CANDIDATE_MOUNT"

echo "== case 19: a suite whose other arm is stale is dropped from the comparison =="
# The hazard this closes: a job that scores only the candidate pairs its fresh
# result against whatever the baseline left on disk. Keys match either way, so
# the macro would mix two measurement conditions and say nothing.
setup; SUITES="alpha"; REPLICATES=1
mkdir -p "$RUN_DIR/raw/baseline" "$RUN_DIR/raw/candidate" "$RUN_DIR/metadata"
echo '{"id":"b"}' > "$RUN_DIR/raw/baseline/alpha-r0.jsonl"
echo '{"id":"c"}' > "$RUN_DIR/raw/candidate/alpha-r0.jsonl"
# alpha is configured at --max-tokens 1000. The candidate was scored there; the
# baseline is left over from a run at 512.
cat > "$RUN_DIR/metadata/alpha-candidate-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":60,"checkpoint":{"fingerprint":"sha256:bbb"}}
JSON
cat > "$RUN_DIR/metadata/alpha-baseline-r0.json" <<'JSON'
{"max_tokens":512,"request_timeout_seconds":60,"checkpoint":{"fingerprint":"sha256:aaa"}}
JSON
out="$(stale_suites 2>&1)"
check "the suite is named stale" "alpha" "$(stale_suites 2>/dev/null)"
check "and says which arm" 1 "$(grep -c 'its baseline arm was not scored' <<<"$out")"

echo "== case 19b: both arms current leaves the comparison alone =="
cat > "$RUN_DIR/metadata/alpha-baseline-r0.json" <<'JSON'
{"max_tokens":1000,"request_timeout_seconds":60,"checkpoint":{"fingerprint":"sha256:aaa"}}
JSON
check "nothing excluded" "" "$(stale_suites 2>/dev/null)"

echo "== case 19c: a suite the eval suite does not define is dropped, not crashed on =="
echo '{"id":"x"}' > "$RUN_DIR/raw/baseline/retired-r0.jsonl"
check "named" "retired" "$(stale_suites 2>/dev/null)"

echo
echo "passed $PASS, failed $FAIL"
[[ "$FAIL" -eq 0 ]]
