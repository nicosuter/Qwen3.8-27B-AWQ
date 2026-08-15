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
awk '/^score_variant\(\) \{/,/^\}/'   "$SBATCH" >> "$WORK/fns.sh"
grep -q "score_variant" "$WORK/fns.sh" || { echo "could not extract functions"; exit 1; }

cat > "$WORK/stub" <<'STUB'
#!/usr/bin/env bash
# arg1 "-" is suite_is_current's inline python; run it for real.
if [[ "${1:-}" == "-" ]]; then exec python3 "$@"; fi
case "${1:-}" in
    *gpqa_diamond.py) echo "probe ok"; exit 0 ;;
esac
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
    PYTHON="$WORK/stub"; BASE_URL="http://x"; SERVED_NAME="m"
    CONCURRENCY=""; CONCURRENCY_SCALE="0.5"; PAIRED_FORCE=0; TIMEOUT_SCALE="1.0"
    BASELINE_FP="sha256:aaa"; CANDIDATE_FP="sha256:bbb"
    BASELINE_INFO='{"label":"baseline","fingerprint":"sha256:aaa"}'
    CANDIDATE_INFO='{"label":"candidate","fingerprint":"sha256:bbb"}'
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
check "scale split 4 ways" 4 "$(grep -c 'scale=0.125' <<<"$out")"
check "replicate 0 and 1 both run" 2 "$(grep -c 'rep=1' <<<"$out")"
check "all four ran concurrently" 4 "$(max_overlap)"

echo "== case 2: PARALLEL caps the lanes =="
setup; SUITES="alpha beta"; REPLICATES=2; PARALLEL=2
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 0" 0 "$rc"
check "two lanes announced" 1 "$(grep -c 'across 2 lane' <<<"$out")"
check "scale split 2 ways" 4 "$(grep -c 'scale=0.25' <<<"$out")"
overlap="$(max_overlap)"
check "never exceeded 2 concurrent" "yes" "$([[ "$overlap" -le 2 ]] && echo yes || echo "no($overlap)")"

echo "== case 3: a failing lane fails the variant but still reports siblings =="
setup; SUITES="alpha failsuite"; REPLICATES=1; PARALLEL=0
out="$(score_variant candidate 2>&1)"; rc=$?
check "exit 1" 1 "$rc"
check "failure named with its code" 1 "$(grep -c 'failsuite-r0 (candidate) failed with exit 3' <<<"$out")"
check "sibling log still printed" 1 "$(grep -c 'suite=alpha rep=0' <<<"$out")"

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

echo
echo "passed $PASS, failed $FAIL"
[[ "$FAIL" -eq 0 ]]
