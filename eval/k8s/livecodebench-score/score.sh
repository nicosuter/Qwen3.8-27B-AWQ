#!/usr/bin/env bash
# Score one deferred LiveCodeBench arm on Kubernetes.
#
# The manifests next door describe the containment. This is the part that was
# only ever a shell history: staging the inputs onto the work volume, pinning the
# adapter, running the job and taking the results back. Three arms had been
# scored that way before it was written down, each leaving differently named
# files on the volume, which is how you end up unable to say which generations a
# results file came from.
#
# Reads nothing from the GPU cluster and writes nothing to it. The caller brings
# the three input files; where they came from is the caller's problem.
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: score.sh --run NAME --generations FILE --meta FILE --key FILE --out DIR

  --run NAME       names the job and its directory on the work volume
  --generations    <suite>-<variant>-r<n>.jsonl from `run --defer-execution`
  --meta           its .meta.json, which records the adapter that produced it
  --key            the suite's answer key, from the run directory's materialized/
  --out DIR        where results.jsonl and metadata.json are written

  LCB_ADAPTER_DIR  adapter to score with (default eval/scripts/adapters). Its
                   pin must equal the one the generations recorded; scoring with
                   a different adapter than generated is not the same measurement.
USAGE
    exit 2
}

RUN="" GENERATIONS="" META="" KEY="" OUT=""
while (( $# )); do
    case "$1" in
        --run)         RUN="$2"; shift 2 ;;
        --generations) GENERATIONS="$2"; shift 2 ;;
        --meta)        META="$2"; shift 2 ;;
        --key)         KEY="$2"; shift 2 ;;
        --out)         OUT="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done
[[ -n "$RUN" && -n "$GENERATIONS" && -n "$META" && -n "$KEY" && -n "$OUT" ]] || usage
for f in "$GENERATIONS" "$META" "$KEY"; do
    test -s "$f" || { echo "$f is missing or empty" >&2; exit 1; }
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
NS=qwen-lcb-score
ADAPTER_DIR="${LCB_ADAPTER_DIR:-$ROOT/eval/scripts/adapters}"

# module_pin, in the form the adapter records it: the file name and the file
# bytes of every source whose contents can change a verdict.
pin() {
    python3 - "$1" <<'PY'
import hashlib, sys
from pathlib import Path
digest = hashlib.sha256()
for path in sorted(Path(sys.argv[1]).resolve() / name
                   for name in ("_common.py", "livecodebench.py")):
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
print("sha256:" + digest.hexdigest())
PY
}

WANT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter"])' "$META")"
HAVE="$(pin "$ADAPTER_DIR")"
if [[ "$WANT" != "$HAVE" ]]; then
    cat >&2 <<EOF
these generations were produced by adapter $WANT
$ADAPTER_DIR is           $HAVE

Scoring with a different adapter than generated is a different measurement, so
this stops here. Extract the one that produced them and point LCB_ADAPTER_DIR at
it -- the commit is recorded in the run directory's code.json:

    mkdir -p /tmp/lcb-adapter && cd "\$(git rev-parse --show-toplevel)"
    git show <commit>:eval/scripts/adapters/livecodebench.py > /tmp/lcb-adapter/livecodebench.py
    git show <commit>:eval/scripts/adapters/_common.py      > /tmp/lcb-adapter/_common.py
    LCB_ADAPTER_DIR=/tmp/lcb-adapter $0 ...
EOF
    exit 1
fi
echo "adapter pin $HAVE matches the generations"

kubectl apply -f "$HERE/00-namespace.yaml" -f "$HERE/10-networkpolicy.yaml" -f "$HERE/20-pvc.yaml" >&2
kubectl -n "$NS" delete pod lcb-stage --ignore-not-found >&2
kubectl apply -f "$HERE/40-stage.yaml" >&2
kubectl -n "$NS" wait --for=condition=Ready pod/lcb-stage --timeout=300s >&2

stage() { kubectl -n "$NS" exec lcb-stage -- "$@"; }
# COPYFILE_DISABLE keeps bsdtar from writing an AppleDouble ._file beside every
# upload from a mac, which is what littered the volume the first three times.
put() { COPYFILE_DISABLE=1 kubectl cp "$1" "$NS/lcb-stage:$2"; }

stage mkdir -p "/work/$RUN"
put "$GENERATIONS" "/work/$RUN/generations.jsonl"
put "$META"        "/work/$RUN/generations.meta.json"

# The key is the verifier and it is hundreds of megabytes, so it is shared
# between runs and checked rather than re-uploaded. Checked, because a shared
# input that is silently the wrong one grades every arm against it.
KEY_NAME="$(basename "$KEY")"
sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || sha256sum "$1" | cut -d' ' -f1; }
WANT_KEY="$(sha "$KEY")"
HAVE_KEY="$(stage python3 -c "
import hashlib, pathlib, sys
p = pathlib.Path('/work/$KEY_NAME')
if not p.exists():
    print('absent'); sys.exit()
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
        h.update(block)
print(h.hexdigest())" | tr -d '\r')"
if [[ "$HAVE_KEY" == "$WANT_KEY" ]]; then
    echo "answer key already on the volume and identical ($WANT_KEY)"
else
    echo "uploading answer key ($HAVE_KEY -> $WANT_KEY)"
    put "$KEY" "/work/$KEY_NAME"
fi

# Named for the pin, so two arms generated by different adapters cannot share a
# configmap and quietly be scored by whichever was applied last.
SHORT="${HAVE#sha256:}"; SHORT="${SHORT:0:7}"
kubectl create configmap "lcb-adapter-$SHORT" -n "$NS" \
    --from-file=livecodebench.py="$ADAPTER_DIR/livecodebench.py" \
    --from-file=_common.py="$ADAPTER_DIR/_common.py" \
    --dry-run=client -o yaml | kubectl apply -f - >&2

JOB="lcb-score-$RUN"
kubectl -n "$NS" delete job "$JOB" --ignore-not-found >&2
sed -e "s/name: lcb-score\$/name: $JOB/" \
    -e "s#--generations=/work/generations.jsonl#--generations=/work/$RUN/generations.jsonl#" \
    -e "s#--key=/work/livecodebench_v6.key.json#--key=/work/$KEY_NAME#" \
    -e "s#--results=/work/results.jsonl#--results=/work/$RUN/results.jsonl#" \
    -e "s#--metadata=/work/metadata.json#--metadata=/work/$RUN/metadata.json#" \
    -e "s/configMap: {name: lcb-adapter}/configMap: {name: lcb-adapter-$SHORT}/" \
    "$HERE/30-job.yaml" | kubectl apply -f - >&2

kubectl -n "$NS" wait --for=condition=complete "job/$JOB" --timeout="${LCB_TIMEOUT:-3600}s" >&2
kubectl -n "$NS" logs "job/$JOB" --tail=5 >&2

mkdir -p "$OUT"
COPYFILE_DISABLE=1 kubectl cp "$NS/lcb-stage:/work/$RUN/results.jsonl" "$OUT/results.jsonl"
COPYFILE_DISABLE=1 kubectl cp "$NS/lcb-stage:/work/$RUN/metadata.json"  "$OUT/metadata.json"

# A results file that still says deferred is the failure this whole split exists
# to make impossible to miss: it would concatenate, reach the comparator, and be
# refused there instead of here.
python3 - "$OUT/results.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
left = sum(1 for row in rows if row.get("deferred"))
if not rows or left:
    sys.exit(f"{len(rows)} rows, {left} still deferred")
print(f"scored {len(rows)} items, pass@1 {sum(r['score'] for r in rows) / len(rows):.4f}")
PY
echo "results in $OUT; copy them into the run directory as raw/<variant>/<suite>-r<n>.jsonl"
