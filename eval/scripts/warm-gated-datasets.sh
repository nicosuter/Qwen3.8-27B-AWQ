#!/usr/bin/env bash
# Populate the Hugging Face cache on one or more hosts with the gated datasets
# this protocol needs, so that batch jobs never need a token.
#
#   ./eval/scripts/with-hf-token.sh -- ./eval/scripts/warm-gated-datasets.sh HOST [HOST...]
#
# Host names are arguments and are never written down here. Which machines a
# deployment owns is not a property of the protocol, the same reason the
# interpreter, the RULER corpus and the tokenizer are named rather than spelled
# out in the suite configs.
#
# The token reaches each host on stdin, never in argv. A remote command line is
# world-readable through ps for as long as it runs, so passing a credential that
# way publishes it to every user on the node. Because stdin is spoken for, the
# remote script cannot arrive by `bash -s` the usual way; it is base64'd into
# argv instead, which is safe precisely because it holds no secret.
#
# Nothing is written to disk: HF_HUB_DISABLE_IMPLICIT_TOKEN_PERSISTENCE stops
# huggingface_hub caching the token under ~/.cache, which is what separates this
# from `hf login`. Only the dataset stays behind, which is the point. Gated data
# has to be fetched once by a human who accepted its terms; after that the cache
# serves it and eval/scripts/with-hf-token.sh can go on refusing to hand tokens to
# sbatch.

set -euo pipefail

# Gated datasets in the protocol, as repo:config:split. GPQA is the only one;
# MMMU-Pro, BFCL, LiveCodeBench, MathArena and the rest are public.
DATASETS=("Idavidrein/gpqa:gpqa_diamond:train")
# Resolved on the far side, so $USER is the remote user.
REMOTE_ROOT='${REMOTE_ROOT_OVERRIDE:-/scratch/$USER/qwen38-27b-awq}'

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -lt 1 ]]; then
    echo "usage: $0 [--dry-run] HOST [HOST...]" >&2
    echo "  run under eval/scripts/with-hf-token.sh so HF_TOKEN is set" >&2
    exit 64
fi
if [[ -z "${HF_TOKEN:-}" && "$DRY_RUN" -eq 0 ]]; then
    echo "error: HF_TOKEN is not set; run this under eval/scripts/with-hf-token.sh" >&2
    exit 64
fi

read -r -d '' REMOTE <<'REMOTE_SCRIPT' || true
set -euo pipefail
# The first line of stdin is the token and nothing else reads stdin after it.
read -rs TOKEN
export HF_TOKEN="$TOKEN" HUGGING_FACE_HUB_TOKEN="$TOKEN"
export HF_HUB_DISABLE_IMPLICIT_TOKEN_PERSISTENCE=1
unset TOKEN
ROOT="$1"; shift
PY="$ROOT/venv/bin/python"
test -x "$PY" || { echo "no interpreter at $PY" >&2; exit 1; }
export HF_HOME="${HF_HOME:-$ROOT/huggingface}"
for spec in "$@"; do
    IFS=: read -r repo config split <<<"$spec"
    "$PY" -c '
import sys
from datasets import load_dataset
repo, config, split = sys.argv[1:4]
print(f"cached {repo} {config} {split}: {len(load_dataset(repo, config, split=split))} rows")
' "$repo" "$config" "$split"
done
REMOTE_SCRIPT

ENCODED="$(printf '%s' "$REMOTE" | base64 | tr -d '\n')"
# Decode to a file rather than piping into bash, so the remote script's own stdin
# stays free for the token.
COMMAND="T=\$(mktemp); trap 'rm -f \$T' EXIT; printf %s $ENCODED | base64 -d > \$T"
COMMAND="$COMMAND; bash \$T \"$REMOTE_ROOT\" ${DATASETS[*]@Q}"

status=0
for host in "$@"; do
    echo "=== $host"
    if (( DRY_RUN )); then
        echo "    would run ${#REMOTE} bytes of script for: ${DATASETS[*]}"
        continue
    fi
    if printf '%s\n' "$HF_TOKEN" \
        | ssh -o BatchMode=yes "$host" "$COMMAND" 2>&1 | sed 's/^/    /'; then
        echo "    ok"
    else
        echo "    FAILED" >&2
        status=1
    fi
done
exit "$status"
