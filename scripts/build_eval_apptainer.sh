#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFINITION="$PROJECT_DIR/apptainer/eval-vllm.def"

usage() {
    cat <<'EOF'
Usage: scripts/build_eval_apptainer.sh BASE_IMAGE OUTPUT.sif

BASE_IMAGE must be an immutable docker:// reference ending in @sha256:<64 hex>.
Example:
  scripts/build_eval_apptainer.sh \
    docker://vllm/vllm-openai@sha256:REPLACE_WITH_VERIFIED_DIGEST \
    /scratch/$USER/containers/qwen38-eval-vllm.sif
EOF
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 64
fi

BASE_IMAGE="$1"
OUTPUT_IMAGE="$2"
if [[ ! "$BASE_IMAGE" =~ ^docker://.+@sha256:[0-9a-fA-F]{64}$ ]]; then
    echo "error: BASE_IMAGE must be pinned by an sha256 digest" >&2
    exit 64
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "error: apptainer is not available" >&2
    exit 69
fi

mkdir -p "$(dirname "$OUTPUT_IMAGE")"
VLLM_BASE="${BASE_IMAGE#docker://}"
apptainer build --build-arg "VLLM_BASE=$VLLM_BASE" "$OUTPUT_IMAGE" "$DEFINITION"
sha256sum "$OUTPUT_IMAGE" > "$OUTPUT_IMAGE.sha256"
apptainer test "$OUTPUT_IMAGE"
echo "eval-image=$OUTPUT_IMAGE"
