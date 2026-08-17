#!/usr/bin/env bash
#SBATCH --job-name=qwen38-full-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=qwen38-full-eval-%j.out

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./common/scripts/load_env.sh
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_BASE/v2/model}"
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"
: "${EVAL_CONFIG:?set EVAL_CONFIG to a completed protocol JSON file}"
: "${EVAL_APPTAINER_IMAGE:?set EVAL_APPTAINER_IMAGE to the pinned vLLM .sif}"
: "${EVAL_BASE_URL:?set EVAL_BASE_URL to an endpoint reachable from Harbor containers, ending in /v1}"

EVAL_RUN_DIR="${EVAL_RUN_DIR:-$RUN_BASE/v2/eval/$(date -u +%Y%m%dT%H%M%SZ)}"
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

OVERLAP_ARGS=()
if [[ -n "${EVAL_OVERLAP_REVIEW:-}" ]]; then
    OVERLAP_ARGS=(--overlap-review "$EVAL_OVERLAP_REVIEW")
fi

python3 eval/scripts/run_eval_protocol.py \
    --config "$EVAL_CONFIG" \
    --image "$EVAL_APPTAINER_IMAGE" \
    --run-dir "$EVAL_RUN_DIR" \
    "${OVERLAP_ARGS[@]}" \
    --phase all
