#!/usr/bin/env bash
#SBATCH --job-name=qwen38-eval-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=qwen38-eval-image-%j.out

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

: "${EVAL_VLLM_BASE_IMAGE:?set EVAL_VLLM_BASE_IMAGE to docker://...@sha256:...}"
: "${EVAL_APPTAINER_IMAGE:?set EVAL_APPTAINER_IMAGE to the output .sif path}"

./scripts/build_eval_apptainer.sh "$EVAL_VLLM_BASE_IMAGE" "$EVAL_APPTAINER_IMAGE"
