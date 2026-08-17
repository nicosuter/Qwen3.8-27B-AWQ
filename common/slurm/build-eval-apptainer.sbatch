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

source ./common/scripts/load_env.sh

# Cache the 7+ GiB OCI download and the expanded build workspace on persistent
# scratch. A retry should reuse layers instead of starting from zero.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$RUN_BASE/cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$RUN_BASE/tmp/apptainer}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

./common/scripts/build_eval_apptainer.sh "$EVAL_VLLM_BASE_IMAGE" "$EVAL_APPTAINER_IMAGE"
