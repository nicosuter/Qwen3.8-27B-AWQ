#!/usr/bin/env bash
#SBATCH --job-name=qwen38-awq-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=qwen38-awq-smoke-%j.out

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./common/scripts/load_env.sh
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_BASE/v2/model}"
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"
export SMOKE_DEADLINE_SECONDS=1200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate
test -s "$OUTPUT_DIR/config.json"
timeout --signal=TERM --kill-after=30s 20m ./quant/scripts/smoke_test.sh
timeout --signal=TERM --kill-after=30s 20m python quant/scripts/validate_generate.py
