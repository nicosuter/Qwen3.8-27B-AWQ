#!/usr/bin/env bash
#SBATCH --job-name=qwen38-paired-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --time=00:25:00
#SBATCH --output=qwen38-paired-smoke-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source ./scripts/load_env.sh
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_BASE/v2/model}"
export SMOKE_EVAL_DIR="${SMOKE_EVAL_DIR:-$RUN_BASE/v2/smoke-eval}"
export EVAL_BASELINE_MODEL_ID="${EVAL_BASELINE_MODEL_ID:-Qwen/Qwen3.8-27B-FP8}"
export EVAL_BASELINE_MODEL_REVISION="${EVAL_BASELINE_MODEL_REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}"
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate
test -s "$OUTPUT_DIR/config.json"
./scripts/smoke_test.sh
timeout --signal=TERM --kill-after=30s 15m torchrun --standalone --nproc_per_node=4 scripts/paired_smoke_eval.py
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 5m python scripts/validate_generate.py
