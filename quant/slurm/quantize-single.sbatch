#!/usr/bin/env bash
#SBATCH --job-name=qwen38-awq-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=qwen38-awq-single-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source ./common/scripts/load_env.sh
RUN_ROOT="${RUN_ROOT:-$RUN_BASE/v2}"
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"
export CALIBRATION_DIR="${CALIBRATION_DIR:-$RUN_ROOT/calibration}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/model}"
export MAX_SEQ_LENGTH=4096
export CALIBRATION_SAMPLES=256
export TOKENIZERS_PARALLELISM=true
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate
test -s "$CALIBRATION_DIR/manifest.jsonl"
test "$(wc -l < "$CALIBRATION_DIR/manifest.jsonl")" -eq 256
nvidia-smi
NGPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
test "$NGPUS" -eq 1
export EXPECTED_WORLD_SIZE="$NGPUS"
mkdir -p "$RUN_ROOT/environment"
nvidia-smi -q > "$RUN_ROOT/environment/nvidia-smi.txt"
torchrun --standalone --nproc_per_node="$NGPUS" quant/scripts/quantize.py
