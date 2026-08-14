#!/usr/bin/env bash
#SBATCH --job-name=qwen38-awq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
# Default allocation for a direct `sbatch slurm/quantize.sbatch`;
# scripts/submit_quantize.sh overrides it from the command line.
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=qwen38-awq-%j.out

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

source ./scripts/load_env.sh
RUN_ROOT="${RUN_ROOT:-$RUN_BASE/v2}"
mkdir -p "$RUN_ROOT"
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"
export CALIBRATION_DIR="${CALIBRATION_DIR:-$RUN_ROOT/calibration}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/model}"
export MAX_SEQ_LENGTH=4096
export TOKENIZERS_PARALLELISM=true
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# FlightRecorder. A collective timeout otherwise reports only a sequence number;
# with this set the watchdog dumps the Python stack of the failed collective on
# every rank, which is what a save-phase divergence needs to be diagnosable.
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

source .venv/bin/activate
nvidia-smi
NGPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ ! "$NGPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Slurm exposed no CUDA GPUs" >&2
    exit 1
fi
export EXPECTED_WORLD_SIZE="$NGPUS"
mkdir -p "$RUN_ROOT/environment"
nvidia-smi -q > "$RUN_ROOT/environment/nvidia-smi.txt"
test -s "$CALIBRATION_DIR/manifest.jsonl"
test "$(wc -l < "$CALIBRATION_DIR/manifest.jsonl")" -eq 256
torchrun --standalone --nproc_per_node="$NGPUS" scripts/quantize.py
