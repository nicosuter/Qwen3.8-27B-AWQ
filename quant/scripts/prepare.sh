#!/usr/bin/env bash
set -euo pipefail

source ./common/scripts/load_env.sh
RUN_ROOT="${RUN_ROOT:-$RUN_BASE/v2}"
export RUN_ROOT
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"
export CALIBRATION_DIR="${CALIBRATION_DIR:-$RUN_ROOT/calibration-v2}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/model}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

./common/scripts/bootstrap.sh
source .venv/bin/activate
./quant/scripts/capture_environment.sh
python quant/scripts/preflight.py
python quant/scripts/build_calibration.py
