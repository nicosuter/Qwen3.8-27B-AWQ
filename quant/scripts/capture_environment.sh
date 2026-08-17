#!/usr/bin/env bash
set -euo pipefail

source ./common/scripts/load_env.sh
RUN_ROOT="${RUN_ROOT:-$RUN_BASE/v2}"
mkdir -p "$RUN_ROOT/environment"
python -m pip freeze > "$RUN_ROOT/environment/pip-freeze.txt"
python --version > "$RUN_ROOT/environment/python-version.txt" 2>&1
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -q > "$RUN_ROOT/environment/nvidia-smi.txt" 2>/dev/null; then
    :
else
    printf '%s\n' "GPU metadata unavailable in this allocation; captured by quantize.sbatch." \
        > "$RUN_ROOT/environment/nvidia-smi.txt"
fi
cp requirements.txt "$RUN_ROOT/environment/requirements.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$RUN_ROOT/environment/repository-commit.txt"
fi
