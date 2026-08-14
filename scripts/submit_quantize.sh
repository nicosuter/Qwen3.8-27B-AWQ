#!/usr/bin/env bash
set -euo pipefail

NGPUS="${1:?usage: bash scripts/submit_quantize.sh NGPUS}"
case "$NGPUS" in
    *[!0-9]* | 0 | "")
        echo "NGPUS must be a positive integer, got: $NGPUS" >&2
        exit 2
        ;;
esac

exec sbatch --gres="gpu:$NGPUS" slurm/quantize.sbatch
