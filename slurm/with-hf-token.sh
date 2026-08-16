#!/usr/bin/env bash
# Run a command with an HF token that exists only in this process.
#
#     slurm/with-hf-token.sh -- <command> [args...]
#
# `cais/hle` and `Idavidrein/gpqa` are gated, so materializing them needs
# credentials the cluster does not have and should not keep.
#
# The token is read with echo off, never written to disk, never passed as an
# argument, and is gone when this process exits. It is not in your shell history
# because you never typed it as a command, and not in the environment of
# anything but the child.
#
# What this cannot do is make a *Slurm job* memory-only. `--export=ALL` copies
# your environment into the job record, which slurmctld and slurmd persist to
# their spool directories, and `--export=HF_TOKEN=...` is worse because it lands
# in argv where `scontrol show job` will print it.
#
# So do not send the token to a job. The only step that needs it is the
# download, which needs no GPU:
#
#     slurm/with-hf-token.sh -- python scripts/evalscope_bridge.py materialize \
#         --repo cais/hle --revision <sha> --into eval-materialized/evalscope
#
# After that the data is on shared storage and every job reads it as a local
# path, with no credential anywhere near the batch system.
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
    shift
fi
if (( $# == 0 )); then
    sed -n '2,26p' "$0" >&2
    exit 2
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is already set in this shell; using it." >&2
else
    # -s so it is not echoed, -r so a backslash is not eaten. Reading from the
    # terminal rather than stdin keeps this usable inside a pipeline.
    if [[ -r /dev/tty ]]; then
        read -rsp "Hugging Face token (input hidden): " HF_TOKEN < /dev/tty
        echo >&2
    else
        echo "no terminal to prompt on; export HF_TOKEN yourself" >&2
        exit 1
    fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "no token given" >&2
    exit 1
fi

# Both names exist in the wild; huggingface_hub reads HF_TOKEN first.
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# Do not let the library helpfully persist it to ~/.cache/huggingface/token.
export HF_HUB_DISABLE_IMPLICIT_TOKEN_PERSISTENCE=1

# exec so the token lives in exactly one process and dies with it.
exec "$@"
