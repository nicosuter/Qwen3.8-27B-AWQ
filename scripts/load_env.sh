#!/usr/bin/env bash

# Entry points source this after changing to the repository root. An explicitly
# exported RUN_BASE takes precedence over the local, gitignored .env file.
if [[ -z "${RUN_BASE:-}" && -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

: "${RUN_BASE:?Set RUN_BASE in .env (copy .env.example) or export it}"

# Qwen3.8-specific upstream vLLM image. Keep the platform manifest pinned: the
# floating qwen38 tag is multi-architecture and can change underneath an eval.
export EVAL_VLLM_BASE_IMAGE="${EVAL_VLLM_BASE_IMAGE:-docker://vllm/vllm-openai@sha256:d392f621bb3e372ecc09f0b0cb88099afe9fa05d37a0450de45eeb8c12b6787e}"
export EVAL_APPTAINER_IMAGE="${EVAL_APPTAINER_IMAGE:-$RUN_BASE/containers/qwen38-eval-vllm.sif}"
export RUN_BASE
