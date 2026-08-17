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

# Datasets belong on scratch with everything else this run produces. Left unset,
# huggingface_hub caches under $HOME, which put 1.5 GB of benchmark data on a
# home filesystem while an 81 GB cache sat on scratch unused, and split the two
# so that a dataset fetched by one path was invisible to the other. That is how
# a gated dataset warmed into the scratch cache still failed to load inside a
# batch job: the job was looking somewhere else.
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"

export RUN_BASE
