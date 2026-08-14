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
export RUN_BASE
