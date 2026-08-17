#!/usr/bin/env bash
# Remove commit checkouts nothing is using.
#
# Safe to be aggressive: the repository is public and under a megabyte, so any
# commit can be fetched again, and every run records the commit it used. What
# must never be deleted is a checkout a job is running from, and Slurm is the
# authority on that -- not a lock file, because a job killed by a time limit or
# a node failure never gets to clean one up.
set -euo pipefail

RUN_BASE="${RUN_BASE:-/scratch/$USER/qwen38-27b-awq}"
KEEP_DAYS="${KEEP_DAYS:-14}"
KEEP_RECENT="${KEEP_RECENT:-5}"
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

CODE_DIR="$RUN_BASE/code"
test -d "$CODE_DIR" || { echo "no $CODE_DIR"; exit 0; }

# Every commit named by a queued or running job of ours, read out of the working
# directory Slurm recorded for it.
declare -A IN_USE=()
while read -r workdir; do
    [[ "$workdir" == "$CODE_DIR"/* ]] || continue
    IN_USE["$(basename "${workdir#$CODE_DIR/}")"]=1
done < <(squeue -h -u "$USER" -o "%Z" 2>/dev/null || true)

mapfile -t RECENT < <(ls -1dt "$CODE_DIR"/*/ 2>/dev/null | head -n "$KEEP_RECENT" | xargs -r -n1 basename)
declare -A KEEP_NEW=()
for sha in ${RECENT[@]+"${RECENT[@]}"}; do KEEP_NEW["$sha"]=1; done

removed=0
for path in "$CODE_DIR"/*/; do
    [[ -d "$path" ]] || continue
    sha="$(basename "$path")"
    if [[ -n "${IN_USE[$sha]:-}" ]]; then
        echo "keep    $sha  (a job is running from it)"
    elif [[ -n "${KEEP_NEW[$sha]:-}" ]]; then
        echo "keep    $sha  (one of the $KEEP_RECENT most recent)"
    elif [[ -n "$(find "$path" -maxdepth 0 -mtime "-$KEEP_DAYS")" ]]; then
        echo "keep    $sha  (used within $KEEP_DAYS days)"
    else
        echo "remove  $sha"
        (( APPLY )) && rm -rf "$path"
        removed=$(( removed + 1 ))
    fi
done

if (( APPLY )); then
    echo "removed $removed checkout(s)"
else
    echo "$removed checkout(s) would be removed; rerun with --apply"
fi
