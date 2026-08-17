#!/usr/bin/env bash
# What is running, with the columns that matter and none of the ones that don't.
#
# This existed because a campaign job's slurm name used to be a quota slot --
# `<prefix>-<arch>-s<n>` -- and six identically named jobs is how you lose track
# of a campaign. The quota is held by an afterany chain now, so the name says
# what the job measures and squeue is readable on its own.
#
# What is left is formatting: the lane key still travels in --comment, which
# squeue prints as %k and nobody passes, and it is shorter than the job name.
set -euo pipefail

printf '%-8s %-3s %10s %10s  %-26s %s\n' JOBID ST ELAPSED LIMIT MEASURING WHERE
squeue -u "${1:-$USER}" -h -o '%i|%t|%M|%l|%k|%R|%j' | while IFS='|' read -r id st elapsed limit comment where name; do
    # Jobs submitted before comments were set, and anything submitted by hand,
    # still have only a name. Showing it beats showing an empty column.
    [[ -z "$comment" || "$comment" == "(null)" ]] && comment="$name"
    printf '%-8s %-3s %10s %10s  %-26s %s\n' \
        "$id" "$st" "$elapsed" "$limit" "$comment" "$where"
done
