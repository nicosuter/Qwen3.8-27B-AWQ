#!/usr/bin/env bash
# What is running, by what it measures.
#
# `squeue` shows the job name, and for a campaign the job name is a slot --
# `<prefix>-<arch>-s<n>` -- not a description. That is forced:
# `--dependency=singleton` serialises jobs that share a name and keys on
# nothing else, so a quota of one lane means one name, and the name cannot also
# say which checkpoint is being scored. Two lanes measuring different
# checkpoints have to share it or they would run at once.
#
# So the lane travels in --comment, which squeue prints as %k and nobody passes.
# This is that invocation, and it exists because reading a queue and seeing six
# identical names is how you lose track of a campaign.
set -euo pipefail

printf '%-8s %-3s %10s %10s  %-26s %s\n' JOBID ST ELAPSED LIMIT MEASURING WHERE
squeue -u "${1:-$USER}" -h -o '%i|%t|%M|%l|%k|%R|%j' | while IFS='|' read -r id st elapsed limit comment where name; do
    # Jobs submitted before comments were set, and anything submitted by hand,
    # still have only a name. Showing it beats showing an empty column.
    [[ -z "$comment" || "$comment" == "(null)" ]] && comment="$name"
    printf '%-8s %-3s %10s %10s  %-26s %s\n' \
        "$id" "$st" "$elapsed" "$limit" "$comment" "$where"
done
