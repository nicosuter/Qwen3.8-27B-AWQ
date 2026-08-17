#!/usr/bin/env sh
# Put a SWE-bench Pro task repository where the task means it to be, and remove
# the answer from it. Runs inside the task container, before the agent.
#
# Two jobs, and both are load-bearing.
#
# The tree has to move. These images are tagged at one commit and the task is
# posed at another: `tests/config.json` carries `base_commit`, and the task's
# Dockerfile resets to it. Using the published image without that step grades a
# tree the task never meant to hand the agent.
#
# The history has to go. `git reset --hard` moves HEAD; it removes nothing. On
# the image measured here the repository still held 67,370 commits against the
# 51,852 reachable from the task's base commit, with 61 remote branches and 636
# tags -- so `git log --all` shows the upstream fix, by title, with no network
# involved. That is the exploit reported upstream as git reward hacking, and it
# was reproduced there under `--network none`: denying the network does not
# touch it.
#
# What is deliberately NOT done is deleting .git. The verifier is git-shaped --
# `tests/config.json` carries the patch and `solution/solve.sh` writes a diff to
# apply -- so a repository that cannot run `git apply` cannot be graded. Upstream
# wrote a history-stripping script rather than an rm for the same reason.
set -eu

REPO="${1:?usage: swebenchpro_prepare_repo.sh <repo dir> <base commit>}"
BASE="${2:?usage: swebenchpro_prepare_repo.sh <repo dir> <base commit>}"

cd "$REPO"

# Detached first: refs/heads is deleted below, and a branch cannot be deleted
# while it is checked out. Any local branch left pointing past the base commit
# would keep the fix reachable on its own.
git checkout --quiet --detach "$BASE"
git reset --quiet --hard "$BASE"
# -fd, not -fdx, matching the task Dockerfile: -x would also remove ignored
# files, and these images keep installed dependencies among them.
git clean -qfd

for remote in $(git remote); do
    git remote remove "$remote"
done
git for-each-ref --format='%(refname)' refs/remotes refs/tags refs/heads \
    | while read -r ref; do git update-ref -d "$ref"; done
git reflog expire --expire=now --all
git gc --prune=now --quiet

# `--all` counts every ref plus HEAD, and every ref is gone, so this is the
# check that the future is actually unreachable rather than merely unreferenced.
# Failing here is the point: a silent partial strip leaves the answer in place
# and nothing downstream would notice.
reachable="$(git rev-list --count HEAD)"
present="$(git rev-list --count --all)"
if [ "$reachable" != "$present" ]; then
    echo "strip failed: $present commits present, $reachable reachable from HEAD" >&2
    exit 1
fi

echo "prepared $REPO at $(git rev-parse --short HEAD): $reachable commits, no remotes, no tags"
