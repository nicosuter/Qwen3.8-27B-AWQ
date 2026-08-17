#!/usr/bin/env python3
"""The one place that says where a checkpoint a campaign may score lives.

These paths used to be written out twice, once per cluster campaign file, as
`$HUB/models--<org>--<name>/snapshots/<sha>` string concatenation. Two copies of
a revision is one copy too many: the pair drifted, and a campaign that scores a
different snapshot than the one it claims to is not detectable from its output.

A campaign now names a checkpoint and this resolves it. Which checkpoints a
campaign runs stays a runtime decision, made on the command line; what each name
refers to is a recorded one and lives in `eval/checkpoints.json`.
"""

# Read from shell by whatever `python3` resolves to; on the clusters that is
# 3.9, where `Path | None` in a signature raises at import time.
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]


class CheckpointError(RuntimeError):
    pass


def load(root: Path | None = None) -> dict[str, Any]:
    path = (root or PROJECT_DIR) / "eval" / "checkpoints.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CheckpointError(f"no checkpoint registry at {path}") from error
    except json.JSONDecodeError as error:
        raise CheckpointError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(document.get("candidates"), dict) or not document["candidates"]:
        raise CheckpointError(f"{path} names no candidates")
    baseline = document.get("baseline")
    if not isinstance(baseline, dict) or not baseline.get("repo") or not baseline.get("revision"):
        raise CheckpointError(f"{path}: the baseline needs a repo and a revision")
    for name, entry in document["candidates"].items():
        described = ("repo" in entry and "revision" in entry) or "path" in entry
        if not described:
            raise CheckpointError(
                f"{path}: {name} needs either repo and revision, or a path under RUN_BASE"
            )
        if "repo" in entry and not entry.get("revision"):
            raise CheckpointError(f"{path}: {name} names a repo with no revision")
    return document


def names(root: Path | None = None) -> list[str]:
    return sorted(load(root)["candidates"])


def hub_dir(repo: str, run_base: Path) -> Path:
    """Where huggingface_hub keeps a repository's cache, given HF_HOME."""
    if repo.count("/") != 1:
        raise CheckpointError(f"{repo!r} is not an <org>/<name> repository id")
    home = Path(os.environ.get("HF_HOME") or run_base / "huggingface")
    return home / "hub" / ("models--" + repo.replace("/", "--"))


def baseline(run_base: Path, root: Path | None = None) -> dict[str, str]:
    """The repository root and revision the sbatch binds the baseline from.

    Deliberately not the snapshot directory: a snapshot is a farm of symlinks
    into ../../blobs, so binding it alone leaves every one of them dangling
    inside the container.
    """
    entry = load(root)["baseline"]
    return {
        "repo": str(hub_dir(entry["repo"], run_base)),
        "revision": entry["revision"],
    }


def candidate(name: str, run_base: Path, suite_version: str = "v1",
              root: Path | None = None) -> dict[str, str]:
    """Resolve one candidate to the directory to serve and its run directory."""
    document = load(root)
    entry = document["candidates"].get(name)
    if entry is None:
        raise CheckpointError(
            f"no checkpoint named {name!r}; the registry has {sorted(document['candidates'])}"
        )
    if "path" in entry:
        # Ours, written by the quantization side: a plain directory, served as
        # it stands.
        path = run_base / entry["path"]
    else:
        path = hub_dir(entry["repo"], run_base) / "snapshots" / entry["revision"]
    # A run directory holds exactly one raw/candidate tree, so it is per
    # candidate. The default is derived; a candidate whose results already sit
    # somewhere else records that instead of the campaign remembering it.
    run_dir = entry.get("run_dir") or f"eval-suite-{suite_version}-{name}"
    return {"path": str(path), "run_dir": str(run_base / "v2" / run_dir)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-base",
        type=Path,
        default=os.environ.get("RUN_BASE"),
        help="deployment root; defaults to $RUN_BASE",
    )
    parser.add_argument("--suite-version", default="v1")
    parser.add_argument("--names", action="store_true", help="list the candidate names")
    parser.add_argument(
        "--candidate",
        help="print a candidate's checkpoint path and run directory, tab separated",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="print the baseline's repository root and revision, tab separated",
    )
    args = parser.parse_args(argv)

    if args.names:
        print(" ".join(names()))
        return 0
    if args.baseline or args.candidate:
        if args.run_base is None:
            raise CheckpointError("set RUN_BASE or pass --run-base")
        if args.baseline:
            resolved = baseline(args.run_base)
            print(f"{resolved['repo']}\t{resolved['revision']}")
        else:
            resolved = candidate(args.candidate, args.run_base, args.suite_version)
            print(f"{resolved['path']}\t{resolved['run_dir']}")
        return 0
    document = load()
    print(f"baseline  {document['baseline']['repo']}@{document['baseline']['revision'][:12]}")
    for name in names():
        entry = document["candidates"][name]
        where = entry.get("path") or f"{entry['repo']}@{entry['revision'][:12]}"
        print(f"  {name:12s} {where}")
    return 0


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except CheckpointError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
