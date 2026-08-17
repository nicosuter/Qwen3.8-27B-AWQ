#!/usr/bin/env python3
"""Pick a reproducible SWE-bench Pro subset, sized to a wall-clock budget.

All 731 instances cost roughly 34 H200-hours paired, which is more than an
agentic supporting suite is worth next to the suites that actually decide the
gate. So we run a sample. A sample only means anything if it is fixed before
anyone sees a score, which is what this produces: a deterministic ordering of
the full task set, pinned by digest, from which the protocol takes a prefix.

The ordering is nested, and that is the point of it. Every prefix is itself a
repo-proportional sample, so a calibration run can measure the real turn count
on the first 24 tasks and the campaign can then be cut to 240 or grown to 360
without picking a new sample or explaining why the sample changed. Each item
gets priority `(rank - 0.5) / repo_size` and the whole set sorts by it, which
is the same largest-remainder logic apportionment uses: at any cut point every
repo has been served in proportion to its size.

Stratifying by repo rather than sampling flat matters because the repos are the
language split. SWE-bench Pro is 36% Python, 38% Go and 25% JS/TS, and those
live in disjoint repos, so a flat sample of 300 would let language mix wander
by several points between two runs that were supposed to be comparable.

The registry file is hashed, not trusted. Harbor publishes task lists that
change as adapters land, and a subset drawn from a different 731 is a different
subset no matter what it is called.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

DATASET = "swebenchpro"
VERSION = "1.0"
DEFAULT_SEED = 38027
# `instance_<owner>__<repo>-<base sha>` and then, for some rows, `-v<sha>` or
# `-vnan`. Only the leading base commit is reliably present, so the repo is
# whatever precedes the first 40-hex run.
TASK_RE = re.compile(r"^instance_(?P<repo>.+?)-[0-9a-f]{40}(?:$|-)")


class SubsetError(RuntimeError):
    pass


def task_repo(name: str) -> str:
    match = TASK_RE.match(name)
    if not match:
        raise SubsetError(f"cannot read a repo out of task name {name!r}")
    return match.group("repo")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_registry(path: Path, dataset: str, version: str) -> tuple[list[str], str]:
    """Return the dataset's task names and the digest of the registry bytes."""
    raw = path.read_bytes()
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SubsetError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(entries, list):
        raise SubsetError(f"{path}: expected a list of dataset entries")

    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("name") == dataset
        and str(entry.get("version")) == version
    ]
    if not matches:
        raise SubsetError(f"{path}: no entry for {dataset}@{version}")
    if len(matches) > 1:
        raise SubsetError(f"{path}: {len(matches)} entries for {dataset}@{version}")

    tasks = matches[0].get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SubsetError(f"{dataset}@{version} has no tasks")
    names = [str(task.get("name", "")) for task in tasks]
    if any(not name for name in names):
        raise SubsetError(f"{dataset}@{version} has a task with no name")
    if len(set(names)) != len(names):
        raise SubsetError(f"{dataset}@{version} has duplicate task names")
    return names, "sha256:" + hashlib.sha256(raw).hexdigest()


def nested_order(names: list[str], seed: int) -> list[str]:
    """Order the full set so that every prefix is repo-proportional.

    Within a repo the order is a hash permutation of the names, so it does not
    inherit whatever order the registry happened to list them in.
    """
    by_repo: dict[str, list[str]] = {}
    for name in names:
        by_repo.setdefault(task_repo(name), []).append(name)

    ranked = []
    for repo, members in by_repo.items():
        shuffled = sorted(members, key=lambda name: digest(f"{seed}:{name}"))
        size = len(shuffled)
        for rank, name in enumerate(shuffled, start=1):
            # Ties are possible when two repos have proportional sizes, so the
            # hash is a tiebreak rather than decoration.
            ranked.append(((rank - 0.5) / size, digest(f"{seed}:tie:{name}"), repo, name))
    ranked.sort()
    return [name for _, _, _, name in ranked]


def repo_counts(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        counts[task_repo(name)] = counts.get(task_repo(name), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def subset_pin(names: list[str]) -> str:
    """A digest of the chosen items in order, so the protocol pins the sample."""
    return "sha256:" + hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def build(names: list[str], size: int, seed: int, registry_digest: str) -> dict[str, Any]:
    if size < 1 or size > len(names):
        raise SubsetError(f"size must be between 1 and {len(names)}; got {size}")
    ordered = nested_order(names, seed)
    chosen = ordered[:size]
    return {
        "dataset": DATASET,
        "version": VERSION,
        "seed": seed,
        "population": len(names),
        "size": size,
        "registry_digest": registry_digest,
        # Pins the whole ordering as well as the cut, so a later prefix of a
        # different length is still verifiably the same sample.
        "order_pin": subset_pin(ordered),
        "subset_pin": subset_pin(chosen),
        "repo_counts": repo_counts(chosen),
        "task_names": chosen,
    }


def command_select(args: argparse.Namespace) -> int:
    names, registry_digest = load_registry(args.registry, DATASET, VERSION)
    subset = build(names, args.size, args.seed, registry_digest)
    text = json.dumps(subset, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {subset['size']} of {subset['population']} tasks to {args.out}")
        print(f"subset_pin {subset['subset_pin']}")
    else:
        sys.stdout.write(text)
    return 0


def command_verify(args: argparse.Namespace) -> int:
    stored = json.loads(args.subset.read_text(encoding="utf-8"))
    names, registry_digest = load_registry(args.registry, DATASET, VERSION)
    if registry_digest != stored.get("registry_digest"):
        raise SubsetError(
            "registry has changed since the subset was drawn: "
            f"{stored.get('registry_digest')} then, {registry_digest} now"
        )
    rebuilt = build(names, int(stored["size"]), int(stored["seed"]), registry_digest)
    for field in ("order_pin", "subset_pin", "task_names"):
        if rebuilt[field] != stored.get(field):
            raise SubsetError(f"{field} does not reproduce from the recorded seed")
    print(f"verified {rebuilt['size']} tasks, {rebuilt['subset_pin']}")
    return 0


def command_plan(args: argparse.Namespace) -> int:
    """Show the repo allocation at several sizes, to choose one before drawing."""
    names, _ = load_registry(args.registry, DATASET, VERSION)
    ordered = nested_order(names, args.seed)
    sizes = args.sizes or [120, 180, 240, 300, 360]
    population = repo_counts(names)
    repos = sorted(population, key=lambda repo: -population[repo])
    header = f"{'repo':32s} {'full':>6s}" + "".join(f"{size:>7d}" for size in sizes)
    print(header)
    for repo in repos:
        row = f"{repo:32s} {population[repo]:6d}"
        for size in sizes:
            row += f"{repo_counts(ordered[:size]).get(repo, 0):7d}"
        print(row)
    print(f"{'total':32s} {len(names):6d}" + "".join(f"{size:7d}" for size in sizes))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    for name in ("select", "verify", "plan"):
        command = sub.add_parser(name)
        command.add_argument(
            "--registry",
            type=Path,
            required=True,
            help="Harbor registry.json holding the pinned task list",
        )
        if name != "verify":
            command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        if name == "select":
            command.add_argument("--size", type=int, required=True)
            command.add_argument("--out", type=Path)
        if name == "verify":
            command.add_argument("--subset", type=Path, required=True)
        if name == "plan":
            command.add_argument("--sizes", type=int, nargs="+")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "select":
        return command_select(args)
    if args.action == "verify":
        return command_verify(args)
    return command_plan(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SubsetError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
