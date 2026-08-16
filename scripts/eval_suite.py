#!/usr/bin/env python3
"""The one place that says which suites this protocol measures.

That set used to live in four: a Python constant in the runner, a suite list in
whichever `paired-N.json` a job happened to point at, the `PAIRED_SUITES` string
in a submitted job's environment, and the rows that actually reached the
comparator. Nothing held them equal, and they drifted the way four copies of one
fact always drift. RULER ended up required by the runner and absent from the
config that ran everything else, so the campaign could not have been launched as
specified, and nothing would have said so until the night it failed.

So the set is a file, and the file is versioned. `eval/eval-suite-v1.json` is the
pre-registration artifact: changing what gets measured means writing v2, not
editing a list and hoping the other three copies follow. Everything else derives
from it, which is why this module has no constants of its own.

Batches stay a separate idea on purpose. Which suites a given job runs is a
scheduling decision, and `select` exists to express that; which suites the
protocol *measures* is a pre-registration decision and belongs here. Conflating
them is how a job that scored five of seven suites could still report a macro.
"""

import json
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VERSION = "v1"


class EvalSuiteError(RuntimeError):
    pass


def path_for(version: str, root: Path | None = None) -> Path:
    if not version.startswith("v") or not version[1:].isdigit():
        raise EvalSuiteError(f"eval suite version must look like v1; got {version!r}")
    return (root or PROJECT_DIR) / "eval" / f"eval-suite-{version}.json"


def load(version: str = DEFAULT_VERSION, root: Path | None = None) -> dict[str, Any]:
    """Read a versioned suite definition, refusing anything malformed."""
    path = path_for(version, root)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EvalSuiteError(f"no eval suite definition at {path}") from error
    except json.JSONDecodeError as error:
        raise EvalSuiteError(f"{path}: not valid JSON: {error}") from error

    declared = document.get("eval_suite")
    if declared != version:
        # A file that disagrees with its own name is exactly the drift this
        # module exists to stop, so it is refused rather than guessed at.
        raise EvalSuiteError(
            f"{path} declares eval_suite {declared!r} but is named {version!r}"
        )
    suites = document.get("suites")
    if not isinstance(suites, list) or not suites:
        raise EvalSuiteError(f"{path} defines no suites")
    names = [suite.get("name") for suite in suites]
    if any(not isinstance(name, str) or not name for name in names):
        raise EvalSuiteError(f"{path} has a suite with no name")
    if len(set(names)) != len(names):
        raise EvalSuiteError(f"{path} names a suite twice")
    for suite in suites:
        replicates = suite.get("replicates")
        if not isinstance(replicates, int) or replicates < 1:
            raise EvalSuiteError(
                f"{path}: {suite['name']} needs a replicate count of at least 1"
            )
    return document


def names(version: str = DEFAULT_VERSION, root: Path | None = None) -> set[str]:
    return {suite["name"] for suite in load(version, root)["suites"]}


def replicates(version: str = DEFAULT_VERSION, root: Path | None = None) -> dict[str, int]:
    return {suite["name"]: suite["replicates"] for suite in load(version, root)["suites"]}


def select(
    requested: list[str], version: str = DEFAULT_VERSION, root: Path | None = None
) -> list[str]:
    """Resolve a batch to suite names, in the definition's own order.

    A batch is a subset of the protocol, never an extension of it. Naming a
    suite the definition does not contain is refused: that is how `aa_lcr` and
    `aa_omniscience` came to be scored into a macro they were never part of.
    """
    defined = [suite["name"] for suite in load(version, root)["suites"]]
    unknown = sorted(set(requested) - set(defined))
    if unknown:
        raise EvalSuiteError(
            f"eval suite {version} does not define {unknown}; "
            f"it defines {defined}. A batch selects from the protocol, it does "
            "not add to it."
        )
    return [name for name in defined if name in set(requested)]


def missing(present: set[str], version: str = DEFAULT_VERSION,
            root: Path | None = None) -> list[str]:
    """Suites the protocol requires that a set of results does not carry."""
    return sorted(names(version, root) - present)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--select", nargs="+", help="resolve a batch and print it")
    args = parser.parse_args(argv)

    if args.select:
        print(" ".join(select(args.select, args.version)))
        return 0
    document = load(args.version)
    print(f"eval suite {args.version}: {len(document['suites'])} suites")
    for suite in document["suites"]:
        print(f"  {suite['name']:24s} x{suite['replicates']}")
    parked = document.get("rationale", {}).get("parked", {})
    if parked:
        print("parked:")
        for name, reason in parked.items():
            summary = reason if len(reason) <= 88 else reason[:85].rsplit(" ", 1)[0] + "..."
            print(f"  {name:24s} {summary}")
    return 0


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except EvalSuiteError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
