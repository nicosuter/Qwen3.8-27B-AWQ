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

# The suite definition is read from shell, by whatever `python3` resolves to.
# On the clusters that is 3.9, where `Path | None` in a signature raises at
# import time. Deferring annotations costs nothing and keeps this module
# loadable by any interpreter that can parse the file.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VERSION = "v2"


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


def batches(version: str = DEFAULT_VERSION, root: Path | None = None) -> list[dict[str, Any]]:
    """Read the batch plan and hold it to the suite definition.

    Two invariants, and both have already been violated in this repository once.
    A batch may not name a suite the protocol does not contain, which is how
    `aa_lcr` reached a macro. And the scoring batches must partition the suite
    set exactly: a plan that leaves a suite out means running every batch still
    does not produce the protocol's macro, and a plan that scores one twice means
    the second run silently overwrites the first.
    """
    path = (root or PROJECT_DIR) / "eval" / "batches.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EvalSuiteError(f"no batch plan at {path}") from error
    except json.JSONDecodeError as error:
        raise EvalSuiteError(f"{path}: not valid JSON: {error}") from error
    if document.get("eval_suite") != version:
        raise EvalSuiteError(
            f"{path} plans eval suite {document.get('eval_suite')!r}, not {version!r}"
        )

    plan = document.get("batches")
    if not isinstance(plan, list) or not plan:
        raise EvalSuiteError(f"{path} defines no batches")
    defined = names(version, root)
    seen: list[str] = []
    for batch in plan:
        name = batch.get("name")
        if not name:
            raise EvalSuiteError(f"{path} has a batch with no name")
        select(batch.get("suites", []), version, root)   # refuses unknown suites
        if batch.get("scoring"):
            seen.extend(batch["suites"])
    duplicated = sorted({s for s in seen if seen.count(s) > 1})
    if duplicated:
        raise EvalSuiteError(f"{path}: scoring batches overlap on {duplicated}")
    uncovered = sorted(defined - set(seen))
    if uncovered:
        raise EvalSuiteError(
            f"{path}: no scoring batch covers {uncovered}, so running every batch "
            "would still not produce the protocol's macro"
        )
    return plan


def batch(name: str, version: str = DEFAULT_VERSION,
          root: Path | None = None) -> dict[str, Any]:
    plan = batches(version, root)
    for entry in plan:
        if entry["name"] == name:
            return entry
    raise EvalSuiteError(
        f"no batch named {name!r}; the plan has {[e['name'] for e in plan]}"
    )


def missing(present: set[str], version: str = DEFAULT_VERSION,
            root: Path | None = None) -> list[str]:
    """Suites the protocol requires that a set of results does not carry."""
    return sorted(names(version, root) - present)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--select", nargs="+", help="resolve a suite list and print it")
    parser.add_argument("--batch", help="print the suites in a named batch")
    parser.add_argument(
        "--batch-field",
        nargs=2,
        metavar=("NAME", "FIELD"),
        help="print one field of a batch, so a shell can act on it",
    )
    parser.add_argument("--batches", action="store_true", help="list the batch plan")
    parser.add_argument(
        "--names", action="store_true",
        help="print the suite names, so a shell can filter what the protocol defines",
    )
    args = parser.parse_args(argv)

    if args.select:
        print(" ".join(select(args.select, args.version)))
        return 0
    if args.names:
        print(" ".join(sorted(names(args.version))))
        return 0
    if args.batch_field:
        name, field = args.batch_field
        value = batch(name, args.version).get(field, "")
        print(" ".join(map(str, value)) if isinstance(value, list) else value)
        return 0
    if args.batch:
        print(" ".join(batch(args.batch, args.version)["suites"]))
        return 0
    if args.batches:
        for entry in batches(args.version):
            kind = "scoring" if entry.get("scoring") else entry.get("phase", "setup")
            print(f"{entry['name']:16s} {kind:8s} {' '.join(entry['suites'])}")
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
