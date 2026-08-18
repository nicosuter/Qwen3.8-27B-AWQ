#!/usr/bin/env python3
"""Bring every config's adapter pin back in line with the adapter it names.

The pin is a hash of the adapter together with `_common.py`, so a change to the
shared module restales every pin in every config at once. That is the pin doing
its job -- it exists to stop an adapter changing under a result set -- but the
correction is mechanical and was previously done by hand, which is how it came
to be discovered twice on a cluster after a queue wait.

    eval/scripts/repin_adapters.py --check eval/*.json    # what is stale
    eval/scripts/repin_adapters.py eval/*.json            # fix it

Only `pins.adapter` is touched. Everything else in a pins object is a decision
someone made, not something derivable from the tree.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Adapters import _common either as a bare sibling or as
# eval.scripts.adapters._common, depending on how they were launched. Importing
# one from here has to satisfy whichever branch it takes.
for _path in (PROJECT_ROOT, PROJECT_ROOT / "eval" / "scripts" / "adapters"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def adapter_path(suite: dict[str, Any], root: Path) -> Path | None:
    for part in suite.get("run") or []:
        if str(part).endswith(".py"):
            return root / str(part)
    return None


def current_pin(path: Path) -> str:
    """Import the adapter and ask it, rather than reimplementing the hash."""
    spec = importlib.util.spec_from_file_location(f"pin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.self_pin())


def repin(
    configs: list[Path], *, root: Path = PROJECT_ROOT, check: bool = False
) -> list[tuple[Path, str, str]]:
    """Return the (config, suite, pin) rows that were stale."""
    stale: list[tuple[Path, str, str]] = []
    for config in configs:
        raw = json.loads(Path(config).read_text(encoding="utf-8"))
        touched = False
        for suite in raw.get("suites") or []:
            # Not every file with a "suites" key defines how to run them:
            # batches.json lists names, saying only which ones a job covers.
            if not isinstance(suite, dict):
                continue
            path = adapter_path(suite, root)
            if path is None or not path.is_file():
                continue
            pin = current_pin(path)
            if suite.get("pins", {}).get("adapter") == pin:
                continue
            stale.append((Path(config), str(suite.get("name")), pin))
            if not check:
                suite.setdefault("pins", {})["adapter"] = pin
                touched = True
        if touched:
            # Two-space indent and a trailing newline, matching the configs as
            # they are checked in, so a repin is not also a whitespace diff.
            Path(config).write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return stale


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale pins and exit non-zero without writing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stale = repin(args.configs, check=args.check)
    for config, suite, pin in stale:
        verb = "stale" if args.check else "repinned"
        print(f"{verb}: {config.name} {suite} -> {pin}")
    if not stale:
        print("every adapter pin is current")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
