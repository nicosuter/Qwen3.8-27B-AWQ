#!/usr/bin/env python3
"""Drive one benchmark adapter against an already-served endpoint.

`run_eval_protocol.py` validates all seven suites before it will do anything, so
it cannot execute while most adapters are still missing. This driver runs a
single suite through the same environment contract, reusing the runner's own
prompt and result validators, so a dry run exercises the real contract rather
than an approximation of it.

It deliberately does not start servers, alternate checkpoints, audit overlap, or
produce a release decision. Results from `--limit` runs are diagnostics only.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_eval_protocol as protocol  # noqa: E402


REQUIRED_CONFIG_KEYS = ("order_seed", "served_model_name", "generation", "suites")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "run", "all"), default="all")
    parser.add_argument("--variant", default="candidate")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-url", default="")
    parser.add_argument(
        "--limit",
        type=int,
        help="score only the first N items of the frozen order; dry runs only",
    )
    return parser.parse_args(argv)


def load_config(path: Path, suite: str) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise protocol.ProtocolError(f"{path}: missing {missing}")
    names = [entry["name"] for entry in config["suites"]]
    if suite not in names:
        raise protocol.ProtocolError(f"{path}: no suite named {suite!r}; has {names}")
    entry = next(item for item in config["suites"] if item["name"] == suite)
    for field in ("pins", "prepare", "run"):
        if field not in entry:
            raise protocol.ProtocolError(f"{path}: suite {suite} has no {field}")
    # Same fail-closed rule as the full runner: a placeholder is never a pin.
    for value in list(entry["pins"].values()) + list(entry["prepare"]) + list(entry["run"]):
        if "REPLACE_" in str(value) or "PINNED_" in str(value):
            raise protocol.ProtocolError(f"{path}: suite {suite} still contains a placeholder")
    return config


def order_path(run_dir: Path, suite: str) -> Path:
    return run_dir / "orders" / f"{suite}.json"


def do_prepare(config: dict[str, Any], suite: str, run_dir: Path) -> list[str]:
    entry = next(item for item in config["suites"] if item["name"] == suite)
    env = protocol.adapter_environment(config, run_dir, suite)
    env["EVAL_ACTION"] = "prepare"
    protocol.run_logged(
        entry["prepare"],
        env=env,
        log_path=run_dir / "logs" / f"prepare-{suite}.log",
        dry_run=False,
    )
    ids = protocol.validate_prompts(Path(env["EVAL_PROMPTS_JSONL"]), suite)
    rng = random.Random(config["order_seed"])
    rng.shuffle(ids)
    protocol.write_json(order_path(run_dir, suite), ids)
    print(f"prepared {suite}: {len(ids)} items", flush=True)
    return ids


def do_run(config: dict[str, Any], suite: str, run_dir: Path, args: argparse.Namespace) -> Path:
    if not args.base_url:
        raise protocol.ProtocolError("--base-url is required to score a suite")
    entry = next(item for item in config["suites"] if item["name"] == suite)
    order = json.loads(order_path(run_dir, suite).read_text(encoding="utf-8"))
    if args.limit:
        order = order[: args.limit]
        # The adapter consumes whatever order file it is given; point it at the
        # truncated one so the run stays inside the frozen sequence.
        truncated = run_dir / "orders" / f"{suite}-limit{args.limit}.json"
        protocol.write_json(truncated, order)
        print(f"limiting {suite} to {len(order)} of the frozen order", flush=True)

    seed = args.seed if args.seed is not None else config["order_seed"]
    output = run_dir / "raw" / args.variant / f"{suite}-r{args.replicate}.jsonl"
    env = protocol.adapter_environment(config, run_dir, suite)
    env.update(
        {
            "EVAL_ACTION": "run",
            "EVAL_VARIANT": args.variant,
            "EVAL_REPLICATE": str(args.replicate),
            "EVAL_SEED": str(seed),
            "EVAL_RESULTS_JSONL": str(output),
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_BASE_URL": args.base_url,
        }
    )
    if args.limit:
        env["EVAL_TASK_ORDER_JSON"] = str(run_dir / "orders" / f"{suite}-limit{args.limit}.json")
    protocol.run_logged(
        entry["run"],
        env=env,
        log_path=run_dir / "logs" / f"run-{args.variant}-r{args.replicate}-{suite}.log",
        dry_run=False,
    )
    protocol.validate_results(output, suite, args.replicate, set(order))
    return output


def summarize(results: Path) -> None:
    rows = protocol.read_jsonl(results)
    scores = [row["score"] for row in rows]
    print(f"\n{results}")
    print(f"  items {len(rows)}  mean score {sum(scores)/len(scores):.4f}")
    groups: dict[str, list[float]] = {}
    for row in rows:
        label = row.get("length") or row.get("category") or "all"
        groups.setdefault(str(label), []).append(row["score"])
    if len(groups) > 1:
        for label, values in sorted(groups.items()):
            print(f"    {label:>24s}  n={len(values):<4d} mean {sum(values)/len(values):.4f}")
    for field in protocol.RESULT_BOOL_FIELDS:
        flagged = sum(1 for row in rows if row.get(field))
        if flagged:
            print(f"    {field}: {flagged}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.suite)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("prepare", "all"):
        do_prepare(config, args.suite, args.run_dir)
    if args.phase in ("run", "all"):
        summarize(do_run(config, args.suite, args.run_dir, args))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except protocol.ProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
