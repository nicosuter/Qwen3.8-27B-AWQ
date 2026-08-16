#!/usr/bin/env python3
"""Drive one benchmark adapter against an already-served endpoint.

`run_eval_protocol.py` validates every suite before it will do anything, so
it cannot execute while most adapters are still missing. This driver runs a
single suite through the same environment contract, reusing the runner's own
prompt and result validators, so a dry run exercises the real contract rather
than an approximation of it.

It deliberately does not start servers, alternate checkpoints, audit overlap, or
produce a release decision. Results from `--limit` runs are diagnostics only.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
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
        "--concurrency",
        type=int,
        help="override the suite's --concurrency outright",
    )
    parser.add_argument(
        "--concurrency-scale",
        type=float,
        help="scale the suite's configured --concurrency, for a smaller server than "
             "the config assumes. Preferred over --concurrency across suites, whose "
             "KV footprints differ by an order of magnitude at long context",
    )
    parser.add_argument(
        "--request-timeout-scale",
        type=float,
        help="scale the suite's --request-timeout for slower hardware. A cap the "
             "card cannot reach inside the timeout turns model behavior into a "
             "zero: five matharena items hit exactly 4200s on A100 and scored 0 "
             "with no output, where the same config is safe on H200",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="score only the first N items of the frozen order; dry runs only",
    )
    return parser.parse_args(argv)


def load_config(path: Path, suite: str) -> dict[str, Any]:
    # The interpreter is the one deployment-specific part of a suite command:
    # the venv sits outside the checkout so that per-commit checkouts can share
    # it. Configs name it ${EVAL_PYTHON}, which resolves here under the runner's
    # existing fail-closed rule -- an unset variable stops the run rather than
    # exec'ing a file literally named "${EVAL_PYTHON}".
    config = protocol.expand_environment(json.loads(path.read_text(encoding="utf-8")))
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


def replicate_seed(config: dict[str, Any], replicate: int, override: int | None) -> int:
    """The sampling seed for one replicate.

    Replicates exist to measure run-to-run spread, so they have to be
    independent draws. Reusing one seed across them would leave only batch
    composition to differ, making the replicates far more alike than genuine
    samples and pooled intervals correspondingly too tight. `run_eval_protocol`
    takes these from a `seeds` list; this driver honours that list when present
    and otherwise derives a distinct seed per replicate.
    """
    if override is not None:
        return override
    seeds = config.get("seeds")
    if seeds:
        if replicate >= len(seeds):
            raise protocol.ProtocolError(
                f"config lists {len(seeds)} seeds; replicate {replicate} has none"
            )
        return int(seeds[replicate])
    if replicate == 0:
        return int(config["order_seed"])
    digest = hashlib.sha256(f"{config['order_seed']}:{replicate}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def served_checkpoint() -> dict[str, Any] | None:
    """Which weights produced these results, as fingerprinted by the caller.

    Both variants are served under one model name on purpose, so the served name
    cannot answer this and the results would otherwise be anonymous.
    """
    raw = os.environ.get("EVAL_CHECKPOINT_JSON", "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "EVAL_CHECKPOINT_JSON was not valid JSON", "raw": raw[:200]}


def describe_hardware() -> dict[str, Any]:
    """What this run actually executed on.

    A varying factor that is not recorded cannot be accounted for afterwards,
    and hardware is the factor that decides whether a request timeout is
    generous or fatal. The same config that is safe on H200 scored five
    matharena items zero on A100 purely by running out of wall clock.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {"gpus": None, "gpu": None}
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"gpus": len(names), "gpu": names[0] if names else None}


def annotate_metadata(
    run_dir: Path, suite: str, variant: str, replicate: int, extra: dict[str, Any]
) -> None:
    """Fold run-level facts into the metadata the adapter already wrote.

    Done here rather than inside the adapters so that recording more about a run
    never changes an adapter's self-pin, which would invalidate every config.
    """
    path = run_dir / "metadata" / f"{suite}-{variant}-r{replicate}.json"
    if not path.is_file():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(extra)
    protocol.write_json(path, record)


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
    command = list(entry["run"])
    if args.concurrency_scale and not args.concurrency and "--concurrency" in command:
        configured = int(command[command.index("--concurrency") + 1])
        args.concurrency = max(1, int(configured * args.concurrency_scale))
    if args.concurrency:
        if "--concurrency" in command:
            command[command.index("--concurrency") + 1] = str(args.concurrency)
        else:
            command += ["--concurrency", str(args.concurrency)]
        print(f"concurrency overridden to {args.concurrency}", flush=True)
    if args.request_timeout_scale and "--request-timeout" in command:
        index = command.index("--request-timeout") + 1
        scaled = float(command[index]) * args.request_timeout_scale
        command[index] = str(scaled)
        print(f"request timeout scaled to {scaled:.0f}s", flush=True)
    order = json.loads(order_path(run_dir, suite).read_text(encoding="utf-8"))
    if args.limit:
        order = order[: args.limit]
        # The adapter consumes whatever order file it is given; point it at the
        # truncated one so the run stays inside the frozen sequence.
        truncated = run_dir / "orders" / f"{suite}-limit{args.limit}.json"
        protocol.write_json(truncated, order)
        print(f"limiting {suite} to {len(order)} of the frozen order", flush=True)

    seed = replicate_seed(config, args.replicate, args.seed)
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
        command,
        env=env,
        log_path=run_dir / "logs" / f"run-{args.variant}-r{args.replicate}-{suite}.log",
        dry_run=False,
    )
    protocol.validate_results(output, suite, args.replicate, set(order))
    rows = protocol.read_jsonl(output)
    annotate_metadata(
        run_dir,
        suite,
        args.variant,
        args.replicate,
        {
            "checkpoint": served_checkpoint(),
            "hardware": describe_hardware(),
            "request_timeout_scale": args.request_timeout_scale,
            "concurrency_scale": args.concurrency_scale,
            # Surfaced next to the hardware because together they say whether a
            # zero was the model's answer or the wall clock's.
            "timeouts": sum(1 for row in rows if row.get("timeout")),
            "context_failures": sum(1 for row in rows if row.get("context_failure")),
        },
    )
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
