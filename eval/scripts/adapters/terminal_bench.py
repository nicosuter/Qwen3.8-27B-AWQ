#!/usr/bin/env python3
"""Terminal-Bench adapter driving Harbor's Hermes agent, for run_eval_protocol.py.

Unlike the other adapters this one does not talk to the server itself. Harbor
owns the agent loop, the task containers and the verifier; this builds a
JobConfig, runs it, and translates Harbor's `result.json` into the paired result
schema. `score` is the verifier reward, as EVAL.md requires for agent tasks. The
parts Harbor cares about live in `_harbor.py`, shared with SWE-bench Pro.

Two consequences of Harbor owning execution are worth stating rather than
hiding. Ordering is approximate: Harbor schedules its own trials, so this
adapter guarantees the task *set* and reconciles results back to ids, but not
the exact sequence the frozen order lists. And the environment needs a container
runtime for the task pack, which is why `--environment` exists: Harbor supports
`singularity` as well as `docker`, so this can run on an Apptainer-only cluster
if the pinned task pack ships singularity-compose files.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from _common import (
        AdapterError,
        check_action,
        env_path,
        env_str,
        load_pins,
        module_pin,
        read_jsonl,
        require_pin,
        write_json,
        write_jsonl,
    )
    from _harbor import (
        build_job_config,
        parse_dataset_pin,
        read_task_instructions,
        run_harbor,
        translate,
    )
except ModuleNotFoundError:  # loading by file spec puts the repo root on sys.path
    from eval.scripts.adapters._common import (  # type: ignore[no-redef]
        AdapterError,
        check_action,
        env_path,
        env_str,
        load_pins,
        module_pin,
        read_jsonl,
        require_pin,
        write_json,
        write_jsonl,
    )
    from eval.scripts.adapters._harbor import (  # type: ignore[no-redef]
        build_job_config,
        parse_dataset_pin,
        read_task_instructions,
        run_harbor,
        translate,
    )


SUITE = "terminal_bench_2_1"
PILOT_SUITE = "terminal_bench_2_1_pilot"
HARNESS_ID = "harbor-hermes-v1"
VERIFIER_ID = "harbor-task-verifier-v1"
DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"
DEFAULT_AGENT = "hermes"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    for name in ("prepare", "prepare-pilot"):
        prepare = sub.add_parser(name, help="materialize task instructions")
        prepare.add_argument("--dataset", default=DEFAULT_DATASET)
        prepare.add_argument("--agent", default=DEFAULT_AGENT)
        prepare.add_argument("--items", type=int, help="pilot size; 30 per EVAL.md")
        prepare.add_argument(
            "--tasks-dir",
            type=Path,
            help="directory of downloaded Harbor tasks; defaults to $TB_TASKS_DIR",
        )

    run = sub.add_parser("run", help="run Harbor and translate its results")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--n-attempts", type=int, default=1)
    run.add_argument("--environment", default="docker",
                     help="Harbor environment type, e.g. docker or singularity")
    run.add_argument("--harbor", default="harbor", help="path to the harbor CLI")
    run.add_argument("--timeout-multiplier", type=float, default=1.0)
    run.add_argument("--job-timeout", type=float, default=86400.0)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--dataset-version", default="REPLACE_WITH_DATASET_VERSION")
    pin.add_argument("--harbor-version", default="REPLACE_WITH_HARBOR_VERSION")
    pin.add_argument("--task-checksums", default="REPLACE_WITH_TASK_CHECKSUM_SET")
    return parser.parse_args(argv)


def self_pin() -> str:
    root = Path(__file__).resolve().parent
    return module_pin([Path(__file__), root / "_common.py", root / "_harbor.py"])


def suite_name() -> str:
    """The runner drives the pilot through the same adapter under its own label."""
    declared = os.environ.get("EVAL_SUITE", SUITE)
    if declared not in (SUITE, PILOT_SUITE):
        raise AdapterError(f"EVAL_SUITE is {declared!r}; expected {SUITE} or {PILOT_SUITE}")
    return declared


def validate_pins(pins: dict[str, str]) -> None:
    for field in ("dataset", "harness", "verifier", "adapter"):
        value = pins.get(field, "")
        if not value or "REPLACE_" in value or "PINNED_" in value:
            raise AdapterError(f"pins.{field} is missing or still a placeholder")
    _, _, subset = parse_dataset_pin(pins["dataset"])
    if subset is not None:
        # Terminal-Bench runs whole; a subset pin here would be recorded and then
        # quietly ignored.
        raise AdapterError(f"{SUITE} runs the full pack; drop the +subset pin")
    require_pin(pins, "adapter", self_pin())


def key_path(run_dir: Path, suite: str) -> Path:
    return run_dir / "materialized" / f"{suite}.key.json"


def materialize(
    instructions: dict[str, str], suite: str, limit: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = sorted(instructions)
    if limit is not None:
        if len(names) < limit:
            raise AdapterError(f"pack has {len(names)} tasks, fewer than the {limit} requested")
        names = names[:limit]
    prompts, key = [], {}
    for name in names:
        row = {"id": name, "suite": suite, "text": instructions[name], "category": "terminal"}
        if suite == PILOT_SUITE:
            # The runner requires category and difficulty on pilot rows.
            row["difficulty"] = "unknown"
        prompts.append(row)
        key[name] = {"task_name": name}
    return prompts, key


def command_prepare(args: argparse.Namespace, pilot: bool) -> int:
    check_action("prepare-pilot" if pilot else "prepare", suite_name())
    validate_pins(load_pins())
    suite = suite_name()
    run_dir = env_path("EVAL_RUN_DIR")
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    tasks_dir = args.tasks_dir or Path(os.environ.get("TB_TASKS_DIR", ""))
    if not str(tasks_dir):
        raise AdapterError("--tasks-dir or $TB_TASKS_DIR is required")

    instructions = read_task_instructions(Path(tasks_dir))
    limit = args.items if pilot else None
    prompts, key = materialize(instructions, suite, limit)
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir, suite),
        {
            "suite": suite,
            "dataset": args.dataset,
            "agent": args.agent,
            "tasks_dir": str(tasks_dir),
            "verifier": VERIFIER_ID,
            "adapter": self_pin(),
            "items": key,
        },
    )
    print(f"materialized {len(prompts)} {suite} tasks to {prompts_path}", flush=True)
    return 0


def command_run(args: argparse.Namespace) -> int:
    check_action("run", suite_name())
    pins = load_pins()
    validate_pins(pins)
    suite = suite_name()
    run_dir = env_path("EVAL_RUN_DIR")
    stored = json.loads(key_path(run_dir, suite).read_text(encoding="utf-8"))
    key = stored["items"]
    order = json.loads(env_path("EVAL_TASK_ORDER_JSON").read_text(encoding="utf-8"))
    prompts = {str(row["id"]) for row in read_jsonl(env_path("EVAL_PROMPTS_JSONL"))}
    missing = [item for item in order if item not in prompts or item not in key]
    if missing:
        raise AdapterError(f"task order references unmaterialized tasks: {missing[:10]}")

    replicate = int(env_str("EVAL_REPLICATE"))
    seed = int(env_str("EVAL_SEED"))
    variant = env_str("EVAL_VARIANT")
    model = env_str("EVAL_SERVED_MODEL")
    base_url = env_str("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    results_path = env_path("EVAL_RESULTS_JSONL")

    dataset, version, _ = parse_dataset_pin(pins["dataset"])
    job_name = f"{suite}-{variant}-r{replicate}"
    jobs_dir = run_dir / "harbor"
    config = build_job_config(
        job_name=job_name, jobs_dir=jobs_dir, dataset=dataset, version=version,
        agent=stored.get("agent", DEFAULT_AGENT), model=model, task_names=list(order),
        args=args, base_url=base_url, api_key=api_key,
    )

    started = time.monotonic()
    exit_code, result = run_harbor(
        harbor=args.harbor,
        config=config,
        config_path=jobs_dir / f"{job_name}-config.json",
        job_timeout=args.job_timeout,
    )
    rows = translate(result, key, suite, replicate, dataset_pin=pins["dataset"])
    write_jsonl(results_path, rows)

    write_json(
        run_dir / "metadata" / f"{suite}-{variant}-r{replicate}.json",
        {
            "suite": suite,
            "variant": variant,
            "replicate": replicate,
            "seed": seed,
            "served_model": model,
            "items": len(rows),
            "dataset": pins["dataset"],
            "agent": stored.get("agent", DEFAULT_AGENT),
            "environment": args.environment,
            "n_attempts": args.n_attempts,
            "concurrency": args.concurrency,
            "harbor_exit_code": exit_code,
            "harbor_result": str(jobs_dir / job_name / "result.json"),
            # Harbor schedules its own trials, so the frozen order fixes the task
            # set rather than the sequence. Recorded so the report cannot imply
            # otherwise.
            "task_order_enforced": "set-only",
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "mean_reward": round(sum(row["score"] for row in rows) / len(rows), 6),
        },
    )
    print(f"scored {len(rows)} {suite} tasks to {results_path}", flush=True)
    return 0


def command_pin(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "dataset": f"{DEFAULT_DATASET}@{args.dataset_version}",
                "harness": f"harbor@{args.harbor_version}",
                "verifier": args.task_checksums,
                "adapter": self_pin(),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "pin":
        return command_pin(args)
    if args.action in ("prepare", "prepare-pilot"):
        return command_prepare(args, pilot=args.action == "prepare-pilot")
    return command_run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
