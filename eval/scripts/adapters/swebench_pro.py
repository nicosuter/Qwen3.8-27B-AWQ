#!/usr/bin/env python3
"""SWE-bench Pro adapter, running a pre-registered subset through Harbor.

The medium-horizon agentic suite. Harbor owns the loop, the containers and the
verifier exactly as it does for Terminal-Bench, so the driving lives in
`_harbor.py` and this file supplies the three things that differ: which tasks
run, how the sample is pinned, and what category a row reports.

All 731 instances cost roughly 34 H200-hours paired, so we run a sample drawn by
`eval/scripts/swebenchpro_subset.py` before any score exists. The sample is not a
loose convention here. Its digest is carried *inside* the dataset pin, as
`swebenchpro@1.0+subset:sha256:...`, because the pin is what the comparator
checks across the two arms. Anywhere else, a baseline and a candidate could run
different samples of the same dataset and still be reported as paired.

Rows report their repository as the category. SWE-bench Pro's repositories are
also its language split, so a per-repo breakdown is what tells you whether a
regression is real or lives entirely in, say, the Go half.
"""

import argparse
import hashlib
import json
import os
import re
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


SUITE = "swebench_pro_1_0"
HARNESS_ID = "harbor-hermes-v1"
VERIFIER_ID = "harbor-task-verifier-v1"
DEFAULT_DATASET = "swebenchpro"
DEFAULT_VERSION = "1.0"
# The same agent Terminal-Bench uses. Two agentic suites scored under one harness
# are comparable to each other; two agents would confound the suites.
DEFAULT_AGENT = "hermes"
DEFAULT_SUBSET = "eval/swebenchpro-subset-300.json"
# Kept in step with eval/scripts/swebenchpro_subset.py by a test, because a category
# that silently stopped parsing would look like a repository with no tasks.
TASK_RE = re.compile(r"^instance_(?P<repo>.+?)-[0-9a-f]{40}(?:$|-)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare", help="materialize the pinned subset")
    prepare.add_argument("--agent", default=DEFAULT_AGENT)
    prepare.add_argument(
        "--subset",
        type=Path,
        help=f"subset drawn by swebenchpro_subset.py; defaults to $SWEP_SUBSET or {DEFAULT_SUBSET}",
    )
    prepare.add_argument(
        "--tasks-dir",
        type=Path,
        help="directory of downloaded Harbor tasks; defaults to $SWEP_TASKS_DIR",
    )

    run = sub.add_parser("run", help="run Harbor and translate its results")
    run.add_argument("--concurrency", type=int, default=40)
    run.add_argument("--n-attempts", type=int, default=1)
    run.add_argument("--environment", default="docker",
                     help="Harbor environment type, e.g. docker or singularity")
    run.add_argument("--harbor", default="harbor", help="path to the harbor CLI")
    run.add_argument("--timeout-multiplier", type=float, default=1.0)
    run.add_argument("--job-timeout", type=float, default=86400.0)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--subset", type=Path, help="subset whose digest joins the dataset pin")
    pin.add_argument("--harbor-version", default="REPLACE_WITH_HARBOR_VERSION")
    pin.add_argument("--task-checksums", default="REPLACE_WITH_TASK_CHECKSUM_SET")
    return parser.parse_args(argv)


def self_pin() -> str:
    root = Path(__file__).resolve().parent
    return module_pin([Path(__file__), root / "_common.py", root / "_harbor.py"])


def task_repo(name: str) -> str:
    match = TASK_RE.match(name)
    if not match:
        raise AdapterError(f"cannot read a repo out of task name {name!r}")
    return match.group("repo")


def subset_path(given: Path | None) -> Path:
    return Path(given or os.environ.get("SWEP_SUBSET", "") or DEFAULT_SUBSET)


def subset_pin(names: list[str]) -> str:
    """The digest eval/scripts/swebenchpro_subset.py writes, recomputed from names."""
    return "sha256:" + hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def load_subset(path: Path) -> dict[str, Any]:
    """Read a drawn subset, refusing anything that is not one."""
    try:
        subset = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AdapterError(
            f"{path} does not exist. Draw one first: python3 "
            "eval/scripts/swebenchpro_subset.py select --registry <registry.json> "
            f"--size 300 --out {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise AdapterError(f"{path}: not valid JSON: {error}") from error

    if subset.get("dataset") != DEFAULT_DATASET:
        raise AdapterError(f"{path} is a subset of {subset.get('dataset')!r}, not {DEFAULT_DATASET}")
    names = subset.get("task_names")
    if not isinstance(names, list) or not names:
        raise AdapterError(f"{path} has no task_names")
    if len(set(names)) != len(names):
        raise AdapterError(f"{path} lists a task twice")
    declared = subset.get("subset_pin")
    if not isinstance(declared, str):
        raise AdapterError(f"{path} has no subset_pin")
    # Recompute rather than trust. A declared pin only checks that the protocol
    # and the file agree with each other, so editing task_names and leaving the
    # pin alone would change the sample while every check still passed, which is
    # the one thing this design exists to prevent.
    actual = subset_pin(names)
    if actual != declared:
        raise AdapterError(
            f"{path}: subset_pin does not match its own task_names "
            f"(declared {declared}, actual {actual})"
        )
    return subset


def validate_pins(pins: dict[str, str], expected_subset: str | None = None) -> None:
    for field in ("dataset", "harness", "verifier", "adapter"):
        value = pins.get(field, "")
        if not value or "REPLACE_" in value or "PINNED_" in value:
            raise AdapterError(f"pins.{field} is missing or still a placeholder")
    dataset, _, subset = parse_dataset_pin(pins["dataset"])
    if dataset != DEFAULT_DATASET:
        raise AdapterError(f"pins.dataset must name {DEFAULT_DATASET}; got {dataset!r}")
    if subset is None:
        raise AdapterError(
            "pins.dataset must carry the subset digest, as "
            f"{DEFAULT_DATASET}@{DEFAULT_VERSION}+subset:sha256:<digest>. Without it "
            "the two arms can run different samples and still compare."
        )
    if expected_subset is not None and subset != expected_subset:
        raise AdapterError(
            f"pins.dataset pins subset {subset}, but the materialized sample is "
            f"{expected_subset}"
        )
    require_pin(pins, "adapter", self_pin())


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def materialize(
    subset: dict[str, Any], instructions: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build prompts for exactly the drawn tasks, in the order they were drawn.

    A task pack missing part of the sample is fatal. Quietly running whichever
    tasks happened to be downloaded would be a different sample than the one
    pinned, which is the whole thing the pin exists to prevent.
    """
    names = [str(name) for name in subset["task_names"]]
    missing = [name for name in names if name not in instructions]
    if missing:
        raise AdapterError(
            f"the task pack is missing {len(missing)} of the {len(names)} drawn "
            f"tasks, for example {missing[:3]}"
        )

    prompts, key = [], {}
    for name in names:
        repo = task_repo(name)
        prompts.append({"id": name, "suite": SUITE, "text": instructions[name],
                        "category": repo})
        key[name] = {"task_name": name, "category": repo}
    return prompts, key


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    path = subset_path(args.subset)
    subset = load_subset(path)
    validate_pins(load_pins(), subset["subset_pin"])

    run_dir = env_path("EVAL_RUN_DIR")
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    tasks_dir = args.tasks_dir or Path(os.environ.get("SWEP_TASKS_DIR", ""))
    if not str(tasks_dir):
        raise AdapterError("--tasks-dir or $SWEP_TASKS_DIR is required")

    instructions = read_task_instructions(Path(tasks_dir), "SWEP_TASKS_DIR")
    prompts, key = materialize(subset, instructions)
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {
            "suite": SUITE,
            "dataset": DEFAULT_DATASET,
            "agent": args.agent,
            "tasks_dir": str(tasks_dir),
            "subset": str(path),
            "subset_pin": subset["subset_pin"],
            "population": subset.get("population"),
            "verifier": VERIFIER_ID,
            "adapter": self_pin(),
            "items": key,
        },
    )
    print(f"materialized {len(prompts)} {SUITE} tasks to {prompts_path}", flush=True)
    return 0


def command_run(args: argparse.Namespace) -> int:
    check_action("run", SUITE)
    run_dir = env_path("EVAL_RUN_DIR")
    stored = json.loads(key_path(run_dir).read_text(encoding="utf-8"))
    key = stored["items"]
    pins = load_pins()
    validate_pins(pins, stored.get("subset_pin"))

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

    if args.n_attempts != 1:
        # EVAL.md reports this suite as a resolve rate, and translate() takes the
        # best of a task's attempts. More than one would publish pass@k under a
        # pass@1 heading.
        raise AdapterError(
            f"--n-attempts must be 1 for {SUITE}; got {args.n_attempts}"
        )

    # Harbor resumes a job only when the config it is handed is identical, and
    # n_concurrent_trials lives in that config. The lane splitter scales
    # concurrency by how many lanes are still pending, so a lane resumed in a
    # later job would arrive with a different number and be refused after the
    # trials were already paid for. The first run's value is therefore the one
    # that counts.
    concurrency = stored.get("concurrency")
    if concurrency is None:
        concurrency = args.concurrency
        stored["concurrency"] = concurrency
        write_json(key_path(run_dir), stored)
    elif concurrency != args.concurrency:
        print(
            f"note: reusing the first run's --concurrency {concurrency} rather "
            f"than {args.concurrency}, so harbor can resume this job",
            flush=True,
        )
    args.concurrency = concurrency

    dataset, version, _ = parse_dataset_pin(pins["dataset"])
    job_name = f"{SUITE}-{variant}-r{replicate}"
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
    rows = translate(result, key, SUITE, replicate, dataset_pin=pins["dataset"])
    write_jsonl(results_path, rows)

    by_repo: dict[str, list[float]] = {}
    for row in rows:
        by_repo.setdefault(row["category"], []).append(row["score"])

    write_json(
        run_dir / "metadata" / f"{SUITE}-{variant}-r{replicate}.json",
        {
            "suite": SUITE,
            "variant": variant,
            "replicate": replicate,
            "seed": seed,
            "served_model": model,
            "items": len(rows),
            "dataset": pins["dataset"],
            "subset": stored.get("subset"),
            "subset_pin": stored.get("subset_pin"),
            "population": stored.get("population"),
            "agent": stored.get("agent", DEFAULT_AGENT),
            "environment": args.environment,
            "n_attempts": args.n_attempts,
            "concurrency": concurrency,
            "harbor_exit_code": exit_code,
            "harbor_result": str(jobs_dir / job_name / "result.json"),
            # As for Terminal-Bench: Harbor schedules its own trials, so the
            # frozen order fixes the task set and not the sequence.
            "task_order_enforced": "set-only",
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "mean_reward": round(sum(row["score"] for row in rows) / len(rows), 6),
            "mean_reward_by_repo": {
                repo: round(sum(scores) / len(scores), 6)
                for repo, scores in sorted(by_repo.items())
            },
        },
    )
    print(f"scored {len(rows)} {SUITE} tasks to {results_path}", flush=True)
    return 0


def command_pin(args: argparse.Namespace) -> int:
    subset = load_subset(subset_path(args.subset))
    print(
        json.dumps(
            {
                "dataset": (
                    f"{DEFAULT_DATASET}@{DEFAULT_VERSION}+subset:{subset['subset_pin']}"
                ),
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
    if args.action == "prepare":
        return command_prepare(args)
    return command_run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
