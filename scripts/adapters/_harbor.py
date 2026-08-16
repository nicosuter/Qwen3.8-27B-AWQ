#!/usr/bin/env python3
"""Shared driver for the suites Harbor executes, rather than our server does.

Terminal-Bench and SWE-bench Pro are the same job from our side. Harbor owns the
agent loop, the task containers and the verifier; we choose the task set, hand it
an endpoint, and translate its `result.json` into paired rows. Only the task set,
the pin format and the per-item category differ, so those are what the two
adapters supply and everything else lives here.

Keeping it in one module is not tidiness. `translate` decides what counts as a
timeout, what an unscored trial means, and that a task Harbor never ran is fatal
rather than a zero. Two copies of those rules would eventually disagree, and the
disagreement would show up as a score difference between suites that has nothing
to do with the model.

Harbor resumes on its own: re-running into an existing job directory keeps every
trial that already has a result and re-plans the rest, and it refuses outright if
the config changed. So preemption costs the trials that were in flight, not the
run, and nothing here needs to reimplement that.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from _common import AdapterError, base_row, write_json
except ModuleNotFoundError:  # loading by file spec puts the repo root on sys.path
    from scripts.adapters._common import (  # type: ignore[no-redef]
        AdapterError,
        base_row,
        write_json,
    )

# `dataset@version`, optionally carrying the digest of a pre-registered subset.
# The subset belongs in the dataset pin and not beside it, because the pin is
# what the comparator checks across the two arms: anywhere else, two arms could
# run different samples of the same dataset and still look comparable.
DATASET_PIN_RE = re.compile(
    r"^(?P<dataset>[\w\-./]+)@(?P<version>[\w.\-]+)"
    r"(?:\+subset:(?P<subset>sha256:[0-9a-f]{64}))?$"
)


def parse_dataset_pin(value: str) -> tuple[str, str, str | None]:
    match = DATASET_PIN_RE.match(value or "")
    if not match:
        raise AdapterError(
            "dataset pin must be dataset@version, optionally followed by "
            f"+subset:sha256:<digest>; got {value!r}"
        )
    return match.group("dataset"), match.group("version"), match.group("subset")


def read_task_instructions(tasks_dir: Path, env_var: str = "TB_TASKS_DIR") -> dict[str, str]:
    """Harbor task packs carry one directory per task with an instruction file."""
    if not tasks_dir.is_dir():
        raise AdapterError(
            f"{tasks_dir} is not a directory. Download the pinned pack first, for "
            "example `harbor download dataset <name> --version <version>`, and "
            f"point --tasks-dir or ${env_var} at it."
        )
    instructions = {}
    for candidate in sorted(tasks_dir.iterdir()):
        if not candidate.is_dir():
            continue
        for filename in ("instruction.md", "task.md", "instruction.txt", "prompt.md"):
            path = candidate / filename
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    instructions[candidate.name] = text
                break
    if not instructions:
        raise AdapterError(
            f"{tasks_dir} has no task directories with an instruction file; "
            "the layout is not what this adapter expects"
        )
    return instructions


def build_job_config(
    *,
    job_name: str,
    jobs_dir: Path,
    dataset: str,
    version: str,
    agent: str,
    model: str,
    task_names: list[str],
    args: Any,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """A JobConfig dict matching harbor.models.job.config:JobConfig."""
    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": args.n_attempts,
        "n_concurrent_trials": args.concurrency,
        "timeout_multiplier": args.timeout_multiplier,
        "quiet": True,
        "environment": {"type": args.environment},
        "agents": [
            {
                "name": agent,
                "model_name": model,
                # The agent reads the endpoint from the environment; both models
                # are served under the same name so it cannot branch on it.
                "env": {"OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": api_key},
            }
        ],
        "datasets": [
            {"name": dataset, "version": version, "task_names": sorted(task_names)}
        ],
    }


def run_harbor(
    *, harbor: str, config: dict[str, Any], config_path: Path, job_timeout: float
) -> tuple[int, dict[str, Any]]:
    """Run one Harbor job and return its exit code with the parsed result.

    Harbor writes `result.json` whether or not every trial succeeded, so a
    non-zero exit is reported rather than raised: the per-task detail in the
    result says more about what happened than the exit code does.
    """
    write_json(config_path, config)
    command = [harbor, "run", "--config", str(config_path)]
    print("run: " + " ".join(command), flush=True)
    try:
        completed = subprocess.run(command, timeout=job_timeout, check=False)
    except FileNotFoundError as error:
        raise AdapterError(f"harbor CLI not found at {harbor!r}") from error
    except subprocess.TimeoutExpired as error:
        raise AdapterError(f"harbor exceeded --job-timeout {job_timeout}s") from error

    result_file = Path(config["jobs_dir"]) / config["job_name"] / "result.json"
    if not result_file.is_file():
        raise AdapterError(
            f"harbor exited {completed.returncode} without writing {result_file}"
        )
    return completed.returncode, json.loads(result_file.read_text(encoding="utf-8"))


def extract_reward(trial: dict[str, Any]) -> float | None:
    verifier = trial.get("verifier_result") or {}
    rewards = verifier.get("rewards")
    if not isinstance(rewards, dict) or not rewards:
        return None
    if "reward" in rewards:
        values = [rewards["reward"]]
    else:
        values = list(rewards.values())
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    return max(0.0, min(1.0, sum(numeric) / len(numeric)))


def translate(
    result: dict[str, Any],
    key: dict[str, Any],
    suite: str,
    replicate: int,
    default_category: str = "terminal",
    dataset_pin: str | None = None,
) -> list[dict[str, Any]]:
    """Map Harbor trials onto the paired schema, one row per materialized task.

    Categories come off the key so a suite that strata by repository can report
    them without a second pass over the task names.

    Rows carry `dataset_pin` because the comparator's cross-arm check reads it
    off the rows themselves. Omitting it does not make the check lenient, it
    makes it vacuous: both arms would report None and always agree.
    """
    trials = result.get("trial_results")
    if not isinstance(trials, list):
        raise AdapterError("harbor result.json has no trial_results list")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        name = str(trial.get("task_name", ""))
        if name:
            by_task.setdefault(name, []).append(trial)

    rows = []
    for item_id, entry in key.items():
        attempts = by_task.get(item_id, [])
        row = base_row(suite, item_id, replicate)
        row["category"] = (entry or {}).get("category", default_category)
        if not attempts:
            # A task Harbor never ran is a failure of the run, not a model score.
            raise AdapterError(f"harbor produced no trial for task {item_id}")
        rewards = [extract_reward(trial) for trial in attempts]
        scored = [value for value in rewards if value is not None]
        exception = next(
            (trial.get("exception_info") for trial in attempts if trial.get("exception_info")),
            None,
        )
        agent = next(
            (trial.get("agent_result") for trial in attempts if trial.get("agent_result")), {}
        ) or {}
        row.update(
            {
                "score": max(scored) if scored else 0.0,
                "empty_answer": not scored,
                "timeout": bool(
                    exception and "timeout" in str(exception.get("exception_type", "")).lower()
                ),
                "malformed_tool_call": False,
                "context_failure": False,
                "attempts_run": len(attempts),
                "exception_type": (exception or {}).get("exception_type"),
                "output_tokens": agent.get("n_output_tokens"),
                "input_tokens": agent.get("n_input_tokens"),
                "trial_uri": attempts[0].get("trial_uri"),
                "task_checksum": attempts[0].get("task_checksum"),
            }
        )
        if dataset_pin is not None:
            row["dataset_pin"] = dataset_pin
        rows.append(row)
    return rows
