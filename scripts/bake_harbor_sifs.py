#!/usr/bin/env python3
"""Bake Harbor task Dockerfiles into .sif images, for an Apptainer-only cluster.

Harbor's Singularity backend cannot build. It takes `docker_image` from
`task.toml [environment]`, converts that one image, and reads the task's
Dockerfile only to scrape `WORKDIR`. SWE-bench Pro tasks carry no
`docker_image` at all: they are Dockerfiles, and the Dockerfile is where the
work happens.

    FROM jefzda/sweap-images:ansible.ansible-...-de01db08d0...
    RUN set -e && git reset --hard fc8197e326... && git clean -fd && git checkout fc8197e326...
    RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh || true

The base image is tagged at one commit and the repo has to be reset to a
different one, so a run that skipped those layers would grade a tree the task
never meant to hand the agent. Harbor refuses to start rather than doing that,
which is the correct failure but still a failure.

So we build what Harbor won't. Each Dockerfile becomes an Apptainer definition,
`RUN` becoming `%post` and `ENV` becoming `%environment`, and the resulting .sif
path is written back into `task.toml` as `docker_image`. That lands on the
backend's own supported path: it recognizes a `docker_image` ending in `.sif` as
pre-built and uses it directly.

Two things about the build are not obvious. It runs unprivileged, as uid 0
inside its own user namespace, so no subuid mapping is needed, which matters
because our user has none. And `--ignore-fakeroot-command` is required: these
images ship a fakeroot shim that Apptainer tries to use and cannot drive, which
fails the build with a `kill` usage error that says nothing about the cause.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT = 3600.0

# Terminus-2 drives a task container through tmux, and if tmux is missing it
# installs it -- apt, pip or a source build -- from inside the container. A task
# run with `--network none` has no mirror to install from, so the agent dies on
# its first command with an error about the terminal rather than about the
# network. Build time is the one moment a task image is allowed network, so tmux
# goes in then. The last line asserts under `set -e`: a task image that ends up
# without tmux fails here, where it is cheap, instead of mid-rollout.
ENSURE_TMUX = """\
    command -v tmux >/dev/null 2>&1 || \\
        (apt-get update && apt-get install -y --no-install-recommends tmux) || \\
        apk add --no-cache tmux || \\
        yum install -y tmux || \\
        dnf install -y tmux
    command -v tmux >/dev/null 2>&1"""


class BakeError(RuntimeError):
    pass


def logical_lines(text: str) -> list[str]:
    """Dockerfile lines with continuations joined and comments dropped.

    SWE-bench Pro's `before_repo_set_cmd` is multi-line for some instances, so a
    parser that read raw lines would take half of a `RUN` and silently build an
    image whose repository was never prepared.
    """
    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if raw.rstrip().endswith("\\"):
            buffer += raw.rstrip()[:-1] + " "
            continue
        joined.append((buffer + raw.strip()).strip())
        buffer = ""
    if buffer.strip():
        joined.append(buffer.strip())
    return joined


def parse_dockerfile(text: str) -> dict[str, Any]:
    image = None
    workdir = None
    envs: list[str] = []
    runs: list[str] = []
    for line in logical_lines(text):
        keyword, _, rest = line.partition(" ")
        rest = rest.strip()
        if keyword == "FROM":
            if image is not None:
                raise BakeError("multi-stage Dockerfiles are not supported")
            image = rest
        elif keyword == "WORKDIR":
            workdir = rest
        elif keyword == "ENV":
            envs.append(rest)
        elif keyword == "RUN":
            runs.append(rest)
    if not image:
        raise BakeError("Dockerfile has no FROM")
    return {"image": image, "workdir": workdir, "envs": envs, "runs": runs}


def definition(parsed: dict[str, Any], ensure_tmux: bool = True) -> str:
    exports = "\n".join(f"    export {entry}" for entry in parsed["envs"])
    body = "\n".join(f"    {command}" for command in parsed["runs"])
    lines = [
        "Bootstrap: docker",
        f"From: {parsed['image']}",
        "",
        "%post",
        "    set -e",
        # Docker builds run as root with HOME=/root, and the uv installer writes
        # into $HOME. Without this it would land somewhere else.
        "    export HOME=/root",
    ]
    if ensure_tmux:
        lines.append(ENSURE_TMUX)
    if exports:
        lines.append(exports)
    lines.append(f"    cd {parsed['workdir'] or '/'}")
    if body:
        lines.append(body)
    if exports:
        lines.extend(["", "%environment", exports])
    return "\n".join(lines) + "\n"


def build(definition_path: Path, sif_path: Path, timeout: float) -> None:
    sif_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "apptainer", "build", "--ignore-fakeroot-command", "-F",
        str(sif_path), str(definition_path),
    ]
    try:
        done = subprocess.run(command, timeout=timeout, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise BakeError("apptainer is not on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise BakeError(f"apptainer build exceeded {timeout}s") from error
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-5:]
        raise BakeError(f"apptainer build failed: {' | '.join(tail)}")
    if not sif_path.is_file():
        raise BakeError(f"apptainer reported success but {sif_path} is missing")


def pin_image(task_toml: Path, sif_path: Path) -> bool:
    """Point the task at its baked image. Returns False if it already was."""
    text = task_toml.read_text(encoding="utf-8")
    if "docker_image" in text:
        return False
    if "[environment]" not in text:
        raise BakeError(f"{task_toml} has no [environment] table")
    updated = text.replace(
        "[environment]", f'[environment]\ndocker_image = "{sif_path}"', 1
    )
    task_toml.write_text(updated, encoding="utf-8")
    return True


def bake(task_dir: Path, sif_dir: Path, timeout: float,
         ensure_tmux: bool = True) -> dict[str, Any]:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        raise BakeError(f"{task_dir.name} has no environment/Dockerfile")
    parsed = parse_dockerfile(dockerfile.read_text(encoding="utf-8"))
    sif_path = sif_dir / f"{task_dir.name}.sif"
    definition_path = sif_path.with_suffix(".def")
    definition_path.parent.mkdir(parents=True, exist_ok=True)
    definition_path.write_text(definition(parsed, ensure_tmux), encoding="utf-8")
    if not sif_path.is_file():
        build(definition_path, sif_path, timeout)
    pinned = pin_image(task_dir / "task.toml", sif_path)
    return {
        "task": task_dir.name,
        "image": parsed["image"],
        "runs": len(parsed["runs"]),
        "sif": str(sif_path),
        "bytes": sif_path.stat().st_size,
        "pinned": pinned,
    }


def selected_tasks(tasks_dir: Path, subset: Path | None) -> list[Path]:
    if subset is None:
        return sorted(path for path in tasks_dir.iterdir() if path.is_dir())
    names = json.loads(subset.read_text(encoding="utf-8"))["task_names"]
    missing = [name for name in names if not (tasks_dir / name).is_dir()]
    if missing:
        raise BakeError(
            f"the task pack is missing {len(missing)} of {len(names)} drawn tasks, "
            f"for example {missing[:3]}"
        )
    return [tasks_dir / name for name in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--sif-dir", type=Path, required=True)
    parser.add_argument("--subset", type=Path, help="bake only the drawn tasks")
    parser.add_argument("--limit", type=int, help="stop after this many tasks")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--report", type=Path, help="write a JSON summary here")
    parser.add_argument(
        "--no-ensure-tmux", action="store_true",
        help="skip installing tmux; only for images that already carry it",
    )
    args = parser.parse_args(argv)

    tasks = selected_tasks(args.tasks_dir, args.subset)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    baked, failures = [], []
    for index, task_dir in enumerate(tasks, start=1):
        try:
            record = bake(task_dir, args.sif_dir, args.timeout,
                          ensure_tmux=not args.no_ensure_tmux)
        except BakeError as error:
            # One unbuildable task should not cost the other 299.
            failures.append({"task": task_dir.name, "error": str(error)})
            print(f"[{index}/{len(tasks)}] FAILED {task_dir.name}: {error}", flush=True)
            continue
        baked.append(record)
        print(
            f"[{index}/{len(tasks)}] {task_dir.name} "
            f"{record['bytes'] / 1e9:.2f} GB",
            flush=True,
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"baked": baked, "failures": failures}, indent=2) + "\n",
            encoding="utf-8",
        )
    total = sum(record["bytes"] for record in baked)
    print(f"baked {len(baked)}, failed {len(failures)}, {total / 1e9:.1f} GB", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BakeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
