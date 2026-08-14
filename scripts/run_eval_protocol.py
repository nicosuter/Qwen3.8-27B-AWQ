#!/usr/bin/env python3
"""Execute EVAL.md's paired protocol around an Apptainer vLLM server.

Benchmark-specific adapters are deliberately separate.  Each adapter receives a
strict environment contract, materializes prompts before inference, and emits
the repository's common item-level JSONL schema.  This runner owns everything
that must be identical across adapters: checkpoint order, server flags,
generation policy, endpoint smoke tests, pairing, contamination review, result
validation, and reproducibility metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
REQUIRED_SUITES = {
    "bfcl_v4",
    "terminal_bench_2_1",
    "livecodebench_v6",
    "gpqa_diamond",
    "matharena_2026_06",
    "multimodal",
}
RESULT_BOOL_FIELDS = (
    "must_pass",
    "timeout",
    "empty_answer",
    "repetition_loop",
    "malformed_tool_call",
    "premature_final_answer",
    "context_failure",
)


class ProtocolError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prepare", "run", "all"), default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ProtocolError(f"unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load protocol config {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ProtocolError("protocol config must be a JSON object")
    config = expand_environment(raw)
    if config.get("version") != 1:
        raise ProtocolError("only protocol config version 1 is supported")
    if config.get("served_model_name") != "openai/qwen38-eval":
        raise ProtocolError("served_model_name must be openai/qwen38-eval")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 4 or any(
        not isinstance(seed, int) for seed in seeds
    ):
        raise ProtocolError("seeds must contain at least four integer seeds")
    suites = config.get("suites")
    if not isinstance(suites, list):
        raise ProtocolError("suites must be a list")
    names = [suite.get("name") for suite in suites if isinstance(suite, dict)]
    if len(names) != len(suites) or len(set(names)) != len(names):
        raise ProtocolError("suite entries must have unique names")
    if set(names) != REQUIRED_SUITES:
        raise ProtocolError(
            f"suite labels must be exactly {sorted(REQUIRED_SUITES)}; got {sorted(names)}"
        )
    expected_replicates = {
        "bfcl_v4": 1,
        "terminal_bench_2_1": 3,
        "livecodebench_v6": 4,
        "gpqa_diamond": 4,
        "matharena_2026_06": 4,
        "multimodal": 1,
    }
    for suite in suites:
        name = suite["name"]
        if suite.get("replicates") != expected_replicates[name]:
            raise ProtocolError(
                f"{name}: replicates must be {expected_replicates[name]}"
            )
        for action in ("prepare", "run"):
            command = suite.get(action)
            if not isinstance(command, list) or not command or any(
                not isinstance(part, str) or not part for part in command
            ):
                raise ProtocolError(f"{name}: {action} must be a non-empty argv list")
            if any("REPLACE_" in part or "PINNED_" in part for part in command):
                raise ProtocolError(f"{name}: {action} still contains a placeholder")
    for variant in ("baseline", "candidate"):
        entry = config.get(variant)
        if not isinstance(entry, dict) or not entry.get("model"):
            raise ProtocolError(f"{variant}.model is required")
    if not config["baseline"].get("revision"):
        raise ProtocolError("baseline.revision must be pinned")
    server = config.get("server")
    if not isinstance(server, dict) or not isinstance(server.get("flags"), list):
        raise ProtocolError("server.flags must be an argv list")
    required_flag_values = {
        "--tensor-parallel-size": "1",
        "--data-parallel-size": "8",
        "--max-model-len": "262144",
        "--kv-cache-dtype": "auto",
        "--reasoning-parser": "qwen3",
        "--tool-call-parser": "qwen3_coder",
    }
    flags = server["flags"]
    for flag, expected in required_flag_values.items():
        try:
            actual = flags[flags.index(flag) + 1]
        except (ValueError, IndexError) as error:
            raise ProtocolError(f"server.flags must contain {flag} {expected}") from error
        if actual != expected:
            raise ProtocolError(f"{flag} must be {expected}, got {actual}")
    if "--enable-auto-tool-choice" not in flags:
        raise ProtocolError("server.flags must enable automatic tool choice")
    forbidden = {"--speculative-config", "--speculative-model", "--enable-chunked-prefill"}
    if forbidden & set(flags):
        raise ProtocolError(
            "primary server flags contain a speculative/non-protocol option: "
            + ", ".join(sorted(forbidden & set(flags)))
        )
    expected_generation = {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    if config.get("generation") != expected_generation:
        raise ProtocolError("generation policy does not exactly match EVAL.md")
    if not config.get("calibration_manifest"):
        raise ProtocolError("calibration_manifest is required")
    return config


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def lock_config(run_dir: Path, config: dict[str, Any]) -> None:
    path = run_dir / "protocol.lock.json"
    encoded = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ProtocolError(
            f"{path} differs from the requested config; use a new run directory"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def adapter_environment(
    config: dict[str, Any], run_dir: Path, suite: str
) -> dict[str, str]:
    prompts = run_dir / "materialized" / f"{suite}.jsonl"
    order = run_dir / "orders" / f"{suite}.json"
    env = os.environ.copy()
    env.update(
        {
            "EVAL_RUN_DIR": str(run_dir),
            "EVAL_SUITE": suite,
            "EVAL_PROMPTS_JSONL": str(prompts),
            "EVAL_TASK_ORDER_JSON": str(order),
            "EVAL_ORDER_SEED": str(config["order_seed"]),
            "EVAL_SERVED_MODEL": config["served_model_name"],
            "EVAL_GENERATION_JSON": canonical_json(config["generation"]),
        }
    )
    return env


def run_logged(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
    acceptable: tuple[int, ...] = (0,),
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    display = " ".join(json.dumps(part) for part in command)
    print(f"run: {display}", flush=True)
    if dry_run:
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode not in acceptable:
        raise ProtocolError(
            f"command failed with exit {process.returncode}; see {log_path}"
        )
    return process.returncode


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise ProtocolError(f"cannot read {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ProtocolError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    if not rows:
        raise ProtocolError(f"{path}: no rows")
    return rows


def validate_prompts(path: Path, suite: str) -> list[str]:
    rows = read_jsonl(path)
    ids = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row.get("id"), (str, int)):
            raise ProtocolError(f"{path}:{index}: id is required")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ProtocolError(f"{path}:{index}: non-empty text is required")
        if "suite" in row and row["suite"] != suite:
            raise ProtocolError(f"{path}:{index}: suite must be {suite}")
        ids.append(str(row["id"]))
    if len(ids) != len(set(ids)):
        raise ProtocolError(f"{path}: duplicate prompt IDs")
    return ids


def prepare_suites(config: dict[str, Any], run_dir: Path, dry_run: bool) -> None:
    for suite in config["suites"]:
        name = suite["name"]
        env = adapter_environment(config, run_dir, name)
        env["EVAL_ACTION"] = "prepare"
        run_logged(
            suite["prepare"],
            env=env,
            log_path=run_dir / "logs" / f"prepare-{name}.log",
            dry_run=dry_run,
        )
        if dry_run:
            continue
        ids = validate_prompts(Path(env["EVAL_PROMPTS_JSONL"]), name)
        rng = random.Random(config["order_seed"])
        rng.shuffle(ids)
        write_json(Path(env["EVAL_TASK_ORDER_JSON"]), ids)


def audit_overlap(config: dict[str, Any], run_dir: Path, dry_run: bool) -> set[tuple[str, str]]:
    calibration = Path(config["calibration_manifest"])
    if not dry_run and not calibration.is_file():
        raise ProtocolError(f"missing calibration manifest: {calibration}")
    report_path = run_dir / "overlap" / "audit.json"
    command = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "audit_eval_overlap.py"),
        "--calibration",
        str(calibration),
    ]
    prompt_to_suite = {}
    for suite in config["suites"]:
        path = run_dir / "materialized" / f"{suite['name']}.jsonl"
        command += ["--eval", str(path)]
        prompt_to_suite[str(path)] = suite["name"]
    command += ["--output", str(report_path)]
    run_logged(
        command,
        env=os.environ.copy(),
        log_path=run_dir / "logs" / "overlap-audit.log",
        dry_run=dry_run,
        acceptable=(0, 2),
    )
    if dry_run:
        return set()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matches = report["matches"]
    if not matches:
        return set()
    review_value = config.get("overlap_review")
    if not review_value:
        raise ProtocolError(
            f"overlap audit flagged {len(matches)} pairs; review {report_path}, "
            "record every decision in overlap_review, and resume"
        )
    review_path = Path(review_value)
    try:
        reviews = json.loads(review_path.read_text(encoding="utf-8"))["reviews"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid overlap review {review_path}: {error}") from error
    reviewed = {
        (str(row["eval_file"]), str(row["eval_id"]), str(row["calibration_id"])): row
        for row in reviews
    }
    confirmed: set[tuple[str, str]] = set()
    for match in matches:
        key = (
            str(match["eval_file"]),
            str(match["eval_id"]),
            str(match["calibration_id"]),
        )
        decision = reviewed.get(key)
        if not decision or not isinstance(decision.get("confirmed_overlap"), bool):
            raise ProtocolError(f"missing overlap-review decision for {key}")
        if decision["confirmed_overlap"]:
            suite = prompt_to_suite.get(key[0])
            if suite is None:
                raise ProtocolError(f"unknown eval file in overlap report: {key[0]}")
            confirmed.add((suite, key[1]))
    shutil.copy2(review_path, run_dir / "overlap" / "review.json")
    return confirmed


def apptainer_prefix(
    image: Path, bind_paths: list[Path], config: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    command = ["apptainer", "exec", "--nv", "--cleanenv"]
    seen = set()
    for path in bind_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if "," in str(resolved):
            raise ProtocolError(f"Apptainer bind path may not contain a comma: {resolved}")
        seen.add(resolved)
        command += ["--bind", f"{resolved}:{resolved}"]
    command.append(str(image.resolve()))
    env = os.environ.copy()
    forwarded = (
        "CUDA_VISIBLE_DEVICES",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_DEBUG",
    )
    for name in forwarded:
        if name in os.environ:
            env[f"APPTAINERENV_{name}"] = os.environ[name]
    hf_home = Path(config.get("hf_home", os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")))
    env["APPTAINERENV_HF_HOME"] = str(hf_home)
    env["APPTAINERENV_TOKENIZERS_PARALLELISM"] = "true"
    return command, env


def check_compatibility(
    config: dict[str, Any], image: Path, run_dir: Path, dry_run: bool
) -> None:
    baseline = config["baseline"]
    candidate = config["candidate"]
    bind_paths = [PROJECT_DIR, run_dir, Path(candidate["model"]).parent]
    hf_home = Path(config.get("hf_home", os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")))
    bind_paths.append(hf_home)
    prefix, env = apptainer_prefix(image, bind_paths, config)
    command = prefix + [
        "python3",
        str(PROJECT_DIR / "scripts" / "check_eval_compat.py"),
        "--baseline",
        baseline["model"],
        "--baseline-revision",
        baseline["revision"],
        "--candidate",
        candidate["model"],
        "--output",
        str(run_dir / "metadata" / "checkpoint-compatibility.json"),
    ]
    run_logged(
        command,
        env=env,
        log_path=run_dir / "logs" / "checkpoint-compatibility.log",
        dry_run=dry_run,
    )


def http_json(url: str, *, payload: dict[str, Any] | None = None, timeout: float = 30) -> Any:
    data = canonical_json(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(base_url: str, process: subprocess.Popen[Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ProtocolError(f"vLLM exited during startup with {process.returncode}")
        try:
            http_json(base_url + "/models", timeout=5)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = str(error)
            time.sleep(2)
    raise ProtocolError(f"vLLM did not become ready: {last_error}")


def tool_smoke(base_url: str, served_model: str) -> dict[str, Any]:
    payload = {
        "model": served_model,
        "messages": [
            {
                "role": "user",
                "content": "Call get_weather for Zurich. Use the tool; do not answer directly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 256,
    }
    response = http_json(base_url + "/chat/completions", payload=payload, timeout=180)
    try:
        calls = response["choices"][0]["message"]["tool_calls"]
        call = calls[0]
        name = call["function"]["name"]
        arguments = json.loads(call["function"]["arguments"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"native tool-call smoke returned malformed output: {error}") from error
    if name != "get_weather" or "zurich" not in str(arguments.get("city", "")).lower():
        raise ProtocolError(f"native tool-call smoke selected {name} with {arguments}")
    return response


def start_server(
    config: dict[str, Any],
    image: Path,
    run_dir: Path,
    variant: str,
    replicate: int,
) -> tuple[subprocess.Popen[Any], Any, str, str]:
    server = config["server"]
    model = config[variant]["model"]
    health_base = f"http://{server.get('health_host', '127.0.0.1')}:{server.get('port', 8000)}/v1"
    public_base = server.get("public_base_url", health_base)
    bind_paths = [PROJECT_DIR, run_dir]
    model_path = Path(model)
    if model_path.exists():
        bind_paths.append(model_path.parent)
    hf_home = Path(config.get("hf_home", os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")))
    bind_paths.append(hf_home)
    prefix, env = apptainer_prefix(image, bind_paths, config)
    command = prefix + [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        config["served_model_name"],
        "--host",
        str(server.get("host", "0.0.0.0")),
        "--port",
        str(server.get("port", 8000)),
    ] + [str(value) for value in server["flags"]]
    revision = config[variant].get("revision")
    if revision:
        command += ["--revision", revision]
    log_path = run_dir / "logs" / f"server-{replicate}-{variant}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    print("start: " + " ".join(json.dumps(part) for part in command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        wait_for_server(health_base, process, int(server.get("startup_timeout_seconds", 1800)))
    except Exception:
        stop_server(process, log_handle)
        raise
    return process, log_handle, health_base, public_base


def stop_server(process: subprocess.Popen[Any], log_handle: Any) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    log_handle.close()


def validate_results(path: Path, suite: str, replicate: int, expected_ids: set[str]) -> None:
    rows = read_jsonl(path)
    seen = set()
    for line_number, row in enumerate(rows, 1):
        if row.get("suite") != suite:
            raise ProtocolError(f"{path}:{line_number}: suite must be {suite}")
        if row.get("replicate") != replicate:
            raise ProtocolError(f"{path}:{line_number}: replicate must be {replicate}")
        item_id = str(row.get("id", ""))
        if not item_id or item_id in seen:
            raise ProtocolError(f"{path}:{line_number}: missing or duplicate id")
        seen.add(item_id)
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ProtocolError(f"{path}:{line_number}: score must be in [0, 1]")
        for field in RESULT_BOOL_FIELDS:
            if not isinstance(row.get(field), bool):
                raise ProtocolError(f"{path}:{line_number}: {field} must be boolean")
    if seen != expected_ids:
        raise ProtocolError(
            f"{path}: result IDs differ from materialized prompts; "
            f"missing={sorted(expected_ids - seen)[:10]} extra={sorted(seen - expected_ids)[:10]}"
        )


def run_primary(config: dict[str, Any], image: Path, run_dir: Path) -> None:
    seeds = config["seeds"]
    suites_by_replicate = {
        replicate: [suite for suite in config["suites"] if suite["replicates"] > replicate]
        for replicate in range(4)
    }
    for replicate, seed in enumerate(seeds[:4]):
        variants = ("baseline", "candidate") if replicate % 2 == 0 else ("candidate", "baseline")
        for variant in variants:
            process, log_handle, health_base, public_base = start_server(
                config, image, run_dir, variant, replicate
            )
            try:
                models = http_json(health_base + "/models")
                write_json(run_dir / "smoke" / f"{replicate}-{variant}-models.json", models)
                smoke = tool_smoke(health_base, config["served_model_name"])
                write_json(run_dir / "smoke" / f"{replicate}-{variant}-tool.json", smoke)
                for suite in suites_by_replicate[replicate]:
                    name = suite["name"]
                    output = run_dir / "raw" / variant / f"{name}-r{replicate}.jsonl"
                    env = adapter_environment(config, run_dir, name)
                    env.update(
                        {
                            "EVAL_ACTION": "run",
                            "EVAL_VARIANT": variant,
                            "EVAL_REPLICATE": str(replicate),
                            "EVAL_SEED": str(seed),
                            "EVAL_RESULTS_JSONL": str(output),
                            "OPENAI_API_KEY": "EMPTY",
                            "OPENAI_BASE_URL": public_base,
                        }
                    )
                    run_logged(
                        suite["run"],
                        env=env,
                        log_path=run_dir / "logs" / f"run-{replicate}-{variant}-{name}.log",
                        dry_run=False,
                    )
                    expected = set(
                        validate_prompts(run_dir / "materialized" / f"{name}.jsonl", name)
                    )
                    validate_results(output, name, replicate, expected)
            finally:
                stop_server(process, log_handle)


def merge_results(run_dir: Path, variant: str) -> Path:
    paths = sorted((run_dir / "raw" / variant).glob("*.jsonl"))
    if not paths:
        raise ProtocolError(f"no raw {variant} results")
    rows = [row for path in paths for row in read_jsonl(path)]
    rows.sort(key=lambda row: (row["suite"], str(row["id"]), row["replicate"]))
    output = run_dir / "results" / f"{variant}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output


def clean_results(source: Path, output: Path, excluded: set[tuple[str, str]]) -> None:
    rows = [
        row
        for row in read_jsonl(source)
        if (str(row["suite"]), str(row["id"])) not in excluded
    ]
    if not rows:
        raise ProtocolError("all evaluation rows were excluded as calibration overlaps")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def compare_results(
    run_dir: Path, baseline: Path, candidate: Path, label: str
) -> int:
    output = run_dir / "results" / f"comparison-{label}.json"
    command = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "compare_eval_results.py"),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
    ]
    return run_logged(
        command,
        env=os.environ.copy(),
        log_path=run_dir / "logs" / f"compare-{label}.log",
        dry_run=False,
        acceptable=(0, 2),
    )


def capture_metadata(config_path: Path, image: Path, run_dir: Path) -> None:
    commands = {
        "apptainer": ["apptainer", "version"],
        "nvidia_smi": ["nvidia-smi", "-q"],
        "git_commit": ["git", "rev-parse", "HEAD"],
    }
    captured = {}
    for name, command in commands.items():
        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
            captured[name] = {"returncode": result.returncode, "output": result.stdout}
        except (OSError, subprocess.TimeoutExpired) as error:
            captured[name] = {"error": str(error)}
    captured["files"] = {
        "protocol_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "apptainer_image": {"path": str(image), "sha256": sha256_file(image)},
    }
    calibration = Path(json.loads((run_dir / "protocol.lock.json").read_text())["calibration_manifest"])
    if calibration.is_file():
        captured["files"]["calibration_manifest"] = {
            "path": str(calibration),
            "sha256": sha256_file(calibration),
        }
    write_json(run_dir / "metadata" / "environment.json", captured)


def write_artifact_manifest(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "sha256sums.json":
            rows.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(run_dir / "sha256sums.json", rows)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    config = load_config(config_path)
    lock_config(run_dir, config)

    if args.phase in ("prepare", "all"):
        prepare_suites(config, run_dir, args.dry_run)
    if args.dry_run:
        print("protocol-dry-run=ok")
        return 0
    confirmed_overlaps = audit_overlap(config, run_dir, False)
    if args.phase == "prepare":
        write_artifact_manifest(run_dir)
        print("protocol-prepare=ok")
        return 0

    if args.image is None:
        raise ProtocolError("--image is required for the run phase")
    image = args.image.resolve()
    if not image.is_file():
        raise ProtocolError(f"missing Apptainer image: {image}")
    if shutil.which("apptainer") is None:
        raise ProtocolError("apptainer is not available")
    candidate_path = Path(config["candidate"]["model"])
    if not (candidate_path / "config.json").is_file():
        raise ProtocolError(f"candidate checkpoint is incomplete: {candidate_path}")

    check_compatibility(config, image, run_dir, False)
    capture_metadata(config_path, image, run_dir)
    run_primary(config, image, run_dir)
    baseline = merge_results(run_dir, "baseline")
    candidate = merge_results(run_dir, "candidate")
    full_status = compare_results(run_dir, baseline, candidate, "full")

    clean_baseline = run_dir / "results" / "baseline-calibration-clean.jsonl"
    clean_candidate = run_dir / "results" / "candidate-calibration-clean.jsonl"
    clean_results(baseline, clean_baseline, confirmed_overlaps)
    clean_results(candidate, clean_candidate, confirmed_overlaps)
    clean_status = compare_results(
        run_dir, clean_baseline, clean_candidate, "calibration-clean"
    )
    write_json(
        run_dir / "results" / "decision.json",
        {
            "automated_quality_gate": "pass" if clean_status == 0 else "fail",
            "manual_regression_cluster_review_required": True,
            "full_comparison_exit_code": full_status,
            "calibration_clean_comparison_exit_code": clean_status,
            "confirmed_overlap_items": len(confirmed_overlaps),
        },
    )
    write_artifact_manifest(run_dir)
    print(
        "automated-quality-gate=" + ("PASS" if clean_status == 0 else "FAIL")
    )
    print("manual-regression-cluster-review=REQUIRED")
    return clean_status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProtocolError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
