#!/usr/bin/env python3
"""LiveCodeBench v6 adapter for scripts/run_eval_protocol.py.

`prepare` reads the pinned release file directly rather than through the
dataset's loading script, which recent `datasets` releases refuse to execute,
and stores the decoded tests as the answer key. `run` extracts the model's
solution, executes it against every public and private test in a resource-capped
subprocess, and scores pass@1: an item counts only if every test passes.

The tests are model-independent, so executing generated code is the verifier.
That code is untrusted: each run gets a fresh temporary directory, a CPU-time
limit, an address-space limit, and a wall-clock timeout. It is not a security
sandbox and must not be pointed at a shared filesystem it can damage.
"""

import argparse
import base64
import io
import json
import os
import pickle
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Callable

try:
    from _common import (
        AdapterError,
        base_row,
        build_payload,
        check_action,
        env_path,
        env_str,
        execute_order,
        has_repetition_loop,
        load_pins,
        module_pin,
        post_chat,
        raw_response_path as _raw_response_path,
        read_jsonl,
        reasoning_tokens,
        request_with_retries,
        require_pin,
        split_reasoning,
        timeout_row as _timeout_row,
        unpack_choice,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # loading by file spec puts the repo root on sys.path
    from scripts.adapters._common import (  # type: ignore[no-redef]
        AdapterError,
        base_row,
        build_payload,
        check_action,
        env_path,
        env_str,
        execute_order,
        has_repetition_loop,
        load_pins,
        module_pin,
        post_chat,
        raw_response_path as _raw_response_path,
        read_jsonl,
        reasoning_tokens,
        request_with_retries,
        require_pin,
        split_reasoning,
        timeout_row as _timeout_row,
        unpack_choice,
        write_json,
        write_jsonl,
    )


SUITE = "livecodebench_v6"
HARNESS_ID = "builtin-lcb-exec-v1"
VERIFIER_ID = "stdin-and-functional-exec-v1"
DATASET_REPO = "livecodebench/code_generation_lite"
RELEASE_FILES = {"v6": "test6.jsonl", "v5": "test5.jsonl", "v4": "test4.jsonl"}

DEFAULT_MAX_TOKENS = 65536
DEFAULT_EXEC_TIMEOUT = 10.0
DEFAULT_EXEC_MEMORY_MB = 2048
DEFAULT_ITEM_BUDGET = 120.0

ANSWER_INSTRUCTION = (
    "Write a complete solution. Put the final program in a single Python code "
    "block:\n\n```python\n# your solution\n```\n\n"
    "For problems with a starter class, keep the given class and method names "
    "and do not read from standard input. Otherwise read from standard input "
    "and write to standard output."
)

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
METHOD_RE = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(\s*self", re.MULTILINE)

FUNCTIONAL_DRIVER = '''
import json, sys
_payload = json.loads(sys.stdin.read())
_solution = Solution()
_result = getattr(_solution, {method!r})(*_payload)
print(json.dumps(_result, default=list))
'''


class RestrictedUnpickler(pickle.Unpickler):
    """Refuses every class lookup.

    LiveCodeBench stores its private tests as base64(zlib(pickle(str))). The
    payload is a plain string, so nothing legitimate needs find_class, and
    blocking it keeps a dataset revision from executing code at prepare time.
    """

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare", help="materialize problems and decoded tests")
    prepare.add_argument("--release", choices=sorted(RELEASE_FILES), default="v6")
    prepare.add_argument(
        "--lite", action="store_true", help="accepted for protocol compatibility"
    )

    run = sub.add_parser("run", help="score the frozen task order against the server")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--request-timeout", type=float, default=1800.0)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument("--exec-timeout", type=float, default=DEFAULT_EXEC_TIMEOUT)
    run.add_argument("--exec-memory-mb", type=int, default=DEFAULT_EXEC_MEMORY_MB)
    run.add_argument(
        "--item-budget",
        type=float,
        default=DEFAULT_ITEM_BUDGET,
        help="wall-clock seconds for all tests of one item before it is failed",
    )
    run.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter used to execute generated solutions",
    )

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--dataset", help="the 40-character LiveCodeBench dataset commit")
    pin.add_argument("--resolve-dataset", action="store_true")
    return parser.parse_args(argv)


def self_pin() -> str:
    return module_pin([Path(__file__), Path(__file__).resolve().parent / "_common.py"])


def raw_response_path(run_dir: Path, variant: str, replicate: int, item_id: str) -> Path:
    return _raw_response_path(run_dir, SUITE, variant, replicate, item_id)


def validate_pins(pins: dict[str, str]) -> None:
    dataset = pins.get("dataset", "")
    if not re.fullmatch(r"[0-9a-f]{40}", dataset):
        raise AdapterError(
            "pins.dataset must be the 40-character LiveCodeBench dataset commit; "
            f"got {dataset!r}. A branch or tag is not an immutable pin."
        )
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def decode_tests(raw: str) -> list[dict[str, Any]]:
    """Public tests are plain JSON; private tests are base64(zlib(pickle(str)))."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        payload = RestrictedUnpickler(
            io.BytesIO(zlib.decompress(base64.b64decode(raw.encode())))
        ).load()
        if not isinstance(payload, str):
            raise AdapterError("private tests decoded to an unexpected type")
        parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise AdapterError("test cases must decode to a list")
    return parsed


def download_release(release: str, revision: str) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("huggingface_hub is required for prepare") from error
    path = hf_hub_download(
        DATASET_REPO, RELEASE_FILES[release], repo_type="dataset", revision=revision
    )
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt_text(row: dict[str, Any]) -> str:
    text = str(row["question_content"]).strip()
    starter = str(row.get("starter_code") or "").strip()
    if starter:
        text += "\n\nComplete this class:\n\n" + starter
    return text


def materialize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for row in rows:
        item_id = str(row["question_id"])
        tests = decode_tests(row.get("public_test_cases", "")) + decode_tests(
            row.get("private_test_cases", "")
        )
        if not tests:
            raise AdapterError(f"{item_id}: no test cases")
        starter = str(row.get("starter_code") or "").strip()
        method = None
        if starter:
            found = METHOD_RE.search(starter)
            if not found:
                raise AdapterError(f"{item_id}: starter code has no method to call")
            method = found.group(1)
        prompts.append(
            {
                "id": item_id,
                "suite": SUITE,
                "text": build_prompt_text(row),
                "category": f"{row.get('platform', 'unknown')}/{row.get('difficulty', 'unknown')}",
                "platform": row.get("platform"),
                "difficulty": row.get("difficulty"),
            }
        )
        key[item_id] = {
            "tests": tests,
            "starter_code": starter,
            "method": method,
            "platform": row.get("platform"),
            "difficulty": row.get("difficulty"),
            "contest_date": row.get("contest_date"),
        }
    ids = [prompt["id"] for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise AdapterError("duplicate question_id in the release file")
    return prompts, key


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    pins = load_pins()
    validate_pins(pins)
    run_dir = env_path("EVAL_RUN_DIR")
    prompts_path = env_path("EVAL_PROMPTS_JSONL")

    rows = download_release(args.release, pins["dataset"])
    prompts, key = materialize(rows)
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {
            "suite": SUITE,
            "release": args.release,
            "dataset_revision": pins["dataset"],
            "verifier": VERIFIER_ID,
            "adapter": self_pin(),
            "items": key,
        },
    )
    tests = sum(len(entry["tests"]) for entry in key.values())
    print(
        f"materialized {len(prompts)} {SUITE} problems with {tests} tests to {prompts_path}",
        flush=True,
    )
    return 0


def extract_code(text: str) -> str | None:
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return None


def limit_resources(memory_mb: int, cpu_seconds: int) -> None:  # pragma: no cover - child only
    limits = [
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_NOFILE, (256, 256)),
        (resource.RLIMIT_CORE, (0, 0)),
    ]
    if sys.platform.startswith("linux"):
        # An address-space cap is the useful memory limit on Linux; on macOS it
        # applies to reserved rather than resident pages and stops the
        # interpreter from starting at all.
        limits.insert(0, (resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2))
    for which, values in limits:
        try:
            resource.setrlimit(which, values)
        except (OSError, ValueError):
            continue


def normalize_stdout(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def compare_functional(expected: str, produced: str) -> bool:
    try:
        return json.loads(produced) == json.loads(expected)
    except (json.JSONDecodeError, TypeError):
        return normalize_stdout(produced) == normalize_stdout(expected)


def run_one_test(
    code: str,
    test: dict[str, Any],
    entry: dict[str, Any],
    *,
    python: str,
    exec_timeout: float,
    memory_mb: int,
    workdir: Path,
) -> tuple[bool, str]:
    functional = bool(entry.get("method")) and test.get("testtype") == "functional"
    source = code
    if functional:
        source = code + "\n" + FUNCTIONAL_DRIVER.format(method=entry["method"])
        # Functional inputs are newline-separated JSON literals, one per
        # parameter. A single list argument is one line, so splitting on lines
        # is what keeps [1,2,3] one argument rather than three.
        lines = [part for part in str(test.get("input", "")).strip().splitlines() if part.strip()]
        try:
            stdin_data = json.dumps([json.loads(part) for part in lines])
        except json.JSONDecodeError:
            return False, "unparsable_test_input"
    else:
        stdin_data = str(test.get("input", ""))

    script = workdir / "solution.py"
    script.write_text(source, encoding="utf-8")
    try:
        completed = subprocess.run(
            [python, str(script)],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=exec_timeout,
            cwd=workdir,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            preexec_fn=lambda: limit_resources(memory_mb, int(exec_timeout) + 1),
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except OSError as error:
        return False, f"spawn_failed:{error}"

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        return False, "error:" + (detail[-1][:120] if detail else f"exit{completed.returncode}")

    expected = str(test.get("output", ""))
    if functional:
        return compare_functional(expected, completed.stdout), "wrong_answer"
    return normalize_stdout(completed.stdout) == normalize_stdout(expected), "wrong_answer"


def evaluate(
    code: str,
    entry: dict[str, Any],
    *,
    python: str,
    exec_timeout: float,
    memory_mb: int,
    item_budget: float,
) -> dict[str, Any]:
    workdir = Path(tempfile.mkdtemp(prefix="lcb-"))
    started = time.monotonic()
    passed = 0
    try:
        for index, test in enumerate(entry["tests"]):
            if time.monotonic() - started > item_budget:
                return {
                    "passed": False,
                    "status": "item_budget_exceeded",
                    "tests_passed": passed,
                    "tests_total": len(entry["tests"]),
                    "failed_test": index,
                }
            ok, status = run_one_test(
                code, test, entry,
                python=python, exec_timeout=exec_timeout,
                memory_mb=memory_mb, workdir=workdir,
            )
            if not ok:
                return {
                    "passed": False,
                    "status": status,
                    "tests_passed": passed,
                    "tests_total": len(entry["tests"]),
                    "failed_test": index,
                }
            passed += 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {
        "passed": True,
        "status": "passed",
        "tests_passed": passed,
        "tests_total": len(entry["tests"]),
        "failed_test": None,
    }


def score_response(
    item_id: str,
    response: dict[str, Any],
    *,
    entry: dict[str, Any],
    replicate: int,
    thinking: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    content, raw_reasoning, finish_reason, usage = unpack_choice(item_id, response)
    reasoning, answer = split_reasoning(content, raw_reasoning)
    code = extract_code(answer)
    thought = reasoning_tokens(usage, reasoning, answer)

    if code is None:
        verdict = {
            "passed": False,
            "status": "no_code_block",
            "tests_passed": 0,
            "tests_total": len(entry["tests"]),
            "failed_test": None,
        }
    else:
        verdict = evaluate(
            code,
            entry,
            python=args.python,
            exec_timeout=args.exec_timeout,
            memory_mb=args.exec_memory_mb,
            item_budget=args.item_budget,
        )

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            # pass@1: every test must pass, matching LiveCodeBench.
            "score": 1.0 if verdict["passed"] else 0.0,
            "empty_answer": code is None,
            "repetition_loop": has_repetition_loop(answer or reasoning),
            # vLLM discards an unterminated think block, so a reply that ran to
            # the cap arrives with no text at all -- exactly where a loop is most
            # likely. Record whether there was anything to inspect, so a False
            # here is not read as "checked, and clean".
            "repetition_assessed": bool(answer or reasoning),
            "malformed_tool_call": False,
            "premature_final_answer": bool(thinking and answer.strip() and thought == 0),
            "context_failure": finish_reason == "length",
            # Execution outcomes stay out of the shared failure flags: a slow
            # program is not the server failing to answer.
            "execution_status": verdict["status"],
            "tests_passed": verdict["tests_passed"],
            "tests_total": verdict["tests_total"],
            "failed_test": verdict["failed_test"],
            "category": f"{entry.get('platform')}/{entry.get('difficulty')}",
            "platform": entry.get("platform"),
            "difficulty": entry.get("difficulty"),
            "finish_reason": finish_reason,
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": thought,
        }
    )
    return row


def run_item(
    item_id: str,
    text: str,
    entry: dict[str, Any],
    *,
    generation: dict[str, Any],
    model: str,
    seed: int,
    replicate: int,
    variant: str,
    run_dir: Path,
    base_url: str,
    api_key: str,
    args: argparse.Namespace,
    client: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    payload = build_payload(
        text, generation, model=model, seed=seed,
        max_tokens=args.max_tokens, instruction=ANSWER_INSTRUCTION,
    )
    started = time.monotonic()
    response, attempts = request_with_retries(
        item_id, payload, base_url=base_url, api_key=api_key,
        timeout=args.request_timeout, retries=args.retries, client=client,
    )
    if response is None:
        row = _timeout_row(SUITE, item_id, replicate)
        row.update(
            {
                "execution_status": "not_run",
                "category": f"{entry.get('platform')}/{entry.get('difficulty')}",
                "platform": entry.get("platform"),
                "difficulty": entry.get("difficulty"),
            }
        )
    else:
        row = score_response(
            item_id, response, entry=entry, replicate=replicate,
            thinking=bool(generation["enable_thinking"]), args=args,
        )
        path = raw_response_path(run_dir, variant, replicate, item_id)
        write_json(path, response)
        row["raw_response"] = str(path)
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    row["attempts"] = attempts
    return row


def command_run(
    args: argparse.Namespace, client: Callable[..., dict[str, Any]] = post_chat
) -> int:
    check_action("run", SUITE)
    validate_pins(load_pins())

    run_dir = env_path("EVAL_RUN_DIR")
    prompts = {str(row["id"]): row["text"] for row in read_jsonl(env_path("EVAL_PROMPTS_JSONL"))}
    order = json.loads(env_path("EVAL_TASK_ORDER_JSON").read_text(encoding="utf-8"))
    key = json.loads(key_path(run_dir).read_text(encoding="utf-8"))["items"]
    generation = json.loads(env_str("EVAL_GENERATION_JSON"))

    missing = [item_id for item_id in order if item_id not in prompts or item_id not in key]
    if missing:
        raise AdapterError(f"task order references unmaterialized items: {missing[:10]}")

    replicate = int(env_str("EVAL_REPLICATE"))
    seed = int(env_str("EVAL_SEED"))
    variant = env_str("EVAL_VARIANT")
    model = env_str("EVAL_SERVED_MODEL")
    base_url = env_str("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    results_path = env_path("EVAL_RESULTS_JSONL")

    started = time.monotonic()
    rows = execute_order(
        order,
        lambda item_id: run_item(
            item_id, prompts[item_id], key[item_id],
            generation=generation, model=model, seed=seed, replicate=replicate,
            variant=variant, run_dir=run_dir, base_url=base_url, api_key=api_key,
            args=args, client=client,
        ),
        args.concurrency,
    )
    write_jsonl(results_path, rows)

    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.get("execution_status", "unknown")] = (
            statuses.get(row.get("execution_status", "unknown"), 0) + 1
        )
    write_json(
        run_dir / "metadata" / f"{SUITE}-{variant}-r{replicate}.json",
        {
            "suite": SUITE,
            "variant": variant,
            "replicate": replicate,
            "seed": seed,
            "served_model": model,
            "items": len(rows),
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "execution": {
                "timeout_seconds": args.exec_timeout,
                "memory_mb": args.exec_memory_mb,
                "item_budget_seconds": args.item_budget,
                "interpreter": args.python,
            },
            "generation": generation,
            "generation_overrides": {},
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "pass_at_1": round(sum(row["score"] for row in rows) / len(rows), 6),
            "execution_status_counts": statuses,
        },
    )
    print(f"scored {len(rows)} {SUITE} items to {results_path}", flush=True)
    return 0


def command_pin(args: argparse.Namespace) -> int:
    dataset = args.dataset
    if args.resolve_dataset:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise AdapterError("huggingface_hub is required for --resolve-dataset") from error
        dataset = str(HfApi().dataset_info(DATASET_REPO).sha)
    print(
        json.dumps(
            {
                "dataset": dataset or "REPLACE_WITH_LCB_V6_LITE_REVISION",
                "harness": HARNESS_ID,
                "verifier": VERIFIER_ID,
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
