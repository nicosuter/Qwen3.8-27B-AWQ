#!/usr/bin/env python3
"""Shared machinery for the benchmark adapters driven by run_eval_protocol.py.

Every adapter reads its paths and policy from the environment, verifies its own
pins before issuing a request, and emits the paired result schema from EVAL.md.
The parts that must behave identically across suites live here.
"""

import sys

if sys.version_info < (3, 10):
    # Checked before the first PEP 604 annotation below, which would otherwise
    # fail with an unreadable TypeError on the cluster's system Python 3.9.
    raise SystemExit(
        "the adapters need Python 3.10 or newer; run them with the project venv "
        f"(this is {sys.version.split()[0]} at {sys.executable})"
    )

import concurrent.futures
import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


RESULT_BOOL_FIELDS = (
    "must_pass",
    "timeout",
    "empty_answer",
    "repetition_loop",
    "malformed_tool_call",
    "premature_final_answer",
    "context_failure",
)

REQUIRED_GENERATION_KEYS = (
    "enable_thinking",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
)

UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class AdapterError(RuntimeError):
    pass


def env_str(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AdapterError(f"{name} is required but unset")
    return value


def env_path(name: str) -> Path:
    return Path(env_str(name))


def check_action(expected: str, suite: str) -> None:
    action = os.environ.get("EVAL_ACTION")
    if action is not None and action != expected:
        raise AdapterError(f"EVAL_ACTION is {action!r} but this command is {expected!r}")
    actual_suite = os.environ.get("EVAL_SUITE")
    if actual_suite is not None and actual_suite != suite:
        raise AdapterError(f"EVAL_SUITE is {actual_suite!r}; this adapter only serves {suite}")


def module_pin(paths: Sequence[Path]) -> str:
    """Hash every source file whose contents can change scoring behavior."""
    digest = hashlib.sha256()
    for path in sorted(Path(path).resolve() for path in paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def load_pins() -> dict[str, str]:
    try:
        pins = json.loads(env_str("EVAL_PINS_JSON"))
    except json.JSONDecodeError as error:
        raise AdapterError(f"EVAL_PINS_JSON is not valid JSON: {error}") from error
    if not isinstance(pins, dict):
        raise AdapterError("EVAL_PINS_JSON must be an object")
    return {str(key): str(value) for key, value in pins.items()}


def require_pin(pins: dict[str, str], field: str, expected: str) -> None:
    if pins.get(field) != expected:
        raise AdapterError(f"pins.{field} must be {expected!r}; got {pins.get(field)!r}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AdapterError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise AdapterError(f"{path}: no rows")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_payload(
    text: str,
    generation: dict[str, Any],
    *,
    model: str,
    seed: int,
    max_tokens: int,
    instruction: str,
) -> dict[str, Any]:
    missing = [key for key in REQUIRED_GENERATION_KEYS if key not in generation]
    if missing:
        raise AdapterError(f"EVAL_GENERATION_JSON is missing {missing}")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{text}\n\n{instruction}"}],
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "min_p": generation["min_p"],
        "presence_penalty": generation["presence_penalty"],
        "repetition_penalty": generation["repetition_penalty"],
        "chat_template_kwargs": {"enable_thinking": bool(generation["enable_thinking"])},
    }
    effort = generation.get("reasoning_effort")
    if effort:
        payload["reasoning_effort"] = effort
    return payload


def post_chat(
    base_url: str, api_key: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def is_timeout(error: BaseException) -> bool:
    if isinstance(error, (socket.timeout, TimeoutError)):
        return True
    if isinstance(error, urllib.error.URLError):
        return is_timeout(error.reason) if isinstance(error.reason, BaseException) else False
    return False


def is_permanent_rejection(error: BaseException) -> bool:
    """A 4xx other than 429 will not change on retry; a rejected policy field is one."""
    return (
        isinstance(error, urllib.error.HTTPError)
        and 400 <= error.code < 500
        and error.code != 429
    )


def describe_http_error(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode(errors="replace").strip()
    except Exception:  # noqa: BLE001 - the status alone is still worth reporting
        body = ""
    return f"HTTP {error.code}: {body[:500]}" if body else f"HTTP {error.code}"


def request_with_retries(
    item_id: str,
    payload: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    timeout: float,
    retries: int,
    client: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any] | None, int]:
    """Return (response, attempts), or (None, attempts) when the request timed out.

    Timeouts are model behavior and become scored failures. A 4xx other than 429
    aborts at once, and other transport faults abort after `retries`, so
    infrastructure noise never enters the score as a zero.
    """
    attempt = 0
    while True:
        try:
            return client(base_url, api_key, payload, timeout), attempt + 1
        except Exception as error:  # noqa: BLE001 - classified immediately below
            if is_timeout(error):
                return None, attempt + 1
            if is_permanent_rejection(error):
                raise AdapterError(
                    f"{item_id}: server rejected the request, "
                    f"{describe_http_error(error)}"
                ) from error
            if attempt >= retries:
                raise AdapterError(f"{item_id}: request failed: {error}") from error
            attempt += 1
            time.sleep(min(2**attempt, 30))


def execute_order(
    order: Sequence[str], worker: Callable[[str], dict[str, Any]], concurrency: int
) -> list[dict[str, Any]]:
    """Run the frozen task order, returning rows in that order."""
    if concurrency < 1:
        raise AdapterError("--concurrency must be at least 1")
    if concurrency == 1:
        return [worker(item_id) for item_id in order]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {item_id: pool.submit(worker, item_id) for item_id in order}
        return [futures[item_id].result() for item_id in order]


def base_row(suite: str, item_id: str, replicate: int) -> dict[str, Any]:
    row: dict[str, Any] = {"suite": suite, "id": item_id, "replicate": replicate, "score": 0.0}
    row.update({field: False for field in RESULT_BOOL_FIELDS})
    return row


def timeout_row(suite: str, item_id: str, replicate: int) -> dict[str, Any]:
    row = base_row(suite, item_id, replicate)
    row.update({"timeout": True, "empty_answer": True, "finish_reason": "timeout"})
    return row


def raw_response_path(
    run_dir: Path, suite: str, variant: str, replicate: int, item_id: str
) -> Path:
    safe = UNSAFE_NAME_RE.sub("_", item_id)
    return run_dir / "raw-responses" / suite / f"{variant}-r{replicate}" / f"{safe}.json"


def message_text(message: dict[str, Any], field: str) -> str:
    value = message.get(field)
    return value if isinstance(value, str) else ""


def reasoning_tokens(usage: dict[str, Any], reasoning: str) -> int:
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
        return int(details["reasoning_tokens"])
    return len(reasoning.split())


def unpack_choice(item_id: str, response: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    """Return (content, reasoning, finish_reason, usage) or fail loudly."""
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise AdapterError(f"{item_id}: malformed chat completion: {error}") from error
    return (
        message_text(message, "content"),
        message_text(message, "reasoning_content"),
        str(choice.get("finish_reason") or ""),
        response.get("usage") or {},
    )


def has_repetition_loop(
    content: str, *, repeats: int = 4, min_unit: int = 20, window: int = 2000
) -> bool:
    """True when the reply ends in `repeats` consecutive copies of one segment."""
    tail = " ".join(content.split())[-window:]
    for unit in range(min_unit, len(tail) // repeats + 1):
        segment = tail[-unit * repeats :]
        first = segment[:unit]
        if all(segment[index * unit : (index + 1) * unit] == first for index in range(1, repeats)):
            return True
    return False
