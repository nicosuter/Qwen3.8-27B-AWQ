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


def _load_admission() -> Any:
    """Load the sibling module by path, however this adapter was launched.

    Adapters are run as scripts, imported by the test suite through a spec, and
    exec'd inside the sbatch's inline Python. Only a path-based load works in
    all three.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_common_admission", Path(__file__).resolve().parent / "_admission.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


admission = _load_admission()

# Resolved once from the environment, or set outright by a test. Held here
# rather than passed through every adapter so that the fourteen of them cannot
# drift apart on a policy that has to be identical across both arms.
_ADMISSION: dict[str, Any] = {"budget": None, "priors": None, "suite": "", "resolved": False}


def configure_admission(*, budget: Any = None, priors: Any = None, suite: str = "") -> None:
    _ADMISSION.update(budget=budget, priors=priors, suite=suite, resolved=True)


def admission_settings() -> dict[str, Any]:
    if not _ADMISSION["resolved"]:
        priors_path = os.environ.get("EVAL_ADMISSION_PRIORS", "").strip()
        configure_admission(
            budget=admission.from_environment(dict(os.environ)),
            priors=admission.load_priors(Path(priors_path)) if priors_path else None,
            suite=os.environ.get("EVAL_SUITE", ""),
        )
    return _ADMISSION


def payload_text(payload: dict[str, Any]) -> str:
    """The text a payload carries, for sizing its prompt.

    Multi-part content is where the images live. Their base64 is deliberately
    not counted: it is an order of magnitude longer than the tokens it becomes,
    so counting it would reserve the pool for one picture. The suite's measured
    prompt length covers them instead.
    """
    parts: list[str] = []
    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for piece in content:
                if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                    parts.append(piece["text"])
    return "\n".join(parts)


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
    # Qwen3.8's chat template reads reasoning_effort as a template variable and
    # defaults it to xhigh. Sent as a top-level field it is accepted by the
    # server and silently ignored, so a protocol asking for medium would run at
    # xhigh and record itself as compliant.
    template_kwargs: dict[str, Any] = {"enable_thinking": bool(generation["enable_thinking"])}
    effort = generation.get("reasoning_effort")
    if effort:
        template_kwargs["reasoning_effort"] = effort
    # One seed for every request makes each item draw the same uniform stream,
    # applied to different logits: their sampling noise is then correlated, and
    # the comparator's item-clustered bootstrap assumes items are independent.
    # Derive the seed from the prompt so a run stays reproducible while items
    # stay independent of one another.
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{text}\n\n{instruction}"}],
        "seed": int.from_bytes(digest[:4], "big"),
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "min_p": generation["min_p"],
        "presence_penalty": generation["presence_penalty"],
        "repetition_penalty": generation["repetition_penalty"],
        "chat_template_kwargs": template_kwargs,
    }
    # max_tokens <= 0 means "whatever is left of the context window", which is
    # what the server does when the field is absent. It is not a convenience:
    # nothing tells the model what its budget is -- the API does not carry it
    # and the prompt does not mention it -- so the model cannot spend against
    # one. Under a 131072 cap, nine of the ten truncated MathArena items had
    # spent 131072 tokens on reasoning and emitted zero answer tokens. A cap
    # therefore does not buy a shorter answer, it buys no answer, and it costs
    # the arm that reasons longer roughly twice as often, which is exactly the
    # arm under test.
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
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
    # Resolved at call time, not bound as a default, so a test can patch
    # time.sleep instead of every caller having to thread a stub through.
    sleep: Callable[[float], None] | None = None,
) -> tuple[dict[str, Any], int]:
    """Return (response, attempts), or raise once the attempts are spent.

    A timeout is retried like any other transport fault, and is never scored.
    It used to be treated as model behavior, which was wrong in a way that took
    a campaign to see: the server returns when it reaches the context limit --
    `finish_reason` is `length` -- so the only way to reach a socket timeout is
    for the client to give up while the request sat in a queue. Charging an
    item for contention turned an admission-control mistake into a 15-point
    RULER regression that did not exist.

    A 4xx other than 429 still aborts at once; it will answer the same however
    many times it is asked.

    Each attempt reserves its KV footprint before it is sent and returns it when
    it ends, so a retry still cannot be what overfills the cache while no single
    hold can outlast one request timeout. Holding one reservation across the
    whole chain instead made the worst case the timeout times the attempt count,
    and against a server that had stopped answering that became a deduction from
    the budget nothing gave back.
    """
    settings = admission_settings()
    budget = settings["budget"]
    tokens = 0
    if budget is not None:
        tokens = admission.reservation(
            payload_text(payload),
            settings["suite"],
            settings["priors"] or {"suites": {}, "default": {"prompt": 2048, "output": 4096}},
            max_tokens=int(payload.get("max_tokens") or 0),
        )
    attempt = 0
    while True:
        if budget is not None:
            budget.acquire(item_id, tokens)
        try:
            return client(base_url, api_key, payload, timeout), attempt + 1
        except Exception as error:  # noqa: BLE001 - classified immediately below
            if is_permanent_rejection(error):
                raise AdapterError(
                    f"{item_id}: server rejected the request, "
                    f"{describe_http_error(error)}"
                ) from error
            if attempt >= retries:
                # Failing the suite is the point. A run whose requests will not
                # complete has not measured the checkpoint, and the harness
                # already knows how to exclude a failed suite from the macro --
                # which is strictly better than contributing zeros to it.
                raise AdapterError(
                    f"{item_id}: request failed after {attempt + 1} attempts: {error}"
                ) from error
            attempt += 1
        finally:
            # Released on the failure path too. A leak here ratchets the budget
            # down over a long suite until the run stalls with the server idle.
            if budget is not None:
                budget.release(item_id)
        # Outside the reservation: a request waiting out its backoff is not
        # occupying the cache, and holding through the sleep would price the
        # queue for capacity nobody is using.
        (sleep or time.sleep)(min(2**attempt, 30))


def carried_forward(source: Any = None) -> dict[str, dict[str, Any]]:
    """Rows from an earlier attempt that a rerun does not have to buy again.

    Set EVAL_RESUME_FROM to the superseded results file and a rerun re-measures
    only the items that have no measurement: a timeout wrote a zero the model
    never produced, and a deferred row was never executed at all. Everything
    else is the same checkpoint answering the same item under the same cap, and
    the protocol already reuses that across jobs -- pairing is per item, not per
    process.

    This is not the relaxation the --max-tokens check refuses. Truncation at the
    cap is an observation, and it lands on the longest reasoning, so keeping the
    items that fit and rescoring the rest would select on the outcome along the
    axis the arms differ on. A timeout leaves nothing to select on, and every
    item in the frozen order is still scored exactly once.

    Whoever sets the variable has already decided the rows are comparable;
    paired-suite-eval.sbatch sets it only when the recorded run matches this one
    in checkpoint, cap and timeout scale, and differs solely in having timed out.
    """
    if source is None:
        source = os.environ.get("EVAL_RESUME_FROM", "")
    if not source:
        return {}
    path = Path(source)
    if not path.is_file():
        raise AdapterError(
            f"EVAL_RESUME_FROM={path} does not exist; refusing to silently rerun "
            "the whole suite, which would look the same as a resume that worked"
        )
    carried: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("timeout") or row.get("deferred"):
            continue
        item_id = row.get("id")
        if item_id is not None:
            carried[str(item_id)] = row
    return carried


def execute_order(
    order: Sequence[str],
    worker: Callable[[str], dict[str, Any]],
    concurrency: int,
    *,
    reuse: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the frozen task order, returning rows in that order.

    Items already measured by an earlier attempt are spliced back in rather than
    re-requested, so the returned list is the whole order either way.
    """
    if concurrency < 1:
        raise AdapterError("--concurrency must be at least 1")
    if reuse is None:
        reuse = carried_forward()
    pending = [item_id for item_id in order if item_id not in reuse]
    if reuse:
        print(
            f"resuming: {len(reuse)} of {len(order)} items carried forward, "
            f"{len(pending)} to measure",
            flush=True,
        )
    if concurrency == 1 or len(pending) <= 1:
        fresh = {item_id: worker(item_id) for item_id in pending}
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {item_id: pool.submit(worker, item_id) for item_id in pending}
            fresh = {item_id: future.result() for item_id, future in futures.items()}
    return [reuse[item_id] if item_id in reuse else fresh[item_id] for item_id in order]


def timing(started_wall: float, started_mono: float) -> dict[str, float]:
    """When a request ran, not just how long it took.

    Duration alone cannot say how many requests were in flight at once, so
    occupancy has to be reconstructed by replaying the thread pool and assuming
    how it scheduled. Absolute stamps make that a measurement instead. The
    monotonic clock still supplies the duration, since the wall clock can step.
    """
    elapsed = time.monotonic() - started_mono
    return {
        "started_at": round(started_wall, 3),
        "finished_at": round(started_wall + elapsed, 3),
        "elapsed_seconds": round(elapsed, 3),
    }


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


CHARS_PER_TOKEN = 4


def reasoning_tokens(usage: dict[str, Any], reasoning: str, content: str = "") -> int:
    """Reasoning tokens, inferred when the server strips them without reporting.

    The pinned vLLM build removes the think block from `content` and returns
    nothing in `reasoning_content`, so the only evidence that thinking happened
    is the gap between tokens generated and tokens visible.
    """
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
        return int(details["reasoning_tokens"])
    if reasoning:
        return len(reasoning.split())
    completion = usage.get("completion_tokens")
    if isinstance(completion, int) and not isinstance(completion, bool):
        return max(0, completion - len(content) // CHARS_PER_TOKEN)
    return 0


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


THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_reasoning(content: str, reasoning: str) -> tuple[str, str]:
    """Return (reasoning, answer) even when the server did not separate them.

    Qwen3.8's template opens `<think>` in the prompt itself, so a server whose
    reasoning parser is not splitting hands back one blob whose only marker is
    the closing tag. Scoring that blob as the answer would credit values the
    model merely considered while thinking.
    """
    if reasoning:
        return reasoning, content
    match = THINK_BLOCK_RE.search(content)
    if match:
        return match.group(1), THINK_BLOCK_RE.sub("", content)
    if "</think>" in content:
        head, _, tail = content.partition("</think>")
        return head, tail
    return "", content


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
