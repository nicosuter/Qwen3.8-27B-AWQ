#!/usr/bin/env python3
"""Berkeley Function Calling Leaderboard adapter for scripts/run_eval_protocol.py.

Covers the static AST categories: simple, multiple, parallel, parallel_multiple
and irrelevance. The model is served its tools natively, so this is the one
suite where `malformed_tool_call` measures something real.

Note on the version: the protocol names this suite `bfcl_v4`, but no v4 dataset
is published on the Hub. The pinned data is the v3 static split from
`gorilla-llm/Berkeley-Function-Calling-Leaderboard`, and the run metadata records
exactly which files and revision were scored. Executable, live, multi-turn and
web-search categories are deliberately excluded: they need the Gorilla API
simulators or a live network, neither of which belongs in a quantization gate.
"""

import argparse
import json
import os
import re
import sys
import time
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


SUITE = "bfcl_v4"
HARNESS_ID = "builtin-bfcl-ast-v1"
VERIFIER_ID = "bfcl-ast-match-v1"
DATASET_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Static AST categories only. irrelevance has no ground truth by design: the
# correct behavior is to call nothing.
CATEGORIES = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
}
SCORED_WITH_GROUND_TRUTH = tuple(name for name in CATEGORIES if name != "irrelevance")

DEFAULT_MAX_TOKENS = 32768

ANSWER_INSTRUCTION = (
    "Use the provided tools to satisfy the request. Call every tool the request "
    "requires, with arguments taken from the request. If no provided tool fits "
    "the request, do not call any tool and say so instead."
)

TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
    "tuple": "array",
    "any": None,
}

PUNCTUATION_RE = re.compile(r"[\s_\-.,!?'\"]+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare", help="materialize prompts, tools and ground truth")
    prepare.add_argument(
        "--categories",
        default=",".join(CATEGORIES),
        help="comma-separated subset of " + ", ".join(CATEGORIES),
    )

    run = sub.add_parser("run", help="score the frozen task order against the server")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--request-timeout", type=float, default=1800.0)
    run.add_argument("--retries", type=int, default=2)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--dataset", help="the 40-character BFCL dataset commit")
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
            "pins.dataset must be the 40-character BFCL dataset commit; "
            f"got {dataset!r}. A branch or tag is not an immutable pin."
        )
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def download_json_lines(name: str, revision: str) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("huggingface_hub is required for prepare") from error
    path = hf_hub_download(DATASET_REPO, name, repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert_schema(node: Any) -> Any:
    """BFCL schemas use dict/float/tuple where JSON Schema wants object/number/array."""
    if not isinstance(node, dict):
        return node
    converted = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            mapped = TYPE_MAP.get(value, value)
            if mapped is None:
                continue
            converted[key] = mapped
        elif key in ("properties", "items", "additionalProperties"):
            if isinstance(value, dict) and key == "properties":
                converted[key] = {k: convert_schema(v) for k, v in value.items()}
            else:
                converted[key] = convert_schema(value)
        else:
            converted[key] = value
    return converted


def build_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for function in functions:
        parameters = convert_schema(function.get("parameters") or {"type": "object"})
        if parameters.get("type") != "object":
            parameters = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return tools


def flatten_question(question: Any) -> str:
    """BFCL nests turns as a list of lists of messages; static items have one turn."""
    parts = []
    turns = question if isinstance(question, list) else []
    for turn in turns:
        messages = turn if isinstance(turn, list) else [turn]
        for message in messages:
            if isinstance(message, dict) and message.get("content"):
                parts.append(str(message["content"]).strip())
    if not parts:
        raise AdapterError("question has no content")
    return "\n\n".join(parts)


def materialize(
    by_category: dict[str, list[dict[str, Any]]],
    answers: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for category, rows in by_category.items():
        for row in rows:
            item_id = str(row["id"])
            functions = row.get("function") or []
            if not functions:
                raise AdapterError(f"{item_id}: no tool schemas")
            ground_truth = None
            if category in SCORED_WITH_GROUND_TRUTH:
                answer = answers.get(item_id)
                if answer is None:
                    raise AdapterError(f"{item_id}: no ground truth for a scored category")
                ground_truth = answer["ground_truth"]
            prompts.append(
                {
                    "id": item_id,
                    "suite": SUITE,
                    "text": flatten_question(row["question"]),
                    "category": category,
                }
            )
            key[item_id] = {
                "category": category,
                "tools": build_tools(functions),
                "ground_truth": ground_truth,
                "parameter_names": {
                    function["name"]: sorted(
                        (function.get("parameters") or {}).get("properties", {})
                    )
                    for function in functions
                },
            }
    ids = [prompt["id"] for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise AdapterError("duplicate ids across categories")
    return prompts, key


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    pins = load_pins()
    validate_pins(pins)
    run_dir = env_path("EVAL_RUN_DIR")
    prompts_path = env_path("EVAL_PROMPTS_JSONL")

    selected = [name.strip() for name in args.categories.split(",") if name.strip()]
    unknown = [name for name in selected if name not in CATEGORIES]
    if unknown:
        raise AdapterError(f"unknown categories {unknown}; known: {sorted(CATEGORIES)}")

    by_category, answers = {}, {}
    for category in selected:
        by_category[category] = download_json_lines(CATEGORIES[category], pins["dataset"])
        if category in SCORED_WITH_GROUND_TRUTH:
            for row in download_json_lines(
                f"possible_answer/{CATEGORIES[category]}", pins["dataset"]
            ):
                answers[str(row["id"])] = row

    prompts, key = materialize(by_category, answers)
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {
            "suite": SUITE,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": pins["dataset"],
            "dataset_files": {name: CATEGORIES[name] for name in selected},
            "dataset_note": "BFCL v3 static split; no v4 dataset is published on the Hub",
            "verifier": VERIFIER_ID,
            "adapter": self_pin(),
            "items": key,
        },
    )
    print(f"materialized {len(prompts)} {SUITE} items to {prompts_path}", flush=True)
    return 0


def normalize_value(value: Any) -> Any:
    """Compare 10 with '10', and 'New York' with 'new york'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            return PUNCTUATION_RE.sub("", text.casefold())
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in sorted(value.items())}
    return value


def value_matches(produced: Any, acceptable: Any) -> bool:
    options = acceptable if isinstance(acceptable, list) else [acceptable]
    normalized = normalize_value(produced)
    for option in options:
        if normalize_value(option) == normalized:
            return True
        # A list-valued parameter may be given as a single acceptable list.
        if isinstance(option, list) and normalize_value([produced]) == normalize_value(option):
            return True
    return False


def call_matches(call: dict[str, Any], expected: dict[str, Any], allowed: list[str]) -> bool:
    (name, parameters), = expected.items()
    if call["name"] != name:
        return False
    arguments = call["arguments"]
    if not isinstance(arguments, dict):
        return False
    for argument in arguments:
        if allowed and argument not in allowed:
            return False  # hallucinated parameter
    for parameter, acceptable in parameters.items():
        options = acceptable if isinstance(acceptable, list) else [acceptable]
        optional = any(option == "" for option in options if isinstance(option, str))
        if parameter not in arguments:
            if optional:
                continue
            return False
        if not value_matches(arguments[parameter], options):
            return False
    return True


def score_calls(calls: list[dict[str, Any]], entry: dict[str, Any]) -> float:
    """AST match: every expected call must be produced, and nothing extra."""
    if entry["category"] == "irrelevance":
        # The correct behavior is to call nothing at all.
        return 0.0 if calls else 1.0
    ground_truth = entry["ground_truth"] or []
    if len(calls) != len(ground_truth):
        return 0.0
    remaining = list(calls)
    for expected in ground_truth:
        name = next(iter(expected))
        allowed = entry.get("parameter_names", {}).get(name, [])
        for index, call in enumerate(remaining):
            if call_matches(call, expected, allowed):
                remaining.pop(index)
                break
        else:
            return 0.0
    return 1.0


def extract_calls(message: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Return (calls, malformed). Malformed means a call whose JSON did not parse."""
    raw_calls = message.get("tool_calls") or []
    calls, malformed = [], False
    for raw in raw_calls:
        function = (raw or {}).get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        if not name:
            malformed = True
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                malformed = True
                continue
        if not isinstance(arguments, dict):
            malformed = True
            continue
        calls.append({"name": str(name), "arguments": arguments})
    return calls, malformed


def score_response(
    item_id: str,
    response: dict[str, Any],
    *,
    entry: dict[str, Any],
    replicate: int,
    thinking: bool,
) -> dict[str, Any]:
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise AdapterError(f"{item_id}: malformed chat completion: {error}") from error

    content = message.get("content") if isinstance(message.get("content"), str) else ""
    raw_reasoning = (
        message.get("reasoning_content")
        if isinstance(message.get("reasoning_content"), str)
        else ""
    )
    reasoning, answer = split_reasoning(content, raw_reasoning)
    finish_reason = str(choice.get("finish_reason") or "")
    usage = response.get("usage") or {}
    calls, malformed = extract_calls(message)
    thought = reasoning_tokens(usage, reasoning, answer)
    score = score_calls(calls, entry)

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            "score": score,
            # For irrelevance, saying nothing and calling nothing is correct.
            "empty_answer": not calls and not answer.strip() and entry["category"] != "irrelevance",
            "repetition_loop": has_repetition_loop(answer or reasoning),
            "malformed_tool_call": malformed,
            "premature_final_answer": bool(thinking and calls and thought == 0),
            "context_failure": finish_reason == "length",
            "category": entry["category"],
            "calls": calls,
            "expected_calls": len(entry["ground_truth"] or []),
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
    payload["tools"] = entry["tools"]
    payload["tool_choice"] = "auto"
    started = time.monotonic()
    response, attempts = request_with_retries(
        item_id, payload, base_url=base_url, api_key=api_key,
        timeout=args.request_timeout, retries=args.retries, client=client,
    )
    if response is None:
        row = _timeout_row(SUITE, item_id, replicate)
        row["category"] = entry["category"]
    else:
        row = score_response(
            item_id, response, entry=entry, replicate=replicate,
            thinking=bool(generation["enable_thinking"]),
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
    stored = json.loads(key_path(run_dir).read_text(encoding="utf-8"))
    key = stored["items"]
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

    by_category: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["score"])
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
            "dataset_revision": stored.get("dataset_revision"),
            "dataset_files": stored.get("dataset_files"),
            "dataset_note": stored.get("dataset_note"),
            "generation": generation,
            "generation_overrides": {},
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "accuracy": round(sum(row["score"] for row in rows) / len(rows), 6),
            "accuracy_by_category": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_category.items())
            },
            "malformed_tool_calls": sum(1 for row in rows if row["malformed_tool_call"]),
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
                "dataset": dataset or "REPLACE_WITH_BFCL_DATA_REVISION",
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
