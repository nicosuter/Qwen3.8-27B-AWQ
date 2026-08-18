#!/usr/bin/env python3
"""MMLU-Pro adapter for eval/scripts/run_eval_protocol.py.

Every suite in this protocol is small. GPQA is 198 items, MathArena 77, AA-LCR
96, and the intervals that follow are +-2 to +-4 points -- wide enough that
MathArena cannot resolve its own measured effect. Resolution is the binding
constraint on everything this repository claims, and the only real fix is more
items.

MMLU-Pro is 12,032 of them, ten options rather than four, exact-match scored
against a letter. At one replicate it carries far more information than any of
our four-replicate suites, and it costs less per unit of it.

Choices are presented in the dataset's published order. GPQA's benchmark
definition shuffles them and ours follows suit there; MMLU-Pro's does not, and
the answer letter is only meaningful against the published order.
"""

import argparse
import json
import os
import re
import string
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
        timing,
        unpack_choice,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # loading by file spec puts the repo root on sys.path
    from eval.scripts.adapters._common import (  # type: ignore[no-redef]
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
        timing,
        unpack_choice,
        write_json,
        write_jsonl,
    )


SUITE = "mmlu_pro"
HARNESS_ID = "builtin-mmlu-pro-mcq-v1"
VERIFIER_ID = "exact-choice-v1"
DATASET_REPO = "TIGER-Lab/MMLU-Pro"
LETTERS = string.ascii_uppercase[:10]

DEFAULT_MAX_TOKENS = 32768

ANSWER_INSTRUCTION = (
    "Answer the multiple-choice question above. End your reply with a final "
    "line of exactly this form:\n\nAnswer: <letter>\n\n"
    "where <letter> is one of the option letters."
)

FINAL_ANSWER_RE = re.compile(r"answer\s*:\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE)
LONE_LETTER_RE = re.compile(r"^[^0-9A-Za-z]*([A-J])[^0-9A-Za-z]*$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("prepare", help="materialize prompts and the answer key")

    run = sub.add_parser("run", help="score the frozen task order against the server")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--request-timeout", type=float, default=1800.0)
    run.add_argument("--retries", type=int, default=2)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--dataset", help="the 40-character dataset commit")
    pin.add_argument("--resolve", action="store_true", help="look the commit up on the Hub")
    return parser.parse_args(argv)


def self_pin() -> str:
    return module_pin([Path(__file__), Path(__file__).resolve().parent / "_common.py"])


def raw_response_path(run_dir: Path, variant: str, replicate: int, item_id: str) -> Path:
    return _raw_response_path(run_dir, SUITE, variant, replicate, item_id)


def validate_pins(pins: dict[str, str]) -> None:
    dataset = pins.get("dataset", "")
    if not re.fullmatch(r"[0-9a-f]{40}", dataset):
        raise AdapterError(
            "pins.dataset must be the 40-character MMLU-Pro dataset commit; "
            f"got {dataset!r}. A branch or tag is not an immutable pin."
        )
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def load_split(revision: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("the datasets package is required for prepare") from error
    rows = load_dataset(DATASET_REPO, split="test", revision=revision)
    return [dict(row) for row in rows]


def build_prompt(question: str, options: list[str]) -> str:
    lines = [question.strip(), ""]
    for letter, option in zip(LETTERS, options):
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def materialize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for row in rows:
        question = str(row.get("question", "")).strip()
        options = [str(option) for option in (row.get("options") or [])]
        answer = str(row.get("answer", "")).strip().upper()
        if not question or not options:
            raise AdapterError(f"question {row.get('question_id')} is incomplete")
        if len(options) > len(LETTERS):
            raise AdapterError(
                f"question {row.get('question_id')} has {len(options)} options; "
                f"only {len(LETTERS)} letters are defined"
            )
        if answer not in LETTERS[: len(options)]:
            raise AdapterError(
                f"question {row.get('question_id')} has answer {answer!r}, which is "
                f"not one of its {len(options)} options"
            )
        # The published answer is a letter into the published order, so the two
        # have to agree or every score is silently wrong.
        index = row.get("answer_index")
        if index is not None and LETTERS[int(index)] != answer:
            raise AdapterError(
                f"question {row.get('question_id')}: answer {answer!r} disagrees with "
                f"answer_index {index}"
            )
        item_id = f"mmlupro-{row['question_id']}"
        if item_id in key:
            raise AdapterError(f"duplicate item id {item_id}")
        prompts.append(
            {
                "id": item_id,
                "suite": SUITE,
                "text": build_prompt(question, options),
                "category": str(row.get("category") or "unknown"),
            }
        )
        key[item_id] = {
            "answer": answer,
            "options": len(options),
            "category": str(row.get("category") or "unknown"),
            "src": str(row.get("src") or ""),
        }
    if not prompts:
        raise AdapterError("no items materialized")
    return prompts, key


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    pins = load_pins()
    validate_pins(pins)
    run_dir = env_path("EVAL_RUN_DIR")
    prompts, key = materialize(load_split(pins["dataset"]))
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {"dataset": f"{DATASET_REPO}@{pins['dataset']}", "items": key},
    )
    print(f"materialized {len(prompts)} {SUITE} items to {prompts_path}", flush=True)
    return 0


def extract_answer(content: str) -> str | None:
    matches = FINAL_ANSWER_RE.findall(content or "")
    if matches:
        return matches[-1].upper()
    for line in reversed([line for line in (content or "").splitlines() if line.strip()]):
        match = LONE_LETTER_RE.match(line.strip())
        if match:
            return match.group(1).upper()
    return None


def score_response(
    item_id: str,
    response: dict[str, Any],
    *,
    entry: dict[str, Any],
    replicate: int,
    thinking: bool,
) -> dict[str, Any]:
    content, raw_reasoning, finish_reason, usage = unpack_choice(item_id, response)
    reasoning, answer_text = split_reasoning(content, raw_reasoning)
    predicted = extract_answer(answer_text)
    thought = reasoning_tokens(usage, reasoning, answer_text)
    # A letter past the end of this item's options is not an answer.
    in_range = bool(predicted and predicted in LETTERS[: entry["options"]])

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            "score": 1.0 if in_range and predicted == entry["answer"] else 0.0,
            "empty_answer": predicted is None,
            "out_of_range_choice": bool(predicted) and not in_range,
            "repetition_loop": has_repetition_loop(answer_text or reasoning),
            # vLLM discards an unterminated think block, so a reply that ran to
            # the cap arrives with no text at all -- exactly where a loop is most
            # likely. Record whether there was anything to inspect, so a False
            # here is not read as "checked, and clean".
            "repetition_assessed": bool(answer_text or reasoning),
            "malformed_tool_call": False,
            "premature_final_answer": bool(thinking and predicted is not None and thought == 0),
            "context_failure": finish_reason == "length",
            "predicted": predicted,
            "expected": entry["answer"],
            "category": entry["category"],
            "finish_reason": finish_reason,
            "output_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
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
    started_wall, started = time.time(), time.monotonic()
    response, attempts = request_with_retries(
        item_id, payload, base_url=base_url, api_key=api_key,
        timeout=args.request_timeout, retries=args.retries, client=client,
    )
    row = score_response(
        item_id, response, entry=entry, replicate=replicate,
        thinking=bool(generation["enable_thinking"]),
    )
    path = raw_response_path(run_dir, variant, replicate, item_id)
    write_json(path, response)
    row["raw_response"] = str(path)
    row.update(timing(started_wall, started))
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
            "dataset": stored.get("dataset"),
            "generation": generation,
            "generation_overrides": {},
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "score": round(sum(row["score"] for row in rows) / len(rows), 6),
            "score_by_category": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_category.items())
            },
            "out_of_range_choices": sum(1 for row in rows if row.get("out_of_range_choice")),
        },
    )
    print(f"scored {len(rows)} {SUITE} items to {results_path}", flush=True)
    return 0


def command_pin(args: argparse.Namespace) -> int:
    dataset = args.dataset
    if args.resolve:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise AdapterError("huggingface_hub is required for --resolve") from error
        dataset = str(HfApi().dataset_info(DATASET_REPO).sha)
    print(
        json.dumps(
            {
                "dataset": dataset or "REPLACE_WITH_MMLU_PRO_REVISION",
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
