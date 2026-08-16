#!/usr/bin/env python3
"""Humanity's Last Exam adapter for scripts/run_eval_protocol.py.

HLE is deliberately unsaturated: frontier models score low, which is the
opposite of everything else in this protocol, where 83% of items score
identically on both checkpoints. Items near the decision boundary are where a
paired comparison has power, and this is where they are.

Scoring is an exact match on a pinned final line, not the benchmark's
`model_graded_fact` scorer with `openai/o3-mini`. The dataset supports that: it
types every answer itself, 1,909 as exactMatch and 591 as multipleChoice, and
the median answer is four characters. A judge would put a proprietary model and
its revision inside a quality gate, which EVAL.md rules out, and any systematic
strictness applies to both checkpoints and cancels in the paired delta.

The consequence is that our absolute number is not comparable to a published
HLE score and must not be submitted to that leaderboard. Judged grading accepts
equivalent phrasings that this rejects. The delta is what this suite is for.

Images arrive as data URLs and are passed through unchanged rather than decoded
and re-encoded, so both checkpoints are sent the same bytes by construction.
"""

import argparse
import hashlib
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
        timing,
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
        timing,
        unpack_choice,
        write_json,
        write_jsonl,
    )


SUITE = "hle"
HARNESS_ID = "builtin-hle-exact-v1"
VERIFIER_ID = "exact-answer-v1"
DATASET_REPO = "cais/hle"
DATASET_FILE = "data/test-00000-of-00001.parquet"

DEFAULT_MAX_TOKENS = 65536

ANSWER_INSTRUCTION = (
    "Answer the question. End your reply with a final line of exactly this "
    "form:\n\nAnswer: <answer>\n\n"
    "giving only the answer itself, with no explanation. For a multiple-choice "
    "question give only the option letter."
)

ANSWER_LINE_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")
TRAILING_RE = re.compile(r"[.,;:]+$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("prepare", help="materialize questions, images and the answer key")

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
            "pins.dataset must be the 40-character HLE dataset commit; "
            f"got {dataset!r}. A branch or tag is not an immutable pin."
        )
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def load_split(revision: str) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("huggingface_hub and pyarrow are required for prepare") from error
    path = hf_hub_download(
        DATASET_REPO, DATASET_FILE, repo_type="dataset", revision=revision
    )
    return pq.read_table(path).to_pylist()


def normalize(value: str) -> str:
    text = WHITESPACE_RE.sub(" ", str(value)).strip().casefold()
    return TRAILING_RE.sub("", text)


def image_dir(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}-images"


def materialize(
    rows: list[dict[str, Any]], run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for row in rows:
        item_id = str(row.get("id") or "").strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not item_id or not question or not answer:
            raise AdapterError(f"row {row.get('id')} is incomplete")
        if item_id in key:
            raise AdapterError(f"duplicate item id {item_id}")

        image = row.get("image")
        image_ref = None
        if image:
            data_url = str(image)
            if not data_url.startswith("data:"):
                raise AdapterError(f"{item_id}: image is not a data URL")
            # Written out so the frozen set is inspectable and its bytes fixed,
            # but sent from this stored copy verbatim rather than re-encoded.
            path = image_dir(run_dir) / f"{item_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data_url, encoding="utf-8")
            image_ref = {
                "path": str(path.relative_to(run_dir)),
                "sha256": hashlib.sha256(data_url.encode()).hexdigest(),
            }

        prompts.append(
            {
                "id": item_id,
                "suite": SUITE,
                "text": question,
                "category": str(row.get("category") or "unknown"),
            }
        )
        key[item_id] = {
            "answer": answer,
            "normalized": normalize(answer),
            "answer_type": str(row.get("answer_type") or "unknown"),
            "category": str(row.get("category") or "unknown"),
            "raw_subject": str(row.get("raw_subject") or "unknown"),
            "image": image_ref,
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
    prompts, key = materialize(load_split(pins["dataset"]), run_dir)
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {"dataset": f"{DATASET_REPO}@{pins['dataset']}", "items": key},
    )
    print(f"materialized {len(prompts)} {SUITE} items to {prompts_path}", flush=True)
    return 0


def read_image(run_dir: Path, image: dict[str, Any]) -> str:
    path = run_dir / image["path"]
    data_url = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(data_url.encode()).hexdigest()
    if digest != image["sha256"]:
        raise AdapterError(
            f"{path}: sha256 {digest} does not match the materialized {image['sha256']}"
        )
    return data_url


def extract_answer(text: str) -> str | None:
    matches = ANSWER_LINE_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


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
    correct = bool(predicted is not None and normalize(predicted) == entry["normalized"])

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            "score": 1.0 if correct else 0.0,
            "empty_answer": predicted is None,
            "repetition_loop": has_repetition_loop(answer_text or reasoning),
            # vLLM discards an unterminated think block, so a reply that ran to
            # the cap arrives with no text at all -- exactly where a loop is most
            # likely. Record whether there was anything to inspect, so a False
            # here is not read as "checked, and clean".
            "repetition_assessed": bool(answer_text or reasoning),
            "malformed_tool_call": False,
            "premature_final_answer": bool(thinking and predicted is not None and thought == 0),
            "context_failure": finish_reason == "length",
            "predicted": (predicted or "")[:200],
            "expected": entry["answer"][:200],
            "category": entry["category"],
            # Kept apart because exact-match grading is strict on free-form
            # answers and lenient on letters, and the two should be readable
            # separately.
            "answer_type": entry["answer_type"],
            "has_image": entry["image"] is not None,
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
    if entry["image"]:
        payload["messages"][0]["content"] = [
            {"type": "image_url",
             "image_url": {"url": read_image(run_dir, entry["image"])}},
            {"type": "text", "text": f"{text}\n\n{ANSWER_INSTRUCTION}"},
        ]

    started_wall, started = time.time(), time.monotonic()
    response, attempts = request_with_retries(
        item_id, payload, base_url=base_url, api_key=api_key,
        timeout=args.request_timeout, retries=args.retries, client=client,
    )
    if response is None:
        row = _timeout_row(SUITE, item_id, replicate)
        row.update(
            {
                "category": entry["category"],
                "answer_type": entry["answer_type"],
                "has_image": entry["image"] is not None,
            }
        )
    else:
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
    by_type: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["score"])
        by_type.setdefault(row["answer_type"], []).append(row["score"])
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
            "score_by_answer_type": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_type.items())
            },
            # Not the published protocol: HLE grades with an LLM judge, this
            # matches exactly. Absolute numbers are not comparable to published
            # HLE scores.
            "grading": "exact-match, not model_graded_fact",
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
                "dataset": dataset or "REPLACE_WITH_HLE_REVISION",
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
