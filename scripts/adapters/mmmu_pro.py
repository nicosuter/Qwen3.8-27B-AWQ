#!/usr/bin/env python3
"""MMMU-Pro adapter for scripts/run_eval_protocol.py.

Our multimodal suite is DocVQA, ChartQA and TextVQA: perception and OCR, where
the model already sits near ceiling and 84% of items score identically on both
checkpoints. That measures the vision tower, which this recipe leaves in source
precision, more than it measures the decoder, which this recipe quantizes.

MMMU-Pro asks college-level reasoning about images across thirty subjects, with
ten options rather than four. The perception is the easy part; the reasoning
afterwards runs entirely through the quantized language path. It is the only
suite here that stresses that path through an image.

The standard ten-option config is used rather than the vision config, where the
question itself is rendered into the image: that measures OCR of the prompt,
which is a different question and one the unquantized tower would answer.

Images are written to the run directory as PNG with their sha256 recorded, and
referenced by a path relative to it, so both checkpoints are sent byte-identical
pixels and a materialized set stays usable when it is copied elsewhere.
"""

import argparse
import ast
import base64
import hashlib
import io
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


SUITE = "mmmu_pro"
HARNESS_ID = "builtin-mmmu-pro-mcq-v1"
VERIFIER_ID = "exact-choice-v1"
DATASET_REPO = "MMMU/MMMU_Pro"
DATASET_CONFIG = "standard (10 options)"
IMAGE_FIELDS = tuple(f"image_{index}" for index in range(1, 8))
# The config is named for ten options and does not hold to it. Across the 1730
# test items the counts run 2, 3, 4, 5, 6, 7, 8, 9, 10 and 12, with 1213 at ten
# and exactly one at twelve (test_Computer_Science_61, answer F). Capping the
# alphabet at ten refused that item rather than mislabelling it, which was the
# right instinct and the wrong bound. The whole alphabet costs nothing and the
# guard below still refuses anything past it.
LETTERS = string.ascii_uppercase

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
            "pins.dataset must be the 40-character MMMU-Pro dataset commit; "
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
    rows = load_dataset(DATASET_REPO, DATASET_CONFIG, split="test", revision=revision)
    return [dict(row) for row in rows]


def parse_options(raw: Any) -> list[str]:
    """`options` arrives as the repr of a Python list, not as a list."""
    if isinstance(raw, (list, tuple)):
        return [str(option) for option in raw]
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError) as error:
        raise AdapterError(f"cannot parse options {raw!r}") from error
    if not isinstance(parsed, (list, tuple)):
        raise AdapterError(f"options did not parse to a list: {raw!r}")
    return [str(option) for option in parsed]


def image_dir(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}-images"


def save_image(image: Any, path: Path) -> str:
    """Write PNG bytes and return their sha256 so a rerun can prove they match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    converted = image.convert("RGB") if getattr(image, "mode", "RGB") not in ("RGB",) else image
    converted.save(buffer, format="PNG")
    payload = buffer.getvalue()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def resolve_image(run_dir: Path, stored: str) -> Path:
    """Locate an image recorded in the key file.

    Paths are stored relative to the run directory so a materialized set stays
    usable when it is copied to another run or cluster.
    """
    candidate = Path(stored)
    if not candidate.is_absolute():
        return run_dir / candidate
    local = image_dir(run_dir) / candidate.name
    return local if local.is_file() else candidate


def image_data_url(path: Path, expected_sha256: str) -> str:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise AdapterError(
            f"{path}: sha256 {digest} does not match the materialized {expected_sha256}"
        )
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def build_prompt(question: str, options: list[str]) -> str:
    lines = [question.strip(), ""]
    for letter, option in zip(LETTERS, options):
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def materialize(
    rows: list[dict[str, Any]], run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for row in rows:
        item_id = str(row.get("id") or "").strip()
        question = str(row.get("question", "")).strip()
        options = parse_options(row.get("options"))
        answer = str(row.get("answer", "")).strip().upper()
        if not item_id or not question or not options:
            raise AdapterError(f"row {row.get('id')} is incomplete")
        if len(options) > len(LETTERS):
            raise AdapterError(
                f"{item_id} has {len(options)} options; only {len(LETTERS)} letters exist"
            )
        if answer not in LETTERS[: len(options)]:
            raise AdapterError(
                f"{item_id} has answer {answer!r}, not one of its {len(options)} options"
            )
        if item_id in key:
            raise AdapterError(f"duplicate item id {item_id}")

        images = []
        for index, field in enumerate(IMAGE_FIELDS, start=1):
            image = row.get(field)
            if image is None:
                continue
            path = image_dir(run_dir) / f"{item_id}_{index}.png"
            digest = save_image(image, path)
            images.append(
                {
                    "index": index,
                    "path": str(path.relative_to(run_dir)),
                    "sha256": digest,
                }
            )
        if not images:
            raise AdapterError(f"{item_id} has no images; MMMU-Pro items are multimodal")

        prompts.append(
            {
                "id": item_id,
                "suite": SUITE,
                "text": build_prompt(question, options),
                "category": str(row.get("subject") or "unknown"),
            }
        )
        key[item_id] = {
            "answer": answer,
            "options": len(options),
            "category": str(row.get("subject") or "unknown"),
            "difficulty": str(row.get("topic_difficulty") or "unknown"),
            "images": images,
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
        {
            "dataset": f"{DATASET_REPO}@{pins['dataset']} [{DATASET_CONFIG}]",
            "items": key,
        },
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
            "difficulty": entry["difficulty"],
            "images": len(entry["images"]),
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
    # Images first and in their dataset order, so the <image N> placeholders in
    # the question refer to what the model was actually shown.
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url(
                    resolve_image(run_dir, image["path"]), image["sha256"]
                )
            },
        }
        for image in sorted(entry["images"], key=lambda image: image["index"])
    ]
    content.append({"type": "text", "text": f"{text}\n\n{ANSWER_INSTRUCTION}"})
    payload["messages"][0]["content"] = content

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
                "difficulty": entry["difficulty"],
                "images": len(entry["images"]),
                "out_of_range_choice": False,
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
            "score_by_subject": {
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
                "dataset": dataset or "REPLACE_WITH_MMMU_PRO_REVISION",
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
