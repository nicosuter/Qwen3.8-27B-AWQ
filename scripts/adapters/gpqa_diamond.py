#!/usr/bin/env python3
"""Reference GPQA Diamond adapter for scripts/run_eval_protocol.py.

`prepare` materializes one prompt row per question with a deterministic answer
order and writes the answer key beside it. `run` replays the frozen task order
against the served checkpoint and emits the paired result schema from EVAL.md.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from _common import (
        AdapterError,
        base_row,
        build_payload,
        check_action,
        describe_http_error,
        env_path,
        env_str,
        execute_order,
        has_repetition_loop,
        load_pins,
        message_text,
        module_pin,
        post_chat,
        raw_response_path as _raw_response_path,
        read_jsonl,
        split_reasoning,
        reasoning_tokens,
        request_with_retries,
        require_pin,
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
        describe_http_error,
        env_path,
        env_str,
        execute_order,
        has_repetition_loop,
        load_pins,
        message_text,
        module_pin,
        post_chat,
        raw_response_path as _raw_response_path,
        read_jsonl,
        split_reasoning,
        reasoning_tokens,
        request_with_retries,
        require_pin,
        timeout_row as _timeout_row,
        timing,
        unpack_choice,
        write_json,
        write_jsonl,
    )


SUITE = "gpqa_diamond"
HARNESS_ID = "builtin-gpqa-mcq-v1"
VERIFIER_ID = "exact-choice-v1"
DATASET_REPO = "Idavidrein/gpqa"
SPLITS = ("gpqa_diamond", "gpqa_main", "gpqa_extended", "gpqa_experts")
CHOICES = ("A", "B", "C", "D")

# Kept out of the materialized prompt: the overlap audit must see the question
# and its options, not shared answer-format boilerplate.
ANSWER_INSTRUCTION = (
    "Answer the multiple-choice question above. End your reply with a final "
    "line of exactly this form:\n\nAnswer: <letter>\n\n"
    "where <letter> is one of A, B, C, or D."
)

# xhigh thinking on graduate-level science routinely runs long; a tight cap
# turns a correct-but-truncated reply into a scored zero plus context_failure.
DEFAULT_MAX_TOKENS = 65536

# Only used by `probe`; the scored run always takes EVAL_GENERATION_JSON.
MODEL_CARD_GENERATION = {
    "enable_thinking": True,
    "reasoning_effort": "xhigh",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

# Deliberately not a fact the model can answer by recall: an easy question can
# legitimately produce an empty think block, which is indistinguishable from
# thinking never being switched on.
PROBE_QUESTION = (
    "A 0.40 kg block slides down a frictionless incline of height 1.8 m, then "
    "crosses a rough patch with coefficient of kinetic friction 0.25 before "
    "compressing a spring of constant 320 N/m by 12 cm at maximum compression. "
    "How long is the rough patch?\n\n"
    "A) 1.06 m\nB) 2.42 m\nC) 3.18 m\nD) 4.75 m"
)

# Below this, the reply cannot have contained real deliberation.
MIN_PROBE_REASONING_TOKENS = 50

FINAL_ANSWER_RE = re.compile(
    r"answer\b[^0-9A-Za-z]{0,12}(?:is[^0-9A-Za-z]{0,6})?([A-D])(?![0-9A-Za-z])",
    re.IGNORECASE,
)
LONE_LETTER_RE = re.compile(r"^[^0-9A-Za-z]*([A-D])[^0-9A-Za-z]*$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare", help="materialize prompts and the answer key")
    prepare.add_argument("--split", choices=SPLITS, default=SUITE)

    run = sub.add_parser("run", help="score the frozen task order against the server")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="output cap; truncation scores 0 and sets context_failure",
    )
    run.add_argument("--request-timeout", type=float, default=1800.0)
    run.add_argument(
        "--retries",
        type=int,
        default=2,
        help="retries for transport faults; timeouts and 4xx are never retried",
    )

    probe = sub.add_parser(
        "probe", help="check that the server accepts the generation policy fields"
    )
    probe.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    probe.add_argument(
        "--model", default=os.environ.get("EVAL_SERVED_MODEL", "openai/qwen38-eval")
    )
    probe.add_argument(
        "--generation",
        type=Path,
        help="JSON generation policy; defaults to the model-card policy",
    )
    # An unterminated think block is returned as nothing at all: vLLM drops
    # reasoning that never reaches </think>. The probe cap must be generous
    # enough for xhigh to finish, or the probe measures its own cap.
    probe.add_argument("--max-tokens", type=int, default=16384)
    probe.add_argument("--request-timeout", type=float, default=300.0)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--dataset", help="the 40-character GPQA dataset commit")
    pin.add_argument(
        "--resolve-dataset",
        action="store_true",
        help="look the dataset commit up on the Hub (needs accepted terms and a token)",
    )
    return parser.parse_args(argv)


def self_pin() -> str:
    return module_pin([Path(__file__), Path(__file__).resolve().parent / "_common.py"])


def raw_response_path(run_dir: Path, variant: str, replicate: int, item_id: str) -> Path:
    return _raw_response_path(run_dir, SUITE, variant, replicate, item_id)


def timeout_row(item_id: str, expected: str, replicate: int) -> dict[str, Any]:
    row = _timeout_row(SUITE, item_id, replicate)
    row.update({"predicted": None, "expected": expected, "reasoning_tokens": 0})
    return row


def validate_pins(pins: dict[str, str]) -> None:
    dataset = pins.get("dataset", "")
    if not re.fullmatch(r"[0-9a-f]{40}", dataset):
        raise AdapterError(
            "pins.dataset must be the 40-character GPQA dataset commit; "
            f"got {dataset!r}. A branch or tag is not an immutable pin."
        )
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def load_gpqa(split: str, revision: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("the datasets package is required for prepare") from error
    dataset = load_dataset(DATASET_REPO, split, split="train", revision=revision)
    return [dict(row) for row in dataset]


def extract_examples(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for index, row in enumerate(rows):
        question = normalize_text(row.get("Question", ""))
        correct = normalize_text(row.get("Correct Answer", ""))
        distractors = [
            normalize_text(row.get(f"Incorrect Answer {number}", ""))
            for number in (1, 2, 3)
        ]
        if not question or not correct or not all(distractors):
            raise AdapterError(f"row {index}: incomplete GPQA record")
        record_id = str(row.get("Record ID") or "").strip()
        if not record_id:
            digest = hashlib.sha256(question.encode()).hexdigest()[:16]
            record_id = f"gpqa-{digest}"
        examples.append(
            {
                "id": record_id,
                "question": question,
                "correct": correct,
                "distractors": distractors,
                "category": normalize_text(row.get("High-level domain", "unknown")),
                "subdomain": normalize_text(row.get("Subdomain", "unknown")),
            }
        )
    ids = [example["id"] for example in examples]
    if len(ids) != len(set(ids)):
        raise AdapterError("GPQA records contain duplicate Record IDs")
    return examples


def build_prompt_text(question: str, options: list[str]) -> str:
    lines = [question, ""]
    lines.extend(f"{letter}) {option}" for letter, option in zip(CHOICES, options))
    return "\n".join(lines)


def materialize(
    examples: list[dict[str, Any]], order_seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = []
    key = {}
    for example in examples:
        options = [example["correct"], *example["distractors"]]
        # Seeded per item so the option order is stable across prepare runs and
        # identical for both checkpoints.
        random.Random(f"{order_seed}:{example['id']}").shuffle(options)
        answer = CHOICES[options.index(example["correct"])]
        prompts.append(
            {
                "id": example["id"],
                "suite": SUITE,
                "text": build_prompt_text(example["question"], options),
                "category": example["category"],
                "subdomain": example["subdomain"],
            }
        )
        key[example["id"]] = {
            "answer": answer,
            "options": options,
            "category": example["category"],
            "subdomain": example["subdomain"],
        }
    return prompts, key


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    pins = load_pins()
    validate_pins(pins)
    run_dir = env_path("EVAL_RUN_DIR")
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    order_seed = int(env_str("EVAL_ORDER_SEED"))

    examples = extract_examples(load_gpqa(args.split, pins["dataset"]))
    prompts, key = materialize(examples, order_seed)
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {
            "suite": SUITE,
            "split": args.split,
            "dataset_revision": pins["dataset"],
            "order_seed": order_seed,
            "verifier": VERIFIER_ID,
            "adapter": self_pin(),
            "items": key,
        },
    )
    print(f"materialized {len(prompts)} {SUITE} prompts to {prompts_path}", flush=True)
    return 0


def extract_answer(content: str) -> str | None:
    matches = FINAL_ANSWER_RE.findall(content)
    if matches:
        return matches[-1].upper()
    for line in reversed([line for line in content.splitlines() if line.strip()]):
        match = LONE_LETTER_RE.match(line.strip())
        if match:
            return match.group(1).upper()
    return None


def score_response(
    item_id: str,
    response: dict[str, Any],
    *,
    expected: str,
    replicate: int,
    thinking: bool,
) -> dict[str, Any]:
    content, raw_reasoning, finish_reason, usage = unpack_choice(item_id, response)
    reasoning, answer = split_reasoning(content, raw_reasoning)
    predicted = extract_answer(answer)
    thought = reasoning_tokens(usage, reasoning, answer)

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            "score": 1.0 if predicted == expected else 0.0,
            "empty_answer": predicted is None,
            "repetition_loop": has_repetition_loop(answer or reasoning),
            # vLLM discards an unterminated think block, so a reply that ran to
            # the cap arrives with no text at all -- exactly where a loop is most
            # likely. Record whether there was anything to inspect, so a False
            # here is not read as "checked, and clean".
            "repetition_assessed": bool(answer or reasoning),
            # GPQA is served without tools, so a malformed call cannot occur.
            "malformed_tool_call": False,
            # Thinking was requested but the server returned no reasoning at all.
            "premature_final_answer": bool(thinking and predicted is not None and thought == 0),
            "context_failure": finish_reason == "length",
            "predicted": predicted,
            "expected": expected,
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
    expected: str,
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
        text,
        generation,
        model=model,
        seed=seed,
        max_tokens=args.max_tokens,
        instruction=ANSWER_INSTRUCTION,
    )
    started_wall, started = time.time(), time.monotonic()
    response, attempts = request_with_retries(
        item_id,
        payload,
        base_url=base_url,
        api_key=api_key,
        timeout=args.request_timeout,
        retries=args.retries,
        client=client,
    )
    if response is None:
        row = timeout_row(item_id, expected, replicate)
    else:
        row = score_response(
            item_id, response, expected=expected, replicate=replicate,
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
            item_id,
            prompts[item_id],
            key[item_id]["answer"],
            generation=generation,
            model=model,
            seed=seed,
            replicate=replicate,
            variant=variant,
            run_dir=run_dir,
            base_url=base_url,
            api_key=api_key,
            args=args,
            client=client,
        ),
        args.concurrency,
    )

    for row in rows:
        row["category"] = key[row["id"]]["category"]
        row["subdomain"] = key[row["id"]]["subdomain"]
    write_jsonl(results_path, rows)

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
            "request_timeout_seconds": args.request_timeout,
            "generation": generation,
            "generation_overrides": {},
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "accuracy": round(sum(row["score"] for row in rows) / len(rows), 6),
        },
    )
    print(f"scored {len(rows)} {SUITE} items to {results_path}", flush=True)
    return 0


def command_probe(
    args: argparse.Namespace, client: Callable[..., dict[str, Any]] = post_chat
) -> int:
    """One live request that proves the server honors the generation policy.

    A 4xx here means vLLM rejected a policy field (commonly `reasoning_effort`
    or `chat_template_kwargs`), which would otherwise abort the scored run.
    """
    if not args.base_url:
        raise AdapterError("--base-url is required (or set OPENAI_BASE_URL)")
    generation = (
        json.loads(args.generation.read_text(encoding="utf-8"))
        if args.generation
        else dict(MODEL_CARD_GENERATION)
    )
    payload = build_payload(
        PROBE_QUESTION,
        generation,
        model=args.model,
        seed=0,
        max_tokens=args.max_tokens,
        instruction=ANSWER_INSTRUCTION,
    )
    try:
        response = client(
            args.base_url,
            os.environ.get("OPENAI_API_KEY", "EMPTY"),
            payload,
            args.request_timeout,
        )
    except urllib.error.HTTPError as error:
        raise AdapterError(
            f"server rejected the generation policy, {describe_http_error(error)}"
        ) from error

    content, raw_reasoning, finish_reason, usage = unpack_choice("probe", response)
    reasoning, answer_text = split_reasoning(content, raw_reasoning)
    thought = reasoning_tokens(usage, reasoning, answer_text)
    answer = extract_answer(answer_text)
    separated = bool(raw_reasoning)
    report = {
        "sent_chat_template_kwargs": payload["chat_template_kwargs"],
        "finish_reason": finish_reason,
        "reasoning_returned": bool(reasoning),
        "reasoning_separated_by_server": separated,
        "completion_tokens": usage.get("completion_tokens"),
        "visible_answer_chars": len(answer_text),
        "reasoning_tokens": thought,
        "parsed_answer": answer,
        "content_preview": content[:200],
        "reasoning_preview": reasoning[:200],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if answer is None:
        raise AdapterError("the reply had no parsable answer line; check the cap and template")
    # Thinking is confirmed either by returned reasoning or by generated tokens
    # that never appear in the answer. This build gives only the latter.
    if generation.get("enable_thinking") and thought < MIN_PROBE_REASONING_TOKENS:
        raise AdapterError(
            f"thinking was requested but only {thought} reasoning tokens are "
            "accounted for; chat_template_kwargs is not reaching the template"
        )
    if not separated:
        print(
            "note: this server strips the think block without returning "
            "reasoning_content; reasoning_tokens is inferred from the token gap",
            file=sys.stderr,
        )
    if reasoning and not separated:
        # Recoverable: split_reasoning salvages it, but the server is not doing
        # the job --reasoning-parser was meant to do, so say so loudly.
        print(
            "warning: reasoning arrived inside content, not reasoning_content; "
            "the reasoning parser is not splitting this model's output",
            file=sys.stderr,
        )
    return 0


def resolve_dataset_commit() -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise AdapterError("huggingface_hub is required for --resolve-dataset") from error
    try:
        return str(HfApi().dataset_info(DATASET_REPO).sha)
    except Exception as error:  # noqa: BLE001 - gated access failures are the common case
        raise AdapterError(
            f"could not resolve {DATASET_REPO}: {error}. "
            "GPQA is gated; accept its terms and log in first."
        ) from error


def command_pin(args: argparse.Namespace) -> int:
    dataset = args.dataset
    if args.resolve_dataset:
        dataset = resolve_dataset_commit()
    print(
        json.dumps(
            {
                "dataset": dataset or "REPLACE_WITH_GPQA_DATASET_COMMIT",
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
    if args.action == "probe":
        return command_probe(args)
    return command_run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
