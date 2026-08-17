#!/usr/bin/env python3
"""AA-Omniscience adapter for eval/scripts/run_eval_protocol.py.

Every other suite here asks the model to reason. This one asks what it knows,
which is the thing 4-bit weights are most directly able to damage: a fact that
survives in bf16 and not in int4 is a fact stored in the weights that were
quantized.

It also scores abstention separately from error. The published Omniscience Index
rewards a correct answer, penalizes a confident wrong one, and treats "I don't
know" as neutral, so a model can improve it by knowing less and admitting more.
Quantization plausibly shifts calibration before it shifts accuracy, and nothing
else in this protocol would see that.

The 600-question public split is used in full. The held-out remainder is not
published, and the paired comparison does not need it: both checkpoints answer
the same frozen items.

Answers are short and factual, so scoring is an exact match on a pinned final
line rather than an LLM judge. That diverges from the published methodology,
which judges equivalence with a model. It is the right trade here: a judge adds
a revision to pin and a source of nondeterminism to a gate, and any systematic
strictness applies to both checkpoints and cancels in the paired delta.
"""

import argparse
import csv
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
        timeout_row as _timeout_row,
        timing,
        unpack_choice,
        write_json,
        write_jsonl,
    )


SUITE = "aa_omniscience"
HARNESS_ID = "builtin-omniscience-exact-v1"
VERIFIER_ID = "exact-answer-with-abstention-v1"
DATASET_REPO = "ArtificialAnalysis/AA-Omniscience-Public"
DATASET_FILE = "AA-Omniscience_dataset_public.csv"

DEFAULT_MAX_TOKENS = 32768
ABSTENTION = "i don't know"

# Abstention has to be offered explicitly and in a form that is unambiguous to
# match. Without the offer the model guesses, and a guess scores the same as
# knowledge it does not have.
ANSWER_INSTRUCTION = (
    "Answer the question. End your reply with a final line of exactly this "
    "form:\n\n"
    "Answer: <answer>\n\n"
    "giving only the answer itself, with no explanation. If you do not know the "
    "answer, write exactly:\n\n"
    "Answer: I don't know"
)

ANSWER_LINE_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
PUNCTUATION_RE = re.compile(r"[\s ]+")
TRAILING_RE = re.compile(r"[.,;:]+$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("prepare", help="materialize questions and answers")

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
            "pins.dataset must be the 40-character AA-Omniscience dataset commit; "
            f"got {dataset!r}. A branch or tag is not an immutable pin."
        )
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def download_dataset(revision: str) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("huggingface_hub is required for prepare") from error
    path = hf_hub_download(
        DATASET_REPO, DATASET_FILE, repo_type="dataset", revision=revision
    )
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(value: str) -> str:
    """Fold the differences that are never the point of the question."""
    text = PUNCTUATION_RE.sub(" ", str(value)).strip().casefold()
    text = TRAILING_RE.sub("", text)
    return text


def materialize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            raise AdapterError(f"row {row.get('question_id')} has no question or answer")
        item_id = f"omni-{row['question_id']}"
        if item_id in key:
            raise AdapterError(f"duplicate item id {item_id}")
        prompts.append(
            {
                "id": item_id,
                "suite": SUITE,
                "text": question,
                "category": str(row.get("domain") or "unknown"),
            }
        )
        key[item_id] = {
            "answer": answer,
            "normalized": normalize(answer),
            "domain": str(row.get("domain") or "unknown"),
            "topic": str(row.get("topic") or "unknown"),
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
    rows = download_dataset(pins["dataset"])
    prompts, key = materialize(rows)
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {"dataset": f"{DATASET_REPO}@{pins['dataset']}", "items": key},
    )
    print(f"materialized {len(prompts)} {SUITE} items to {prompts_path}", flush=True)
    return 0


def extract_answer(text: str) -> str | None:
    matches = ANSWER_LINE_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def is_abstention(predicted: str | None) -> bool:
    if predicted is None:
        return False
    folded = normalize(predicted).replace("do not", "don't")
    return folded.startswith(ABSTENTION) or folded in {"unknown", "i dont know"}


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

    abstained = is_abstention(predicted)
    correct = bool(
        predicted is not None and not abstained and normalize(predicted) == entry["normalized"]
    )

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            # Accuracy is what feeds the macro. The index that treats abstention
            # as neutral is reported separately, from the flags below.
            "score": 1.0 if correct else 0.0,
            "abstained": abstained,
            # A wrong answer given confidently, which is the behaviour the
            # published index penalizes.
            "hallucinated": bool(predicted is not None and not abstained and not correct),
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
            "category": entry["domain"],
            "topic": entry["topic"],
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
    if response is None:
        row = _timeout_row(SUITE, item_id, replicate)
        row.update(
            {
                "abstained": False,
                "hallucinated": False,
                "category": entry["domain"],
                "topic": entry["topic"],
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

    correct = sum(1 for row in rows if row["score"] == 1.0)
    abstained = sum(1 for row in rows if row.get("abstained"))
    hallucinated = sum(1 for row in rows if row.get("hallucinated"))
    by_domain: dict[str, list[float]] = {}
    for row in rows:
        by_domain.setdefault(row["category"], []).append(row["score"])
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
            "score": round(correct / len(rows), 6),
            # The published index: right answers minus confident wrong ones,
            # with abstentions neutral. Reported beside accuracy, never instead
            # of it, since the two move independently.
            "omniscience_index": round((correct - hallucinated) / len(rows), 6),
            "abstained": abstained,
            "hallucinated": hallucinated,
            "score_by_domain": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_domain.items())
            },
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
                "dataset": dataset or "REPLACE_WITH_OMNISCIENCE_REVISION",
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
