#!/usr/bin/env python3
"""AA-LCR (long-context reasoning) adapter for scripts/run_eval_protocol.py.

Long context is the one claim this repository makes and has never measured. The
Gated DeltaNet input projections are held at 8 bits rather than 4 because 48 of
64 layers carry their long-range signal in a recurrent state, where error
accumulates along the sequence instead of being bounded per token. RULER was
supposed to check that and could not: nearly every item scored identically on
both checkpoints and the rest hit the output cap.

This is a different shape of test. Each question comes with two to twenty-four
real documents totalling 72k to 115k tokens, and the answer requires finding and
combining facts across them rather than retrieving a planted needle. That is
work the recurrent path has to do over the whole sequence.

`prepare` assembles each question's documents into the exact prompt text and
freezes it, so both checkpoints are sent byte-identical input and a rerun can
prove it.

Answers are short, so scoring is an exact match on a pinned final line rather
than the published LLM judge. A judge would add a revision to pin and
nondeterminism to a gate; systematic strictness applies to both checkpoints and
cancels in the paired delta.

Eight documents referenced by the dataset are absent from its own archive,
affecting four questions. Those cannot be built as published, so `prepare`
refuses unless --skip-incomplete says otherwise, and records exactly which were
dropped.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import zipfile
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


SUITE = "aa_lcr"
HARNESS_ID = "builtin-lcr-exact-v1"
VERIFIER_ID = "exact-answer-v1"
DATASET_REPO = "ArtificialAnalysis/AA-LCR"
DATASET_FILE = "AA-LCR_Dataset.csv"
ARCHIVE_FILE = "extracted_text/AA-LCR_extracted-text.zip"

DEFAULT_MAX_TOKENS = 32768

ANSWER_INSTRUCTION = (
    "Answer the question using only the documents above. End your reply with a "
    "final line of exactly this form:\n\n"
    "Answer: <answer>\n\n"
    "giving only the answer itself, with no explanation."
)

ANSWER_LINE_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")
TRAILING_RE = re.compile(r"[.,;:]+$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare", help="assemble documents and freeze prompts")
    prepare.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="drop questions whose documents are missing from the archive, "
             "recording which; without this a missing document is an error",
    )

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
            "pins.dataset must be the 40-character AA-LCR dataset commit; "
            f"got {dataset!r}. A branch or tag is not an immutable pin."
        )
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def download(revision: str) -> tuple[list[dict[str, Any]], zipfile.ZipFile]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("huggingface_hub is required for prepare") from error
    csv_path = hf_hub_download(
        DATASET_REPO, DATASET_FILE, repo_type="dataset", revision=revision
    )
    zip_path = hf_hub_download(
        DATASET_REPO, ARCHIVE_FILE, repo_type="dataset", revision=revision
    )
    with open(csv_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, zipfile.ZipFile(zip_path)


def archive_index(archive: zipfile.ZipFile) -> dict[str, str]:
    """Map bare filename to archive entry; the dataset references them by name."""
    return {
        Path(name).name: name
        for name in archive.namelist()
        if not name.endswith("/")
    }


def document_names(row: dict[str, Any]) -> list[str]:
    return [name.strip() for name in str(row.get("data_source_filenames", "")).split(";") if name.strip()]


def build_prompt(question: str, documents: list[tuple[str, str]]) -> str:
    """Documents in the order the dataset lists them, each clearly delimited."""
    parts = []
    for index, (name, text) in enumerate(documents, start=1):
        parts.append(f"--- Document {index}: {name} ---\n{text}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def normalize(value: str) -> str:
    text = WHITESPACE_RE.sub(" ", str(value)).strip().casefold()
    return TRAILING_RE.sub("", text)


def materialize(
    rows: list[dict[str, Any]], archive: zipfile.ZipFile, *, skip_incomplete: bool
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    index = archive_index(archive)
    prompts, key, dropped = [], {}, []
    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        names = document_names(row)
        if not question or not answer or not names:
            raise AdapterError(f"row {row.get('question_id')} is incomplete")
        missing = [name for name in names if name not in index]
        if missing:
            if not skip_incomplete:
                raise AdapterError(
                    f"question {row.get('question_id')} references documents absent from "
                    f"the archive: {missing[:3]}. Pass --skip-incomplete to drop such "
                    "questions and record them."
                )
            dropped.append({"question_id": row.get("question_id"), "missing": missing})
            continue
        documents = [
            (name, archive.read(index[name]).decode("utf-8", errors="replace"))
            for name in names
        ]
        item_id = f"lcr-{row['question_id']}"
        if item_id in key:
            raise AdapterError(f"duplicate item id {item_id}")
        prompts.append(
            {
                "id": item_id,
                "suite": SUITE,
                "text": build_prompt(question, documents),
                "category": str(row.get("document_category") or "unknown"),
            }
        )
        key[item_id] = {
            "answer": answer,
            "normalized": normalize(answer),
            "category": str(row.get("document_category") or "unknown"),
            "document_set": str(row.get("document_set_id") or ""),
            "documents": names,
            "input_tokens": int(row.get("input_tokens") or 0),
        }
    if not prompts:
        raise AdapterError("no items materialized")
    return prompts, key, dropped


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    pins = load_pins()
    validate_pins(pins)
    run_dir = env_path("EVAL_RUN_DIR")
    rows, archive = download(pins["dataset"])
    prompts, key, dropped = materialize(
        rows, archive, skip_incomplete=args.skip_incomplete
    )
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {
            "dataset": f"{DATASET_REPO}@{pins['dataset']}",
            "items": key,
            # Never a silent cap: what was dropped travels with the item set.
            "dropped": dropped,
        },
    )
    if dropped:
        ids = ", ".join(str(entry["question_id"]) for entry in dropped)
        print(
            f"warning: dropped {len(dropped)} question(s) whose documents are absent "
            f"from the archive: {ids}",
            file=sys.stderr,
        )
    print(f"materialized {len(prompts)} {SUITE} items to {prompts_path}", flush=True)
    return 0


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
            "document_set": entry["document_set"],
            # The whole point of the suite: how much context this item carried.
            "input_tokens": entry["input_tokens"],
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
                "category": entry["category"],
                "document_set": entry["document_set"],
                "input_tokens": entry["input_tokens"],
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
            "dropped": stored.get("dropped"),
            "generation": generation,
            "generation_overrides": {},
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "score": round(sum(row["score"] for row in rows) / len(rows), 6),
            "score_by_category": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_category.items())
            },
            # A truncated reply on a 100k-token prompt means the context ran out,
            # which is a different failure from getting the answer wrong.
            "context_failures": sum(1 for row in rows if row.get("context_failure")),
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
                "dataset": dataset or "REPLACE_WITH_AA_LCR_REVISION",
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
