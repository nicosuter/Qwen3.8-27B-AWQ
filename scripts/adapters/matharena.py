#!/usr/bin/env python3
"""MathArena adapter for scripts/run_eval_protocol.py.

Note on the snapshots: the protocol names `arxivmath-2026-06` and
`brokenarxiv-2026-06`, neither of which is published by the MathArena
organization. This adapter scores the snapshots that do exist, defaulting to
AIME 2026 plus the Apex shortlist. Both carry exact integer answers, so no judge
model and no judge revision enters the gate, which is what the protocol wanted
from this suite in the first place.

Each snapshot is pinned individually as `repo@commit`, because one suite draws
from several datasets.
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


SUITE = "matharena_2026_06"
HARNESS_ID = "builtin-matharena-exact-v1"
VERIFIER_ID = "exact-integer-answer-v1"
DEFAULT_SNAPSHOTS = ("MathArena/aime_2026", "MathArena/apex-shortlist")

DEFAULT_MAX_TOKENS = 65536

ANSWER_INSTRUCTION = (
    "Solve the problem. End your reply with a final line of exactly this form:\n\n"
    "Answer: <integer>\n\n"
    "giving only the integer answer, with no units, commas, or explanation."
)

BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
ANSWER_LINE_RE = re.compile(r"answer\b[^0-9\-]{0,12}(-?\d[\d,]*)", re.IGNORECASE)
INTEGER_RE = re.compile(r"-?\d[\d,]*")
PIN_RE = re.compile(r"^([\w\-./]+)@([0-9a-f]{40})$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare", help="materialize problems and answers")
    prepare.add_argument(
        "--snapshots",
        default=",".join(DEFAULT_SNAPSHOTS),
        help="comma-separated dataset repos; each needs a pin in pins.dataset",
    )

    run = sub.add_parser("run", help="score the frozen task order against the server")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--request-timeout", type=float, default=1800.0)
    run.add_argument("--retries", type=int, default=2)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--snapshots", default=",".join(DEFAULT_SNAPSHOTS))
    pin.add_argument("--resolve", action="store_true", help="look the commits up on the Hub")
    return parser.parse_args(argv)


def self_pin() -> str:
    return module_pin([Path(__file__), Path(__file__).resolve().parent / "_common.py"])


def raw_response_path(run_dir: Path, variant: str, replicate: int, item_id: str) -> Path:
    return _raw_response_path(run_dir, SUITE, variant, replicate, item_id)


def parse_snapshot_pins(raw: str) -> dict[str, str]:
    """pins.dataset is `repo@commit` per snapshot, because one suite spans several."""
    pins = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        match = PIN_RE.match(entry)
        if not match:
            raise AdapterError(
                "pins.dataset entries must be repo@40-character-commit; "
                f"got {entry!r}. A branch or tag is not an immutable pin."
            )
        pins[match.group(1)] = match.group(2)
    if not pins:
        raise AdapterError("pins.dataset is empty")
    return pins


def validate_pins(pins: dict[str, str], snapshots: list[str] | None = None) -> dict[str, str]:
    resolved = parse_snapshot_pins(pins.get("dataset", ""))
    if snapshots is not None:
        missing = [name for name in snapshots if name not in resolved]
        if missing:
            raise AdapterError(f"pins.dataset has no commit for {missing}")
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())
    return resolved


def load_snapshot(repo: str, revision: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("the datasets package is required for prepare") from error
    rows = load_dataset(repo, split="train", revision=revision)
    return [dict(row) for row in rows]


def normalize_answer(value: Any) -> str | None:
    """Answers are integers; accept 1,234 and 1234 as the same value."""
    text = str(value).strip()
    match = INTEGER_RE.search(text)
    if not match:
        return None
    return str(int(match.group(0).replace(",", "")))


def materialize(
    snapshots: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for repo, rows in snapshots.items():
        short = repo.split("/")[-1]
        for row in rows:
            problem = str(row.get("problem", "")).strip()
            answer = normalize_answer(row.get("answer"))
            if not problem:
                raise AdapterError(f"{short}: a row has no problem text")
            if answer is None:
                raise AdapterError(f"{short}: a row has no integer answer")
            item_id = f"{short}-{row.get('problem_idx', len(key))}"
            if item_id in key:
                raise AdapterError(f"duplicate item id {item_id}")
            prompts.append(
                {
                    "id": item_id,
                    "suite": SUITE,
                    "text": problem,
                    "category": short,
                    "source": row.get("source") or short,
                }
            )
            key[item_id] = {
                "answer": answer,
                "category": short,
                "source": row.get("source") or short,
            }
    if not prompts:
        raise AdapterError("no problems materialized")
    return prompts, key


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    snapshots = [name.strip() for name in args.snapshots.split(",") if name.strip()]
    resolved = validate_pins(load_pins(), snapshots)
    run_dir = env_path("EVAL_RUN_DIR")
    prompts_path = env_path("EVAL_PROMPTS_JSONL")

    loaded = {repo: load_snapshot(repo, resolved[repo]) for repo in snapshots}
    prompts, key = materialize(loaded)
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {
            "suite": SUITE,
            "snapshots": {repo: resolved[repo] for repo in snapshots},
            "snapshot_note": (
                "arxivmath-2026-06 and brokenarxiv-2026-06 are not published; "
                "these are the MathArena snapshots that exist"
            ),
            "verifier": VERIFIER_ID,
            "adapter": self_pin(),
            "items": key,
        },
    )
    print(f"materialized {len(prompts)} {SUITE} problems to {prompts_path}", flush=True)
    return 0


def extract_answer(text: str) -> str | None:
    boxed = BOXED_RE.findall(text)
    if boxed:
        answer = normalize_answer(boxed[-1])
        if answer is not None:
            return answer
    matches = ANSWER_LINE_RE.findall(text)
    if matches:
        return normalize_answer(matches[-1])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and INTEGER_RE.fullmatch(lines[-1].replace(" ", "")):
        return normalize_answer(lines[-1])
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

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            "score": 1.0 if predicted is not None and predicted == entry["answer"] else 0.0,
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
            "predicted": predicted,
            "expected": entry["answer"],
            "category": entry["category"],
            "source": entry["source"],
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
        row.update({"category": entry["category"], "source": entry["source"],
                    "expected": entry["answer"], "predicted": None})
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
            "snapshots": stored.get("snapshots"),
            "snapshot_note": stored.get("snapshot_note"),
            "generation": generation,
            "generation_overrides": {},
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "accuracy": round(sum(row["score"] for row in rows) / len(rows), 6),
            "accuracy_by_snapshot": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_category.items())
            },
        },
    )
    print(f"scored {len(rows)} {SUITE} items to {results_path}", flush=True)
    return 0


def command_pin(args: argparse.Namespace) -> int:
    snapshots = [name.strip() for name in args.snapshots.split(",") if name.strip()]
    if args.resolve:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise AdapterError("huggingface_hub is required for --resolve") from error
        api = HfApi()
        dataset = ",".join(f"{repo}@{api.dataset_info(repo).sha}" for repo in snapshots)
    else:
        dataset = ",".join(f"{repo}@REPLACE_WITH_COMMIT" for repo in snapshots)
    print(
        json.dumps(
            {
                "dataset": dataset,
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
