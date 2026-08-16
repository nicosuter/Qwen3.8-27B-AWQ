#!/usr/bin/env python3
"""Multimodal adapter (DocVQA, ChartQA, TextVQA) for run_eval_protocol.py.

Each public set keeps its own published metric rather than being flattened into
one accuracy: ANLS for DocVQA, relaxed accuracy for ChartQA, and the VQA
consensus score for TextVQA. They report under the single `multimodal` suite
label with the set as a category, so the suite counts once in the macro average.

`prepare` writes every image to the run directory as PNG and records its sha256,
so both checkpoints are sent byte-identical pixels and a rerun can prove it.
The private UI-screenshot pack named in EVAL.md is not included: it does not
exist in this repository, and inventing one would put an unpinnable set inside a
release gate.
"""

import argparse
import base64
import hashlib
import io
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


SUITE = "multimodal"
HARNESS_ID = "builtin-vqa-v1"
VERIFIER_ID = "anls-relaxed-vqa-v1"

# repo, config, split, metric
SETS = {
    "docvqa": ("lmms-lab/DocVQA", "DocVQA", "validation", "anls"),
    "chartqa": ("lmms-lab/ChartQA", None, "test", "relaxed"),
    "textvqa": ("lmms-lab/textvqa", None, "validation", "vqa"),
}

DEFAULT_MAX_TOKENS = 16384
DEFAULT_ITEMS_PER_SET = 200
ANLS_THRESHOLD = 0.5
RELAXED_TOLERANCE = 0.05

ANSWER_INSTRUCTION = (
    "Answer the question about the image. End your reply with a final line of "
    "exactly this form:\n\nAnswer: <answer>\n\n"
    "giving the shortest answer that is correct, with no explanation."
)

ANSWER_SEGMENT_RE = re.compile(r"answer\s*[::]\s*", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
ARTICLES = {"a", "an", "the"}
PIN_RE = re.compile(r"^([\w\-./]+)@([0-9a-f]{40})$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare", help="materialize questions and images")
    prepare.add_argument("--sets", default=",".join(SETS))
    prepare.add_argument("--items-per-set", type=int, default=DEFAULT_ITEMS_PER_SET)

    run = sub.add_parser("run", help="score the frozen task order against the server")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--request-timeout", type=float, default=1800.0)
    run.add_argument("--retries", type=int, default=2)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--sets", default=",".join(SETS))
    pin.add_argument("--resolve", action="store_true")
    return parser.parse_args(argv)


def self_pin() -> str:
    return module_pin([Path(__file__), Path(__file__).resolve().parent / "_common.py"])


def raw_response_path(run_dir: Path, variant: str, replicate: int, item_id: str) -> Path:
    return _raw_response_path(run_dir, SUITE, variant, replicate, item_id)


def parse_set_pins(raw: str) -> dict[str, str]:
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


def validate_pins(pins: dict[str, str], names: list[str] | None = None) -> dict[str, str]:
    resolved = parse_set_pins(pins.get("dataset", ""))
    if names is not None:
        missing = [name for name in names if SETS[name][0] not in resolved]
        if missing:
            raise AdapterError(f"pins.dataset has no commit for {missing}")
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())
    return resolved


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


def normalize_text(value: Any) -> str:
    text = re.sub(r"[^\w\s.%-]", " ", str(value).casefold())
    words = [word for word in text.split() if word not in ARTICLES]
    return " ".join(words).strip()


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, 1):
        current = [i]
        for j, rchar in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (lchar != rchar)))
        previous = current
    return previous[-1]


def anls_score(prediction: str, answers: list[str]) -> float:
    """DocVQA's metric: 1 - normalized edit distance, zeroed below the threshold."""
    predicted = normalize_text(prediction)
    best = 0.0
    for answer in answers:
        target = normalize_text(answer)
        if not predicted and not target:
            best = max(best, 1.0)
            continue
        longest = max(len(predicted), len(target)) or 1
        similarity = 1.0 - levenshtein(predicted, target) / longest
        best = max(best, similarity)
    return best if best >= ANLS_THRESHOLD else 0.0


def relaxed_score(prediction: str, answers: list[str]) -> float:
    """ChartQA's metric: numeric answers within 5%, everything else exact."""
    predicted = normalize_text(prediction)
    for answer in answers:
        target = normalize_text(answer)
        if predicted == target:
            return 1.0
        left, right = NUMBER_RE.search(predicted), NUMBER_RE.search(target)
        if left and right:
            try:
                value, expected = float(left.group()), float(right.group())
            except ValueError:
                continue
            if expected == 0:
                if value == 0:
                    return 1.0
            elif abs(value - expected) / abs(expected) <= RELAXED_TOLERANCE:
                return 1.0
    return 0.0


def vqa_score(prediction: str, answers: list[str]) -> float:
    """TextVQA's metric: agreement with ten human answers, capped at one."""
    predicted = normalize_text(prediction)
    matches = sum(1 for answer in answers if normalize_text(answer) == predicted)
    return min(matches / 3.0, 1.0)


SCORERS = {"anls": anls_score, "relaxed": relaxed_score, "vqa": vqa_score}


def extract_rows(name: str, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    extracted = []
    for index, row in enumerate(rows):
        if len(extracted) >= limit:
            break
        question = str(row.get("question", "")).strip()
        if name == "chartqa":
            answers = [row.get("answer")]
        else:
            answers = list(row.get("answers") or [])
        answers = [str(a) for a in answers if str(a).strip()]
        image = row.get("image")
        if not question or not answers or image is None:
            raise AdapterError(f"{name}: row {index} is incomplete")
        identifier = (
            row.get("questionId") or row.get("question_id") or f"{name}-{index:04d}"
        )
        extracted.append(
            {
                "id": f"{name}-{identifier}",
                "question": question,
                "answers": answers,
                "image": image,
                "kind": row.get("type") or name,
            }
        )
    if len(extracted) < limit:
        raise AdapterError(f"{name}: only {len(extracted)} usable rows for a limit of {limit}")
    return extracted


def materialize(
    by_set: dict[str, list[dict[str, Any]]], run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for name, rows in by_set.items():
        metric = SETS[name][3]
        for row in rows:
            item_id = str(row["id"])
            if item_id in key:
                raise AdapterError(f"duplicate item id {item_id}")
            path = image_dir(run_dir) / f"{item_id}.png"
            digest = save_image(row["image"], path)
            prompts.append(
                {
                    "id": item_id,
                    "suite": SUITE,
                    "text": row["question"],
                    "category": name,
                }
            )
            key[item_id] = {
                "set": name,
                "metric": metric,
                "answers": row["answers"],
                # Relative to the run directory: a materialized set is meant to
                # be copied to another run, cluster or filesystem, and an
                # absolute path silently points back at the run that built it.
                "image": str(path.relative_to(run_dir)),
                "image_sha256": digest,
                "kind": row["kind"],
            }
    if not prompts:
        raise AdapterError("no items materialized")
    return prompts, key


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def load_set(name: str, revision: str, limit: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("the datasets package is required for prepare") from error
    repo, config, split, _ = SETS[name]
    rows = load_dataset(repo, config, split=split, revision=revision, streaming=True)
    collected = []
    for row in rows:
        collected.append(row)
        if len(collected) >= limit:
            break
    return collected


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    names = [name.strip() for name in args.sets.split(",") if name.strip()]
    unknown = [name for name in names if name not in SETS]
    if unknown:
        raise AdapterError(f"unknown sets {unknown}; known: {sorted(SETS)}")
    resolved = validate_pins(load_pins(), names)
    run_dir = env_path("EVAL_RUN_DIR")
    prompts_path = env_path("EVAL_PROMPTS_JSONL")

    by_set = {}
    for name in names:
        repo = SETS[name][0]
        rows = load_set(name, resolved[repo], args.items_per_set)
        by_set[name] = extract_rows(name, rows, args.items_per_set)

    prompts, key = materialize(by_set, run_dir)
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {
            "suite": SUITE,
            "sets": {name: {"repo": SETS[name][0], "revision": resolved[SETS[name][0]],
                            "split": SETS[name][2], "metric": SETS[name][3]}
                     for name in names},
            "items_per_set": args.items_per_set,
            "private_ui_pack": "not included; no such pack exists in this repository",
            "verifier": VERIFIER_ID,
            "adapter": self_pin(),
            "items": key,
        },
    )
    print(f"materialized {len(prompts)} {SUITE} items to {prompts_path}", flush=True)
    return 0


def answer_segment(content: str) -> str:
    matches = list(ANSWER_SEGMENT_RE.finditer(content))
    return content[matches[-1].end():].strip() if matches else content.strip()


def resolve_image(run_dir: Path, stored: str) -> Path:
    """Locate an image recorded in the key file.

    Key files written before paths were made relative carry an absolute path
    into whichever run first materialized the set, which does not resolve once
    the set is copied elsewhere. Prefer this run's own copy; the sha256 check on
    read is what makes that substitution safe.
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


def score_response(
    item_id: str,
    response: dict[str, Any],
    *,
    entry: dict[str, Any],
    replicate: int,
    thinking: bool,
) -> dict[str, Any]:
    content, raw_reasoning, finish_reason, usage = unpack_choice(item_id, response)
    reasoning, answer = split_reasoning(content, raw_reasoning)
    segment = answer_segment(answer)
    scorer = SCORERS[entry["metric"]]
    score = scorer(segment, entry["answers"]) if segment else 0.0
    thought = reasoning_tokens(usage, reasoning, answer)

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            "score": float(score),
            "empty_answer": not segment,
            "repetition_loop": has_repetition_loop(answer or reasoning),
            # vLLM discards an unterminated think block, so a reply that ran to
            # the cap arrives with no text at all -- exactly where a loop is most
            # likely. Record whether there was anything to inspect, so a False
            # here is not read as "checked, and clean".
            "repetition_assessed": bool(answer or reasoning),
            "malformed_tool_call": False,
            "premature_final_answer": bool(thinking and segment and thought == 0),
            "context_failure": finish_reason == "length",
            "category": entry["set"],
            "metric": entry["metric"],
            "predicted": segment[:200],
            "expected": entry["answers"][:5],
            "image_sha256": entry["image_sha256"],
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
    # Swap the text-only message for a text-plus-image one; the instruction and
    # every sampling field stay exactly as the shared builder produced them.
    payload["messages"][0]["content"] = [
        {"type": "image_url", "image_url": {"url": image_data_url(
            resolve_image(run_dir, entry["image"]), entry["image_sha256"])}},
        {"type": "text", "text": f"{text}\n\n{ANSWER_INSTRUCTION}"},
    ]
    started_wall, started = time.time(), time.monotonic()
    response, attempts = request_with_retries(
        item_id, payload, base_url=base_url, api_key=api_key,
        timeout=args.request_timeout, retries=args.retries, client=client,
    )
    if response is None:
        row = _timeout_row(SUITE, item_id, replicate)
        row.update({"category": entry["set"], "metric": entry["metric"],
                    "image_sha256": entry["image_sha256"]})
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

    by_set: dict[str, list[float]] = {}
    for row in rows:
        by_set.setdefault(row["category"], []).append(row["score"])
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
            "sets": stored.get("sets"),
            "private_ui_pack": stored.get("private_ui_pack"),
            "generation": generation,
            "generation_overrides": {},
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "score": round(sum(row["score"] for row in rows) / len(rows), 6),
            # Each set keeps its own published metric; the mean above is only
            # the suite's contribution to the macro average.
            "score_by_set": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_set.items())
            },
        },
    )
    print(f"scored {len(rows)} {SUITE} items to {results_path}", flush=True)
    return 0


def command_pin(args: argparse.Namespace) -> int:
    names = [name.strip() for name in args.sets.split(",") if name.strip()]
    repos = [SETS[name][0] for name in names if name in SETS]
    if args.resolve:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise AdapterError("huggingface_hub is required for --resolve") from error
        api = HfApi()
        dataset = ",".join(f"{repo}@{api.dataset_info(repo).sha}" for repo in repos)
    else:
        dataset = ",".join(f"{repo}@REPLACE_WITH_COMMIT" for repo in repos)
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
