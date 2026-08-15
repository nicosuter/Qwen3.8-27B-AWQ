#!/usr/bin/env python3
"""RULER long-context adapter for scripts/run_eval_protocol.py.

This is a self-contained synthesizer, not an upstream RULER reproduction. Scores
are comparable between FP8 and AWQ on identical materialized items; they are not
comparable to published RULER numbers. `prepare` builds every haystack from a
content-pinned corpus and the synthesis seed, so both checkpoints receive
byte-identical prompts. `run` scores by string-match recall over the expected
values.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Protocol

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
        split_reasoning,
        reasoning_tokens,
        request_with_retries,
        require_pin,
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
        split_reasoning,
        reasoning_tokens,
        request_with_retries,
        require_pin,
        timeout_row as _timeout_row,
        unpack_choice,
        write_json,
        write_jsonl,
    )


SUITE = "ruler"
HARNESS_ID = "builtin-ruler-synth-v1"
VERIFIER_ID = "recall-string-match-v1"

TASKS = (
    "niah_single",
    "niah_multikey",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
)

# 131072 is the top length by decision: the server runs --max-model-len 262144,
# and a 262144-token prompt leaves no output budget for a thinking model.
DEFAULT_LENGTHS = (4096, 32768, 131072)
DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_OUTPUT_RESERVE = 16384
# Slack for the chat template wrapper, which prepare cannot see.
TEMPLATE_ALLOWANCE = 256
LENGTH_TOLERANCE = 0.02

ANSWER_INSTRUCTION = (
    "End your reply with a final line of exactly this form:\n\n"
    "Answer: <comma-separated values>\n\n"
    "listing every value the question asks for and nothing else."
)

KEY_WORDS = (
    "anchor", "basil", "cobalt", "dahlia", "ember", "fjord", "granite", "harbor",
    "indigo", "juniper", "kestrel", "lantern", "marble", "nectar", "obsidian",
    "pewter", "quartz", "ripple", "saffron", "tundra", "umber", "velvet",
    "walnut", "xenon", "yarrow", "zephyr", "amber", "bramble", "cinder",
    "driftwood", "elder", "fennel", "gossamer", "hollow", "ivory", "jasper",
    "kelp", "lichen", "meadow", "nimbus", "onyx", "prairie", "quiver", "rosin",
    "sable", "thistle", "upland", "vellum", "willow", "yonder",
)

COMMON_WORD_POOL = tuple(f"item{index:03d}" for index in range(400))
FWE_WORD_POOL = tuple(f"token{index:03d}" for index in range(600))

# The frequent words form a descending ladder from TOP_RATIO down to
# TIGHT_RATIO times the filler count. The lowest rung sits deliberately close to
# the filler: a task both checkpoints ace measures nothing about degradation.
TOP_RATIO = 3.0
TIGHT_RATIO = 2.0
TARGET_FILLER_REPEATS = 8
MIN_FILLER_WORDS = 20
MIN_FILLER_REPEATS = 3
# Counting is bounded by how many distinct words must be tallied, not by list
# length. Left uncapped, a 128k list asks for the top 3 of 400 buckets and the
# model reasons until it hits the output cap at every length, scoring 0 for both
# checkpoints and measuring nothing. The list still scales with context; the
# bookkeeping does not.
MAX_DISTINCT_FILLER = 24

NEEDLE_INSTRUCTION = (
    "Some special magic numbers are hidden in the following text. Memorize them; "
    "you will be asked about them afterwards."
)
VT_INSTRUCTION = (
    "The following text contains variable assignments. Track them; you will be "
    "asked which variables hold a given value."
)
CWE_INSTRUCTION = (
    "Below is a long list of words. Count how often each word appears; you will "
    "be asked for the most frequent ones."
)
# Asking for "the most frequent words" outright invites a thinking model to tally
# every word, which exhausts any output budget: every such item returned nothing
# after 16384 tokens while the retrieval tasks finished in under 2500. Offering
# candidates bounds the work to comparing a shortlist without removing the need
# to attend over the whole list.
WORD_CANDIDATE_RATIO = 3
WORD_QUESTIONS = {
    "cwe": "Which {k} of these words appear most often in the list above?",
    "fwe": "Which {k} of these words appear most often in the list above?",
}

ANSWER_SEGMENT_RE = re.compile(r"answer\s*[::]\s*", re.IGNORECASE)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare", help="synthesize haystacks and the answer key")
    prepare.add_argument(
        "--lengths",
        default=",".join(str(length) for length in DEFAULT_LENGTHS),
        help="comma-separated context lengths in tokens",
    )
    prepare.add_argument("--synthesis-seed", type=int, required=True)
    prepare.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="UTF-8 text file, or directory of .txt files, hashed into pins.dataset",
    )
    prepare.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "")),
        help="checkpoint whose tokenizer defines the length targets",
    )
    prepare.add_argument("--items-per-task", type=int, default=10)
    prepare.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    prepare.add_argument("--output-reserve", type=int, default=DEFAULT_OUTPUT_RESERVE)

    run = sub.add_parser("run", help="score the frozen task order against the server")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_OUTPUT_RESERVE)
    run.add_argument("--concurrency-note", help=argparse.SUPPRESS)
    run.add_argument("--request-timeout", type=float, default=3600.0)
    run.add_argument("--retries", type=int, default=2)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--corpus", type=Path, required=True)
    return parser.parse_args(argv)


def self_pin() -> str:
    return module_pin([Path(__file__), Path(__file__).resolve().parent / "_common.py"])


def raw_response_path(run_dir: Path, variant: str, replicate: int, item_id: str) -> Path:
    return _raw_response_path(run_dir, SUITE, variant, replicate, item_id)


def read_corpus(path: Path) -> str:
    if path.is_dir():
        parts = [file.read_text(encoding="utf-8") for file in sorted(path.glob("*.txt"))]
        if not parts:
            raise AdapterError(f"{path}: no .txt files in the corpus directory")
        return "\n\n".join(parts)
    if not path.is_file():
        raise AdapterError(f"{path}: corpus not found")
    return path.read_text(encoding="utf-8")


def corpus_pin(path: Path) -> str:
    """Content hash, so both checkpoints demonstrably share one haystack source."""
    return "sha256:" + hashlib.sha256(read_corpus(path).encode()).hexdigest()


def validate_pins(pins: dict[str, str], corpus: Path | None = None) -> None:
    dataset = pins.get("dataset", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", dataset):
        raise AdapterError(
            "pins.dataset must be the sha256 content hash of the haystack corpus; "
            f"got {dataset!r}. Run `ruler.py pin --corpus <path>` to produce it."
        )
    if corpus is not None and corpus_pin(corpus) != dataset:
        raise AdapterError(
            f"corpus at {corpus} does not match pins.dataset; the haystack source changed"
        )
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def parse_lengths(raw: str, *, max_model_len: int, output_reserve: int) -> list[int]:
    lengths = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit() or int(part) <= 0:
            raise AdapterError(f"--lengths must be positive integers; got {part!r}")
        lengths.append(int(part))
    if not lengths:
        raise AdapterError("--lengths is empty")
    budget = max_model_len - output_reserve - TEMPLATE_ALLOWANCE
    over = [length for length in lengths if length > budget]
    if over:
        raise AdapterError(
            f"lengths {over} exceed the usable window: --max-model-len {max_model_len} "
            f"minus --output-reserve {output_reserve} leaves {budget} prompt tokens. "
            "A prompt that fills the whole window leaves no room for an answer."
        )
    return sorted(set(lengths))


def length_label(length: int) -> str:
    return f"{length // 1024}k" if length % 1024 == 0 else str(length)


def load_tokenizer(path: Path) -> Tokenizer:
    if not str(path):
        raise AdapterError("--tokenizer is required (or set OUTPUT_DIR)")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("the transformers package is required for prepare") from error
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def scatter(filler_ids: list[int], blocks: list[list[int]], depths: list[float]) -> list[int]:
    """Insert needle token blocks into filler at fractional depths, in order."""
    positions = [min(len(filler_ids), max(0, round(depth * len(filler_ids)))) for depth in depths]
    positions.sort()
    result: list[int] = []
    previous = 0
    for position, block in zip(positions, blocks):
        result.extend(filler_ids[previous:position])
        result.extend(block)
        previous = position
    result.extend(filler_ids[previous:])
    return result


def filler_slice(corpus_ids: list[int], count: int, rng: random.Random) -> list[int]:
    if count <= 0:
        return []
    if not corpus_ids:
        raise AdapterError("the corpus tokenized to nothing")
    start = rng.randrange(len(corpus_ids))
    repeats = count // len(corpus_ids) + 2
    extended = (corpus_ids * repeats)[start : start + count]
    return extended


def build_needle_task(
    task: str, rng: random.Random
) -> tuple[str, list[str], str, list[str]]:
    """Return (instruction, needle sentences, query, expected values)."""
    def number() -> str:
        return str(rng.randrange(1000000, 9999999))

    if task == "niah_single":
        key = rng.choice(KEY_WORDS)
        value = number()
        needles = [f"One of the special magic numbers for {key} is: {value}."]
        query = f"What is the special magic number for {key} mentioned in the text?"
        return NEEDLE_INSTRUCTION, needles, query, [value]

    if task == "niah_multikey":
        keys = rng.sample(KEY_WORDS, 4)
        values = [number() for _ in keys]
        needles = [
            f"One of the special magic numbers for {key} is: {value}."
            for key, value in zip(keys, values)
        ]
        index = rng.randrange(len(keys))
        query = f"What is the special magic number for {keys[index]} mentioned in the text?"
        return NEEDLE_INSTRUCTION, needles, query, [values[index]]

    if task == "niah_multivalue":
        key = rng.choice(KEY_WORDS)
        values = [number() for _ in range(4)]
        needles = [f"One of the special magic numbers for {key} is: {value}." for value in values]
        query = f"What are all the special magic numbers for {key} mentioned in the text?"
        return NEEDLE_INSTRUCTION, needles, query, values

    if task == "niah_multiquery":
        keys = rng.sample(KEY_WORDS, 4)
        values = [number() for _ in keys]
        needles = [
            f"One of the special magic numbers for {key} is: {value}."
            for key, value in zip(keys, values)
        ]
        listed = ", ".join(keys)
        query = f"What are the special magic numbers for {listed} mentioned in the text?"
        return NEEDLE_INSTRUCTION, needles, query, values

    if task == "vt":
        names = [f"VAR{index}" for index in rng.sample(range(100, 999), 20)]
        target_chain = names[:5]
        value = str(rng.randrange(10000, 99999))
        statements = [f"VAR {target_chain[0]} = {value}"]
        statements += [
            f"VAR {target_chain[index]} = {target_chain[index - 1]}"
            for index in range(1, len(target_chain))
        ]
        for offset in range(5, 20, 5):  # distractor chains
            chain = names[offset : offset + 5]
            if len(chain) < 5:
                break
            statements.append(f"VAR {chain[0]} = {rng.randrange(10000, 99999)}")
            statements += [
                f"VAR {chain[index]} = {chain[index - 1]}" for index in range(1, len(chain))
            ]
        query = (
            f"Find every variable that is assigned the value {value}, "
            "directly or through another variable."
        )
        return VT_INSTRUCTION, statements, query, target_chain

    raise AdapterError(f"{task} is not a needle task")


def frequency_ladder(count: int) -> list[float]:
    """Descending multipliers over the filler count, ending at TIGHT_RATIO."""
    if count < 2:
        return [TOP_RATIO]
    step = (TOP_RATIO - TIGHT_RATIO) / (count - 1)
    return [TOP_RATIO - step * index for index in range(count)]


def compose_word_list(
    total: int,
    frequent: list[str],
    filler_pool: list[str],
    rng: random.Random,
) -> list[str]:
    """Lay out the list so the frequent words genuinely dominate at every length.

    A fixed repeat count does not survive scaling: at 131k tokens each of the
    few hundred filler words would recur more often than a "frequent" word, and
    the task would have no well-defined answer.
    """
    ladder = frequency_ladder(len(frequent))
    weight = sum(ladder)
    # Spend as many distinct filler words as the list can carry while each still
    # recurs often enough for the ladder to sit above it.
    usable = min(
        len(filler_pool),
        MAX_DISTINCT_FILLER,
        max(MIN_FILLER_WORDS, int(total / TARGET_FILLER_REPEATS - weight)),
    )
    base = total / (usable + weight)
    if usable < MIN_FILLER_WORDS or base < MIN_FILLER_REPEATS:
        raise AdapterError(
            f"a {total}-word list is too short for a word-frequency task: "
            f"{usable} filler words would recur only {base:.1f} times each"
        )
    counts = [max(round(rung * base), round(base) + 2) for rung in ladder]
    filler_slots = total - sum(counts)
    per_filler, remainder = divmod(filler_slots, usable)
    top_filler = per_filler + bool(remainder)
    if min(counts) <= top_filler:
        raise AdapterError(
            f"a {total}-word list cannot separate {len(frequent)} frequent words "
            f"(lowest {min(counts)}) from {usable} filler words (up to {top_filler})"
        )
    words = [word for word, count in zip(frequent, counts) for _ in range(count)]
    # The remainder is spread one word at a time; integer division alone would
    # quantize the list length to steps of the whole filler pool.
    for index, word in enumerate(filler_pool[:usable]):
        words += [word] * (per_filler + (1 if index < remainder else 0))
    rng.shuffle(words)
    return words


def build_word_list(
    task: str, budget: int, tokenizer: Tokenizer, rng: random.Random
) -> tuple[str, list[str], list[str]]:
    """Return (word list text, expected words, candidate shortlist)."""
    pool, frequent_count = (COMMON_WORD_POOL, 10) if task == "cwe" else (FWE_WORD_POOL, 3)
    frequent = list(rng.sample(pool, frequent_count))
    filler_pool = [word for word in pool if word not in frequent]

    sample = ", ".join(rng.choices(pool, k=200))
    ratio = max(len(tokenizer.encode(sample)) / 200, 0.25)
    total = max(int(budget / ratio), frequent_count * 8)
    text = ""
    for _ in range(4):
        words = compose_word_list(total, frequent, filler_pool, rng)
        text = ", ".join(words)
        tokens = len(tokenizer.encode(text))
        if abs(tokens - budget) <= max(64, budget * LENGTH_TOLERANCE / 2):
            break
        total = max(int(total * budget / max(tokens, 1)), frequent_count * 8)
    # Distractors are drawn from words that really are in the list, so absence
    # cannot be used to eliminate them without reading it.
    present = [word for word in filler_pool if word in set(text.split(", "))]
    distractors = rng.sample(present, min(len(present), frequent_count * WORD_CANDIDATE_RATIO))
    candidates = frequent + distractors
    rng.shuffle(candidates)
    return text, frequent, candidates


def build_item(
    task: str,
    length: int,
    index: int,
    *,
    corpus_ids: list[int],
    tokenizer: Tokenizer,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item_id = f"{SUITE}-{length_label(length)}-{task}-{index:02d}"
    rng = random.Random(f"{seed}:{item_id}")
    # The allowance is deliberately left unfilled for the chat template wrapper,
    # which prepare cannot measure.
    target = length - TEMPLATE_ALLOWANCE

    if task in ("cwe", "fwe"):
        instruction = CWE_INSTRUCTION
        frequent_count = 10 if task == "cwe" else 3
        # The query carries the shortlist, so its length is known only after the
        # list is built; reserve a generous allowance and measure exactly after.
        overhead = len(tokenizer.encode(instruction)) + 256
        body, expected, candidates = build_word_list(task, target - overhead, tokenizer, rng)
        query = (
            WORD_QUESTIONS[task].format(k=frequent_count)
            + "\n\nCandidates: "
            + ", ".join(candidates)
        )
    else:
        instruction, needles, query, expected = build_needle_task(task, rng)
        overhead = len(tokenizer.encode(instruction)) + len(tokenizer.encode(query))
        blocks = [tokenizer.encode(f" {needle} ") for needle in needles]
        filler_count = target - overhead - sum(len(block) for block in blocks)
        if filler_count <= 0:
            raise AdapterError(f"{item_id}: length {length} is too small for this task")
        depths = sorted(
            (position + 1) / (len(blocks) + 1) + rng.uniform(-0.04, 0.04)
            for position in range(len(blocks))
        )
        body = tokenizer.decode(
            scatter(filler_slice(corpus_ids, filler_count, rng), blocks, depths)
        )

    text = f"{instruction}\n\n{body}\n\n{query}"
    achieved = len(tokenizer.encode(text))
    if abs(achieved - target) > max(TEMPLATE_ALLOWANCE // 2, length * LENGTH_TOLERANCE):
        raise AdapterError(
            f"{item_id}: assembled {achieved} tokens for a {target}-token target"
        )

    prompt = {
        "id": item_id,
        "suite": SUITE,
        "text": text,
        "category": f"{length_label(length)}/{task}",
        "task": task,
        "length": length_label(length),
    }
    key = {
        "task": task,
        "length": length_label(length),
        "nominal_tokens": length,
        "achieved_tokens": achieved,
        "expected": expected,
    }
    return prompt, key


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    pins = load_pins()
    validate_pins(pins, args.corpus)
    lengths = parse_lengths(
        args.lengths, max_model_len=args.max_model_len, output_reserve=args.output_reserve
    )
    if args.items_per_task < 1:
        raise AdapterError("--items-per-task must be at least 1")

    run_dir = env_path("EVAL_RUN_DIR")
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    tokenizer = load_tokenizer(args.tokenizer)
    corpus_ids = tokenizer.encode(read_corpus(args.corpus))

    prompts = []
    key = {}
    for length in lengths:
        for task in TASKS:
            for index in range(args.items_per_task):
                prompt, entry = build_item(
                    task,
                    length,
                    index,
                    corpus_ids=corpus_ids,
                    tokenizer=tokenizer,
                    seed=args.synthesis_seed,
                )
                prompts.append(prompt)
                key[prompt["id"]] = entry

    write_jsonl(prompts_path, prompts)
    total_tokens = sum(entry["achieved_tokens"] for entry in key.values())
    write_json(
        key_path(run_dir),
        {
            "suite": SUITE,
            "lengths": lengths,
            "tasks": list(TASKS),
            "items_per_task": args.items_per_task,
            "synthesis_seed": args.synthesis_seed,
            "corpus": str(args.corpus),
            "corpus_sha256": pins["dataset"],
            "max_model_len": args.max_model_len,
            "output_reserve": args.output_reserve,
            "verifier": VERIFIER_ID,
            "adapter": self_pin(),
            "prompt_tokens_total": total_tokens,
            "items": key,
        },
    )
    print(
        f"materialized {len(prompts)} {SUITE} prompts "
        f"({total_tokens} prompt tokens per replicate per checkpoint) to {prompts_path}",
        flush=True,
    )
    return 0


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def answer_segment(content: str) -> str:
    matches = list(ANSWER_SEGMENT_RE.finditer(content))
    return content[matches[-1].end() :] if matches else content


def score_answer(expected: list[str], answer: str) -> float:
    if not expected:
        raise AdapterError("an item has no expected values")
    found = sum(
        1
        for value in expected
        if re.search(rf"(?<![0-9A-Za-z]){re.escape(value)}(?![0-9A-Za-z])", answer, re.IGNORECASE)
    )
    return found / len(expected)


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
    score = score_answer(entry["expected"], segment)
    thought = reasoning_tokens(usage, reasoning, answer)

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            "score": score,
            "empty_answer": not answer.strip(),
            "repetition_loop": has_repetition_loop(answer or reasoning),
            # RULER is served without tools, so a malformed call cannot occur.
            "malformed_tool_call": False,
            "premature_final_answer": bool(thinking and answer.strip() and thought == 0),
            # Prompts are built to fit the window, so this is output truncation only.
            "context_failure": finish_reason == "length",
            "task": entry["task"],
            "length": entry["length"],
            "category": f"{entry['length']}/{entry['task']}",
            "nominal_tokens": entry["nominal_tokens"],
            "achieved_tokens": entry["achieved_tokens"],
            "expected": entry["expected"],
            "matched": round(score * len(entry["expected"])),
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
        text,
        generation,
        model=model,
        seed=seed,
        max_tokens=args.max_tokens,
        instruction=ANSWER_INSTRUCTION,
    )
    started = time.monotonic()
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
        row = _timeout_row(SUITE, item_id, replicate)
        row.update(
            {
                "task": entry["task"],
                "length": entry["length"],
                "category": f"{entry['length']}/{entry['task']}",
                "nominal_tokens": entry["nominal_tokens"],
                "achieved_tokens": entry["achieved_tokens"],
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
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    row["attempts"] = attempts
    return row


def per_length_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    lengths: dict[str, list[float]] = {}
    for row in rows:
        lengths.setdefault(row["length"], []).append(row["score"])
    return {
        label: round(sum(scores) / len(scores), 6) for label, scores in sorted(lengths.items())
    }


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
            key[item_id],
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
            # Report per length; a single scalar hides where the recurrent state fails.
            "accuracy_by_length": per_length_accuracy(rows),
            "context_failures_by_length": {
                label: sum(
                    1 for row in rows if row["length"] == label and row["context_failure"]
                )
                for label in sorted({row["length"] for row in rows})
            },
        },
    )
    print(f"scored {len(rows)} {SUITE} items to {results_path}", flush=True)
    return 0


def command_pin(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "dataset": corpus_pin(args.corpus),
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
