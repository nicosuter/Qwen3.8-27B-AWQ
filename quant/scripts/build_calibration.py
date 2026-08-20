#!/usr/bin/env python3
"""Build the pinned public multimodal calibration manifest."""

import gc
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoProcessor

# Dependency-free on purpose so the rule is testable without a GPU node.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_trim import closes_in_window  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.8-27B")
MODEL_REVISION = os.environ.get(
    "MODEL_REVISION", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
ROOT = Path(os.environ.get("CALIBRATION_DIR", "artifacts/calibration")).resolve()
MANIFEST = ROOT / "manifest.jsonl"
IMAGE_ROOT = ROOT / "images"
SOURCE_CACHE_ROOT = ROOT / "source-cache"
SEED = int(os.environ.get("CALIBRATION_SEED", "38027"))
MAX_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
LONG_MIN_TOKENS = int(os.environ.get("LONG_MIN_TOKENS", "1536"))
FORCE_REBUILD = os.environ.get("FORCE_CALIBRATION_REBUILD", "0") == "1"
TEXT_SHUFFLE_MIN = int(os.environ.get("TEXT_SHUFFLE_MIN", "256"))
TEXT_SHUFFLE_MAX = int(os.environ.get("TEXT_SHUFFLE_MAX", "1024"))
TEXT_SHUFFLE_MULTIPLIER = int(os.environ.get("TEXT_SHUFFLE_MULTIPLIER", "16"))
LONG_SHUFFLE_BUFFER = int(os.environ.get("LONG_SHUFFLE_BUFFER", "512"))
VISION_SHUFFLE_MIN = int(os.environ.get("VISION_SHUFFLE_MIN", "64"))
VISION_SHUFFLE_MULTIPLIER = int(
    os.environ.get("VISION_SHUFFLE_MULTIPLIER", "8")
)
SOURCE_CACHE_VERSION = 1

LAMBDA_REVISION = "b92885e4f0161d4b2536512710e004d4892cac6e"
NEMOTRON_REVISION = "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8"
OPEN_SWE_REVISION = "ad4805a5aa7de70d99cab0bb8f99b15304c76de0"
CAULDRON_REVISION = "847a98a779b1652d65111daf20c972dfcd333605"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

TEXT_SOURCES = (
    ("nvidia/Open-SWE-Traces", "openhands", "qwen35_122b", 52, OPEN_SWE_REVISION),
    ("nvidia/Open-SWE-Traces", "sweagent", "qwen35_122b", 52, OPEN_SWE_REVISION),
    ("lambda/hermes-agent-reasoning-traces", "kimi", "train", 32, LAMBDA_REVISION),
    ("lambda/hermes-agent-reasoning-traces", "glm-5.1", "train", 4, LAMBDA_REVISION),
    ("nvidia/Nemotron-Post-Training-Dataset-v1", None, "tool_calling", 4, NEMOTRON_REVISION),
    ("nvidia/Nemotron-Post-Training-Dataset-v1", None, "stem", 28, NEMOTRON_REVISION),
    ("nvidia/Nemotron-Post-Training-Dataset-v1", None, "math", 28, NEMOTRON_REVISION),
)
VISION_SOURCES = (
    ("vqav2", 12),
    ("textvqa", 9),
    ("chartqa", 9),
    ("docvqa", 9),
    ("ai2d", 9),
)
LONG_COUNT = 8
TOTAL = sum(source[3] for source in TEXT_SOURCES) + sum(
    count for _, count in VISION_SOURCES
) + LONG_COUNT
assert TOTAL == 256

TOOLS_BLOCK_RE = re.compile(r"<tools>.*?</tools>", re.DOTALL | re.IGNORECASE)


def normalize_role(role: str) -> str:
    return {"human": "user", "gpt": "assistant", "function": "tool"}.get(
        role, role
    )


def messages_from(row: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    raw: Any = row
    if isinstance(row, dict):
        raw = (
            row.get("messages")
            or row.get("trajectory")
            or row.get("conversations")
            or row.get("conversation")
        )
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        raw = raw.get("messages") or raw.get("trajectory")
    if isinstance(raw, list):
        result: list[dict[str, Any]] = []
        for message in raw:
            if not isinstance(message, dict):
                continue
            role = normalize_role(
                str(message.get("role") or message.get("from") or "user")
            )
            content = message.get("content", message.get("value", ""))
            if isinstance(content, list):
                content = "\n".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            normalized: dict[str, Any] = {"role": role, "content": str(content or "")}
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if reasoning:
                normalized["reasoning_content"] = str(reasoning)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, str):
                tool_calls = json.loads(tool_calls)
            if tool_calls:
                normalized_calls = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        raise ValueError("tool call must be a mapping")
                    function = call.get("function")
                    if not isinstance(function, dict):
                        raise ValueError("tool call function must be a mapping")
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool call arguments must decode to a mapping")
                    normalized_calls.append(
                        {**call, "function": {**function, "arguments": arguments}}
                    )
                normalized["tool_calls"] = normalized_calls
            if normalized["content"] or normalized.get("tool_calls"):
                result.append(normalized)
        if result:
            return result
    if isinstance(row, dict):
        prompt = row.get("prompt") or row.get("question") or row.get("input")
        answer = row.get("response") or row.get("answer") or row.get("output")
        if prompt and answer:
            return [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(answer)},
            ]
        raise ValueError(f"unsupported row schema: {sorted(row)}")
    raise ValueError(f"unsupported trajectory type: {type(row).__name__}")


def tools_from(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw = row.get("tools")
    if raw is None:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = None
        if isinstance(metadata, dict):
            raw = metadata.get("tools")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list):
        return None
    tools: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            tools.append(item)
        elif isinstance(item, list):
            tools.extend(tool for tool in item if isinstance(tool, dict))
    return tools or None


def tools_for_messages(
    tools: list[dict[str, Any]] | None, messages: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """Keep only schemas invoked by this trajectory window."""
    if not tools:
        return None
    names: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, dict) else call.get("name")
            if name:
                names.add(str(name))
    if not names:
        return None
    selected = []
    for tool in tools:
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if name in names:
            selected.append(tool)
    return selected or None


def compact_text(value: str, limit: int = 2_048) -> str:
    if len(value) <= limit:
        return value
    marker = "\n...[calibration window truncated]...\n"
    head = (limit - len(marker)) * 2 // 3
    tail = limit - len(marker) - head
    return value[:head] + marker + value[-tail:]


def compact_open_swe_window(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    compact_messages = []
    for message in messages:
        compact = dict(message)
        compact["content"] = compact_text(str(compact.get("content", "")))
        if compact.get("reasoning_content"):
            compact["reasoning_content"] = compact_text(
                str(compact["reasoning_content"])
            )
        compact_messages.append(compact)
    compact_tools = json.loads(json.dumps(tools))
    for tool in compact_tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("description"), str):
            function["description"] = compact_text(function["description"], 768)
    return compact_messages, compact_tools


def close_stream(iterator: Any, stream: Any) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()
    del iterator
    del stream
    gc.collect()


def source_cache_path(
    kind: str,
    dataset_id: str,
    config: str | None,
    split: str,
    count: int,
    revision: str,
) -> Path:
    specification = {
        "version": SOURCE_CACHE_VERSION,
        "kind": kind,
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "count": count,
        "revision": revision,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "long_min_tokens": LONG_MIN_TOKENS,
    }
    digest = hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{kind}-{dataset_id}-{config}-{split}")
    return SOURCE_CACHE_ROOT / f"{label}-{digest}.jsonl"


def load_source_cache(
    path: Path,
    count: int,
    expected: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if FORCE_REBUILD or not path.is_file():
        return None
    try:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != count:
            raise ValueError(f"rows={len(records)}, expected={count}")
        for record in records:
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError(
                        f"unexpected {key}={record.get(key)!r}, expected={value!r}"
                    )
            if record["kind"] == "vision":
                image_path = ROOT / record["image"]
                if not image_path.is_file() or image_path.stat().st_size == 0:
                    raise ValueError(f"missing cached image: {image_path}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"source-cache=invalid path={path.name} reason={error}", flush=True)
        return None
    print(f"source-cache=valid path={path.name} rows={len(records)}", flush=True)
    return records


def write_source_cache(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"source-cache=wrote path={path.name} rows={len(records)}", flush=True)


def backfill_source_caches(records: list[dict[str, Any]]) -> None:
    specifications = [
        ("text", dataset_id, config, split, count, revision)
        for dataset_id, config, split, count, revision in TEXT_SOURCES
    ]
    specifications.append(
        (
            "long",
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            "train",
            LONG_COUNT,
            FINEWEB_REVISION,
        )
    )
    specifications.extend(
        (
            "vision",
            "HuggingFaceM4/the_cauldron",
            config,
            "train",
            count,
            CAULDRON_REVISION,
        )
        for config, count in VISION_SOURCES
    )
    for kind, dataset_id, config, split, count, revision in specifications:
        expected_kind = "vision" if kind == "vision" else "text"
        expected = {
            "kind": expected_kind,
            "source": dataset_id,
            "config": config,
            "split": split,
            "revision": revision,
        }
        path = source_cache_path(kind, dataset_id, config, split, count, revision)
        if load_source_cache(path, count, expected) is not None:
            continue
        selected = [
            record
            for record in records
            if all(record.get(key) == value for key, value in expected.items())
        ]
        if len(selected) != count:
            raise ValueError(
                f"cannot backfill {path.name}: rows={len(selected)}, expected={count}"
            )
        write_source_cache(path, selected)


def load_stream(
    dataset_id: str,
    config: str | None,
    split: str,
    revision: str,
    seed_offset: int,
    shuffle_buffer: int,
):
    if dataset_id == "nvidia/Nemotron-Post-Training-Dataset-v1":
        prefix = "tool" if split == "tool_calling" else split
        files = f"hf://datasets/{dataset_id}@{revision}/data/{prefix}-*.parquet"
        stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
    else:
        stream = load_dataset(
            dataset_id, config, split=split, revision=revision, streaming=True
        )
    return stream.shuffle(seed=SEED + seed_offset, buffer_size=shuffle_buffer)


def render_row(processor: Any, row: dict[str, Any]) -> tuple[str, bool]:
    messages = messages_from(row)
    tools = tools_from(row)
    if tools:
        # Hermes embeds legacy JSON tool definitions in the system content.
        # Qwen receives the canonical schemas through tools=, so remove that
        # duplicate legacy block (and a now-empty system turn).
        cleaned = []
        for message in messages:
            if message["role"] == "system":
                content = TOOLS_BLOCK_RE.sub("", message.get("content", "")).strip()
                if not content or "function calling AI model" in content:
                    continue
                message = {**message, "content": content}
            cleaned.append(message)
        messages = cleaned
    text = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text, bool(tools)


def render_open_swe_window(
    processor: Any, row: dict[str, Any], window_index: int
) -> tuple[str, bool]:
    """Render a complete-turn prefix/interior/tail window in Qwen's format."""
    messages = messages_from(row.get("trajectory"))
    all_tools = tools_from(row)
    system = [message for message in messages[:1] if message["role"] == "system"]
    body = messages[len(system) :]
    if len(body) < 2:
        raise ValueError("Open-SWE trajectory has fewer than two non-system turns")
    initial_user = next(
        (message for message in body if message["role"] == "user"), None
    )
    if initial_user is None:
        raise ValueError("Open-SWE trajectory has no user query")

    mode = ("prefix", "interior", "interior", "tail")[window_index % 4]
    if mode == "prefix":
        spans = [(0, end) for end in range(min(len(body), 24), 1, -1)]
    elif mode == "tail":
        first = max(0, len(body) - 24)
        spans = [(start, len(body)) for start in range(first, len(body) - 1)]
    else:
        tool_turns = [
            index
            for index, message in enumerate(body)
            if message["role"] == "tool" or message.get("tool_calls")
        ]
        anchor = tool_turns[window_index % len(tool_turns)] if tool_turns else len(body) // 2
        spans = []
        for width in range(min(len(body), 24), 1, -1):
            start = min(max(0, anchor - width // 2), len(body) - width)
            spans.append((start, start + width))

    smallest_tokens: int | None = None
    template_errors: list[str] = []
    for start, end in spans:
        while start < end and body[start]["role"] == "tool":
            start += 1
        if end - start < 2:
            continue
        window = body[start:end]
        # Interior and tail spans commonly begin after the single original SWE
        # issue. Qwen's template requires that user query, so carry it into the
        # standalone calibration window before the selected agent trajectory.
        if window[0]["role"] != "user":
            window = [initial_user, *window]
        window_tools = tools_for_messages(all_tools, window)
        if not window_tools:
            continue
        # Prefer the original system turn. Agent scaffolds often have very long
        # system prompts, so fall back to the complete trajectory window while
        # retaining native Qwen tool schemas and call/result ordering.
        contexts = (system + window, window) if system else (window,)
        for context in contexts:
            compact_context, compact_tools = compact_open_swe_window(
                context, window_tools
            )
            for candidate_context, candidate_tools in (
                (context, window_tools),
                (compact_context, compact_tools),
            ):
                try:
                    text = processor.apply_chat_template(
                        candidate_context,
                        tools=candidate_tools,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    token_count = len(
                        processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                    )
                except Exception as error:
                    if len(template_errors) < 3:
                        template_errors.append(f"{type(error).__name__}: {error}")
                    continue
                smallest_tokens = (
                    token_count
                    if smallest_tokens is None
                    else min(smallest_tokens, token_count)
                )
                if 128 <= token_count <= MAX_LENGTH:
                    return text, True
    raise ValueError(
        f"no valid {mode} Open-SWE window fits {MAX_LENGTH} tokens; "
        f"smallest={smallest_tokens}; template_errors={template_errors}"
    )


def build_text_records(processor: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_index, (dataset_id, config, split, count, revision) in enumerate(
        TEXT_SOURCES
    ):
        expected = {
            "kind": "text",
            "source": dataset_id,
            "config": config,
            "split": split,
            "revision": revision,
        }
        cache_path = source_cache_path(
            "text", dataset_id, config, split, count, revision
        )
        cached = load_source_cache(cache_path, count, expected)
        if cached is not None:
            records.extend(cached)
            continue
        shuffle_buffer = max(
            TEXT_SHUFFLE_MIN,
            min(TEXT_SHUFFLE_MAX, count * TEXT_SHUFFLE_MULTIPLIER),
        )
        max_candidates = max(512, count * 20)
        print(
            f"source-build=start dataset={dataset_id} config={config} split={split} "
            f"target={count} shuffle_buffer={shuffle_buffer} "
            f"max_candidates={max_candidates}",
            flush=True,
        )
        stream = load_stream(
            dataset_id,
            config,
            split,
            revision,
            source_index,
            shuffle_buffer,
        )
        iterator = iter(stream)
        source_records: list[dict[str, Any]] = []
        accepted = 0
        scanned = 0
        skipped_open_think = 0
        think_open = processor.tokenizer.convert_tokens_to_ids("<think>")
        think_close = processor.tokenizer.convert_tokens_to_ids("</think>")
        errors: list[str] = []
        try:
            for row_index, row in enumerate(iterator):
                if row_index >= max_candidates:
                    break
                scanned = row_index + 1
                try:
                    if dataset_id == "nvidia/Open-SWE-Traces":
                        window_errors = []
                        for offset in range(4):
                            try:
                                text, has_tools = render_open_swe_window(
                                    processor, row, accepted + offset
                                )
                                break
                            except Exception as error:
                                window_errors.append(
                                    f"{type(error).__name__}: {error}"
                                )
                        else:
                            raise ValueError(
                                "all Open-SWE window modes failed: "
                                + " | ".join(window_errors)
                            )
                    else:
                        text, has_tools = render_row(processor, row)
                    tokens = processor.tokenizer(
                        text, truncation=True, max_length=MAX_LENGTH
                    )["input_ids"]
                    if len(tokens) < 32:
                        continue
                    # A trace whose reasoning does not close inside the window
                    # would reach the quantizer as an opening tag and an
                    # interior with no matching close, which is the asymmetry
                    # that biases a stop probability. Pass it over and draw
                    # another rather than let calibration trim it to a prompt:
                    # the rows this rejects are the longest chains, so trimming
                    # them would select against exactly what the window is for.
                    if not closes_in_window(
                        tokens, open_id=think_open, close_id=think_close
                    ):
                        skipped_open_think += 1
                        continue
                    source_records.append(
                        {
                            "kind": "text",
                            "text": text,
                            "source": dataset_id,
                            "config": config,
                            "split": split,
                            "revision": revision,
                            "has_tools": has_tools,
                            "has_reasoning": bool(
                                re.search(r"<think>\s*(?!</think>)\S", text)
                            ),
                            "has_qwen_functions": "<function=" in text,
                        }
                    )
                    accepted += 1
                    if accepted == count:
                        break
                except Exception as error:
                    if len(errors) < 3:
                        errors.append(f"{type(error).__name__}: {error}")
        finally:
            close_stream(iterator, stream)
        if accepted != count:
            raise RuntimeError(
                f"only rendered {accepted}/{count} from {dataset_id}:{config}:{split}; "
                f"scanned={scanned}; first errors={errors}"
            )
        write_source_cache(cache_path, source_records)
        records.extend(source_records)
        print(
            f"source-build=done dataset={dataset_id} config={config} split={split} "
            f"accepted={accepted} scanned={scanned} "
            f"skipped_open_think={skipped_open_think}",
            flush=True,
        )
    return records


def build_long_records(processor: Any) -> list[dict[str, Any]]:
    expected = {
        "kind": "text",
        "source": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "revision": FINEWEB_REVISION,
    }
    cache_path = source_cache_path(
        "long",
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "train",
        LONG_COUNT,
        FINEWEB_REVISION,
    )
    cached = load_source_cache(cache_path, LONG_COUNT, expected)
    if cached is not None:
        return cached
    print(
        f"source-build=start dataset=HuggingFaceFW/fineweb-edu "
        f"config=sample-10BT split=train target={LONG_COUNT} "
        f"shuffle_buffer={LONG_SHUFFLE_BUFFER}",
        flush=True,
    )
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        revision=FINEWEB_REVISION,
        streaming=True,
    ).shuffle(seed=SEED + 100, buffer_size=LONG_SHUFFLE_BUFFER)
    iterator = iter(stream)
    records: list[dict[str, Any]] = []
    chunks: list[str] = []
    try:
        for row in iterator:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            chunks.append(text)
            joined = "\n\n".join(chunks)
            messages = [
                {
                    "role": "user",
                    "content": "Read the following material carefully.\n\n" + joined,
                }
            ]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_count = len(
                processor.tokenizer(
                    rendered, truncation=True, max_length=MAX_LENGTH
                )["input_ids"]
            )
            if token_count < LONG_MIN_TOKENS:
                continue
            records.append(
                {
                    "kind": "text",
                    "text": rendered,
                    "source": "HuggingFaceFW/fineweb-edu",
                    "config": "sample-10BT",
                    "split": "train",
                    "revision": FINEWEB_REVISION,
                    "has_tools": False,
                    "has_reasoning": False,
                    "has_qwen_functions": False,
                    "token_count": token_count,
                }
            )
            chunks = []
            if len(records) == LONG_COUNT:
                break
    finally:
        close_stream(iterator, stream)
    if len(records) != LONG_COUNT:
        raise RuntimeError(f"only built {len(records)}/{LONG_COUNT} long records")
    write_source_cache(cache_path, records)
    print(
        f"source-build=done dataset=HuggingFaceFW/fineweb-edu "
        f"config=sample-10BT split=train accepted={len(records)}",
        flush=True,
    )
    return records


def build_vision_records() -> list[dict[str, Any]]:
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source_index, (config, count) in enumerate(VISION_SOURCES):
        expected = {
            "kind": "vision",
            "source": "HuggingFaceM4/the_cauldron",
            "config": config,
            "split": "train",
            "revision": CAULDRON_REVISION,
        }
        cache_path = source_cache_path(
            "vision",
            "HuggingFaceM4/the_cauldron",
            config,
            "train",
            count,
            CAULDRON_REVISION,
        )
        cached = load_source_cache(cache_path, count, expected)
        if cached is not None:
            records.extend(cached)
            continue
        shuffle_buffer = max(VISION_SHUFFLE_MIN, count * VISION_SHUFFLE_MULTIPLIER)
        print(
            f"source-build=start dataset=HuggingFaceM4/the_cauldron "
            f"config={config} split=train target={count} "
            f"shuffle_buffer={shuffle_buffer}",
            flush=True,
        )
        stream = load_dataset(
            "HuggingFaceM4/the_cauldron",
            config,
            split="train",
            revision=CAULDRON_REVISION,
            streaming=True,
        ).shuffle(seed=SEED + 200 + source_index, buffer_size=shuffle_buffer)
        iterator = iter(stream)
        source_records: list[dict[str, Any]] = []
        accepted = 0
        scanned = 0
        try:
            for row_index, row in enumerate(iterator):
                if row_index >= count * 30:
                    break
                scanned = row_index + 1
                images = row.get("images") or []
                texts = row.get("texts") or []
                if len(images) != 1 or not texts:
                    continue
                turn = texts[0]
                user = str(turn.get("user", "")).strip()
                assistant = str(turn.get("assistant", "")).strip()
                if not user or not assistant:
                    continue
                relative_path = Path(config) / f"{accepted:04d}.jpg"
                destination = IMAGE_ROOT / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                image = images[0].convert("RGB")
                image.thumbnail((1536, 1536))
                image.save(destination, format="JPEG", quality=92)
                source_records.append(
                    {
                        "kind": "vision",
                        "image": str(Path("images") / relative_path),
                        "user": user,
                        "assistant": assistant,
                        "source": "HuggingFaceM4/the_cauldron",
                        "config": config,
                        "split": "train",
                        "revision": CAULDRON_REVISION,
                        "has_tools": False,
                        "has_reasoning": False,
                        "has_qwen_functions": False,
                    }
                )
                accepted += 1
                if accepted == count:
                    break
        finally:
            close_stream(iterator, stream)
        if accepted != count:
            raise RuntimeError(f"only built {accepted}/{count} images from {config}")
        write_source_cache(cache_path, source_records)
        records.extend(source_records)
        print(
            f"source-build=done dataset=HuggingFaceM4/the_cauldron "
            f"config={config} split=train accepted={accepted} scanned={scanned}",
            flush=True,
        )
    return records


def validate_manifest() -> str:
    expected = Counter(
        {
            (dataset_id, config, split, revision): count
            for dataset_id, config, split, count, revision in TEXT_SOURCES
        }
    )
    expected[
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", FINEWEB_REVISION)
    ] = LONG_COUNT
    for config, count in VISION_SOURCES:
        expected[
            ("HuggingFaceM4/the_cauldron", config, "train", CAULDRON_REVISION)
        ] = count
    actual: Counter[tuple[str, str | None, str, str]] = Counter()
    digest = hashlib.sha256()
    rows = 0
    reasoning_rows = 0
    function_rows = 0
    long_rows = 0
    with MANIFEST.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            record = json.loads(line)
            actual[
                (
                    record["source"],
                    record.get("config"),
                    record["split"],
                    record["revision"],
                )
            ] += 1
            reasoning_rows += bool(record.get("has_reasoning"))
            function_rows += bool(record.get("has_qwen_functions"))
            long_rows += int(record.get("token_count", 0)) >= LONG_MIN_TOKENS
            if record["kind"] == "text":
                if not record.get("text", "").strip():
                    raise ValueError(f"empty text at line {line_number}")
            elif record["kind"] == "vision":
                image_path = ROOT / record["image"]
                if not image_path.is_file() or image_path.stat().st_size == 0:
                    raise ValueError(f"missing image at line {line_number}: {image_path}")
            else:
                raise ValueError(f"invalid kind at line {line_number}")
            rows += 1
    if rows != TOTAL or actual != expected:
        raise ValueError(f"manifest mismatch: rows={rows}; actual={actual}; expected={expected}")
    if reasoning_rows < 32:
        raise ValueError(f"reasoning gate failed: {reasoning_rows}/32")
    if function_rows < 32:
        raise ValueError(f"Qwen function gate failed: {function_rows}/32")
    if long_rows != LONG_COUNT:
        raise ValueError(f"long-context gate failed: {long_rows}/{LONG_COUNT}")
    return (
        f"rows={rows}; sha256={digest.hexdigest()}; reasoning_rows={reasoning_rows}; "
        f"function_rows={function_rows}; long_rows={long_rows}"
    )


def main() -> None:
    if MANIFEST.exists() and not FORCE_REBUILD:
        try:
            result = validate_manifest()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"calibration-cache=invalid; rebuilding; reason={error}")
        else:
            print(f"calibration-cache=valid; {result}")
            cached_records = [
                json.loads(line) for line in MANIFEST.read_text().splitlines()
            ]
            backfill_source_caches(cached_records)
            return
    ROOT.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    records = build_text_records(processor)
    records.extend(build_long_records(processor))
    records.extend(build_vision_records())
    random.Random(SEED).shuffle(records)
    temporary = MANIFEST.with_name(f".{MANIFEST.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote calibration: {validate_manifest()}")


if __name__ == "__main__":
    main()
    # datasets/fsspec can leave native HTTP worker threads alive after streamed
    # Parquet iteration. On this cluster those workers abort CPython during
    # PyGILState finalization even after the atomically written manifest has
    # been validated. Exit directly only on success; exceptions still unwind
    # normally and return non-zero.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
