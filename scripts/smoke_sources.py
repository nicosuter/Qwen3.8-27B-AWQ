#!/usr/bin/env python3
"""Render representative public rows through the pinned Qwen template."""

from transformers import AutoProcessor

from build_calibration import (
    LAMBDA_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    NEMOTRON_REVISION,
    OPEN_SWE_REVISION,
    close_stream,
    load_stream,
    render_open_swe_window,
    render_row,
)

SOURCES = (
    ("nvidia/Open-SWE-Traces", "openhands", "qwen35_122b", OPEN_SWE_REVISION, True),
    ("lambda/hermes-agent-reasoning-traces", "kimi", "train", LAMBDA_REVISION, True),
    ("lambda/hermes-agent-reasoning-traces", "glm-5.1", "train", LAMBDA_REVISION, True),
    ("nvidia/Nemotron-Post-Training-Dataset-v1", None, "tool_calling", NEMOTRON_REVISION, True),
)
SMOKE_SHUFFLE_BUFFER = 64

processor = AutoProcessor.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
)
for source_index, (dataset_id, config, split, revision, requires_functions) in enumerate(SOURCES):
    stream = load_stream(
        dataset_id,
        config,
        split,
        revision,
        900 + source_index,
        SMOKE_SHUFFLE_BUFFER,
    )
    iterator = iter(stream)
    first_errors = []
    text = ""
    try:
        for row_index, row in enumerate(iterator):
            if row_index >= 50:
                break
            try:
                if dataset_id == "nvidia/Open-SWE-Traces":
                    # Tail mode previously omitted the original user issue and
                    # deadlocked production construction at accepted row 3.
                    text, has_tools = render_open_swe_window(processor, row, 3)
                else:
                    text, has_tools = render_row(processor, row)
                if not text.strip():
                    raise ValueError("empty render")
                if requires_functions and (not has_tools or "<function=" not in text):
                    raise ValueError("Qwen function schema missing")
                break
            except Exception as error:
                if len(first_errors) < 3:
                    first_errors.append(f"{type(error).__name__}: {error}")
        if not text:
            raise RuntimeError(
                f"no valid row from {dataset_id}:{config}:{split}; errors={first_errors}"
            )
    finally:
        close_stream(iterator, stream)
    print(
        f"source-render=ok dataset={dataset_id} config={config} "
        f"chars={len(text)} qwen_functions={'<function=' in text}"
    )
