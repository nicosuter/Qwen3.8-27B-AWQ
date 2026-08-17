#!/usr/bin/env python3
"""Reload the checkpoint and enforce the sub-20-minute release smoke gates."""

import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from preserve_mtp import (
    QWEN38_MTP_KEYS,
    QWEN38_MTP_LINEAR_MODULES,
    QWEN38_MTP_SHAPES,
    validate_mtp_artifact,
)

MODEL = Path(os.environ.get("OUTPUT_DIR", "artifacts/Qwen3.8-27B-AWQ")).resolve()
STARTED = time.monotonic()
DEADLINE_SECONDS = int(os.environ.get("SMOKE_DEADLINE_SECONDS", "1200"))
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}

artifact = validate_mtp_artifact(
    MODEL,
    expected_keys=QWEN38_MTP_KEYS,
    expected_modules=QWEN38_MTP_LINEAR_MODULES,
    expected_shapes=QWEN38_MTP_SHAPES,
)
print(
    f"artifact-index=ok packed_weights={artifact['packed_weights']} "
    f"mtp_parameters={artifact['mtp_parameters']} mtp_dtype=torch.bfloat16 "
    f"mtp_ignored_modules={len(artifact['mtp_ignored_modules'])}"
)

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype="auto", device_map="cuda:0", trust_remote_code=True
)
model.eval()
vision_parameters = [
    (name, parameter)
    for name, parameter in model.named_parameters()
    if "visual" in name.lower() or "vision" in name.lower()
]
assert vision_parameters, "saved checkpoint has no vision parameters"
bad_vision_dtypes = [
    (name, str(parameter.dtype))
    for name, parameter in vision_parameters
    if parameter.dtype not in (torch.float16, torch.bfloat16)
]
assert not bad_vision_dtypes, f"vision parameters were quantized: {bad_vision_dtypes[:5]}"
mtp_layers = int(getattr(model.config.text_config, "mtp_num_hidden_layers", 0))
assert mtp_layers > 0, "saved checkpoint config no longer advertises MTP"
print(
    f"artifact-dtypes=ok vision_parameters={len(vision_parameters)} "
    f"mtp_parameters={artifact['mtp_parameters']} mtp_layers={mtp_layers}"
)


def generate(prompt: str, *, tools=None, max_new_tokens: int = 96) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        reasoning_effort="low",
    )
    inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
    )


prompt = "Solve 3x^2 - 10x + 3 = 0. Give both roots in one sentence."
completion = generate(prompt)
assert completion.strip(), "empty text completion"
assert "3" in completion and ("1/3" in completion or "0.333" in completion), (
    f"text arithmetic smoke failed: {completion}"
)
print(f"TEXT OUTPUT: {completion}\n")

tool_completion = generate(
    "Use the weather tool to get the weather in Zurich in Celsius.",
    tools=[WEATHER_TOOL],
    max_new_tokens=128,
)
assert "<tool_call>" in tool_completion, "model did not emit a structured tool call"
assert "get_weather" in tool_completion, "model called the wrong tool"
print(f"TOOL OUTPUT: {tool_completion}\n")

needle = "The deployment verification code is HELVETIA-38027."
long_context = ("Filler context about systems and mathematics. " * 500) + needle
long_completion = generate(
    f"{long_context}\nWhat is the deployment verification code? Answer with only the code.",
    max_new_tokens=64,
)
assert "HELVETIA-38027" in long_completion, "long-context retrieval failed"
print(f"LONG-CONTEXT OUTPUT: {long_completion}\n")

image = Image.new("RGB", (256, 256), color=(220, 20, 20))
image_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is the dominant color? Answer with one word."},
        ],
    }
]
image_text = processor.apply_chat_template(
    image_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
    reasoning_effort="low",
)
image_inputs = processor(text=[image_text], images=[image], return_tensors="pt").to(
    model.device
)
with torch.inference_mode():
    image_output = model.generate(**image_inputs, max_new_tokens=32, do_sample=False)
image_completion = processor.tokenizer.decode(
    image_output[0, image_inputs["input_ids"].shape[1] :], skip_special_tokens=False
)
assert "red" in image_completion.lower(), f"vision smoke failed: {image_completion}"
print(f"VISION OUTPUT: {image_completion}\n")
elapsed = time.monotonic() - STARTED
assert elapsed <= DEADLINE_SECONDS, (
    f"smoke exceeded deadline: {elapsed:.1f}s > {DEADLINE_SECONDS}s"
)
print(f"release-smoke=ok elapsed_seconds={elapsed:.1f}")
