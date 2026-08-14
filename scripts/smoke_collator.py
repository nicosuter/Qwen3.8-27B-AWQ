#!/usr/bin/env python3
"""No-weight smoke for text/vision preprocessing and collation."""

import tempfile
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

import quantize

processor = AutoProcessor.from_pretrained(
    quantize.MODEL_ID,
    revision=quantize.MODEL_REVISION,
    trust_remote_code=True,
)
with tempfile.TemporaryDirectory(prefix="qwen38-collator-") as directory:
    root = Path(directory)
    image_path = root / "sample.jpg"
    Image.new("RGB", (256, 192), color=(30, 120, 210)).save(image_path)
    quantize.CALIBRATION_DIR = root
    records = [
        {"kind": "text", "text": "A short text calibration record."},
        {
            "kind": "vision",
            "image": "sample.jpg",
            "user": "What is the dominant color?",
            "assistant": "Blue.",
        },
    ]
    dataset = quantize.RankCalibrationDataset(records, processor, 0, 1)
    text_batch = quantize.single_item_collator([dataset[0]])
    vision_batch = quantize.single_item_collator([dataset[1]])
    assert "input_ids" in text_batch and text_batch["input_ids"].ndim == 2
    assert "input_ids" in vision_batch and vision_batch["input_ids"].ndim == 2
    assert "pixel_values" in vision_batch and vision_batch["pixel_values"].ndim >= 2
    assert "image_grid_thw" in vision_batch
    print(
        "collator=ok "
        f"text_tokens={text_batch['input_ids'].shape[-1]} "
        f"vision_tokens={vision_batch['input_ids'].shape[-1]} "
        f"pixels={tuple(vision_batch['pixel_values'].shape)}"
    )
