#!/usr/bin/env python3
"""Cheap compatibility check before loading 27B of weights."""

import os
from accelerate import init_empty_weights
from llmcompressor.modifiers.transform.awq.dynamic_mappings import (
    get_layer_mappings_from_model,
)
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.8-27B")
model_revision = os.environ.get(
    "MODEL_REVISION", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
config = AutoConfig.from_pretrained(
    model_id, revision=model_revision, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    model_id, revision=model_revision, trust_remote_code=True
)
with init_empty_weights():
    model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)
mappings = get_layer_mappings_from_model(model)
vision_parameters = [
    name
    for name, _ in model.named_parameters()
    if "visual" in name.lower() or "vision" in name.lower()
]
architecture = (getattr(config, "architectures", None) or [type(config).__name__])[0]
assert hasattr(processor, "apply_chat_template"), "processor has no chat template"
assert hasattr(processor, "tokenizer"), "processor has no tokenizer"
assert any("linear_attn" in str(mapping) for mapping in mappings), (
    "AWQ hybrid mappings missing linear_attn; refusing default mappings"
)
assert vision_parameters, "multimodal model exposed no visual/vision parameters"
print(f"model={model_id}")
print(f"revision={model_revision}")
print(f"architecture={architecture}")
print(f"model_type={config.model_type}")
print(f"awq_mappings={len(mappings)}")
print(f"vision_parameters={len(vision_parameters)}")
print(f"vision_parameter_example={vision_parameters[0]}")
print("preflight=ok")
