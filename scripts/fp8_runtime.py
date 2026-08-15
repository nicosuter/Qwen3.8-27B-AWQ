"""Runtime selection for the Transformers FP8 correctness baseline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def configure_triton_fp8_baseline(
    model: Any,
    module_types: tuple[type, ...] | None = None,
) -> int:
    """Route all fine-grained FP8 modules through the Triton implementation."""
    if module_types is None:
        from transformers.integrations.finegrained_fp8 import FP8Experts, FP8Linear

        module_types = (FP8Linear, FP8Experts)

    modules: Iterable[Any] = model.modules()
    fp8_modules = [module for module in modules if isinstance(module, module_types)]
    if not fp8_modules:
        raise RuntimeError("FP8 baseline contains no fine-grained FP8 modules")

    for module in fp8_modules:
        module._deepgemm_disabled = True
    return len(fp8_modules)
