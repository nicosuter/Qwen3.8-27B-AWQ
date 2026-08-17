#!/usr/bin/env python3
"""Preserve Qwen's source-precision MTP head in a Transformers export.

Qwen3.5-family checkpoints store their speculative head under top-level
``mtp.*`` keys.  Transformers' ``Qwen3_5ForConditionalGeneration`` does not
instantiate that inference-only module, so a normal ``save_pretrained`` drops
those weights even when the quantization recipe ignores ``mtp``.  vLLM loads
the same keys separately when native MTP is enabled.

This module copies only those tensors from the pinned source checkpoint into a
dedicated shard and atomically adds them to the exported safetensors index.

Shipping the tensors is necessary but not sufficient.  vLLM decides whether a
layer is quantized when it *constructs* it, from the layer's prefix string:
``CompressedTensorsConfig.get_quant_method`` asks ``should_ignore_layer`` against
``quantization_config.ignore``.  The MTP projections are ordinary ``Linear``
modules, so unless they are named in that list vLLM builds them with packed
int4 parameters and the grafted BF16 ``mtp.*.weight`` has nowhere to land.  This
module therefore repairs ``config.json`` as well.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

MTP_PREFIX = "mtp."
MTP_SHARD = "model-mtp.safetensors"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
SOURCE_DTYPE = "BF16"
WEIGHT_SUFFIX = ".weight"
QWEN38_MTP_SHAPES = {
    "mtp.fc.weight": (5120, 10240),
    "mtp.layers.0.input_layernorm.weight": (5120,),
    "mtp.layers.0.mlp.down_proj.weight": (5120, 17408),
    "mtp.layers.0.mlp.gate_proj.weight": (17408, 5120),
    "mtp.layers.0.mlp.up_proj.weight": (17408, 5120),
    "mtp.layers.0.post_attention_layernorm.weight": (5120,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (1024, 5120),
    "mtp.layers.0.self_attn.o_proj.weight": (5120, 6144),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.q_proj.weight": (12288, 5120),
    "mtp.layers.0.self_attn.v_proj.weight": (1024, 5120),
    "mtp.norm.weight": (5120,),
    "mtp.pre_fc_norm_embedding.weight": (5120,),
    "mtp.pre_fc_norm_hidden.weight": (5120,),
}
QWEN38_MTP_KEYS = frozenset(QWEN38_MTP_SHAPES)
QWEN38_MTP_LINEAR_MODULES = frozenset(
    {
        "mtp.fc",
        "mtp.layers.0.mlp.down_proj",
        "mtp.layers.0.mlp.gate_proj",
        "mtp.layers.0.mlp.up_proj",
        "mtp.layers.0.self_attn.k_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.self_attn.v_proj",
    }
)


def mtp_weight_map(index: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact top-level MTP mapping from a safetensors index."""
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index has no weight_map object")
    result = {
        name: shard
        for name, shard in weight_map.items()
        if isinstance(name, str)
        and name.startswith(MTP_PREFIX)
        and isinstance(shard, str)
    }
    if not result:
        raise ValueError("checkpoint index contains no top-level mtp.* tensors")
    return result


def packed_weight_count(index: Mapping[str, Any]) -> int:
    """Count compressed-tensors packed weights in an exported index."""
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index has no weight_map object")
    return sum(name.endswith(".weight_packed") for name in weight_map)


def require_packed_export(index: Mapping[str, Any]) -> int:
    """Reject a checkpoint that silently serialized its BF16 snapshot."""
    count = packed_weight_count(index)
    if count == 0:
        raise RuntimeError(
            "checkpoint contains no weight_packed tensors; compressed export failed"
        )
    return count


def merged_index(
    output_index: Mapping[str, Any],
    mtp_names: set[str],
    *,
    mtp_numel: int,
    mtp_nbytes: int,
) -> dict[str, Any]:
    """Return an output index extended with one source-precision MTP shard."""
    result = json.loads(json.dumps(output_index))
    weight_map = result.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("output safetensors index has no weight_map object")
    existing = {name for name in weight_map if name.startswith(MTP_PREFIX)}
    if existing:
        raise ValueError(
            f"output index already contains MTP tensors: {sorted(existing)}"
        )
    for name in sorted(mtp_names):
        weight_map[name] = MTP_SHARD

    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("output safetensors index metadata is not an object")
    metadata["total_size"] = int(metadata.get("total_size", 0)) + mtp_nbytes
    if "total_parameters" in metadata:
        metadata["total_parameters"] = int(metadata["total_parameters"]) + mtp_numel
    return result


def mtp_linear_modules(shapes: Mapping[str, Sequence[int]]) -> list[str]:
    """Return the MTP module names vLLM will construct as quantizable Linears.

    Tensor rank is the discriminator: Qwen ships the MTP head as 2-D projection
    weights plus 1-D RMSNorm weights, and only the projections become ``Linear``
    modules. Deriving the list beats hardcoding eight names, which would quietly
    under-cover a revision that adds a second MTP layer or renames a projection.

    The names stay in the checkpoint's ``mtp.*`` namespace on purpose: vLLM
    builds the head under ``prefix="mtp"`` and only rewrites ``mtp.`` to
    ``model.`` later, while loading weights, so the ignore list has to match the
    construction-time prefix rather than the runtime attribute path.
    """
    modules: list[str] = []
    for name in sorted(shapes):
        if not name.startswith(MTP_PREFIX):
            raise ValueError(f"not a top-level MTP tensor: {name}")
        if len(shapes[name]) != 2:
            continue
        if not name.endswith(WEIGHT_SUFFIX):
            raise ValueError(f"2-D MTP tensor is not a module weight: {name}")
        modules.append(name[: -len(WEIGHT_SUFFIX)])
    if not modules:
        raise ValueError(
            "no 2-D MTP tensors; cannot derive the quantization ignore list"
        )
    return modules


def merged_ignore(
    config: Mapping[str, Any], modules: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    """Return a config whose ignore list also excludes the MTP projections.

    Every projection must be listed. vLLM maps its fused ``qkv_proj`` and
    ``gate_up_proj`` back through ``packed_modules_mapping`` and raises if the
    shards disagree about being ignored, so a partial list is worse than none.
    """
    result = json.loads(json.dumps(config))
    quantization_config = result.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("config.json has no quantization_config object")
    ignore = quantization_config.get("ignore")
    if not isinstance(ignore, list) or any(
        not isinstance(item, str) for item in ignore
    ):
        raise ValueError("quantization_config.ignore is not a string list")
    added = [module for module in modules if module not in ignore]
    quantization_config["ignore"] = list(ignore) + added
    return result, added


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Return the tensor entries of a safetensors file without loading tensors.

    The container is a little-endian ``uint64`` header length followed by that
    many bytes of JSON, so shape and dtype are readable with the standard
    library alone.  That keeps the ``config.json`` repair usable on a checkpoint
    without importing torch.
    """
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        length = int.from_bytes(length_bytes, "little")
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    entries = {
        name: entry
        for name, entry in header.items()
        if name != "__metadata__" and isinstance(entry, dict)
    }
    data_size = path.stat().st_size - 8 - length
    for name, entry in entries.items():
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
        ):
            raise ValueError(f"{path}: invalid data offsets for {name}")
    return entries


def validate_mtp_artifact(
    output_dir: Path,
    *,
    expected_keys: frozenset[str] | None = None,
    expected_modules: frozenset[str] | None = None,
    expected_shapes: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Validate the complete compressed-checkpoint/MTP serving contract.

    This deliberately uses the raw index, safetensors headers, and config. A
    Transformers reload cannot validate the top-level MTP module because the
    architecture intentionally does not instantiate it. The checks here are
    also dependency-free, so release and evaluation launchers can fail before
    importing torch or starting vLLM.
    """
    output_dir = output_dir.resolve()
    index_path = output_dir / INDEX_NAME
    config_path = output_dir / CONFIG_NAME
    if not index_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"incomplete checkpoint at {output_dir}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    packed_weights = require_packed_export(index)
    mapping = mtp_weight_map(index)
    if expected_keys is not None and set(mapping) != expected_keys:
        raise RuntimeError(
            "checkpoint MTP keyset differs from the pinned source: "
            f"missing={sorted(expected_keys - set(mapping))} "
            f"unexpected={sorted(set(mapping) - expected_keys)}"
        )

    entries: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(mapping.values())):
        shard_path = output_dir / shard
        if not shard_path.is_file():
            raise RuntimeError(f"MTP index references missing shard: {shard}")
        header = read_safetensors_header(shard_path)
        expected = {name for name, filename in mapping.items() if filename == shard}
        missing = expected - set(header)
        if missing:
            raise RuntimeError(
                f"MTP tensors absent from indexed shard {shard}: {sorted(missing)}"
            )
        entries.update({name: header[name] for name in expected})

    wrong_dtype = sorted(
        name for name, entry in entries.items() if entry.get("dtype") != SOURCE_DTYPE
    )
    if wrong_dtype:
        raise RuntimeError(f"MTP tensors are not {SOURCE_DTYPE}: {wrong_dtype}")
    shapes: dict[str, tuple[int, ...]] = {}
    for name, entry in entries.items():
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or dimension < 0 for dimension in shape
        ):
            raise RuntimeError(f"MTP tensor has an invalid shape: {name}={shape!r}")
        shapes[name] = tuple(shape)
        elements = 1
        for dimension in shape:
            elements *= dimension
        start, end = entry["data_offsets"]
        if end - start != elements * 2:
            raise RuntimeError(
                f"MTP tensor byte span does not match BF16 shape: {name}"
            )
    if expected_shapes is not None:
        normalized_expected = {
            name: tuple(shape) for name, shape in expected_shapes.items()
        }
        if shapes != normalized_expected:
            mismatched = sorted(
                name
                for name in set(shapes) | set(normalized_expected)
                if shapes.get(name) != normalized_expected.get(name)
            )
            raise RuntimeError(
                "checkpoint MTP shapes differ from the pinned source: "
                f"{[(name, shapes.get(name), normalized_expected.get(name)) for name in mismatched]}"
            )
    modules = mtp_linear_modules(shapes)
    if expected_modules is not None and set(modules) != expected_modules:
        raise RuntimeError(
            "checkpoint MTP Linear set differs from the pinned architecture: "
            f"missing={sorted(expected_modules - set(modules))} "
            f"unexpected={sorted(set(modules) - expected_modules)}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise RuntimeError("config.json has no quantization_config object")
    ignore = quantization_config.get("ignore")
    if not isinstance(ignore, list) or any(
        not isinstance(item, str) for item in ignore
    ):
        raise RuntimeError("quantization_config.ignore is not a string list")
    missing_ignores = sorted(set(modules) - set(ignore))
    if missing_ignores:
        raise RuntimeError(
            "MTP BF16 projections are absent from quantization_config.ignore: "
            f"{missing_ignores}"
        )

    text_config = config.get("text_config")
    try:
        mtp_layers = int(text_config.get("mtp_num_hidden_layers", 0))
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("config.json has an invalid MTP layer count") from error
    if mtp_layers <= 0:
        raise RuntimeError("config.json no longer advertises an MTP head")
    return {
        "packed_weights": packed_weights,
        "mtp_parameters": len(mapping),
        "mtp_dtype": SOURCE_DTYPE,
        "mtp_shards": sorted(set(mapping.values())),
        "mtp_ignored_modules": modules,
    }


def repair_mtp_ignores(output_dir: Path) -> dict[str, Any]:
    """Add the MTP Linear exclusions to a checkpoint that already has the shard.

    An export grafted before the ``config.json`` half of this repair existed has
    all fifteen tensors and still cannot serve: vLLM builds the projections as
    packed int4 and rejects the BF16 ``.weight``.  ``preserve_mtp_weights``
    refuses to touch such a checkpoint, by design, so recovery needs its own
    door.  Shapes come from the shard's own header, which keeps this runnable
    without torch.
    """
    output_dir = output_dir.resolve()
    index = json.loads((output_dir / INDEX_NAME).read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("output safetensors index has no weight_map object")
    indexed = {name for name in weight_map if name.startswith(MTP_PREFIX)}
    if not indexed:
        raise RuntimeError(
            "output index has no mtp.* tensors; graft them before repairing config"
        )

    entries: dict[str, dict[str, Any]] = {}
    for shard in sorted({weight_map[name] for name in indexed}):
        for name, entry in read_safetensors_header(output_dir / shard).items():
            if name in indexed:
                entries[name] = entry
    if set(entries) != indexed:
        raise RuntimeError(
            "indexed MTP tensors are absent from their shards: "
            f"{sorted(indexed - set(entries))}"
        )
    wrong_dtype = sorted(
        name for name, entry in entries.items() if entry.get("dtype") != SOURCE_DTYPE
    )
    if wrong_dtype:
        raise RuntimeError(f"MTP tensors are not {SOURCE_DTYPE}: {wrong_dtype}")

    shapes = {name: tuple(entry.get("shape", ())) for name, entry in entries.items()}
    if set(shapes) != QWEN38_MTP_KEYS:
        raise RuntimeError(
            "checkpoint MTP keyset differs from the pinned source: "
            f"missing={sorted(QWEN38_MTP_KEYS - set(shapes))} "
            f"unexpected={sorted(set(shapes) - QWEN38_MTP_KEYS)}"
        )
    if shapes != QWEN38_MTP_SHAPES:
        mismatched = sorted(
            name
            for name in QWEN38_MTP_KEYS
            if shapes.get(name) != QWEN38_MTP_SHAPES.get(name)
        )
        raise RuntimeError(
            "checkpoint MTP shapes differ from the pinned source: "
            f"{[(name, shapes.get(name), QWEN38_MTP_SHAPES.get(name)) for name in mismatched]}"
        )
    modules = mtp_linear_modules(shapes)
    if set(modules) != QWEN38_MTP_LINEAR_MODULES:
        raise RuntimeError("checkpoint MTP Linear set differs from pinned architecture")
    config_path = output_dir / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    updated_config, added = merged_ignore(config, modules)
    if added:
        _atomic_json(config_path, updated_config)
    validated = validate_mtp_artifact(
        output_dir,
        expected_keys=QWEN38_MTP_KEYS,
        expected_modules=QWEN38_MTP_LINEAR_MODULES,
        expected_shapes=QWEN38_MTP_SHAPES,
    )
    return {**validated, "mtp_ignores_added": added}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _default_download(repo_id: str, revision: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, revision=revision, filename=filename)


def preserve_mtp_weights(
    repo_id: str,
    revision: str,
    output_dir: Path,
    *,
    download: Callable[[str, str, str], str] = _default_download,
) -> dict[str, Any]:
    """Copy BF16 ``mtp.*`` tensors from a pinned source into ``output_dir``.

    The output index is replaced only after the new shard has been completely
    written and reopened successfully.  Existing partial or duplicate MTP
    mappings are rejected instead of silently masking a damaged export.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    output_dir = output_dir.resolve()
    output_index_path = output_dir / INDEX_NAME
    output_index = json.loads(output_index_path.read_text(encoding="utf-8"))
    packed_weights = require_packed_export(output_index)
    source_index_path = Path(download(repo_id, revision, INDEX_NAME))
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    source_mtp = mtp_weight_map(source_index)

    output_map = output_index.get("weight_map", {})
    existing = {name for name in output_map if name.startswith(MTP_PREFIX)}
    if existing:
        if existing != set(source_mtp):
            raise RuntimeError(
                "output contains an incomplete or foreign MTP keyset: "
                f"output={len(existing)} source={len(source_mtp)}"
            )
        raise RuntimeError(
            "output index already contains the complete MTP keyset; refusing "
            "to overwrite it"
        )

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in source_mtp.items():
        by_shard[shard].append(name)

    tensors: dict[str, torch.Tensor] = {}
    for shard, names in sorted(by_shard.items()):
        shard_path = download(repo_id, revision, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            missing = set(names) - set(source.keys())
            if missing:
                raise RuntimeError(
                    f"source shard {shard} is missing indexed MTP tensors: "
                    f"{sorted(missing)}"
                )
            for name in sorted(names):
                tensor = source.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"source MTP tensor {name} is {tensor.dtype}, expected BF16"
                    )
                tensors[name] = tensor.clone()

    if set(tensors) != set(source_mtp):
        raise RuntimeError("loaded MTP keyset does not match the pinned source index")
    mtp_numel = sum(tensor.numel() for tensor in tensors.values())
    mtp_nbytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    ignored_modules = mtp_linear_modules(
        {name: tuple(tensor.shape) for name, tensor in tensors.items()}
    )

    final_shard = output_dir / MTP_SHARD
    temporary_shard = output_dir / f".{MTP_SHARD}.tmp"
    temporary_shard.unlink(missing_ok=True)
    try:
        save_file(
            tensors,
            temporary_shard,
            metadata={
                "format": "pt",
                "source_model": repo_id,
                "source_revision": revision,
            },
        )
        with safe_open(temporary_shard, framework="pt", device="cpu") as saved:
            if set(saved.keys()) != set(source_mtp):
                raise RuntimeError("written MTP shard has the wrong tensor keyset")
            for name, source_tensor in tensors.items():
                saved_tensor = saved.get_tensor(name)
                if saved_tensor.dtype != torch.bfloat16:
                    raise RuntimeError(f"written MTP tensor {name} is not BF16")
                if not torch.equal(saved_tensor, source_tensor):
                    raise RuntimeError(f"written MTP tensor {name} differs from source")
        os.replace(temporary_shard, final_shard)
    finally:
        temporary_shard.unlink(missing_ok=True)

    # vLLM constructs quantized modules from config before loading any weights.
    # Mark every 2-D MTP tensor's module as unquantized before exposing the new
    # shard in the index; otherwise vLLM creates weight_packed parameters and
    # rejects the source BF16 `.weight` tensors.
    config_path = output_dir / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    updated_config, _ = merged_ignore(config, ignored_modules)
    _atomic_json(config_path, updated_config)

    updated_index = merged_index(
        output_index,
        set(source_mtp),
        mtp_numel=mtp_numel,
        mtp_nbytes=mtp_nbytes,
    )
    _atomic_json(output_index_path, updated_index)
    validated = validate_mtp_artifact(output_dir)
    if validated["mtp_parameters"] != len(source_mtp):
        raise RuntimeError("final MTP artifact validation changed the tensor count")
    if set(validated["mtp_ignored_modules"]) != set(ignored_modules):
        raise RuntimeError("final MTP artifact validation changed the ignore set")
    return {
        "packed_weights": packed_weights,
        "mtp_parameters": len(source_mtp),
        "mtp_numel": mtp_numel,
        "mtp_nbytes": mtp_nbytes,
        "mtp_dtype": "torch.bfloat16",
        "mtp_shard": MTP_SHARD,
        "mtp_ignored_modules": ignored_modules,
        "source_shards": sorted(by_shard),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the pinned BF16 MTP head into an existing AWQ export"
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument(
        "--ignore-only",
        action="store_true",
        help=(
            "skip the weight graft and only add the MTP Linear exclusions to "
            "config.json; for an export grafted before that half existed"
        ),
    )
    args = parser.parse_args()
    if args.ignore_only:
        result = repair_mtp_ignores(args.output_dir)
    else:
        if not args.model_id or not args.revision:
            parser.error("--model-id and --revision are required without --ignore-only")
        result = preserve_mtp_weights(
            args.model_id,
            args.revision,
            args.output_dir,
        )
    metadata_path = args.output_dir.resolve() / "run-metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"mtp_preserved": True, **result})
        _atomic_json(metadata_path, metadata)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
