#!/usr/bin/env python3
"""Public multimodal/long-context W4A16 AWQ recipe for Qwen3.8-27B."""

import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from compressed_tensors.offload import OffloadCache, init_dist, to_accelerate
from compressed_tensors.quantization import preset_name_to_scheme
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from distributed_lifecycle import run_rank0_after_group_teardown
from gdn_in_proj import (
    DEFAULT_GDN_IN_PROJ_PRECISION,
    FOUR_BIT_TARGETS,
    GDN_IN_PROJ_PRECISIONS,
    GDN_IN_PROJ_TARGETS,
    gdn_in_proj_plan,
    unbalanced_norm_mappings,
)
from preserve_mtp import (
    QWEN38_MTP_KEYS,
    QWEN38_MTP_LINEAR_MODULES,
    QWEN38_MTP_SHAPES,
    preserve_mtp_weights,
    validate_mtp_artifact,
)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.8-27B")
MODEL_REVISION = os.environ.get(
    "MODEL_REVISION", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
CALIBRATION_DIR = Path(os.environ.get("CALIBRATION_DIR", "artifacts/calibration")).resolve()
MANIFEST = CALIBRATION_DIR / "manifest.jsonl"
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", "artifacts/Qwen3.8-27B-AWQ")
).resolve()
MAX_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
# What the Gated DeltaNet input projections are built at; see gdn_in_proj.py
# for what each mode means and why the set is four rather than a boolean.
# Validated here rather than at first use: until this check existed, any value
# that was not "source" selected FP8, so a typo shipped the most aggressive mode
# in the set and recorded itself in run-metadata as whatever was misspelled.
GDN_IN_PROJ_PRECISION = os.environ.get("GDN_IN_PROJ_PRECISION", DEFAULT_GDN_IN_PROJ_PRECISION)
if GDN_IN_PROJ_PRECISION not in GDN_IN_PROJ_PRECISIONS:
    raise SystemExit(
        f"GDN_IN_PROJ_PRECISION must be one of {', '.join(GDN_IN_PROJ_PRECISIONS)}; got {GDN_IN_PROJ_PRECISION!r}"
    )
# "awq" scales activations, then rounds each weight to the nearest representable
# value independently. "awq+gptq" keeps the scaling and replaces the rounding
# with GPTQ, which pushes each column's rounding error into the columns it has
# not quantized yet, correcting the layer's output rather than each weight in
# isolation. Our reconstruction error rises sharply with depth -- 6.4e-2 at
# layers.63.mlp.up_proj against a 3.98e-4 median -- which is where compensation
# has the most to recover.
QUANT_ALGORITHM = os.environ.get("QUANT_ALGORITHM", "awq")
if QUANT_ALGORITHM not in ("awq", "awq+gptq"):
    raise SystemExit(f"QUANT_ALGORITHM must be awq or awq+gptq, got {QUANT_ALGORITHM!r}")
# Added to the Hessian diagonal before inversion. Too small and a rank-deficient
# Hessian makes the Cholesky fail; llm-compressor's default is 0.01.
GPTQ_DAMPENING_FRAC = float(os.environ.get("GPTQ_DAMPENING_FRAC", "0.01"))
FULL_MANIFEST_SAMPLES = 256
NUM_SAMPLES = int(os.environ.get("CALIBRATION_SAMPLES", str(FULL_MANIFEST_SAMPLES)))


class RankCalibrationDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], processor: Any) -> None:
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        if record["kind"] == "text":
            return dict(
                self.processor(
                    text=[record["text"]],
                    padding=False,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
            )
        image_path = CALIBRATION_DIR / record["image"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": record["user"]},
                ],
            },
            {"role": "assistant", "content": record["assistant"]},
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        with Image.open(image_path) as image:
            processed = dict(
                self.processor(
                    text=[rendered],
                    images=[image.convert("RGB")],
                    padding=False,
                    return_tensors="pt",
                )
            )
        # Truncating a vision row cuts image placeholder tokens away from the
        # embeddings the vision tower produced for them, and the mismatch only
        # surfaces later inside get_placeholder_mask. Reject the row instead.
        tokens = processed["input_ids"].shape[-1]
        if tokens > MAX_LENGTH:
            raise RuntimeError(
                f"vision row exceeds MAX_SEQ_LENGTH: image={record['image']} "
                f"tokens={tokens} max={MAX_LENGTH}"
            )
        return processed


def single_item_collator(batch: list[dict[str, torch.Tensor]]):
    if len(batch) != 1:
        raise ValueError(f"calibration requires batch size 1, got {len(batch)}")
    return batch[0]


def capture_pip_freeze() -> str:
    """Capture dependency versions without invalidating a completed checkpoint."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "# pip freeze timed out after 60 seconds\n"
    if result.returncode:
        detail = result.stderr.strip().replace("\n", " ")
        return f"# pip freeze failed with exit {result.returncode}: {detail}\n"
    return result.stdout


def shard_records(
    records: list[dict[str, Any]], world_size: int
) -> list[list[dict[str, Any]]]:
    """Split the manifest into equal shards that each carry both modalities.

    A contiguous split leaves modality coverage to chance: only 48 of the 256
    manifest rows carry pixels, so a higher rank count or a reduced
    ``CALIBRATION_SAMPLES`` can leave a shard text-only. Deal the vision rows
    round-robin first, then fill each shard from the text rows.
    """
    indexed = list(enumerate(records))
    vision = [item for item in indexed if item[1]["kind"] == "vision"]
    text = [item for item in indexed if item[1]["kind"] != "vision"]
    per_rank = len(records) // world_size
    shards = [vision[shard_rank::world_size] for shard_rank in range(world_size)]
    cursor = 0
    for shard in shards:
        take = per_rank - len(shard)
        shard.extend(text[cursor : cursor + take])
        cursor += take
    for shard_rank, shard in enumerate(shards):
        kinds = {item[1]["kind"] for item in shard}
        if len(shard) != per_rank or not {"text", "vision"} <= kinds:
            raise RuntimeError(
                f"shard {shard_rank} is unusable: rows={len(shard)} "
                f"expected={per_rank} kinds={sorted(kinds)}; "
                f"vision_rows={len(vision)} text_rows={len(text)} "
                f"world_size={world_size}"
            )
    # Restore manifest order inside each shard so a rank's calibration sequence
    # stays reproducible from the manifest alone.
    return [
        [record for _, record in sorted(shard, key=lambda item: item[0])]
        for shard in shards
    ]


def validate_vision_rows(dataset: RankCalibrationDataset, rank: int) -> None:
    """Process this rank's image rows before the 27B load.

    ``__getitem__`` rejects a vision row that does not fit ``MAX_LENGTH``.
    Forcing that here surfaces it in seconds rather than minutes into the job.
    """
    checked = 0
    for index, record in enumerate(dataset.records):
        if record["kind"] == "vision":
            dataset[index]
            checked += 1
    print(f"vision-rows=ok rank={rank} checked={checked}", flush=True)


def materialize_decoder_inputs(
    dataloader: DataLoader,
    model: Any,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Build one uniform decoder-input schema from text and real image rows.

    LLM Compressor's sequential tracer assumes that the first batch fixes all
    control-flow decisions. Qwen's top-level multimodal forward branches on
    ``pixel_values is not None``, so tracing a mixture of text and image batch
    schemas either drops the vision path or makes pixel arguments mandatory.

    Run the source-precision embedding stage once before AWQ instead: text rows
    use the BF16 token embedding table, while image rows additionally execute
    the BF16 vision tower and splice its real embeddings into the token stream.
    AWQ then receives a uniform ``inputs_embeds``/``position_ids`` schema and
    still calibrates every decoder layer on the true visual embeddings.
    """
    prepared: list[dict[str, torch.Tensor]] = []
    text_rows = 0
    vision_rows = 0

    with torch.inference_mode():
        for batch in dataloader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            inputs_embeds = model.model.get_input_embeddings()(input_ids)

            text_position_ids = attention_mask.long().cumsum(-1) - 1
            text_position_ids.masked_fill_(attention_mask == 0, 0)

            if "pixel_values" in batch:
                required = ("image_grid_thw", "mm_token_type_ids")
                missing = [name for name in required if name not in batch]
                if missing:
                    raise RuntimeError(
                        f"vision calibration row is missing processor fields: {missing}"
                    )
                image_outputs = model.model.get_image_features(
                    batch["pixel_values"],
                    batch["image_grid_thw"],
                    return_dict=True,
                )
                image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = model.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                vision_position_ids, _ = model.model.get_rope_index(
                    input_ids,
                    mm_token_type_ids=batch["mm_token_type_ids"],
                    image_grid_thw=batch["image_grid_thw"],
                    attention_mask=attention_mask,
                )
                vision_rows += 1
            else:
                vision_position_ids = text_position_ids.unsqueeze(0).expand(3, -1, -1)
                text_rows += 1

            # Qwen3.5 uses axis 0 for ordinary text positions (causal masks)
            # and axes 1..3 for temporal/height/width rotary positions.
            position_ids = torch.cat(
                [text_position_ids.unsqueeze(0), vision_position_ids], dim=0
            )
            prepared.append(
                {
                    "inputs_embeds": inputs_embeds.detach().cpu(),
                    "attention_mask": attention_mask.detach().cpu(),
                    "position_ids": position_ids.detach().cpu(),
                }
            )

    if text_rows == 0 or vision_rows == 0:
        raise RuntimeError(
            "each distributed calibration shard must contain both modalities; "
            f"text_rows={text_rows}, vision_rows={vision_rows}"
        )
    print(
        f"decoder-inputs=ok rank={dist.get_rank()} rows={len(prepared)} "
        f"text_rows={text_rows} vision_rows={vision_rows} "
        "vision_tower=bf16 schema=inputs_embeds",
        flush=True,
    )
    return prepared


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    init_dist()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != local_world_size:
        raise RuntimeError(
            "quantization supports one node only: "
            f"WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}"
        )
    expected_world_size = int(os.environ.get("EXPECTED_WORLD_SIZE", str(world_size)))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, expected EXPECTED_WORLD_SIZE={expected_world_size}"
        )
    full_records = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    if len(full_records) != FULL_MANIFEST_SAMPLES:
        raise RuntimeError(
            f"manifest has {len(full_records)} rows, expected {FULL_MANIFEST_SAMPLES}"
        )
    if not 1 <= NUM_SAMPLES <= len(full_records):
        raise RuntimeError(f"invalid CALIBRATION_SAMPLES={NUM_SAMPLES}")
    if NUM_SAMPLES % world_size:
        raise RuntimeError(
            f"CALIBRATION_SAMPLES={NUM_SAMPLES} must divide evenly across "
            f"WORLD_SIZE={world_size}"
        )
    # The manifest is deterministically shuffled by the builder. Evenly spaced
    # selection preserves coverage better than taking one source-grouped prefix.
    records = [
        full_records[(index * len(full_records)) // NUM_SAMPLES]
        for index in range(NUM_SAMPLES)
    ]
    if rank == 0:
        print(
            f"calibration-selection=ok manifest_rows={len(full_records)} "
            f"selected_rows={len(records)} world_size={world_size} "
            f"rows_per_rank={NUM_SAMPLES // world_size} "
            f"vision_rows={sum(record.get('kind') == 'vision' for record in records)}",
            flush=True,
        )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    dataset = RankCalibrationDataset(shard_records(records, world_size)[rank], processor)
    validate_vision_rows(dataset, rank)
    source_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=single_item_collator,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    vision_dtypes = {
        parameter.dtype for parameter in model.model.visual.parameters()
    }
    if vision_dtypes != {torch.bfloat16}:
        raise RuntimeError(f"vision tower is not uniformly BF16: {vision_dtypes}")
    prepared = materialize_decoder_inputs(
        source_dataloader,
        model,
        torch.device("cuda", local_rank),
    )
    dataloader = DataLoader(
        prepared,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=single_item_collator,
    )
    del source_dataloader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    ignores = [
        "lm_head",
        "re:.*visual.*",
        "re:.*vision.*",
        "re:.*mtp.*",
        "re:.*linear_attn.in_proj_a$",
        "re:.*linear_attn.in_proj_b$",
    ]
    plan = gdn_in_proj_plan(GDN_IN_PROJ_PRECISION)
    ignores.extend(plan["ignore"])
    config_groups = {
        "group_0": preset_name_to_scheme(
            "W4A16_ASYM", list(FOUR_BIT_TARGETS) + list(plan["four_bit"])
        )
    }
    if plan["own_group"]:
        # Activations are stripped unconditionally, whatever the preset carries.
        # sm_89 is needed to perform FP8 ones and the serving cards are sm_86, so
        # declaring any would describe numerics no deployment of ours executes.
        # There is no flag for this on purpose, and doing it here rather than
        # per-preset means a preset added later cannot bring them back.
        config_groups["group_1"] = preset_name_to_scheme(
            plan["own_group"], list(GDN_IN_PROJ_TARGETS)
        ).model_copy(update={"input_activations": None})
    # AWQ equalization is only function-preserving when every consumer of a
    # smoothed norm receives the scale. Neither mapping here smooths
    # input_layernorm, so the Gated DeltaNet input projections get no AWQ
    # rescaling at all and there is nothing to propagate -- which is also why
    # quantizing them to four bits is a blunter operation than quantizing the
    # MLP. Adding that mapping later means listing all four consumers,
    # in_proj_a and in_proj_b included, even though no mode quantizes them.
    mapping_spec = [
        ("re:.*post_attention_layernorm$", ["re:.*gate_proj$", "re:.*up_proj$"]),
        ("re:.*up_proj$", ["re:.*down_proj$"]),
    ]
    problems = unbalanced_norm_mappings(mapping_spec)
    if problems:
        raise SystemExit("AWQ mappings would change the function:\n  " + "\n  ".join(problems))
    mappings = [AWQMapping(smooth, balances) for smooth, balances in mapping_spec]
    # GPTQ subsumes QuantizationModifier rather than running beside it: it
    # applies the same config groups itself, and two modifiers writing schemes
    # onto the same modules would fight over them.
    if QUANT_ALGORITHM == "awq+gptq":
        quantizer = GPTQModifier(
            config_groups=config_groups,
            ignore=ignores,
            dampening_frac=GPTQ_DAMPENING_FRAC,
            # A down_proj Hessian is intermediate_size squared in floats, over a
            # gigabyte each here, and one is live per module being quantized.
            # Keeping them off the accelerator costs transfer time and buys back
            # the memory the model replicas already want.
            offload_hessians=True,
        )
    else:
        quantizer = QuantizationModifier(config_groups=config_groups, ignore=ignores)
    recipe = [
        AWQModifier(duo_scaling="both", n_grid=20, mappings=mappings),
        # No kv_cache_scheme: it attaches quantization to the attention module
        # itself, and the distributed branch bin-packs scheme-bearing modules by
        # mod.weight.numel(), which Qwen3_5Attention does not have.
        quantizer,
    ]
    oneshot(
        model=model,
        processor=processor,
        dataset=dataloader,
        recipe=recipe,
        max_seq_length=MAX_LENGTH,
        num_calibration_samples=len(prepared),
        sequential_targets=["Qwen3_5DecoderLayer"],
        pad_to_max_length=False,
        shuffle_calibration_samples=False,
    )
    quantized_vision_modules = [
        name
        for name, module in model.model.visual.named_modules()
        if getattr(module, "quantization_scheme", None) is not None
    ]
    if quantized_vision_modules:
        raise RuntimeError(
            "vision tower unexpectedly received a quantization scheme: "
            f"{quantized_vision_modules[:5]}"
        )
    if {parameter.dtype for parameter in model.model.visual.parameters()} != {
        torch.bfloat16
    }:
        raise RuntimeError("vision tower dtype changed during quantization")
    def save_checkpoint() -> None:
        # The direct per-rank device map above gives rank 0 a complete replica.
        # Tear down WORLD before entering the compression wrapper so its
        # save-time recompression selects compressed-tensors' supported serial
        # branch. The distributed branch deadlocks while recoupling this direct,
        # non-offloaded Qwen replica. The final barrier is load-bearing: every
        # rank must finish oneshot collectives before any rank destroys WORLD.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(
            OUTPUT_DIR,
            save_compressed=True,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        # Qwen stores its inference-only speculative head as top-level mtp.*
        # weights. Transformers does not instantiate that module, so its normal
        # save_pretrained path silently omits the weights. Copy them verbatim
        # from the pinned source checkpoint and extend the output index before
        # calling this export complete.
        mtp_metadata = preserve_mtp_weights(
            MODEL_ID,
            MODEL_REVISION,
            OUTPUT_DIR,
        )
        validate_mtp_artifact(
            OUTPUT_DIR,
            expected_keys=QWEN38_MTP_KEYS,
            expected_modules=QWEN38_MTP_LINEAR_MODULES,
            expected_shapes=QWEN38_MTP_SHAPES,
        )
        processor.save_pretrained(OUTPUT_DIR)
        (OUTPUT_DIR / "pip-freeze.txt").write_text(
            capture_pip_freeze(), encoding="utf-8"
        )
        metadata = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "method": "AWQ W4A16 asymmetric group-size 128",
            "calibration_manifest": str(MANIFEST),
            "calibration_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "calibration_samples": NUM_SAMPLES,
            "calibration_modality": "text-and-vision",
            "max_seq_length": MAX_LENGTH,
            "awq_grid_points": 20,
            "awq_duo_scaling": "both",
            "calibration_pipeline": "sequential",
            "calibration_input_schema": "precomputed-bf16-text-and-vision-embeddings",
            "vision_tower_dtype": "torch.bfloat16",
            "vision_tower_quantized": False,
            "mtp_preserved": True,
            **mtp_metadata,
            "awq_cache_device": "cuda",
            "world_size": world_size,
            "loading_strategy": "one BF16 model replica per H200; no offload wrapper",
            "ignored_modules": ignores,
            "gdn_in_proj": GDN_IN_PROJ_PRECISION,
            # config.json is identical for both algorithms, so without this the
            # only thing separating an AWQ checkpoint from an AWQ+GPTQ one is
            # the weights themselves.
            "quant_algorithm": QUANT_ALGORITHM,
            **(
                {"gptq_dampening_frac": GPTQ_DAMPENING_FRAC, "gptq_block_size": 128}
                if QUANT_ALGORITHM == "awq+gptq"
                else {}
            ),
            "config_groups": {
                name: {
                    "num_bits": group.weights.num_bits,
                    "type": str(group.weights.type),
                    "strategy": str(group.weights.strategy),
                    "targets": list(group.targets),
                }
                for name, group in config_groups.items()
            },
            "gpu": torch.cuda.get_device_name(local_rank),
            "python": platform.python_version(),
            "elapsed_seconds": round(time.time() - started, 2),
        }
        (OUTPUT_DIR / "run-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))

    def prepare_checkpoint_model() -> None:
        # oneshot leaves DistributedDeviceCache objects installed even though
        # each rank owns a complete GPU replica. The compression wrapper mutates
        # those caches during save and their __setitem__ broadcasts. Convert them
        # to Accelerate's local representation while WORLD is still available.
        to_accelerate(model)
        remaining_caches = [
            name
            for name, module in model.named_modules(remove_duplicate=False)
            if isinstance(module._parameters, OffloadCache)
            or isinstance(module._buffers, OffloadCache)
        ]
        if remaining_caches:
            raise RuntimeError(
                "distributed offload caches remain before process-group teardown: "
                f"{remaining_caches[:10]}"
            )
        print(f"checkpoint-model=local rank={rank}", flush=True)

    run_rank0_after_group_teardown(
        rank,
        dist,
        prepare_checkpoint_model,
        save_checkpoint,
    )


if __name__ == "__main__":
    main()
