---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - qwen3_6
  - qwen3_8
  - awq
  - w4a16
  - compressed-tensors
  - llm-compressor
  - multimodal
  - tool-use
  - vllm
datasets:
  - nvidia/Open-SWE-Traces
  - nvidia/Nemotron-Post-Training-Dataset-v1
  - lambda/hermes-agent-reasoning-traces
  - HuggingFaceM4/the_cauldron
  - HuggingFaceFW/fineweb-edu
---

# Qwen3.8-27B-AWQ

> Work in progress: smoke tested so far, scored evals over the coming days.

A W4A16 asymmetric AWQ quantization of the language path in
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). The vision tower
stays in source precision, so this is still a multimodal checkpoint: image
inputs run through an unquantized encoder and a quantized decoder.

## Provenance

| | |
|---|---|
| Upstream model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Pinned revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Method | AWQ, W4A16 asymmetric, group size 128 |
| Format | `compressed-tensors` |
| Quantized with | [`llm-compressor`](https://github.com/vllm-project/llm-compressor) @ `623c8ce`, `compressed-tensors` 0.18.1a20260806, Transformers @ `a597f97`, PyTorch 2.10.0 |
| Calibration | 256 pinned public text, long-context, and vision samples, up to 4,096 tokens |
| Hardware | 4x NVIDIA H200, one BF16 replica per GPU, disjoint 64-row calibration partitions with AWQ statistics reduced across ranks |
| Recipe source | [`nicosuter/Qwen3.8-27B-AWQ`](https://github.com/nicosuter/Qwen3.8-27B-AWQ) — the `scripts/` and `slurm/` directories |

The full recipe, calibration builder, and evaluation protocol live in the
[GitHub repository](https://github.com/nicosuter/Qwen3.8-27B-AWQ).
`run-metadata.json`, `pip-freeze.txt`, the exact calibration `manifest.jsonl`
and its SHA256 ship alongside the weights in this model repository. Anything you
want to reproduce or audit should start from those rather than from this card.

## What is and is not quantized

These modules stay in source precision:

- the vision tower (`visual`/`vision`)
- the MTP head
- decoder layer 0
- Gated DeltaNet input projections (`in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`)
- full-attention `q_proj`, `k_proj`, `v_proj`
- `lm_head`

Everything else that is a `Linear` gets W4A16. The AWQ scale search itself is
restricted to two MLP mappings, `post_attention_layernorm → gate_proj/up_proj`
and `up_proj → down_proj`, with `duo_scaling="both"` over a 20-point grid. The
remaining quantized projections are quantized without smoothing.

The exclusion set follows QuantTrio's conservative target list. Confining AWQ
mappings to the MLP paths also keeps calibration from ever wrapping
`Qwen3_5GatedDeltaNet`, which sidesteps a compressed-tensors offload-wrapper
bug that drops the positional `hidden_states` argument during replay.

## Calibration data

256 samples, deterministically selected at seed 38027, every source pinned to a
dataset revision. The blend targets agent trajectories, native tool calls,
code, math, STEM, vision, and long-context material. All 48 Cauldron records
run through the model with their real pixels, calibrating the quantized decoder
on visual embeddings while the vision tower itself remains source precision.

Image rows are 48 of 256 samples but only 29,594 of 797,190 calibration tokens,
or 3.7%. Cauldron images average 616 visual tokens per row against 3,690 tokens
per text row, so text dominates the activation statistics AWQ uses to pick
per-channel scales.

| Samples | Source | Config / split |
|---:|---|---|
| 52 | `nvidia/Open-SWE-Traces` | `openhands` / `qwen35_122b` |
| 52 | `nvidia/Open-SWE-Traces` | `sweagent` / `qwen35_122b` |
| 32 | `lambda/hermes-agent-reasoning-traces` | `kimi` |
| 4 | `lambda/hermes-agent-reasoning-traces` | `glm-5.1` |
| 28 | `nvidia/Nemotron-Post-Training-Dataset-v1` | `stem` |
| 28 | `nvidia/Nemotron-Post-Training-Dataset-v1` | `math` |
| 4 | `nvidia/Nemotron-Post-Training-Dataset-v1` | `tool_calling` |
| 12 | `HuggingFaceM4/the_cauldron` | `vqav2` |
| 9 each | `HuggingFaceM4/the_cauldron` | `textvqa`, `chartqa`, `docvqa`, `ai2d` |
| 8 | `HuggingFaceFW/fineweb-edu` | coherent windows of at least 1,536 tokens |

Revisions: Open-SWE `ad4805a`, Lambda `b92885e`, Nemotron `74e23eb`, Cauldron
`847a98a`, FineWeb-Edu `87f0914`.

The Open-SWE rows are stratified by scaffold and model and target a 25% prefix,
50% tool- and code-centered interior, 25% tail mix. If a preferred span cannot
fit the token budget, the builder tries the other complete-turn window modes
for that row. Foreign tool schemas and calls are parsed and re-emitted through
Qwen3.8's own chat template rather than left in their original serialization,
so the calibration text matches what the model actually sees at inference.
Cauldron rows carry real pixels, not placeholders.

Each rank first runs its image rows through the BF16 vision tower and splices
the resulting real visual embeddings into the token stream. Sequential AWQ then
receives one uniform `inputs_embeds` schema for both text and image rows. This
avoids specializing Qwen's optional-pixel branch to the first batch while
retaining the sequential pipeline's bounded activation cache.

The builder enforces gates for reasoning content, Qwen-native tool calls,
long-context length, and image files, and validates the exact per-source
allocation before writing the manifest. A run that cannot fill a source fails
instead of quietly substituting.

Cauldron aggregates upstream datasets under their own licenses. If you
redistribute this calibration set, attribute each subset separately.

## Evaluation

Not yet scored. Evaluation runs over the coming days: a controlled paired
comparison against `Qwen/Qwen3.8-27B-FP8`, and against other public
quantizations of the same model.

The suites are BFCL v4 for tool calling, Terminal-Bench 2.1 for agentic coding,
LiveCodeBench v6, GPQA Diamond and MathArena for reasoning, and DocVQA, ChartQA
and TextVQA for the multimodal path. That set matches what the calibration blend
targets and where 4-bit weights are most likely to cost something.

## Usage

### vLLM

```bash
vllm serve nicosuter/Qwen3.8-27B-AWQ \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --kv-cache-dtype auto \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Use the upstream generation policy: thinking enabled, `temperature=1.0`,
`top_p=0.95`, `top_k=20`, `min_p=0`, no presence penalty, repetition penalty
1.0. Qwen warns that greedy decoding degrades thinking-mode output and can
trigger repetition loops, and that warning carries over here.

### MTP / speculative decoding

The MTP tensors are preserved in source precision, so speculation should load:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Preserved weights are not the same as verified behavior. The MTP acceptance
rate and quality delta are part of the pending validation, so this card does
not claim either one yet.

## Limitations

- Runtime compatibility and quality retention are unestablished until the
  checks above are posted. If you need a validated W4A16 Qwen3.8, wait.
- An unquantized vision tower does not mean multimodal output is unaffected,
  since image tokens still pass through a quantized decoder. That is what the
  vision suites are there to measure.
- Calibration is 256 samples at 4,096 tokens. The recipe was not tuned for
  behavior well past that length, or for languages and domains the blend does
  not cover.
- Quantization inherits every limitation and bias of the upstream model and
  fixes none of them.

## License

Apache 2.0, following the upstream model. Calibration datasets keep their own
licenses; see the linked sources above.
