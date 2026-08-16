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
  - 4-bit
  - int4
  - fp8
  - mixed-precision
  - quantized
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

> Work in progress: preliminary paired evaluation below, on a suite set that is
> still growing.

A mixed-precision quantization of the language path in
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B): W4A16 asymmetric
AWQ on the MLP and attention projections, FP8 block quantization on the Gated
DeltaNet input projections. The vision tower stays in source precision, so this
is still a multimodal checkpoint: image inputs run through an unquantized
encoder and a quantized decoder.

## Provenance

| | |
|---|---|
| Upstream model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Pinned revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Method | AWQ W4A16 asymmetric, group size 128, on the MLP and attention projections; `FP8_BLOCK` (e4m3, 128x128 weight blocks, dynamic per-group activations) on `in_proj_qkv` and `in_proj_z` |
| Format | `compressed-tensors`, `mixed-precision` |
| Quantized with | [`llm-compressor`](https://github.com/vllm-project/llm-compressor) @ `623c8ce`, `compressed-tensors` 0.18.1a20260806, Transformers @ `a597f97`, PyTorch 2.10.0 |
| Calibration | 256 pinned public text, long-context, and vision samples, up to 4,096 tokens |
| Hardware | 4x NVIDIA H200, one BF16 replica per GPU, disjoint 64-row calibration partitions with AWQ statistics reduced across ranks |
| Recipe source | [`nicosuter/Qwen3.8-27B-AWQ`](https://github.com/nicosuter/Qwen3.8-27B-AWQ), the `scripts/` and `slurm/` directories |

The full recipe, calibration builder, and evaluation protocol live in the
[GitHub repository](https://github.com/nicosuter/Qwen3.8-27B-AWQ).
`run-metadata.json`, `pip-freeze.txt`, the exact calibration `manifest.jsonl`
and its SHA256 ship alongside the weights in this model repository. Anything you
want to reproduce or audit should start from those rather than from this card.

## What is and is not quantized

These modules stay in source precision:

- the vision tower (`visual`/`vision`)
- the MTP head
- Gated DeltaNet `in_proj_a` and `in_proj_b`
- `lm_head`

The Gated DeltaNet `in_proj_qkv` and `in_proj_z` projections are FP8 rather
than 4-bit. Everything else that is a `Linear` gets W4A16. The AWQ scale search
itself is restricted to two MLP mappings, `post_attention_layernorm →
gate_proj/up_proj` and `up_proj → down_proj`, with `duo_scaling="both"` over a
20-point grid. The remaining projections get no smoothing, and the FP8 group
gets no AWQ scales at all.

`in_proj_qkv` and `in_proj_z` are 4.0B parameters across all 48
linear-attention layers, roughly 15% of the model, and they are the difference
between a 25.5 GB checkpoint and this 21.5 GB one. They are held at 8 bits
rather than 4 because 48 of the 64 layers carry their long-range signal in a
recurrent state rather than a renormalized attention pattern, so error
introduced there accumulates along the sequence instead of being bounded per
token. `FP8_BLOCK` is the scheme Qwen's own FP8 release applies to these same
layers. Whether 4 bits would actually damage that path is a measurement this
repository has not made.

Confining AWQ mappings to the MLP paths also keeps calibration from ever
wrapping `Qwen3_5GatedDeltaNet`, which sidesteps a compressed-tensors
offload-wrapper bug that drops the positional `hidden_states` argument during
replay.

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

**Preliminary.** The suite set is not final, so the macro average below covers
only the four suites scored so far and will change as more are added.

These numbers come from a paired comparison against `Qwen/Qwen3.8-27B-FP8` on the
same items in the same order. Recovery is candidate/baseline, averaged across
suites with the geometric mean; intervals are an item-clustered bootstrap.
Scored on 4x H200 NVL, four replicates per suite per checkpoint. The protocol is
in [`EVAL.md`](https://github.com/nicosuter/Qwen3.8-27B-AWQ/blob/master/EVAL.md).

| suite | items | reps | responses/ckpt | FP8 | AWQ | delta | 95% CI | recovery | recovery 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| BFCL v4 | 1240 | 4 | 4960 | 87.42 | 87.74 | +0.32 | [-0.44, +1.11] | 100.37% | [99.50, 101.27] |
| GPQA Diamond | 198 | 4 | 792 | 88.89 | 89.77 | +0.88 | [-1.01, +2.90] | 100.99% | [98.84, 103.30] |
| MathArena 2026-06 | 77 | 4 | 308 | 80.52 | 79.87 | -0.65 | [-4.22, +3.25] | 99.19% | [94.74, 104.02] |
| Multimodal | 600 | 4 | 2400 | 86.75 | 86.95 | +0.20 | [-0.70, +1.12] | 100.23% | [99.19, 101.30] |
| **macro (4 suites)** | | | | **85.89** | **86.08** | **+0.19** | [-0.87, +1.28] | **100.20%** | [98.89, 101.53] |

Multimodal is DocVQA, ChartQA and TextVQA at 200 items apiece, scored with their
own published metrics. MathArena is AIME 2026 plus the Apex shortlist.

The pre-registered rule was macro geometric-mean recovery of at least 99% on the
point estimate. It measured 100.20%. The interval's lower bound is 98.89%, and no
individual suite's interval excludes zero.

Qwen publishes GPQA Diamond 89.2 for this model. The FP8 baseline measured 88.89,
95% CI [87.99, 89.79] across its four replicates, and the AWQ checkpoint 89.77,
[88.19, 91.36]. Both intervals contain the published value, which the protocol
requires before any delta is interpreted.

### What this does not cover

- No executable coding or agentic suite has run. LiveCodeBench v6 and
  Terminal-Bench 2.1 are both pending, and those are the workloads where 4-bit
  weights are most likely to cost something.
- MathArena cannot resolve its own effect at 77 items. Its interval is +-3.7
  points against a measured -0.65.
- Around 83% of items score identically on both checkpoints, mostly at ceiling.
  That is partly the result itself, but it means the effective sample is smaller
  than the item counts suggest.
- No suite here resolves to a tenth of a point. Two draws of BFCL v4 under
  identical conditions differed by 0.7, inside the interval but worth knowing
  before quoting a single figure.

Third-party quantizations of this model were scored under the same protocol.
They are not reported here: one replicate each is too few to publish.

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

The 15 MTP tensors are copied unchanged from the pinned source checkpoint into
a dedicated BF16 shard after AWQ serialization. The export fails if their
keyset, dtype, or values differ, or if the main model contains no packed
weights. All eight MTP projection modules are also added to the compressed
tensors ignore list so vLLM constructs them as BF16 Linears. Native speculation
can be enabled with:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Preserved weights are not the same as verified behavior. The MTP acceptance
rate and quality delta are part of the pending validation, so this card does
not claim either one yet.

## Limitations

- Runtime compatibility and quality retention are unestablished until the
  checks above are posted. If you need a validated 4-bit Qwen3.8, wait.
- The recurrent path is quantized here, at 8 bits. FP8 weights on the DeltaNet
  input projections need no calibration and carry per-block scales, but error
  in a recurrent state accumulates along the sequence and only shows at long
  context. RULER at 128K is that check, and its result is not posted.
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
