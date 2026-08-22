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
- int8
- mixed-precision
- quantized
- compressed-tensors
- llm-compressor
- multimodal
- tool-use
- vllm
- hermes-agent
- mtp
- speculative-decoding
- long-context
datasets:
- nvidia/Open-SWE-Traces
- nvidia/Nemotron-Post-Training-Dataset-v1
- lambda/hermes-agent-reasoning-traces
- HuggingFaceM4/the_cauldron
- HuggingFaceFW/fineweb-edu
---

# Qwen3.8-27B-AWQ

A mixed-precision quantization of the language path in
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B): W4A16 asymmetric
AWQ on the MLP and attention projections, int8 group quantization on the Gated
DeltaNet input projections. The vision tower is left in source precision, so
this is still a multimodal checkpoint. Images run through an unquantized
encoder into a quantized decoder.

**Recipe, calibration builder, evaluation protocol and raw results:
[github.com/nicosuter/Qwen3.8-27B-AWQ](https://github.com/nicosuter/Qwen3.8-27B-AWQ)**

## Provenance

| | |
|---|---|
| Upstream model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Pinned revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Method | AWQ W4A16 asymmetric, group size 128, on the MLP and attention projections; int8 symmetric, group size 128, weights only, on `in_proj_qkv` and `in_proj_z` |
| Format | `compressed-tensors`, `mixed-precision` |
| Quantized with | [`llm-compressor`](https://github.com/vllm-project/llm-compressor) @ `623c8ce`, `compressed-tensors` 0.18.1a20260806, Transformers @ `a597f97`, PyTorch 2.10.0 |
| Calibration | 256 pinned public text, long-context, and vision samples, up to 4,096 tokens |
| Hardware | 2x NVIDIA H200, one BF16 replica per GPU, disjoint 128-row calibration partitions with AWQ statistics reduced across ranks |
| Recipe source | [`nicosuter/Qwen3.8-27B-AWQ`](https://github.com/nicosuter/Qwen3.8-27B-AWQ), the `quant/`, `eval/` and `common/` directories |

`run-metadata.json`, `pip-freeze.txt`, the exact calibration `manifest.jsonl`
and its SHA256 ship alongside the weights here. If you want to reproduce or
audit any of this, start there and in the repository, not from this card.

## What is and is not quantized

These modules stay in source precision:

- the vision tower (`visual`/`vision`)
- the MTP head
- Gated DeltaNet `in_proj_a` and `in_proj_b`
- `lm_head`

The Gated DeltaNet `in_proj_qkv` and `in_proj_z` projections are int8 rather
than 4-bit. Everything else that is a `Linear` gets W4A16.

## Calibration data

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

## Evaluation

Across six suites this checkpoint is within a quarter of a point of
`Qwen/Qwen3.8-27B-FP8` overall, and two points behind it on MMMU-Pro. The
suites cover executable coding, tool calls, long context and two kinds of
multimodal input. Nothing here tests agentic use. This model card will be
expanded with further evals in the near future.

Both checkpoints were scored on the same items in the same order. Recovery is
the AWQ score divided by the FP8 one, averaged across suites with the geometric
mean. Intervals are a bootstrap over 20,000 resamples, clustered by item. Each
suite ran once per checkpoint over its whole item set, on 4x H200 NVL: at a
fixed budget, more items are worth more than repeat passes. The protocol is in
[`EVAL.md`](https://github.com/nicosuter/Qwen3.8-27B-AWQ/blob/master/EVAL.md).

| suite | items | FP8 | AWQ | delta | recovery (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| BFCL | 3486 | 81.27 | 81.33 | +0.06 | 100.07% [98.99, 101.16] |
| GPQA Diamond | 198 | 89.90 | 88.89 | -1.01 | 98.88% [95.05, 102.82] |
| LiveCodeBench v6 | 175 | 88.00 | 88.57 | +0.57 | 100.65% [96.18, 105.41] |
| MMMU-Pro | 1730 | 77.17 | 75.20 | -1.97 | 97.45% [95.53, 99.40] |
| Multimodal | 600 | 86.08 | 86.34 | +0.26 | 100.31% [98.76, 101.88] |
| RULER | 200 | 92.17 | 92.90 | +0.72 | 100.79% [98.00, 103.68] |
| **macro** | 6 suites | **85.76** | **85.54** | **-0.23** | **99.68%** [98.50, 100.88] |

BFCL is the static split of
`gorilla-llm/Berkeley-Function-Calling-Leaderboard`: simple through
parallel-multiple, irrelevance, and their six `live` counterparts, which are
real user-submitted prompts shipped as static data. The executable, REST,
multi-turn and chatable categories need the Gorilla simulators, so they are
excluded, and the Java, JavaScript and SQL splits have answers this Python
matcher cannot read. Tools are passed to the model natively. Neither checkpoint
produced a malformed tool call.

LiveCodeBench v6 is pass@1: an item counts only if it passes every public and
private test. Multimodal is DocVQA, ChartQA and TextVQA, 200 items each, scored
with their published metrics. MMMU-Pro is the ten-option config across thirty
subjects. RULER is synthesized here at 4k, 32k and 128k rather than the
upstream benchmark, so its scores compare these two checkpoints and nothing
else.

MMMU-Pro is the only suite whose interval excludes zero.

### What this does not cover

- No agentic suite.
- MMMU-Pro is the only suite that separates the checkpoints. It is also the
  hardest one here, and the one where an image feeds the quantized decoder. A
  single comparison cannot say which of those explains the gap.
- 92% of items score the same on both checkpoints, mostly at ceiling, so the
  effective sample is far smaller than the item counts suggest. On BFCL, 124
  items improve and 122 regress, and the +0.06 is what is left once they
  cancel.
- Nothing here resolves to a tenth of a point, and the small suites resolve
  worst: GPQA Diamond's interval is +/-3.5 points around a measured -1.01, and
  LiveCodeBench's +/-4.0 around +0.57.
- RULER truncated 15 baseline and 12 candidate items at the 262,144-token
  window even with the output cap removed. Its 128k counting task is left out:
  one pass over the word list needs more output than the window leaves, and it
  scored zero on both checkpoints.

### What this cost

The six-suite comparison took **25 H200-hours**: one job on 4x H200, 6h13, both
checkpoints and every suite.

Note to other quantizers: a paired quality check against another model is a few
GPU hours on four cards. If you publish a quantization, you can afford to
measure it rather than inherit the upstream model's numbers.

## Usage

### vLLM

```bash
vllm serve nicosuter/Qwen3.8-27B-AWQ \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

The weights are 21.5 GB before any KV cache. To save VRAM, drop what you are not
using: `--limit-mm-per-prompt '{"image": 0}'` for the vision tower, and the
0.85 GB MTP shard if you are not running speculation.

Use the upstream generation policy: thinking enabled, `temperature=1.0`,
`top_p=0.95`, `top_k=20`, `min_p=0`, no presence penalty, repetition penalty
1.0. Qwen warns that greedy decoding degrades thinking-mode output and can
trigger repetition loops, and that warning carries over here.

### MTP / speculative decoding

The 15 MTP tensors are copied unchanged from the pinned source checkpoint into
a dedicated BF16 shard after AWQ serialization. Native speculation can be
enabled with:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

## Limitations

- Long context is only tested synthetically. RULER at 4k, 32k and 128k found no
  difference. But the recurrent path is quantized at 8 bits, and error in a
  recurrent state builds up along the sequence instead of staying bounded per
  token, so finding a planted string in generated text is a weak check for
  that.
- An unquantized vision tower does not make multimodal output safe: image
  tokens still pass through a quantized decoder. On document, chart and scene
  text the multimodal suite found no difference. On MMMU-Pro, where the
  reasoning after perception runs through the quantized path, this checkpoint is
  1.97 points behind the FP8 release. That is the largest gap measured here.

## License

Apache 2.0, following the upstream model.
