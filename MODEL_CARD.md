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

256 samples, deterministically selected at seed 38027, every source pinned to a
dataset revision. The blend targets agent trajectories, native tool calls,
code, math, STEM, vision, and long-context material. All 48 Cauldron records
run through the model with their real pixels, calibrating the quantized decoder
on visual embeddings while the vision tower itself remains source precision.

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

Cauldron aggregates upstream datasets under their own licenses. If you
redistribute this calibration set, attribute each subset separately.

## Evaluation

On the four suites scored so far, nothing separates this checkpoint from
`Qwen/Qwen3.8-27B-FP8` at a resolution of about one point. The suite set is not
final and will grow; nothing here covers executable coding or agentic use.

These numbers come from a paired comparison against the FP8 release on the same
items in the same order. Recovery is candidate/baseline, averaged across suites
with the geometric mean; intervals are an item-clustered bootstrap. Scored on
4x H200 NVL, four replicates per suite per checkpoint. The protocol is in
[`EVAL.md`](https://github.com/nicosuter/Qwen3.8-27B-AWQ/blob/master/EVAL.md).

| suite | items x reps | FP8 | AWQ | delta | recovery (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| BFCL v3 | 1240 x 4 | 87.42 | 87.74 | +0.32 | 100.37% [99.50, 101.27] |
| GPQA Diamond | 198 x 4 | 88.89 | 89.77 | +0.88 | 100.99% [98.84, 103.30] |
| MathArena 2026-06 | 77 x 4 | 80.52 | 79.87 | -0.65 | 99.19% [94.74, 104.02] |
| Multimodal | 600 x 4 | 86.75 | 86.95 | +0.20 | 100.23% [99.19, 101.30] |
| **macro** | 4 suites | **85.89** | **86.08** | **+0.19** | **100.20%** [98.89, 101.53] |

Multimodal is DocVQA, ChartQA and TextVQA at 200 items apiece, scored with their
own published metrics. MathArena is AIME 2026 plus the Apex shortlist. BFCL is
the v3 static split, simple through parallel-multiple plus irrelevance; the
executable, live, multi-turn and web-search categories need the Gorilla
simulators or a live network and are excluded.

The pre-registered rule was macro geometric-mean recovery of at least 99% on the
point estimate. It measured 100.20%. No individual suite's interval excludes
zero.

Qwen publishes GPQA Diamond 89.2 for this model. The FP8 baseline measured 88.89,
95% CI [87.99, 89.79] across its four replicates, and the AWQ checkpoint 89.77,
[88.19, 91.36]. Both intervals contain the published value, which the protocol
requires before any delta is interpreted.

### What this does not cover

- No executable coding or agentic suite has run. LiveCodeBench v6 and
  Terminal-Bench 2.1 are both pending, and those are the workloads where 4-bit
  weights are most likely to cost something.
- MathArena cannot resolve its own effect at 77 items. Its interval is ±3.7
  points against a measured -0.65.
- Around 83% of items score identically on both checkpoints, mostly at ceiling,
  so the effective sample is smaller than the item counts suggest.
- No suite here resolves to a tenth of a point. Two draws of BFCL v3 under
  identical conditions differed by 0.7, and the FP8 baseline's own four GPQA
  replicates spanned 2.0 points without any quantization involved.

Third-party quantizations of this model were scored under the same protocol.
They are not reported here: one replicate each is too few to publish.

### What this cost

The four-suite comparison above took **48 H200-hours**: two jobs on 4x H200, of
2h43 and 9h22. Most of that is replicates and one slow suite.

Scoring a different quantization of this model against the same FP8 baseline,
three suites at one replicate, including re-running the baseline half on the
same hardware, took **5 to 7 A100-hours** per checkpoint.

That second figure is the one worth knowing. A paired quality check against the
model you quantized is a few GPU-hours on four cards. If you publish a
quantization, you can afford to measure it rather than inherit the upstream
model's numbers.

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
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Limitations

- Long context is unmeasured. The recurrent path is quantized, at 8 bits, and
  error in a recurrent state accumulates along the sequence instead of being
  bounded per token, so if it costs anything that is where it would show.
- An unquantized vision tower does not mean multimodal output is safe: image
  tokens still pass through a quantized decoder. The multimodal suite above is
  the check for that, on document, chart and scene text, and it found no
  difference.

## License

Apache 2.0, following the upstream model. Calibration datasets keep their own
licenses; see the linked sources above.
