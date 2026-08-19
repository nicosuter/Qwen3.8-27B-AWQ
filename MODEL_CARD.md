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
AWQ on the MLP and attention projections, int8 group-128 on the Gated DeltaNet
input projections. The vision tower stays in source precision, so this is still
a multimodal checkpoint: image inputs run through an unquantized encoder and a
quantized decoder.

**Recipe, calibration builder, evaluation protocol and raw results:
[github.com/nicosuter/Qwen3.8-27B-AWQ](https://github.com/nicosuter/Qwen3.8-27B-AWQ)**

> This checkpoint is still being iterated. The recipe for the quantized
> projections has changed at least once and may change again, so the weights
> behind this repository name are not stable over time. Pin a revision if you
> need a build you can return to.

## Provenance

| | |
|---|---|
| Upstream model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Pinned revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Method | AWQ W4A16 asymmetric, group size 128, on the MLP and attention projections; int8 symmetric, group size 128, weight-only, on `in_proj_qkv` and `in_proj_z` |
| Format | `compressed-tensors`, `mixed-precision` |
| Quantized with | [`llm-compressor`](https://github.com/vllm-project/llm-compressor) @ `623c8ce`, `compressed-tensors` 0.18.1a20260806, Transformers @ `a597f97`, PyTorch 2.10.0 |
| Calibration | 256 pinned public text, long-context, and vision samples, up to 4,096 tokens |
| Hardware | 2x NVIDIA H200, one BF16 replica per GPU, disjoint 128-row calibration partitions with AWQ statistics reduced across ranks |
| Recipe source | [`nicosuter/Qwen3.8-27B-AWQ`](https://github.com/nicosuter/Qwen3.8-27B-AWQ), the `quant/`, `eval/` and `common/` directories |

`run-metadata.json`, `pip-freeze.txt` and `recipe.yaml` ship alongside the
weights here, including the SHA256 of the calibration manifest that produced
them. The manifest itself, and the builder that writes it, are in the
repository. Anything you want to reproduce or audit should start from those
rather than from this card.

## What is and is not quantized

These modules stay in source precision:

- the vision tower (`visual`/`vision`)
- the MTP head
- Gated DeltaNet `in_proj_a` and `in_proj_b`
- `lm_head`

The Gated DeltaNet `in_proj_qkv` and `in_proj_z` projections are int8 rather
than 4-bit: symmetric, group size 128, weights only, with activations left
unquantized. Everything else that is a `Linear` gets W4A16. The AWQ scale search
itself is restricted to two MLP mappings, `post_attention_layernorm →
gate_proj/up_proj` and `up_proj → down_proj`, with `duo_scaling="both"` over a
20-point grid. The remaining projections get no smoothing, and the int8 group
gets no AWQ scales at all.

`in_proj_qkv` and `in_proj_z` are 4.0B parameters across all 48
linear-attention layers, roughly 15% of the model, and they are the difference
between a 25.5 GB checkpoint and this 21.5 GB one. They are held at 8 bits
rather than 4 because 48 of the 64 layers carry their long-range signal in a
recurrent state rather than a renormalized attention pattern, so error
introduced there accumulates along the sequence instead of being bounded per
token. That is an argument from the architecture, not from a measurement:
whether 4 bits would actually damage that path is something this repository has
not established, and neither is the exact 8-bit scheme load-bearing.

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

This checkpoint is scored with a paired protocol against
[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8): the same
frozen items in the same order for both arms, recovery as candidate/baseline
averaged across suites with the geometric mean, and item-clustered bootstrap
intervals.

The protocol is in
[`EVAL.md`](https://github.com/nicosuter/Qwen3.8-27B-AWQ/blob/master/EVAL.md)
and the current results are in the repository. They are deliberately not copied
into this card: the recipe is still moving, and numbers pinned here would
outlive the weights they describe and quietly become claims about a checkpoint
nobody can download any more. Read them against a commit.

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

- Long context is unmeasured. The recurrent path is quantized, at 8 bits, and
  error in a recurrent state accumulates along the sequence instead of being
  bounded per token, so if it costs anything that is where it would show. The
  suite meant to settle this has not produced a clean read.
- An unquantized vision tower does not mean multimodal output is unaffected,
  since image tokens still pass through a quantized decoder.
- No executable coding or agentic workload is part of the settled result set,
  and those are where 4-bit weights are most likely to cost something.
- Calibration is 256 samples at 4,096 tokens. The recipe was not tuned for
  behavior well past that length, or for languages and domains the blend does
  not cover.

## License

Apache 2.0, following the upstream model. Calibration datasets keep their own
licenses; see the linked sources above.
