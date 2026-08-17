# Qwen3.8-27B calibrated AWQ

Reproducible W4A16 AWQ quantization of the language path in
`Qwen/Qwen3.8-27B`. The multimodal vision tower is deliberately excluded from
quantization and retained in BF16/FP16. Post-save validation checks its dtype
and runs an actual image prompt. The calibration blend targets Hermes-style
tool use, multi-turn agent trajectories, mathematics, and STEM.

The recipe has one axis: what happens to the Gated DeltaNet input projections
`in_proj_qkv` and `in_proj_z`, which are 4.0B parameters and most of the
checkpoint's size. `--gdn-in-proj` selects `int8` (the default), `int4`, or
`source`. Everything else is W4A16 in every mode and every build comes from the
same calibration, so they can be compared directly.

## Fast path

On the cluster login node:

```bash
git clone <this-repository> qwen38-awq
cd qwen38-awq
cp .env.example .env
# Edit .env and set RUN_BASE to persistent storage with at least 250 GB free.
sbatch quant/slurm/prepare.sbatch
# After preparation succeeds:
sbatch quant/slurm/quantize.sbatch
# After quantization succeeds:
sbatch eval/slurm/paired-smoke-eval.sbatch
```

The tree is grouped by what the code is for, not by which tool runs it:

```
quant/    scripts/ slurm/            producing the checkpoint
eval/     *.json scripts/ slurm/ k8s/   measuring it, and what it measures
common/   scripts/ slurm/ apptainer/    what both need
```

Cross-bucket use is one-directional: `eval/` reads the quantization side's
description of the MTP artifact, never the reverse.

All entry points read `RUN_BASE` from the local, gitignored `.env` file. The
cache, calibration corpus, and model are stored beneath that directory. An
explicitly exported `RUN_BASE` takes precedence over `.env`; `RUN_ROOT` can be
overridden independently when needed.

For a direct eight-GPU H200 host instead of Slurm:

```bash
./common/scripts/bootstrap.sh
source .venv/bin/activate
python quant/scripts/build_calibration.py
python quant/scripts/preflight.py
torchrun --standalone --nproc_per_node=gpu quant/scripts/quantize.py
./quant/scripts/smoke_test.sh
python quant/scripts/validate_generate.py
```

Defaults can be overridden without editing files:

```bash
MODEL_ID=Qwen/Qwen3.8-27B \
OUTPUT_DIR="$RUN_BASE/Qwen3.8-27B-AWQ" \
bash quant/scripts/submit_quantize.sh 8
```

The argument is the only GPU-count setting: the wrapper requests that many
GPUs, and the batch job derives both the expected distributed world size and
`torchrun` process count from the CUDA devices Slurm actually exposes.

`--gdn-in-proj` selects what those two projections are built at:

```bash
bash quant/scripts/submit_quantize.sh 8 --gdn-in-proj=int4
```

Eight bits is the default because the AWQ mappings do not reach this path: four
bits here would be bare round-to-nearest on the inputs to a recurrent state,
where every other four-bit tensor in the model is rescaled by AWQ first.

Each mode defaults to its own output directory -- `v2/model`,
`v2/model-inproj-int8`, `v2/model-inproj-int4` -- so they can all exist at once
while they are compared, and the wrapper refuses to
overwrite an existing checkpoint. `--output-dir=PATH` overrides the default.

The preparation job owns dependency setup, preflight, and calibration
construction. The quantization job only validates that the prepared manifest
exists, runs AWQ, serializes the checkpoint, and exits. The evaluation job owns
checkpoint reload and all scored/generation checks. Source and collator smokes
remain explicit diagnostics in `quant/scripts/smoke_sources.py` and
`quant/scripts/smoke_collator.py`; preparation does not rerun them.

The quantization job writes `run-metadata.json`, the exact calibration JSONL and SHA256,
package versions, GPU details, and the compressed checkpoint into the output
directory. Keep these files with any published model. Tool definitions are
included when the source row provides them. Quantization does not run source or
generation smoke tests; submit `eval/slurm/paired-smoke-eval.sbatch` separately after
the checkpoint has been saved.

The quantization workflow preserves and validates the source-precision MTP head
automatically. For an artifact created through another export path, add MTP
without recalibration or requantization:

```bash
source .venv/bin/activate
source common/scripts/load_env.sh
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"
python quant/scripts/preserve_mtp.py "$RUN_BASE/v2/model" \
  --model-id Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

If the artifact already contains the MTP tensors but not their vLLM
quantization exclusions, update only `config.json`:

```bash
python quant/scripts/preserve_mtp.py "$RUN_BASE/v2/model" --ignore-only
```

Reruns reuse the persistent virtual environment and calibration manifest after
validating its row count, exact source allocations, pinned
revisions, reasoning/tool gates, and SHA256. Set
`FORCE_CALIBRATION_REBUILD=1` to deliberately rebuild calibration data.
Successful sources are also written atomically under `calibration/source-cache`;
an existing valid manifest backfills those caches without another Hub read.

## Expected time and disk

Require at least 250 GB free on fast scratch: roughly 55 GB for the source
cache, up to another source-sized temporary/offload footprint, calibration and
environment artifacts, and roughly 28 GB for the result. Against the 55.6 GB
BF16 source the checkpoint is about 21.6 GB at `int8` (2.6x), about 19.5 GB at
`int4` (2.9x) and 25.5 GB at `source` (2.2x). All are larger than a fully 4-bit
checkpoint of this size, because the exclusions leave 4.9B of the model's 27.8B
parameters in BF16, or 8.9B under `source`. The `int8` and `int4` figures are
arithmetic rather than measurements: no build has been made under either yet,
though a third-party checkpoint of the `int4` shape measures 19.5 GB.

Once dependencies are installed and preparation has run, a quantization finishes
inside 30 minutes. The recorded `elapsed_seconds` is 6.1 minutes at eight H200s
and 9.4 at four. First download and environment creation add 10-30 minutes, and
preparation reserves an hour. The quantize job reserves whole-node host memory and 24 hours
to leave production margin. DDP gives each rank a disjoint calibration partition
(64 rows at four ranks) and synchronizes AWQ activation statistics and
scale-search errors.

Qwen3.8 is new. `quant/scripts/preflight.py` intentionally fails before the expensive
step if the installed Transformers build cannot instantiate its architecture.
Do not silently fall back to round-to-nearest and call that result calibrated
AWQ.

## Calibration blend

The release recipe uses 256 pinned public samples at up to 4,096 tokens. All
48 Cauldron records are processed with their real pixels so the quantized
decoder is calibrated on actual visual embeddings while the vision tower
itself remains in source precision.

Those 48 rows are 19% of the samples but 3.7% of the calibration tokens: 29,594
visual against 767,596 text. Cauldron images average 616 visual tokens per row,
well under the 2,304 a 1536px image would produce, while text rows average 3,690
against a 4,096 cap. Raising image resolution is the cheapest way to shift that
ratio, and it tops out near 12%.

| Samples | Source |
|---:|---|
| 104 | Open-SWE-Traces, Qwen direct, split evenly across OpenHands/SWE-agent |
| 32 | `lambda/hermes-agent-reasoning-traces`, `kimi` |
| 4 | `lambda/hermes-agent-reasoning-traces`, `glm-5.1` |
| 4 | Nemotron `tool_calling` |
| 28 | Nemotron `stem` |
| 28 | Nemotron `math` |
| 12 | Cauldron `vqav2` |
| 9 each | Cauldron `textvqa`, `chartqa`, `docvqa`, `ai2d` |
| 8 | FineWeb-Edu coherent long-form windows (at least 1,536 tokens) |

The builder saves real Cauldron pixels under persistent scratch, validates
the exact source/revision allocation, and requires reasoning, Qwen-native tool,
long-context, and image-file gates. Cauldron aggregates datasets with their own
licenses; a public model card must attribute each selected subset and its
upstream license.

Open-SWE replaces both generated self-traces and raw Nemotron code. Its 104
rows are scaffold/model stratified and target a 25% prefix, 50%
tool/code-centered interior, and 25% tail mix. If a preferred window cannot fit
the token budget, the builder tries the other complete-turn window modes for
that row. Foreign tool schemas and calls are parsed and re-rendered through
Qwen3.8's pinned chat template rather than retained in their original
serialization.

The H200 path deliberately loads one BF16 model replica directly on each GPU.
This avoids a pinned compressed-tensors offload-wrapper bug that drops the
positional `hidden_states` argument when AWQ replays
`Qwen3_5GatedDeltaNet`. Rank-local calibration is disjoint and AWQ statistics
are still reduced across ranks; only rank 0 saves the identical result. Before
sequential AWQ starts, each rank runs its image rows through the BF16 vision
tower and splices those real visual embeddings into the token stream. Text and
image rows then share one `inputs_embeds` schema, which avoids the sequential FX
tracer's static optional-pixel branch and still keeps vision calibration.
The full BF16 model and rank-local AWQ cache remain on each H200. The target
set matches the reference quantizations except in one place: vision, MTP,
`lm_head`, and the DeltaNet `in_proj_{a,b}` projections stay in source
precision, and `in_proj_{qkv,z}` are whatever `--gdn-in-proj` says. Qwen's own FP8
checkpoint keeps the vision tower and `lm_head` in BF16 too, so those exclusions
are not conservatism.

`in_proj_qkv` and `in_proj_z` are the exception, and they are most of the
remaining difference from `cyankiwi/Qwen3.8-27B-AWQ-INT4` at 19.6 GB. Across all
48 linear-attention layers they are 4.0B parameters, roughly 15% of the model:
held in BF16 the checkpoint is 25.5 GB, and at FP8 it is 21.5 GB, within 1.9 GB
of a checkpoint that quantizes them to 4 bits. `FP8_BLOCK` is what Qwen's own
FP8 release applies to these same layers, but 8 bits with per-block scales is
not evidence about 4-bit AWQ: the reported failure mode is recurrent-state
corruption that only shows at long context, and the FP8 variant does not settle
it either. RULER at 128K is the measurement that separates them, and both
variants come from one calibration so that comparison is paired. AWQ mappings
are restricted to the MLP paths, so calibration never wraps
`Qwen3_5GatedDeltaNet`; this avoids the positional-`hidden_states` bug in
compressed-tensors cache offload.

## Publishing the checkpoint

Use `quant/scripts/publish_checkpoint.py`. It plans the commit, refuses to publish a
structurally broken artifact, and only uploads when told to. Publishing happens
from a workstation, not the cluster, so it needs an interpreter that can import
`huggingface_hub`. The `hf` CLI keeps its copy in a private virtualenv that is
not on the path, so make one:

```bash
python3 -m venv .venv && .venv/bin/pip install huggingface_hub
```

A quantization run does not write a `README.md`, and the script leaves Hub-only
files alone instead of deleting them. If you do not copy the card in, the old
one stays live. Copy it first:

```bash
cp MODEL_CARD.md artifacts/Qwen3.8-27B-AWQ-FP8GDN/README.md

.venv/bin/python quant/scripts/publish_checkpoint.py \
    --repo nicosuter/Qwen3.8-27B-AWQ \
    --path artifacts/Qwen3.8-27B-AWQ-FP8GDN \
    --message "Requantize"          # add --execute to actually publish
```

It exists because `hf upload` adds and updates but never removes. Publishing a
reshard without pruning leaves the previous shards in place: a repository that
held `model-0000{1,2}-of-00002.safetensors` keeps them beside the new five,
twice the download and two sets of weights the index does not reference. The
script prunes remote `*.safetensors` that this upload does not replace, refuses
an artifact whose shards and index disagree, and excludes local state such as
`.omc/`, which `hf upload` would otherwise publish since it does not honour a
`.gitignore`. Files kept only on the Hub, the model card included, are reported
and left alone.

Check what a fresh clone would receive before publishing:

```bash
ls -a artifacts/Qwen3.8-27B-AWQ-FP8GDN
```

## Rapid paired release smoke

```bash
sbatch eval/slurm/paired-smoke-eval.sbatch
```

Ranks 0-1 run complete `Qwen/Qwen3.8-27B-FP8` replicas, pinned by
`EVAL_BASELINE_MODEL_REVISION`, and ranks 2-3 run complete AWQ replicas over the
same 64 pinned, deterministic, category-stratified MMLU-Pro items.
The gate permits at most a three-point AWQ accuracy loss and no malformed AWQ
answers. It then runs checkpoint reload, text, native tool-call, short-context,
image, vision-dtype, and MTP-dtype checks on the AWQ artifact. The scored phase
has a 15-minute hard timeout and the artifact phase has a 5-minute timeout.
Results are written under `v2/smoke-eval/`.

## Full paired evaluation

The complete `EVAL.md` protocol is implemented by the fail-closed Apptainer
runner in [`eval/README.md`](eval/README.md). It freezes and audits prompts
before inference, serves both checkpoints with the same vLLM SIF under one
served name so no harness behaviour can branch on which is loaded, validates
adapter output, and produces the paired report.

What is measured is `eval/eval-suite-v2.json`, which names the suites and their
dataset, harness, verifier and adapter pins; how it is divided across jobs is
`eval/batches.json`, which is scheduling and not pre-registration. A campaign
is submitted with `eval/slurm/campaign.sh`, one lane per candidate, and the
lanes hold the allocation by waiting on each other rather than by sharing a job
name. Every scored suite runs on one server: they are complementary rather than
competing, and the telemetry that says so is in `batches.json`.

If a distributed run reaches serialization but fails due to mismatched save
collectives, recover on one H200 without rebuilding calibration:

```bash
sbatch quant/slurm/quantize-single.sbatch
```

The recovery recipe uses all 256 rows from the deterministically shuffled
manifest. This is semantically equivalent to the distributed calibration, but
requires eight times the per-GPU activation cache and may exceed H200 memory.
Quantization and serialization occur in the same single-rank process group.
