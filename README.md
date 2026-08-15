# Qwen3.8-27B calibrated AWQ

Reproducible W4A16 AWQ quantization of the language path in
`Qwen/Qwen3.8-27B`. The multimodal vision tower is deliberately excluded from
quantization and retained in BF16/FP16. Post-save validation checks its dtype
and runs an actual image prompt. The calibration blend targets Hermes-style
tool use, multi-turn agent trajectories, mathematics, and STEM.

## Fast path

On the cluster login node:

```bash
git clone <this-repository> qwen38-awq
cd qwen38-awq
cp .env.example .env
# Edit .env and set RUN_BASE to persistent storage with at least 250 GB free.
sbatch slurm/prepare.sbatch
# After preparation succeeds:
sbatch slurm/quantize.sbatch
# After quantization succeeds:
sbatch slurm/paired-smoke-eval.sbatch
```

All entry points read `RUN_BASE` from the local, gitignored `.env` file. The
cache, calibration corpus, and model are stored beneath that directory. An
explicitly exported `RUN_BASE` takes precedence over `.env`; `RUN_ROOT` can be
overridden independently when needed.

For a direct eight-GPU H200 host instead of Slurm:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
python scripts/build_calibration.py
python scripts/preflight.py
torchrun --standalone --nproc_per_node=gpu scripts/quantize.py
./scripts/smoke_test.sh
python scripts/validate_generate.py
```

Defaults can be overridden without editing files:

```bash
MODEL_ID=Qwen/Qwen3.8-27B \
OUTPUT_DIR="$RUN_BASE/Qwen3.8-27B-AWQ" \
bash scripts/submit_quantize.sh 8
```

The argument is the only GPU-count setting: the wrapper requests that many
GPUs, and the batch job derives both the expected distributed world size and
`torchrun` process count from the CUDA devices Slurm actually exposes.

The preparation job owns dependency setup, preflight, and calibration
construction. The quantization job only validates that the prepared manifest
exists, runs AWQ, serializes the checkpoint, and exits. The evaluation job owns
checkpoint reload and all scored/generation checks. Source and collator smokes
remain explicit diagnostics in `scripts/smoke_sources.py` and
`scripts/smoke_collator.py`; preparation does not rerun them.

The quantization job writes `run-metadata.json`, the exact calibration JSONL and SHA256,
package versions, GPU details, and the compressed checkpoint into the output
directory. Keep these files with any published model. Tool definitions are
included when the source row provides them. Quantization does not run source or
generation smoke tests; submit `slurm/paired-smoke-eval.sbatch` separately after
the checkpoint has been saved.

The quantization workflow preserves and validates the source-precision MTP head
automatically. For an artifact created through another export path, add MTP
without recalibration or requantization:

```bash
source .venv/bin/activate
source scripts/load_env.sh
export HF_HOME="${HF_HOME:-$RUN_BASE/huggingface}"
python scripts/preserve_mtp.py "$RUN_BASE/v2/model" \
  --model-id Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

If the artifact already contains the MTP tensors but not their vLLM
quantization exclusions, update only `config.json`:

```bash
python scripts/preserve_mtp.py "$RUN_BASE/v2/model" --ignore-only
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
environment artifacts, and roughly 28 GB for the result. The conservative
exclusion set leaves about 8.9B of the model's 27.8B parameters in BF16, so the
checkpoint is larger than a fully quantized W4 model of this size would be:
about 2.0x compression against the 55.6 GB BF16 source, against 2.5x for
QuantTrio's less conservative `Qwen3.6-27B-AWQ` (21.9 GB, same architecture) and
3.4x for a hypothetical W4A16 group-128 quantization of every `Linear`. A
warm-cache H200 run is
expected to take roughly 1-2 hours; first download and environment
creation can add 10-30 minutes. The job reserves whole-node host memory and 12
hours to leave production margin. DDP gives each rank a disjoint
calibration partition (64 rows at four ranks) and synchronizes AWQ activation
statistics and
scale-search errors.

Qwen3.8 is new. `scripts/preflight.py` intentionally fails before the expensive
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
image rows then share one `inputs_embeds` schema, avoiding the sequential FX
tracer's static optional-pixel branch without discarding vision calibration.
The full BF16 model and rank-local AWQ cache remain on each H200. The
target set is deliberately more conservative than QuantTrio's: vision, MTP,
layer 0, DeltaNet `in_proj_{a,b,qkv,z}`, and full-attention `q/k/v` stay in
source precision. QuantTrio's `Qwen3.6-27B-AWQ` excludes only `in_proj_a` and
`in_proj_b` from the DeltaNet input projection and quantizes `in_proj_qkv` and
`in_proj_z`. Holding those two in BF16 across the 47 quantizable linear-attention
layers is 3.9B parameters, or roughly 5.8 GB of checkpoint size, and it is the
entire difference between this artifact and theirs. AWQ mappings are restricted
to the MLP paths, so calibration never wraps `Qwen3_5GatedDeltaNet`; this avoids
the positional-`hidden_states` bug in compressed-tensors cache offload.

## Rapid paired release smoke

```bash
sbatch slurm/paired-smoke-eval.sbatch
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
before inference, alternates checkpoint order across seeds, serves both models
with the same eight-replica vLLM SIF, validates adapter output, produces full
and calibration-clean paired reports, and runs the separate MTP gate. The
benchmark adapters and their dataset/verifier revisions must be supplied and
pinned in a copy of `eval/protocol.example.json`; unresolved placeholders are
rejected rather than silently selecting moving benchmark versions.

If a distributed run reaches serialization but fails due to mismatched save
collectives, recover on one H200 without rebuilding calibration:

```bash
sbatch slurm/quantize-single.sbatch
```

The recovery recipe uses all 256 rows from the deterministically shuffled
manifest. This is semantically equivalent to the distributed calibration, but
requires eight times the per-GPU activation cache and may exceed H200 memory.
Quantization and serialization occur in the same single-rank process group.
