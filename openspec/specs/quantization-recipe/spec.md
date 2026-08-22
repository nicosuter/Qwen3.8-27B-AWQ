# Quantization recipe

## Purpose

Produce a W4A16 AWQ checkpoint of `Qwen/Qwen3.8-27B` whose exclusions are
deliberate and whose one variable axis is isolated.

## Requirements

### Requirement: The recipe has one axis and everything else is held fixed

`--gdn-in-proj` SHALL select `int8`, `int4` or `source` for the Gated DeltaNet
input projections `in_proj_qkv` and `in_proj_z`, and SHALL be the only recipe
variable.

Those two projections are 4.0B parameters across 48 linear-attention layers,
roughly 15% of the model and most of the difference in checkpoint size: BF16
gives 25.5 GB, FP8 21.5 GB, 4-bit about 19.5 GB. Every mode is W4A16 everywhere
else and every build comes from the same calibration, so modes compare directly
under the paired protocol.

Each mode SHALL default to its own output directory, and the wrapper SHALL
refuse to overwrite an existing checkpoint, so modes can coexist while being
compared.

#### Scenario: Building a second mode
- **WHEN** a build is submitted with a different `--gdn-in-proj`
- **THEN** it writes to that mode's own directory
- **AND** it reuses the existing calibration manifest after validating its hash

### Requirement: Eight bits is the default because AWQ does not reach that path

`int8` SHALL be the default for the GDN input projections.

AWQ mappings are restricted to the MLP paths, so these projections receive no
smoothing scales. Four bits here would be bare round-to-nearest on the inputs to
a recurrent state, where every other four-bit tensor in the model is rescaled by
AWQ first.

`FP8_BLOCK` is what Qwen's own FP8 release applies to these same layers, but 8
bits with per-block scales is not evidence about 4-bit AWQ. The reported failure
mode is recurrent-state corruption that only shows at long context, and the FP8
variant does not settle it either.

#### Scenario: Choosing four bits for the GDN input projections
- **WHEN** `--gdn-in-proj=int4` is built
- **THEN** it is compared against `int8` from the same calibration
- **AND** RULER at long context is the measurement that separates them

### Requirement: Exclusions are exclusions of record, not conservatism

Vision, MTP, `lm_head` and the DeltaNet `in_proj_{a,b}` projections SHALL remain
in source precision.

Qwen's own FP8 checkpoint keeps the vision tower and `lm_head` in BF16, so these
are not extra caution. They leave 4.9B of 27.8B parameters in BF16, or 8.9B
under `source`, which is why every mode is larger than a fully 4-bit checkpoint
of this size.

#### Scenario: Validating a saved checkpoint
- **WHEN** post-save validation runs
- **THEN** it checks the vision tower's dtype and runs an actual image prompt
- **AND** it fails if the main model carries no packed weights

### Requirement: AWQ mappings cover the MLP paths only

The scale search SHALL cover `post_attention_layernorm -> gate_proj/up_proj` and
`up_proj -> down_proj` with `duo_scaling="both"` over a 20-point grid.

Restricting mappings to the MLP paths means calibration never wraps
`Qwen3_5GatedDeltaNet`, which avoids a pinned compressed-tensors offload-wrapper
bug that drops the positional `hidden_states` argument when AWQ replays that
module. Every other projection is quantized without smoothing, and the 8-bit
group gets no AWQ scales at all.

#### Scenario: A mapping is added that reaches the attention path
- **WHEN** a mapping would cause AWQ to replay `Qwen3_5GatedDeltaNet`
- **THEN** the offload-wrapper interaction is verified before the mapping is kept

### Requirement: The MTP head is preserved and excluded, and the export proves it

The source-precision MTP head SHALL survive quantization, and export SHALL fail
if its 15 tensors' keyset, dtype or values differ from source.

The eight MTP projection modules SHALL be added to the compressed-tensors ignore
list so vLLM builds them as BF16 Linears. An artifact created through another
export path can be repaired without recalibration via
`quant/scripts/preserve_mtp.py`, including `--ignore-only` for the case where
the tensors are present but the exclusions are not.

#### Scenario: Exporting a checkpoint
- **WHEN** serialization completes
- **THEN** the MTP tensors are byte-compared against source
- **AND** `config.json` carries the MTP projection exclusions

### Requirement: Calibration is pinned, validated and reused

The blend SHALL be 256 pinned public samples at up to 4,096 tokens, and reruns
SHALL reuse an existing manifest only after validating row count, exact source
allocations, pinned revisions, reasoning and tool gates, and SHA256.

All 48 Cauldron records are processed with their real pixels, so the quantized
decoder is calibrated on actual visual embeddings while the vision tower stays
in source precision. Those rows are 19% of samples but 3.7% of calibration
tokens — 29,594 visual against 767,596 text — because Cauldron images average
616 visual tokens against a 3,690-token text average. Raising image resolution
is the cheapest way to shift that ratio and it tops out near 12%.

Foreign tool schemas and calls SHALL be re-rendered through Qwen3.8's pinned
chat template rather than retained in their original serialization.

#### Scenario: A rerun finds an existing manifest
- **WHEN** the manifest validates
- **THEN** calibration is reused and source caches are backfilled without another Hub read
- **AND** `FORCE_CALIBRATION_REBUILD=1` is the only way to rebuild deliberately

### Requirement: Preflight fails before the expensive step, never falls back

`quant/scripts/preflight.py` SHALL fail if the installed Transformers build
cannot instantiate the architecture.

Qwen3.8 is new. A silent fall back to round-to-nearest produces an artifact that
is not calibrated AWQ and does not say so.

#### Scenario: The installed Transformers cannot build the architecture
- **WHEN** preflight runs
- **THEN** it fails before quantization starts

### Requirement: Distributed calibration is disjoint and reduced, and one rank saves

Each rank SHALL load one BF16 replica directly on its GPU, take a disjoint
calibration partition, and participate in a reduction of AWQ activation
statistics and scale-search errors; rank 0 SHALL save.

Loading per-GPU rather than offloading avoids the compressed-tensors
offload-wrapper bug. Before sequential AWQ starts, each rank runs its image rows
through the BF16 vision tower and splices the real visual embeddings into the
token stream, so text and image rows share one `inputs_embeds` schema — which
avoids the sequential FX tracer's static optional-pixel branch while keeping
vision calibration.

#### Scenario: Recovering a run that failed at serialization
- **WHEN** `quant/slurm/quantize-single.sbatch` is used
- **THEN** it uses all 256 rows from the deterministically shuffled manifest
- **AND** quantization and serialization happen in one process group
