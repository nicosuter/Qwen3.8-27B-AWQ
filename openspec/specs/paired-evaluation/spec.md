# Paired evaluation

## Purpose

Measure an AWQ checkpoint against `Qwen/Qwen3.8-27B-FP8` so that the difference
between them is the checkpoint and not the harness, the schedule, or the draw.

## Requirements

### Requirement: Both arms see the same items in the same order

Materialized items SHALL be byte-identical between arms, and the task order
SHALL be randomized once and reused by both.

Preparation writes `EVAL_PROMPTS_JSONL` and the runner writes
`EVAL_TASK_ORDER_JSON` once from those IDs. Adapters consume that order without
resampling. RULER synthesizes its haystacks from a pinned corpus revision and
`--synthesis-seed` rather than downloading prebuilt prompts, because two
downloads are not two identical inputs.

#### Scenario: A second arm runs against an existing run directory
- **WHEN** a candidate is scored in a run directory that already holds a baseline
- **THEN** it reads the same `materialized/` and `orders/` the baseline used
- **AND** no item is re-synthesized

#### Scenario: The two arms disagree on which items exist
- **WHEN** the comparator finds a result key present in one arm and not the other
- **THEN** it exits non-zero and writes no comparison

### Requirement: Nothing in the harness can tell which checkpoint is loaded

Both checkpoints SHALL be served under one `--served-model-name`, with the same
vLLM image, chat template, tool parser, reasoning parser, context limit,
KV-cache dtype and scheduler settings.

This is why the served name is `qwen38-eval` for both and why the checkpoint
identity lives in `EVAL_CHECKPOINT_JSON`, written into result metadata after the
fact, rather than in anything the adapter can read at request time.

Speculative decoding SHALL be disabled for the primary comparison, so it
isolates weight recovery. MTP is measured separately.

#### Scenario: Serving the two arms
- **WHEN** either arm's server starts
- **THEN** `/v1/models` reports the same served name
- **AND** the tokenizer and effective chat-template hashes are identical between arms
- **AND** a native function-call smoke test passes before any suite is scored

### Requirement: A draw is the rows on disk, and the baseline is drawn once

A generation event SHALL produce exactly one draw, identified by suite, variant
and replicate, and persisted as its rows. Copying rows copies the draw;
generating rows makes a new one.

Per-request seeds are fully determined — run seed is `order_seed + replicate`,
and each request uses `sha256(f"{run_seed}:{text}")[:4]` — but determinism of the
seed is not reproducibility of the output. Continuous batching varies batch
composition, so the same seed against the same weights on the same GPUs returns
a different completion. Measured on unchanged checkpoints: GPQA moved 2.02
points and RULER 2.62 points between two draws.

The baseline half is identical work for every candidate, so it SHALL be bought
once and inherited by file copy (`PAIRED_INHERIT_BASELINE_FROM`) rather than
regenerated per arm. This is common random numbers, and it is what makes
candidates comparable to each other rather than each carrying its own baseline
noise.

Consequence to state wherever a recovery interval is published: the interval is
conditional on that baseline realization. An idiosyncrasy in the frozen baseline
draw is inherited identically by every arm and no amount of candidate
replication removes it.

#### Scenario: Inheriting a baseline
- **WHEN** `PAIRED_INHERIT_BASELINE_FROM` names a finished run directory
- **THEN** `orders/`, `materialized/`, `raw/baseline/*.jsonl` and
  `metadata/*-baseline-r*.json` are copied without clobbering rows this job produced
- **AND** the inherited baseline's compute capability is checked, because below
  sm_89 the FP8 baseline's declared float8 activations are served weight-only
  and a baseline measured there is not paired with a candidate served here

#### Scenario: Re-running an arm that already has rows
- **WHEN** a suite is regenerated for a checkpoint that already has a draw
- **THEN** the existing rows are archived rather than overwritten
- **AND** the new rows are a new draw of that checkpoint, not a recovery of the old one

### Requirement: Replicates are spent on generation noise, not on more questions

Additional replicates SHALL be added only where item sets are exhausted, and
SHALL be applied to both arms.

Where the items run out. GPQA Diamond at 198 is the full set. LiveCodeBench v6
at 175 is the full release. MMMU-Pro is complete. On those suites there is no
item sampling error left to reduce, so the only remaining reducible variance is
generation noise, and replicates are the only lever.

Why both arms. Replicating the candidate alone halves within-item variance on
one side of a paired difference and narrows the interval by under 10%.
Replicating both halves it on both sides. A single-arm replicate is close to
worthless and should not be mistaken for a cheaper version of this.

Why the interval does not shrink as fast as expected. At fixed spend the
variance model is `SE² = R·var_between/C + var_within/C`, so replicates trade
against item coverage; on an exhausted suite `var_between` no longer has items
to buy, which is exactly the condition that makes replicates worth it and not
before.

The comparator SHALL average replicates within an item before resampling, so
repeated generations of one question are never counted as independent
questions.

#### Scenario: Adding a replicate to an existing comparison
- **WHEN** replicate 1 is generated for a run that has replicate 0
- **THEN** it differs from replicate 0 only in the run seed
- **AND** it is generated for both arms
- **AND** the eval commit is unchanged, so no suite's item count or cap moves between replicates

#### Scenario: A replicate is generated for one arm only
- **WHEN** the comparator sees replicate 1 for the candidate and not the baseline
- **THEN** it exits non-zero rather than comparing unequal replicate sets

### Requirement: The noise floor is measured, not assumed

Before a difference between two checkpoints is reported as real, the same-
checkpoint spread SHALL be available for that suite.

Scoring a checkpoint against itself is the calibration for everything else.
Measured here: FP8 recovers 102.84% of itself; GPQA r0 against r1 of one
unchanged AWQ checkpoint gives 102.34% [98.85, 106.21] with 10 of 198 items
disagreeing. A 1-point difference between two candidates on a suite whose
same-checkpoint spread is 2.8 points is not a finding.

#### Scenario: Reporting a per-suite difference between two candidates
- **WHEN** two checkpoints differ on a suite by less than that suite's measured
  replicate-to-replicate spread
- **THEN** the difference is reported as within noise rather than as a ranking

### Requirement: Suites run concurrently within an arm and arms run sequentially

Lanes within a variant SHALL run concurrently against one server; variants SHALL
run one after another.

One vLLM server costs about four minutes to start, and the suites are
complementary rather than competing for the same resource — the telemetry that
says so is in `batches.json`. Wall clock is therefore the longest lane plus
startup, not the sum of the suites.

A suite that fails on one variant SHALL be dropped from both, because a suite
scored on one arm only is not a paired measurement.

#### Scenario: One suite fails mid-run
- **WHEN** a lane exits non-zero
- **THEN** its sibling lanes continue
- **AND** the suite is recorded in `logs/suite-failures.tsv` and excluded from both arms
- **AND** the run aborts only if no suite is left to compare

### Requirement: One suite label per suite, with drill-down as fields

Each primary suite SHALL contribute exactly one label to the macro average.

RULER keeps context length and task name as category fields under the one
`ruler` label; multimodal keeps `docvqa`, `chartqa`, `textvqa` and `private-ui`
as categories under `multimodal`. Splitting a weak category into several labels
changes its macro weight, which is a way of editing the result rather than
reporting it.

#### Scenario: Reporting per-length RULER accuracy
- **WHEN** RULER results are broken out by context length
- **THEN** the breakdown comes from the category field
- **AND** `ruler` still counts once in the macro average
