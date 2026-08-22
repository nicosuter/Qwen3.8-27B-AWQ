# Checkpoint publishing

## Purpose

Put an artifact on the Hub such that a fresh clone receives exactly the
checkpoint that was measured, and nothing else.

## Requirements

### Requirement: Publishing prunes what it replaces

`quant/scripts/publish_checkpoint.py` SHALL remove remote `*.safetensors` that
the upload does not replace.

`hf upload` adds and updates but never removes. Publishing a reshard without
pruning leaves the previous shards in place: a repository holding
`model-0000{1,2}-of-00002.safetensors` keeps them beside the new five — twice
the download, and two sets of weights the index does not reference.

#### Scenario: Publishing a checkpoint with a different shard count
- **WHEN** the new artifact has fewer or differently named shards
- **THEN** the shards the index does not reference are removed from the remote

### Requirement: A structurally broken artifact is refused

The publisher SHALL refuse an artifact whose shards and index disagree, and
SHALL plan the commit before uploading anything.

Publishing SHALL require an explicit `--execute`; without it the script reports
what it would do.

#### Scenario: Index and shards disagree
- **WHEN** the index references a shard the artifact does not contain
- **THEN** the publisher refuses, before any upload

### Requirement: Local state is never published

The publisher SHALL exclude local operational state such as `.omc/`.

`hf upload` does not honour a `.gitignore`, so exclusion is the publisher's job
rather than something inherited.

#### Scenario: The artifact directory contains local state
- **WHEN** the artifact holds `.omc/` or similar
- **THEN** it is excluded from the upload

### Requirement: Hub-only files are reported and left alone

Files that exist on the Hub and not in the artifact SHALL be reported and
retained.

A quantization run does not write a `README.md`. Deleting Hub-only files would
remove the model card on every publish; leaving them means an un-copied card
stays live, which is why the card must be copied into the artifact deliberately
before publishing.

#### Scenario: The artifact has no model card
- **WHEN** publishing an artifact without `README.md`
- **THEN** the existing Hub card is retained and reported as Hub-only

### Requirement: The reproducibility record travels with the artifact

`run-metadata.json`, the exact calibration JSONL and its SHA256, package
versions and GPU details SHALL be written into the output directory and kept
with any published model.

#### Scenario: Publishing a measured checkpoint
- **WHEN** an artifact is published
- **THEN** it carries the calibration hash and run metadata that identify what was measured

### Requirement: The model card is a claim surface, not an implementation zoo

The published card SHALL carry what the model is and what was measured. Recipe
internals belong in the repository README.

Where the card states recovery it SHALL follow the reporting rule in
`release-gates`: per-suite figures with intervals, and a bound that holds rather
than a point estimate presented as equivalence.

Cauldron aggregates datasets under their own licenses, so a public card SHALL
attribute each selected subset and its upstream license.

#### Scenario: A recipe detail is proposed for the card
- **WHEN** the detail describes how the checkpoint was built rather than what it is
- **THEN** it goes in the README
