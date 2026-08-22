# Result provenance

## Purpose

A number is a result only if the code, data and checkpoint that produced it can
be named. Everything here exists because something was reused, compared or
published that could not be.

## Requirements

### Requirement: Every suite pins its dataset, harness, verifier and adapter

Each suite in the eval config SHALL carry a `pins` object, and the runner SHALL
reject missing or placeholder pins before any model request.

Pins are passed to the adapter as `EVAL_PINS_JSON`, and the adapter verifies the
installed checkout, image and dataset revision against them rather than trusting
that the right thing is installed.

#### Scenario: A pin is absent or a placeholder
- **WHEN** the runner resolves a suite whose pins are incomplete
- **THEN** it fails before preparation, not after the GPUs are held

### Requirement: The eval code is a checkout of a pushed commit

A run SHALL execute from `$RUN_BASE/code/<sha>`, cloned from the remote, and
SHALL record that commit in the run directory's `code.json`.

Deployments used to be an rsync of whatever the working tree held, which meant a
result could correspond to no commit that existed anywhere. A checkout is only
useful if the commit can be fetched again later, which is what makes discarding
old checkouts safe, so an unpushed commit is refused.

#### Scenario: Submitting against an unpushed commit
- **WHEN** the submit wrapper cannot find the commit on the remote
- **THEN** it refuses to submit

### Requirement: The run commands come from the same file as the suite list

The eval config version SHALL select both which suites run and what commands
run them.

These were separate: the suite list defaulted to v2 and the command file to a
hard-coded v1, so every run selected v2's suites, executed v1's commands, and
printed a header reading `eval-suite=v2`. RULER is the only suite whose lines
differ between the two files, which is why nothing else drifted — and why the
decision that RULER defers to the context window had never once executed. Every
RULER row on disk was taken under v1's 131072-token cap.

#### Scenario: Resolving the config
- **WHEN** the sbatch prologue resolves `CONFIG`
- **THEN** it is derived from `EVAL_SUITE_VERSION`, assigned after it
- **AND** requesting an older protocol version gets that version's commands too

### Requirement: Recorded rows are reusable only if their provenance matches

`suite_is_current` SHALL refuse rows whose checkpoint fingerprint, token cap,
recorded timeouts, or adapter pin differ from what is now running.

The adapter check exists because it was missing. Streaming landed in
`_common.py` without carrying `delta.tool_calls`, so BFCL recorded 2193 of 3486
items as empty answers — exactly 0.0000 on all nine categories that require
emitting a call and exactly 1.0000 on the two where the correct behaviour is to
emit nothing. Checkpoint, cap and timeouts all matched, so the rows certified as
current, were reused across later jobs, and read as a 32.07 model rather than a
broken harness. The adapter that produced them was in their metadata the whole
time and was never looked at.

The token cap is compared exactly and is deliberately not relaxed the way the
timeout check is. A cap no item reached is a property of one draw, not of the
distribution: keeping the draws that happened to fit and rescoring the ones that
did not selects on the outcome, biasing what is kept toward shorter reasoning —
the axis the arms actually differ on.

A recorded timeout invalidates the rows. A timed-out item is not a score, so its
zero was never the model's answer.

#### Scenario: Rows scored by a different adapter
- **WHEN** the recorded adapter pin differs from the running one
- **AND** the pair is not declared equivalent for that suite
- **THEN** the suite is rescored, and the message names both pins

#### Scenario: Rows that do not say what scored them
- **WHEN** metadata carries no adapter field
- **THEN** the rows are not reusable, because missing is not a match

### Requirement: Adapter equivalence is declared with its evidence, never assumed

Pins that score identically MAY be declared equivalent per suite in
`eval/adapter-equivalence.json`, and each group SHALL state what was compared.

An adapter pin is `sha256` over the suite adapter and `_common.py` together, and
most of `_common.py` is the generation path — streaming, retries, admission. A
change there moves the pin without changing how an existing row was scored, and
refusing every such row would rescore an inherited baseline for nothing. That
judgement cannot be made from a hash, so the check refuses and defers here.

Equivalence is per suite. Two adapters scoring alike on one suite says nothing
about another. Membership is a claim about evidence, and cost of rescoring is
not evidence.

#### Scenario: Declaring an equivalence
- **WHEN** a group is added
- **THEN** it names the files compared and how they were compared
- **AND** it lists more than one pin

### Requirement: Deferred scoring runs isolated and pinned to its generations

Suites whose scoring executes model-written code SHALL generate and score in
separate steps, with execution confined to a network-denied namespace.

LiveCodeBench generates on the GPU cluster with `--defer-execution` and executes
elsewhere. The scoring step reads nothing from the GPU cluster and writes
nothing to it; the caller brings the generations, their metadata and the answer
key.

The scoring adapter's pin SHALL be checked against the pin the generations
recorded, because scoring with a different adapter than generated is not the
same measurement. The shared answer key is hashed rather than re-uploaded, since
a shared input that is silently the wrong one grades every arm against it.

Scored rows SHALL be folded back into the run directory as an explicit step, and
the generation wall clock SHALL survive that fold. The scoring pass reports its
own wall clock, in seconds of sandboxed execution; letting it win destroys the
generation figure, which has happened.

#### Scenario: Scoring generations with a mismatched adapter
- **WHEN** the adapter directory's pin differs from the generations' recorded pin
- **THEN** scoring stops and names both, rather than producing a number

#### Scenario: A results file still marked deferred
- **WHEN** scoring finishes with any row still `deferred`
- **THEN** it fails there, rather than concatenating and being refused by the comparator

### Requirement: Results name the generations they came from

Every scored artifact SHALL be identifiable back to the generations that
produced it.

Three arms were scored by hand before this was written down, each leaving
differently named files on a shared volume, which is how you end up unable to
say which generations a results file came from. Two files claiming to be the
same baseline turned out to be different draws — 88.57 against 90.29 — and
neither could be attributed. A third pair differed by 16.6 points because one
scoring left 38 of 175 items `not_run` and counted them as zeros.

#### Scenario: Two results claim to be the same arm
- **WHEN** two scored files for one arm disagree
- **THEN** the generations are compared by hash before either is used
- **AND** a file whose generations cannot be located is not a measurement
