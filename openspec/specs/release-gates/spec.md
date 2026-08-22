# Release gates

## Purpose

Decide whether a checkpoint ships, and separately what its model card may claim.
Both thresholds are pre-registered and both are derived from simulation under
the null rather than from the result being judged.

## Requirements

### Requirement: Shipping and claiming are two decisions

The comparator SHALL emit `automated-quality-gate` and `near-lossless-claim`
independently, and neither SHALL be derived from the other.

The gate decides deployment. The claim decides what the card is allowed to say.
A checkpoint can ship without being near-lossless, and the two have different
costs when wrong.

#### Scenario: A run passes the gate and misses the claim
- **WHEN** macro recovery is above the gate's margin but below the claim's bar
- **THEN** the checkpoint is shippable
- **AND** the model card does not say near-lossless

### Requirement: The gate is on the macro, with one evidential per-suite escape

The automated gate SHALL fail when the equally weighted macro point estimate is
more than 3 points below baseline, or when a single suite's 95% paired-bootstrap
interval sits entirely below -5 points.

This replaced a rule that failed the run whenever any suite's point estimate
fell 3 points. Eight suites is eight chances and the smaller ones carry
intervals twice the width of that margin. Re-derived from the per-suite
intervals the paired runs actually produced, 200,000 draws under the null
(`eval/scripts/simulate_gates.py`):

| rule | six suites | with the agentic family |
|---|---:|---:|
| any suite falls 3 points | 33.3% | 52.0% |
| macro falls 3 points | 0.02% | 0.02% |
| a suite falls 5 points, interval clear of zero | 4.2% | 7.4% |
| near-lossless denied at 98% | 3.6% | 16.7% |

One suite can still sink the run, but only on evidence. Suites that fall past 3
points without clearing the 5-point bar are printed as review flags and belong
in the manual cluster review, not in an automatic verdict.

#### Scenario: A small suite drops on a wide interval
- **WHEN** a suite's point estimate falls more than 3 points but its interval crosses -5
- **THEN** the gate does not fail
- **AND** the suite is printed as a review flag

### Requirement: Failure modes are gated separately from accuracy

Malformed tool calls, premature final answers, empty answers, repetition loops,
context failures and timeouts SHALL each be held to no more than a 1-point
absolute increase, measured per suite and averaged with equal suite weight.

A checkpoint can hold its score while changing how it fails, and the failure
mode is what a deployment feels.

At least 95% of baseline-passed must-pass tasks SHALL still pass, where passing
means the full verifier reward, counted once per task rather than once per
replicate.

#### Scenario: Accuracy holds and malformation rises
- **WHEN** macro accuracy is within margin but malformed tool calls rise 2 points
- **THEN** the gate fails on the failure-mode rule

### Requirement: Baseline floors alert, they do not fail

A baseline score below its recorded floor SHALL print `ALERT baseline floor` and
set `baseline_floor_alerted`, and SHALL NOT fail the gate.

Floors are the only check that would notice a harness broken identically for
both checkpoints. But a broken harness is not a bad checkpoint, and the two want
different responses: a floor that fires sends someone to look at the run.

#### Scenario: Both arms score below the anchor
- **WHEN** the baseline falls under its floor
- **THEN** the comparator alerts and continues to a verdict

### Requirement: The near-lossless bar is 98%, set by what the instrument resolves

The claim SHALL be judged at 98% macro geometric-mean recovery, and SHALL be
published with its interval.

Recovery is a ratio, so its noise is a suite's standard error divided by its
baseline, and an equally weighted geometric mean is set by whichever suite is
least precise rather than by the average. Over the suites we run that interval
is about 2.1 points wide under the null. A 99% bar was asking a 1-point question
of a measurement that cannot resolve one: it would deny the claim to 19% of
checkpoints with no degradation at all, against 3.6% at 98%.

The bar moved on that simulation, before the result it now judges. It SHALL NOT
move again on a result it would have failed.

#### Scenario: The interval is wider than the margin
- **WHEN** the recovery interval's half-width exceeds the distance from 100% to the bar
- **THEN** the comparator prints a note to read the interval rather than the verdict

### Requirement: A claim of non-inferiority is reported against the interval

Where the published claim is that the candidate is not meaningfully worse, the
supporting statement SHALL be the confidence bound, not the point estimate.

The verdict field is a point-estimate test: `recovery_geomean >= bar`. The claim
it supports is an equivalence claim, and the standard rule for those is that the
lower confidence bound clears the margin. Those disagree exactly when the bar
sits inside the interval, and they have disagreed: 98.18% [97.15, 99.19] against
a 98% bar passes on the point estimate and fails on the bound.

A bound-based rule at 98% is not reachable here — it needs a half-width of 0.18
points against the 1.02 measured at R=2, roughly a 30x replicate budget — so the
margin is what moves, not the rule. State the bound that holds.

#### Scenario: Publishing a recovery figure
- **WHEN** a model card states recovery
- **THEN** it states the interval and the confidence bound that holds
- **AND** it does not present a point estimate clearing a bar as evidence of equivalence

### Requirement: Per-suite recovery is reported alongside any aggregate

Published results SHALL carry per-suite recovery, not only the macro.

The macro is retained as the internal gate because per-suite gating has a
documented false-alarm rate of 33% at six suites. It is not retained as the
headline: an equally weighted mean over suites of wildly unequal size and
saturation is a summary the reader cannot audit, and a saturated suite biases it
upward — RULER's 75 items at 1.0000 for both arms move the macro about 0.9
points on their own.

#### Scenario: A suite is saturated
- **WHEN** a suite's items score identically for both arms across most of the set
- **THEN** that is reported with the suite, because a ceiling term inflates recovery

### Requirement: A passing gate is not a deployment decision

`decision.json` SHALL leave `manual_regression_cluster_review_required` true.

The last check is whether a regression cluster has a credible common cause —
long context, parallel tool calls, vision/OCR, chemistry, dynamic programming —
merely hidden by the macro. That review is manual, and the script reports
`automated-quality-gate` rather than claiming it passed.

#### Scenario: Every automated gate passes
- **WHEN** the comparator writes a passing decision
- **THEN** the manual review flag remains true until a person clears it
