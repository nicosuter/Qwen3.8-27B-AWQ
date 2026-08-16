# Preliminary: AWQ + FP8-GDN hybrid vs Qwen3.8-27B-FP8

**Preliminary.** The numbers below are what was measured; the suite set they are
averaged over is not yet frozen, so the macro is expected to change. See
"Why this is preliminary".

Recorded because the run directory is not durable: `comparison.json` under
`v2/paired-2` was overwritten once already when the chained matharena job
rescored two suites, and the earlier three-suite result would have been lost.

## Result

Recovery is candidate/baseline, summarized across suites with the geometric
mean and reported beside the absolute scores it derives from.

| suite | items | baseline | candidate | delta | 95% CI | recovery | recovery 95% CI |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| bfcl_v4 | 1240 | 87.42 | 87.74 | +0.32 | [-0.44, +1.11] | 100.37% | [99.50, 101.27] |
| gpqa_diamond | 198 | 88.89 | 89.77 | +0.88 | [-1.01, +2.90] | 100.99% | [98.84, 103.30] |
| matharena_2026_06 | 77 | 80.52 | 79.87 | -0.65 | [-4.22, +3.25] | 99.19% | [94.74, 104.02] |
| multimodal | 600 | 86.75 | 86.95 | +0.20 | [-0.70, +1.12] | 100.23% | [99.19, 101.30] |
| **MACRO** | 4 | 85.89 | 86.08 | +0.19 | [-0.87, +1.28] | **100.20%** | [98.89, 101.53] |

`bfcl_v4` is the suite identifier, not the dataset version. The data is BFCL's
v3 static split, five categories of it; no v4 is published on the Hub. The name
is kept because it keys the frozen order and every result file here.

`near-lossless-claim=PASS` against a 99.00% bar, `automated-quality-gate=PASS`.

The near-lossless rule was pre-registered on the **point estimate**. On this
suite set the interval's lower bound is 98.89%, below the bar; on the
three-suite set before matharena was added it was 99.38%, above it. The claim
rests on the pre-registered rule, not on the lower bound.

## Upstream anchor

`EVAL.md` requires the FP8 baseline to land near Qwen's published GPQA Diamond
of 89.2 before any AWQ delta is interpreted.

| | replicates | mean | 95% CI | vs 89.2 |
| --- | --- | ---: | --- | ---: |
| FP8 baseline | 89.90 / 89.39 / 88.38 / 87.88 | 88.89 | [87.99, 89.79] | -0.31 |
| AWQ candidate | 87.37 / 90.91 / 90.40 / 90.40 | 89.77 | [88.19, 91.36] | +0.57 |

Both intervals contain the published value. This is the only upstream anchor
currently built; LiveCodeBench v6 (published 90.3) is not yet run.

## Conditions

- **Checkpoints**: `checkpoints.jsonl`, fingerprinted before serving. Baseline
  `Qwen/Qwen3.8-27B-FP8` at `017b9c7a`, 30.9 GB. Candidate `v2/model-fp8gdn`,
  21.5 GB, W4A16 asymmetric group-128 plus an FP8 block-wise group on the GDN
  projections, 311 ignored modules.
- **Hardware**: 4x NVIDIA H200 NVL, one node, `timeout-scale=1.0`.
- **Serving**: `DP=4`, both variants under one `--served-model-name` so no
  harness behavior can branch on which checkpoint is loaded.
- **Config**: `eval/paired-2.json` for the three fast suites,
  `eval/paired-1.json` for matharena; dataset revisions and per-adapter source
  hashes are pinned there.
- **Replicates**: 4 per suite per variant, pooled per item before bootstrapping.
- **Statistics**: item-clustered bootstrap resampling pairs, so recovery draws
  both sides from the same sample.
- **Jobs**: 4621 (02:43:02) scored the three fast suites; 4622 (09:22:32) added
  matharena and rescored bfcl and multimodal.

## Why this is preliminary

- **The suite set is not frozen.** LiveCodeBench v6 is a primary suite in
  `EVAL.md` and has never been run; adding it will change the macro. Whether
  matharena votes at all is undecided, and its apex-shortlist half is under
  review as a substitute for two snapshots that were never published.
- **Two draws of the same comparison disagree by more than a rounding error.**
  bfcl_v4 measured -0.40 in job 4621 and +0.32 in 4622 -- a sign flip, 1.3
  sigma, consistent with the interval but a caution against reading any suite
  to a tenth of a point.
- **matharena cannot resolve its own effect.** 77 items, `|delta|/half = 0.17`.
  Including it widened the macro half-width from 0.77 to 1.08.
- **Roughly 83% of items are constant** across all eight observations, mostly
  at ceiling. That is partly the result -- a near-lossless candidate moves
  little -- but it means the effective sample is far smaller than the item
  counts suggest.

## Files

- `comparison.json` -- the comparator's full output, including per-suite
  bootstrap distributions and the gate decision.
- `checkpoints.jsonl` -- the fingerprint and quantization descriptor of each
  checkpoint as served.

Raw per-item results (~6.5 MB per variant) stay on the cluster under
`v2/paired-2/raw/`.
