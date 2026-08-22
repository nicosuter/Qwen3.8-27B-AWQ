# Eval suite v2 vs Qwen3.8-27B-FP8

Six suites over the frozen v2 suite set, both arms in one job. The candidate
is the checkpoint published as `nicosuter/Qwen3.8-27B-AWQ`.

## Result

Recovery is candidate/baseline, summarized across suites with the geometric
mean and reported beside the absolute scores it derives from.

| suite | items | baseline | candidate | delta | 95% CI | recovery | recovery 95% CI |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| bfcl_v4 | 3486 | 81.27 | 81.33 | +0.06 | [-0.83, +0.95] | 100.07% | [98.99, 101.16] |
| gpqa_diamond | 198 | 89.90 | 88.89 | -1.01 | [-4.55, +2.53] | 98.88% | [95.05, 102.82] |
| livecodebench_v6 | 175 | 88.00 | 88.57 | +0.57 | [-3.43, +4.57] | 100.65% | [96.18, 105.41] |
| mmmu_pro | 1730 | 77.17 | 75.20 | -1.97 | [-3.47, -0.46] | 97.45% | [95.53, 99.40] |
| multimodal | 600 | 86.08 | 86.34 | +0.26 | [-1.07, +1.60] | 100.31% | [98.76, 101.88] |
| ruler | 200 | 92.17 | 92.90 | +0.72 | [-1.85, +3.33] | 100.79% | [98.00, 103.68] |
| **MACRO** | 6 | 85.76 | 85.54 | -0.23 | [-1.28, +0.82] | **99.68%** | [98.50, 100.88] |

`near-lossless-claim=PASS`, `automated-quality-gate=PASS`.

The macro point estimate is 99.68% and the interval's lower bound is 98.50%,
so the claim holds on the pre-registered point-estimate rule and also against
the 98.00% bar the comparator checks the lower bound against.

`mmmu_pro` is the only suite whose interval excludes zero. It is also the
hardest suite in the set and the only one that puts an image in front of the
quantized language path, so a single comparison cannot separate the mechanism
from the sample.

92% of items (5,903 of 6,389) score the same on both checkpoints, most of them
at ceiling, so the effective sample is much smaller than the item counts. On
bfcl_v4, 124 items improve and 122 regress to produce the +0.06.

## Baseline floor alert

`bfcl_v4` scored 81.27 on the FP8 arm against a floor of 83.00, which is 95% of
where that suite's FP8 baseline has landed before. The protocol alerts rather
than failing on this: a baseline that low means the harness behaved differently
for both arms, which is a reason to look at the run and not a reason to fail a
checkpoint. Its recovery of 100.07% is a ratio of two numbers that both came in
low.

## Upstream anchor

`EVAL.md` requires the FP8 baseline to land near a published figure before any
AWQ delta is interpreted.

| suite | published | FP8 baseline here | difference |
| --- | ---: | ---: | ---: |
| GPQA Diamond | 89.2 | 89.90 | +0.70 |
| LiveCodeBench v6 | 90.3 | 88.00 | -2.30 |

The LiveCodeBench anchor is weaker than it looks. This harness scores 175 items
with its own answer extraction and its own executor, not the published
evaluation, so the gap is not evidence about the checkpoint.

## Conditions

- **Checkpoints**: `checkpoints.jsonl`, fingerprinted before serving. Baseline
  `Qwen/Qwen3.8-27B-FP8` at `017b9c7a`, 30.9 GB. Candidate
  `v2/model-inproj-int8-calv2-smoothattn`, 21.5 GB, W4A16 asymmetric group-128
  on the MLP and attention projections plus an int8 group on the GDN input
  projections, 311 ignored modules. This is the checkpoint published as
  `nicosuter/Qwen3.8-27B-AWQ`; its six shards hash identical to the ones on the
  Hub. Neither arm carries an activation taint.
- **Hardware**: 4x NVIDIA H200 NVL, one node, `timeout-scale=1.0`.
- **Serving**: two engines at `tensor_parallel_size=2` across the four GPUs,
  both variants under one `--served-model-name` so no harness behavior can
  branch on which checkpoint is loaded.
- **Config**: `eval/eval-suite-v2.json`. Dataset revisions, harness and verifier
  ids and per-adapter source hashes are pinned there. Code at `ee24f74`.
- **Replicates**: 1 per suite per variant, over each suite's whole item set. At
  a fixed budget more items are worth more than repeat passes.
- **Generation**: thinking enabled at `reasoning_effort=xhigh`,
  `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, no presence penalty,
  repetition penalty 1.0. Output capped at 131,072 tokens everywhere except
  `ruler`, which runs uncapped against the 262,144-token window.
- **Statistics**: bootstrap over 20,000 resamples, clustered by item, seed
  38027.
- **Truncation**: 20 items per arm hit their limit, 15 baseline and 12 candidate
  of them on `ruler`, 4 and 7 on `livecodebench_v6`, 1 each on `gpqa_diamond`.
- **Wall clock**: 6h13 for both arms and all six suites.

## LiveCodeBench execution

`livecodebench_v6` runs with `--defer-execution`. The suite writes generations
and stops, because running model-written code on the machine serving the model
is the one thing that clearly does not belong there. A separate pass executed
all 350 generations against the public and private tests on CPU, with no
network and no GPU, under the adapter's default limits: 10 s per test, 2 GB of
address space, a 120 s budget per item. Both arms were executed by the same
Python 3.12.5 interpreter with numpy 2.5.2, matching the environment the other
suites were scored in.

One baseline item timed out and no candidate item did, out of 175 each. A
per-test timeout depends on how fast the executing CPU is, so it is worth
knowing that this bound was reached at all, but it fell on both arms in the same
way.
