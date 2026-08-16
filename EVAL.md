# FP8 vs W4A16 AWQ evaluation

The executable implementation is the Apptainer runner documented in
[`eval/README.md`](eval/README.md). It deliberately rejects the checked-in
example until every external benchmark adapter, dataset, harness, verifier,
request set, and container image has an immutable revision or digest.

This is a paired non-inferiority evaluation, not a model leaderboard. Its
question is: **does the AWQ checkpoint retain the official FP8 model's quality in the
text workloads for which this repository is intended?** Speed and memory are
reported separately and cannot compensate for a failed quality gate.

The current Qwen model card reports strong results on Terminal-Bench 2.1,
LiveCodeBench v6, GPQA Diamond, and HLE. Those are useful upstream reproduction
anchors, but they are not sufficient by themselves: some are old enough to
have been seen during model training, and none measures our actual Hermes
Agent integration. The primary suite therefore mixes protocol-level tests,
executable tasks, the real Hermes scaffold, and recent reasoning problems.

## Primary suite

| Domain | Evaluation | Run | Main metrics | Why it is here |
|---|---|---:|---|---|
| Tool use | BFCL v4 static single- and multi-turn categories | full, 1 seeded sample | AST accuracy by category; irrelevance accuracy; malformed-call rate | Large, executable check of native function selection, arguments, parallel calls, missing tools/parameters, and long multi-turn context. |
| Agentic / coding | Terminal-Bench 2.1 through Harbor's `hermes` adapter | 30-task locked pilot x 3 seeds, then full x 3 | task reward; pass@1; tool errors; turns; tokens; timeout rate | This is the actual Hermes Agent loop with terminal feedback, not a tool-call proxy. |
| Coding | LiveCodeBench v6 code generation | full lite set, 4 seeds | item-level pass@1; compile/runtime/timeout failure counts | Executable competitive-programming signal and a published Qwen3.8 upstream anchor. |
| Science | GPQA Diamond | full, 4 seeds | exact-choice accuracy by biology/chemistry/physics | Hard expert-written science questions with objective scoring. |
| Current reasoning | MathArena ArXivMath June 2026 plus BrokenArXiv June 2026 | full snapshots, 4 seeds | answer/proof score; correct refusal on false premises | Recent research-level material. BrokenArXiv also catches confident reasoning through a false premise. |
| Multimodal | DocVQA, ChartQA, TextVQA, and a frozen UI-screenshot pack | public validation splits plus private screenshots, 1 deterministic sample | accuracy/ANLS; OCR, grounding, empty-answer failures | Source-precision vision outputs still traverse the quantized language path. |
| Long context | RULER at 4K, 32K, and 128K | full synthetic set per length, 1 seeded sample | per-length string-match accuracy; effective context length; `context_failure` rate | Every other suite here runs at a few thousand tokens. In 48 of 64 layers the DeltaNet recurrent state is the only long-range carrier, and quantization error in that path accumulates through the state rather than being renormalized away per token. Nothing else in this protocol would see it. |

The locked Terminal-Bench pilot must be sampled once, before either model is
run, stratified by task category and difficulty. It is only a cheap failure
detector. A publishable or deployment decision uses the full set.

RULER is here as a paired FP8-versus-AWQ measurement, not as an upstream
reproduction: Qwen publishes RULER for earlier generations but reports no
long-context benchmark for Qwen3.8-27B, so there is no published number to
anchor against. It suits this protocol for three other reasons. The haystacks
are synthesized from a corpus revision, a generator commit, and a fixed seed
rather than pulled from a dataset that can move under a pinned name; scoring is
deterministic string match, so no judge revision enters the gate; and synthetic
haystacks cannot overlap the calibration manifest, so the full and
`calibration-clean` reports agree by construction.

Report RULER per length, never as a single scalar. The whole point is to locate
where degradation begins, and a 4K-to-128K average will hide a cliff at the top
end behind two healthy buckets. The per-length numbers are diagnostics; the
value that enters the macro-average is RULER's own
average over its three lengths, so the suite counts once rather than three times.

The top length is 128K, not the model's full 256K window. The paired protocol
serves both checkpoints with `--max-model-len 262144`, and a prompt that fills
the window leaves no output budget for a thinking model, so a 256K point would
score truncation rather than long-range recall. Raising the served context for
one suite would break the identical-server rule that makes the pairing valid.
`scripts/adapters/ruler.py` rejects any requested length that does not leave
room for `--output-reserve`.
A per-length cliff that survives that averaging is exactly the kind of regression
cluster gate 6 exists to catch. Budget the 128K bucket against KV memory rather
than wall clock: at 16 full-attention layers, 4 KV heads, and 256-wide heads,
one 128K sequence holds roughly 8.5 GB of KV cache, which bounds per-replica
concurrency well below what the shorter suites tolerate. Measured on one H200
serving this checkpoint, four concurrent 128K requests occupied about 25% of the
KV pool against roughly 1.2% for a short-prompt request, so RULER wants about a
dozen requests in flight per replica where the short suites comfortably take
forty or more.

### Secondary diagnostics

- HLE text-only reproduces another score on the Qwen model card and supplies
  broad academic coverage. Keep it secondary because it uses an LLM judge,
  its public test set is no longer clean with respect to pre-training, and its
  rolling corrections must be pinned. We grade with our own pinned judge rather
  than the published `openai/o3-mini`, which means our absolute is not
  comparable to a published HLE score. The judge may be served here or hosted,
  but it is pinned either way: `hf:owner/name@<40-hex>` for weights we hold, or
  `api:provider/model-id@<snapshot>` for one we call. The second is the weaker
  record and should be read as such, since a hosted model cannot be hashed and
  a moving alias could grade the two arms differently. It must not be either
  checkpoint under test. A string match settles every item it can before the
  judge sees anything, since string equality has no false positives, and
  verdicts are cached on (item, normalized answer) so both arms inherit one
  ruling per distinct answer. That last point is not an optimization: it is
  what stops the judge from turning two responses that agreed into a discordant
  pair. Record the judge pin beside the result, and do not let a judged suite
  set a per-suite floor.
- IFBench is a cheap structured-instruction sentinel. Report its 58 constraint
  types separately; do not let a high score mask agent or reasoning losses.
- BFCL `memory` can be added if Hermes persistent memory is a deployment
  requirement. BFCL web-search tasks are excluded from the quality gate because
  changing search results add noise unrelated to weight quantization.
- A private Harbor pack of 20-50 recurring, executable tasks from real Hermes
  usage is highly valuable. Freeze the tasks and verifiers before looking at
  either model's outputs, keep networks disabled unless essential, and never
  include oracle solutions in the agent-visible container.

## Deliberate exclusions

- Exclude confirmed overlaps with Open-SWE-Traces, Lambda Hermes traces,
  selected Nemotron splits, FineWeb-Edu, and selected Cauldron subsets. Publish
  the full score and a primary `calibration-clean` score.
- Do not make MMLU, GSM8K, MATH-500, HumanEval, or MBPP primary evidence. They
  are useful compatibility smoke tests but are saturated and have substantial
  exposure risk for a model released in 2026.
- Do not use SWE-bench Verified as the main coding result. It is expensive,
  scaffold-sensitive, and now has documented contamination and task-quality
  problems. Terminal-Bench under Hermes plus LiveCodeBench gives cleaner
  information for this deployment. SWE-bench Pro can be an optional release
  run if its licensed task snapshot and verifier revisions are available.
- Smoke prompts from `scripts/validate_generate.py` are gates for a broken
  artifact, not capability evidence.

## Contamination control

There are two different contamination questions and both must be reported:

1. **AWQ-calibration overlap.** Materialize every eval's agent-visible prompt
   as JSONL with `id` and `text`, then run `scripts/audit_eval_overlap.py`
   against the exact `manifest.jsonl` saved with the checkpoint. The audit
   looks for exact containment and high token-shingle containment. Any flagged
   item is manually reviewed. Publish both the full benchmark score and a
   `calibration-clean` score with confirmed overlaps removed. The clean score
   is the primary comparison.
2. **Model-training exposure.** We cannot prove what was in Qwen pre-training
   or post-training. Label older public benchmarks as exposure-unknown. Keep
   the June 2026 MathArena snapshots and the private frozen Hermes pack as the
   strongest contemporary evidence; record their publication/creation dates.

Example audit after prompt export:

```bash
python scripts/audit_eval_overlap.py \
  --calibration "${RUN_BASE}/v2/calibration/manifest.jsonl" \
  --eval eval-materialized/bfcl.jsonl \
  --eval eval-materialized/livecodebench-v6.jsonl \
  --eval eval-materialized/gpqa-diamond.jsonl \
  --eval eval-materialized/matharena-2026-06.jsonl \
  --output artifacts/eval-overlap.json
```

Pin dataset revisions before the first inference call. A dataset update is a
new experiment; never silently mix questions or corrected labels between FP8
and AWQ.

Export only the task/question/instruction text for this audit, not a shared
system prompt or tool-schema boilerplate. Shared scaffolding is expected and
would otherwise create uninteresting matches.

## Paired serving protocol

Run `Qwen/Qwen3.8-27B-FP8` and AWQ sequentially on the same eight-GPU host. Pin
the FP8 checkpoint to the revision recorded by the smoke evaluator's
`EVAL_BASELINE_MODEL_REVISION`. Each checkpoint uses eight complete single-GPU
replicas (`TP=1`, `DP=8`), using the same vLLM commit, CUDA stack, chat template,
tool parser, context limit, KV-cache dtype, and scheduler settings. Use the same
served model name so harness behavior cannot branch on the checkpoint label.
Disable MTP/speculative decoding for the primary quality comparison so it
isolates weight recovery. Test MTP separately afterward.

The verified text-serving shape for Qwen3.8 is:

```bash
vllm serve MODEL_PATH \
  --served-model-name openai/qwen38-eval \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --max-model-len 262144 \
  --kv-cache-dtype auto \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Use the model-card generation policy for primary runs: thinking enabled,
`reasoning_effort=xhigh`, `temperature=1.0`, `top_p=0.95`, `top_k=20`,
`min_p=0`, no presence penalty, and repetition penalty 1.0. Fix and record the
seed list. Do not substitute greedy decoding: Qwen explicitly warns that it can
degrade thinking-mode output and cause repetition. A benchmark's mandatory
protocol may override these values, but the override must be identical for both
checkpoints and written into the result metadata.

Randomize task order once and reuse it for both models. For multi-replicate
runs, alternate checkpoint order by seed (FP8/AWQ, then AWQ/FP8) so transient
cluster load or an external service drift is not perfectly confounded with the
candidate checkpoint.

Hermes through Harbor uses an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://REACHABLE_INFERENCE_HOST:8000/v1

harbor run \
  -d terminal-bench/terminal-bench-2-1@PINNED_VERSION \
  -a hermes \
  -m openai/qwen38-eval \
  -n 1
```

Use `-n 1` only for the smoke run. For the pilot and release run, increase to
4-8 concurrent trials after verifying that neither checkpoint is saturated or
memory-constrained; use the same concurrency for both. Start with `-n 8` for the
full run so Harbor can keep the eight data-parallel replicas occupied.

The endpoint must be reachable from the Harbor task containers; `localhost`
usually points at the container, not the inference host. Pin the Harbor commit,
Hermes Agent version, dataset package, Docker image digests, toolset, 90-turn
limit, compression policy, task timeout, and network policy. Start every trial
with empty Hermes sessions, skills, user profile, and memory. Harbor's Hermes
adapter already disables memory/profile and exports ATIF trajectories; retain
those trajectories for paired failure analysis.

Before the full run, require:

- tokenizer and chat-template file hashes identical between checkpoints;
- FP8 and AWQ `/v1/models` and one native tool-call smoke test pass;
- no request truncation at the intended context length;
- the FP8 score is plausibly close to the Qwen-published anchor (GPQA Diamond
  89.2, LiveCodeBench v6 90.3). A large miss means the harness or inference
  protocol must be explained before interpreting the AWQ delta.

## MTP gate

`scripts/validate_generate.py` first asserts that MTP tensors exist and remain
BF16/FP16. Then serve the AWQ checkpoint once without speculation and once with
the sole additional flag:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Export identical request IDs to two JSONL files with `score`, `failed`,
`elapsed_seconds`, `output_tokens`, and, for MTP, vLLM's
`accepted_draft_tokens` and `draft_tokens`. Compare them with:

```bash
python scripts/compare_mtp_results.py \
  --disabled artifacts/eval/awq-mtp-disabled.jsonl \
  --enabled artifacts/eval/awq-mtp-enabled.jsonl \
  --output artifacts/eval/mtp-comparison.json
```

The compatibility gate allows at most a 1-point quality drop, a 1-point
failure-rate increase, and requires at least 40% draft acceptance. Speed is
reported, not gated; run concurrency 1 and production concurrency separately.
The acceptance fields must be deltas, never repeated snapshots of vLLM's
cumulative counters. For a concurrent run, record the server-wide counter delta
on exactly one result row and explicit zeros on the others. The comparator sums
accepted and drafted tokens before dividing. Its per-request latency-derived
speed figure is diagnostic only and is not wall-clock server throughput when
requests overlap; record wall-clock batch throughput separately in the adapter
report.

## Scoring and decision rule

Never compare only aggregate dashboard numbers. Export one JSONL row per
`suite`, `id`, and `replicate`:

```json
{"suite":"gpqa_diamond","id":"example-id","replicate":0,"score":1.0,"must_pass":false,"timeout":false,"empty_answer":false,"repetition_loop":false,"malformed_tool_call":false,"premature_final_answer":false,"context_failure":false}
```

`score` is in `[0, 1]`; use the verifier reward for agent tasks. The two files
must contain exactly the same keys. `scripts/compare_eval_results.py` averages
replicates within an item, computes paired item deltas, and bootstraps over
items rather than pretending repeated generations are independent questions.
Use one fixed suite label for each primary row in the table above; do not split
a weak category into many labels to change its macro weight. Keep category and
failure-mode fields in the raw result for drill-down.

```bash
python scripts/compare_eval_results.py \
  --baseline artifacts/eval/fp8.jsonl \
  --candidate artifacts/eval/awq.jsonl \
  --output artifacts/eval/comparison.json
```

The default automated release gate is intentionally practical for a W4A16 27B model:

1. the equally weighted macro-average point estimate is no worse than FP8 by 3
   percentage points;
2. no single suite's 95% paired-bootstrap interval sits entirely below -5
   points, so one suite can still sink the run, but only on evidence;
3. malformed tool calls, premature final answers, empty answers, repetition
   loops, context failures, and timeouts show no absolute increase over 1 point,
   measured per suite and averaged with equal suite weight;
4. at least 95% of FP8-passed private must-pass tasks still pass under AWQ,
   where passing means reaching the full verifier reward, counted once per task
   rather than once per replicate;
5. every suite with a declared baseline floor clears it, which is the only check
   that would notice a harness broken identically for both checkpoints;
6. no regression cluster has a credible common cause (for example long context,
   parallel tool calls, vision/OCR, chemistry, or dynamic programming) merely
   hidden by the macro average. This last review remains manual; the script
   reports `automated-quality-gate` rather than claiming it passed.

Gates 1 and 2 replace an earlier rule that failed the run whenever any suite's
point estimate fell 3 points. Seven suites is seven chances, and the smaller
ones carry intervals twice the width of that margin: simulated against the
planned sample sizes, that rule rejected 60% of runs with no degradation at all,
while the macro rule rejects 2%. Suites that fall past 3 points without clearing
the 5-point evidential bar are printed as review flags and belong in the manual
cluster review, not in an automatic verdict.

Lock these margins before viewing either model's outputs. If a 3-point loss is
not acceptable for the intended deployment, lower the margin now and increase
the number of independent private tasks accordingly.

For the smaller GPQA and MathArena sets, confidence intervals will be wide.
Repeated samples reduce generation noise but do not create more independent
questions. Treat their per-suite confidence intervals as diagnostics and use
the macro gate plus qualitative regression review. If a deployment needs a
tighter margin than 3 points, enlarge the private task bank; changing bootstrap
settings cannot manufacture statistical power.

Report, but do not fold into quality:

- TTFT p50/p95, output tokens/s, end-to-end task time, and total generated
  tokens at concurrency 1 and the intended production concurrency;
- peak GPU memory, KV-cache utilization, host memory, and model load time;
- tool calls and failed tool calls per solved task.

## Reproducibility record

Keep this beside each result set:

- FP8 model revision and AWQ `run-metadata.json` plus all artifact hashes;
- calibration JSONL hash and overlap report;
- vLLM, Transformers, Harbor, Hermes, benchmark, verifier, and judge revisions;
- GPU type/count, driver, CUDA, container digest, server flags, and environment;
- `TP=1`, `DP=8`, replica routing, request concurrency, and per-request seeds;
- exact prompts/system prompt, chat-template hash, generation parameters, seed
  list, token/context limits, stop strings, and reasoning/tool parsers;
- every raw response, ATIF trajectory, verifier output, item score, latency, and
  error; and
- exclusions, retries, crashes, label corrections, and judge model/version.

## Source choices

- [Qwen3.8-27B-FP8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) —
  official evaluation baseline.
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — official
  benchmark anchors, 262K context, thinking defaults, and generation settings.
- [vLLM Qwen3.8 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) — verified
  reasoning and tool parser configuration.
- [Harbor](https://github.com/harbor-framework/harbor) and its
  [Terminal-Bench guide](https://www.harborframework.com/docs/tutorials/running-terminal-bench)
  — official Terminal-Bench harness with a native Hermes adapter.
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
  — official function-calling evaluator and category definitions.
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) — official
  executable code-generation evaluator and release snapshots.
- [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) — authors' dataset;
  do not publish its question text in reports.
- [MathArena](https://matharena.ai/competitions) — continuously updated math
  competition, ArXivMath, and BrokenArXiv snapshots.
- [HLE](https://github.com/centerforaisafety/hle) and
  [IFBench](https://github.com/instruction-following/IFBench) — secondary broad
  reasoning and instruction-following diagnostics.
