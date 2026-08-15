# Apptainer evaluation runner

`scripts/run_eval_protocol.py` implements the experiment controls in
`EVAL.md`. It runs vLLM inside one immutable Apptainer image and invokes pinned,
benchmark-specific adapters on the host. Keeping adapters outside the server
image is intentional: Harbor needs access to its task-container runtime, GPQA
may require authenticated dataset access, and the private screenshot pack must
not be baked into a public SIF.

The checked-in `protocol.example.json` is a fail-closed template, not a runnable
lock file. Copy it to persistent scratch and replace every `REPLACE_...` command
and `PINNED_VERSION` with reviewed adapter argv and immutable revisions. The
runner refuses placeholders.

## Build and run

The repository defaults to the official Qwen3.8-specific vLLM image for
linux/amd64/CUDA 13.0, pinned to its platform manifest digest. Build the SIF on
a node with Apptainer:

```bash
sbatch slurm/build-eval-apptainer.sbatch
```

The default output is `$RUN_BASE/containers/qwen38-eval-vllm.sif`. OCI layers
and the expanded build workspace are cached under `$RUN_BASE`, so a retry does
not download the roughly 7 GiB compressed image again. The builder tests the
SIF before atomically installing it and writes a sibling `.sha256` file.

Once the AWQ checkpoint exists, the short end-to-end serving gate is:

```bash
sbatch slurm/serve-smoke.sbatch
```

It rejects an unpacked checkpoint before starting vLLM, then serves one H200
twice with a 4K context window: once without speculation and once with native
MTP. Both modes must pass `/v1/models` and a chat completion. The MTP run also
uses a longer request to require nonzero draft tokens and at least 40% draft
acceptance. This is a load/generation smoke, not the scored protocol below.

After filling the protocol lock file, choose an address on the inference host
that Harbor task containers can reach. `localhost` is normally wrong from
inside those containers.

```bash
export RUN_BASE="/scratch/$USER/qwen38-27b-awq"
export EVAL_CONFIG="$RUN_BASE/eval/protocol.json"
export EVAL_BASE_URL='http://<inference-host-address>:8000/v1'
sbatch --export=ALL slurm/eval.sbatch
```

To freeze and audit prompts without allocating GPUs, run only preparation:

```bash
python3 scripts/run_eval_protocol.py \
  --config "$EVAL_CONFIG" \
  --run-dir "$RUN_BASE/v2/eval/release-1" \
  --phase prepare
```

If overlap candidates are found, preparation stops before inference. Review
`overlap/audit.json` and write decisions in JSON of this form:

```json
{
  "reviews": [
    {
      "eval_file": "/absolute/path/to/materialized/bfcl_v4.jsonl",
      "eval_id": "item-id",
      "calibration_id": "calibration-id",
      "confirmed_overlap": false,
      "note": "shared API name only"
    }
  ]
}
```

Every flagged triple needs an explicit Boolean decision. Confirmed overlaps are
removed symmetrically from FP8 and AWQ only for the primary
`calibration-clean` report; the full report is retained.

Resume the same locked run without changing its protocol JSON:

```bash
python3 scripts/run_eval_protocol.py \
  --config "$EVAL_CONFIG" \
  --image "$EVAL_APPTAINER_IMAGE" \
  --run-dir "$RUN_BASE/v2/eval/release-1" \
  --overlap-review "$RUN_BASE/eval/overlap-review.json" \
  --phase run
```

For Slurm, export the review path as `EVAL_OVERLAP_REVIEW`. A review path may
also be locked in `overlap_review` before preparation, but changing a protocol
JSON after its run directory is created is intentionally rejected.

## Reference adapter: GPQA Diamond

`scripts/adapters/gpqa_diamond.py` implements the contract below and is the
worked example for the remaining suites. `prepare` materializes one row per
question with a per-item deterministic option order, keeps the shared
answer-format instruction out of the prompt text so the overlap audit sees only
the question and its options, and writes the answer key to
`materialized/gpqa_diamond.key.json` inside the run directory. `run` replays the
frozen task order, applies the generation policy, and emits the paired result
schema plus category, timings, token counts, and a retained raw response per
item.

Its pins are self-checked and fail closed. `dataset` must be the 40-character
GPQA commit, not a branch; `harness` is `builtin-gpqa-mcq-v1` because the
adapter is its own harness; `verifier` is `exact-choice-v1`; and `adapter` is
the adapter's own source hash, so editing the file without repinning stops the
run. Print the current value with:

```bash
python3 scripts/adapters/gpqa_diamond.py pin
```

Then replace the `gpqa_diamond` suite in your protocol JSON with:

```json
{
  "name": "gpqa_diamond",
  "replicates": 4,
  "pins": {
    "dataset": "<40-char GPQA dataset commit>",
    "harness": "builtin-gpqa-mcq-v1",
    "verifier": "exact-choice-v1",
    "adapter": "<output of the pin command>"
  },
  "prepare": ["python3", "scripts/adapters/gpqa_diamond.py", "prepare", "--split", "gpqa_diamond"],
  "run": ["python3", "scripts/adapters/gpqa_diamond.py", "run", "--concurrency", "8"]
}
```

Adapter argv runs with the repository root as the working directory, so the
relative path above resolves; use an absolute path if you invoke the runner from
elsewhere.

GPQA is a gated dataset; the account running `prepare` needs accepted terms and
a token in `HF_HOME`. Scoring is exact-choice, so no judge model is involved.
Every item carries `must_pass: false`: the must-pass bank is the private task
set, not a public benchmark. `malformed_tool_call` is always false because the
suite is served without tools, and `premature_final_answer` means the server
returned an answer with zero reasoning tokens while thinking was enabled.
Timeouts are recorded as failed items rather than retried; transport faults are
retried and then abort the run, and a 4xx other than 429 aborts immediately, so
infrastructure noise never scores as model behavior.

`--max-tokens` defaults to 65536. xhigh thinking on graduate-level science runs
long, and a truncated reply is scored 0 with `context_failure` set; raise the
cap rather than accepting truncation, and keep it identical for both
checkpoints.

Before the first scored run, confirm the server actually honors the generation
policy. `reasoning_effort` and `chat_template_kwargs` are the fields a vLLM
build is most likely to reject or ignore, and either failure would otherwise
surface as an aborted run or as silently non-thinking output:

```bash
python3 scripts/adapters/gpqa_diamond.py probe \
  --base-url "$EVAL_BASE_URL" --model openai/qwen38-eval
```

It sends one request with the model-card policy and fails if the server rejects
a field, returns no reasoning while thinking is enabled, or produces no parsable
answer line.

## Reference adapter: RULER

`scripts/adapters/ruler.py` synthesizes every haystack locally. It is not an
upstream RULER reproduction and its scores are not comparable to published RULER
numbers; it exists to measure FP8 against AWQ on byte-identical items, which is
what `EVAL.md` asks of this suite. Seven synthetic tasks are generated per
length: `niah_single`, `niah_multikey`, `niah_multivalue`, `niah_multiquery`,
`vt`, `cwe`, and `fwe`. RULER's two QA tasks are deliberately excluded, since
they would pull in an external QA dataset and a second verifier.

Lengths are 4096, 32768, and 131072. `prepare` refuses any length that does not
leave `--output-reserve` tokens inside `--max-model-len`, so the combination
that would score truncation instead of recall cannot be configured by accident:

```console
$ ruler.py prepare --lengths 262144 ...
error: lengths [262144] exceed the usable window: --max-model-len 262144 minus
--output-reserve 16384 leaves 245504 prompt tokens. A prompt that fills the
whole window leaves no room for an answer.
```

Because prompts are built to fit, `context_failure` on this suite means output
truncation only, never a prompt that never fit. Rows carry `length` and `task`
as category fields under the single `ruler` label, and the run metadata reports
`accuracy_by_length` and `context_failures_by_length` so a cliff at 128K is
visible without the suite counting more than once in the macro gate.

The haystack corpus is pinned by content hash rather than by a hub revision: a
revision can be re-tagged, a hash cannot. Point `--corpus` at a UTF-8 text file
or a directory of `.txt` files on persistent scratch and produce all four pins
with:

```bash
python3 scripts/adapters/ruler.py pin --corpus "$RUN_BASE/eval/haystack"
```

`prepare` re-hashes the corpus and refuses to run if it no longer matches
`pins.dataset`, so a silently edited haystack cannot reach a scored run. Then
use:

```json
{
  "name": "ruler",
  "replicates": 1,
  "pins": { "...": "output of the pin command" },
  "prepare": ["python3", "scripts/adapters/ruler.py", "prepare",
              "--lengths", "4096,32768,131072", "--synthesis-seed", "38027",
              "--corpus", "/scratch/.../haystack", "--tokenizer", "/scratch/.../v2/model"],
  "run": ["python3", "scripts/adapters/ruler.py", "run", "--concurrency", "8"]
}
```

`prepare` needs a tokenizer to hit its length targets and defaults to
`$OUTPUT_DIR`; the runner already asserts that both checkpoints share a
tokenizer hash, so either one gives the same items. It prints the total prompt
tokens per replicate per checkpoint, which is the number to look at before
committing the GPU hours: at the default ten items per task that is 210 items
and roughly 12M prompt tokens per checkpoint.

## Adapter contract

Adapter commands are argv arrays and are never evaluated through a shell. They
receive paths and policy through environment variables.

Every suite also has an explicit `pins` object for its dataset, harness,
verifier, and adapter. MTP pins its request set and adapter. The runner rejects
missing or placeholder pins and passes the resolved object as `EVAL_PINS_JSON`;
adapters should verify the installed checkout/image/dataset against it before
materializing prompts.

During `prepare` or `prepare-pilot`, the adapter must write
`EVAL_PROMPTS_JSONL`, one row per item, with unique `id` and agent-visible
`text`. Do not include a shared system prompt or tool-schema boilerplate. The
pilot must contain exactly 30 rows and also supply `category` and `difficulty`.
Preparation must pin the dataset/verifier revision and finish before any model
request. The runner writes `EVAL_TASK_ORDER_JSON` once from those IDs.

During `run` or `run-pilot`, the adapter must:

- use `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `EVAL_SERVED_MODEL`;
- consume items in `EVAL_TASK_ORDER_JSON` order without resampling;
- apply `EVAL_GENERATION_JSON`, except for a benchmark-mandated override that
  it records in every run's metadata;
- use `EVAL_SEED` and write all items, including failures, to
  `EVAL_RESULTS_JSONL`;
- start Hermes trials with empty session, skills, profile, and memory, retain
  ATIF trajectories, and enforce the pinned 90-turn/timeout/network policy;
- emit one row per item with the exact `suite`, `id`, integer `replicate`, a
  `score` in `[0,1]`, and Boolean values for `must_pass`, `timeout`,
  `empty_answer`, `repetition_loop`, `malformed_tool_call`,
  `premature_final_answer`, and `context_failure`.

Keep category, failure details, timings, token counts, raw-response paths,
verifier output paths, and protocol overrides as additional fields. For the
multimodal suite, keep `docvqa`, `chartqa`, `textvqa`, and `private-ui` as a
category field under the one `multimodal` suite label. For the `ruler` suite,
keep the context length (`4k`, `32k`, `128k`, `256k`) and the RULER task name as
category fields under the one `ruler` label, so per-length accuracy can be
reported without the suite counting more than once in the macro-average gate.
The RULER adapter must synthesize its haystacks from the pinned corpus revision
and `--synthesis-seed` rather than downloading prebuilt prompts, and both models
must receive byte-identical materialized items.

The MTP adapter runs four times: disabled/enabled at concurrency 1 and 8. It
receives `EVAL_MTP_MODE` and `EVAL_CONCURRENCY`, and writes the schema consumed
by `scripts/compare_mtp_results.py`: `id`, `score`, `failed`,
`elapsed_seconds`, `output_tokens`, plus `accepted_draft_tokens` and
`draft_tokens` when enabled.

## What the runner enforces

- exact primary suite labels and replicate counts;
- the four fixed seeds, shared materialized items, and one shared randomized
  order;
- FP8/AWQ then AWQ/FP8 alternating checkpoint order;
- TP=1, DP=8, 262K context, parser settings, no primary speculation, and the
  model-card generation policy;
- exact tokenizer and effective chat-template hashes before scoring;
- `/v1/models` readiness and a native function-call smoke for every server;
- paired result keys, item-clustered bootstrap gates, failure-mode gates,
  must-pass retention, contamination review, and separate full/clean reports;
- native MTP disabled/enabled comparisons at both requested concurrencies;
- SIF, configuration, calibration, environment, raw result, and report hashes.

The final `decision.json` deliberately leaves
`manual_regression_cluster_review_required` true. A passing automated gate is
not a deployment decision until trajectories and regression clusters are
reviewed and the FP8 anchors are judged plausible.
