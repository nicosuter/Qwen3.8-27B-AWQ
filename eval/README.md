# Apptainer evaluation runner

`eval/scripts/run_eval_protocol.py` implements the experiment controls in
`EVAL.md`. It runs vLLM inside one immutable Apptainer image and invokes pinned,
benchmark-specific adapters on the host. Keeping adapters outside the server
image is intentional: Harbor needs access to its task-container runtime, GPQA
may require authenticated dataset access, and the private screenshot pack must
not be baked into a public SIF.

The checked-in `protocol.example.json` is a fail-closed template, not a runnable
lock file. Copy it to persistent scratch and replace every `REPLACE_...` command
and `PINNED_VERSION` with reviewed adapter argv and immutable revisions. The
runner refuses placeholders.

## What gets measured

`eval/eval-suite-v1.json` is the one place that says which suites this protocol
measures, and everything derives from it: `run_eval_protocol.py` reads its
`REQUIRED_SUITES` from there rather than declaring one, the paired sbatch
resolves its lane list through it, and `compare_eval_results.py --eval-suite v1`
refuses results carrying a suite it does not contain.

That set previously lived in four places at once, and they drifted. RULER was
required by the runner and absent from the config that ran everything else, so
the campaign could not have been launched as specified; meanwhile `aa_lcr` and
`aa_omniscience` were scored into a macro they were never part of. Both are now
loud failures.

```bash
python3 eval/scripts/eval_suite.py                      # what v1 measures, and what is parked
python3 eval/scripts/eval_suite.py --select ruler bfcl_v4   # resolve a batch
```

Changing what is measured means writing `eval-suite-v2.json` and pointing
`eval_suite` in the protocol at it.

## Batches

`eval/batches.json` divides that set across jobs. Batches are scheduling and not
pre-registration, which is why they live in their own file, but they are held to
the definition: the scoring batches must partition the suite set exactly. A gap
means running every batch still would not produce the protocol's macro, and an
overlap means the second job silently overwrites the first in the shared run
directory. Both are refused.

| batch | suites | why |
|---|---|---|
| `prepare` | all seven, `--phase prepare` | no GPU; resolves every pin and materializes every dataset, so a bad pin fails before an allocation |
| `short-context` | bfcl_v4, gpqa_diamond, livecodebench_v6, matharena_2026_06, mmmu_pro, multimodal | everything that tolerates forty or more requests in flight |
| `long-context` | ruler | one 128k sequence holds roughly 8.5 GB of KV, so it wants about a dozen per replica where the others take forty |

```bash
python3 eval/scripts/eval_suite.py --batches
PAIRED_BATCH=short-context sbatch eval/slurm/paired-suite-eval.sbatch
```

RULER is separated because sharing a server with forty-way lanes means one
starves the other, and it would do so to whichever arm happened to be running.
The rest are kept together deliberately: lanes fill each other's tails, and each
split costs another server load per arm. A batch's own comparison is
`--allow-partial` by construction; the protocol's macro appears once every
scoring batch has written into the same run directory.

## Build and run

The repository defaults to the official Qwen3.8-specific vLLM image for
linux/amd64/CUDA 13.0, pinned to its platform manifest digest. Build the SIF on
a node with Apptainer:

```bash
sbatch common/slurm/build-eval-apptainer.sbatch
```

The default output is `$RUN_BASE/containers/qwen38-eval-vllm.sif`. OCI layers
and the expanded build workspace are cached under `$RUN_BASE`, so a retry does
not download the roughly 7 GiB compressed image again. The builder tests the
SIF before atomically installing it and writes a sibling `.sha256` file.

Once the AWQ checkpoint exists, the short end-to-end serving gate is:

```bash
sbatch quant/slurm/serve-smoke.sbatch
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
sbatch --export=ALL eval/slurm/protocol.sbatch
```

To freeze and audit prompts without allocating GPUs, run only preparation:

```bash
python3 eval/scripts/run_eval_protocol.py \
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
python3 eval/scripts/run_eval_protocol.py \
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

`eval/scripts/adapters/gpqa_diamond.py` implements the contract below and is the
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
python3 eval/scripts/adapters/gpqa_diamond.py pin
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
  "prepare": ["python3", "eval/scripts/adapters/gpqa_diamond.py", "prepare", "--split", "gpqa_diamond"],
  "run": ["python3", "eval/scripts/adapters/gpqa_diamond.py", "run", "--concurrency", "384"]
}
```

Adapter argv runs with the repository root as the working directory, so the
relative path above resolves; use an absolute path if you invoke the runner from
elsewhere. Point the argv at the project venv's interpreter rather than bare
`python3` — the cluster's system Python is 3.9, which cannot even import the
adapters, and the adapters need `datasets` and `transformers` from that venv
anyway:

```json
"prepare": ["/scratch/$USER/qwen38-27b-awq/repo/.venv/bin/python",
            "eval/scripts/adapters/gpqa_diamond.py", "prepare", "--split", "gpqa_diamond"]
```

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
python3 eval/scripts/adapters/gpqa_diamond.py probe \
  --base-url "$EVAL_BASE_URL" --model openai/qwen38-eval
```

It sends one request with the model-card policy and fails if the server rejects
a field, returns no reasoning while thinking is enabled, or produces no parsable
answer line.

## MMMU-Pro

`eval/scripts/adapters/mmmu_pro.py` covers the multimodal reasoning the DocVQA and
ChartQA suite does not. Those are perception, scored near ceiling, and 86.5% of
their items came back identical on both checkpoints in the last paired run: they
measure the vision tower, which this recipe leaves in source precision. MMMU-Pro
asks college-level questions about an image across thirty subjects with ten
options rather than four, so the reasoning after the perception runs through the
quantized decoder.

It uses the standard config rather than the vision config, where the question
itself is rendered into the image. That variant measures OCR of the prompt,
which the unquantized tower would answer, and is a different question from the
one this protocol asks.

```bash
python3 eval/scripts/adapters/mmmu_pro.py pin --resolve
```

## Reference adapter: RULER

`eval/scripts/adapters/ruler.py` synthesizes every haystack locally. It is not an
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
python3 eval/scripts/adapters/ruler.py pin --corpus "$RUN_BASE/eval/haystack"
```

`prepare` re-hashes the corpus and refuses to run if it no longer matches
`pins.dataset`, so a silently edited haystack cannot reach a scored run. Then
use:

```json
{
  "name": "ruler",
  "replicates": 1,
  "pins": { "...": "output of the pin command" },
  "prepare": ["python3", "eval/scripts/adapters/ruler.py", "prepare",
              "--lengths", "4096,32768,131072", "--synthesis-seed", "38027",
              "--corpus", "/scratch/.../haystack", "--tokenizer", "/scratch/.../v2/model"],
  "run": ["python3", "eval/scripts/adapters/ruler.py", "run", "--concurrency", "96"]
}
```

Size `--concurrency` from KV cache, not intuition. Measured on one H200 serving
this checkpoint: a short-prompt request occupies about 1.2% of the KV pool and a
128K request about 6%, so a replica comfortably holds roughly 48 short requests
or a dozen long ones. Adapter concurrency is total in-flight across the whole
endpoint, so multiply by the number of data-parallel replicas: at `DP=8` that is
about 384 for the short suites and 96 for RULER. At 8 in flight on one GPU the
server generated 480 tokens/s in total, about 60 per stream, which is roughly
single-stream speed with the GPU idle between tokens.

Keep the number identical for both checkpoints. Batch composition changes
reduction order, so the same seed does not produce the same tokens at a
different concurrency, and a run at 8 is not comparable to a run at 384.

`prepare` needs a tokenizer to hit its length targets and defaults to
`$OUTPUT_DIR`; the runner already asserts that both checkpoints share a
tokenizer hash, so either one gives the same items. It prints the total prompt
tokens per replicate per checkpoint, which is the number to look at before
committing the GPU hours: at the default ten items per task that is 210 items
and roughly 12M prompt tokens per checkpoint.

## The remaining four adapters

Each prints its pins with `pin`, refuses a branch name where a commit belongs,
and hashes its own source together with `_common.py` so an edit without a repin
stops the run. The two Harbor adapters hash `_harbor.py` as well, since that is
where their scoring rules live, though neither is in the protocol right now. Three of them score data that differs from what
`EVAL.md` names, and each records the substitution in its own run metadata:

| suite | adapter | data actually scored | verifier |
|---|---|---|---|
| `bfcl_v4` | `bfcl.py` | BFCL **v3** static split, 1240 items; no v4 exists on the Hub | AST match |
| `livecodebench_v6` | `livecodebench.py` | release v6, 175 problems, 7000 tests | sandboxed execution, pass@1 |
| `matharena_2026_06` | `matharena.py` | AIME 2026 + Apex shortlist; the named 2026-06 snapshots are **not published** | exact integer |
| `multimodal` | `multimodal.py` | DocVQA, ChartQA, TextVQA; the private UI pack does **not exist** here | ANLS / relaxed / VQA |

```bash
python3 eval/scripts/adapters/bfcl.py           pin --resolve-dataset
python3 eval/scripts/adapters/livecodebench.py  pin --resolve-dataset
python3 eval/scripts/adapters/matharena.py      pin --resolve
python3 eval/scripts/adapters/multimodal.py     pin --resolve
python3 eval/scripts/adapters/terminal_bench.py pin --dataset-version <v> --harbor-version <v> --task-checksums <set>
```

SWE-bench Pro is not in the protocol. Harbor's Singularity backend hardcodes
`--fakeroot`, this account has no `/etc/subuid` mapping, and the bundled
`faked` needs a newer glibc than the task images carry, so the containers
never start. `eval/scripts/adapters/swebench_pro.py`,
`eval/scripts/swebenchpro_subset.py` and `eval/scripts/bake_harbor_sifs.py` are kept
against that being fixed; the runner does not reference them.

Three things worth knowing before running them.

**LiveCodeBench executes model-generated code.** Each solution runs in a fresh
temporary directory under CPU-time, file-descriptor and address-space limits,
but that is a resource cap, not a security sandbox. Do not point it at a
filesystem it could damage. Its private tests arrive as pickled data, decoded
with an unpickler that refuses every class lookup so a dataset revision cannot
execute code at prepare time.

**Terminal-Bench does not enforce the frozen sequence.** Harbor schedules its
own trials, so the order fixes the task set only; the metadata records
`task_order_enforced: set-only`. It also needs a container runtime for the task
pack — `--environment singularity` works where Docker is unavailable, provided
the pinned pack ships singularity-compose files.

**BFCL is the only suite where `malformed_tool_call` can be true.** It serves
tools natively; every other suite runs without tools, so that flag is
structurally false there and contributes nothing to the failure-mode gate.

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
by `eval/scripts/compare_mtp_results.py`: `id`, `score`, `failed`,
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

## Running a suite through EvalScope

EvalScope implements the task definitions for eight of our suites and grades
BFCL with Gorilla's own `bfcl-eval`. What it does not have is an immutable
dataset pin or the paired statistics, so `eval/scripts/evalscope_bridge.py` supplies
both ends and the comparator is unchanged.

Set `PAIRED_RUNNER=evalscope` and a lane runs EvalScope instead of an adapter.
Everything else is the same: one server per variant, the same lane fan-out and
failure ledger, the same reuse check, the same comparison. A lane writes the
same two files either way, `raw/<variant>/<suite>-r<n>.jsonl` and
`metadata/<suite>-<variant>-r<n>.json`, so nothing downstream knows which
harness produced a result.

### Pinning

EvalScope names ModelScope mirrors as its dataset ids, but its adapters read the
original Hugging Face column names, so a mirror is a straight copy and each
benchmark can be pointed at our own snapshot instead. That is what makes this a
cross-check rather than a second measurement: both harnesses read identical
bytes at the same commit, so a disagreement is scoring, not data.

Pinning has to be supplied because `DefaultDataAdapter.load_subset` builds its
loader without passing the `version` that `DataLoader` accepts and forwards to
`load_dataset(revision=...)`. A benchmark run from the Hub therefore tracks
whatever `main` is that day. Instead:

    python eval/scripts/evalscope_bridge.py materialize \
        --repo TIGER-Lab/MMLU-Pro --revision <40-hex> \
        --into eval-materialized/evalscope

`cais/hle` and `Idavidrein/gpqa` are gated. Materialize those through
`eval/scripts/with-hf-token.sh`, which keeps the credential in one process and refuses
to hand it to `sbatch`, because Slurm persists a job's environment to its spool
directories.

BFCL needs building rather than downloading: EvalScope reads one record carrying
prompt and key together, while upstream ships them as two files.

    python eval/scripts/evalscope_bridge.py bfcl-dataset --into eval-materialized/evalscope

### Two things worth knowing before trusting a number

**Sample ids are positional.** EvalScope defaults to `auto_id=True`, so a
sample id is that item's index in the loaded split. Two arms scored against
revisions differing by one row would join perfectly and compare unrelated
questions. The bridge therefore keys rows on a digest of the item's own prompt
and target, so a mismatch surfaces as a missing key rather than a plausible
wrong number.

**No per-request seed is sent, deliberately.** `TaskConfig.seed` only feeds
`seed_everything()` and the loader's shuffle; it is never copied into
`GenerateConfig`. Leave it that way. One seed on every request makes each item
draw the same uniform stream against different logits, correlating their
sampling noise, and the item-clustered bootstrap assumes items are independent.
It would report an interval narrower than the truth.

### LiveCodeBench

Scoring runs code the model wrote, and EvalScope executes it in the local
environment by default with no generate-only mode. Load
`eval/scripts/evalscope_plugins/lcb_deferred.py` and the generating pass returns a
deferred marker without reaching the executor; the isolated pass then runs
`--use-cache <dir> --rerun-review` with `execute: True`. Deferred rows are
marked, and the comparator refuses any file containing one, so a pass that
generated but did not execute cannot be read as a suite that scored zero.
