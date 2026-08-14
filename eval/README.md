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

Resolve a Qwen3.8-capable official vLLM image to an immutable registry digest,
then build the SIF on a node with Apptainer:

```bash
export EVAL_VLLM_BASE_IMAGE='docker://vllm/vllm-openai@sha256:<64-hex-digest>'
export EVAL_APPTAINER_IMAGE="/scratch/$USER/containers/qwen38-eval-vllm.sif"
sbatch --export=ALL slurm/build-eval-apptainer.sbatch
```

After filling the protocol lock file, choose an address on the inference host
that Harbor task containers can reach. `localhost` is normally wrong from
inside those containers.

```bash
export RUN_BASE="/scratch/$USER/qwen38-27b-awq"
export EVAL_CONFIG="$RUN_BASE/eval/protocol.json"
export EVAL_APPTAINER_IMAGE="/scratch/$USER/containers/qwen38-eval-vllm.sif"
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
category field under the one `multimodal` suite label.

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
