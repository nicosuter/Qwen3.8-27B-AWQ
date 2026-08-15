#!/usr/bin/env python3
"""MTP gate adapter for scripts/run_eval_protocol.py.

The runner invokes this four times: speculation disabled and enabled, at each
requested concurrency. It replays one pinned request set and writes the schema
`scripts/compare_mtp_results.py` consumes.

Acceptance counters come from vLLM's Prometheus endpoint, which reports
cumulative totals. EVAL.md requires deltas: this snapshots the counters around
the run and writes the delta to exactly one row, with explicit zeros on the
others, so summing the column gives the run's real acceptance rather than a
count multiplied by the number of requests.

The per-request speed here is latency-derived and is not server throughput when
requests overlap. Wall-clock batch throughput is recorded separately in the run
metadata, which is the number to quote when comparing concurrency settings.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

SCRIPTS_DIR = Path(__file__).resolve().parent.parent

try:
    from _common import (
        AdapterError,
        build_payload,
        env_path,
        env_str,
        execute_order,
        load_pins,
        module_pin,
        post_chat,
        request_with_retries,
        require_pin,
        unpack_choice,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # loading by file spec puts the repo root on sys.path
    from scripts.adapters._common import (  # type: ignore[no-redef]
        AdapterError,
        build_payload,
        env_path,
        env_str,
        execute_order,
        load_pins,
        module_pin,
        post_chat,
        request_with_retries,
        require_pin,
        unpack_choice,
        write_json,
        write_jsonl,
    )

sys.path.insert(0, str(SCRIPTS_DIR))
from check_mtp_metrics import metric_total, ACCEPTED_NAMES, DRAFTED_NAMES  # noqa: E402


HARNESS_ID = "builtin-mtp-replay-v1"
DEFAULT_MAX_TOKENS = 2048

# One fixed request set: deterministic, moderate-length, and varied enough that
# acceptance is not measured on a single degenerate continuation.
REQUESTS = [
    ("mtp-arith", "Compute 17 * 23 + 41, showing each step briefly."),
    ("mtp-list", "List the first ten prime numbers, comma separated."),
    ("mtp-prose", "Write three sentences about why checksums matter in backups."),
    ("mtp-code", "Write a Python function that reverses a string, no explanation."),
    ("mtp-json", 'Return a JSON object with keys "a" and "b" set to 1 and 2.'),
    ("mtp-steps", "Explain in four steps how to safely restart a network service."),
    ("mtp-table", "Name four SI base units and what each measures."),
    ("mtp-summary", "Summarize what a hash collision is in two sentences."),
]
ANSWER_INSTRUCTION = "Answer directly and stop when the answer is complete."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="replay the pinned request set")
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--request-timeout", type=float, default=600.0)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument(
        "--metrics-url",
        default="",
        help="vLLM /metrics endpoint; defaults to the base URL with /metrics",
    )

    sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    return parser.parse_args(argv)


def self_pin() -> str:
    return module_pin([Path(__file__), Path(__file__).resolve().parent / "_common.py"])


def request_set_pin() -> str:
    payload = json.dumps(REQUESTS, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def expand_requests(concurrency: int) -> tuple[dict[str, str], list[str]]:
    """Repeat the pinned set until it can saturate the requested concurrency."""
    repeats = max(1, -(-concurrency * 2 // len(REQUESTS)))
    prompts, order = {}, []
    for cycle in range(repeats):
        for item_id, prompt in REQUESTS:
            key = item_id if repeats == 1 else f"{item_id}#{cycle}"
            prompts[key] = prompt
            order.append(key)
    return prompts, order


def validate_pins(pins: dict[str, str]) -> None:
    require_pin(pins, "request_set", request_set_pin())
    require_pin(pins, "adapter", self_pin())


def metrics_url_for(base_url: str, override: str) -> str:
    if override:
        return override
    # vLLM serves /metrics at the server root, beside the /v1 API prefix.
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed + "/metrics"


def fetch_metrics(url: str, timeout: float = 30.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode()
    except Exception as error:  # noqa: BLE001 - any failure here is fatal to the gate
        raise AdapterError(f"could not read speculative counters from {url}: {error}") from error


def counter_totals(text: str) -> tuple[float, float]:
    return metric_total(text, ACCEPTED_NAMES) or 0.0, metric_total(text, DRAFTED_NAMES) or 0.0


def run_request(
    item_id: str,
    prompt: str,
    *,
    generation: dict[str, Any],
    model: str,
    seed: int,
    base_url: str,
    api_key: str,
    args: argparse.Namespace,
    client: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    payload = build_payload(
        prompt, generation, model=model, seed=seed,
        max_tokens=args.max_tokens, instruction=ANSWER_INSTRUCTION,
    )
    started = time.monotonic()
    response, _ = request_with_retries(
        item_id, payload, base_url=base_url, api_key=api_key,
        timeout=args.request_timeout, retries=args.retries, client=client,
    )
    elapsed = max(time.monotonic() - started, 1e-6)
    if response is None:
        return {
            "id": item_id, "score": 0.0, "failed": True,
            "elapsed_seconds": round(elapsed, 6), "output_tokens": 0,
            "finish_reason": "timeout",
        }
    content, _, finish_reason, usage = unpack_choice(item_id, response)
    tokens = usage.get("completion_tokens")
    return {
        "id": item_id,
        # The gate compares quality between speculation modes; a reply that
        # arrived and said something is the unit of quality here.
        "score": 1.0 if content.strip() else 0.0,
        "failed": not content.strip(),
        "elapsed_seconds": round(elapsed, 6),
        "output_tokens": int(tokens) if isinstance(tokens, int) else 0,
        "finish_reason": finish_reason,
    }


def command_run(
    args: argparse.Namespace, client: Callable[..., dict[str, Any]] = post_chat
) -> int:
    validate_pins(load_pins())
    mode = env_str("EVAL_MTP_MODE")
    if mode not in ("disabled", "enabled"):
        raise AdapterError(f"EVAL_MTP_MODE must be disabled or enabled; got {mode!r}")
    concurrency = int(env_str("EVAL_CONCURRENCY"))
    if concurrency < 1:
        raise AdapterError("EVAL_CONCURRENCY must be at least 1")

    run_dir = env_path("EVAL_RUN_DIR")
    results_path = env_path("EVAL_RESULTS_JSONL")
    model = env_str("EVAL_SERVED_MODEL")
    base_url = env_str("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    seed = int(os.environ.get("EVAL_SEED", "0"))
    generation = json.loads(env_str("EVAL_GENERATION_JSON"))
    metrics_url = metrics_url_for(base_url, args.metrics_url)

    # Eight requests at concurrency 384 would measure an almost-empty pipe, and
    # acceptance at batch 1 is not acceptance under load. Replay the base set
    # enough times to keep the batch full; the count is derived from the
    # concurrency, so both modes at one concurrency share their keys.
    prompts, order = expand_requests(concurrency)

    before_accepted = before_drafted = 0.0
    if mode == "enabled":
        before_accepted, before_drafted = counter_totals(fetch_metrics(metrics_url))

    started = time.monotonic()
    rows = execute_order(
        order,
        lambda item_id: run_request(
            item_id, prompts[item_id], generation=generation, model=model, seed=seed,
            base_url=base_url, api_key=api_key, args=args, client=client,
        ),
        concurrency,
    )
    wall_clock = max(time.monotonic() - started, 1e-6)

    accepted = drafted = 0.0
    if mode == "enabled":
        after_accepted, after_drafted = counter_totals(fetch_metrics(metrics_url))
        accepted = after_accepted - before_accepted
        drafted = after_drafted - before_drafted
        if accepted < 0 or drafted < 0:
            raise AdapterError("speculative counters decreased between snapshots")
        if drafted == 0:
            raise AdapterError(
                "speculation was requested but vLLM drafted no tokens; "
                "the server is not running with a speculative config"
            )
        # The delta is server-wide, so it belongs on one row. Spreading it would
        # multiply the run's draft count by the number of requests.
        for index, row in enumerate(rows):
            row["accepted_draft_tokens"] = int(accepted) if index == 0 else 0
            row["draft_tokens"] = int(drafted) if index == 0 else 0

    write_jsonl(results_path, rows)
    total_output = sum(row["output_tokens"] for row in rows)
    write_json(
        run_dir / "metadata" / f"mtp-{mode}-c{concurrency}.json",
        {
            "mode": mode,
            "concurrency": concurrency,
            "served_model": model,
            "requests": len(rows),
            "request_set": request_set_pin(),
            "adapter": self_pin(),
            "max_tokens": args.max_tokens,
            "generation": generation,
            "accepted_draft_tokens": int(accepted),
            "draft_tokens": int(drafted),
            "acceptance_rate": round(accepted / drafted, 6) if drafted else None,
            # Wall-clock throughput is the honest number when requests overlap;
            # the comparator's per-request figure is latency-derived.
            "wall_clock_seconds": round(wall_clock, 3),
            "output_tokens_total": total_output,
            "wall_clock_output_tokens_per_second": round(total_output / wall_clock, 2),
        },
    )
    print(
        f"mtp mode={mode} concurrency={concurrency} requests={len(rows)} "
        f"tokens={total_output} wall_clock={wall_clock:.1f}s "
        f"throughput={total_output / wall_clock:.1f} tok/s",
        flush=True,
    )
    return 0


def command_pin() -> int:
    print(json.dumps({"request_set": request_set_pin(), "adapter": self_pin()}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "pin":
        return command_pin()
    return command_run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
