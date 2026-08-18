#!/usr/bin/env python3
"""Hold the KV budget for every lane on one server, and correct it as it runs.

Lanes are separate processes, so the budget they share has to live somewhere.
It lives here, behind a unix socket, alongside a loop that watches the server's
own preemption counter and resizes the budget from it.

Why a controller at all: the reservation each request makes is a median, so it
is wrong on the tail by construction, and the tail is where the cache overfills.
Preemption is the only observable that distinguishes a cache that is working
from one that is thrashing -- queue depth cannot, because a deep queue is the
intended state whenever items outnumber capacity.

    eval/scripts/admission_broker.py \\
        --socket "$RUN_DIR/admission.sock" \\
        --metrics-url http://127.0.0.1:8000/metrics \\
        --server-log "$RUN_DIR/logs/server.log"
"""

import argparse
import importlib.util
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import sample_telemetry  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_admission", SCRIPTS / "adapters" / "_admission.py"
)
admission = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(admission)

# Leave a fifth of the pool unreserved. The reservation is a median, so roughly
# half the requests in flight are running longer than they booked; the headroom
# is what absorbs that between one controller tick and the next.
POOL_FRACTION = 0.8

# vLLM prints this once at startup, after it has actually allocated the cache.
# It is the only authoritative statement of how many tokens fit -- every other
# route to the number involves multiplying out a config and hoping.
POOL_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")


def pool_from_log(text: str) -> int | None:
    match = POOL_RE.search(text)
    return int(match.group(1).replace(",", "")) if match else None


def ceiling_from_pool(pool_tokens: int) -> int:
    return int(pool_tokens * POOL_FRACTION)


def controller_step(
    broker: Any, sample: dict[str, float], previous_preemptions: float, ceiling: int
) -> float:
    """Apply one observation. Returns the preemption count to compare next.

    An empty sample means the server did not answer -- restarting, or not up
    yet. Holding still is the only safe reading: growing would size the budget
    against a cache that is not there, and backing off would punish the lanes
    for the scrape failing.
    """
    if not sample:
        return previous_preemptions
    total = float(sample.get("vllm:num_preemptions_total", previous_preemptions))
    # The counter is monotonic across the whole server lifetime, so only the
    # change since the last tick says anything about the budget we just set.
    preempted = max(0.0, total - previous_preemptions)
    broker.resize(
        admission.next_capacity(
            broker.budget.capacity,
            preempted=preempted,
            waiting=float(sample.get("vllm:num_requests_waiting", 0.0)),
            ceiling=ceiling,
        )
    )
    return total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument(
        "--server-log",
        type=Path,
        help="where to read the reported KV pool size from; falls back to --pool-tokens",
    )
    parser.add_argument(
        "--pool-tokens",
        type=int,
        help="KV pool size, when the log is not available to read it from",
    )
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument(
        "--status",
        type=Path,
        help="append one line per tick, so a run can be explained afterwards",
    )
    return parser.parse_args(argv)


def resolve_pool(args: argparse.Namespace) -> int:
    if args.server_log and args.server_log.is_file():
        found = pool_from_log(args.server_log.read_text(encoding="utf-8", errors="replace"))
        if found:
            return found
    if args.pool_tokens:
        return args.pool_tokens
    raise SystemExit(
        "could not determine the KV pool size; pass --pool-tokens or a --server-log "
        "containing vLLM's 'GPU KV cache size' line"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ceiling = ceiling_from_pool(resolve_pool(args))
    # The path that actually gets bound may not be the one asked for: a run
    # directory deep enough to exceed sun_path is redirected to a short one,
    # by the same rule the lanes apply, so both still meet.
    bound = admission.socket_path(args.socket)
    bound.parent.mkdir(parents=True, exist_ok=True)
    if bound.exists():
        bound.unlink()
    broker = admission.serve(args.socket, capacity=ceiling)
    print(f"admission broker on {bound} ceiling={ceiling} tokens", flush=True)

    running = {"go": True}

    def stop(_signum, _frame):
        running["go"] = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    previous = 0.0
    first = sample_telemetry.scrape(args.metrics_url)
    if first:
        previous = float(first.get("vllm:num_preemptions_total", 0.0))
    while running["go"]:
        time.sleep(args.interval)
        sample = sample_telemetry.scrape(args.metrics_url)
        before = broker.budget.capacity
        previous = controller_step(broker, sample, previous, ceiling)
        if args.status and sample:
            line = {
                "capacity": broker.budget.capacity,
                "outstanding": broker.outstanding(),
                "preemptions_total": previous,
                "waiting": sample.get("vllm:num_requests_waiting"),
            }
            with args.status.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
        if broker.budget.capacity != before:
            print(f"budget {before} -> {broker.budget.capacity}", flush=True)
    broker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
