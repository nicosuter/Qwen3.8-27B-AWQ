#!/usr/bin/env python3
"""Sample server and GPU state while a suite is being scored.

Per-request durations can say how long each call took, but not how many were in
flight at once, how full the KV cache was, or whether the scheduler was
preempting. Those have to be reconstructed from durations by assuming how the
client's thread pool scheduled -- and that reconstruction put gpu occupancy at
19% of the requested concurrency, which is too important a number to infer.
vLLM publishes the real thing, so record it.

Writes one JSON object per sample to --output, appending, until killed.
"""

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

# Summed across data-parallel engines, which each publish their own series.
# generation_tokens_total is monotonic: differencing two samples gives the real
# aggregate decode rate, which is the number every throughput claim rests on.
COUNTERS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
)
# Averaged instead: a cache 80% full on every engine is 80% full, not 320% full.
# vLLM renamed this from gpu_ to kv_; accept whichever the server publishes so a
# container upgrade does not silently drop the series.
FRACTIONS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    return parser.parse_args(argv)


def scrape(url: str, timeout: float = 5.0) -> dict[str, float]:
    """Prometheus text format to a dict, tolerating a server that is not up."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - an unreachable server is a normal sample
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in COUNTERS and name not in FRACTIONS:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1
    return {
        name: (total / counts[name] if name in FRACTIONS else total)
        for name, total in sums.items()
    }


def sample_gpus() -> list[dict[str, float]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "util_percent": float(parts[1]),
                    "memory_mib": float(parts[2]),
                    "power_w": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return rows


def sample(metrics_url: str) -> dict[str, Any]:
    return {"t": round(time.time(), 3), "vllm": scrape(metrics_url), "gpus": sample_gpus()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        while True:
            handle.write(json.dumps(sample(args.metrics_url)) + "\n")
            handle.flush()  # killed abruptly at suite end; unflushed lines are lost
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
