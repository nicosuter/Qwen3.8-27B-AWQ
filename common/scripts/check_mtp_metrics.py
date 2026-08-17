#!/usr/bin/env python3
"""Check speculative-decoding counter deltas from two vLLM metrics snapshots."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+(?P<value>[^\s]+)"
)
ACCEPTED_NAMES = {
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_total",
}
DRAFTED_NAMES = {
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_draft_tokens_total",
}


def metric_total(text: str, names: set[str]) -> float | None:
    total = 0.0
    found = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if match is None or match.group("name") not in names:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid Prometheus counter sample: {line}")
        total += value
        found = True
    return total if found else None


def acceptance_report(
    before: str, after: str, *, min_acceptance: float = 0.40
) -> dict[str, float | bool]:
    if not math.isfinite(min_acceptance) or not 0 <= min_acceptance <= 1:
        raise ValueError("minimum acceptance must be between zero and one")

    accepted_before = metric_total(before, ACCEPTED_NAMES) or 0.0
    drafted_before = metric_total(before, DRAFTED_NAMES) or 0.0
    accepted_after = metric_total(after, ACCEPTED_NAMES)
    drafted_after = metric_total(after, DRAFTED_NAMES)
    if accepted_after is None or drafted_after is None:
        raise ValueError("vLLM did not expose speculative decoding counters")

    accepted = accepted_after - accepted_before
    drafted = drafted_after - drafted_before
    if accepted < 0 or drafted < 0:
        raise ValueError("speculative decoding counters decreased between snapshots")
    if drafted == 0:
        raise ValueError("MTP request produced no draft tokens")
    if accepted > drafted:
        raise ValueError("accepted draft tokens exceed drafted tokens")

    rate = accepted / drafted
    return {
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
        "acceptance_rate": rate,
        "minimum_acceptance": min_acceptance,
        "passed": rate >= min_acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--min-acceptance", type=float, default=0.40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = acceptance_report(
        args.before.read_text(encoding="utf-8"),
        args.after.read_text(encoding="utf-8"),
        min_acceptance=args.min_acceptance,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
