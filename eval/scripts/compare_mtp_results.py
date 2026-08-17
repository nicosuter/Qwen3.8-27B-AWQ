#!/usr/bin/env python3
"""Gate native-MTP compatibility and report its performance separately."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disabled", type=Path, required=True)
    parser.add_argument("--enabled", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-quality-drop", type=float, default=0.01)
    parser.add_argument("--max-failure-increase", type=float, default=0.01)
    parser.add_argument("--min-acceptance", type=float, default=0.40)
    return parser.parse_args()


def load(path: Path, *, mtp: bool) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                item_id = str(row["id"])
                raw_score = row["score"]
                failed = row["failed"]
                raw_elapsed = row["elapsed_seconds"]
                raw_output_tokens = row["output_tokens"]
                if isinstance(raw_score, bool) or not isinstance(
                    raw_score, (int, float)
                ):
                    raise TypeError("score must be numeric")
                if isinstance(raw_elapsed, bool) or not isinstance(
                    raw_elapsed, (int, float)
                ):
                    raise TypeError("elapsed_seconds must be numeric")
                if isinstance(raw_output_tokens, bool) or not isinstance(
                    raw_output_tokens, int
                ):
                    raise TypeError("output_tokens must be an integer")
                score = float(raw_score)
                elapsed = float(raw_elapsed)
                output_tokens = raw_output_tokens
                if mtp:
                    accepted = row["accepted_draft_tokens"]
                    drafted = row["draft_tokens"]
                    if (
                        isinstance(accepted, bool)
                        or not isinstance(accepted, int)
                        or isinstance(drafted, bool)
                        or not isinstance(drafted, int)
                    ):
                        raise TypeError("MTP acceptance counters must be integers")
                else:
                    accepted = 0
                    drafted = 0
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid row: {error}"
                ) from error
            if not isinstance(failed, bool):
                raise ValueError(f"{path}:{line_number}: failed must be boolean")
            if (
                item_id in rows
                or not math.isfinite(score)
                or not 0 <= score <= 1
                or not math.isfinite(elapsed)
                or elapsed <= 0
                or output_tokens < 0
            ):
                raise ValueError(f"{path}:{line_number}: invalid or duplicate metrics")
            if mtp and (drafted < 0 or not 0 <= accepted <= drafted):
                raise ValueError(
                    f"{path}:{line_number}: invalid MTP acceptance counters"
                )
            rows[item_id] = {
                "score": score,
                "failed": failed,
                "elapsed_seconds": elapsed,
                "output_tokens": output_tokens,
                "accepted_draft_tokens": accepted,
                "draft_tokens": drafted,
            }
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def compare(
    disabled: dict[str, dict[str, Any]],
    enabled: dict[str, dict[str, Any]],
    *,
    max_quality_drop: float = 0.01,
    max_failure_increase: float = 0.01,
    min_acceptance: float = 0.40,
) -> dict[str, Any]:
    """Return the paired MTP report.

    Acceptance counters may be per request or a server-wide run delta assigned
    to one row, with explicit zeros on the others. They must never be repeated
    cumulative counter snapshots.
    """
    if not disabled or not enabled:
        raise ValueError("MTP comparison requires non-empty result sets")
    if set(disabled) != set(enabled):
        raise ValueError("MTP-disabled and MTP-enabled item IDs differ")
    thresholds = (max_quality_drop, max_failure_increase, min_acceptance)
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("MTP comparison thresholds must be finite")
    if max_quality_drop < 0 or max_failure_increase < 0:
        raise ValueError("quality and failure tolerances must be non-negative")
    if not 0 <= min_acceptance <= 1:
        raise ValueError("minimum acceptance must be between 0 and 1")

    quality_delta = statistics.fmean(
        enabled[key]["score"] - disabled[key]["score"] for key in disabled
    )
    failure_increase = statistics.fmean(
        float(enabled[key]["failed"]) - float(disabled[key]["failed"])
        for key in disabled
    )
    accepted = sum(row["accepted_draft_tokens"] for row in enabled.values())
    drafted = sum(row["draft_tokens"] for row in enabled.values())
    if drafted <= 0:
        raise ValueError("MTP-enabled results contain no drafted tokens")
    acceptance = accepted / drafted
    disabled_request_tps = sum(row["output_tokens"] for row in disabled.values()) / sum(
        row["elapsed_seconds"] for row in disabled.values()
    )
    enabled_request_tps = sum(row["output_tokens"] for row in enabled.values()) / sum(
        row["elapsed_seconds"] for row in enabled.values()
    )
    failures = []
    if quality_delta < -max_quality_drop:
        failures.append("quality")
    if failure_increase > max_failure_increase:
        failures.append("failures")
    if acceptance < min_acceptance:
        failures.append("acceptance")
    speed_ratio = (
        enabled_request_tps / disabled_request_tps if disabled_request_tps > 0 else None
    )
    return {
        "items": len(disabled),
        "quality_delta": quality_delta,
        "failure_increase": failure_increase,
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
        "acceptance_rate": acceptance,
        "latency_derived_tokens_per_second": {
            "disabled": disabled_request_tps,
            "enabled": enabled_request_tps,
        },
        "latency_derived_speed_ratio": speed_ratio,
        "speed_note": (
            "sum(output_tokens) / sum(per-request elapsed_seconds); this is a "
            "latency diagnostic, not wall-clock server throughput when requests overlap"
        ),
        "gate": {"passed": not failures, "failures": failures},
    }


def main() -> int:
    args = parse_args()
    disabled = load(args.disabled, mtp=False)
    enabled = load(args.enabled, mtp=True)
    report = compare(
        disabled,
        enabled,
        max_quality_drop=args.max_quality_drop,
        max_failure_increase=args.max_failure_increase,
        min_acceptance=args.min_acceptance,
    )
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
