#!/usr/bin/env python3
"""Gate native-MTP compatibility and report its performance separately."""

import argparse
import json
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
                score = float(row["score"])
                failed = row.get("failed", False)
                elapsed = float(row["elapsed_seconds"])
                output_tokens = int(row["output_tokens"])
                accepted = int(row.get("accepted_draft_tokens", 0))
                drafted = int(row.get("draft_tokens", 0))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid row: {error}") from error
            if not isinstance(failed, bool):
                raise ValueError(f"{path}:{line_number}: failed must be boolean")
            if item_id in rows or not 0 <= score <= 1 or elapsed <= 0 or output_tokens < 0:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate metrics")
            if mtp and (drafted <= 0 or not 0 <= accepted <= drafted):
                raise ValueError(f"{path}:{line_number}: invalid MTP acceptance counters")
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


def main() -> int:
    args = parse_args()
    disabled = load(args.disabled, mtp=False)
    enabled = load(args.enabled, mtp=True)
    if set(disabled) != set(enabled):
        raise ValueError("MTP-disabled and MTP-enabled item IDs differ")
    quality_delta = statistics.fmean(
        enabled[key]["score"] - disabled[key]["score"] for key in disabled
    )
    failure_increase = statistics.fmean(
        float(enabled[key]["failed"]) - float(disabled[key]["failed"])
        for key in disabled
    )
    accepted = sum(row["accepted_draft_tokens"] for row in enabled.values())
    drafted = sum(row["draft_tokens"] for row in enabled.values())
    acceptance = accepted / drafted
    disabled_tps = sum(row["output_tokens"] for row in disabled.values()) / sum(
        row["elapsed_seconds"] for row in disabled.values()
    )
    enabled_tps = sum(row["output_tokens"] for row in enabled.values()) / sum(
        row["elapsed_seconds"] for row in enabled.values()
    )
    failures = []
    if quality_delta < -args.max_quality_drop:
        failures.append("quality")
    if failure_increase > args.max_failure_increase:
        failures.append("failures")
    if acceptance < args.min_acceptance:
        failures.append("acceptance")
    report = {
        "items": len(disabled),
        "quality_delta": quality_delta,
        "failure_increase": failure_increase,
        "acceptance_rate": acceptance,
        "tokens_per_second": {"disabled": disabled_tps, "enabled": enabled_tps},
        "speed_ratio": enabled_tps / disabled_tps,
        "gate": {"passed": not failures, "failures": failures},
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
