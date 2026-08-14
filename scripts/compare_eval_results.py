#!/usr/bin/env python3
"""Paired, item-clustered comparison of baseline and candidate eval results."""

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


Key = tuple[str, str, int]
FAILURE_FIELDS = (
    "malformed_tool_call",
    "premature_final_answer",
    "empty_answer",
    "repetition_loop",
    "context_failure",
    "timeout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=38_027)
    parser.add_argument("--max-suite-drop", type=float, default=0.03)
    parser.add_argument("--macro-ci-margin", type=float, default=0.03)
    parser.add_argument("--max-failure-increase", type=float, default=0.01)
    parser.add_argument("--must-pass-retention", type=float, default=0.95)
    return parser.parse_args()


def load_rows(path: Path) -> dict[Key, dict[str, Any]]:
    rows: dict[Key, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                suite = str(row["suite"])
                item_id = str(row["id"])
                replicate = int(row.get("replicate", 0))
                score = float(row["score"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid result row: {exc}") from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{path}:{line_number}: score must be finite and in [0, 1]")
            key = (suite, item_id, replicate)
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate key {key}")
            failures = {}
            for field in FAILURE_FIELDS:
                value = row.get(field, False)
                if not isinstance(value, bool):
                    raise ValueError(f"{path}:{line_number}: {field} must be boolean")
                failures[field] = value
            must_pass = row.get("must_pass", False)
            if not isinstance(must_pass, bool):
                raise ValueError(f"{path}:{line_number}: must_pass must be boolean")
            rows[key] = {"score": score, "must_pass": must_pass, **failures}
    if not rows:
        raise ValueError(f"{path}: no result rows")
    return rows


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(
    baseline: dict[Key, dict[str, Any]],
    candidate: dict[Key, dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    missing_candidate = sorted(set(baseline) - set(candidate))
    missing_baseline = sorted(set(candidate) - set(baseline))
    if missing_candidate or missing_baseline:
        raise ValueError(
            "result keys differ: "
            f"missing from candidate={missing_candidate[:10]} "
            f"missing from baseline={missing_baseline[:10]}"
        )

    by_item: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for key, baseline_row in baseline.items():
        suite, item_id, _ = key
        by_item[(suite, item_id)].append(
            (baseline_row["score"], candidate[key]["score"])
        )

    suites: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for (suite, item_id), pairs in by_item.items():
        suites[suite].append(
            (
                item_id,
                statistics.fmean(pair[0] for pair in pairs),
                statistics.fmean(pair[1] for pair in pairs),
            )
        )

    rng = random.Random(seed)
    suite_results: dict[str, Any] = {}
    bootstrap_suite_deltas: dict[str, list[float]] = {}
    for suite in sorted(suites):
        items = suites[suite]
        replicate_counts = [len(by_item[(suite, item_id)]) for item_id, _, _ in items]
        deltas = [candidate_score - baseline_score for _, baseline_score, candidate_score in items]
        bootstrap = []
        for _ in range(samples):
            drawn = [deltas[rng.randrange(len(deltas))] for _ in items]
            bootstrap.append(statistics.fmean(drawn))
        bootstrap_suite_deltas[suite] = bootstrap
        regressions = sum(candidate_score < baseline_score for _, baseline_score, candidate_score in items)
        improvements = sum(candidate_score > baseline_score for _, baseline_score, candidate_score in items)
        suite_results[suite] = {
            "items": len(items),
            "observations": sum(replicate_counts),
            "replicates_per_item": {
                "min": min(replicate_counts),
                "max": max(replicate_counts),
            },
            "baseline": statistics.fmean(item[1] for item in items),
            "candidate": statistics.fmean(item[2] for item in items),
            "delta": statistics.fmean(deltas),
            "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
            "improved_items": improvements,
            "regressed_items": regressions,
            "tied_items": len(items) - improvements - regressions,
        }

    suite_names = sorted(suites)
    macro_bootstrap = [
        statistics.fmean(bootstrap_suite_deltas[suite][index] for suite in suite_names)
        for index in range(samples)
    ]
    macro_baseline = statistics.fmean(suite_results[suite]["baseline"] for suite in suite_names)
    macro_candidate = statistics.fmean(suite_results[suite]["candidate"] for suite in suite_names)
    return {
        "suites": suite_results,
        "macro": {
            "suites": len(suite_names),
            "baseline": macro_baseline,
            "candidate": macro_candidate,
            "delta": macro_candidate - macro_baseline,
            "ci95": [percentile(macro_bootstrap, 0.025), percentile(macro_bootstrap, 0.975)],
        },
        "bootstrap": {"samples": samples, "seed": seed, "cluster": "item"},
    }


def auxiliary_gates(
    baseline: dict[Key, dict[str, Any]],
    candidate: dict[Key, dict[str, Any]],
    *,
    max_failure_increase: float,
    must_pass_retention: float,
) -> dict[str, Any]:
    failure_rates = {}
    failure_failures = []
    for field in FAILURE_FIELDS:
        baseline_rate = statistics.fmean(float(row[field]) for row in baseline.values())
        candidate_rate = statistics.fmean(float(row[field]) for row in candidate.values())
        increase = candidate_rate - baseline_rate
        failure_rates[field] = {
            "baseline": baseline_rate,
            "candidate": candidate_rate,
            "increase": increase,
        }
        if increase > max_failure_increase:
            failure_failures.append(field)

    must_pass_keys = [
        key
        for key, row in baseline.items()
        if row["must_pass"] and row["score"] > 0
    ]
    retained = (
        statistics.fmean(float(candidate[key]["score"] > 0) for key in must_pass_keys)
        if must_pass_keys
        else None
    )
    return {
        "failure_rates": failure_rates,
        "max_failure_increase": max_failure_increase,
        "failure_rate_failures": failure_failures,
        "must_pass_baseline_passes": len(must_pass_keys),
        "must_pass_retention": retained,
        "required_must_pass_retention": must_pass_retention,
        "must_pass_failure": retained is not None and retained < must_pass_retention,
    }


def format_points(value: float) -> str:
    return f"{100 * value:+.2f}"


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    baseline = load_rows(args.baseline)
    candidate = load_rows(args.candidate)
    report = summarize(
        baseline,
        candidate,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    auxiliary = auxiliary_gates(
        baseline,
        candidate,
        max_failure_increase=args.max_failure_increase,
        must_pass_retention=args.must_pass_retention,
    )

    suite_failures = [
        suite
        for suite, result in report["suites"].items()
        if result["delta"] < -args.max_suite_drop
    ]
    macro_failure = report["macro"]["ci95"][0] < -args.macro_ci_margin
    report["gate"] = {
        "passed": not suite_failures
        and not macro_failure
        and not auxiliary["failure_rate_failures"]
        and not auxiliary["must_pass_failure"],
        "max_suite_drop": args.max_suite_drop,
        "macro_ci_margin": args.macro_ci_margin,
        "suite_point_failures": suite_failures,
        "macro_ci_failure": macro_failure,
        **auxiliary,
    }

    print("suite                         n baseline candidate delta       95% CI")
    for suite, result in report["suites"].items():
        interval = result["ci95"]
        print(
            f"{suite[:28]:28} {result['items']:4d} "
            f"{100 * result['baseline']:7.2f} {100 * result['candidate']:7.2f} "
            f"{format_points(result['delta']):>7} "
            f"[{format_points(interval[0])}, {format_points(interval[1])}]"
        )
    macro = report["macro"]
    print(
        f"{'MACRO':28} {macro['suites']:4d} "
        f"{100 * macro['baseline']:7.2f} {100 * macro['candidate']:7.2f} "
        f"{format_points(macro['delta']):>7} "
        f"[{format_points(macro['ci95'][0])}, {format_points(macro['ci95'][1])}]"
    )
    print(
        "automated-quality-gate="
        f"{'PASS' if report['gate']['passed'] else 'FAIL'}"
    )

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
