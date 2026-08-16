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
    # The hard bar is the equally weighted macro point estimate. Gating on every
    # suite's point estimate instead fails 60% of runs that have no degradation
    # at all: every suite is another chance, and the small ones carry confidence
    # intervals twice the width of the margin. That 60% was simulated over seven
    # suites; swebench_pro_1_0 makes eight and has not been re-simulated.
    parser.add_argument("--max-macro-drop", type=float, default=0.03)
    # A single suite can still sink the run, but only on evidence: its interval
    # must clear a wider margin, so noise cannot trip it.
    parser.add_argument("--suite-confident-drop", type=float, default=0.05)
    # Reported for the manual regression review, never a failure by itself.
    parser.add_argument("--review-suite-drop", type=float, default=0.03)
    # The near-lossless claim, judged on the point estimate with its interval
    # published beside it. Separate from the ship/no-ship gate above: this is
    # what the model card is allowed to say, not whether the run passed.
    # 98, not 99. Simulated over the suites we actually run
    # (scripts/simulate_gates.py), the geometric mean's own 95% interval is about
    # ±2.2 points wide under the null, so a 99% bar denied the claim to 19% of
    # checkpoints with no degradation at all. A 1-point margin was never inside
    # what this measurement can resolve. At 98% that falls to 4%.
    parser.add_argument("--near-lossless-recovery", type=float, default=0.98)
    parser.add_argument("--max-failure-increase", type=float, default=0.01)
    parser.add_argument("--must-pass-retention", type=float, default=0.95)
    # Agent verifiers award partial credit, so "score > 0" is not a pass.
    parser.add_argument("--must-pass-threshold", type=float, default=1.0)
    # Everything else here is paired, so a harness broken for both checkpoints
    # passes silently. Floors are the only check that looks at absolute quality.
    parser.add_argument(
        "--baseline-floor",
        action="append",
        default=[],
        metavar="SUITE=VALUE",
        help="alert (not fail) when the baseline scores below VALUE on SUITE",
    )
    return parser.parse_args()


def parse_floors(raw: list[str]) -> dict[str, float]:
    floors = {}
    for entry in raw:
        suite, _, value = entry.partition("=")
        if not suite or not value:
            raise ValueError(f"--baseline-floor expects SUITE=VALUE; got {entry!r}")
        floors[suite] = float(value)
    return floors


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
            # A deferred row carries a zero that means "not executed yet". Left
            # to stand it would read as a failed item, so refuse the file rather
            # than average it.
            if row.get("deferred"):
                raise ValueError(
                    f"{path}:{line_number}: {suite}/{item_id} is deferred and has not been "
                    "scored; run the adapter's `score` step first"
                )
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
            rows[key] = {
                "score": score,
                "must_pass": must_pass,
                # Optional: rows from our own adapters do not carry it. Where it
                # is present it names the dataset revision the item came from,
                # and the two arms are checked against each other below.
                "dataset_pin": row.get("dataset_pin"),
                **failures,
            }
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


def recovery(baseline: float, candidate: float) -> float:
    """Candidate score as a fraction of the baseline's.

    A zero baseline means the harness failed for both checkpoints, so the ratio
    carries no information. Return the geometric identity and leave
    --baseline-floor to be the thing that reports it.
    """
    if baseline <= 0:
        return 1.0
    return candidate / baseline


def geometric_mean(values: list[float]) -> float:
    """The only average of ratios that has an interpretation.

    Zero survives as zero: a suite the candidate scores nothing on should sink
    the summary rather than be averaged away by the others.
    """
    if not values:
        return 1.0
    if any(value <= 0 for value in values):
        return 0.0
    return math.exp(statistics.fmean(math.log(value) for value in values))


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

    # Two arms scored against different dataset revisions would still join on
    # every key and produce a plausible number. Where the rows say which
    # revision they came from, the arms have to agree.
    disagreeing = sorted(
        {
            f"{key[0]}: baseline {baseline_row['dataset_pin']} "
            f"vs candidate {candidate[key]['dataset_pin']}"
            for key, baseline_row in baseline.items()
            if baseline_row["dataset_pin"] != candidate[key]["dataset_pin"]
        }
    )
    if disagreeing:
        raise ValueError(
            "the two arms were scored against different dataset revisions, so "
            "this is not a paired comparison: " + "; ".join(disagreeing[:5])
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
    bootstrap_suite_recovery: dict[str, list[float]] = {}
    for suite in sorted(suites):
        items = suites[suite]
        replicate_counts = [len(by_item[(suite, item_id)]) for item_id, _, _ in items]
        pairs = [(baseline_score, candidate_score) for _, baseline_score, candidate_score in items]
        deltas = [candidate_score - baseline_score for baseline_score, candidate_score in pairs]
        bootstrap = []
        recovery_bootstrap = []
        for _ in range(samples):
            # Resample items, not deltas: recovery needs both sides of the pair
            # from the same draw, and the delta stays exactly what it was.
            drawn = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            drawn_baseline = statistics.fmean(pair[0] for pair in drawn)
            drawn_candidate = statistics.fmean(pair[1] for pair in drawn)
            bootstrap.append(drawn_candidate - drawn_baseline)
            recovery_bootstrap.append(recovery(drawn_baseline, drawn_candidate))
        bootstrap_suite_deltas[suite] = bootstrap
        bootstrap_suite_recovery[suite] = recovery_bootstrap
        regressions = sum(candidate_score < baseline_score for baseline_score, candidate_score in pairs)
        improvements = sum(candidate_score > baseline_score for baseline_score, candidate_score in pairs)
        baseline_mean = statistics.fmean(pair[0] for pair in pairs)
        candidate_mean = statistics.fmean(pair[1] for pair in pairs)
        suite_results[suite] = {
            "items": len(items),
            "observations": sum(replicate_counts),
            "replicates_per_item": {
                "min": min(replicate_counts),
                "max": max(replicate_counts),
            },
            "baseline": baseline_mean,
            "candidate": candidate_mean,
            "delta": statistics.fmean(deltas),
            "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
            # Reported beside the absolute scores, never instead of them: a ratio
            # without its base case is uninterpretable.
            "recovery": recovery(baseline_mean, candidate_mean),
            "recovery_ci95": [
                percentile(recovery_bootstrap, 0.025),
                percentile(recovery_bootstrap, 0.975),
            ],
            "improved_items": improvements,
            "regressed_items": regressions,
            "tied_items": len(items) - improvements - regressions,
        }

    suite_names = sorted(suites)
    macro_bootstrap = [
        statistics.fmean(bootstrap_suite_deltas[suite][index] for suite in suite_names)
        for index in range(samples)
    ]
    macro_recovery_bootstrap = [
        geometric_mean([bootstrap_suite_recovery[suite][index] for suite in suite_names])
        for index in range(samples)
    ]
    macro_baseline = statistics.fmean(suite_results[suite]["baseline"] for suite in suite_names)
    macro_candidate = statistics.fmean(suite_results[suite]["candidate"] for suite in suite_names)
    macro_recovery = geometric_mean(
        [suite_results[suite]["recovery"] for suite in suite_names]
    )
    return {
        "suites": suite_results,
        "macro": {
            "suites": len(suite_names),
            "baseline": macro_baseline,
            "candidate": macro_candidate,
            "delta": macro_candidate - macro_baseline,
            "ci95": [percentile(macro_bootstrap, 0.025), percentile(macro_bootstrap, 0.975)],
            # Geometric, not arithmetic: averaging ratios arithmetically has no
            # meaningful interpretation (Hoefler & Belli, SC15, rules 3 and 4).
            # The absolute baseline and candidate above are what it summarizes.
            "recovery_geomean": macro_recovery,
            "recovery_ci95": [
                percentile(macro_recovery_bootstrap, 0.025),
                percentile(macro_recovery_bootstrap, 0.975),
            ],
        },
        "bootstrap": {"samples": samples, "seed": seed, "cluster": "item"},
    }


def auxiliary_gates(
    baseline: dict[Key, dict[str, Any]],
    candidate: dict[Key, dict[str, Any]],
    *,
    max_failure_increase: float,
    must_pass_retention: float,
    must_pass_threshold: float = 1.0,
) -> dict[str, Any]:
    suites = sorted({key[0] for key in baseline})
    failure_rates = {}
    failure_failures = []
    for field in FAILURE_FIELDS:
        per_suite = {}
        for suite in suites:
            rows = [key for key in baseline if key[0] == suite]
            base_rate = statistics.fmean(float(baseline[key][field]) for key in rows)
            cand_rate = statistics.fmean(float(candidate[key][field]) for key in rows)
            per_suite[suite] = {
                "baseline": base_rate,
                "candidate": cand_rate,
                "increase": cand_rate - base_rate,
            }
        # Equal weight per suite, matching the quality gate. Pooling every row
        # instead weights suites by items x replicates, which let a 15-point
        # context_failure regression confined to RULER show up as 0.86 points.
        macro_increase = statistics.fmean(v["increase"] for v in per_suite.values())
        failure_rates[field] = {
            "macro_increase": macro_increase,
            "worst_suite": max(per_suite, key=lambda s: per_suite[s]["increase"]),
            "by_suite": per_suite,
        }
        if macro_increase > max_failure_increase:
            failure_failures.append(field)

    # Per item, after averaging replicates: an item with more replicates must not
    # carry more weight in a bank where every task counts once.
    base_items = aggregate_items(baseline)
    cand_items = aggregate_items(candidate)
    must_pass_items = [
        key
        for key, (score, must_pass) in base_items.items()
        if must_pass and score >= must_pass_threshold
    ]
    retained = (
        statistics.fmean(
            float(cand_items[key][0] >= must_pass_threshold) for key in must_pass_items
        )
        if must_pass_items
        else None
    )
    declared = {key for key, (_, must_pass) in base_items.items() if must_pass}
    return {
        "failure_rates": failure_rates,
        "max_failure_increase": max_failure_increase,
        "failure_rate_failures": failure_failures,
        "must_pass_threshold": must_pass_threshold,
        "must_pass_declared_items": len(declared),
        "must_pass_baseline_passes": len(must_pass_items),
        "must_pass_retention": retained,
        "required_must_pass_retention": must_pass_retention,
        "must_pass_failure": retained is not None and retained < must_pass_retention,
        # A bank whose tasks the baseline never fully passes cannot gate anything.
        "must_pass_unusable": bool(declared) and not must_pass_items,
    }


def aggregate_items(rows: dict[Key, dict[str, Any]]) -> dict[tuple[str, str], tuple[float, bool]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (suite, item_id, _), row in rows.items():
        grouped[(suite, item_id)].append(row)
    return {
        key: (
            statistics.fmean(row["score"] for row in group),
            any(row["must_pass"] for row in group),
        )
        for key, group in grouped.items()
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
        must_pass_threshold=args.must_pass_threshold,
    )

    # Hard bar: the equally weighted macro point estimate.
    macro_failure = report["macro"]["delta"] < -args.max_macro_drop
    # A single suite fails the run only when its interval clears a wider margin,
    # so one noisy small suite cannot veto an otherwise healthy candidate.
    suite_failures = [
        suite
        for suite, result in report["suites"].items()
        if result["ci95"][1] < -args.suite_confident_drop
    ]
    # Reported for the manual regression review; never a failure by itself.
    suite_reviews = [
        suite
        for suite, result in report["suites"].items()
        if result["delta"] < -args.review_suite_drop
    ]
    floors = parse_floors(args.baseline_floor)
    floor_failures = {
        suite: {"baseline": report["suites"][suite]["baseline"], "floor": floor}
        for suite, floor in floors.items()
        if suite in report["suites"] and report["suites"][suite]["baseline"] < floor
    }
    missing_floor_suites = sorted(set(floors) - set(report["suites"]))

    report["gate"] = {
        # Floors are deliberately absent from "passed". A floor firing means the
        # absolute number looks wrong, which is usually a broken harness rather
        # than a bad checkpoint, and the two need different responses: a failed
        # gate stops a release, whereas this should send someone to look at the
        # run. It stays loud in the printed output and in the report.
        "passed": not macro_failure
        and not suite_failures
        and not auxiliary["failure_rate_failures"]
        and not auxiliary["must_pass_failure"],
        "max_macro_drop": args.max_macro_drop,
        "suite_confident_drop": args.suite_confident_drop,
        "review_suite_drop": args.review_suite_drop,
        # Deliberately absent from "passed": this decides what the model card may
        # claim, not whether the checkpoint ships.
        "near_lossless": report["macro"]["recovery_geomean"] >= args.near_lossless_recovery,
        "near_lossless_recovery": args.near_lossless_recovery,
        "near_lossless_ci95": report["macro"]["recovery_ci95"],
        "macro_point_failure": macro_failure,
        "suite_confident_failures": suite_failures,
        "suite_review_flags": suite_reviews,
        "baseline_floor_alerts": floor_failures,
        "baseline_floor_missing_suites": missing_floor_suites,
        "baseline_floor_alerted": bool(floor_failures or missing_floor_suites),
        **auxiliary,
    }

    print("suite                         n baseline candidate delta       95% CI"
          "        recovery  recovery 95% CI")
    for suite, result in report["suites"].items():
        interval = result["ci95"]
        recovery_interval = result["recovery_ci95"]
        print(
            f"{suite[:28]:28} {result['items']:4d} "
            f"{100 * result['baseline']:7.2f} {100 * result['candidate']:7.2f} "
            f"{format_points(result['delta']):>7} "
            f"[{format_points(interval[0])}, {format_points(interval[1])}]"
            f"  {100 * result['recovery']:7.2f}%"
            f"  [{100 * recovery_interval[0]:6.2f}, {100 * recovery_interval[1]:6.2f}]"
        )
    macro = report["macro"]
    print(
        f"{'MACRO':28} {macro['suites']:4d} "
        f"{100 * macro['baseline']:7.2f} {100 * macro['candidate']:7.2f} "
        f"{format_points(macro['delta']):>7} "
        f"[{format_points(macro['ci95'][0])}, {format_points(macro['ci95'][1])}]"
        f"  {100 * macro['recovery_geomean']:7.2f}%"
        f"  [{100 * macro['recovery_ci95'][0]:6.2f}, {100 * macro['recovery_ci95'][1]:6.2f}]"
    )
    gate = report["gate"]
    # The interval, not the verdict, is the honest object here: the bar is a
    # convenience and the width is what says whether the number means anything.
    recovery_interval = macro["recovery_ci95"]
    print(
        f"near-lossless-claim="
        f"{'PASS' if gate['near_lossless'] else 'FAIL'}"
        f" (geometric mean recovery {100 * macro['recovery_geomean']:.2f}%,"
        f" 95% CI {100 * recovery_interval[0]:.2f}-{100 * recovery_interval[1]:.2f}%,"
        f" against a {100 * args.near_lossless_recovery:.2f}% bar)"
    )
    margin = 1.0 - args.near_lossless_recovery
    half_width = (recovery_interval[1] - recovery_interval[0]) / 2
    if half_width > margin:
        print(
            f"note: that interval is ±{100 * half_width:.2f} points, wider than the "
            f"{100 * margin:.2f}-point margin it is being compared against, so read "
            "the interval rather than the verdict"
        )
    if gate["suite_review_flags"]:
        print("review (not a failure): " + ", ".join(gate["suite_review_flags"]))
    for field in gate["failure_rate_failures"]:
        rates = gate["failure_rates"][field]
        print(
            f"failure-mode gate: {field} +{100 * rates['macro_increase']:.2f} points "
            f"macro, worst in {rates['worst_suite']}"
        )
    if gate["must_pass_unusable"]:
        print(
            f"must-pass bank unusable: {gate['must_pass_declared_items']} tasks declared, "
            f"none reach the {gate['must_pass_threshold']:.2f} pass threshold on the baseline"
        )
    if gate["must_pass_retention"] is not None:
        print(
            f"must-pass retention: {100 * gate['must_pass_retention']:.1f}% of "
            f"{gate['must_pass_baseline_passes']} baseline-passing tasks"
        )
    for suite, detail in gate["baseline_floor_alerts"].items():
        print(
            f"ALERT baseline floor: {suite} scored {100 * detail['baseline']:.2f} "
            f"below the {100 * detail['floor']:.2f} floor; the harness may be broken "
            f"for both arms. This does not fail the gate; go and look at the run."
        )
    for suite in gate["baseline_floor_missing_suites"]:
        print(f"ALERT baseline floor: no results for {suite}")
    print(
        "automated-quality-gate="
        f"{'PASS' if gate['passed'] else 'FAIL'}"
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
