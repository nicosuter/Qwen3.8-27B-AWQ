#!/usr/bin/env python3
"""Re-derive the gate rejection rates, at whatever suite count we actually run.

EVAL.md justifies gates 1 and 2 with a simulation: the old rule, failing the run
whenever any suite's point estimate fell 3 points, rejected 60% of runs with no
degradation at all, while the macro rule rejects 2%. That simulation was run over
seven suites and was never committed, so it could not be checked and could not be
re-run when SWE-bench Pro made it eight. This is it, written down.

The parameters are measured wherever a paired run has already produced them: the
per-suite 95% interval from `comparison.json` gives SE(delta) directly, since the
comparator's interval is a bootstrap over item-clustered pairs and already
carries whatever replicate correlation that suite has. Three suites have no
paired run yet and are modelled instead, from the paired-binomial

    SE(delta) = sqrt(2 p (1 - p) (1 - rho) / n)

with `rho` the between-arm agreement. Those three are marked `modelled` in the
output, and `--discordance-scale` sweeps them, because a number nobody has
measured should not be quoted as though somebody had.

What the simulation asks is narrow and worth stating plainly. Under the null the
two checkpoints are equal in expectation, so every suite's observed delta is
noise; the question is how often each candidate rule fires anyway. That is a
false-positive rate, not a power calculation, and it is the thing that gets worse
as suites are added.
"""

import argparse
import json
import math
import random
from typing import Any

Z95 = 1.959963984540054

# `items`/`reps` are what the measured `half` was measured at, from v2/paired-2
# and v2/paired-fp8gdn. `run_items`/`run_reps` are what the protocol now runs,
# and `k` rescales between the two.
#
# k is the ratio of replicate-to-replicate variance to item-to-item variance in
# the paired delta, decomposed from those same runs. It is around 10, which is
# worth knowing: replicates are not the wasted spend they are often assumed to
# be, and dropping to one replicate costs real precision on every suite that has
# no further items to buy. What remains true is the budget argument, that
# SE^2 = R * var_between / C + var_within / C rises with R at fixed spend.
SUITES: dict[str, dict[str, Any]] = {
    "bfcl_v4": {"items": 1240, "reps": 4, "baseline": 0.8742, "half": 0.00776,
                "k": 11.56, "run_items": 3486, "run_reps": 1},
    "gpqa_diamond": {"items": 198, "reps": 4, "baseline": 0.8889, "half": 0.01957,
                     "k": 15.41, "run_reps": 1},
    "multimodal": {"items": 600, "reps": 4, "baseline": 0.8675, "half": 0.00912,
                   "k": 8.88, "run_reps": 1},
    "ruler": {"items": 105, "reps": 1, "baseline": 0.8000, "half": 0.02857,
              "k": 12.0, "run_reps": 1},
    # No paired run yet. Ten-way multiple choice on hard material, so the
    # baseline is a guess and the agreement is assumed lower than the
    # four-option suites.
    "mmmu_pro": {"items": 1730, "reps": 1, "baseline": 0.55, "rho": 0.75},
    # Baseline measured 2026-08-17 once the deferred execution was scored;
    # it came in at 0.8743, well above the 0.65 that was assumed.
    "livecodebench_v6": {"items": 175, "reps": 1, "baseline": 0.8743, "rho": 0.80},
}
# Parked, not deleted. Both agentic suites were costed and left out: SWE-bench
# Pro because its containers will not start here, Terminal-Bench because 89 items
# at a 0.35 baseline set the width of the whole macro. `--with-candidate` puts
# them back, so the price of readmitting a low-baseline suite stays visible.
CANDIDATE = {
    "matharena_2026_06": {"items": 77, "reps": 4, "baseline": 0.8052, "half": 0.03734,
                          # var_between measured as zero at 77 items, so k is a
                          # lower bound standing in for "replicates are all of it".
                          "k": 60.0, "run_reps": 1},
    "terminal_bench_2_1": {"items": 89, "reps": 3, "baseline": 0.35, "rho": 0.70},
    "swebench_pro_1_0": {"items": 300, "reps": 1, "baseline": 0.25, "rho": 0.65},
}


def standard_error(spec: dict[str, Any], discordance_scale: float) -> tuple[float, bool]:
    """SE of a suite's paired delta at the configuration the protocol runs."""
    if "half" in spec:
        measured = spec["half"] / Z95
        k = spec.get("k")
        if k is None:
            return measured, False
        items, reps = spec["items"], spec["reps"]
        run_items = spec.get("run_items", items)
        run_reps = spec.get("run_reps", reps)
        scale = math.sqrt(
            ((1 + k / run_reps) / run_items) / ((1 + k / reps) / items)
        )
        return measured * scale, False
    p = spec["baseline"]
    discordance = 2 * p * (1 - p) * (1 - spec["rho"]) * discordance_scale
    return math.sqrt(max(discordance, 1e-12) / spec["items"]), True


def simulate(
    names: list[str],
    *,
    trials: int,
    seed: int,
    any_suite_drop: float,
    max_macro_drop: float,
    confident_drop: float,
    near_lossless: float,
    discordance_scale: float,
    recovery_names: list[str] | None = None,
    weighting: str = "equal",
) -> dict[str, Any]:
    rng = random.Random(seed)
    errors = {name: standard_error(SUITES[name], discordance_scale)[0] for name in names}
    baselines = {name: SUITES[name]["baseline"] for name in names}

    # Which suites enter the recovery geomean, and how heavily. Recovery is a
    # ratio, so its noise is SE divided by baseline: a suite with a small
    # baseline contributes noise out of all proportion to its size, and equal
    # weighting hands the aggregate to whichever suite is least precise.
    entering = [n for n in (recovery_names or names) if n in names]
    if weighting == "inverse-variance":
        precision = {n: (baselines[n] / errors[n]) ** 2 for n in entering}
        total = sum(precision.values())
        weights = {n: precision[n] / total for n in entering}
    elif weighting == "equal":
        weights = {n: 1 / len(entering) for n in entering}
    else:
        raise ValueError(f"unknown weighting {weighting!r}")

    any_rule = macro_rule = confident_rule = lossless_fail = 0
    recoveries: list[float] = []
    for _ in range(trials):
        deltas = {name: rng.gauss(0.0, errors[name]) for name in names}
        if any(delta <= -any_suite_drop for delta in deltas.values()):
            any_rule += 1
        macro = sum(deltas.values()) / len(names)
        if macro <= -max_macro_drop:
            macro_rule += 1
        # Gate 2 wants a confident drop, so the interval has to exclude zero as
        # well. With a symmetric interval that is |delta| > z * SE.
        if any(
            delta <= -confident_drop and abs(delta) > Z95 * errors[name]
            for name, delta in deltas.items()
        ):
            confident_rule += 1
        geomean = math.exp(
            sum(weights[name] * math.log(1 + deltas[name] / baselines[name])
                for name in entering)
        )
        recoveries.append(geomean)
        if geomean < near_lossless:
            lossless_fail += 1

    mean = sum(recoveries) / trials
    variance = sum((value - mean) ** 2 for value in recoveries) / (trials - 1)
    return {
        "suites": len(names),
        "any_suite_rule": any_rule / trials,
        "macro_rule": macro_rule / trials,
        "confident_suite_rule": confident_rule / trials,
        "near_lossless_failure": lossless_fail / trials,
        "recovery_geomean_sd": math.sqrt(variance),
        "recovery_geomean_95_halfwidth": Z95 * math.sqrt(variance),
        "suite_delta_se": {name: round(errors[name], 5) for name in names},
        "recovery_suites": entering,
        "weighting": weighting,
        "modelled": sorted(
            name for name in names if standard_error(SUITES[name], discordance_scale)[1]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=38027)
    parser.add_argument("--any-suite-drop", type=float, default=0.03)
    parser.add_argument("--max-macro-drop", type=float, default=0.03)
    parser.add_argument("--confident-drop", type=float, default=0.05)
    # Matches compare_eval_results.py, so the two cannot drift.
    parser.add_argument("--near-lossless", type=float, default=0.98)
    parser.add_argument(
        "--discordance-scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="sweep the modelled suites' discordance, which nobody has measured",
    )
    parser.add_argument(
        "--with-candidate",
        action="store_true",
        help="also report the protocol with the parked SWE-bench Pro added back",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    runs = [(list(SUITES), "protocol")]
    if args.with_candidate:
        SUITES.update(CANDIDATE)
        runs.append((list(SUITES), "plus candidate"))

    results = []
    for scale in args.discordance_scale:
        for names, label in runs:
            report = simulate(
                names,
                trials=args.trials,
                seed=args.seed,
                any_suite_drop=args.any_suite_drop,
                max_macro_drop=args.max_macro_drop,
                confident_drop=args.confident_drop,
                near_lossless=args.near_lossless,
                discordance_scale=scale,
            )
            report["label"] = label
            report["discordance_scale"] = scale
            results.append(report)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"{args.trials} trials under the null, no degradation in either arm\n")
    header = (
        f"{'suites':>7} {'disc':>5} {'any-3pt':>9} {'macro-3pt':>10} "
        f"{'confident-5pt':>14} {'near-lossless fail':>19} {'recovery 95%':>13}"
    )
    print(header)
    for report in results:
        print(
            f"{report['suites']:>7} {report['discordance_scale']:>5.2f} "
            f"{report['any_suite_rule']:>8.1%} {report['macro_rule']:>10.2%} "
            f"{report['confident_suite_rule']:>14.2%} "
            f"{report['near_lossless_failure']:>19.1%} "
            f"±{100 * report['recovery_geomean_95_halfwidth']:>11.2f}pp"
        )
    # Show the widest suite table for the central assumption, not whichever
    # sweep point happened to run last.
    central = min(results, key=lambda r: (abs(r["discordance_scale"] - 1.0), -r["suites"]))
    if central["modelled"]:
        print(f"\nmodelled rather than measured: {', '.join(central['modelled'])}")
    print(f"per-suite noise at discordance {central['discordance_scale']:.2f}:")
    print(f"  {'suite':24s} {'SE(delta)':>10s} {'baseline':>9s} {'SE(recovery)':>13s}")
    for name, value in sorted(
        central["suite_delta_se"].items(), key=lambda item: -item[1] / SUITES[item[0]]["baseline"]
    ):
        baseline = SUITES[name]["baseline"]
        print(f"  {name:24s} {value:>10.5f} {baseline:>9.3f} {value / baseline:>12.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
