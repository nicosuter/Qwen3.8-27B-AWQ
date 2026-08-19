#!/usr/bin/env python3
"""Measure the per-suite output length that admission control reserves against.

Admission reserves `prompt + expected output` per request. The prompt is known
exactly; the output has to come from somewhere, and the only honest source is
what the suite actually generated last time. Run this over the run directories
on disk and check the result in, so the first job against a new checkpoint
starts calibrated instead of discovering the number by thrashing.

    eval/scripts/build_token_priors.py --output eval/token-priors.json RUN_DIR...
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

# What a suite reserves when it has never run. Wide enough that an unmeasured
# suite is throttled rather than allowed to flood, narrow enough that it still
# makes progress. Deliberately not the largest observed prior: an unknown suite
# resembling RULER is rarer than one resembling the rest.
DEFAULT_OUTPUT_TOKENS = 4096
# Enough to cover a text prompt with one image attached, so an unmeasured
# multimodal suite is not priced as if its prompt were the caption alone.
DEFAULT_PROMPT_TOKENS = 2048


def usable_output(row: dict[str, Any]) -> int | None:
    """Output length, or None when the row is not evidence of one.

    A timed-out or empty row records zero tokens generated. Folding those in
    would shrink the reservation in exactly the situation that says it was
    already too small, so the loop would run away from the right answer.
    """
    if row.get("timeout") or row.get("deferred"):
        return None
    tokens = row.get("output_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        return None
    return tokens


def collect(run_dirs: list[Path]) -> dict[str, Any]:
    """Median prompt and mean output length per suite, over both arms of every run.

    The prompt half is recorded because the client cannot recover it: a
    multimodal prompt is mostly image, and the only party that counted those
    tokens is the server that encoded them. It stays a median because it is a
    floor under a per-item estimate, applied as `max(estimate, prior)`, and the
    max of a draw and a constant sits above both -- a mean floor over-reserved
    RULER and multimodal by a quarter where the median lands both within a tenth.

    The output half is added rather than floored, and there the mean is the
    right statistic: a budget admits many requests at once and holds their sum,
    and the sum of N draws concentrates on N times the mean however skewed each
    one is. The median answers how long a single request runs, which is not what
    a shared budget is deciding, and on these suites the two are far apart --
    mean over median is 1.68x on livecodebench_v6, 2.16x on gpqa_diamond, 2.56x
    on mmmu_pro, 4.14x on multimodal and 30.94x on ruler, whose median item
    emits 653 tokens while a tenth of them run to the 131,072 cap.

    Not p90, which this file used to argue against and still does: p90 is 6.2x
    the median on gpqa_diamond and would idle most of the pool, against the
    mean's 2.16x. Measured over the runs on disk, the mean brings every suite's
    actual occupancy to between 0.88 and 1.10 of what it reserved, from as much
    as 2.27x under it before.
    """
    outputs: dict[str, list[int]] = {}
    prompts: dict[str, list[int]] = {}
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).glob("raw/*/*.jsonl")):
            suite = path.name.rsplit("-r", 1)[0]
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                tokens = usable_output(row)
                if tokens is None:
                    continue
                outputs.setdefault(suite, []).append(tokens)
                prompt = row.get("prompt_tokens")
                if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt > 0:
                    prompts.setdefault(suite, []).append(prompt)
    suites = {
        suite: {
            # Rounded up: truncation on the output half is a systematic
            # under-reservation, which is the whole failure being fixed.
            "prompt": int(statistics.median(prompts.get(suite) or [DEFAULT_PROMPT_TOKENS])),
            "output": math.ceil(statistics.fmean(values)),
        }
        for suite, values in sorted(outputs.items())
        if values
    }
    return {
        "suites": suites,
        "default": {"prompt": DEFAULT_PROMPT_TOKENS, "output": DEFAULT_OUTPUT_TOKENS},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    priors = collect(args.run_dirs)
    if not priors["suites"]:
        print("no scored rows found; refusing to write an empty priors file", file=sys.stderr)
        return 1
    priors["note"] = (
        "Median prompt and mean output tokens per suite, measured by "
        "eval/scripts/build_token_priors.py. Admission control reserves "
        "max(prompt estimate, prompt) + output. Rebuild after a checkpoint changes how "
        "long it reasons, or after a suite changes what it sends."
    )
    args.output.write_text(json.dumps(priors, indent=2) + "\n", encoding="utf-8")
    for suite, value in priors["suites"].items():
        print(f"{suite:20s} prompt {value['prompt']:8d}  output {value['output']:8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
