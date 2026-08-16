#!/usr/bin/env python3
"""Run EvalScope benchmarks against a pinned dataset and feed our comparator.

EvalScope already has the task definitions, the per-item output and a
generate/score split. What it does not have is an immutable dataset pin or the
paired statistics, and those are the two things our gate rests on. This bridges
the gap at both ends rather than reimplementing a hundred benchmarks.

`materialize` downloads a dataset at a 40-character commit and writes it where
EvalScope can be pointed at it as a local path. That is the whole pinning story:
`DataLoader` accepts a `version` and passes it to `load_dataset(revision=...)`,
but `DefaultDataAdapter.load_subset` never populates it, so a benchmark run from
the Hub always tracks whatever `main` happens to be that day.

`rows` converts `reviews/*.jsonl` into the result rows compare_eval_results.py
reads.

The id deserves the most care here. EvalScope defaults to `auto_id=True`, so a
sample id is that item's *position* in the loaded split. Pair two arms on those
and a dataset that shifted by one row silently compares different questions
rather than failing. Ids therefore carry a digest of the item's own prompt and
target, so a mismatch surfaces as a missing key in the comparator instead of a
plausible wrong number.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
# Mirrors _common.RESULT_BOOL_FIELDS. Every one defaults False: EvalScope does
# not report them, and inventing them from a truncated string would be worse
# than admitting we do not know.
BOOL_FIELDS = (
    "must_pass",
    "timeout",
    "empty_answer",
    "repetition_loop",
    "malformed_tool_call",
    "premature_final_answer",
    "context_failure",
)


class BridgeError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    materialize = sub.add_parser(
        "materialize", help="download a dataset at a pinned commit for local use"
    )
    materialize.add_argument("--repo", required=True, help="e.g. cais/hle")
    materialize.add_argument("--revision", required=True, help="40-character commit")
    materialize.add_argument("--into", required=True, type=Path)

    rows = sub.add_parser("rows", help="convert EvalScope reviews into result rows")
    rows.add_argument("--reviews", required=True, type=Path, help="reviews/*.jsonl")
    rows.add_argument("--suite", required=True)
    rows.add_argument("--output", required=True, type=Path)
    rows.add_argument(
        "--dataset-pin",
        required=True,
        help="repo@<40-hex> the reviews were produced against; recorded per row "
             "so a comparison across two revisions cannot be assembled by accident",
    )
    rows.add_argument("--metric", help="score key to read; defaults to the main one")
    return parser.parse_args(argv)


def command_materialize(args: argparse.Namespace) -> int:
    if not REVISION_RE.match(args.revision):
        raise BridgeError(
            f"--revision must be a 40-character commit; got {args.revision!r}. "
            "A branch or tag is not a pin."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - environment guard
        raise BridgeError("huggingface_hub is required for materialize") from error

    target = args.into / args.repo.replace("/", "__") / args.revision
    snapshot_download(
        args.repo, revision=args.revision, repo_type="dataset", local_dir=str(target)
    )
    # Written beside the data so a run directory carries the pin even when the
    # config that produced it is long gone.
    (target / ".pin.json").write_text(
        json.dumps({"repo": args.repo, "revision": args.revision}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"path": str(target), "pin": f"{args.repo}@{args.revision}"}))
    return 0


def content_id(record: dict[str, Any]) -> str:
    """A digest of the item itself, so a shifted dataset cannot masquerade.

    EvalScope's default sample id is positional. Two arms scored against
    different revisions would still join cleanly on those, and report a
    comparison of unrelated questions.
    """
    payload = json.dumps(
        [record.get("input"), record.get("target")], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main_score(score: dict[str, Any], metric: str | None) -> float:
    values = score.get("value")
    if not isinstance(values, dict) or not values:
        raise BridgeError(f"review has no score value: {score!r}")
    name = metric or score.get("main_score_name") or next(iter(values))
    if name not in values:
        raise BridgeError(f"metric {name!r} not in {sorted(values)}")
    value = values[name]
    if isinstance(value, bool):
        value = float(value)
    if not isinstance(value, (int, float)):
        raise BridgeError(f"metric {name!r} is {type(value).__name__}, not a number")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        # The comparator requires [0, 1] and rescaling here would silently
        # change what a recovery ratio means.
        raise BridgeError(
            f"metric {name!r} is {value}, outside [0, 1]; pick a normalized "
            "metric with --metric rather than rescaling after the fact"
        )
    return value


def convert(
    records: Iterable[dict[str, Any]],
    *,
    suite: str,
    dataset_pin: str,
    metric: str | None = None,
) -> list[dict[str, Any]]:
    if "@" not in dataset_pin or not REVISION_RE.match(dataset_pin.split("@", 1)[1]):
        raise BridgeError(
            f"--dataset-pin must be repo@<40-hex>; got {dataset_pin!r}. Without it "
            "there is nothing stopping two revisions being compared to each other."
        )

    by_group: dict[str, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        sample = record.get("sample_score")
        if not isinstance(sample, dict):
            raise BridgeError(f"review row has no sample_score: {record!r}")
        # repeats=k emits k samples sharing one group_id; the replicate index is
        # the position within that group, ordered by sample id for determinism.
        group = str(sample.get("group_id") if sample.get("group_id") is not None
                    else sample.get("sample_id"))
        by_group[group].append((sample.get("sample_id"), record))

    rows: list[dict[str, Any]] = []
    for group, entries in sorted(by_group.items()):
        entries.sort(key=lambda pair: (pair[0] is None, pair[0]))
        for replicate, (sample_id, record) in enumerate(entries):
            sample = record["sample_score"]
            row: dict[str, Any] = {
                "suite": suite,
                "id": f"{group}:{content_id(record)}",
                "replicate": replicate,
                "score": main_score(sample.get("score") or {}, metric),
            }
            row.update({field: False for field in BOOL_FIELDS})
            extracted = (sample.get("score") or {}).get("extracted_prediction")
            # The one failure flag the review actually supports.
            row["empty_answer"] = extracted is not None and not str(extracted).strip()
            row["evalscope_sample_id"] = sample_id
            row["dataset_pin"] = dataset_pin
            rows.append(row)

    if not rows:
        raise BridgeError("no reviews to convert")
    return rows


def command_rows(args: argparse.Namespace) -> int:
    records = [
        json.loads(line)
        for line in args.reviews.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = convert(
        records, suite=args.suite, dataset_pin=args.dataset_pin, metric=args.metric
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    replicates = len({row["replicate"] for row in rows})
    print(
        f"wrote {len(rows)} rows for {args.suite} "
        f"({len(rows) // max(replicates, 1)} items x {replicates} replicates) "
        f"to {args.output}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "materialize":
        return command_materialize(args)
    return command_rows(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BridgeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
