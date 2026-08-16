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
import os
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


BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
# Every category scored by matching a call against a key or by whether a call
# was made. multi_turn is excluded on purpose: its rows need an initial_config
# and the Gorilla simulators' state, neither of which exists in the static files.
BFCL_AST_CATEGORIES = (
    "simple", "multiple", "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance",
)
BFCL_NO_GROUND_TRUTH = ("irrelevance", "live_irrelevance", "live_relevance")


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

    bfcl = sub.add_parser(
        "bfcl-dataset",
        help="build EvalScope's BFCL layout from our pinned upstream revision",
    )
    bfcl.add_argument("--revision", default=BFCL_REVISION)
    bfcl.add_argument("--into", required=True, type=Path)
    bfcl.add_argument(
        "--categories",
        default=",".join(BFCL_AST_CATEGORIES),
        help="comma-separated; multi_turn is not derivable and is refused",
    )

    run = sub.add_parser("run", help="materialize a pinned dataset and score it")
    run.add_argument("--suites", type=Path, default=Path("eval/evalscope-suites.json"))
    run.add_argument("--suite", required=True, help="a `suite` name from that file")
    run.add_argument("--model", required=True, help="served model name")
    run.add_argument("--api-url", required=True)
    run.add_argument("--api-key", default="EMPTY")
    run.add_argument("--work-dir", required=True, type=Path)
    # Shared across lanes on purpose: a per-lane default would have every
    # (suite, replicate) re-download its own copy of the same pinned dataset.
    run.add_argument(
        "--datasets-root", type=Path,
        help="where materialize put the pinned datasets; defaults to the "
             "materialize_root in the suites file, relative to it",
    )
    run.add_argument("--variant", required=True, choices=("baseline", "candidate"))
    run.add_argument("--repeats", type=int, default=1)
    # EvalScope defaults eval_batch_size to 1, i.e. one request in flight. A
    # 12k-item suite would take days, and the server sits idle between replies.
    run.add_argument("--concurrency", type=int, default=64)
    # Applied per subset, not per suite: mmlu_pro has 14 subsets, so --limit 20
    # scores 280 items. Useful for smoke tests, misleading otherwise.
    run.add_argument("--limit", type=float, default=None)
    # A lane writes exactly where the sbatch's concatenation and reuse check
    # look, so swapping the runner changes nothing downstream of it.
    run.add_argument("--replicate", type=int, default=0)
    run.add_argument("--results", type=Path, help="raw/<variant>/<suite>-r<n>.jsonl")
    run.add_argument("--metadata", type=Path, help="metadata/<suite>-<variant>-r<n>.json")
    run.add_argument("--request-timeout", type=float, default=5400.0)
    # Matches the protocol's own cap, so a comparison against our adapters is
    # not confounded by one side getting a different budget.
    run.add_argument("--max-tokens", type=int, default=131072)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=38027)
    run.add_argument("--print-only", action="store_true")

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
            # A deferred review is an unscored item, not a failed one. The
            # comparator refuses any file containing one, so a half-scored
            # suite cannot be read as a result.
            if ((sample.get("score") or {}).get("metadata") or {}).get("deferred"):
                row["deferred"] = True
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


def bfcl_rows(
    by_category: dict[str, list[dict[str, Any]]],
    answers: dict[str, dict[str, Any]],
    build_tools: Any,
) -> list[dict[str, Any]]:
    """Reshape upstream BFCL into the layout EvalScope's adapter expects.

    Upstream ships prompts and possible_answer as two files; EvalScope reads one
    record carrying both, with functions/tools/turns/ground_truth as JSON
    strings. Everything here comes from the pinned upstream files, so the only
    thing being trusted is the reshaping.
    """
    rows: list[dict[str, Any]] = []
    for category, records in by_category.items():
        if category not in BFCL_AST_CATEGORIES:
            raise BridgeError(
                f"{category} is not derivable from the static files: multi_turn "
                "needs an initial_config and the Gorilla simulators' state"
            )
        for record in records:
            item_id = str(record["id"])
            functions = record.get("function") or []
            if not functions:
                # Same degenerate rows our own adapter drops: nothing to call,
                # so abstaining is automatic and the item separates nothing.
                continue
            ground_truth: Any = {}
            if category not in BFCL_NO_GROUND_TRUTH:
                answer = answers.get(item_id)
                if answer is None:
                    continue  # upstream id typo; our adapter drops it too
                ground_truth = answer["ground_truth"]
            rows.append(
                {
                    "id": item_id,
                    "subset": category,
                    "multi_turn": False,
                    "functions": json.dumps(functions),
                    "tools": json.dumps(build_tools(functions)),
                    "turns": json.dumps(record["question"]),
                    "missed_functions": json.dumps([]),
                    "initial_config": json.dumps({}),
                    "ground_truth": json.dumps(ground_truth),
                }
            )
    if not rows:
        raise BridgeError("no BFCL rows built")
    return rows


def command_bfcl_dataset(args: argparse.Namespace) -> int:
    if not REVISION_RE.match(args.revision):
        raise BridgeError(f"--revision must be a 40-character commit; got {args.revision!r}")
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    sys.path.insert(0, str(Path(__file__).resolve().parent / "adapters"))
    try:
        import bfcl as our_bfcl  # our adapter already converts BFCL schemas
    except ImportError as error:  # pragma: no cover - environment guard
        raise BridgeError("scripts/adapters/bfcl.py is required") from error

    by_category, answers = {}, {}
    for category in categories:
        name = our_bfcl.CATEGORIES.get(category)
        if name is None:
            raise BridgeError(f"unknown category {category!r}")
        by_category[category] = our_bfcl.download_json_lines(name, args.revision)
        if category not in BFCL_NO_GROUND_TRUTH:
            for row in our_bfcl.download_json_lines(
                f"possible_answer/{name}", args.revision
            ):
                answers[str(row["id"])] = row

    rows = bfcl_rows(by_category, answers, our_bfcl.build_tools)
    target = args.into / "bfcl_v3" / args.revision
    target.mkdir(parents=True, exist_ok=True)
    out = target / "train.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (target / ".pin.json").write_text(
        json.dumps(
            {"repo": BFCL_REPO, "revision": args.revision,
             "categories": categories, "rows": len(rows),
             "built_by": "evalscope_bridge.py bfcl-dataset"},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["subset"]] = counts.get(row["subset"], 0) + 1
    print(json.dumps({"path": str(target), "rows": len(rows),
                      "by_subset": dict(sorted(counts.items()))}, indent=2))
    return 0


def load_suite(path: Path, name: str) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    for entry in spec["suites"]:
        if entry["suite"] == name:
            break
    else:
        names = sorted(e["suite"] for e in spec["suites"])
        raise BridgeError(f"unknown suite {name!r}; {path} has {names}")
    if not entry.get("ported"):
        raise BridgeError(
            f"{name} is listed but not ported: {entry.get('note', 'no reason recorded')}"
        )
    if not REVISION_RE.match(entry.get("revision") or ""):
        raise BridgeError(f"{name} has no pinned revision in {path}")
    # Carried on the entry so `run` can find the shared materialized datasets
    # without every caller having to pass the same path.
    entry["_root"] = spec.get("materialize_root") or "eval-materialized/evalscope"
    return entry


def command_run(args: argparse.Namespace) -> int:
    entry = load_suite(args.suites, args.suite)
    pin = f"{entry['repo']}@{entry['revision']}"

    root = args.datasets_root or (
        args.suites.parent.parent / (entry.get("_root") or "eval-materialized/evalscope")
    )
    if entry.get("build") == "bfcl-dataset":
        # Built rather than downloaded: upstream ships prompts and keys apart.
        local = root / "bfcl_v3" / entry["revision"]
    else:
        local = root / entry["repo"].replace("/", "__") / entry["revision"]
    materialize = argparse.Namespace(
        repo=entry["repo"], revision=entry["revision"], into=root
    )
    if not args.print_only and not local.is_dir():
        command_materialize(materialize)

    # dataset_id is overridden to the materialized path. EvalScope names
    # ModelScope mirrors, but its adapters read the original column names, so
    # pointing it at our own pinned snapshot gives both harnesses the same bytes
    # -- and is the only way to pin at all, since load_subset never passes the
    # `version` its DataLoader accepts.
    # Per-suite overrides: gpqa passes its subset as the HF config name and
    # EvalScope declares 'default', which is not a config of Idavidrein/gpqa;
    # mmmu_pro otherwise falls back to the 4-option set with a warning.
    dataset_args = dict(entry.get("dataset_args") or {})
    dataset_args["dataset_id"] = str(local)

    task = {
        "model": args.model,
        "eval_type": "openai_api",
        "api_url": args.api_url,
        "api_key": args.api_key,
        "datasets": [entry["benchmark"]],
        "dataset_args": {entry["benchmark"]: dataset_args},
        "dataset_hub": "local",
        "work_dir": str(args.work_dir / "runs" / args.variant),
        # TaskConfig.seed feeds seed_everything() and the loader's shuffle,
        # which we do not use. It is never copied into GenerateConfig, so no
        # per-request seed reaches the server -- and that is the behaviour we
        # want. A single seed sent on every request would make each item draw
        # the same uniform stream against different logits, correlating their
        # sampling noise, and the item-clustered bootstrap assumes items are
        # independent. It would report an interval narrower than the truth.
        # Our own adapters avoid this by deriving a seed per item from the
        # prompt; EvalScope cannot, so it gets independence without
        # reproducibility, which is the half that the statistics need.
        "seed": args.seed,
        "repeats": args.repeats,
        "eval_batch_size": args.concurrency,
        "generation_config": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    }
    if args.limit is not None:
        task["limit"] = args.limit

    print(json.dumps({"suite": args.suite, "pin": pin, "dataset": str(local),
                      "task": task}, indent=2))
    if args.print_only:
        return 0

    try:
        from evalscope import TaskConfig, run_task
    except ImportError as error:
        raise BridgeError("evalscope is required for run") from error
    if entry.get("plugin"):
        # LiveCodeBench executes generated code locally unless told not to.
        import importlib.util as _il
        spec = _il.spec_from_file_location("_es_plugin", entry["plugin"])
        module = _il.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]

    run_task(TaskConfig(**task))

    if args.results:
        rows = collect_rows(Path(task["work_dir"]), entry, args)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with args.results.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.metadata:
            write_lane_metadata(args, entry, pin, rows)
        print(f"scored {len(rows)} {args.suite} rows to {args.results}", flush=True)
    print(f"pin={pin} work_dir={task['work_dir']}", flush=True)
    return 0


def collect_rows(
    work_dir: Path, entry: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Convert every subset's reviews from the newest run under work_dir.

    A benchmark writes one reviews file per subset, and mmlu_pro alone has 14 of
    them, so a lane's rows are the union rather than a single file.
    """
    runs = sorted(p for p in work_dir.glob("*") if p.is_dir())
    if not runs:
        raise BridgeError(f"{work_dir}: evalscope wrote no run directory")
    reviews = sorted((runs[-1] / "reviews").rglob("*.jsonl"))
    if not reviews:
        raise BridgeError(f"{runs[-1]}: no reviews written")
    records: list[dict[str, Any]] = []
    for path in reviews:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    rows = convert(
        records,
        suite=args.suite,
        dataset_pin=f"{entry['repo']}@{entry['revision']}",
        metric=entry.get("metric"),
    )
    # The lane owns the replicate index; EvalScope's group_id only orders
    # repeats within one invocation, and we run one replicate per lane.
    for row in rows:
        row["replicate"] = args.replicate
    return rows


def write_lane_metadata(
    args: argparse.Namespace, entry: dict[str, Any], pin: str, rows: list[dict[str, Any]]
) -> None:
    """Write what suite_is_current reads, so reuse works as it does for adapters."""
    checkpoint: Any = None
    raw = os.environ.get("EVAL_CHECKPOINT_JSON", "").strip()
    if raw:
        try:
            checkpoint = json.loads(raw)
        except json.JSONDecodeError:
            checkpoint = {"error": "EVAL_CHECKPOINT_JSON was not valid JSON"}
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "variant": args.variant,
                "replicate": args.replicate,
                "seed": args.seed,
                "served_model": args.model,
                "items": len(rows),
                "max_tokens": args.max_tokens,
                "request_timeout_seconds": args.request_timeout,
                # EvalScope reports no per-request timeout, so none can be
                # claimed. suite_is_current treats a non-zero value as "a
                # timeout may have fired", which is the safe reading.
                "timeouts": None,
                "checkpoint": checkpoint,
                "runner": "evalscope",
                "benchmark": entry["benchmark"],
                "dataset_pin": pin,
                "score": round(sum(r["score"] for r in rows) / len(rows), 6) if rows else None,
                "deferred": any(r.get("deferred") for r in rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "materialize":
        return command_materialize(args)
    if args.action == "bfcl-dataset":
        return command_bfcl_dataset(args)
    if args.action == "run":
        return command_run(args)
    return command_rows(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BridgeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
