#!/usr/bin/env python3
"""Flag exact and near prompt overlap with the saved AWQ calibration corpus."""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-field", default="text")
    parser.add_argument("--eval-field", default="text")
    parser.add_argument("--shingle-size", type=int, default=8)
    parser.add_argument("--anchors", type=int, default=6)
    parser.add_argument("--containment-threshold", type=float, default=0.80)
    return parser.parse_args()


def read_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"{path}: expected a JSON array")
            for row in data:
                if isinstance(row, dict):
                    yield row
            return
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold()))


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def shingle_hashes(words: list[str], size: int) -> list[int]:
    if len(words) < size:
        return []
    result = []
    for index in range(len(words) - size + 1):
        value = "\x1f".join(words[index : index + size]).encode()
        result.append(int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big"))
    return result


def row_id(row: dict[str, Any], fallback: str) -> str:
    for field in ("id", "task_id", "question_id", "instance_id"):
        if field in row:
            return str(row[field])
    return fallback


def main() -> int:
    args = parse_args()
    if args.shingle_size < 2 or args.anchors < 1:
        raise ValueError("shingle size must be >= 2 and anchors must be positive")
    if not 0 < args.containment_threshold <= 1:
        raise ValueError("containment threshold must be in (0, 1]")

    calibration: list[tuple[str, str, list[str]]] = []
    for index, row in enumerate(read_json_rows(args.calibration)):
        raw = row.get(args.calibration_field)
        if isinstance(raw, str) and raw.strip():
            calibration.append((row_id(row, f"calibration:{index}"), normalize(raw), tokens(raw)))
    if not calibration:
        raise ValueError("no calibration text found")

    eval_rows: list[tuple[str, str, str, list[str], set[int]]] = []
    anchor_to_eval: dict[int, set[int]] = defaultdict(set)
    for path in args.eval:
        for index, row in enumerate(read_json_rows(path)):
            raw = row.get(args.eval_field)
            if not isinstance(raw, str) or not raw.strip():
                continue
            words = tokens(raw)
            hashes = set(shingle_hashes(words, args.shingle_size))
            eval_index = len(eval_rows)
            eval_rows.append((str(path), row_id(row, f"{path.name}:{index}"), normalize(raw), words, hashes))
            for anchor in sorted(hashes)[: args.anchors]:
                anchor_to_eval[anchor].add(eval_index)
    if not eval_rows:
        raise ValueError("no eval text found; export rows with id and text first")

    candidate_pairs: set[tuple[int, int]] = set()
    short_eval_indices = [index for index, row in enumerate(eval_rows) if not row[4]]
    for calibration_index, (_, _, words) in enumerate(calibration):
        for value in set(shingle_hashes(words, args.shingle_size)):
            for eval_index in anchor_to_eval.get(value, ()):
                candidate_pairs.add((calibration_index, eval_index))
        calibration_text = calibration[calibration_index][1]
        for eval_index in short_eval_indices:
            if eval_rows[eval_index][2] in calibration_text:
                candidate_pairs.add((calibration_index, eval_index))

    matches = []
    for calibration_index, eval_index in sorted(candidate_pairs):
        calibration_id, calibration_text, calibration_words = calibration[calibration_index]
        eval_path, eval_id, eval_text, _, eval_hashes = eval_rows[eval_index]
        calibration_hashes = set(shingle_hashes(calibration_words, args.shingle_size))
        exact_containment = bool(eval_text and eval_text in calibration_text)
        shared = len(eval_hashes & calibration_hashes)
        containment = shared / len(eval_hashes) if eval_hashes else 0.0
        if exact_containment or containment >= args.containment_threshold:
            union = len(eval_hashes | calibration_hashes)
            matches.append(
                {
                    "eval_file": eval_path,
                    "eval_id": eval_id,
                    "calibration_id": calibration_id,
                    "exact_containment": exact_containment,
                    "shingle_containment": containment,
                    "shingle_jaccard": shared / union if union else 0.0,
                }
            )

    report = {
        "calibration_file": str(args.calibration),
        "calibration_rows": len(calibration),
        "eval_rows": len(eval_rows),
        "shingle_size": args.shingle_size,
        "anchors": args.anchors,
        "containment_threshold": args.containment_threshold,
        "matches": matches,
    }
    print(
        f"calibration_rows={len(calibration)} eval_rows={len(eval_rows)} "
        f"flagged_matches={len(matches)}"
    )
    for match in matches[:20]:
        print(
            f"{match['eval_file']}:{match['eval_id']} -> {match['calibration_id']} "
            f"exact={match['exact_containment']} "
            f"containment={match['shingle_containment']:.3f}"
        )
    if len(matches) > 20:
        print(f"... {len(matches) - 20} more matches; see JSON report")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 2 if matches else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
