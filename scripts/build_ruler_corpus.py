#!/usr/bin/env python3
"""Materialize the RULER haystack corpus from a pinned dataset revision.

The corpus is pinned by content hash rather than by revision, so this script
exists to make that hash reproducible: the same dataset revision and the same
normalization must yield byte-identical output on any machine.
"""

import argparse
import hashlib
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_DATASET = "baber/paul_graham_essays"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--revision", required=True, help="immutable dataset commit, not a branch"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_documents(dataset: str, revision: str, split: str, field: str) -> list[str]:
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split, revision=revision)
    if field not in rows.column_names:
        raise ValueError(f"{dataset} has no {field!r} column; got {rows.column_names}")
    return [str(row) for row in rows[field]]


def normalize(documents: Iterable[str]) -> str:
    """Deterministic text: NFC, Unix line endings, no trailing space, one blank line between."""
    cleaned = []
    for document in documents:
        text = unicodedata.normalize("NFC", document).replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        stripped = "\n".join(lines).strip()
        if stripped:
            cleaned.append(stripped)
    if not cleaned:
        raise ValueError("the dataset produced no non-empty documents")
    return "\n\n".join(cleaned) + "\n"


def build(
    args: argparse.Namespace, loader: Callable[..., list[str]] = load_documents
) -> dict[str, Any]:
    documents = loader(args.dataset, args.revision, args.split, args.text_field)
    corpus = normalize(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    return {
        "dataset": args.dataset,
        "revision": args.revision,
        "documents": len(documents),
        "characters": len(corpus),
        "output": str(args.output),
        "sha256": "sha256:" + hashlib.sha256(corpus.encode()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    report = build(parse_args(argv))
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
