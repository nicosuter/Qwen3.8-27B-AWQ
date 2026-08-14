#!/usr/bin/env python3
"""Require exact tokenizer and chat-template identity for paired evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from transformers.utils import cached_file


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_file(model: str, filename: str, revision: str | None) -> Path | None:
    local = Path(model)
    if local.is_dir():
        path = local / filename
        return path if path.is_file() else None
    resolved = cached_file(
        model,
        filename,
        revision=revision,
        _raise_exceptions_for_gated_repo=True,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=True,
    )
    return Path(resolved) if resolved else None


def fingerprint(model: str, revision: str | None) -> dict[str, Any]:
    files = {}
    for filename in TOKENIZER_FILES:
        path = resolve_file(model, filename, revision)
        files[filename] = sha256_bytes(path.read_bytes()) if path else None
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=True
    )
    chat_template = tokenizer.chat_template
    if isinstance(chat_template, dict):
        rendered_template = json.dumps(
            chat_template, sort_keys=True, separators=(",", ":")
        )
    else:
        rendered_template = str(chat_template or "")
    return {
        "model": model,
        "revision": revision,
        "files": files,
        "chat_template_sha256": sha256_bytes(rendered_template.encode()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = fingerprint(args.baseline, args.baseline_revision)
    candidate = fingerprint(args.candidate, None)
    mismatches = [
        filename
        for filename in TOKENIZER_FILES
        if baseline["files"][filename] != candidate["files"][filename]
    ]
    if baseline["chat_template_sha256"] != candidate["chat_template_sha256"]:
        mismatches.append("effective_chat_template")
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(
            "tokenizer/chat-template mismatch: " + ", ".join(mismatches)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
