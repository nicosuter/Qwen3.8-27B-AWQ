#!/usr/bin/env python3
"""Publish a quantized checkpoint to the Hub without leaving stale weights behind.

`hf upload` adds and updates but never removes, so publishing a reshard over an
earlier layout leaves both sets of weights in the repository: twice the
download, and shards the index does not reference. It also ignores
`.gitignore`, so local state is published unless excluded.

This plans the commit first, refuses to publish a structurally broken artifact,
and only uploads when told to. Deletion is limited to weight files by default,
so nothing maintained solely on the Hub, the model card included, can be removed
by accident.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "quant" / "scripts"))

EXCLUDED = (".omc/*", ".gitignore", "*.log", "__pycache__/*", ".DS_Store", "*.tmp")
DEFAULT_DELETE = ("*.safetensors",)
REQUIRED_FILES = ("config.json", "model.safetensors.index.json")


class PublishError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="for example nicosuter/Qwen3.8-27B-AWQ")
    parser.add_argument("--path", type=Path, required=True, help="artifact directory")
    parser.add_argument("--message", default="Publish checkpoint")
    parser.add_argument(
        "--delete",
        action="append",
        default=None,
        help=f"remote patterns to prune; defaults to {' '.join(DEFAULT_DELETE)}",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually upload; without it the plan is printed and nothing changes",
    )
    parser.add_argument(
        "--skip-mtp-check",
        action="store_true",
        help="skip the MTP structural check, for checkpoints that carry no MTP head",
    )
    return parser.parse_args(argv)


def local_files(path: Path) -> list[str]:
    """Every file that would be published, relative and sorted."""
    if not path.is_dir():
        raise PublishError(f"{path} is not a directory")
    names = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path).as_posix()
        if any(part in (".omc", "__pycache__") for part in relative.split("/")):
            continue
        if relative in (".gitignore", ".DS_Store") or relative.endswith((".log", ".tmp")):
            continue
        names.append(relative)
    return names


def check_artifact(path: Path, *, skip_mtp: bool = False) -> dict[str, Any]:
    """Refuse to publish something that would not load."""
    for name in REQUIRED_FILES:
        if not (path / name).is_file():
            raise PublishError(f"{path} has no {name}")

    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PublishError("model.safetensors.index.json has no weight_map")

    referenced = set(weight_map.values())
    present = {candidate.name for candidate in path.glob("*.safetensors")}
    missing = sorted(referenced - present)
    if missing:
        raise PublishError(f"index references shards that are not here: {missing[:5]}")
    # An unreferenced shard is exactly the stale-weight problem, caught locally.
    orphaned = sorted(present - referenced)
    if orphaned:
        raise PublishError(
            f"these shards are not referenced by the index: {orphaned[:5]}. "
            "Remove them before publishing; they are almost certainly from an "
            "earlier layout."
        )

    report: dict[str, Any] = {
        "shards": len(referenced),
        "tensors": len(weight_map),
        "bytes": sum((path / name).stat().st_size for name in referenced),
    }
    if not skip_mtp:
        try:
            from preserve_mtp import (
                QWEN38_MTP_KEYS,
                QWEN38_MTP_LINEAR_MODULES,
                QWEN38_MTP_SHAPES,
                validate_mtp_artifact,
            )
        except ImportError as error:  # pragma: no cover - environment guard
            raise PublishError("preserve_mtp is required for the MTP check") from error
        result = validate_mtp_artifact(
            path,
            expected_keys=QWEN38_MTP_KEYS,
            expected_modules=QWEN38_MTP_LINEAR_MODULES,
            expected_shapes=QWEN38_MTP_SHAPES,
        )
        report["mtp_tensors"] = result["mtp_parameters"]
        report["packed_tensors"] = result["packed_weights"]
    return report


def plan_commit(
    names: list[str], remote: list[str], delete_patterns: tuple[str, ...]
) -> dict[str, list[str]]:
    """Split the remote side into kept, replaced and pruned."""
    from fnmatch import fnmatch

    local = set(names)
    pruned = sorted(
        name for name in remote
        if name not in local and any(fnmatch(name, pattern) for pattern in delete_patterns)
    )
    orphans = sorted(
        name for name in remote
        if name not in local and name not in pruned
    )
    return {
        "upload": sorted(local - set(remote)),
        "replace": sorted(local & set(remote)),
        "prune": pruned,
        "left_alone": orphans,
    }


def publish(
    args: argparse.Namespace,
    *,
    list_remote: Callable[[str], list[str]] | None = None,
    upload: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    delete_patterns = tuple(args.delete) if args.delete else DEFAULT_DELETE
    report = check_artifact(args.path, skip_mtp=args.skip_mtp_check)
    names = local_files(args.path)
    if "README.md" not in names:
        print("note: no README.md locally; the Hub copy is kept", file=sys.stderr)

    if list_remote is None or upload is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise PublishError("huggingface_hub is required") from error
        api = HfApi()
        list_remote = lambda repo: api.list_repo_files(repo, repo_type="model")  # noqa: E731
        upload = api.upload_folder

    try:
        remote = list(list_remote(args.repo))
    except Exception as error:  # noqa: BLE001 - a new repository has no files yet
        print(f"note: could not list {args.repo} ({error}); treating it as empty", file=sys.stderr)
        remote = []

    plan = plan_commit(names, remote, delete_patterns)
    print(f"repository      : {args.repo}")
    print(f"artifact        : {args.path}")
    print(f"shards/tensors  : {report['shards']} / {report['tensors']}"
          f"  ({report['bytes'] / 2**30:.2f} GB)")
    if "mtp_tensors" in report:
        print(f"mtp/packed      : {report['mtp_tensors']} / {report['packed_tensors']}")
    print(f"new files       : {len(plan['upload'])}")
    print(f"replaced files  : {len(plan['replace'])}")
    print(f"pruned remotely : {len(plan['prune'])}" +
          (f"  {plan['prune'][:4]}" if plan["prune"] else ""))
    if plan["left_alone"]:
        print(f"left on the Hub : {len(plan['left_alone'])}  {plan['left_alone'][:4]}")

    if not args.execute:
        print("\nplan only; rerun with --execute to publish")
        return {"report": report, "plan": plan, "executed": False}

    upload(
        repo_id=args.repo,
        folder_path=str(args.path),
        repo_type="model",
        commit_message=args.message,
        ignore_patterns=list(EXCLUDED),
        delete_patterns=list(delete_patterns),
    )
    print("\npublished")
    return {"report": report, "plan": plan, "executed": True}


def main(argv: list[str] | None = None) -> int:
    publish(parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PublishError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
