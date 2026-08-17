#!/usr/bin/env python3
"""Warm the shared fine-grained FP8 kernel cache before distributed loading."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


KERNELS = (("kernels-community/finegrained-fp8", 4),)


def prefetch(
    resolve: Callable[..., str] | None = None,
    install: Callable[..., Path] | None = None,
) -> list[tuple[str, str, Path]]:
    """Download kernel builds serially without importing their Triton modules."""
    if resolve is None or install is None:
        from kernels._versions import select_revision_or_version
        from kernels.utils import install_kernel

        resolve = select_revision_or_version
        install = install_kernel

    loaded = []
    for repo_id, version in KERNELS:
        revision = resolve(repo_id, revision=None, version=version)
        path = install(
            repo_id,
            revision=revision,
            validate_dependencies=True,
        )
        if not path.is_dir():
            raise RuntimeError(f"{repo_id} cache path is not a directory: {path}")
        loaded.append((repo_id, revision, path))
        print(
            f"fp8-kernel-cache=ok repo={repo_id} version={version} "
            f"revision={revision} path={path}"
        )
    return loaded


if __name__ == "__main__":
    prefetch()
