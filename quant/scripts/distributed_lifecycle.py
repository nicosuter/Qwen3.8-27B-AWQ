"""Small, dependency-free helpers for distributed process lifecycle."""

from collections.abc import Callable
from typing import Any


def run_rank0_after_group_teardown(
    rank: int,
    distributed: Any,
    prepare: Callable[[], None],
    action: Callable[[], None],
) -> bool:
    """Prepare every replica, tear down the group, then act only on rank 0.

    ``prepare`` runs while distributed communication is still available and
    must remove any model state whose later mutation would issue collectives.
    This is valid only when rank 0 owns a complete model replica. Destroying the
    group before ``action`` prevents save-time helpers from selecting distributed
    algorithms whose collectives cannot be matched while other ranks wait.
    """
    prepare()
    distributed.barrier()
    distributed.destroy_process_group()
    if rank != 0:
        return False
    action()
    return True
