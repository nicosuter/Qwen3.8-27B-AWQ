"""Small, dependency-free helpers for distributed process lifecycle."""

from collections.abc import Callable
from typing import Any


def run_rank0_after_group_teardown(
    rank: int,
    distributed: Any,
    action: Callable[[], None],
) -> bool:
    """Synchronize all ranks, destroy the group, then run ``action`` on rank 0.

    This is valid only when rank 0 owns a complete model replica. Destroying the
    group before serialization prevents save-time helpers from accidentally
    selecting distributed algorithms whose collectives cannot be matched while
    the other ranks wait for rank 0 to write.
    """
    distributed.barrier()
    distributed.destroy_process_group()
    if rank != 0:
        return False
    action()
    return True
