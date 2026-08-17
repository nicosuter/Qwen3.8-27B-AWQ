import unittest

from quant.scripts.distributed_lifecycle import run_rank0_after_group_teardown


class FakeDistributed:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.initialized = True

    def barrier(self) -> None:
        if not self.initialized:
            raise AssertionError("barrier called after process-group teardown")
        self.events.append("barrier")

    def destroy_process_group(self) -> None:
        if not self.initialized:
            raise AssertionError("process group destroyed twice")
        self.initialized = False
        self.events.append("destroy")


class DistributedLifecycleTests(unittest.TestCase):
    def test_rank0_saves_only_after_group_teardown(self) -> None:
        events: list[str] = []
        distributed = FakeDistributed(events)
        state = {"distributed_cache": True}

        def prepare() -> None:
            self.assertTrue(distributed.initialized)
            state["distributed_cache"] = False
            events.append("prepare")

        def save() -> None:
            self.assertFalse(distributed.initialized)
            self.assertFalse(state["distributed_cache"])
            events.append("save")

        saved = run_rank0_after_group_teardown(
            0,
            distributed,
            prepare,
            save,
        )
        self.assertTrue(saved)
        self.assertEqual(events, ["prepare", "barrier", "destroy", "save"])

    def test_nonzero_rank_never_saves(self) -> None:
        events: list[str] = []
        saved = run_rank0_after_group_teardown(
            3,
            FakeDistributed(events),
            lambda: events.append("prepare"),
            lambda: events.append("save"),
        )
        self.assertFalse(saved)
        self.assertEqual(events, ["prepare", "barrier", "destroy"])

    def test_save_failure_happens_after_all_ranks_are_released(self) -> None:
        events: list[str] = []
        distributed = FakeDistributed(events)

        def fail_save() -> None:
            self.assertFalse(distributed.initialized)
            events.append("save")
            raise RuntimeError("write failed")

        with self.assertRaisesRegex(RuntimeError, "write failed"):
            run_rank0_after_group_teardown(
                0,
                distributed,
                lambda: events.append("prepare"),
                fail_save,
            )
        self.assertEqual(events, ["prepare", "barrier", "destroy", "save"])


if __name__ == "__main__":
    unittest.main()
