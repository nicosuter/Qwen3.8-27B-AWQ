"""Rerunning a suite that timed out re-measures the gaps, not the whole suite.

A timeout is a missing observation, not a bad one: the row carries a zero the
model never produced. The items that did answer are still evidence, so the
rerun only has to fill the holes.

This is deliberately not the relaxation the `--max-tokens` check in
paired-suite-eval.sbatch refuses. Truncation at the cap IS an observation, and
it lands on the longest reasoning, so keeping the items that fit and rescoring
the ones that did not would select on the outcome along the axis the two arms
differ on. A timeout produces no observation to select on, and every item in the
frozen order is still scored exactly once.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "scripts" / "adapters"))

_SPEC = importlib.util.spec_from_file_location(
    "_common", ROOT / "eval" / "scripts" / "adapters" / "_common.py"
)
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class CarriedForwardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "previous.jsonl"

    def test_nothing_is_carried_without_a_source(self):
        """Off by default. A rerun that was not told to resume is a full rerun."""
        self.assertEqual(common.carried_forward(None), {})
        self.assertEqual(common.carried_forward(""), {})

    def test_an_answered_item_is_carried(self):
        write(self.path, [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.0}])
        carried = common.carried_forward(self.path)
        self.assertEqual(sorted(carried), ["a", "b"])
        self.assertEqual(carried["a"]["score"], 1.0)

    def test_a_timed_out_item_is_not_carried(self):
        """The hole the rerun exists to fill."""
        write(self.path, [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.0, "timeout": True}])
        self.assertEqual(sorted(common.carried_forward(self.path)), ["a"])

    def test_a_deferred_item_is_not_carried(self):
        """Deferred means nobody has executed it yet, not that it scored zero."""
        write(self.path, [{"id": "a", "score": 0.0, "deferred": True}])
        self.assertEqual(common.carried_forward(self.path), {})

    def test_a_missing_source_is_an_error_not_a_silent_full_rerun(self):
        """Asked to resume from a file that is not there, stop.

        Continuing would quietly rerun everything, which is a correct result
        bought at the price the caller was trying not to pay -- and it would
        look identical to a resume that worked.
        """
        with self.assertRaises(common.AdapterError):
            common.carried_forward(Path(self.tmp.name) / "absent.jsonl")


class ExecuteOrderResumeTests(unittest.TestCase):
    def test_carried_items_do_not_reach_the_worker(self):
        called = []

        def worker(item_id):
            called.append(item_id)
            return {"id": item_id, "score": 1.0, "fresh": True}

        rows = common.execute_order(
            ["a", "b", "c"], worker, concurrency=2, reuse={"a": {"id": "a", "score": 0.5}}
        )
        self.assertEqual(called, ["b", "c"])
        self.assertEqual([r["id"] for r in rows], ["a", "b", "c"])
        self.assertEqual(rows[0]["score"], 0.5, "the carried row was re-measured")
        self.assertTrue(rows[1]["fresh"])

    def test_the_frozen_order_survives_a_resume(self):
        """Every item is present exactly once, in the order the seed fixed.

        The comparator pairs per item, so a resume that dropped or reordered one
        would silently compare different items to each other.
        """
        order = [str(i) for i in range(20)]
        reuse = {i: {"id": i, "score": 1.0} for i in order if int(i) % 3 == 0}
        rows = common.execute_order(
            order, lambda i: {"id": i, "score": 0.0}, concurrency=4, reuse=reuse
        )
        self.assertEqual([r["id"] for r in rows], order)

    def test_no_reuse_runs_everything(self):
        called = []
        common.execute_order(["a", "b"], lambda i: called.append(i) or {"id": i}, concurrency=1)
        self.assertEqual(called, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
