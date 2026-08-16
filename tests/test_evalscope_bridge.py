import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = load_module("evalscope_bridge", "scripts/evalscope_bridge.py")

PIN = "cais/hle@" + "a" * 40


def review(sample_id, *, group_id=None, acc=1.0, text="q", target="18",
           extracted="18", value=None, main=None):
    return {
        "index": str(sample_id),
        "input": text,
        "target": target,
        "sample_score": {
            "score": {
                "value": {"acc": acc} if value is None else value,
                "extracted_prediction": extracted,
                "prediction": extracted,
                "main_score_name": main,
                "metadata": {},
            },
            "sample_id": sample_id,
            "group_id": sample_id if group_id is None else group_id,
            "sample_metadata": {},
        },
    }


class ConvertTests(unittest.TestCase):
    def test_a_review_becomes_a_result_row(self) -> None:
        rows = bridge.convert([review(0)], suite="hle", dataset_pin=PIN)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["suite"], "hle")
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["replicate"], 0)
        self.assertEqual(row["dataset_pin"], PIN)
        for field in bridge.BOOL_FIELDS:
            self.assertIn(field, row)

    def test_the_id_is_derived_from_the_item_not_its_position(self) -> None:
        # EvalScope's sample id is positional. A dataset that shifted would
        # otherwise join cleanly and compare different questions.
        same = bridge.convert([review(0, text="q1")], suite="s", dataset_pin=PIN)
        moved = bridge.convert([review(7, text="q1")], suite="s", dataset_pin=PIN)
        self.assertNotEqual(same[0]["id"], moved[0]["id"])
        self.assertEqual(same[0]["id"].split(":")[1], moved[0]["id"].split(":")[1])

    def test_two_arms_on_the_same_item_produce_the_same_id(self) -> None:
        a = bridge.convert([review(3, acc=1.0)], suite="s", dataset_pin=PIN)
        b = bridge.convert([review(3, acc=0.0)], suite="s", dataset_pin=PIN)
        self.assertEqual(a[0]["id"], b[0]["id"])
        self.assertNotEqual(a[0]["score"], b[0]["score"])

    def test_a_changed_question_changes_the_id(self) -> None:
        a = bridge.convert([review(3, text="q1")], suite="s", dataset_pin=PIN)
        b = bridge.convert([review(3, text="q2")], suite="s", dataset_pin=PIN)
        self.assertNotEqual(a[0]["id"], b[0]["id"])

    def test_completion_order_does_not_change_the_rows(self) -> None:
        # Reviews are written in completion order, not item order.
        forward = bridge.convert(
            [review(0, text="a"), review(1, text="b")], suite="s", dataset_pin=PIN
        )
        backward = bridge.convert(
            [review(1, text="b"), review(0, text="a")], suite="s", dataset_pin=PIN
        )
        self.assertEqual(
            sorted(r["id"] for r in forward), sorted(r["id"] for r in backward)
        )

    def test_repeats_become_replicates(self) -> None:
        records = [review(i, group_id=5, text="q") for i in (2, 0, 1)]
        rows = bridge.convert(records, suite="s", dataset_pin=PIN)
        self.assertEqual(sorted(r["replicate"] for r in rows), [0, 1, 2])
        self.assertEqual(len({r["id"] for r in rows}), 1, "one item, three draws")

    def test_an_empty_extraction_is_flagged(self) -> None:
        rows = bridge.convert([review(0, extracted="  ")], suite="s", dataset_pin=PIN)
        self.assertTrue(rows[0]["empty_answer"])
        rows = bridge.convert([review(0, extracted="18")], suite="s", dataset_pin=PIN)
        self.assertFalse(rows[0]["empty_answer"])

    def test_the_named_main_metric_wins(self) -> None:
        rec = review(0, value={"pass": 0.0, "acc": 1.0}, main="acc")
        self.assertEqual(bridge.convert([rec], suite="s", dataset_pin=PIN)[0]["score"], 1.0)

    def test_an_explicit_metric_overrides(self) -> None:
        rec = review(0, value={"acc": 1.0, "f1": 0.5})
        rows = bridge.convert([rec], suite="s", dataset_pin=PIN, metric="f1")
        self.assertEqual(rows[0]["score"], 0.5)

    def test_a_score_outside_the_unit_interval_is_refused(self) -> None:
        # Rescaling here would quietly change what a recovery ratio means.
        rec = review(0, value={"bleu": 42.0})
        with self.assertRaises(bridge.BridgeError):
            bridge.convert([rec], suite="s", dataset_pin=PIN)

    def test_a_missing_metric_is_refused(self) -> None:
        with self.assertRaises(bridge.BridgeError):
            bridge.convert([review(0)], suite="s", dataset_pin=PIN, metric="nope")

    def test_an_unpinned_dataset_is_refused(self) -> None:
        for pin in ("", "cais/hle", "cais/hle@main", "a" * 40):
            with self.subTest(pin=pin):
                with self.assertRaises(bridge.BridgeError):
                    bridge.convert([review(0)], suite="s", dataset_pin=pin)

    def test_a_row_without_a_sample_score_is_refused(self) -> None:
        with self.assertRaises(bridge.BridgeError):
            bridge.convert([{"input": "q", "target": "a"}], suite="s", dataset_pin=PIN)


class ComparatorContractTests(unittest.TestCase):
    """The rows have to survive compare_eval_results.py's own loader."""

    def test_rows_load_in_the_comparator(self) -> None:
        comparator = load_module("compare_eval", "scripts/compare_eval_results.py")
        rows = bridge.convert(
            [review(0, text="a", acc=1.0), review(1, text="b", acc=0.0)],
            suite="hle", dataset_pin=PIN,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            loaded = comparator.load_rows(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual({v["score"] for v in loaded.values()}, {0.0, 1.0})


class MaterializeTests(unittest.TestCase):
    def test_a_branch_is_not_a_pin(self) -> None:
        args = bridge.parse_args(
            ["materialize", "--repo", "cais/hle", "--revision", "main", "--into", "/tmp/x"]
        )
        with self.assertRaises(bridge.BridgeError):
            bridge.command_materialize(args)


if __name__ == "__main__":
    unittest.main()
