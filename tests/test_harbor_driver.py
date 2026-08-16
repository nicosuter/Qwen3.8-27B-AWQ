"""Rules shared by every Harbor-executed suite, tested once where they live.

Reward extraction and dataset-pin parsing used to sit in the Terminal-Bench
adapter. They now serve SWE-bench Pro as well, and a second copy of "what counts
as a reward" would eventually drift from the first and show up as a score
difference between suites that had nothing to do with the model.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "adapters"))

SPEC = importlib.util.spec_from_file_location(
    "_harbor", ROOT / "scripts" / "adapters" / "_harbor.py"
)
assert SPEC and SPEC.loader
harbor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harbor)

DIGEST = "sha256:" + "1" * 64


def trial(rewards=None):
    row = {"task_name": "a"}
    if rewards is not None:
        row["verifier_result"] = {"rewards": rewards}
    return row


class RewardTests(unittest.TestCase):
    def test_named_reward_key_wins(self) -> None:
        self.assertEqual(harbor.extract_reward(trial({"reward": 1.0, "partial": 0.0})), 1.0)

    def test_multiple_checks_average(self) -> None:
        self.assertAlmostEqual(
            harbor.extract_reward(trial({"check1": 1.0, "check2": 0.0})), 0.5)

    def test_reward_is_clamped(self) -> None:
        self.assertEqual(harbor.extract_reward(trial({"reward": 5.0})), 1.0)
        self.assertEqual(harbor.extract_reward(trial({"reward": -2.0})), 0.0)

    def test_missing_verifier_result(self) -> None:
        self.assertIsNone(harbor.extract_reward(trial()))

    def test_non_numeric_rewards_are_not_scored(self) -> None:
        """An unscored trial has to stay distinguishable from a zero-scoring one."""
        self.assertIsNone(harbor.extract_reward(trial({"reward": "passed"})))

    def test_an_empty_reward_map_is_not_scored(self) -> None:
        self.assertIsNone(harbor.extract_reward(trial({})))


class DatasetPinTests(unittest.TestCase):
    def test_a_plain_pin_parses(self) -> None:
        self.assertEqual(
            harbor.parse_dataset_pin("terminal-bench/terminal-bench-2-1@2.1.0"),
            ("terminal-bench/terminal-bench-2-1", "2.1.0", None),
        )

    def test_a_subset_pin_parses(self) -> None:
        self.assertEqual(
            harbor.parse_dataset_pin(f"swebenchpro@1.0+subset:{DIGEST}"),
            ("swebenchpro", "1.0", DIGEST),
        )

    def test_a_missing_version_is_refused(self) -> None:
        with self.assertRaises(harbor.AdapterError):
            harbor.parse_dataset_pin("swebenchpro")

    def test_a_truncated_digest_is_refused(self) -> None:
        with self.assertRaises(harbor.AdapterError):
            harbor.parse_dataset_pin("swebenchpro@1.0+subset:sha256:abc")

    def test_an_unhashed_subset_marker_is_refused(self) -> None:
        with self.assertRaises(harbor.AdapterError):
            harbor.parse_dataset_pin("swebenchpro@1.0+subset:300-tasks")

    def test_an_empty_pin_is_refused(self) -> None:
        with self.assertRaises(harbor.AdapterError):
            harbor.parse_dataset_pin("")


class CategoryTests(unittest.TestCase):
    """Categories ride on the key so a per-repository suite needs no second pass."""

    def result(self):
        return {"trial_results": [
            {"task_name": "alpha", "verifier_result": {"rewards": {"reward": 1.0}}},
            {"task_name": "beta", "verifier_result": {"rewards": {"reward": 0.0}}},
        ]}

    def test_the_key_supplies_the_category(self) -> None:
        key = {"alpha": {"category": "ansible__ansible"}, "beta": {"category": "nodebb__nodebb"}}
        rows = harbor.translate(self.result(), key, "suite", 0)
        self.assertEqual([row["category"] for row in rows],
                         ["ansible__ansible", "nodebb__nodebb"])

    def test_a_key_without_categories_falls_back(self) -> None:
        key = {"alpha": {"task_name": "alpha"}, "beta": {"task_name": "beta"}}
        rows = harbor.translate(self.result(), key, "suite", 0)
        self.assertEqual({row["category"] for row in rows}, {"terminal"})


if __name__ == "__main__":
    unittest.main()
