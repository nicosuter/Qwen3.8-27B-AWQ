"""Repinning is mechanical, so it should not be done by hand.

The adapter pin is a hash of the adapter and _common.py together. Any change to
the shared module restales every pin in every config at once, and a stale pin
stops the run -- correctly, but on a cluster, after a queue wait. Twice.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "scripts"))

_SPEC = importlib.util.spec_from_file_location(
    "repin_adapters", ROOT / "eval" / "scripts" / "repin_adapters.py"
)
repin = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(repin)

REAL_ADAPTER = "eval/scripts/adapters/gpqa_diamond.py"


def config_with(pin: str) -> dict:
    return {
        "suites": [
            {
                "name": "gpqa_diamond",
                "pins": {"dataset": "abc", "adapter": pin},
                "run": ["python", REAL_ADAPTER, "run"],
            }
        ]
    }


class RepinTests(unittest.TestCase):
    def write(self, tmp: str, pin: str) -> Path:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(config_with(pin), indent=2), encoding="utf-8")
        return path

    def test_a_stale_pin_is_rewritten_to_the_adapters_current_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "sha256:stale")
            changed = repin.repin([path], root=ROOT)
            written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(changed), 1)
        self.assertTrue(written["suites"][0]["pins"]["adapter"].startswith("sha256:"))
        self.assertNotEqual(written["suites"][0]["pins"]["adapter"], "sha256:stale")

    def test_repinning_an_already_current_config_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "sha256:stale")
            repin.repin([path], root=ROOT)
            before = path.read_text(encoding="utf-8")
            changed = repin.repin([path], root=ROOT)
            after = path.read_text(encoding="utf-8")
        self.assertEqual(changed, [])
        self.assertEqual(after, before)

    def test_other_pins_are_left_alone(self):
        """Only the adapter hash is derivable here. A dataset revision is a
        decision, and rewriting one silently would change what is measured."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "sha256:stale")
            repin.repin([path], root=ROOT)
            written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written["suites"][0]["pins"]["dataset"], "abc")

    def test_a_config_that_lists_suites_by_name_is_skipped_not_crashed_on(self):
        """batches.json says which suites a job runs, not how to run them."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batches.json"
            path.write_text(
                json.dumps({"suites": ["gpqa_diamond", "ruler"]}), encoding="utf-8"
            )
            self.assertEqual(repin.repin([path], root=ROOT), [])

    def test_check_mode_reports_staleness_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "sha256:stale")
            stale = repin.repin([path], root=ROOT, check=True)
            unchanged = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(stale), 1)
        self.assertEqual(unchanged["suites"][0]["pins"]["adapter"], "sha256:stale")


if __name__ == "__main__":
    unittest.main()
