"""The priors are measured from runs on disk, never hand-set.

An invented number here is an invented admission policy: it decides how many
requests of each suite are in flight, and nothing downstream reveals that it was
wrong except a preemption storm hours later.
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
    "build_token_priors", ROOT / "eval" / "scripts" / "build_token_priors.py"
)
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class BuildPriorsTests(unittest.TestCase):
    def test_prior_is_the_median_output_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_rows(
                run / "raw" / "baseline" / "gpqa_diamond-r0.jsonl",
                [
                    {"id": str(i), "prompt_tokens": 100, "output_tokens": n}
                    for i, n in enumerate([10, 20, 30, 40, 500])
                ],
            )
            priors = builder.collect([run])
        self.assertEqual(priors["suites"]["gpqa_diamond"]["output"], 30)

    def test_both_arms_contribute_because_the_candidate_may_reason_longer(self):
        """Sizing admission on the baseline alone under-reserves the arm under test."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_rows(
                run / "raw" / "baseline" / "ruler-r0.jsonl",
                [{"id": "a", "output_tokens": 100}, {"id": "b", "output_tokens": 100}],
            )
            write_rows(
                run / "raw" / "candidate" / "ruler-r0.jsonl",
                [{"id": "a", "output_tokens": 900}, {"id": "b", "output_tokens": 900}],
            )
            priors = builder.collect([run])
        self.assertEqual(priors["suites"]["ruler"]["output"], 500)

    def test_rows_that_never_produced_output_are_not_evidence(self):
        """A timed-out row records 0 output tokens; averaging it in would shrink
        the reservation because the cache was too full, which is backwards."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_rows(
                run / "raw" / "baseline" / "ruler-r0.jsonl",
                [
                    {"id": "a", "output_tokens": 800},
                    {"id": "b", "output_tokens": 0, "timeout": True},
                    {"id": "c", "output_tokens": 900},
                    {"id": "d", "output_tokens": 0, "timeout": True},
                ],
            )
            priors = builder.collect([run])
        self.assertEqual(priors["suites"]["ruler"]["output"], 850)

    def test_a_suite_with_no_usable_rows_is_left_out_rather_than_defaulted(self):
        """Absent is honest; a fabricated prior is not."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_rows(
                run / "raw" / "baseline" / "ruler-r0.jsonl",
                [{"id": "a", "output_tokens": 0, "timeout": True}],
            )
            priors = builder.collect([run])
        self.assertNotIn("ruler", priors["suites"])

    def test_the_prompt_half_is_recorded_too_because_pixels_are_not_characters(self):
        """A multimodal prompt is mostly image. Only the server counted it."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_rows(
                run / "raw" / "baseline" / "multimodal-r0.jsonl",
                [
                    {"id": "a", "prompt_tokens": 800, "output_tokens": 50},
                    {"id": "b", "prompt_tokens": 900, "output_tokens": 60},
                    {"id": "c", "prompt_tokens": 1000, "output_tokens": 70},
                ],
            )
            priors = builder.collect([run])
        self.assertEqual(priors["suites"]["multimodal"]["prompt"], 900)


if __name__ == "__main__":
    unittest.main()
