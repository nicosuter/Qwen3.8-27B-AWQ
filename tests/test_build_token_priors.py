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
    def test_prior_is_the_mean_because_the_budget_is_a_sum(self):
        """The budget holds N requests at once, so it holds N times the mean.

        The median answers "how long is this one request", which is not the
        question a shared budget asks. Every suite in this protocol is
        right-skewed -- measured over the runs on disk, mean/median is 1.68x on
        livecodebench_v6, 2.56x on mmmu_pro and 30.94x on ruler -- so a median
        reservation tells the budget it is holding a fraction of what it will
        actually hold, and the difference is charged to the cache as preemption.
        """
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
        self.assertEqual(priors["suites"]["gpqa_diamond"]["output"], 120)

    def test_a_suite_whose_tail_reaches_the_cap_reserves_far_above_its_median(self):
        """RULER's shape: most items short, a tenth of them at the token cap.

        Its median output is 653 tokens against a mean of 20,206. Reserving the
        median priced 96 concurrent lanes at a thirtieth of the cache they went
        on to occupy.
        """
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_rows(
                run / "raw" / "baseline" / "ruler-r0.jsonl",
                [{"id": str(i), "output_tokens": 600} for i in range(9)]
                + [{"id": "tail", "output_tokens": 131_072}],
            )
            priors = builder.collect([run])
        self.assertGreater(priors["suites"]["ruler"]["output"], 600)

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

    def test_the_prompt_half_stays_a_median_because_it_is_a_floor_not_a_summand(self):
        """Only the output is added to the reservation.

        The prompt prior is a floor under a per-item estimate the client already
        has, applied as `max(estimate, prior)`, and the max of a draw and a
        constant sits above both -- so raising the floor to the mean overshoots
        instead of centring. Measured over the runs on disk it over-reserved
        RULER and multimodal by a quarter, while the median floor lands both
        within a tenth of what they actually occupied.
        """
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_rows(
                run / "raw" / "baseline" / "ruler-r0.jsonl",
                [{"id": str(i), "prompt_tokens": 100, "output_tokens": 10} for i in range(9)]
                + [{"id": "tail", "prompt_tokens": 10_000, "output_tokens": 10}],
            )
            priors = builder.collect([run])
        self.assertEqual(priors["suites"]["ruler"]["prompt"], 100)
        self.assertEqual(priors["suites"]["ruler"]["output"], 10)

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


class RulerPriorFloorTests(unittest.TestCase):
    """RULER's output prior must survive the suite it now describes.

    Admission reserves `prompt + expected output`. The prompt half corrects
    itself from the text, but the output half is taken from this file as-is, so
    a stale value under-reserves every request in flight -- the failure that
    left the cache holding 2.27x what the budget believed it had booked.

    `--max-tokens 0` is what moved it. RULER keeps all twenty categories --
    narrowing to the five that separated a checkpoint was rejected, because
    selecting on the outcome moves ruler from 96.67% to 80.00% -- so the mean is
    not the problem. The tail is. As the lane drains, every request still in
    flight is cwe or fwe, so a prior sized on the mean falls short on all of
    them at once, which is the one shape the preemption controller cannot claw
    back.

    So the floor is not a measured mean at all; it is what the window allows.
    Uncapped, a request can occupy prompt + (window - prompt) = the whole
    window, and a 32k item reserving on the 66,210 prompt floor therefore needs
    an output prior of at least window - 66,210 to book what it can hold. Below
    that the budget admits 34 long requests where 21 fit. An 80,000 floor, which
    is what this test used to assert, passes a prior of 100,000 that
    over-commits by 1.57x.
    """

    WINDOW = 262_144
    PROMPT_FLOOR = 66_210
    MEASURED_FLOOR = WINDOW - PROMPT_FLOOR

    def test_the_ruler_output_prior_covers_the_narrowed_suite(self) -> None:
        priors = json.loads(
            (Path(__file__).resolve().parent.parent / "eval" / "token-priors.json").read_text()
        )
        output = priors["suites"]["ruler"]["output"]
        self.assertGreaterEqual(
            output,
            self.MEASURED_FLOOR,
            "ruler's output prior is below what its surviving categories generate; "
            "admission would under-reserve every request in flight",
        )
