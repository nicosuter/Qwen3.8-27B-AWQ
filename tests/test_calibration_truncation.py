"""Calibration must not hand the quantizer a think block that never closes.

AWQ fits per-channel scales to the activations it observes. Truncating a
calibration sample at MAX_SEQ_LENGTH cuts long reasoning traces mid-block, so
the quantizer sees the states of reasoning-in-progress without ever seeing the
transition that ends it. Measured on the shipped calibration set, 73.9% of the
in-think token mass inside the window belongs to a block that never terminates:
for every token of reasoning that concludes, 2.8 tokens of reasoning that does
not.

An empty block is not the same fault and needs no repair. `<think></think>`
carries both tags adjacently, so the pair is represented in proportion and the
statistics for opening and closing move together. A cut block is asymmetric --
it inflates the opening and the interior while contributing nothing to the
close -- which is the shape that biases a stop probability.

Trimming back to the last block that closes costs no memory and leaves the
surviving text untouched. It is the one repair that does not trade the problem
for its mirror image: tail-aligning the window instead would guarantee a
closing tag and cut the opening, inflating `</think>` without `<think>`.
"""

import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "calibration_trim", ROOT / "quant" / "scripts" / "calibration_trim.py"
)
trim_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trim_module)
trim_to_closed_think = trim_module.trim_to_closed_think

OPEN, CLOSE = 248068, 248069


class TrimToClosedThinkTests(unittest.TestCase):
    """`<think>` and `</think>` are single tokens, so this is exact."""

    def trim(self, ids):
        return list(trim_to_closed_think(ids, open_id=OPEN, close_id=CLOSE))

    def test_a_sequence_with_no_think_block_is_untouched(self) -> None:
        ids = [1, 2, 3, 4, 5]
        self.assertEqual(self.trim(ids), ids)

    def test_a_closed_block_is_untouched(self) -> None:
        ids = [1, OPEN, 7, 8, CLOSE, 9]
        self.assertEqual(self.trim(ids), ids)

    def test_an_empty_block_is_untouched(self) -> None:
        """Both tags present and adjacent: represented in proportion, not a fault."""
        ids = [1, OPEN, CLOSE, 2]
        self.assertEqual(self.trim(ids), ids)

    def test_a_dangling_block_is_cut_back_to_before_its_opening(self) -> None:
        """The reasoning that never concludes leaves, and takes its tag with it."""
        ids = [1, 2, OPEN, 7, 8, 9]
        self.assertEqual(self.trim(ids), [1, 2])

    def test_earlier_closed_blocks_survive_a_later_dangling_one(self) -> None:
        ids = [1, OPEN, 5, CLOSE, 2, OPEN, 7, 8]
        self.assertEqual(self.trim(ids), [1, OPEN, 5, CLOSE, 2])

    def test_the_opening_tag_alone_at_the_tail_is_removed(self) -> None:
        ids = [1, 2, 3, OPEN]
        self.assertEqual(self.trim(ids), [1, 2, 3])

    def test_a_sequence_that_is_only_a_dangling_block_becomes_empty(self) -> None:
        """The caller decides what to do with a row that trims away to nothing."""
        self.assertEqual(self.trim([OPEN, 4, 5]), [])

    def test_trimming_never_lengthens_and_never_reorders(self) -> None:
        ids = [3, OPEN, 4, CLOSE, 5, OPEN, 6, 7, 8]
        out = self.trim(ids)
        self.assertLessEqual(len(out), len(ids))
        self.assertEqual(out, ids[: len(out)], "trim must be a prefix, not a rewrite")

    def test_it_accepts_any_integer_sequence(self) -> None:
        """The collator holds ids as a 1-D tensor, so element types vary."""
        class Boxed:
            def __init__(self, v): self.v = v
            def __int__(self): return self.v
        out = trim_to_closed_think(
            [Boxed(1), Boxed(OPEN), Boxed(2)], open_id=OPEN, close_id=CLOSE
        )
        self.assertEqual(out, [1])


if __name__ == "__main__":
    unittest.main()


class CalibrationWindowTests(unittest.TestCase):
    """The window has to be wide enough that trimming is a net, not a censor.

    Trimming back to the last closed block removes the asymmetry, but it removes
    it from the deep end: at 4096 every reasoning block that survives is short,
    and the calibration set contains zero in-think tokens deeper than 4k. The
    deployment reasons far past that -- GPQA's p90 chain is around 23k tokens --
    so calibrating only on shallow reasoning trades one mismatch for another.

    Measured with the real tokenizer over the shipped sources, after trimming:

        window   tokens kept   in-think mass   deeper than 4k   deepest block
         16384     1,183,743         263,770          114,613          14,455
         32768     1,731,695         559,883          354,059          29,080
         65536     2,036,099         600,561          384,798          34,005

    32768 takes 92% of the deep coverage available at any window; 16384 takes
    only 30%. It is not memory-bound -- 17.7 GB of layer-input cache summed over
    every sample, beside a 55 GB BF16 replica on a 141 GB card -- so the cost is
    calibration-time compute, roughly 2.3x the forward passes of 4096.
    """

    MINIMUM = 32768

    def sbatch_window(self, name: str) -> int:
        text = (ROOT / "quant" / "slurm" / name).read_text()
        match = re.search(r"^export MAX_SEQ_LENGTH=(\d+)", text, re.M)
        self.assertIsNotNone(match, f"{name} does not set MAX_SEQ_LENGTH")
        return int(match.group(1))

    def test_the_calibration_window_reaches_deep_reasoning(self) -> None:
        for name in ("quantize.sbatch", "quantize-single.sbatch"):
            with self.subTest(sbatch=name):
                self.assertGreaterEqual(
                    self.sbatch_window(name),
                    self.MINIMUM,
                    "window too narrow: trimming would drop every deep reasoning state",
                )

    def test_both_entry_points_agree(self) -> None:
        """Two jobs quantizing the same recipe must not calibrate differently."""
        self.assertEqual(
            self.sbatch_window("quantize.sbatch"),
            self.sbatch_window("quantize-single.sbatch"),
        )
