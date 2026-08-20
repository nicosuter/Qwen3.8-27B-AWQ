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


class ClosesInWindowTests(unittest.TestCase):
    """Reject a sample at build time rather than gut it at calibration time.

    The trim is a safety net, and a net that catches a whole sample is not
    doing anyone a favour: the worst row in the shipped set is 34,383 tokens
    whose single reasoning block never closes, so trimming leaves 148 tokens of
    prompt and the sample slot is spent on nothing. Worse, that row is the
    longest chain in the set -- exactly the deep coverage the wide window exists
    to capture -- so trimming silently selects against the samples that matter
    most.

    Rejecting it while building means the loop draws another candidate instead,
    and every accepted sample contributes in full.

    A bare trailing `<think>` is not that fault. The chat template ends a row
    with the generation prompt, so eight fineweb-edu rows close with `<think>\\n`
    and nothing after it -- two tokens the trim removes cleanly. Rejecting those
    would throw away good samples over a rendering artefact, so the predicate
    tolerates a dangling block with no substance in it.
    """

    def closes(self, ids, **kw):
        return trim_module.closes_in_window(
            ids, open_id=OPEN, close_id=CLOSE, **kw
        )

    def test_a_row_whose_reasoning_closes_is_accepted(self) -> None:
        self.assertTrue(self.closes([1, OPEN, 2, 3, CLOSE, 4]))

    def test_a_row_with_no_reasoning_is_accepted(self) -> None:
        self.assertTrue(self.closes([1, 2, 3]))

    def test_a_row_cut_mid_reasoning_is_rejected(self) -> None:
        self.assertFalse(self.closes([1, OPEN] + list(range(100, 200))))

    def test_a_trailing_generation_prompt_is_tolerated(self) -> None:
        """`...<|im_start|>assistant\\n<think>\\n` -- an artefact, not lost reasoning."""
        self.assertTrue(self.closes([1, 2, 3, OPEN, 9]))

    def test_the_tolerance_is_adjustable_and_bounded(self) -> None:
        self.assertFalse(self.closes([1, OPEN, 9, 9, 9], max_dangling=2))
        self.assertTrue(self.closes([1, OPEN, 9, 9, 9], max_dangling=8))

    def test_an_earlier_closed_block_does_not_excuse_a_cut_one(self) -> None:
        ids = [1, OPEN, 2, CLOSE, 3, OPEN] + list(range(100, 200))
        self.assertFalse(self.closes(ids))


class IteratorSafetyTests(unittest.TestCase):
    """Both helpers must not depend on `ids` being re-iterable.

    The collator passes a tensor row and the builder passes a list, but a
    generator is the natural thing for a caller to reach for, and a predicate
    that silently returns True on one would wave through every cut sample.
    """

    def test_closes_in_window_handles_a_one_shot_iterator(self) -> None:
        cut = [1, OPEN] + list(range(100, 200))
        self.assertFalse(trim_module.closes_in_window(
            iter(cut), open_id=OPEN, close_id=CLOSE))

    def test_trim_handles_a_one_shot_iterator(self) -> None:
        self.assertEqual(
            trim_module.trim_to_closed_think(iter([1, 2, OPEN, 3]), open_id=OPEN, close_id=CLOSE),
            [1, 2],
        )


class WindowConsistencyTests(unittest.TestCase):
    """Build and calibration must agree on the window, or the rule misfires.

    The builder rejects a candidate whose reasoning does not close inside
    MAX_SEQ_LENGTH, and the collator trims what is left at the same bound. Set
    them differently and both go wrong: build at 4096 and quantize at 32768 and
    the set was filtered against a window far tighter than the one used, so the
    long traces were discarded for nothing; build wide and quantize narrow and
    the filter passes rows the trim then guts.
    """

    def window_of(self, relative: str) -> int:
        text = (ROOT / relative).read_text()
        match = re.search(r"MAX_SEQ_LENGTH(?:=|:-|:=)\"?(\d+)", text)
        self.assertIsNotNone(match, f"{relative} does not pin MAX_SEQ_LENGTH")
        return int(match.group(1))

    def test_prepare_and_quantize_use_the_same_window(self) -> None:
        prepare = self.window_of("quant/scripts/prepare.sh")
        for name in ("quant/slurm/quantize.sbatch", "quant/slurm/quantize-single.sbatch"):
            with self.subTest(sbatch=name):
                self.assertEqual(
                    prepare,
                    self.window_of(name),
                    "the set would be filtered against a different window than it is used at",
                )

    def test_the_source_cache_version_moved_with_the_selection_rule(self) -> None:
        """A cached render predates the completeness filter and must not be reused.

        The cache key carries max_length, so widening the window already
        invalidates it -- but the filter changes which candidates are accepted at
        the same width, and nothing in the key describes that.
        """
        text = (ROOT / "quant" / "scripts" / "build_calibration.py").read_text()
        match = re.search(r"SOURCE_CACHE_VERSION = (\d+)", text)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(int(match.group(1)), 2)
