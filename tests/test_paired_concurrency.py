"""How wide each lane offers, and the two ways that has been wrong.

Dividing the configured concurrency by the other lanes in the job starved a
server that schedules its own queue: 32 requests running against an empty
queue at 21% cache, with the share fixed at lane launch so it could not
recover as siblings finished. Removing that *and* the GPU scaling replaced
starvation with thrash -- a suite's eight-GPU width on two GPUs left ~90
sequences in permanent evict-and-recompute, 124 preemptions a minute against
12.5 completions.

What survives both: proportional to the hardware, with a margin to keep the
queue fed, and never a function of how many other lanes exist.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SBATCH = ROOT / "eval" / "slurm" / "paired-suite-eval.sbatch"


class ConcurrencyScaleTests(unittest.TestCase):
    """The offered width has been wrong in both directions; keep it derived.

    Dividing by the lanes in the job starved the server -- 32 requests running
    against an empty queue at 21% cache. Removing that *and* the GPU scaling
    replaced it with thrash: a suite's eight-GPU width on two GPUs held ~90
    sequences in permanent evict-and-recompute. The surviving rule is that the
    default is proportional to the hardware, never a literal and never a
    function of how many other lanes happen to exist.
    """

    def setUp(self) -> None:
        self.source = SBATCH.read_text(encoding="utf-8")
        line = next(
            l for l in self.source.splitlines()
            if l.startswith("CONCURRENCY_SCALE=")
        )
        self.default = line

    def test_the_client_does_not_ration_the_server(self) -> None:
        """No default scale: a lane offers its configured width, undivided.

        Any client-side number is blind to request shape, so one value either
        starves the short suites or thrashes the long ones.
        """
        self.assertEqual(
            self.default, 'CONCURRENCY_SCALE="${PAIRED_CONCURRENCY_SCALE:-}"',
            "the client is rationing again; admission belongs to the scheduler",
        )

    def test_the_running_batch_is_capped_server_side(self) -> None:
        """--max-num-seqs is the lever; unset, vLLM admits then evicts."""
        self.assertIn('--max-num-seqs "$MAX_NUM_SEQS"', self.source)
        self.assertIn('MAX_NUM_SEQS="${PAIRED_MAX_NUM_SEQS:-$MAX_NUM_SEQS_DEFAULT}"', self.source)

    def test_the_cap_is_sized_from_vram_not_from_the_card_count(self) -> None:
        """A card is not a unit of cache.

        The partitions this runs on differ by more than 4x per card, so a cap
        keyed on how many GPUs were granted sets the same batch for an
        allocation with four times the pool.
        """
        self.assertIn("memory.total", self.source)
        self.assertNotIn("128 * TP * DP", self.source)
        self.assertNotRegex(
            self.source, r"MAX_NUM_SEQS_DEFAULT=\$\(\( *[0-9]+ *\* *TP",
            "the cap is counting cards again",
        )

    def test_a_missing_gpu_query_does_not_invent_a_cap(self) -> None:
        """Sizing against an allocation that cannot be read is guessing."""
        self.assertIn("MAX_NUM_SEQS_DEFAULT=1024", self.source)

    def test_nothing_divides_by_the_lane_count_again(self) -> None:
        for gone in ("class_mates", "kv_class_of", "$CONCURRENCY_SCALE/$mates"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, self.source)

    def test_the_cap_is_visible_in_the_job_log(self) -> None:
        """Which side of starve-or-thrash a run was on has to be recoverable."""
        self.assertIn('echo "max-num-seqs=$MAX_NUM_SEQS', self.source)
        self.assertIn("vram", self.source, "the log has to say what it sized against")


if __name__ == "__main__":
    unittest.main()
