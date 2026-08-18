"""The loop that corrects the reservation from what the server reports.

The reservation is a median, so it is wrong on the tail by construction. The
server publishes the one counter that says when that matters -- preemption --
and the sampler was already scraping it for the record. Here it becomes an
input instead.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "scripts"))

_SPEC = importlib.util.spec_from_file_location(
    "admission_broker", ROOT / "eval" / "scripts" / "admission_broker.py"
)
broker_main = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(broker_main)


def sample(preemptions: float, waiting: float) -> dict[str, float]:
    return {
        "vllm:num_preemptions_total": preemptions,
        "vllm:num_requests_waiting": waiting,
    }


class ControllerStepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.broker = broker_main.admission.serve(
            Path(self.tmp.name) / "sock", capacity=2_000_000
        )
        self.addCleanup(self.broker.stop)

    def test_preemptions_are_read_as_a_delta_not_a_level(self):
        """The counter is monotonic. Reading it as a level would back off
        forever after the first preemption of the job."""
        seen = broker_main.controller_step(
            self.broker, sample(50, 0), previous_preemptions=50, ceiling=2_000_000
        )
        self.assertEqual(self.broker.budget.capacity, 2_000_000, "an old count backed us off")
        self.assertEqual(seen, 50)

    def test_new_preemptions_shrink_the_budget(self):
        broker_main.controller_step(
            self.broker, sample(57, 0), previous_preemptions=50, ceiling=2_000_000
        )
        self.assertLess(self.broker.budget.capacity, 2_000_000)

    def test_a_queue_with_a_calm_cache_grows_the_budget_back(self):
        self.broker.resize(1_000_000)
        broker_main.controller_step(
            self.broker, sample(50, 40), previous_preemptions=50, ceiling=2_000_000
        )
        self.assertGreater(self.broker.budget.capacity, 1_000_000)

    def test_a_missing_sample_changes_nothing(self):
        """A server that is restarting scrapes empty. Treating that as calm
        would grow the budget into a cache that is not there yet."""
        before = self.broker.budget.capacity
        seen = broker_main.controller_step(
            self.broker, {}, previous_preemptions=50, ceiling=2_000_000
        )
        self.assertEqual(self.broker.budget.capacity, before)
        self.assertEqual(seen, 50, "an empty scrape must not reset the preemption baseline")


class CeilingTests(unittest.TestCase):
    def test_ceiling_is_a_fraction_of_the_pool_the_server_reported(self):
        """vLLM prints its own pool at startup. Sizing from that rather than
        from VRAM-in-gigabytes is the whole point: max_num_seqs was set to the
        number of GiB on the card, which is a unit coincidence."""
        self.assertEqual(broker_main.ceiling_from_pool(3_700_000), 2_960_000)

    def test_the_pool_is_read_from_the_server_log(self):
        line = "INFO 08-17 22:23 [kv_cache_utils.py:1229] GPU KV cache size: 3,708,453 tokens"
        self.assertEqual(broker_main.pool_from_log(line), 3_708_453)

    def test_a_log_without_the_line_yields_nothing_rather_than_a_guess(self):
        self.assertIsNone(broker_main.pool_from_log("nothing to see here"))


if __name__ == "__main__":
    unittest.main()
