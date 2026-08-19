"""The loop that corrects the reservation from what the server reports.

The reservation is a median, so it is wrong on the tail by construction. The
server publishes the one counter that says when that matters -- preemption --
and the sampler was already scraping it for the record. Here it becomes an
input instead.
"""

import importlib.util
import sys
import tempfile
import threading
import time
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

    def test_a_queue_inside_the_broker_grows_the_budget(self):
        """The growth signal has to include the queue the budget is causing.

        A request blocked on capacity has never been sent, so vLLM's
        `num_requests_waiting` reads zero exactly when the budget is the
        bottleneck. Growing only on the server's queue makes the backoff
        one-way: one run sat at the 262k floor for three hours with the server
        running 18 requests and nothing queued anywhere it could see.
        """
        self.broker.resize(1_000_000)
        self.broker.budget.acquire("held", 1_000_000)
        for index in range(broker_main.admission.WAITING_TARGET + 1):
            threading.Thread(
                target=self.broker.budget.acquire, args=(f"lane{index}", 1000), daemon=True
            ).start()
        deadline = time.monotonic() + 3.0
        while self.broker.waiting() <= broker_main.admission.WAITING_TARGET:
            self.assertLess(time.monotonic(), deadline, "waiters never registered")
            time.sleep(0.02)
        broker_main.controller_step(
            self.broker, sample(50, 0), previous_preemptions=50, ceiling=2_000_000
        )
        self.assertGreater(self.broker.budget.capacity, 1_000_000)

    def test_a_budget_almost_entirely_held_still_backs_off(self):
        """Nearly full is the normal state under load, not a reason to stand
        down. The budget is still what admits the next request, so the cut
        reaches it."""
        self.broker.budget.acquire("held", 1_800_000)
        broker_main.controller_step(
            self.broker, sample(60, 0), previous_preemptions=50, ceiling=2_000_000
        )
        self.assertEqual(self.broker.budget.capacity, 1_600_000)

    def test_an_overdrawn_budget_is_left_where_it_is(self):
        """Held above capacity means admission is already shut. A further cut
        un-admits nobody and only has to be repaid out of completions."""
        self.broker.budget.acquire("held", 1_800_000)
        self.broker.resize(1_000_000)
        broker_main.controller_step(
            self.broker, sample(60, 0), previous_preemptions=50, ceiling=2_000_000
        )
        self.assertEqual(self.broker.budget.capacity, 1_000_000)

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

    def test_every_data_parallel_engine_counts_toward_the_pool(self):
        """Each engine allocates and reports its own cache. Reading the first
        line and stopping sized the budget at 40% of the endpoint rather than
        80%, which booked the budget solid while the cache sat at 35%."""
        log = (
            "GPU KV cache size: 3,540,097 tokens\n"
            "GPU KV cache size: 3,543,172 tokens\n"
        )
        self.assertEqual(broker_main.pool_from_log(log, engines=2), 3_540_097 + 3_543_172)

    def test_an_engine_that_has_not_logged_yet_is_scaled_not_dropped(self):
        """Engines are configured identically, so one line understates the pool
        by exactly the number that have not flushed. Under-sizing is the failure
        being fixed, so scale rather than take what is there."""
        log = "GPU KV cache size: 3,540,097 tokens\n"
        self.assertEqual(broker_main.pool_from_log(log, engines=2), 3_540_097 * 2)

    def test_a_restarted_server_does_not_double_count(self):
        """A log holding two startups has four lines for two engines; the budget
        is the current server's, not the sum of every one it has ever had."""
        log = "\n".join(f"GPU KV cache size: {n} tokens" for n in
                        ("3,000,000", "3,000,000", "3,540,097", "3,543,172"))
        self.assertEqual(broker_main.pool_from_log(log, engines=2), 3_540_097 + 3_543_172)

    def test_a_log_without_the_line_yields_nothing_rather_than_a_guess(self):
        self.assertIsNone(broker_main.pool_from_log("nothing to see here"))


if __name__ == "__main__":
    unittest.main()
