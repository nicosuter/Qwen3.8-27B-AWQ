"""Admission control by KV footprint rather than by request count.

The server admits on a request's current size, not its eventual one: a GPQA item
enters at 275 prompt tokens and grows to 52k, so 280 of them are trivially
admissible and then collectively will not fit. Only the client knows the
eventual size, from the previous run's distribution, so the reservation is made
here.
"""

import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "_admission", ROOT / "eval" / "scripts" / "adapters" / "_admission.py"
)
admission = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(admission)


class LocalBudgetTests(unittest.TestCase):
    def test_second_holder_waits_until_the_first_releases(self):
        """Two reservations that do not fit together are not held together."""
        budget = admission.TokenBudget(1000)
        budget.acquire("a", 600)
        released = threading.Event()

        def second() -> None:
            budget.acquire("b", 600)
            released.set()

        thread = threading.Thread(target=second, daemon=True)
        thread.start()
        self.assertFalse(released.wait(timeout=0.2), "b was admitted while a still held 600")
        budget.release("a")
        self.assertTrue(released.wait(timeout=2.0), "b was not admitted after a released")
        thread.join(timeout=2.0)

    def test_a_reservation_larger_than_the_budget_still_runs(self):
        """Alone, but it runs.

        A 131k RULER item against a budget the controller has backed off to
        100k would otherwise wait for capacity that can never appear. The suite
        would hang rather than score, which is worse than the contention the
        backoff was avoiding.
        """
        budget = admission.TokenBudget(1000)
        granted = budget.acquire("huge", 5000)
        self.assertEqual(granted, 1000, "an oversized reservation should clamp to the budget")
        self.assertEqual(budget.outstanding(), 1000)

    def test_an_oversized_holder_still_excludes_everyone_else(self):
        """Clamping must not become a free pass that lets the cache overfill."""
        budget = admission.TokenBudget(1000)
        budget.acquire("huge", 5000)
        admitted = threading.Event()

        def other() -> None:
            budget.acquire("small", 10)
            admitted.set()

        thread = threading.Thread(target=other, daemon=True)
        thread.start()
        self.assertFalse(admitted.wait(timeout=0.2), "another request ran beside the oversized one")
        budget.release("huge")
        self.assertTrue(admitted.wait(timeout=2.0))
        thread.join(timeout=2.0)


class ControllerTests(unittest.TestCase):
    """The estimate is a p50, so it is wrong on the tail. Preemption says so.

    Reserving p90 instead would run the pool at a sixth of its capacity, because
    GPQA's output length spans 8k at p50 to 52k at p90. Reserve the middle and
    let the measured preemption counter correct it.
    """

    def test_preemption_shrinks_the_budget(self):
        after = admission.next_capacity(1_000_000, preempted=7, waiting=0, ceiling=3_000_000)
        self.assertLess(after, 1_000_000)
        self.assertEqual(after, 800_000)

    def test_a_standing_queue_without_preemption_grows_the_budget(self):
        """Requests waiting while the cache copes means the reservation is too fat."""
        after = admission.next_capacity(1_000_000, preempted=0, waiting=32, ceiling=3_000_000)
        self.assertGreater(after, 1_000_000)

    def test_an_empty_queue_holds_the_budget_steady(self):
        """Nothing is waiting, so a larger budget would admit nothing new."""
        after = admission.next_capacity(1_000_000, preempted=0, waiting=0, ceiling=3_000_000)
        self.assertEqual(after, 1_000_000)

    def test_growth_stops_at_the_ceiling(self):
        after = admission.next_capacity(2_999_000, preempted=0, waiting=99, ceiling=3_000_000)
        self.assertEqual(after, 3_000_000)

    def test_backoff_stops_at_a_floor_that_still_admits_the_longest_item(self):
        """Backing off below one max-length request would serialize the suite."""
        after = 4_000_000
        for _ in range(50):
            after = admission.next_capacity(after, preempted=1, waiting=0, ceiling=4_000_000)
        self.assertGreaterEqual(after, admission.FLOOR_TOKENS)

    def test_shrinking_below_what_is_held_admits_nobody_until_it_drains(self):
        """A resize is not a revocation: holders keep what they were granted."""
        budget = admission.TokenBudget(1000)
        budget.acquire("a", 800)
        budget.resize(500)
        admitted = threading.Event()

        def other() -> None:
            budget.acquire("b", 100)
            admitted.set()

        thread = threading.Thread(target=other, daemon=True)
        thread.start()
        self.assertFalse(admitted.wait(timeout=0.2), "admitted past a budget already overdrawn")
        budget.release("a")
        self.assertTrue(admitted.wait(timeout=2.0))
        thread.join(timeout=2.0)


class ReservationTests(unittest.TestCase):
    """What one request costs the cache: its prompt, plus what it will generate.

    The prompt half is known exactly -- the adapter just built it. Only the
    output half is a guess, and the suites where the guess is worst are the
    small ones: RULER is 130k of prompt against 621 tokens of output, so its
    estimate is nearly exact, while GPQA's is almost entirely guess.
    """

    PRIORS = {
        "suites": {
            "ruler": {"prompt": 32597, "output": 621},
            "gpqa_diamond": {"prompt": 275, "output": 8081},
            "multimodal": {"prompt": 869, "output": 89},
        },
        "default": {"prompt": 1024, "output": 4096},
    }

    def test_reservation_is_the_prompt_plus_the_suites_expected_output(self):
        text = "x" * (400 * admission.CHARS_PER_TOKEN)
        got = admission.reservation(text, "gpqa_diamond", self.PRIORS, max_tokens=131072)
        self.assertEqual(got, 400 + 8081)

    def test_an_unrecorded_suite_falls_back_to_the_default_rather_than_to_zero(self):
        """A zero prior would reserve the prompt alone and admit without bound."""
        text = "x" * (2000 * admission.CHARS_PER_TOKEN)
        got = admission.reservation(text, "suite_that_has_never_run", self.PRIORS, max_tokens=0)
        self.assertEqual(got, 2000 + 4096)

    def test_the_reservation_cannot_exceed_what_the_cap_allows(self):
        """A prior larger than --max-tokens describes a request that cannot happen."""
        text = "x" * (100 * admission.CHARS_PER_TOKEN)
        got = admission.reservation(text, "gpqa_diamond", self.PRIORS, max_tokens=500)
        self.assertEqual(got, 275 + 500)

    def test_an_image_prompt_reserves_more_than_its_text_length(self):
        """The text of a multimodal item is a caption; the image is the prompt.

        Estimating from characters alone would reserve a few dozen tokens for an
        item whose prompt reaches 4078 at p90. At the hundreds-in-flight the
        cheap suites are meant to run at, that shortfall is the whole pool.
        """
        text = "What is shown?"
        got = admission.reservation(text, "multimodal", self.PRIORS, max_tokens=131072)
        self.assertEqual(got, 869 + 89, "should fall back to the measured prompt length")

    def test_a_long_prompt_beats_the_suite_median(self):
        """RULER's items are 4k, 32k and 128k; a median would misprice two of them."""
        text = "x" * (120_000 * admission.CHARS_PER_TOKEN)
        got = admission.reservation(text, "ruler", self.PRIORS, max_tokens=131072)
        self.assertEqual(got, 120_000 + 621)

    def test_priors_shipped_in_the_repo_cover_every_scored_suite(self):
        """A suite with no prior silently reserves the default, which is 20x wrong
        for RULER. Catch that here rather than in a run."""
        priors = admission.load_priors(ROOT / "eval" / "token-priors.json")
        suite_names = {
            entry["name"]
            for entry in __import__("json").loads(
                (ROOT / "eval" / "eval-suite-v1.json").read_text(encoding="utf-8")
            )["suites"]
        }
        self.assertTrue(suite_names, "no suites found in eval-suite-v1.json")
        missing = sorted(suite_names - set(priors["suites"]))
        self.assertEqual(missing, [], f"no output-length prior for {missing}")


class BrokerTests(unittest.TestCase):
    """One budget per server, shared by every lane process on it.

    A per-lane share cannot be right: it is fixed when the lane starts, so it
    cannot recover as siblings finish. A six-lane run measured 32 requests
    against an empty queue with two lanes pinned at their share while the
    server idled.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.socket_path = Path(self.tmp.name) / "admission.sock"

    def test_two_lanes_sharing_a_broker_are_mutually_excluded(self):
        broker = admission.serve(self.socket_path, capacity=1000)
        self.addCleanup(broker.stop)
        lane_a = admission.RemoteBudget(self.socket_path)
        lane_b = admission.RemoteBudget(self.socket_path)
        lane_a.acquire("a", 700)
        admitted = threading.Event()

        def second() -> None:
            lane_b.acquire("b", 700)
            admitted.set()

        thread = threading.Thread(target=second, daemon=True)
        thread.start()
        self.assertFalse(admitted.wait(timeout=0.3), "both lanes held 1400 of a 1000 budget")
        lane_a.release("a")
        self.assertTrue(admitted.wait(timeout=3.0), "b never got the capacity a released")
        thread.join(timeout=3.0)

    def test_a_lane_that_dies_does_not_strand_its_reservation(self):
        """A lane is a process. If it is killed mid-request its tokens must come
        back, or the budget ratchets down to nothing over a long job."""
        broker = admission.serve(self.socket_path, capacity=1000)
        self.addCleanup(broker.stop)
        casualty = admission.RemoteBudget(self.socket_path)
        casualty.acquire("doomed", 900)
        self.assertEqual(broker.outstanding(), 900)

        casualty.close()

        survivor = admission.RemoteBudget(self.socket_path)
        deadline = time.monotonic() + 3.0
        while broker.outstanding() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(broker.outstanding(), 0, "the dead lane's tokens were never returned")
        survivor.acquire("next", 900)
        survivor.close()


class SocketPathTests(unittest.TestCase):
    """A unix socket path is capped near 104 bytes by the sockaddr struct.

    The natural place for it is inside the run directory, and those are already
    close: `.../v2/eval-suite-v2-inproj-int8/admission-candidate.sock` is 88
    characters under the current RUN_BASE. A slightly longer campaign name would
    have failed at bind with an OSError that says nothing about paths.
    """

    def test_a_short_path_is_left_alone(self):
        self.assertEqual(admission.socket_path("/tmp/adm.sock"), Path("/tmp/adm.sock"))

    def test_an_overlong_path_is_replaced_deterministically(self):
        long_path = "/" + "d" * 200 + "/admission-candidate.sock"
        once = admission.socket_path(long_path)
        twice = admission.socket_path(long_path)
        self.assertEqual(once, twice, "server and client must agree without talking")
        self.assertLess(len(str(once)), 100)

    def test_different_run_directories_do_not_collide(self):
        first = admission.socket_path("/" + "a" * 200 + "/admission-candidate.sock")
        second = admission.socket_path("/" + "b" * 200 + "/admission-candidate.sock")
        self.assertNotEqual(first, second)

    def test_the_two_arms_of_one_run_do_not_collide(self):
        """Baseline and candidate get their own broker; sharing one would carry
        a budget sized for one checkpoint's cache over to the other."""
        base = "/" + "d" * 200 + "/admission-baseline.sock"
        cand = "/" + "d" * 200 + "/admission-candidate.sock"
        self.assertNotEqual(admission.socket_path(base), admission.socket_path(cand))

    def test_a_broker_on_an_overlong_path_is_reachable_by_the_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / ("x" * 120) / "admission.sock"
            broker = admission.serve(requested, capacity=1000)
            self.addCleanup(broker.stop)
            client = admission.RemoteBudget(requested)
            self.addCleanup(client.close)
            self.assertEqual(client.acquire("item", 100), 100)


class EnvironmentTests(unittest.TestCase):
    def test_no_configuration_means_no_admission_control(self):
        """Adapters run standalone against a dev server with nothing to share."""
        self.assertIsNone(admission.from_environment({}))

    def test_a_capacity_alone_gives_a_process_local_budget(self):
        budget = admission.from_environment({"EVAL_ADMISSION_TOKENS": "2048"})
        self.assertIsInstance(budget, admission.TokenBudget)
        self.assertEqual(budget.capacity, 2048)


if __name__ == "__main__":
    unittest.main()
