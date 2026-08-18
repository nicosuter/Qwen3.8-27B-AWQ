"""A client-side timeout is not model behavior, so it must not be scored as one.

The server returns when it hits the context limit -- `finish_reason` is
`length`, and RULER's own baseline has seven such rows. The only way to reach a
socket timeout is for the client to give up while the request was queued or
generating, which charges an item for contention it did not choose. Retrying is
the only reading that makes progress monotonic.
"""

import importlib.util
import io
import threading
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_ADMISSION_SPEC = importlib.util.spec_from_file_location(
    "_admission", ROOT / "eval" / "scripts" / "adapters" / "_admission.py"
)
admission = importlib.util.module_from_spec(_ADMISSION_SPEC)
_ADMISSION_SPEC.loader.exec_module(admission)

_SPEC = importlib.util.spec_from_file_location(
    "_common", ROOT / "eval" / "scripts" / "adapters" / "_common.py"
)
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)

PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class Client:
    """A stand-in server, scripted with what each attempt should do."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, base_url, api_key, payload, timeout):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def call(client, retries=2, sleeps=None):
    return common.request_with_retries(
        "item-1",
        PAYLOAD,
        base_url="http://server",
        api_key="EMPTY",
        timeout=5.0,
        retries=retries,
        client=client,
        sleep=(sleeps.append if sleeps is not None else (lambda _seconds: None)),
    )


class TimeoutRetryTests(unittest.TestCase):
    def test_a_timeout_that_clears_is_retried_and_its_answer_kept(self):
        client = Client([TimeoutError("timed out"), {"choices": [{"message": {}}]}])
        response, attempts = call(client)
        self.assertIsNotNone(response, "a cleared timeout must not score as a failure")
        self.assertEqual(attempts, 2)
        self.assertEqual(client.calls, 2)

    def test_a_timeout_that_never_clears_fails_the_run_rather_than_scoring_zero(self):
        """Silently returning a zero is how a contended server became a quality
        regression: 36 of RULER's 105 items, none of them the model's answer."""
        client = Client([TimeoutError("timed out")])
        with self.assertRaises(common.AdapterError) as caught:
            call(client, retries=2)
        self.assertIn("item-1", str(caught.exception))
        self.assertIn("timed out", str(caught.exception).lower())
        self.assertEqual(client.calls, 3, "should have used every attempt before giving up")

    def test_backoff_grows_between_timeout_attempts(self):
        """Retrying a timeout immediately would hammer a server that is merely busy."""
        sleeps: list[float] = []
        client = Client([TimeoutError("timed out"), TimeoutError("timed out"), {"ok": True}])
        call(client, retries=3, sleeps=sleeps)
        self.assertEqual(len(sleeps), 2)
        self.assertGreater(sleeps[1], sleeps[0])

    def test_a_rejected_request_still_aborts_without_retrying(self):
        """A 400 will say the same thing however many times it is asked."""
        client = Client([urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad"))])
        with self.assertRaises(common.AdapterError):
            call(client)
        self.assertEqual(client.calls, 1)

    def test_a_transport_fault_is_still_retried(self):
        client = Client([urllib.error.URLError("connection reset"), {"ok": True}])
        response, attempts = call(client)
        self.assertEqual(attempts, 2)
        self.assertIsNotNone(response)


class AdmissionHoldTests(unittest.TestCase):
    """Every request reserves its KV footprint for as long as it is in flight.

    Hooked in here rather than in each adapter so that the fourteen of them
    cannot drift apart on the one policy that has to be identical across arms.
    """

    # Reservation for the payload below: prompt floor 10, output capped at
    # max_tokens 100, so 110 tokens per request.
    PRIORS = {
        "suites": {"gpqa_diamond": {"prompt": 10, "output": 100}},
        "default": {"prompt": 10, "output": 100},
    }

    def configure(self, capacity):
        budget = admission.TokenBudget(capacity)
        common.configure_admission(budget=budget, priors=self.PRIORS, suite="gpqa_diamond")
        self.addCleanup(common.configure_admission, budget=None, priors=None, suite="")
        return budget

    def test_requests_that_do_not_fit_together_do_not_fly_together(self):
        budget = self.configure(capacity=150)
        in_flight = threading.Semaphore(0)
        may_finish = threading.Event()
        peak = []

        def client(base_url, api_key, payload, timeout):
            peak.append(budget.outstanding())
            in_flight.release()
            may_finish.wait(timeout=3.0)
            return {"ok": True}

        payload = dict(PAYLOAD, max_tokens=100)
        threads = [
            threading.Thread(
                target=lambda: common.request_with_retries(
                    f"item-{n}", payload, base_url="u", api_key="k",
                    timeout=5.0, retries=0, client=client,
                ),
                daemon=True,
            )
            for n in range(2)
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(in_flight.acquire(timeout=2.0), "no request was admitted")
        self.assertFalse(
            in_flight.acquire(timeout=0.3),
            "both requests were in flight against a budget that fits one",
        )
        may_finish.set()
        for thread in threads:
            thread.join(timeout=3.0)
        self.assertLessEqual(max(peak), budget.capacity)

    def test_a_failed_request_still_returns_its_reservation(self):
        """A leak here ratchets the budget to zero over a long suite and the
        run stalls with the server idle."""
        budget = self.configure(capacity=100_000)
        client = Client([urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad"))])
        with self.assertRaises(common.AdapterError):
            call(client)
        self.assertEqual(budget.outstanding(), 0)

    def test_a_retry_takes_its_reservation_out_again_rather_than_holding_it(self):
        """One hold is one attempt, so the worst case is one timeout long.

        Holding across the whole retry chain makes the worst case the request
        timeout times the attempt count -- three hours on these lanes -- and a
        server that stops answering turns that into a deduction from the budget
        that nothing gives back. The retry still reserves before it is sent, so
        a retry still cannot be what overfills the cache.
        """
        budget = self.configure(capacity=110)
        between: list[int] = []
        client = Client([TimeoutError("timed out"), {"ok": True}])
        common.request_with_retries(
            "item-1",
            dict(PAYLOAD, max_tokens=100),
            base_url="http://server",
            api_key="EMPTY",
            timeout=5.0,
            retries=2,
            client=client,
            sleep=lambda _seconds: between.append(budget.outstanding()),
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(between, [0], "the reservation was held across the backoff")
        self.assertEqual(budget.outstanding(), 0)

    def test_with_no_budget_configured_requests_are_unrestricted(self):
        common.configure_admission(budget=None, priors=None, suite="")
        response, _ = call(Client([{"ok": True}]))
        self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
