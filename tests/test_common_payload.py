"""What the request says about the token budget, which is nothing by default.

A cap does not persuade the model to answer sooner. Nothing carries the budget
to it -- not the API, not the prompt -- so it cannot spend against one, and when
it runs long it runs into the cap mid-sentence. Under 131072, nine of the ten
truncated MathArena items had put all 131072 tokens into reasoning and emitted
no answer at all. The cap bought no answer rather than a shorter one, and it
did so about twice as often for the arm that reasons longer, which is the arm
under test.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "adapters"))

_SPEC = importlib.util.spec_from_file_location(
    "_common", ROOT / "scripts" / "adapters" / "_common.py"
)
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)

GENERATION = {
    "enable_thinking": True,
    "reasoning_effort": "xhigh",
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def payload(max_tokens: int) -> dict:
    return common.build_payload(
        "2 + 2 = ?",
        GENERATION,
        model="qwen38-eval",
        seed=38027,
        max_tokens=max_tokens,
        instruction="Answer with a number.",
    )


class TokenBudgetTests(unittest.TestCase):
    def test_a_positive_cap_is_sent(self) -> None:
        self.assertEqual(payload(4096)["max_tokens"], 4096)

    def test_zero_means_the_rest_of_the_context_window(self) -> None:
        """Absent, not zero: the server reads an absent field as 'all of it'.

        Sending max_tokens=0 would be a request for no output at all.
        """
        self.assertNotIn("max_tokens", payload(0))

    def test_a_negative_cap_is_treated_the_same_way(self) -> None:
        self.assertNotIn("max_tokens", payload(-1))

    def test_nothing_else_about_the_request_changes(self) -> None:
        """Whether the cap is sent must not disturb what is sampled.

        This is what lets an item that finished under a cap be carried across a
        change to it: max_tokens bounds the length, it does not steer the draw.
        """
        capped = payload(131072)
        uncapped = payload(0)
        capped.pop("max_tokens")
        self.assertEqual(capped, uncapped)
        self.assertEqual(capped["seed"], uncapped["seed"])


if __name__ == "__main__":
    unittest.main()
