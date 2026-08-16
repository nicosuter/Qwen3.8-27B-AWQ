import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("mmlu_pro", "scripts/adapters/mmlu_pro.py")

ROW = {
    "question_id": 70,
    "question": "Adverts must not encourage ____.",
    "options": ["Safe practices", "Unsafe practices", "Joy", "Trivial",
                "Wants", "Fear", "Jealousy", "Distress", "Nothing", "Everything"],
    "answer": "I",
    "answer_index": 8,
    "category": "business",
    "src": "ori_mmlu-business_ethics",
}
SHORT = {
    "question_id": 71, "question": "Pick one.", "options": ["a", "b", "c"],
    "answer": "B", "answer_index": 1, "category": "math", "src": "x",
}


def completion(content: str, *, reasoning: str = "", finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": 50},
    }


class MaterializeTests(unittest.TestCase):
    def test_options_are_lettered_in_published_order(self) -> None:
        prompts, key = adapter.materialize([ROW])
        text = prompts[0]["text"]
        self.assertIn("A. Safe practices", text)
        self.assertIn("I. Nothing", text)
        self.assertIn("J. Everything", text)
        self.assertEqual(key["mmlupro-70"]["answer"], "I")
        self.assertEqual(key["mmlupro-70"]["options"], 10)

    def test_the_prompt_does_not_leak_the_answer(self) -> None:
        prompts, _ = adapter.materialize([ROW])
        self.assertNotIn("Answer:", prompts[0]["text"])

    def test_fewer_than_ten_options_is_fine(self) -> None:
        prompts, key = adapter.materialize([SHORT])
        self.assertIn("C. c", prompts[0]["text"])
        self.assertNotIn("D.", prompts[0]["text"])
        self.assertEqual(key["mmlupro-71"]["options"], 3)

    def test_answer_disagreeing_with_answer_index_is_refused(self) -> None:
        # The letter indexes into the published order. If the two disagree, every
        # score computed from them is silently wrong.
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize([dict(ROW, answer_index=3)])

    def test_answer_outside_the_options_is_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize([dict(SHORT, answer="J", answer_index=9)])

    def test_too_many_options_is_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize([dict(ROW, options=ROW["options"] + ["k"])])


class ScoringTests(unittest.TestCase):
    def entry(self, row=ROW):
        return adapter.materialize([row])[1][f"mmlupro-{row['question_id']}"]

    def score(self, content, row=ROW, finish="stop"):
        return adapter.score_response(
            f"mmlupro-{row['question_id']}", completion(content, finish=finish),
            entry=self.entry(row), replicate=0, thinking=True,
        )

    def test_correct_letter(self) -> None:
        row = self.score("Thinking.\nAnswer: I")
        self.assertEqual(row["score"], 1.0)
        self.assertFalse(row["out_of_range_choice"])

    def test_wrong_letter(self) -> None:
        self.assertEqual(self.score("Answer: A")["score"], 0.0)

    def test_parenthesised_and_lowercase(self) -> None:
        self.assertEqual(self.score("answer: (i)")["score"], 1.0)

    def test_last_answer_wins(self) -> None:
        self.assertEqual(self.score("Answer: A\non reflection\nAnswer: I")["score"], 1.0)

    def test_lone_letter_on_the_final_line(self) -> None:
        self.assertEqual(self.score("The reasoning leads to\nI")["score"], 1.0)

    def test_a_letter_past_the_options_is_not_an_answer(self) -> None:
        # Ten letters exist, but this item has three options.
        row = self.score("Answer: J", row=SHORT)
        self.assertEqual(row["score"], 0.0)
        self.assertTrue(row["out_of_range_choice"])

    def test_no_answer_at_all(self) -> None:
        row = self.score("I am not sure.")
        self.assertTrue(row["empty_answer"])
        self.assertEqual(row["score"], 0.0)

    def test_truncated_reply_is_a_context_failure(self) -> None:
        self.assertTrue(self.score("Answer: I", finish="length")["context_failure"])


class PinTests(unittest.TestCase):
    def base(self, **over):
        pins = {"dataset": "a" * 40, "harness": adapter.HARNESS_ID,
                "verifier": adapter.VERIFIER_ID, "adapter": adapter.self_pin()}
        pins.update(over)
        return pins

    def test_resolved_commit_passes(self) -> None:
        adapter.validate_pins(self.base())

    def test_branch_is_not_a_pin(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(self.base(dataset="main"))

    def test_changed_adapter_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(self.base(adapter="sha256:" + "0" * 64))


if __name__ == "__main__":
    unittest.main()
