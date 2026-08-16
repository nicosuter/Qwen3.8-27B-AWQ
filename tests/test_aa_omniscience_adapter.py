import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("aa_omniscience", "scripts/adapters/aa_omniscience.py")

ROWS = [
    {"domain": "Finance", "topic": "Accounting", "question_id": "1",
     "question": "Which reference lists the two criteria?", "answer": "ASC 606-10-25-15"},
    {"domain": "Law", "topic": "Contract", "question_id": "2",
     "question": "Which article governs?", "answer": "Article 2"},
]


def completion(content: str, *, reasoning: str = "", finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": 40},
    }


def key_for(rows=None) -> dict:
    return adapter.materialize(rows or ROWS)[1]


def score(content: str, item="omni-1", rows=None):
    key = key_for(rows)
    return adapter.score_response(
        item, completion(content), entry=key[item], replicate=0, thinking=True
    )


class MaterializeTests(unittest.TestCase):
    def test_prompts_and_key(self) -> None:
        prompts, key = adapter.materialize(ROWS)
        self.assertEqual([p["id"] for p in prompts], ["omni-1", "omni-2"])
        self.assertEqual(key["omni-1"]["answer"], "ASC 606-10-25-15")
        self.assertEqual(key["omni-1"]["domain"], "Finance")
        # The prompt must not leak the answer.
        self.assertNotIn("606-10-25-15", prompts[0]["text"])

    def test_a_row_without_an_answer_is_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize([dict(ROWS[0], answer="")])

    def test_duplicate_ids_are_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize([ROWS[0], dict(ROWS[0])])


class ScoringTests(unittest.TestCase):
    def test_exact_answer_scores(self) -> None:
        row = score("Reasoning here.\nAnswer: ASC 606-10-25-15")
        self.assertEqual(row["score"], 1.0)
        self.assertFalse(row["abstained"])
        self.assertFalse(row["hallucinated"])

    def test_case_and_spacing_are_not_the_question(self) -> None:
        row = score("Answer:   asc 606-10-25-15  ")
        self.assertEqual(row["score"], 1.0)

    def test_a_trailing_period_is_tolerated(self) -> None:
        self.assertEqual(score("Answer: Article 2.", item="omni-2")["score"], 1.0)

    def test_a_confident_wrong_answer_is_a_hallucination(self) -> None:
        row = score("Answer: ASC 606-10-25-99")
        self.assertEqual(row["score"], 0.0)
        self.assertFalse(row["abstained"])
        self.assertTrue(row["hallucinated"])

    def test_abstention_is_neither_correct_nor_hallucinated(self) -> None:
        # The distinction the published index turns on: not knowing and saying
        # so must not be scored the same as answering wrongly.
        row = score("Answer: I don't know")
        self.assertEqual(row["score"], 0.0)
        self.assertTrue(row["abstained"])
        self.assertFalse(row["hallucinated"])

    def test_abstention_phrasing_variants(self) -> None:
        for text in ("Answer: I do not know", "Answer: i don't know.", "Answer: unknown"):
            with self.subTest(text=text):
                self.assertTrue(score(text)["abstained"], text)

    def test_the_last_answer_line_wins(self) -> None:
        row = score("Answer: wrong first\nmore thinking\nAnswer: ASC 606-10-25-15")
        self.assertEqual(row["score"], 1.0)

    def test_a_reply_with_no_answer_line_falls_back_to_the_last_line(self) -> None:
        row = score("I think it is\nASC 606-10-25-15")
        self.assertEqual(row["score"], 1.0)

    def test_an_empty_reply_is_flagged_not_hallucinated(self) -> None:
        row = score("")
        self.assertTrue(row["empty_answer"])
        self.assertFalse(row["hallucinated"])
        self.assertEqual(row["score"], 0.0)

    def test_a_truncated_reply_is_a_context_failure(self) -> None:
        key = key_for()
        row = adapter.score_response(
            "omni-1", completion("Answer: ASC", finish="length"),
            entry=key["omni-1"], replicate=0, thinking=True,
        )
        self.assertTrue(row["context_failure"])


class PinTests(unittest.TestCase):
    def test_a_branch_is_not_a_pin(self) -> None:
        pins = {"dataset": "main", "harness": adapter.HARNESS_ID,
                "verifier": adapter.VERIFIER_ID, "adapter": adapter.self_pin()}
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_a_resolved_commit_passes(self) -> None:
        pins = {"dataset": "a" * 40, "harness": adapter.HARNESS_ID,
                "verifier": adapter.VERIFIER_ID, "adapter": adapter.self_pin()}
        adapter.validate_pins(pins)

    def test_a_changed_adapter_is_refused(self) -> None:
        pins = {"dataset": "a" * 40, "harness": adapter.HARNESS_ID,
                "verifier": adapter.VERIFIER_ID, "adapter": "sha256:" + "0" * 64}
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)


if __name__ == "__main__":
    unittest.main()
