import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "scripts" / "adapters"))


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("matharena", "eval/scripts/adapters/matharena.py")
protocol = load_module("run_eval_protocol", "eval/scripts/run_eval_protocol.py")

SNAPSHOTS = {
    "MathArena/aime_2026": [
        {"problem_idx": "1", "answer": "277", "problem": "Patrick walked to the park..."},
        {"problem_idx": "2", "answer": "42", "problem": "Find n such that..."},
    ],
    "MathArena/apex-shortlist": [
        {"problem_idx": "1", "answer": "171", "problem": "Ana and Banana play...",
         "source": "usa-tst-2025-p1"},
    ],
}
SHA_A = "a" * 40
SHA_B = "b" * 40


def valid_pins() -> dict:
    return {
        "dataset": f"MathArena/aime_2026@{SHA_A},MathArena/apex-shortlist@{SHA_B}",
        "harness": adapter.HARNESS_ID,
        "verifier": adapter.VERIFIER_ID,
        "adapter": adapter.self_pin(),
    }


def completion(content: str, *, reasoning: str = "", finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": 500},
    }


class PinTests(unittest.TestCase):
    def test_per_snapshot_pins_are_parsed(self) -> None:
        resolved = adapter.validate_pins(valid_pins())
        self.assertEqual(resolved["MathArena/aime_2026"], SHA_A)
        self.assertEqual(resolved["MathArena/apex-shortlist"], SHA_B)

    def test_branch_name_is_not_a_pin(self) -> None:
        pins = dict(valid_pins(), dataset="MathArena/aime_2026@main")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_placeholder_rejected(self) -> None:
        pins = dict(valid_pins(), dataset="MathArena/aime_2026@REPLACE_WITH_COMMIT")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_snapshot_without_a_pin_is_refused(self) -> None:
        pins = dict(valid_pins(), dataset=f"MathArena/aime_2026@{SHA_A}")
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.validate_pins(pins, ["MathArena/aime_2026", "MathArena/apex-shortlist"])
        self.assertIn("apex-shortlist", str(caught.exception))

    def test_edited_adapter_invalidates_the_pin(self) -> None:
        pins = dict(valid_pins(), adapter="sha256:" + "0" * 64)
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)


class MaterializeTests(unittest.TestCase):
    def test_items_carry_snapshot_and_source(self) -> None:
        prompts, key = adapter.materialize(SNAPSHOTS)
        self.assertEqual(len(prompts), 3)
        self.assertEqual(prompts[0]["id"], "aime_2026-1")
        self.assertEqual(key["apex-shortlist-1"]["source"], "usa-tst-2025-p1")
        self.assertEqual(key["aime_2026-1"]["answer"], "277")

    def test_prompt_excludes_the_shared_instruction(self) -> None:
        prompts, _ = adapter.materialize(SNAPSHOTS)
        for prompt in prompts:
            self.assertNotIn(adapter.ANSWER_INSTRUCTION, prompt["text"])
            self.assertNotIn("Answer:", prompt["text"])

    def test_non_integer_answer_is_fatal(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize({"x/y": [{"problem_idx": "1", "answer": "banana",
                                          "problem": "p"}]})

    def test_missing_problem_text_is_fatal(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize({"x/y": [{"problem_idx": "1", "answer": "1", "problem": "  "}]})

    def test_prompts_satisfy_the_runner(self) -> None:
        prompts, _ = adapter.materialize(SNAPSHOTS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matharena.jsonl"
            adapter.write_jsonl(path, prompts)
            self.assertEqual(len(protocol.validate_prompts(path, adapter.SUITE)), 3)


class AnswerTests(unittest.TestCase):
    def test_answer_forms(self) -> None:
        cases = {
            "Answer: 277": "277",
            "so the value is \\boxed{277}": "277",
            "Answer: 1,234": "1234",
            "The answer is 42.": "42",
            "long working\n\n277": "277",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(adapter.extract_answer(text), expected)

    def test_last_boxed_value_wins(self) -> None:
        self.assertEqual(adapter.extract_answer("\\boxed{1} then \\boxed{2}"), "2")

    def test_no_answer(self) -> None:
        self.assertIsNone(adapter.extract_answer("I could not finish the algebra."))

    def test_normalization_strips_separators(self) -> None:
        self.assertEqual(adapter.normalize_answer(" 1,234 "), "1234")
        self.assertEqual(adapter.normalize_answer(277), "277")
        self.assertIsNone(adapter.normalize_answer("none"))


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        _, self.key = adapter.materialize(SNAPSHOTS)

    def score(self, content, item="aime_2026-1", **kwargs):
        return adapter.score_response(item, completion(content, **kwargs),
                                      entry=self.key[item], replicate=0, thinking=True)

    def test_correct_and_incorrect(self) -> None:
        self.assertEqual(self.score("Answer: 277")["score"], 1.0)
        self.assertEqual(self.score("Answer: 278")["score"], 0.0)

    def test_comma_form_still_matches(self) -> None:
        self.assertEqual(self.score("Answer: 277")["score"], 1.0)

    def test_unanswered_is_empty(self) -> None:
        row = self.score("I ran out of ideas.")
        self.assertTrue(row["empty_answer"])
        self.assertEqual(row["score"], 0.0)

    def test_truncated_reply_is_a_context_failure(self) -> None:
        self.assertTrue(self.score("Answer: 277", finish="length")["context_failure"])

    def test_reasoning_answer_is_not_scored(self) -> None:
        # The value considered mid-thought must not beat the final line.
        row = self.score("maybe 100</think>\n\nAnswer: 277")
        self.assertEqual(row["score"], 1.0)

    def test_rows_satisfy_the_runner_contract(self) -> None:
        row = self.score("Answer: 277")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            adapter.write_jsonl(path, [row])
            protocol.validate_results(path, adapter.SUITE, 0, {"aime_2026-1"})


if __name__ == "__main__":
    unittest.main()
