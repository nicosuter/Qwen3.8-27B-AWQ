import base64
import importlib.util
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


adapter = load_module("hle", "scripts/adapters/hle.py")

TINY_PNG = "data:image/png;base64," + base64.b64encode(b"not-really-a-png").decode()

EXACT = {"id": "hle_1", "question": "How many?", "answer": "18",
         "answer_type": "exactMatch", "category": "Math",
         "raw_subject": "Mathematics", "image": None}
CHOICE = {"id": "hle_2", "question": "Which?\n\nAnswer Choices:\nA. x\nB. y\nC. z\nD. w",
          "answer": "D", "answer_type": "multipleChoice",
          "category": "Humanities/Social Science", "raw_subject": "Philosophy",
          "image": None}
WITH_IMAGE = dict(EXACT, id="hle_3", image=TINY_PNG)


def completion(content: str, *, reasoning: str = "", finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": 40},
    }


class MaterializeTests(unittest.TestCase):
    def test_both_answer_types_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts, key = adapter.materialize([EXACT, CHOICE], Path(tmp))
            self.assertEqual([p["id"] for p in prompts], ["hle_1", "hle_2"])
            self.assertEqual(key["hle_1"]["answer_type"], "exactMatch")
            self.assertEqual(key["hle_2"]["answer_type"], "multipleChoice")
            self.assertIsNone(key["hle_1"]["image"])

    def test_an_image_is_stored_relatively_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _, key = adapter.materialize([WITH_IMAGE], run_dir)
            ref = key["hle_3"]["image"]
            self.assertFalse(Path(ref["path"]).is_absolute())
            self.assertTrue((run_dir / ref["path"]).is_file())
            # Passed through unchanged rather than decoded and re-encoded, so
            # both checkpoints get the same bytes by construction.
            self.assertEqual(adapter.read_image(run_dir, ref), TINY_PNG)

    def test_a_corrupted_image_is_refused_on_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _, key = adapter.materialize([WITH_IMAGE], run_dir)
            ref = key["hle_3"]["image"]
            (run_dir / ref["path"]).write_text("data:image/png;base64,dGFtcGVyZWQ=")
            with self.assertRaises(adapter.AdapterError):
                adapter.read_image(run_dir, ref)

    def test_a_non_data_url_image_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(adapter.AdapterError):
                adapter.materialize([dict(EXACT, image="https://example.com/a.png")], Path(tmp))

    def test_an_answerless_row_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(adapter.AdapterError):
                adapter.materialize([dict(EXACT, answer="")], Path(tmp))

    def test_the_prompt_does_not_leak_the_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts, _ = adapter.materialize([EXACT], Path(tmp))
            self.assertNotIn("18", prompts[0]["text"])


class ScoringTests(unittest.TestCase):
    def entry(self, row=EXACT):
        with tempfile.TemporaryDirectory() as tmp:
            return adapter.materialize([row], Path(tmp))[1][row["id"]]

    def score(self, content, row=EXACT, finish="stop"):
        return adapter.score_response(
            row["id"], completion(content, finish=finish), entry=self.entry(row),
            replicate=0, thinking=True,
        )

    def test_exact_answer(self) -> None:
        result = self.score("Working.\nAnswer: 18")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["answer_type"], "exactMatch")

    def test_multiple_choice_letter(self) -> None:
        self.assertEqual(self.score("Answer: D", row=CHOICE)["score"], 1.0)

    def test_wrong_answer(self) -> None:
        self.assertEqual(self.score("Answer: 19")["score"], 0.0)

    def test_case_and_trailing_punctuation_folded(self) -> None:
        self.assertEqual(self.score("Answer: d.", row=CHOICE)["score"], 1.0)

    def test_last_answer_line_wins(self) -> None:
        self.assertEqual(self.score("Answer: 3\nrecheck\nAnswer: 18")["score"], 1.0)

    def test_truncation_is_a_context_failure(self) -> None:
        self.assertTrue(self.score("Answer: 18", finish="length")["context_failure"])

    def test_no_answer_line_falls_back_to_the_last_line(self) -> None:
        self.assertEqual(self.score("the total is\n18")["score"], 1.0)


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


if __name__ == "__main__":
    unittest.main()
