import importlib.util
import io
import json
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("aa_lcr", "scripts/adapters/aa_lcr.py")

ROWS = [
    {"document_category": "Legal", "document_set_id": "eu_ai", "question_id": "1",
     "question": "How many airlines are listed?", "answer": "12",
     "data_source_filenames": "a.txt;b.txt", "input_tokens": "94494"},
    {"document_category": "Academia", "document_set_id": "ac", "question_id": "2",
     "question": "Which body published it?", "answer": "The Commission",
     "data_source_filenames": "gone.txt", "input_tokens": "71691"},
]


def archive(names=("a.txt", "b.txt")) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("lcr/", "")
        for name in names:
            zf.writestr(f"lcr/Legal/{name}", f"contents of {name}")
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def completion(content: str, *, reasoning: str = "", finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": 60},
    }


class MaterializeTests(unittest.TestCase):
    def test_documents_are_inlined_in_dataset_order(self) -> None:
        prompts, key, dropped = adapter.materialize(
            [ROWS[0]], archive(), skip_incomplete=False
        )
        text = prompts[0]["text"]
        self.assertIn("Document 1: a.txt", text)
        self.assertIn("Document 2: b.txt", text)
        self.assertLess(text.index("a.txt"), text.index("b.txt"))
        self.assertIn("contents of a.txt", text)
        # The question comes after the documents it is about.
        self.assertGreater(text.index("Question:"), text.index("contents of b.txt"))
        self.assertEqual(key["lcr-1"]["input_tokens"], 94494)
        self.assertEqual(dropped, [])

    def test_the_prompt_does_not_leak_the_answer(self) -> None:
        prompts, _, _ = adapter.materialize([ROWS[0]], archive(), skip_incomplete=False)
        self.assertNotIn("Answer:", prompts[0]["text"])

    def test_a_missing_document_is_an_error_by_default(self) -> None:
        # Eight documents the published dataset references are absent from its
        # own archive. Dropping those questions has to be asked for.
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.materialize(ROWS, archive(), skip_incomplete=False)
        self.assertIn("skip-incomplete", str(caught.exception))

    def test_skip_incomplete_drops_and_records(self) -> None:
        prompts, key, dropped = adapter.materialize(
            ROWS, archive(), skip_incomplete=True
        )
        self.assertEqual([p["id"] for p in prompts], ["lcr-1"])
        self.assertNotIn("lcr-2", key)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["question_id"], "2")
        self.assertEqual(dropped[0]["missing"], ["gone.txt"])

    def test_everything_dropped_is_an_error(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize([ROWS[1]], archive(), skip_incomplete=True)


class ScoringTests(unittest.TestCase):
    def key(self):
        return adapter.materialize([ROWS[0]], archive(), skip_incomplete=False)[1]

    def score(self, content, finish="stop"):
        return adapter.score_response(
            "lcr-1", completion(content, finish=finish),
            entry=self.key()["lcr-1"], replicate=0, thinking=True,
        )

    def test_exact_answer(self) -> None:
        row = self.score("Working through it.\nAnswer: 12")
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["input_tokens"], 94494)

    def test_wrong_answer(self) -> None:
        self.assertEqual(self.score("Answer: 4")["score"], 0.0)

    def test_whitespace_and_case_folded(self) -> None:
        key = adapter.materialize([dict(ROWS[0], answer="The Commission")],
                                  archive(), skip_incomplete=False)[1]
        row = adapter.score_response(
            "lcr-1", completion("Answer:  the   commission "),
            entry=key["lcr-1"], replicate=0, thinking=True,
        )
        self.assertEqual(row["score"], 1.0)

    def test_truncation_on_a_long_prompt_is_a_context_failure(self) -> None:
        # The failure this suite exists to catch is distinct from a wrong answer.
        row = self.score("Answer: 1", finish="length")
        self.assertTrue(row["context_failure"])

    def test_last_answer_line_wins(self) -> None:
        self.assertEqual(self.score("Answer: 9\nno wait\nAnswer: 12")["score"], 1.0)


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
