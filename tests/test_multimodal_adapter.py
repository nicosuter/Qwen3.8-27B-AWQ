import base64
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "adapters"))


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("multimodal", "scripts/adapters/multimodal.py")
protocol = load_module("run_eval_protocol", "scripts/run_eval_protocol.py")

SHA = {name: chr(97 + index) * 40 for index, name in enumerate(adapter.SETS)}


def valid_pins() -> dict:
    dataset = ",".join(
        f"{adapter.SETS[name][0]}@{SHA[name]}" for name in adapter.SETS
    )
    return {
        "dataset": dataset,
        "harness": adapter.HARNESS_ID,
        "verifier": adapter.VERIFIER_ID,
        "adapter": adapter.self_pin(),
    }


def fake_image(color=(255, 0, 0)):
    from PIL import Image

    return Image.new("RGB", (8, 8), color)


def completion(content: str, *, reasoning: str = "", finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": 40},
    }


class MetricTests(unittest.TestCase):
    def test_anls_accepts_near_matches_and_zeroes_far_ones(self) -> None:
        self.assertEqual(adapter.anls_score("0.28", ["0.28"]), 1.0)
        self.assertGreater(adapter.anls_score("0.283", ["0.28"]), 0.5)
        self.assertEqual(adapter.anls_score("completely other", ["0.28"]), 0.0)

    def test_anls_is_case_and_article_insensitive(self) -> None:
        self.assertEqual(adapter.anls_score("The Invoice", ["invoice"]), 1.0)

    def test_relaxed_accuracy_allows_five_percent_on_numbers(self) -> None:
        self.assertEqual(adapter.relaxed_score("14", ["14"]), 1.0)
        self.assertEqual(adapter.relaxed_score("14.5", ["14"]), 1.0)     # within 5%
        self.assertEqual(adapter.relaxed_score("16", ["14"]), 0.0)       # beyond 5%

    def test_relaxed_accuracy_is_exact_for_text(self) -> None:
        self.assertEqual(adapter.relaxed_score("apples", ["apples"]), 1.0)
        self.assertEqual(adapter.relaxed_score("pears", ["apples"]), 0.0)

    def test_relaxed_handles_zero_expected(self) -> None:
        self.assertEqual(adapter.relaxed_score("0", ["0"]), 1.0)
        self.assertEqual(adapter.relaxed_score("1", ["0"]), 0.0)

    def test_vqa_consensus_scoring(self) -> None:
        answers = ["dakota"] * 3 + ["nikon"] * 7
        self.assertEqual(adapter.vqa_score("dakota", answers), 1.0)
        self.assertAlmostEqual(adapter.vqa_score("dakota", ["dakota"] + ["x"] * 9), 1 / 3)
        self.assertEqual(adapter.vqa_score("sony", answers), 0.0)

    def test_levenshtein(self) -> None:
        self.assertEqual(adapter.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(adapter.levenshtein("", "abc"), 3)


class PinTests(unittest.TestCase):
    def test_per_set_pins_parse(self) -> None:
        resolved = adapter.validate_pins(valid_pins(), list(adapter.SETS))
        self.assertEqual(resolved["lmms-lab/ChartQA"], SHA["chartqa"])

    def test_branch_is_not_a_pin(self) -> None:
        pins = dict(valid_pins(), dataset="lmms-lab/DocVQA@main")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_set_without_a_pin_is_refused(self) -> None:
        pins = dict(valid_pins(), dataset=f"lmms-lab/DocVQA@{SHA['docvqa']}")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins, ["docvqa", "chartqa"])


class MaterializeTests(unittest.TestCase):
    def rows(self):
        return {
            "docvqa": [{"id": "docvqa-1", "question": "What is the value?",
                        "answers": ["0.28"], "image": fake_image(), "kind": "figure"}],
            "chartqa": [{"id": "chartqa-1", "question": "How many bars?",
                         "answers": ["14"], "image": fake_image((0, 255, 0)),
                         "kind": "human_test"}],
        }

    def test_images_are_written_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            prompts, key = adapter.materialize(self.rows(), run_dir)
            self.assertEqual(len(prompts), 2)
            for item_id, entry in key.items():
                self.assertFalse(Path(entry["image"]).is_absolute())
                self.assertTrue((run_dir / entry["image"]).exists())
                self.assertEqual(len(entry["image_sha256"]), 64)
            # Identical pixels for both checkpoints is the point of hashing them.
            again = adapter.materialize(self.rows(), run_dir)[1]
            self.assertEqual(
                {k: v["image_sha256"] for k, v in key.items()},
                {k: v["image_sha256"] for k, v in again.items()},
            )

    def test_metric_is_recorded_per_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, key = adapter.materialize(self.rows(), Path(tmp))
            self.assertEqual(key["docvqa-1"]["metric"], "anls")
            self.assertEqual(key["chartqa-1"]["metric"], "relaxed")

    def test_prompts_satisfy_the_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts, _ = adapter.materialize(self.rows(), Path(tmp))
            path = Path(tmp) / "multimodal.jsonl"
            adapter.write_jsonl(path, prompts)
            self.assertEqual(len(protocol.validate_prompts(path, adapter.SUITE)), 2)

    def test_incomplete_row_is_fatal(self) -> None:
        rows = [{"question": "", "answers": ["x"], "image": fake_image()}]
        with self.assertRaises(adapter.AdapterError):
            adapter.extract_rows("docvqa", rows, 1)

    def test_extract_rows_refuses_a_short_set(self) -> None:
        rows = [{"question": "q", "answers": ["a"], "image": fake_image(),
                 "questionId": "1"}]
        with self.assertRaises(adapter.AdapterError):
            adapter.extract_rows("docvqa", rows, 5)


class RequestTests(unittest.TestCase):
    def test_image_is_attached_as_a_data_url(self) -> None:
        seen = []

        def client(base_url, api_key, payload, timeout):
            seen.append(payload)
            return completion("Answer: 0.28")

        generation = {"enable_thinking": True, "reasoning_effort": "xhigh",
                      "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                      "presence_penalty": 0.0, "repetition_penalty": 1.0}
        args = SimpleNamespace(max_tokens=512, request_timeout=30.0, retries=0)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _, key = adapter.materialize(
                {"docvqa": [{"id": "docvqa-1", "question": "What is the value?",
                             "answers": ["0.28"], "image": fake_image(), "kind": "f"}]},
                run_dir,
            )
            row = adapter.run_item(
                "docvqa-1", "What is the value?", key["docvqa-1"], generation=generation,
                model="m", seed=1, replicate=0, variant="candidate", run_dir=run_dir,
                base_url="http://x/v1", api_key="EMPTY", args=args, client=client,
            )
        content = seen[0]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])
        self.assertIn(adapter.ANSWER_INSTRUCTION, content[1]["text"])
        # The sampling policy from the shared builder must survive the swap.
        self.assertEqual(seen[0]["chat_template_kwargs"]["reasoning_effort"], "xhigh")
        self.assertEqual(seen[0]["temperature"], 1.0)
        self.assertEqual(row["score"], 1.0)


class ImageResolutionTests(unittest.TestCase):
    """A materialized set has to survive being copied to another run."""

    def materialized(self, run_dir: Path) -> dict:
        return adapter.materialize(
            {"docvqa": [{"id": "docvqa-1", "question": "q", "answers": ["0.28"],
                         "image": fake_image(), "kind": "f"}]},
            run_dir,
        )[1]

    def test_relative_path_resolves_against_the_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            entry = self.materialized(run_dir)["docvqa-1"]
            resolved = adapter.resolve_image(run_dir, entry["image"])
            self.assertTrue(resolved.is_file())
            self.assertEqual(resolved.parent, adapter.image_dir(run_dir))

    def test_stale_absolute_path_falls_back_to_this_run(self) -> None:
        # What broke the third-party evals: a key file copied from another run
        # carried absolute paths into a directory that did not exist here.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.materialized(run_dir)
            stale = "/scratch/somewhere/else/materialized/multimodal-images/docvqa-1.png"
            resolved = adapter.resolve_image(run_dir, stale)
            self.assertEqual(resolved, adapter.image_dir(run_dir) / "docvqa-1.png")
            self.assertTrue(resolved.is_file())

    def test_absolute_path_is_kept_when_this_run_has_no_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stale = "/scratch/somewhere/else/docvqa-1.png"
            self.assertEqual(adapter.resolve_image(Path(tmp), stale), Path(stale))

    def test_mismatched_image_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            entry = self.materialized(run_dir)["docvqa-1"]
            path = adapter.resolve_image(run_dir, entry["image"])
            self.assertTrue(adapter.image_data_url(path, entry["image_sha256"]))
            # Substituting this run's copy is only safe because the bytes are
            # checked; a different image under the same name must not go out.
            with self.assertRaises(adapter.AdapterError):
                adapter.image_data_url(path, "0" * 64)


class ScoringTests(unittest.TestCase):
    def entry(self, metric="anls", answers=("0.28",)):
        return {"set": "docvqa", "metric": metric, "answers": list(answers),
                "image": "unused", "image_sha256": "0" * 64, "kind": "f"}

    def score(self, content, **kwargs):
        return adapter.score_response(
            "docvqa-1", completion(content, **kwargs), entry=self.entry(),
            replicate=0, thinking=True,
        )

    def test_answer_segment_is_scored(self) -> None:
        row = self.score("The chart shows 9.99 somewhere.\n\nAnswer: 0.28")
        self.assertEqual(row["score"], 1.0)

    def test_unanswered(self) -> None:
        row = adapter.score_response("docvqa-1", completion(""), entry=self.entry(),
                                     replicate=0, thinking=True)
        self.assertTrue(row["empty_answer"])
        self.assertEqual(row["score"], 0.0)

    def test_truncation_is_a_context_failure(self) -> None:
        self.assertTrue(self.score("Answer: 0.28", finish="length")["context_failure"])

    def test_rows_satisfy_the_runner_contract(self) -> None:
        row = self.score("Answer: 0.28")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            adapter.write_jsonl(path, [row])
            protocol.validate_results(path, adapter.SUITE, 0, {"docvqa-1"})


if __name__ == "__main__":
    unittest.main()
