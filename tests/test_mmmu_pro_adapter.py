import base64
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


adapter = load_module("mmmu_pro", "scripts/adapters/mmmu_pro.py")

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:  # pragma: no cover - environment guard
    HAVE_PIL = False


def fake_image(color=(255, 0, 0)):
    return Image.new("RGB", (4, 4), color)


def row(**over):
    data = {
        "id": "test_History_1",
        "question": "What does <image 1> show?",
        # The dataset stores options as the repr of a list, not as a list.
        "options": "['Alpha', 'Beta', 'Gamma', 'Delta', 'Eps', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa']",
        "answer": "B",
        "subject": "History",
        "topic_difficulty": "Medium",
        "image_1": fake_image() if HAVE_PIL else None,
    }
    for field in adapter.IMAGE_FIELDS[1:]:
        data.setdefault(field, None)
    data.update(over)
    return data


def completion(content: str, *, reasoning: str = "", finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": 50},
    }


class OptionParsingTests(unittest.TestCase):
    def test_a_repr_of_a_list_is_parsed(self) -> None:
        self.assertEqual(adapter.parse_options("['a', 'b']"), ["a", "b"])

    def test_an_actual_list_passes_through(self) -> None:
        self.assertEqual(adapter.parse_options(["a", "b"]), ["a", "b"])

    def test_unparseable_options_are_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.parse_options("not a list at all [")

    def test_a_non_list_literal_is_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.parse_options("42")


@unittest.skipUnless(HAVE_PIL, "Pillow is required to materialize images")
class MaterializeTests(unittest.TestCase):
    def test_images_are_written_hashed_and_referenced_relatively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            prompts, key = adapter.materialize([row()], run_dir)
            entry = key["test_History_1"]
            self.assertEqual(len(entry["images"]), 1)
            image = entry["images"][0]
            self.assertFalse(Path(image["path"]).is_absolute())
            self.assertTrue((run_dir / image["path"]).is_file())
            self.assertEqual(len(image["sha256"]), 64)
            self.assertIn("A. Alpha", prompts[0]["text"])
            self.assertIn("J. Kappa", prompts[0]["text"])

    def test_several_images_keep_their_dataset_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _, key = adapter.materialize(
                [row(image_2=fake_image((0, 255, 0)), image_5=fake_image((0, 0, 255)))],
                run_dir,
            )
            indexes = [i["index"] for i in key["test_History_1"]["images"]]
            self.assertEqual(indexes, [1, 2, 5])

    def test_an_item_without_images_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(adapter.AdapterError):
                adapter.materialize([row(image_1=None)], Path(tmp))

    def test_an_answer_outside_the_options_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(adapter.AdapterError):
                adapter.materialize([row(options="['a','b']", answer="J")], Path(tmp))

    def test_the_prompt_does_not_leak_the_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts, _ = adapter.materialize([row()], Path(tmp))
            self.assertNotIn("Answer:", prompts[0]["text"])

    def test_mismatched_pixels_are_refused_on_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _, key = adapter.materialize([row()], run_dir)
            image = key["test_History_1"]["images"][0]
            path = adapter.resolve_image(run_dir, image["path"])
            self.assertTrue(adapter.image_data_url(path, image["sha256"]))
            with self.assertRaises(adapter.AdapterError):
                adapter.image_data_url(path, "0" * 64)


@unittest.skipUnless(HAVE_PIL, "Pillow is required to materialize images")
class RequestTests(unittest.TestCase):
    def test_every_image_is_attached_before_the_text(self) -> None:
        seen = []

        def client(base_url, api_key, payload, timeout):
            seen.append(payload)
            return completion("Answer: B")

        generation = {"enable_thinking": True, "reasoning_effort": "xhigh",
                      "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                      "presence_penalty": 0.0, "repetition_penalty": 1.0}
        args = SimpleNamespace(max_tokens=512, request_timeout=30.0, retries=0)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            prompts, key = adapter.materialize(
                [row(image_2=fake_image((0, 255, 0)))], run_dir
            )
            result = adapter.run_item(
                "test_History_1", prompts[0]["text"], key["test_History_1"],
                generation=generation, model="m", seed=1, replicate=0,
                variant="candidate", run_dir=run_dir, base_url="http://x/v1",
                api_key="EMPTY", args=args, client=client,
            )
        content = seen[0]["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content],
                         ["image_url", "image_url", "text"])
        base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["images"], 2)


class ScoringTests(unittest.TestCase):
    def entry(self, options=10):
        return {"answer": "B", "options": options, "category": "History",
                "difficulty": "Medium", "images": [{"index": 1}]}

    def score(self, content, options=10, finish="stop"):
        return adapter.score_response(
            "x", completion(content, finish=finish), entry=self.entry(options),
            replicate=0, thinking=True,
        )

    def test_correct_letter(self) -> None:
        self.assertEqual(self.score("Answer: B")["score"], 1.0)

    def test_wrong_letter(self) -> None:
        self.assertEqual(self.score("Answer: C")["score"], 0.0)

    def test_a_letter_past_the_options_is_flagged(self) -> None:
        result = self.score("Answer: J", options=3)
        self.assertEqual(result["score"], 0.0)
        self.assertTrue(result["out_of_range_choice"])

    def test_truncation_is_a_context_failure(self) -> None:
        self.assertTrue(self.score("Answer: B", finish="length")["context_failure"])


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
