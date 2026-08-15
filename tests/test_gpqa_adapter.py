import importlib.util
import io
import json
import os
import socket
import tempfile
import unittest
import unittest.mock
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("gpqa_diamond", "scripts/adapters/gpqa_diamond.py")
protocol = load_module("run_eval_protocol", "scripts/run_eval_protocol.py")

DATASET_REVISION = "a" * 40


def valid_pins() -> dict:
    return {
        "dataset": DATASET_REVISION,
        "harness": adapter.HARNESS_ID,
        "verifier": adapter.VERIFIER_ID,
        "adapter": adapter.self_pin(),
    }


def gpqa_rows(count: int = 3) -> list[dict]:
    return [
        {
            "Record ID": f"rec{index}",
            "Question": f"Question {index}?",
            "Correct Answer": f"correct {index}",
            "Incorrect Answer 1": f"wrong {index} a",
            "Incorrect Answer 2": f"wrong {index} b",
            "Incorrect Answer 3": f"wrong {index} c",
            "High-level domain": "Physics",
            "Subdomain": "Optics",
        }
        for index in range(count)
    ]


def http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://inference:8000/v1/chat/completions", code, "error", {}, io.BytesIO(body)
    )


def completion(content: str, *, reasoning: str = "thinking", finish: str = "stop") -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": content, "reasoning_content": reasoning},
            }
        ],
        "usage": {
            "completion_tokens": 128,
            "completion_tokens_details": {"reasoning_tokens": 96},
        },
    }


class PinTests(unittest.TestCase):
    def test_valid_pins_accepted(self) -> None:
        adapter.validate_pins(valid_pins())

    def test_placeholder_dataset_rejected(self) -> None:
        pins = valid_pins()
        pins["dataset"] = "REPLACE_WITH_GPQA_REVISION"
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_branch_name_is_not_a_pin(self) -> None:
        pins = valid_pins()
        pins["dataset"] = "main"
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_edited_adapter_invalidates_its_pin(self) -> None:
        pins = valid_pins()
        pins["adapter"] = "sha256:" + "0" * 64
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_foreign_verifier_rejected(self) -> None:
        pins = valid_pins()
        pins["verifier"] = "llm-judge-v2"
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)


class MaterializeTests(unittest.TestCase):
    def test_option_order_is_seed_stable(self) -> None:
        examples = adapter.extract_examples(gpqa_rows())
        first, first_key = adapter.materialize(examples, 38027)
        second, second_key = adapter.materialize(examples, 38027)
        self.assertEqual(first, second)
        self.assertEqual(first_key, second_key)

    def test_answer_letter_matches_option_position(self) -> None:
        examples = adapter.extract_examples(gpqa_rows())
        prompts, key = adapter.materialize(examples, 38027)
        for prompt in prompts:
            entry = key[prompt["id"]]
            letter_index = adapter.CHOICES.index(entry["answer"])
            correct = next(
                example["correct"] for example in examples if example["id"] == prompt["id"]
            )
            self.assertEqual(entry["options"][letter_index], correct)
            self.assertIn(f"{entry['answer']}) {correct}", prompt["text"])

    def test_prompt_text_excludes_shared_boilerplate(self) -> None:
        prompts, _ = adapter.materialize(adapter.extract_examples(gpqa_rows()), 1)
        for prompt in prompts:
            self.assertNotIn("Answer:", prompt["text"])
            self.assertNotIn(adapter.ANSWER_INSTRUCTION, prompt["text"])

    def test_materialized_prompts_satisfy_the_runner(self) -> None:
        prompts, _ = adapter.materialize(adapter.extract_examples(gpqa_rows()), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpqa_diamond.jsonl"
            adapter.write_jsonl(path, prompts)
            ids = protocol.validate_prompts(path, adapter.SUITE)
        self.assertEqual(ids, [prompt["id"] for prompt in prompts])

    def test_incomplete_record_rejected(self) -> None:
        rows = gpqa_rows(1)
        rows[0]["Incorrect Answer 2"] = "  "
        with self.assertRaises(adapter.AdapterError):
            adapter.extract_examples(rows)

    def test_duplicate_record_ids_rejected(self) -> None:
        rows = gpqa_rows(2)
        rows[1]["Record ID"] = rows[0]["Record ID"]
        with self.assertRaises(adapter.AdapterError):
            adapter.extract_examples(rows)


class ParsingTests(unittest.TestCase):
    def test_answer_forms(self) -> None:
        cases = {
            "Answer: C": "C",
            "**Answer:** B": "B",
            "The answer is (D).": "D",
            "answer  ->  a": "A",
            "Reasoning...\n\nAnswer: B\n": "B",
            "Answer: A\nOn reflection, Answer: D": "D",
        }
        for content, expected in cases.items():
            with self.subTest(content=content):
                self.assertEqual(adapter.extract_answer(content), expected)

    def test_lone_letter_fallback(self) -> None:
        self.assertEqual(adapter.extract_answer("Long reasoning\n\nC"), "C")

    def test_unanswered_content(self) -> None:
        self.assertIsNone(adapter.extract_answer("I cannot determine this."))
        self.assertIsNone(adapter.extract_answer(""))

    def test_answering_prose_is_not_an_answer(self) -> None:
        self.assertIsNone(adapter.extract_answer("Answering this requires more thought."))

    def test_repetition_loop_detection(self) -> None:
        loop = "the same clause repeated over and over again forever " * 8
        self.assertTrue(adapter.has_repetition_loop(loop))
        self.assertFalse(adapter.has_repetition_loop("A varied explanation. " * 3 + "Answer: A"))


class PayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generation = {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }

    def test_reasoning_effort_goes_to_the_template_not_the_api(self) -> None:
        # Qwen3.8 reads reasoning_effort as a template variable and defaults it
        # to xhigh. Sent top-level it is accepted and ignored, so a policy
        # asking for medium would silently run at xhigh.
        payload = adapter.build_payload(
            "Q?", self.generation, model="m", seed=1, max_tokens=16, instruction="x"
        )
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
        )

    def test_generation_policy_is_applied(self) -> None:
        payload = adapter.build_payload(
            "Q?",
            self.generation,
            model="openai/qwen38-eval",
            seed=38027,
            max_tokens=4096,
            instruction=adapter.ANSWER_INSTRUCTION,
        )
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["repetition_penalty"], 1.0)
        self.assertEqual(payload["chat_template_kwargs"]["reasoning_effort"], "xhigh")
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["seed"], 38027)
        self.assertTrue(payload["messages"][0]["content"].startswith("Q?"))
        self.assertIn(adapter.ANSWER_INSTRUCTION, payload["messages"][0]["content"])

    def test_missing_generation_key_rejected(self) -> None:
        del self.generation["top_k"]
        with self.assertRaises(adapter.AdapterError):
            adapter.build_payload(
                "Q?", self.generation, model="m", seed=1, max_tokens=16, instruction="x"
            )


class LeakedReasoningTests(unittest.TestCase):
    """The template opens <think> in the prompt, so an unsplitting server
    returns one blob whose only marker is the closing tag."""

    def test_reasoning_in_content_is_recovered(self) -> None:
        reasoning, answer = adapter.split_reasoning("thinking hard</think>\n\nAnswer: C", "")
        self.assertEqual(reasoning, "thinking hard")
        self.assertEqual(answer.strip(), "Answer: C")

    def test_full_think_block_is_stripped(self) -> None:
        reasoning, answer = adapter.split_reasoning("<think>weighing</think>Answer: B", "")
        self.assertEqual(reasoning, "weighing")
        self.assertEqual(answer.strip(), "Answer: B")

    def test_separated_reasoning_is_left_alone(self) -> None:
        self.assertEqual(adapter.split_reasoning("Answer: A", "sep"), ("sep", "Answer: A"))

    def test_no_reasoning_at_all(self) -> None:
        self.assertEqual(adapter.split_reasoning("Answer: A", ""), ("", "Answer: A"))

    def test_answer_considered_while_thinking_is_not_scored(self) -> None:
        # Without the split, the last "Answer:" match would come from the
        # discarded line of reasoning.
        response = completion("Maybe Answer: A\u2026 no.</think>\n\nAnswer: B", reasoning="")
        row = adapter.score_response(
            "rec0", response, expected="B", replicate=0, thinking=True
        )
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["predicted"], "B")


class ScoringTests(unittest.TestCase):
    def score(self, response: dict, *, expected: str = "B", thinking: bool = True) -> dict:
        return adapter.score_response(
            "rec0", response, expected=expected, replicate=0, thinking=thinking
        )

    def test_correct_and_incorrect(self) -> None:
        self.assertEqual(self.score(completion("Answer: B"))["score"], 1.0)
        self.assertEqual(self.score(completion("Answer: C"))["score"], 0.0)

    def test_empty_answer_flag(self) -> None:
        row = self.score(completion("No conclusion reached."))
        self.assertTrue(row["empty_answer"])
        self.assertEqual(row["score"], 0.0)

    def test_context_failure_flag(self) -> None:
        row = self.score(completion("Answer: B", finish="length"))
        self.assertTrue(row["context_failure"])

    def test_premature_final_answer_when_thinking_produced_nothing(self) -> None:
        response = completion("Answer: B", reasoning="")
        response["usage"]["completion_tokens_details"]["reasoning_tokens"] = 0
        self.assertTrue(self.score(response)["premature_final_answer"])
        self.assertFalse(self.score(response, thinking=False)["premature_final_answer"])

    def test_malformed_response_rejected(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            self.score({"choices": []})


class ProbeTests(unittest.TestCase):
    def args(self, **overrides):
        defaults = {
            "action": "probe",
            "base_url": "http://inference:8000/v1",
            "model": "openai/qwen38-eval",
            "generation": None,
            "max_tokens": 2048,
            "request_timeout": 60.0,
        }
        defaults.update(overrides)
        return unittest.mock.Mock(**defaults)

    def test_healthy_server_passes(self) -> None:
        seen: list = []

        def client(base_url, api_key, payload, timeout):
            seen.append(payload)
            return completion("Answer: B")

        self.assertEqual(adapter.command_probe(self.args(), client=client), 0)
        self.assertEqual(
            seen[0]["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
        )

    def test_rejected_policy_field_is_reported(self) -> None:
        def client(base_url, api_key, payload, timeout):
            raise http_error(400, b'{"message":"unknown field reasoning_effort"}')

        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.command_probe(self.args(), client=client)
        self.assertIn("reasoning_effort", str(caught.exception))

    def test_missing_reasoning_fails_the_probe(self) -> None:
        def client(base_url, api_key, payload, timeout):
            response = completion("Answer: B", reasoning="")
            response["usage"] = {"completion_tokens": 4}
            return response

        with self.assertRaises(adapter.AdapterError):
            adapter.command_probe(self.args(), client=client)

    def test_stripped_reasoning_is_inferred_from_the_token_gap(self) -> None:
        # The pinned vLLM build removes the think block and returns nothing in
        # reasoning_content; the only evidence is tokens generated but unseen.
        def client(base_url, api_key, payload, timeout):
            response = completion("Answer: B", reasoning="")
            response["usage"] = {"completion_tokens": 2431}
            return response

        self.assertEqual(adapter.command_probe(self.args(), client=client), 0)

    def test_base_url_is_required(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.command_probe(self.args(base_url=""), client=lambda *a: {})


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run"
        self.prompts, key = adapter.materialize(adapter.extract_examples(gpqa_rows()), 38027)
        self.prompts_path = self.run_dir / "materialized" / "gpqa_diamond.jsonl"
        adapter.write_jsonl(self.prompts_path, self.prompts)
        adapter.write_json(adapter.key_path(self.run_dir), {"suite": adapter.SUITE, "items": key})
        self.key = key
        self.order = [prompt["id"] for prompt in reversed(self.prompts)]
        self.order_path = self.run_dir / "orders" / "gpqa_diamond.json"
        adapter.write_json(self.order_path, self.order)
        self.results_path = self.run_dir / "raw" / "candidate" / "gpqa_diamond-r0.jsonl"

        self.environ = {
            "EVAL_ACTION": "run",
            "EVAL_SUITE": adapter.SUITE,
            "EVAL_RUN_DIR": str(self.run_dir),
            "EVAL_PROMPTS_JSONL": str(self.prompts_path),
            "EVAL_TASK_ORDER_JSON": str(self.order_path),
            "EVAL_RESULTS_JSONL": str(self.results_path),
            "EVAL_PINS_JSON": json.dumps(valid_pins()),
            "EVAL_SERVED_MODEL": "openai/qwen38-eval",
            "EVAL_VARIANT": "candidate",
            "EVAL_REPLICATE": "0",
            "EVAL_SEED": "38027",
            "EVAL_ORDER_SEED": "38027",
            "EVAL_GENERATION_JSON": json.dumps(
                {
                    "enable_thinking": True,
                    "reasoning_effort": "xhigh",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 0.0,
                    "repetition_penalty": 1.0,
                }
            ),
            "OPENAI_BASE_URL": "http://inference:8000/v1",
            "OPENAI_API_KEY": "EMPTY",
        }
        patch = unittest.mock.patch.dict(os.environ, self.environ, clear=False)
        patch.start()
        self.addCleanup(patch.stop)

    def args(self, **overrides):
        defaults = {
            "action": "run",
            "concurrency": 1,
            "max_tokens": 4096,
            "request_timeout": 60.0,
            "retries": 0,
        }
        defaults.update(overrides)
        return unittest.mock.Mock(**defaults)

    def answering_client(self, calls: list):
        def client(base_url, api_key, payload, timeout):
            calls.append(payload)
            question = payload["messages"][0]["content"]
            item = next(
                prompt for prompt in self.prompts if question.startswith(prompt["text"])
            )
            return completion(f"Answer: {self.key[item['id']]['answer']}")

        return client

    def test_rows_satisfy_the_runner_contract(self) -> None:
        adapter.command_run(self.args(), client=self.answering_client([]))
        expected = set(protocol.validate_prompts(self.prompts_path, adapter.SUITE))
        protocol.validate_results(self.results_path, adapter.SUITE, 0, expected)
        rows = protocol.read_jsonl(self.results_path)
        self.assertTrue(all(row["score"] == 1.0 for row in rows))
        self.assertEqual([row["id"] for row in rows], self.order)

    def test_requests_follow_the_frozen_order(self) -> None:
        calls: list = []
        adapter.command_run(self.args(), client=self.answering_client(calls))
        asked = [
            next(
                prompt["id"]
                for prompt in self.prompts
                if payload["messages"][0]["content"].startswith(prompt["text"])
            )
            for payload in calls
        ]
        self.assertEqual(asked, self.order)

    def test_raw_responses_and_metadata_are_retained(self) -> None:
        adapter.command_run(self.args(), client=self.answering_client([]))
        for item_id in self.order:
            self.assertTrue(
                adapter.raw_response_path(self.run_dir, "candidate", 0, item_id).exists()
            )
        metadata = json.loads(
            (self.run_dir / "metadata" / "gpqa_diamond-candidate-r0.json").read_text()
        )
        self.assertEqual(metadata["accuracy"], 1.0)
        self.assertEqual(metadata["adapter"], adapter.self_pin())

    def test_timeout_is_recorded_not_retried(self) -> None:
        attempts = []

        def client(base_url, api_key, payload, timeout):
            attempts.append(payload)
            raise socket.timeout("timed out")

        adapter.command_run(self.args(), client=client)
        self.assertEqual(len(attempts), len(self.order))
        rows = protocol.read_jsonl(self.results_path)
        self.assertTrue(all(row["timeout"] and row["score"] == 0.0 for row in rows))
        protocol.validate_results(
            self.results_path,
            adapter.SUITE,
            0,
            set(protocol.validate_prompts(self.prompts_path, adapter.SUITE)),
        )

    def test_rejected_request_aborts_without_retrying(self) -> None:
        attempts = []

        def client(base_url, api_key, payload, timeout):
            attempts.append(payload)
            raise http_error(400, b'{"message":"unknown field reasoning_effort"}')

        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.command_run(self.args(retries=3), client=client)
        self.assertEqual(len(attempts), 1)
        self.assertIn("HTTP 400", str(caught.exception))

    def test_rate_limit_is_retried(self) -> None:
        attempts = []

        def client(base_url, api_key, payload, timeout):
            attempts.append(payload)
            raise http_error(429, b"slow down")

        with unittest.mock.patch.object(adapter.time, "sleep"):
            with self.assertRaises(adapter.AdapterError):
                adapter.command_run(self.args(retries=1), client=client)
        self.assertEqual(len(attempts), 2)

    def test_transport_error_aborts_the_run(self) -> None:
        def client(base_url, api_key, payload, timeout):
            raise ConnectionResetError("server went away")

        with self.assertRaises(adapter.AdapterError):
            adapter.command_run(self.args(), client=client)

    def test_concurrent_results_keep_task_order(self) -> None:
        adapter.command_run(self.args(concurrency=3), client=self.answering_client([]))
        rows = protocol.read_jsonl(self.results_path)
        self.assertEqual([row["id"] for row in rows], self.order)

    def test_pins_are_checked_before_any_request(self) -> None:
        os.environ["EVAL_PINS_JSON"] = json.dumps({**valid_pins(), "dataset": "main"})

        def client(base_url, api_key, payload, timeout):
            raise AssertionError("no request may be issued with invalid pins")

        with self.assertRaises(adapter.AdapterError):
            adapter.command_run(self.args(), client=client)


if __name__ == "__main__":
    unittest.main()
