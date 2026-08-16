import base64
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
import unittest.mock
import zlib
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


adapter = load_module("livecodebench", "scripts/adapters/livecodebench.py")
protocol = load_module("run_eval_protocol", "scripts/run_eval_protocol.py")

STDIN_ROW = {
    "question_id": "abc387_b",
    "question_content": "Read n and print n doubled.",
    "starter_code": "",
    "platform": "atcoder",
    "difficulty": "easy",
    "contest_date": "2025-01-04T00:00:00",
    "public_test_cases": json.dumps(
        [{"input": "2\n", "output": "4\n", "testtype": "stdin"}]
    ),
    "private_test_cases": "",
}

FUNCTIONAL_ROW = {
    "question_id": "lc_double",
    "question_content": "Return the doubled list.",
    "starter_code": "class Solution:\n    def doubleAll(self, nums: List[int]) -> List[int]:",
    "platform": "leetcode",
    "difficulty": "medium",
    "contest_date": "2025-02-01T00:00:00",
    "public_test_cases": json.dumps(
        [{"input": "[1, 2, 3]", "output": "[2, 4, 6]", "testtype": "functional"}]
    ),
    "private_test_cases": "",
}


def exec_args(**overrides):
    defaults = {
        "python": sys.executable,
        "exec_timeout": 10.0,
        "exec_memory_mb": 2048,
        "item_budget": 60.0,
        "max_tokens": 4096,
        "request_timeout": 60.0,
        "retries": 0,
        "concurrency": 1,
        "action": "run",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def completion(content: str, *, reasoning: str = "", finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": 900},
    }


class DecodeTests(unittest.TestCase):
    def test_plain_json_tests(self) -> None:
        raw = json.dumps([{"input": "1", "output": "2", "testtype": "stdin"}])
        self.assertEqual(adapter.decode_tests(raw)[0]["output"], "2")

    def test_pickled_string_payload_is_decoded(self) -> None:
        # LiveCodeBench stores private tests as base64(zlib(pickle(json_string))).
        payload = json.dumps([{"input": "9", "output": "18", "testtype": "stdin"}])
        raw = base64.b64encode(zlib.compress(pickle.dumps(payload))).decode()
        self.assertEqual(adapter.decode_tests(raw)[0]["input"], "9")

    def test_pickle_that_would_execute_code_is_refused(self) -> None:
        # A dataset revision must not be able to run code at prepare time.
        class Evil:
            def __reduce__(self):
                return (os.system, ("echo owned",))

        raw = base64.b64encode(zlib.compress(pickle.dumps(Evil()))).decode()
        with self.assertRaises(pickle.UnpicklingError) as caught:
            adapter.decode_tests(raw)
        self.assertIn("refusing to load", str(caught.exception))

    def test_empty_tests(self) -> None:
        self.assertEqual(adapter.decode_tests(""), [])


class MaterializeTests(unittest.TestCase):
    def test_stdin_and_functional_rows(self) -> None:
        prompts, key = adapter.materialize([STDIN_ROW, FUNCTIONAL_ROW])
        self.assertEqual(prompts[0]["id"], "abc387_b")
        self.assertIsNone(key["abc387_b"]["method"])
        self.assertEqual(key["lc_double"]["method"], "doubleAll")
        self.assertIn("Complete this class", prompts[1]["text"])
        self.assertEqual(prompts[1]["category"], "leetcode/medium")

    def test_prompt_excludes_the_shared_instruction(self) -> None:
        prompts, _ = adapter.materialize([STDIN_ROW])
        self.assertNotIn(adapter.ANSWER_INSTRUCTION, prompts[0]["text"])

    def test_prompts_satisfy_the_runner(self) -> None:
        prompts, _ = adapter.materialize([STDIN_ROW, FUNCTIONAL_ROW])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lcb.jsonl"
            adapter.write_jsonl(path, prompts)
            self.assertEqual(len(protocol.validate_prompts(path, adapter.SUITE)), 2)

    def test_missing_tests_rejected(self) -> None:
        row = dict(STDIN_ROW, public_test_cases="", private_test_cases="")
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize([row])

    def test_starter_without_method_rejected(self) -> None:
        row = dict(FUNCTIONAL_ROW, starter_code="class Solution:\n    pass")
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize([row])


class ExtractionTests(unittest.TestCase):
    def test_last_python_block_wins(self) -> None:
        text = "first\n```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```"
        self.assertEqual(adapter.extract_code(text), "print(2)")

    def test_unlabelled_block(self) -> None:
        self.assertEqual(adapter.extract_code("```\nprint(3)\n```"), "print(3)")

    def test_no_block(self) -> None:
        self.assertIsNone(adapter.extract_code("I cannot solve this."))

    def test_stdout_normalization(self) -> None:
        self.assertEqual(adapter.normalize_stdout("4 \n\n"), "4")

    def test_functional_comparison_is_structural(self) -> None:
        self.assertTrue(adapter.compare_functional("[2, 4, 6]", "[2,4,6]"))
        self.assertFalse(adapter.compare_functional("[2, 4, 6]", "[2,4,7]"))


class ExecutionTests(unittest.TestCase):
    """Runs real subprocesses; the programs are fixtures written by this test."""

    def setUp(self) -> None:
        _, self.key = adapter.materialize([STDIN_ROW, FUNCTIONAL_ROW])

    def evaluate(self, code: str, item: str = "abc387_b", **overrides):
        args = exec_args(**overrides)
        return adapter.evaluate(
            code, self.key[item], python=args.python, exec_timeout=args.exec_timeout,
            memory_mb=args.exec_memory_mb, item_budget=args.item_budget,
        )

    def test_correct_stdin_solution_passes(self) -> None:
        verdict = self.evaluate("print(int(input()) * 2)")
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["status"], "passed")
        self.assertEqual(verdict["tests_passed"], 1)

    def test_wrong_answer_fails(self) -> None:
        verdict = self.evaluate("print(int(input()) * 3)")
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["status"], "wrong_answer")

    def test_runtime_error_is_reported(self) -> None:
        verdict = self.evaluate("raise ValueError('boom')")
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict["status"].startswith("error:"))

    def test_infinite_loop_is_killed(self) -> None:
        verdict = self.evaluate("while True:\n    pass", exec_timeout=2.0)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["status"], "timeout")

    def test_functional_solution_is_called_correctly(self) -> None:
        code = ("from typing import List\n"
                "class Solution:\n"
                "    def doubleAll(self, nums: List[int]) -> List[int]:\n"
                "        return [n * 2 for n in nums]\n")
        verdict = self.evaluate(code, item="lc_double")
        self.assertTrue(verdict["passed"])

    def test_functional_wrong_result_fails(self) -> None:
        code = ("from typing import List\n"
                "class Solution:\n"
                "    def doubleAll(self, nums: List[int]) -> List[int]:\n"
                "        return nums\n")
        self.assertFalse(self.evaluate(code, item="lc_double")["passed"])


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        _, self.key = adapter.materialize([STDIN_ROW])

    def score(self, content: str, **kwargs):
        return adapter.score_response(
            "abc387_b", completion(content, **kwargs), entry=self.key["abc387_b"],
            replicate=0, thinking=True, args=exec_args(),
        )

    def test_passing_solution_scores_one(self) -> None:
        row = self.score("```python\nprint(int(input()) * 2)\n```")
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["execution_status"], "passed")

    def test_missing_code_block_is_an_empty_answer(self) -> None:
        row = self.score("I would solve it with dynamic programming.")
        self.assertEqual(row["score"], 0.0)
        self.assertTrue(row["empty_answer"])
        self.assertEqual(row["execution_status"], "no_code_block")

    def test_execution_failure_does_not_set_the_timeout_flag(self) -> None:
        # A slow program is not the server failing to answer; conflating them
        # would pollute a gate that compares failure-mode rates.
        row = self.score("```python\nwhile True: pass\n```")
        self.assertFalse(row["timeout"])
        self.assertEqual(row["execution_status"], "timeout")

    def test_truncated_reply_is_a_context_failure(self) -> None:
        row = self.score("```python\nprint(1)\n```", finish="length")
        self.assertTrue(row["context_failure"])

    def test_rows_satisfy_the_runner_contract(self) -> None:
        row = self.score("```python\nprint(int(input()) * 2)\n```")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            adapter.write_jsonl(path, [row])
            protocol.validate_results(path, adapter.SUITE, 0, {"abc387_b"})


class PinTests(unittest.TestCase):
    def valid(self) -> dict:
        return {
            "dataset": "0" * 40,
            "harness": adapter.HARNESS_ID,
            "verifier": adapter.VERIFIER_ID,
            "adapter": adapter.self_pin(),
        }

    def test_valid_pins(self) -> None:
        adapter.validate_pins(self.valid())

    def test_placeholder_rejected(self) -> None:
        pins = self.valid()
        pins["dataset"] = "REPLACE_WITH_LCB_V6_LITE_REVISION"
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_edited_adapter_invalidates_the_pin(self) -> None:
        pins = self.valid()
        pins["adapter"] = "sha256:" + "0" * 64
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)


if __name__ == "__main__":
    unittest.main()


class DeferredExecutionTests(unittest.TestCase):
    """Generation and execution must be separable without changing the result.

    Executing model-written code is the verifier, and it has no business running
    on the machine serving the model. Splitting it is only safe if the two-step
    path produces exactly what the single pass produced.
    """

    SOLUTION = "```python\nn = int(input())\nprint(n * 2)\n```"

    def key(self) -> dict:
        _, key = adapter.materialize([dict(STDIN_ROW)])
        return key

    def test_deferred_run_executes_nothing_and_flags_every_row(self) -> None:
        key = self.key()
        item_id = next(iter(key))
        # A solution that would fail loudly if it were ever executed.
        row = adapter.score_response(
            item_id, completion("```python\nraise SystemExit(3)\n```"),
            entry=key[item_id], replicate=0, thinking=True,
            args=exec_args(defer_execution=True), execute=False,
        )
        self.assertTrue(row["deferred"])
        self.assertEqual(row["execution_status"], "deferred")
        self.assertEqual(row["score"], 0.0)
        self.assertEqual(row["tests_total"], len(key[item_id]["tests"]))

    def test_scoring_a_deferred_generation_matches_a_single_pass(self) -> None:
        key = self.key()
        item_id = next(iter(key))
        response = completion(self.SOLUTION)

        direct = adapter.score_response(
            item_id, response, entry=key[item_id], replicate=0,
            thinking=True, args=exec_args(), execute=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            gen = work / "gen.jsonl"
            gen.write_text(json.dumps({
                "id": item_id, "replicate": 0, "response": response,
                "timeout": False, "attempts": 1,
                "started_at": 1.0, "finished_at": 2.0, "elapsed_seconds": 1.0,
            }) + "\n", encoding="utf-8")
            (work / "gen.meta.json").write_text(json.dumps({
                "variant": "candidate", "replicate": 0, "seed": 7,
                "served_model": "m", "concurrency": 1, "max_tokens": 4096,
                "generation": {"enable_thinking": True},
            }), encoding="utf-8")
            (work / "key.json").write_text(json.dumps({"items": key}), encoding="utf-8")

            rc = adapter.command_score(exec_args(
                action="score", generations=gen, key=work / "key.json",
                results=work / "results.jsonl", metadata=work / "meta.json",
            ))
            self.assertEqual(rc, 0)
            rows = [json.loads(l) for l in (work / "results.jsonl").read_text().splitlines() if l.strip()]
            meta = json.loads((work / "meta.json").read_text())

        self.assertEqual(len(rows), 1)
        scored = rows[0]
        self.assertFalse(scored["deferred"])
        # The verdict must be identical to the one-pass run.
        for field in ("score", "execution_status", "tests_passed", "tests_total", "category"):
            self.assertEqual(scored[field], direct[field], field)
        # Request-side facts survive the hand-off; scoring cannot recompute them.
        self.assertEqual(scored["elapsed_seconds"], 1.0)
        self.assertEqual(scored["attempts"], 1)
        self.assertEqual(meta["seed"], 7)
        self.assertEqual(meta["variant"], "candidate")
        self.assertTrue(meta["execution"]["deferred"])

    def test_a_timed_out_generation_scores_without_a_response(self) -> None:
        key = self.key()
        item_id = next(iter(key))
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            gen = work / "gen.jsonl"
            gen.write_text(json.dumps({
                "id": item_id, "replicate": 0, "response": None, "timeout": True,
            }) + "\n", encoding="utf-8")
            (work / "key.json").write_text(json.dumps({"items": key}), encoding="utf-8")
            adapter.command_score(exec_args(
                action="score", generations=gen, key=work / "key.json",
                results=work / "results.jsonl", metadata=None,
            ))
            row = json.loads((work / "results.jsonl").read_text().strip())
        self.assertTrue(row["timeout"])
        self.assertFalse(row["deferred"])
        self.assertEqual(row["execution_status"], "not_run")

    def test_generations_referencing_unknown_items_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            gen = work / "gen.jsonl"
            gen.write_text(json.dumps({"id": "nope", "replicate": 0, "response": None}) + "\n",
                           encoding="utf-8")
            (work / "key.json").write_text(json.dumps({"items": self.key()}), encoding="utf-8")
            with self.assertRaises(adapter.AdapterError):
                adapter.command_score(exec_args(
                    action="score", generations=gen, key=work / "key.json",
                    results=work / "results.jsonl", metadata=None,
                ))
