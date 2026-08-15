import collections
import importlib.util
import io
import json
import os
import random
import socket
import sys
import tempfile
import unittest
import unittest.mock
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "adapters"))


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("ruler", "scripts/adapters/ruler.py")
protocol = load_module("run_eval_protocol", "scripts/run_eval_protocol.py")


class WordTokenizer:
    """Whitespace tokenizer: one token per word, so lengths are word counts."""

    def __init__(self) -> None:
        self.words: list[str] = []
        self.ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        return [self._id(word) for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.words[index] for index in ids)

    def _id(self, word: str) -> int:
        if word not in self.ids:
            self.ids[word] = len(self.words)
            self.words.append(word)
        return self.ids[word]


def corpus_text(words: int = 40000) -> str:
    return " ".join(f"filler{index % 977:04d}" for index in range(words))


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
            "completion_tokens": 64,
            "completion_tokens_details": {"reasoning_tokens": 32},
        },
    }


class LengthBudgetTests(unittest.TestCase):
    def test_default_lengths_stop_at_131072(self) -> None:
        self.assertEqual(adapter.DEFAULT_LENGTHS, (4096, 32768, 131072))

    def test_lengths_are_parsed_and_sorted(self) -> None:
        lengths = adapter.parse_lengths(
            "131072,4096,32768", max_model_len=262144, output_reserve=16384
        )
        self.assertEqual(lengths, [4096, 32768, 131072])

    def test_full_window_length_is_rejected(self) -> None:
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.parse_lengths("262144", max_model_len=262144, output_reserve=16384)
        self.assertIn("leaves no room for an answer", str(caught.exception))

    def test_length_fits_when_the_window_is_larger(self) -> None:
        lengths = adapter.parse_lengths(
            "262144", max_model_len=524288, output_reserve=16384
        )
        self.assertEqual(lengths, [262144])

    def test_non_numeric_length_rejected(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.parse_lengths("4096,long", max_model_len=262144, output_reserve=1024)

    def test_length_labels(self) -> None:
        self.assertEqual(adapter.length_label(4096), "4k")
        self.assertEqual(adapter.length_label(131072), "128k")
        self.assertEqual(adapter.length_label(5000), "5000")


class SynthesisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WordTokenizer()
        self.corpus_ids = self.tokenizer.encode(corpus_text())

    def build(self, task: str, length: int = 4096, index: int = 0):
        return adapter.build_item(
            task,
            length,
            index,
            corpus_ids=self.corpus_ids,
            tokenizer=self.tokenizer,
            seed=38027,
        )

    def test_every_task_hits_its_length_target(self) -> None:
        # The allowance is deliberately left unfilled for the chat template, so
        # the contract is the target minus it, not the nominal length.
        target = 4096 - adapter.TEMPLATE_ALLOWANCE
        tolerance = max(adapter.TEMPLATE_ALLOWANCE // 2, int(4096 * adapter.LENGTH_TOLERANCE))
        for task in adapter.TASKS:
            with self.subTest(task=task):
                prompt, key = self.build(task)
                achieved = len(self.tokenizer.encode(prompt["text"]))
                self.assertLessEqual(abs(achieved - target), tolerance)
                self.assertEqual(key["achieved_tokens"], achieved)

    def test_synthesis_is_deterministic(self) -> None:
        for task in adapter.TASKS:
            with self.subTest(task=task):
                self.assertEqual(self.build(task)[0]["text"], self.build(task)[0]["text"])

    def test_needles_are_present_and_answerable(self) -> None:
        for task in ("niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery"):
            with self.subTest(task=task):
                prompt, key = self.build(task)
                for value in key["expected"]:
                    self.assertIn(value, prompt["text"])

    def test_variable_tracking_chain_is_traceable(self) -> None:
        prompt, key = self.build("vt")
        self.assertEqual(len(key["expected"]), 5)
        for name in key["expected"]:
            self.assertIn(name, prompt["text"])

    def test_counting_tasks_offer_a_candidate_shortlist(self) -> None:
        # Without candidates the model tallies the whole list and never stops:
        # every cwe and fwe item burned the full output cap and scored zero.
        for task, count in (("cwe", 10), ("fwe", 3)):
            with self.subTest(task=task):
                prompt, key = self.build(task, length=32768)
                self.assertIn("Candidates:", prompt["text"])
                listed = [w.strip() for w in prompt["text"].rsplit("Candidates:", 1)[1].split(",")]
                # Distractors are bounded by how many distinct filler words the
                # list actually carries, so the shortlist can be shorter than the
                # ratio asks for; it must still be long enough to punish guessing.
                self.assertLessEqual(len(listed), count * (1 + adapter.WORD_CANDIDATE_RATIO))
                self.assertGreaterEqual(len(listed), count * 2)
                self.assertEqual(len(listed), len(set(listed)))
                for word in key["expected"]:
                    self.assertIn(word, listed)

    def test_frequent_words_dominate_the_list(self) -> None:
        for task, count in (("cwe", 10), ("fwe", 3)):
            with self.subTest(task=task):
                prompt, key = self.build(task, length=32768)
                words = prompt["text"].replace(",", " ").split()
                counts = {word: words.count(word) for word in set(words)}
                self.assertEqual(len(key["expected"]), count)
                least_frequent = min(counts[word] for word in key["expected"])
                others = [
                    counts[word]
                    for word in counts
                    if word not in key["expected"] and word.startswith(("item", "token"))
                ]
                self.assertGreater(least_frequent, max(others))
                self.assertLessEqual(least_frequent / max(others), 2.6)

    def test_short_list_uses_part_of_the_filler_pool(self) -> None:
        # A 4k context cannot carry 600 distinct filler words; it must use fewer
        # rather than refuse the item.
        words = adapter.compose_word_list(
            900, ["a", "b", "c"], [f"f{index}" for index in range(600)], random.Random(0)
        )
        self.assertEqual(len(words), 900)
        self.assertGreater(min(words.count(word) for word in "abc"), max(
            words.count(word) for word in set(words) if word.startswith("f")
        ))

    def test_word_list_refuses_a_context_that_is_too_short(self) -> None:
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.compose_word_list(
                30, ["a", "b", "c"], [f"f{index}" for index in range(400)], random.Random(0)
            )
        self.assertIn("too short", str(caught.exception))

    def test_distinct_filler_words_are_capped(self) -> None:
        # Counting cost is set by bucket count, not list length: uncapped, the
        # 128k list asked for the top 3 of 400 buckets and the model reasoned
        # until the output cap at every length, scoring 0 on both checkpoints.
        words = adapter.compose_word_list(
            32000, ["a", "b", "c"], [f"f{index}" for index in range(600)], random.Random(0)
        )
        distinct = {word for word in words if word.startswith("f")}
        self.assertLessEqual(len(distinct), adapter.MAX_DISTINCT_FILLER)

    def test_frequency_ladder_descends_to_the_tight_ratio(self) -> None:
        ladder = adapter.frequency_ladder(10)
        self.assertEqual(ladder[0], adapter.TOP_RATIO)
        self.assertAlmostEqual(ladder[-1], adapter.TIGHT_RATIO)
        self.assertEqual(ladder, sorted(ladder, reverse=True))

    def test_lowest_frequent_word_stays_near_the_filler(self) -> None:
        # The point of the ladder: discriminating rank 3 from rank 4 must take
        # real counting, or both checkpoints score 1.0 and the task is blind to
        # any degradation.
        for total in (2000, 32000):
            with self.subTest(total=total):
                words = adapter.compose_word_list(
                    total, ["a", "b", "c"], [f"f{index}" for index in range(600)], random.Random(0)
                )
                counts = collections.Counter(words)
                lowest = min(counts[word] for word in "abc")
                top_filler = max(count for word, count in counts.items() if word.startswith("f"))
                self.assertGreater(lowest, top_filler)
                self.assertLessEqual(lowest / top_filler, 2.6)

    def test_item_ids_carry_length_and_task(self) -> None:
        prompt, _ = self.build("niah_single", length=131072, index=7)
        self.assertEqual(prompt["id"], "ruler-128k-niah_single-07")
        self.assertEqual(prompt["length"], "128k")
        self.assertEqual(prompt["category"], "128k/niah_single")

    def test_prompts_satisfy_the_runner(self) -> None:
        prompts = [self.build(task)[0] for task in adapter.TASKS]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ruler.jsonl"
            adapter.write_jsonl(path, prompts)
            ids = protocol.validate_prompts(path, adapter.SUITE)
        self.assertEqual(len(ids), len(adapter.TASKS))


class PinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = Path(self.tmp.name) / "corpus.txt"
        self.corpus.write_text("some haystack text", encoding="utf-8")

    def pins(self) -> dict:
        return {
            "dataset": adapter.corpus_pin(self.corpus),
            "harness": adapter.HARNESS_ID,
            "verifier": adapter.VERIFIER_ID,
            "adapter": adapter.self_pin(),
        }

    def test_valid_pins_accepted(self) -> None:
        adapter.validate_pins(self.pins(), self.corpus)

    def test_changed_corpus_invalidates_the_pin(self) -> None:
        pins = self.pins()
        self.corpus.write_text("a different haystack", encoding="utf-8")
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.validate_pins(pins, self.corpus)
        self.assertIn("haystack source changed", str(caught.exception))

    def test_placeholder_dataset_rejected(self) -> None:
        pins = self.pins()
        pins["dataset"] = "REPLACE_WITH_RULER_HAYSTACK_CORPUS_REVISION"
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_edited_adapter_invalidates_its_pin(self) -> None:
        pins = self.pins()
        pins["adapter"] = "sha256:" + "0" * 64
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_pin_covers_the_shared_module(self) -> None:
        common = ROOT / "scripts" / "adapters" / "_common.py"
        original = common.read_bytes()
        before = adapter.self_pin()
        try:
            common.write_bytes(original + b"\n# touched\n")
            self.assertNotEqual(adapter.self_pin(), before)
        finally:
            common.write_bytes(original)
        self.assertEqual(adapter.self_pin(), before)

    def test_corpus_directory_is_hashed_in_order(self) -> None:
        directory = Path(self.tmp.name) / "corpus"
        directory.mkdir()
        (directory / "b.txt").write_text("second", encoding="utf-8")
        (directory / "a.txt").write_text("first", encoding="utf-8")
        self.assertEqual(adapter.read_corpus(directory), "first\n\nsecond")


class ScoringTests(unittest.TestCase):
    def entry(self, expected: list[str]) -> dict:
        return {
            "task": "niah_multivalue",
            "length": "32k",
            "nominal_tokens": 32768,
            "achieved_tokens": 32700,
            "expected": expected,
        }

    def score(self, content: str, expected: list[str], **kwargs) -> dict:
        return adapter.score_response(
            "ruler-32k-niah_multivalue-00",
            completion(content, **kwargs),
            entry=self.entry(expected),
            replicate=0,
            thinking=True,
        )

    def test_full_and_partial_recall(self) -> None:
        self.assertEqual(self.score("Answer: 111, 222", ["111", "222"])["score"], 1.0)
        self.assertEqual(self.score("Answer: 111", ["111", "222"])["score"], 0.5)
        self.assertEqual(self.score("Answer: 999", ["111", "222"])["score"], 0.0)

    def test_only_the_answer_segment_counts(self) -> None:
        content = "I considered 111 while reasoning.\n\nAnswer: 222"
        self.assertEqual(self.score(content, ["111", "222"])["score"], 0.5)

    def test_substring_matches_are_not_credited(self) -> None:
        self.assertEqual(self.score("Answer: 1112223", ["111", "222"])["score"], 0.0)

    def test_matching_is_case_insensitive_for_variables(self) -> None:
        row = self.score("Answer: var123, VAR456", ["VAR123", "var456"])
        self.assertEqual(row["score"], 1.0)

    def test_truncated_output_is_a_context_failure(self) -> None:
        row = self.score("Answer: 111", ["111"], finish="length")
        self.assertTrue(row["context_failure"])

    def test_empty_reply(self) -> None:
        row = self.score("", ["111"])
        self.assertTrue(row["empty_answer"])
        self.assertEqual(row["score"], 0.0)

    def test_rows_carry_length_and_task_for_drilldown(self) -> None:
        row = self.score("Answer: 111", ["111"])
        self.assertEqual(row["length"], "32k")
        self.assertEqual(row["task"], "niah_multivalue")
        self.assertEqual(row["category"], "32k/niah_multivalue")

    def test_per_length_accuracy(self) -> None:
        rows = [
            {"length": "4k", "score": 1.0},
            {"length": "4k", "score": 0.0},
            {"length": "128k", "score": 0.5},
        ]
        self.assertEqual(
            adapter.per_length_accuracy(rows), {"128k": 0.5, "4k": 0.5}
        )


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run"
        tokenizer = WordTokenizer()
        corpus_ids = tokenizer.encode(corpus_text(20000))

        prompts, key = [], {}
        for task in ("niah_single", "vt", "cwe"):
            prompt, entry = adapter.build_item(
                task, 4096, 0, corpus_ids=corpus_ids, tokenizer=tokenizer, seed=38027
            )
            prompts.append(prompt)
            key[prompt["id"]] = entry
        self.prompts, self.key = prompts, key

        self.prompts_path = self.run_dir / "materialized" / "ruler.jsonl"
        adapter.write_jsonl(self.prompts_path, prompts)
        adapter.write_json(adapter.key_path(self.run_dir), {"suite": "ruler", "items": key})
        self.order = [prompt["id"] for prompt in reversed(prompts)]
        self.order_path = self.run_dir / "orders" / "ruler.json"
        adapter.write_json(self.order_path, self.order)
        self.results_path = self.run_dir / "raw" / "candidate" / "ruler-r0.jsonl"

        corpus = Path(self.tmp.name) / "corpus.txt"
        corpus.write_text("unused at run time", encoding="utf-8")
        environ = {
            "EVAL_ACTION": "run",
            "EVAL_SUITE": "ruler",
            "EVAL_RUN_DIR": str(self.run_dir),
            "EVAL_PROMPTS_JSONL": str(self.prompts_path),
            "EVAL_TASK_ORDER_JSON": str(self.order_path),
            "EVAL_RESULTS_JSONL": str(self.results_path),
            "EVAL_PINS_JSON": json.dumps(
                {
                    "dataset": adapter.corpus_pin(corpus),
                    "harness": adapter.HARNESS_ID,
                    "verifier": adapter.VERIFIER_ID,
                    "adapter": adapter.self_pin(),
                }
            ),
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
        patch = unittest.mock.patch.dict(os.environ, environ, clear=False)
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

    def perfect_client(self, calls: list | None = None):
        def client(base_url, api_key, payload, timeout):
            if calls is not None:
                calls.append(payload)
            text = payload["messages"][0]["content"]
            item = next(
                prompt for prompt in self.prompts if text.startswith(prompt["text"][:200])
            )
            return completion("Answer: " + ", ".join(self.key[item["id"]]["expected"]))

        return client

    def test_rows_satisfy_the_runner_contract(self) -> None:
        adapter.command_run(self.args(), client=self.perfect_client())
        expected = set(protocol.validate_prompts(self.prompts_path, adapter.SUITE))
        protocol.validate_results(self.results_path, adapter.SUITE, 0, expected)
        rows = protocol.read_jsonl(self.results_path)
        self.assertTrue(all(row["score"] == 1.0 for row in rows))
        self.assertEqual([row["id"] for row in rows], self.order)

    def test_metadata_reports_per_length_accuracy(self) -> None:
        adapter.command_run(self.args(), client=self.perfect_client())
        metadata = json.loads(
            (self.run_dir / "metadata" / "ruler-candidate-r0.json").read_text()
        )
        self.assertEqual(metadata["accuracy_by_length"], {"4k": 1.0})
        self.assertEqual(metadata["context_failures_by_length"], {"4k": 0})

    def test_timeout_rows_keep_their_length_label(self) -> None:
        def client(base_url, api_key, payload, timeout):
            raise socket.timeout("timed out")

        adapter.command_run(self.args(), client=client)
        rows = protocol.read_jsonl(self.results_path)
        self.assertTrue(all(row["timeout"] and row["length"] == "4k" for row in rows))
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

        with self.assertRaises(adapter.AdapterError):
            adapter.command_run(self.args(retries=3), client=client)
        self.assertEqual(len(attempts), 1)

    def test_concurrent_results_keep_task_order(self) -> None:
        adapter.command_run(self.args(concurrency=3), client=self.perfect_client())
        rows = protocol.read_jsonl(self.results_path)
        self.assertEqual([row["id"] for row in rows], self.order)

    def test_answer_instruction_is_appended_at_run_time(self) -> None:
        calls: list = []
        adapter.command_run(self.args(), client=self.perfect_client(calls))
        for payload in calls:
            self.assertIn(adapter.ANSWER_INSTRUCTION, payload["messages"][0]["content"])
        for prompt in self.prompts:
            self.assertNotIn(adapter.ANSWER_INSTRUCTION, prompt["text"])


if __name__ == "__main__":
    unittest.main()
