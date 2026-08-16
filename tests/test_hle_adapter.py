import base64
import importlib.util
import json
import os
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


class FakeJudge:
    """Records what it was asked and rules however the test says."""

    def __init__(self, correct=True):
        self.correct = correct
        self.seen = []

    def verdict(self, item_id, question, reference, submitted):
        self.seen.append({"item_id": item_id, "question": question,
                          "reference": reference, "submitted": submitted})
        return {"correct": self.correct, "reply": "correct: yes", "cached": False}


class ScoringTests(unittest.TestCase):
    def entry(self, row=EXACT):
        with tempfile.TemporaryDirectory() as tmp:
            return adapter.materialize([row], Path(tmp))[1][row["id"]]

    def score(self, content, row=EXACT, finish="stop", judge=None):
        return adapter.score_response(
            row["id"], completion(content, finish=finish), entry=self.entry(row),
            replicate=0, thinking=True, judge=judge or FakeJudge(),
        )

    def test_exact_answer_never_reaches_the_judge(self) -> None:
        judge = FakeJudge(correct=False)
        result = self.score("Working.\nAnswer: 18", judge=judge)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["graded_by"], "exact")
        self.assertEqual(judge.seen, [])

    def test_multiple_choice_letter(self) -> None:
        result = self.score("Answer: D", row=CHOICE)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["graded_by"], "exact")

    def test_a_named_option_is_read_off_without_a_judge(self) -> None:
        judge = FakeJudge(correct=False)
        result = self.score("Answer: D. w", row=CHOICE, judge=judge)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["graded_by"], "option-letter")
        self.assertEqual(judge.seen, [])

    def test_a_wrong_option_letter_needs_no_judge_either(self) -> None:
        judge = FakeJudge(correct=True)
        result = self.score("Answer: B", row=CHOICE, judge=judge)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["graded_by"], "option-letter")
        self.assertEqual(judge.seen, [])

    def test_a_mismatch_goes_to_the_judge(self) -> None:
        judge = FakeJudge(correct=True)
        result = self.score("Answer: eighteen", judge=judge)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["graded_by"], "judge")
        self.assertEqual(len(judge.seen), 1)
        self.assertEqual(judge.seen[0]["submitted"], "eighteen")
        self.assertEqual(judge.seen[0]["reference"], "18")
        self.assertEqual(judge.seen[0]["question"], EXACT["question"])

    def test_the_judge_can_rule_against(self) -> None:
        result = self.score("Answer: 19", judge=FakeJudge(correct=False))
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["graded_by"], "judge")

    def test_the_judge_is_not_told_which_arm_it_is_grading(self) -> None:
        judge = FakeJudge()
        self.score("Answer: eighteen", judge=judge)
        self.assertEqual(set(judge.seen[0]),
                         {"item_id", "question", "reference", "submitted"})

    def test_an_absent_answer_is_wrong_without_a_judge_call(self) -> None:
        judge = FakeJudge(correct=True)
        result = adapter.score_response(
            "hle_1", completion(""), entry=self.entry(), replicate=0,
            thinking=True, judge=judge,
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["graded_by"], "no-answer")
        self.assertEqual(judge.seen, [])

    def test_case_and_trailing_punctuation_folded(self) -> None:
        self.assertEqual(self.score("Answer: d.", row=CHOICE)["score"], 1.0)

    def test_last_answer_line_wins(self) -> None:
        self.assertEqual(self.score("Answer: 3\nrecheck\nAnswer: 18")["score"], 1.0)

    def test_truncation_is_a_context_failure(self) -> None:
        self.assertTrue(self.score("Answer: 18", finish="length")["context_failure"])

    def test_no_answer_line_falls_back_to_the_last_line(self) -> None:
        self.assertEqual(self.score("the total is\n18")["score"], 1.0)

    def test_grading_without_a_configured_judge_is_refused(self) -> None:
        # Silently scoring zero here would read as a model failure.
        with self.assertRaises(adapter.JudgeError):
            adapter.score_response(
                "hle_1", completion("Answer: eighteen"), entry=self.entry(),
                replicate=0, thinking=True, judge=None,
            )


class JudgeTests(unittest.TestCase):
    def build(self, tmp, replies, **over):
        self.sent = []

        def client(base_url, api_key, payload, timeout):
            self.sent.append(payload)
            return completion(replies[min(len(self.sent) - 1, len(replies) - 1)])

        kwargs = dict(
            base_url="http://judge/v1", api_key="EMPTY", model="openai/gpt-oss-20b",
            pin="hf:openai/gpt-oss-20b@" + "b" * 40,
            cache=adapter.JudgeCache(Path(tmp) / "judgements" / "hle.jsonl"),
            max_tokens=256, timeout=30.0, retries=0, client=client,
        )
        kwargs.update(over)
        return adapter.Judge(**kwargs)

    def test_a_yes_verdict_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["reasoning: same value.\ncorrect: yes"])
            ruling = judge.verdict("hle_1", "How many?", "18", "eighteen")
            self.assertTrue(ruling["correct"])
            self.assertFalse(ruling["cached"])

    def test_a_no_verdict_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["reasoning: different.\ncorrect: no"])
            self.assertFalse(judge.verdict("hle_1", "q", "18", "19")["correct"])

    def test_the_judge_is_greedy_and_seeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["correct: yes"])
            judge.verdict("hle_1", "q", "18", "eighteen")
            self.assertEqual(self.sent[0]["temperature"], 0.0)
            self.assertEqual(self.sent[0]["seed"], 0)

    def test_the_judge_prompt_carries_all_three_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["correct: yes"])
            judge.verdict("hle_1", "How many stars?", "18", "eighteen")
            prompt = self.sent[0]["messages"][0]["content"]
            for part in ("How many stars?", "18", "eighteen"):
                self.assertIn(part, prompt)
            self.assertNotIn("candidate", prompt.lower())
            self.assertNotIn("baseline", prompt.lower())

    def test_an_unparseable_reply_is_refused_rather_than_scored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["I think it is probably fine"])
            with self.assertRaises(adapter.JudgeError):
                judge.verdict("hle_1", "q", "18", "eighteen")

    def test_the_last_verdict_line_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["correct: no\nrechecking\ncorrect: yes"])
            self.assertTrue(judge.verdict("hle_1", "q", "18", "eighteen")["correct"])

    def test_the_same_answer_is_judged_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["correct: yes"])
            first = judge.verdict("hle_1", "q", "18", "eighteen")
            second = judge.verdict("hle_1", "q", "18", "Eighteen.")
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(second["correct"], first["correct"])
            self.assertEqual(len(self.sent), 1)
            self.assertEqual(judge.calls, 1)
            self.assertEqual(judge.hits, 1)

    def test_a_different_item_with_the_same_string_is_judged_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["correct: yes"])
            judge.verdict("hle_1", "q", "18", "eighteen")
            judge.verdict("hle_2", "q", "18", "eighteen")
            self.assertEqual(len(self.sent), 2)

    def test_a_verdict_survives_a_new_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp, ["correct: yes"]).verdict("hle_1", "q", "18", "eighteen")
            reopened = self.build(tmp, ["correct: no"])
            ruling = reopened.verdict("hle_1", "q", "18", "eighteen")
            self.assertTrue(ruling["correct"])
            self.assertTrue(ruling["cached"])
            self.assertEqual(self.sent, [])

    def test_a_judge_timeout_raises_rather_than_scoring_zero(self) -> None:
        def timing_out(base_url, api_key, payload, timeout):
            raise TimeoutError("judge is gone")

        with tempfile.TemporaryDirectory() as tmp:
            judge = self.build(tmp, ["correct: yes"], client=timing_out)
            with self.assertRaises(adapter.JudgeError):
                judge.verdict("hle_1", "q", "18", "eighteen")


class DeferredTests(unittest.TestCase):
    """Generation and judging split, so the judge never contends for a card."""

    JUDGE_PIN = "hf:openai/gpt-oss-20b@" + "b" * 40

    def entry(self, row=EXACT):
        with tempfile.TemporaryDirectory() as tmp:
            return adapter.materialize([row], Path(tmp))[1][row["id"]]

    def defer(self, content, row=EXACT):
        return adapter.score_response(
            row["id"], completion(content), entry=self.entry(row),
            replicate=0, thinking=True, judge=None, defer=True,
        )

    def test_a_mismatch_is_left_open_rather_than_scored(self) -> None:
        deferred = self.defer("Answer: eighteen")
        self.assertTrue(deferred["deferred"])
        self.assertEqual(deferred["graded_by"], "deferred")
        # Untruncated, because this is the string the judge will rule on.
        self.assertEqual(deferred["submitted"], "eighteen")

    def test_what_can_be_settled_mechanically_still_is(self) -> None:
        for content, expected in (("Answer: 18", "exact"), ("", "no-answer")):
            with self.subTest(content=content):
                row = self.defer(content)
                self.assertFalse(row["deferred"])
                self.assertEqual(row["graded_by"], expected)

    def write_run(self, tmp, rows, judge=None):
        run = Path(tmp)
        generations = run / "hle-candidate-r0.jsonl"
        with generations.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        (run / "hle-candidate-r0.meta.json").write_text(json.dumps(
            {"suite": "hle", "variant": "candidate", "replicate": 0,
             "judge": self.JUDGE_PIN if judge is None else judge}))
        key = run / "hle.key.json"
        with tempfile.TemporaryDirectory() as inner:
            _, items = adapter.materialize([EXACT, CHOICE], Path(inner))
        key.write_text(json.dumps({"dataset": "cais/hle@x", "items": items}))
        return generations, key

    def score(self, tmp, rows, replies=("reasoning: same.\ncorrect: yes",), judge=None):
        sent = []

        def client(base_url, api_key, payload, timeout):
            sent.append(payload)
            return completion(replies[min(len(sent) - 1, len(replies) - 1)])

        generations, key = self.write_run(tmp, rows, judge=judge)
        results = Path(tmp) / "results.jsonl"
        metadata = Path(tmp) / "meta.json"
        args = adapter.parse_args([
            "score", "--generations", str(generations), "--key", str(key),
            "--results", str(results), "--metadata", str(metadata),
        ])
        os.environ["EVAL_JUDGE_BASE_URL"] = "http://judge/v1"
        try:
            adapter.command_score(args, client=client)
        finally:
            os.environ.pop("EVAL_JUDGE_BASE_URL", None)
        rows_out = [json.loads(line) for line in results.read_text().splitlines()]
        return rows_out, json.loads(metadata.read_text()), sent

    def test_a_deferred_row_is_judged_and_a_settled_one_is_not(self) -> None:
        rows = [
            {"id": "hle_1", "score": 0.0, "graded_by": "deferred", "deferred": True,
             "submitted": "eighteen", "replicate": 0},
            {"id": "hle_2", "score": 1.0, "graded_by": "exact", "deferred": False,
             "replicate": 0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out, meta, sent = self.score(tmp, rows)
            by_id = {row["id"]: row for row in out}
            self.assertEqual(by_id["hle_1"]["score"], 1.0)
            self.assertEqual(by_id["hle_1"]["graded_by"], "judge")
            self.assertNotIn("submitted", by_id["hle_1"])
            self.assertEqual(by_id["hle_2"]["graded_by"], "exact")
            self.assertEqual(len(sent), 1)
            self.assertFalse(any(row["deferred"] for row in out))
            self.assertEqual(meta["judge"], self.JUDGE_PIN)
            self.assertEqual(meta["deferred"], False)

    def test_both_arms_share_one_verdict_for_one_string(self) -> None:
        rows = [
            {"id": "hle_1", "score": 0.0, "graded_by": "deferred", "deferred": True,
             "submitted": "eighteen", "replicate": 0},
            {"id": "hle_1", "score": 0.0, "graded_by": "deferred", "deferred": True,
             "submitted": "Eighteen", "replicate": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out, meta, sent = self.score(tmp, rows)
            self.assertEqual(len(sent), 1, "the same answer was judged twice")
            self.assertEqual({row["score"] for row in out}, {1.0})
            self.assertEqual(meta["judge_calls"], 1)
            self.assertEqual(meta["judge_cache_hits"], 1)

    def test_an_unpinned_judge_in_the_metadata_is_refused(self) -> None:
        rows = [{"id": "hle_1", "score": 0.0, "graded_by": "deferred",
                 "deferred": True, "submitted": "eighteen", "replicate": 0}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(adapter.JudgeError):
                self.score(tmp, rows, judge="openai/gpt-oss-20b")


class PinTests(unittest.TestCase):
    def base(self, **over):
        pins = {"dataset": "a" * 40, "judge": "hf:openai/gpt-oss-20b@" + "b" * 40,
                "harness": adapter.HARNESS_ID, "verifier": adapter.VERIFIER_ID,
                "adapter": adapter.self_pin()}
        pins.update(over)
        return pins

    def test_resolved_commit_passes(self) -> None:
        adapter.validate_pins(self.base())

    def test_branch_is_not_a_pin(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(self.base(dataset="main"))

    def test_a_served_judge_and_a_hosted_one_both_pin(self) -> None:
        for value, scheme, model in (
            ("hf:openai/gpt-oss-20b@" + "b" * 40, "hf", "openai/gpt-oss-20b"),
            ("api:anthropic/claude-opus-5@2026-08-16", "api", "claude-opus-5"),
        ):
            with self.subTest(judge=value):
                adapter.validate_pins(self.base(judge=value))
                parts = adapter.judge_pin_parts(value)
                self.assertEqual(parts["scheme"], scheme)
                self.assertEqual(parts["model"], model)

    def test_an_unpinned_judge_is_refused(self) -> None:
        for value in ("", "openai/gpt-oss-20b", "hf:openai/gpt-oss-20b@main",
                      "b" * 40, "api:anthropic/claude-opus-5"):
            with self.subTest(judge=value):
                with self.assertRaises(adapter.AdapterError):
                    adapter.validate_pins(self.base(judge=value))

    def test_a_moving_alias_is_refused(self) -> None:
        # It can resolve to different weights for the two arms of one run.
        for value in ("api:anthropic/claude-opus-5-latest@2026-08-16",
                      "api:openai/some-model@latest",
                      "api:anthropic/claude-opus-5@preview"):
            with self.subTest(judge=value):
                with self.assertRaises(adapter.AdapterError):
                    adapter.validate_pins(self.base(judge=value))

    def test_a_hosted_judge_is_not_sent_vllm_only_fields(self) -> None:
        sent = []

        def client(base_url, api_key, payload, timeout):
            sent.append(payload)
            return completion("correct: yes")

        with tempfile.TemporaryDirectory() as tmp:
            judge = adapter.Judge(
                base_url="https://api.example/v1", api_key="k",
                model="claude-opus-5", pin="api:anthropic/claude-opus-5@2026-08-16",
                scheme="api", cache=adapter.JudgeCache(Path(tmp) / "j.jsonl"),
                max_tokens=256, timeout=30.0, retries=0, client=client,
            )
            judge.verdict("hle_1", "q", "18", "eighteen")
        self.assertEqual(sent[0]["temperature"], 0.0)
        for field in ("seed", "top_p", "chat_template_kwargs"):
            self.assertNotIn(field, sent[0])

    def test_a_judge_that_is_not_the_pinned_one_is_refused(self) -> None:
        import os
        args = adapter.parse_args(["run"])
        env = {"EVAL_JUDGE_MODEL": "Qwen/Qwen3-32B", "EVAL_JUDGE_BASE_URL": "http://x/v1"}
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(adapter.JudgeError):
                    adapter.build_judge(self.base(), Path(tmp), args, lambda *a: {})
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


if __name__ == "__main__":
    unittest.main()
