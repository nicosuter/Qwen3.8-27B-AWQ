import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "scripts" / "adapters"))


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("bfcl", "eval/scripts/adapters/bfcl.py")
protocol = load_module("run_eval_protocol", "eval/scripts/run_eval_protocol.py")

SIMPLE_ROW = {
    "id": "simple_0",
    "question": [[{"role": "user", "content": "Find the area of a triangle, base 10, height 5."}]],
    "function": [
        {
            "name": "calculate_triangle_area",
            "description": "Area of a triangle.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer", "description": "base"},
                    "height": {"type": "integer", "description": "height"},
                    "unit": {"type": "string", "description": "unit"},
                },
                "required": ["base", "height"],
            },
        }
    ],
}
SIMPLE_ANSWER = {
    "id": "simple_0",
    "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}],
}
PARALLEL_ROW = {
    "id": "parallel_0",
    "question": [[{"role": "user", "content": "Play Taylor Swift for 20 and Maroon 5 for 15."}]],
    "function": [
        {
            "name": "spotify.play",
            "description": "Play music.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "artist": {"type": "string", "description": "artist"},
                    "duration": {"type": "integer", "description": "minutes"},
                },
                "required": ["artist", "duration"],
            },
        }
    ],
}
PARALLEL_ANSWER = {
    "id": "parallel_0",
    "ground_truth": [
        {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
        {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
    ],
}
IRRELEVANCE_ROW = {
    "id": "irrelevance_0",
    "question": [[{"role": "user", "content": "What is the meaning of life?"}]],
    "function": [
        {
            "name": "get_weather",
            "description": "Weather.",
            "parameters": {"type": "dict", "properties": {"city": {"type": "string"}}},
        }
    ],
}


def build_key():
    _, key, _ = adapter.materialize(
        {"simple": [SIMPLE_ROW], "parallel": [PARALLEL_ROW], "irrelevance": [IRRELEVANCE_ROW]},
        {"simple_0": SIMPLE_ANSWER, "parallel_0": PARALLEL_ANSWER},
    )
    return key


def tool_call(name, arguments, as_string=True):
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if as_string else arguments,
        },
    }


def completion(calls=None, content="", *, finish="stop", reasoning=""):
    message = {"content": content, "reasoning_content": reasoning}
    if calls is not None:
        message["tool_calls"] = calls
    return {
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": {"completion_tokens": 300},
    }


class SchemaTests(unittest.TestCase):
    def test_bfcl_types_become_json_schema_types(self) -> None:
        converted = adapter.convert_schema(
            {"type": "dict", "properties": {
                "a": {"type": "float"}, "b": {"type": "tuple", "items": {"type": "integer"}}}}
        )
        self.assertEqual(converted["type"], "object")
        self.assertEqual(converted["properties"]["a"]["type"], "number")
        self.assertEqual(converted["properties"]["b"]["type"], "array")

    def test_any_type_is_left_unconstrained(self) -> None:
        converted = adapter.convert_schema({"type": "any", "description": "whatever"})
        self.assertNotIn("type", converted)
        self.assertEqual(converted["description"], "whatever")

    def test_tools_are_openai_shaped(self) -> None:
        tools = adapter.build_tools(SIMPLE_ROW["function"])
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "calculate_triangle_area")
        self.assertEqual(tools[0]["function"]["parameters"]["type"], "object")

    def test_question_turns_are_flattened(self) -> None:
        self.assertIn("triangle", adapter.flatten_question(SIMPLE_ROW["question"]))
        with self.assertRaises(adapter.AdapterError):
            adapter.flatten_question([])


class MaterializeTests(unittest.TestCase):
    def test_ground_truth_attaches_and_irrelevance_has_none(self) -> None:
        key = build_key()
        self.assertEqual(len(key["parallel_0"]["ground_truth"]), 2)
        self.assertIsNone(key["irrelevance_0"]["ground_truth"])

    def test_an_unkeyed_item_is_dropped_and_named(self) -> None:
        # One upstream id is typo'd at the pinned revision. Pairing it to the
        # near-miss key would invent a correspondence, so it is dropped.
        prompts, key, dropped = adapter.materialize({"simple": [SIMPLE_ROW]}, {})
        self.assertEqual(prompts, [])
        self.assertEqual(key, {})
        self.assertEqual(dropped["unkeyed"], [str(SIMPLE_ROW["id"])])
        self.assertEqual(dropped["no_tools"], [])

    def test_many_unkeyed_items_are_fatal(self) -> None:
        # A handful is upstream raggedness; a flood means the answer files
        # changed shape, and scoring a silently smaller suite is the real risk.
        rows = [dict(SIMPLE_ROW, id=f"simple_{i}") for i in range(adapter.MAX_UNKEYED + 1)]
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize({"simple": rows}, {})

    def test_a_toolless_abstention_item_is_dropped_as_undiscriminating(self) -> None:
        # With no tools offered, calling nothing is automatic: both checkpoints
        # score 1.0 and the item cannot separate them.
        row = {"id": "live_irrelevance_0", "question": [[{"role": "user", "content": "hi"}]],
               "function": []}
        prompts, _, dropped = adapter.materialize({"live_irrelevance": [row]}, {})
        self.assertEqual(prompts, [])
        self.assertEqual(dropped["no_tools"], ["live_irrelevance_0"])

    def test_a_toolless_scored_item_is_still_fatal(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize({"simple": [dict(SIMPLE_ROW, function=[])]}, {})

    def test_a_decision_only_category_needs_no_key(self) -> None:
        for category in adapter.NO_GROUND_TRUTH:
            with self.subTest(category=category):
                self.assertNotIn(category, adapter.SCORED_WITH_GROUND_TRUTH)

    def test_live_relevance_is_the_mirror_of_irrelevance(self) -> None:
        call = [{"name": "f", "arguments": {}}]
        for category, calls, expected in (
            ("irrelevance", [], 1.0),
            ("irrelevance", call, 0.0),
            ("live_irrelevance", [], 1.0),
            ("live_irrelevance", call, 0.0),
            ("live_relevance", [], 0.0),
            ("live_relevance", call, 1.0),
        ):
            with self.subTest(category=category, called=bool(calls)):
                entry = {"category": category, "ground_truth": None}
                self.assertEqual(adapter.score_calls(calls, entry), expected)

    def test_silence_is_success_only_where_abstaining_is_correct(self) -> None:
        self.assertIn("live_irrelevance", adapter.ABSTENTION_IS_CORRECT)
        self.assertNotIn("live_relevance", adapter.ABSTENTION_IS_CORRECT)

    def test_a_colliding_id_is_disambiguated_and_both_items_kept(self) -> None:
        # live_relevance_3-3-0 is two different questions under one id upstream.
        # Dropping either would lose a real item.
        prompts, key, notes = adapter.materialize(
            {"simple": [SIMPLE_ROW], "multiple": [SIMPLE_ROW]},
            {"simple_0": SIMPLE_ANSWER},
        )
        self.assertEqual([p["id"] for p in prompts], ["simple_0", "simple_0#2"])
        self.assertEqual(notes["renamed"], ["simple_0"])
        # The key is looked up under the upstream id, so the renamed copy is
        # still scored rather than silently losing its ground truth.
        self.assertIsNotNone(key["simple_0#2"]["ground_truth"])

    def test_a_flood_of_colliding_ids_is_fatal(self) -> None:
        rows = [SIMPLE_ROW] * (adapter.MAX_UNKEYED + 2)
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize({"simple": rows}, {"simple_0": SIMPLE_ANSWER})

    def test_prompts_satisfy_the_runner(self) -> None:
        prompts, _, _ = adapter.materialize(
            {"simple": [SIMPLE_ROW]}, {"simple_0": SIMPLE_ANSWER}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bfcl.jsonl"
            adapter.write_jsonl(path, prompts)
            self.assertEqual(protocol.validate_prompts(path, adapter.SUITE), ["simple_0"])


class MatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = build_key()

    def score(self, item, calls):
        return adapter.score_calls(calls, self.key[item])

    def test_exact_match(self) -> None:
        self.assertEqual(
            self.score("simple_0", [{"name": "calculate_triangle_area",
                                     "arguments": {"base": 10, "height": 5}}]), 1.0)

    def test_string_and_number_forms_are_equivalent(self) -> None:
        self.assertEqual(
            self.score("simple_0", [{"name": "calculate_triangle_area",
                                     "arguments": {"base": "10", "height": 5}}]), 1.0)

    def test_optional_parameter_may_be_supplied_or_omitted(self) -> None:
        # "" among the acceptable values means the parameter is optional.
        self.assertEqual(
            self.score("simple_0", [{"name": "calculate_triangle_area",
                                     "arguments": {"base": 10, "height": 5, "unit": "units"}}]), 1.0)

    def test_wrong_value_fails(self) -> None:
        self.assertEqual(
            self.score("simple_0", [{"name": "calculate_triangle_area",
                                     "arguments": {"base": 10, "height": 6}}]), 0.0)

    def test_hallucinated_parameter_fails(self) -> None:
        self.assertEqual(
            self.score("simple_0", [{"name": "calculate_triangle_area",
                                     "arguments": {"base": 10, "height": 5, "colour": "red"}}]), 0.0)

    def test_wrong_function_fails(self) -> None:
        self.assertEqual(
            self.score("simple_0", [{"name": "other", "arguments": {"base": 10, "height": 5}}]), 0.0)

    def test_parallel_calls_match_in_any_order(self) -> None:
        calls = [
            {"name": "spotify.play", "arguments": {"artist": "Maroon 5", "duration": 15}},
            {"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}},
        ]
        self.assertEqual(self.score("parallel_0", calls), 1.0)

    def test_parallel_requires_every_call(self) -> None:
        calls = [{"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}}]
        self.assertEqual(self.score("parallel_0", calls), 0.0)

    def test_extra_call_fails(self) -> None:
        calls = [
            {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}},
            {"name": "calculate_triangle_area", "arguments": {"base": 1, "height": 1}},
        ]
        self.assertEqual(self.score("simple_0", calls), 0.0)

    def test_case_and_punctuation_insensitive_strings(self) -> None:
        calls = [
            {"name": "spotify.play", "arguments": {"artist": "taylor swift", "duration": 20}},
            {"name": "spotify.play", "arguments": {"artist": "MAROON-5", "duration": 15}},
        ]
        self.assertEqual(self.score("parallel_0", calls), 1.0)

    def test_irrelevance_rewards_calling_nothing(self) -> None:
        self.assertEqual(self.score("irrelevance_0", []), 1.0)
        self.assertEqual(
            self.score("irrelevance_0", [{"name": "get_weather", "arguments": {"city": "Zurich"}}]),
            0.0,
        )


class ExtractionTests(unittest.TestCase):
    def test_string_arguments_are_parsed(self) -> None:
        calls, malformed = adapter.extract_calls(
            {"tool_calls": [tool_call("f", {"a": 1})]}
        )
        self.assertEqual(calls, [{"name": "f", "arguments": {"a": 1}}])
        self.assertFalse(malformed)

    def test_unparsable_arguments_are_malformed(self) -> None:
        broken = {"id": "1", "type": "function",
                  "function": {"name": "f", "arguments": "{not json"}}
        calls, malformed = adapter.extract_calls({"tool_calls": [broken]})
        self.assertEqual(calls, [])
        self.assertTrue(malformed)

    def test_missing_name_is_malformed(self) -> None:
        broken = {"function": {"arguments": "{}"}}
        _, malformed = adapter.extract_calls({"tool_calls": [broken]})
        self.assertTrue(malformed)

    def test_no_tool_calls(self) -> None:
        self.assertEqual(adapter.extract_calls({"content": "hi"}), ([], False))


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = build_key()

    def score(self, item, response):
        return adapter.score_response(item, response, entry=self.key[item],
                                      replicate=0, thinking=True)

    def test_correct_call_scores_one(self) -> None:
        row = self.score("simple_0", completion(
            [tool_call("calculate_triangle_area", {"base": 10, "height": 5})]))
        self.assertEqual(row["score"], 1.0)
        self.assertFalse(row["malformed_tool_call"])

    def test_malformed_call_is_flagged_and_scored_zero(self) -> None:
        broken = {"function": {"name": "calculate_triangle_area", "arguments": "{oops"}}
        row = self.score("simple_0", completion([broken]))
        self.assertTrue(row["malformed_tool_call"])
        self.assertEqual(row["score"], 0.0)

    def test_silence_on_irrelevance_is_not_an_empty_answer(self) -> None:
        row = self.score("irrelevance_0", completion([], content=""))
        self.assertEqual(row["score"], 1.0)
        self.assertFalse(row["empty_answer"])

    def test_rows_satisfy_the_runner_contract(self) -> None:
        row = self.score("simple_0", completion(
            [tool_call("calculate_triangle_area", {"base": 10, "height": 5})]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            adapter.write_jsonl(path, [row])
            protocol.validate_results(path, adapter.SUITE, 0, {"simple_0"})

    def test_tools_are_attached_to_the_request(self) -> None:
        seen = []

        def client(base_url, api_key, payload, timeout):
            seen.append(payload)
            return completion([tool_call("calculate_triangle_area", {"base": 10, "height": 5})])

        generation = {"enable_thinking": True, "reasoning_effort": "xhigh", "temperature": 1.0,
                      "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                      "repetition_penalty": 1.0}
        args = SimpleNamespace(max_tokens=1024, request_timeout=30.0, retries=0)
        with tempfile.TemporaryDirectory() as tmp:
            row = adapter.run_item(
                "simple_0", "question", self.key["simple_0"], generation=generation,
                model="m", seed=1, replicate=0, variant="candidate", run_dir=Path(tmp),
                base_url="http://x/v1", api_key="EMPTY", args=args, client=client,
            )
        self.assertEqual(seen[0]["tool_choice"], "auto")
        self.assertEqual(seen[0]["tools"][0]["function"]["name"], "calculate_triangle_area")
        self.assertEqual(row["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
