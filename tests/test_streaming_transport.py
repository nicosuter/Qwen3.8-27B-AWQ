"""The transport streams, so the request timeout stops measuring contention.

A non-streaming completion sends nothing until the whole answer is ready, so a
socket timeout on it is a total-duration timeout by accident: it fires when the
server is busy, not when the model misbehaves. Measured on one run, filling the
131072-token cap took 132 minutes per stream under load against a 90-minute
timeout, so the item most likely to be censored was the one generating the most
tokens -- the hardest item, on the axis the two arms differ on.

Streaming gives each failure its own detector. Degenerate output hits max_tokens
and returns `finish_reason: length`, which is an observation both arms get under
identical rules. A server that has actually stopped produces no chunk, and the
socket deadline -- now between chunks rather than across the whole request --
fires on that alone. Contention just takes longer and is no longer an outcome.

The cost of streaming is memory: an SSE envelope is ~200 bytes carrying a ~4
byte token, so retaining raw chunks inflates a full-length answer from ~500KB to
~26MB, times the lane concurrency. Hence the retention test below, which is a
correctness test and not a benchmark.
"""

import importlib.util
import json
import sys
import tracemalloc
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "_common", ROOT / "eval" / "scripts" / "adapters" / "_common.py"
)
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)


def sse(*chunks: dict) -> list[bytes]:
    """Render chunks as the wire format, one `data:` line each, then [DONE]."""
    lines = [b"data: " + json.dumps(chunk).encode() + b"\n" for chunk in chunks]
    return lines + [b"data: [DONE]\n"]


def delta(content=None, reasoning=None, finish_reason=None, reasoning_key="reasoning_content") -> dict:
    body: dict = {}
    if content is not None:
        body["content"] = content
    if reasoning is not None:
        body[reasoning_key] = reasoning
    return {"choices": [{"delta": body, "finish_reason": finish_reason}]}


def usage_chunk(prompt=7, completion=11) -> dict:
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


class CollectStreamTests(unittest.TestCase):
    def test_deltas_join_into_one_message(self):
        stream = sse(
            delta(content="Hel"),
            delta(content="lo "),
            delta(content="world", finish_reason="stop"),
            usage_chunk(),
        )
        response = common.collect_stream(stream, item_id="a")
        content, reasoning, finish, usage = common.unpack_choice("a", response)
        self.assertEqual(content, "Hello world")
        self.assertEqual(reasoning, "")
        self.assertEqual(finish, "stop")
        self.assertEqual(usage["completion_tokens"], 11)

    def test_reasoning_and_content_stay_separate(self):
        """`enable_thinking` is on, and the comparator scores them differently."""
        stream = sse(
            delta(reasoning="think "),
            delta(reasoning="harder"),
            delta(content="42", finish_reason="stop"),
            usage_chunk(),
        )
        response = common.collect_stream(stream, item_id="a")
        content, reasoning, _, _ = common.unpack_choice("a", response)
        self.assertEqual(reasoning, "think harder")
        self.assertEqual(content, "42")

    def test_the_shape_matches_a_buffered_response(self):
        """Downstream reads `choices[0].message`, so streaming must rebuild it.

        Keeping the shape is what confines this change to the transport: every
        adapter, the raw-response archive and the scorers stay untouched.
        """
        response = common.collect_stream(
            sse(delta(content="x", finish_reason="stop"), usage_chunk()), item_id="a"
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "x")
        self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")

    def test_length_finish_reason_survives(self):
        """Hitting the cap is the degeneracy signal, so it must reach the row."""
        response = common.collect_stream(
            sse(delta(content="x", finish_reason="length"), usage_chunk()), item_id="a"
        )
        self.assertEqual(response["choices"][0]["finish_reason"], "length")

    def test_keepalives_and_comments_are_ignored(self):
        stream = [
            b"\n",
            b": ping\n",
            b"data: " + json.dumps(delta(content="a", finish_reason="stop")).encode() + b"\n",
            b"\n",
            b"data: " + json.dumps(usage_chunk()).encode() + b"\n",
            b"data: [DONE]\n",
        ]
        response = common.collect_stream(stream, item_id="a")
        self.assertEqual(response["choices"][0]["message"]["content"], "a")

    def test_usage_is_required_because_admission_is_sized_from_it(self):
        """No usage means no token counts, and the priors are built from those.

        Silently defaulting would shrink every reservation to the fallback and
        put the cache straight back into the preemption storm this run spent
        five hours climbing out of -- with nothing in the output to say so.
        """
        stream = sse(delta(content="x", finish_reason="stop"))
        with self.assertRaises(common.AdapterError) as caught:
            common.collect_stream(stream, item_id="item-9")
        self.assertIn("item-9", str(caught.exception))
        self.assertIn("usage", str(caught.exception))

    def test_a_truncated_stream_is_an_error_not_a_short_answer(self):
        """A connection cut mid-answer must not read as a finished one.

        This is the failure streaming introduces that buffering could not have:
        a partial body used to be a JSON parse error, and now it is a
        well-formed prefix. Scored, it would be a short answer the model never
        gave; raised, it is a transport fault the retry already handles.
        """
        chunks = [
            b"data: " + json.dumps(delta(content="half an ans")).encode() + b"\n",
        ]
        with self.assertRaises(common.AdapterError) as caught:
            common.collect_stream(chunks, item_id="item-3")
        self.assertIn("item-3", str(caught.exception))

    def test_an_empty_stream_is_an_error(self):
        with self.assertRaises(common.AdapterError):
            common.collect_stream([], item_id="a")

    def test_retained_memory_tracks_text_not_chunk_count(self):
        """Hold the text, never the envelopes.

        40,000 one-character tokens is ~40KB of answer inside ~2.9MB of wire
        format, and a real envelope is fatter than this fixture's. Retaining the
        parsed chunks would show up here as two orders of magnitude; the bound is
        deliberately loose enough that it only fails if the accumulator starts
        keeping per-chunk objects.
        """
        tokens = 40_000
        stream = sse(
            *[delta(content="x") for _ in range(tokens)],
            delta(content="", finish_reason="stop"),
            usage_chunk(),
        )
        wire = sum(len(line) for line in stream)
        self.assertGreater(wire, 2_000_000, "the fixture must be big enough to matter")

        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        response = common.collect_stream(iter(stream), item_id="a")
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        self.assertEqual(len(response["choices"][0]["message"]["content"]), tokens)
        self.assertLess(
            peak - before,
            20 * tokens,
            "the collector is retaining chunk objects, not just the text",
        )


class ReasoningFieldNameTests(unittest.TestCase):
    """The server names the thinking text `reasoning`, not `reasoning_content`.

    Every fixture in this repository was written against `reasoning_content`, so
    the suite stayed green while the harness discarded the thinking text of
    every item of every suite. What it actually sends, from a scored run's own
    archive:

        message keys = ['annotations', 'audio', 'content', 'function_call',
                        'reasoning', 'refusal', 'role']

    Reading one name and being sent the other is silent by construction: a reply
    that finished still carries its answer in `content`, so only the reasoning
    goes missing. A reply that ran to the token cap never leaves the think
    block, so all of it goes missing and the row arrives empty.
    """

    def test_buffered_reasoning_is_read(self):
        response = {
            "choices": [
                {
                    "message": {"content": "42", "reasoning": "think harder"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 11},
        }
        content, reasoning, _, _ = common.unpack_choice("a", response)
        self.assertEqual(reasoning, "think harder")
        self.assertEqual(content, "42")

    def test_streamed_reasoning_is_read(self):
        stream = sse(
            delta(reasoning="think ", reasoning_key="reasoning"),
            delta(reasoning="harder", reasoning_key="reasoning"),
            delta(content="42", finish_reason="stop"),
            usage_chunk(),
        )
        response = common.collect_stream(stream, item_id="a")
        content, reasoning, _, _ = common.unpack_choice("a", response)
        self.assertEqual(reasoning, "think harder")
        self.assertEqual(content, "42")

    def test_a_reply_truncated_inside_the_think_block_keeps_its_text(self):
        """The production failure, reduced.

        A RULER item that ran to the cap generated 131072 tokens and reached the
        scorer as `content=''`, `reasoning_content=''` -- so `repetition_loop`
        was computed over an empty string and `repetition_assessed` was False on
        every truncated row. The whole point of recording truncation is to be
        able to look at what the model was doing when it happened.
        """
        stream = sse(
            delta(reasoning="counting: item001, ", reasoning_key="reasoning"),
            delta(reasoning="item002, ", reasoning_key="reasoning", finish_reason="length"),
            usage_chunk(),
        )
        response = common.collect_stream(stream, item_id="a")
        content, reasoning, finish, _ = common.unpack_choice("a", response)
        self.assertEqual(finish, "length")
        self.assertEqual(content, "")
        self.assertEqual(reasoning, "counting: item001, item002, ")

    def test_either_name_is_accepted(self):
        """Both spellings are in the archive; neither may start being ignored.

        The buffered runs of this protocol carry `reasoning` and the fixtures
        carry `reasoning_content`, and a rescored run directory mixes the two.
        """
        for key in ("reasoning", "reasoning_content"):
            with self.subTest(key=key):
                response = {
                    "choices": [{"message": {"content": "", key: "t"}, "finish_reason": "stop"}],
                    "usage": {},
                }
                _, reasoning, _, _ = common.unpack_choice("a", response)
                self.assertEqual(reasoning, "t")


def tool_delta(index=0, call_id=None, name=None, arguments=None, finish_reason=None) -> dict:
    """One `delta.tool_calls` fragment, the way a server streams a call.

    The id and name arrive once, on the fragment that opens the call; the
    arguments arrive as a JSON string split across any number of later
    fragments, tied together only by `index`.
    """
    call: dict = {"index": index}
    if call_id is not None:
        call["id"] = call_id
        call["type"] = "function"
    function: dict = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    if function:
        call["function"] = function
    return {"choices": [{"delta": {"tool_calls": [call]}, "finish_reason": finish_reason}]}


class StreamedToolCallTests(unittest.TestCase):
    """BFCL scores `message.tool_calls`, and the stream was dropping them.

    `collect_stream` rebuilt the message from a fixed list of keys, so anything
    the server sent outside that list vanished. On a streamed BFCL run 2193 of
    3486 items came back `empty_answer` against 0 on a buffered run of the same
    items -- read as a checkpoint scoring 32.07 against a baseline of 81.61.
    """

    def test_a_call_split_across_fragments_is_reassembled(self):
        stream = sse(
            tool_delta(call_id="call_1", name="get_weather", arguments=""),
            tool_delta(arguments='{"city": '),
            tool_delta(arguments='"Zurich"}', finish_reason="tool_calls"),
            usage_chunk(),
        )
        response = common.collect_stream(stream, item_id="a")
        calls = response["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_1")
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Zurich"})

    def test_parallel_calls_stay_separate_and_ordered(self):
        """`index` is the only thing tying a fragment to its call."""
        stream = sse(
            tool_delta(index=0, call_id="a", name="first", arguments='{"x":'),
            tool_delta(index=1, call_id="b", name="second", arguments='{"y":'),
            tool_delta(index=1, arguments=" 2}"),
            tool_delta(index=0, arguments=" 1}", finish_reason="tool_calls"),
            usage_chunk(),
        )
        response = common.collect_stream(stream, item_id="a")
        calls = response["choices"][0]["message"]["tool_calls"]
        self.assertEqual([c["function"]["name"] for c in calls], ["first", "second"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"x": 1})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"y": 2})

    def test_the_scorer_can_read_what_the_stream_rebuilt(self):
        """The shape is only right if the adapter that consumes it agrees."""
        bfcl_spec = importlib.util.spec_from_file_location(
            "_bfcl", ROOT / "eval" / "scripts" / "adapters" / "bfcl.py"
        )
        bfcl = importlib.util.module_from_spec(bfcl_spec)
        sys.modules["_bfcl"] = bfcl
        bfcl_spec.loader.exec_module(bfcl)

        response = common.collect_stream(
            sse(
                tool_delta(call_id="call_1", name="get_weather", arguments='{"city":'),
                tool_delta(arguments='"Zurich"}', finish_reason="tool_calls"),
                usage_chunk(),
            ),
            item_id="a",
        )
        calls, malformed = bfcl.extract_calls(response["choices"][0]["message"])
        self.assertFalse(malformed)
        self.assertEqual(calls, [{"name": "get_weather", "arguments": {"city": "Zurich"}}])

    def test_a_reply_without_tool_calls_has_no_tool_calls(self):
        """Every other suite reads `content`; none of them should grow a key."""
        response = common.collect_stream(
            sse(delta(content="42", finish_reason="stop"), usage_chunk()), item_id="a"
        )
        self.assertNotIn("tool_calls", response["choices"][0]["message"])


class ReasoningTokenEstimateTests(unittest.TestCase):
    """The estimate must not change units when the text arrives.

    `reasoning_tokens_median` is compared between arms, so the estimator has to
    mean the same thing in both. It used to fall back to `completion_tokens`
    minus the visible answer, because the text was never there; once the text is
    read, a word count would answer the same question in different units and
    every rescored row would move by roughly the tokens-per-word ratio while
    nothing about the run had changed.
    """

    def test_the_estimate_is_the_same_with_and_without_the_text(self):
        usage = {"completion_tokens": 1000}
        reasoning = "word " * 700
        without = common.reasoning_tokens(usage, "", "answer")
        with_text = common.reasoning_tokens(usage, reasoning, "answer")
        self.assertEqual(with_text, without)

    def test_a_reported_count_still_wins(self):
        """When the server counts them itself, nothing here should guess."""
        usage = {"completion_tokens": 1000, "completion_tokens_details": {"reasoning_tokens": 640}}
        self.assertEqual(common.reasoning_tokens(usage, "word " * 700, "answer"), 640)

    def test_text_alone_is_still_measurable(self):
        """Rows exist whose usage was not retained; they must not report zero."""
        self.assertGreater(common.reasoning_tokens({}, "word " * 700), 0)


class StreamRequestTests(unittest.TestCase):
    def test_the_request_asks_for_usage_with_the_stream(self):
        """vLLM omits usage from a stream unless stream_options requests it."""
        body = common.stream_body({"model": "m", "messages": []})
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_the_caller_payload_is_not_mutated(self):
        """The payload is hashed for the raw-response archive, so it must not
        grow transport fields after the fact."""
        payload = {"model": "m", "messages": []}
        common.stream_body(payload)
        self.assertNotIn("stream", payload)


if __name__ == "__main__":
    unittest.main()
