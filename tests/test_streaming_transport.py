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


def delta(content=None, reasoning=None, finish_reason=None) -> dict:
    body: dict = {}
    if content is not None:
        body["content"] = content
    if reasoning is not None:
        body["reasoning_content"] = reasoning
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
