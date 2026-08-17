import importlib.util
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "scripts" / "adapters"))
sys.path.insert(0, str(ROOT / "common" / "scripts"))


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("mtp_adapter", "eval/scripts/adapters/mtp.py")
comparator = load_module("compare_mtp_results", "eval/scripts/compare_mtp_results.py")

GENERATION = {"enable_thinking": False, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
              "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}


def metrics(accepted: int, drafted: int) -> str:
    return (
        "# HELP vllm:spec_decode_num_accepted_tokens_total accepted\n"
        f"vllm:spec_decode_num_accepted_tokens_total {accepted}\n"
        f"vllm:spec_decode_num_draft_tokens_total {drafted}\n"
    )


def completion(content="ok", tokens=64):
    return {
        "choices": [{"finish_reason": "stop",
                     "message": {"content": content, "reasoning_content": ""}}],
        "usage": {"completion_tokens": tokens},
    }


def args(**overrides):
    defaults = {"action": "run", "max_tokens": 256, "request_timeout": 30.0,
                "retries": 0, "metrics_url": ""}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PinTests(unittest.TestCase):
    def test_request_set_pin_is_content_derived(self) -> None:
        self.assertTrue(adapter.request_set_pin().startswith("sha256:"))
        self.assertEqual(adapter.request_set_pin(), adapter.request_set_pin())

    def test_wrong_request_set_pin_is_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins({"request_set": "sha256:" + "0" * 64,
                                   "adapter": adapter.self_pin()})

    def test_edited_adapter_invalidates_the_pin(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins({"request_set": adapter.request_set_pin(),
                                   "adapter": "sha256:" + "0" * 64})


class MetricsTests(unittest.TestCase):
    def test_metrics_url_derived_from_the_api_base(self) -> None:
        self.assertEqual(adapter.metrics_url_for("http://h:8000/v1", ""),
                         "http://h:8000/metrics")
        self.assertEqual(adapter.metrics_url_for("http://h:8000/v1/", ""),
                         "http://h:8000/metrics")

    def test_explicit_override_wins(self) -> None:
        self.assertEqual(adapter.metrics_url_for("http://h:8000/v1", "http://x/m"),
                         "http://x/m")

    def test_counter_totals(self) -> None:
        self.assertEqual(adapter.counter_totals(metrics(10, 20)), (10.0, 20.0))

    def test_absent_counters_read_as_zero(self) -> None:
        self.assertEqual(adapter.counter_totals("# nothing here\n"), (0.0, 0.0))


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name)
        self.results = self.run_dir / "mtp.jsonl"
        environ = {
            "EVAL_RUN_DIR": str(self.run_dir),
            "EVAL_RESULTS_JSONL": str(self.results),
            "EVAL_SERVED_MODEL": "openai/qwen38-eval",
            "OPENAI_BASE_URL": "http://inference:8000/v1",
            "OPENAI_API_KEY": "EMPTY",
            "EVAL_GENERATION_JSON": json.dumps(GENERATION),
            "EVAL_PINS_JSON": json.dumps({"request_set": adapter.request_set_pin(),
                                          "adapter": adapter.self_pin()}),
            "EVAL_SEED": "38027",
        }
        patch = unittest.mock.patch.dict(os.environ, environ, clear=False)
        patch.start()
        self.addCleanup(patch.stop)

    def run_mode(self, mode, concurrency=1, accepted=(0, 800), drafted=(0, 1000)):
        os.environ["EVAL_MTP_MODE"] = mode
        os.environ["EVAL_CONCURRENCY"] = str(concurrency)
        snapshots = iter([metrics(accepted[0], drafted[0]), metrics(accepted[1], drafted[1])])
        with unittest.mock.patch.object(adapter, "fetch_metrics",
                                        side_effect=lambda url, timeout=30.0: next(snapshots)):
            adapter.command_run(args(), client=lambda *a: completion())
        return [json.loads(line) for line in self.results.read_text().splitlines() if line]

    def test_request_set_saturates_the_concurrency(self) -> None:
        # At batch 1 the base set is enough; at 384 it would measure an empty pipe.
        _, order = adapter.expand_requests(1)
        self.assertEqual(len(order), len(adapter.REQUESTS))
        _, wide = adapter.expand_requests(384)
        self.assertGreaterEqual(len(wide), 384 * 2)
        self.assertEqual(len(wide), len(set(wide)))

    def test_both_modes_share_keys_at_one_concurrency(self) -> None:
        # compare_mtp_results requires identical keys across the two files.
        self.assertEqual(adapter.expand_requests(16)[1], adapter.expand_requests(16)[1])

    def test_disabled_mode_writes_no_acceptance_fields(self) -> None:
        rows = self.run_mode("disabled")
        self.assertEqual(len(rows), len(adapter.REQUESTS))
        self.assertNotIn("accepted_draft_tokens", rows[0])

    def test_delta_lands_on_exactly_one_row(self) -> None:
        # EVAL.md: the server-wide counter delta goes on one row with explicit
        # zeros elsewhere, so summing does not multiply by the request count.
        rows = self.run_mode("enabled", accepted=(100, 900), drafted=(200, 1200))
        self.assertEqual(sum(r["accepted_draft_tokens"] for r in rows), 800)
        self.assertEqual(sum(r["draft_tokens"] for r in rows), 1000)
        self.assertEqual(sum(1 for r in rows if r["draft_tokens"]), 1)

    def test_server_without_speculation_is_fatal(self) -> None:
        with self.assertRaises(adapter.AdapterError) as caught:
            self.run_mode("enabled", accepted=(5, 5), drafted=(7, 7))
        self.assertIn("drafted no tokens", str(caught.exception))

    def test_decreasing_counters_are_fatal(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            self.run_mode("enabled", accepted=(900, 100), drafted=(1200, 200))

    def test_bad_mode_is_refused(self) -> None:
        os.environ["EVAL_MTP_MODE"] = "sometimes"
        os.environ["EVAL_CONCURRENCY"] = "1"
        with self.assertRaises(adapter.AdapterError):
            adapter.command_run(args(), client=lambda *a: completion())

    def test_metadata_records_wall_clock_throughput(self) -> None:
        self.run_mode("enabled", concurrency=4)
        meta = json.loads((self.run_dir / "metadata" / "mtp-enabled-c4.json").read_text())
        self.assertEqual(meta["concurrency"], 4)
        self.assertEqual(meta["acceptance_rate"], 0.8)
        self.assertGreater(meta["wall_clock_output_tokens_per_second"], 0)

    def test_rows_satisfy_the_comparator(self) -> None:
        disabled = self.run_mode("disabled")
        enabled = self.run_mode("enabled")
        with tempfile.TemporaryDirectory() as tmp:
            off, on = Path(tmp) / "off.jsonl", Path(tmp) / "on.jsonl"
            off.write_text("\n".join(json.dumps(r) for r in disabled))
            on.write_text("\n".join(json.dumps(r) for r in enabled))
            loaded_off = comparator.load(off, mtp=False)
            loaded_on = comparator.load(on, mtp=True)
        self.assertEqual(set(loaded_off), set(loaded_on))
        self.assertEqual(sum(r["draft_tokens"] for r in loaded_on.values()), 1000)


if __name__ == "__main__":
    unittest.main()
