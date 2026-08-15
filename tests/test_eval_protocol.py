import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "run_eval_protocol", ROOT / "scripts" / "run_eval_protocol.py"
)
assert SPEC and SPEC.loader
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def valid_config() -> dict:
    commands = ["/usr/bin/true"]
    suites = []
    replicas = {
        "bfcl_v4": 1,
        "terminal_bench_2_1": 3,
        "livecodebench_v6": 4,
        "gpqa_diamond": 4,
        "matharena_2026_06": 4,
        "multimodal": 1,
        "ruler": 1,
    }
    for name, count in replicas.items():
        suite = {
            "name": name,
            "replicates": count,
            "pins": {
                "dataset": "dataset-revision",
                "harness": "harness-revision",
                "verifier": "verifier-revision",
                "adapter": "adapter-revision",
            },
            "prepare": commands,
            "run": commands,
        }
        if name == "terminal_bench_2_1":
            suite["pilot_prepare"] = commands
            suite["pilot_run"] = commands
        suites.append(suite)
    return {
        "version": 1,
        "served_model_name": "openai/qwen38-eval",
        "order_seed": 38027,
        "seeds": [38027, 38028, 38029, 38030],
        "baseline": {"model": "baseline", "revision": "deadbeef"},
        "candidate": {"model": "candidate"},
        "calibration_manifest": "manifest.jsonl",
        "overlap_review": None,
        "server": {
            "host": "0.0.0.0",
            "health_host": "127.0.0.1",
            "public_base_url": "http://inference:8000/v1",
            "port": 8000,
            "flags": [
                "--tensor-parallel-size",
                "1",
                "--data-parallel-size",
                "8",
                "--max-model-len",
                "262144",
                "--kv-cache-dtype",
                "auto",
                "--reasoning-parser",
                "qwen3",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
            ],
        },
        "generation": {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "suites": suites,
        "mtp": {
            "run": commands,
            "pins": {"request_set": "request-set-hash", "adapter": "adapter-revision"},
            "concurrencies": [1, 8],
            "enabled_server_flags": [
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":1}',
            ],
        },
    }


class ConfigTests(unittest.TestCase):
    def write_config(self, directory: Path, data: dict) -> Path:
        path = directory / "protocol.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_valid_protocol_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_config(Path(temporary), valid_config())
            loaded = protocol.load_config(path)
            self.assertEqual(len(loaded["suites"]), 7)
            self.assertIn("ruler", {suite["name"] for suite in loaded["suites"]})

    def test_rejects_speculation_in_primary_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = valid_config()
            config["server"]["flags"] += ["--speculative-config", "{}"]
            path = self.write_config(Path(temporary), config)
            with self.assertRaisesRegex(protocol.ProtocolError, "speculative"):
                protocol.load_config(path)

    def test_rejects_unpinned_adapter_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = valid_config()
            config["suites"][0]["run"] = ["REPLACE_WITH_ADAPTER"]
            path = self.write_config(Path(temporary), config)
            with self.assertRaisesRegex(protocol.ProtocolError, "placeholder"):
                protocol.load_config(path)

    def test_unresolved_environment_variable_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = valid_config()
            config["candidate"]["model"] = "${DEFINITELY_UNSET_EVAL_VAR}"
            os.environ.pop("DEFINITELY_UNSET_EVAL_VAR", None)
            path = self.write_config(Path(temporary), config)
            with self.assertRaisesRegex(protocol.ProtocolError, "unresolved"):
                protocol.load_config(path)


class ArtifactValidationTests(unittest.TestCase):
    def test_candidate_checkpoint_preflight_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(protocol.ProtocolError, "checkpoint is invalid"):
                protocol.validate_candidate_checkpoint(Path(temporary))

    def test_result_rows_require_complete_failure_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            row = {
                "suite": "gpqa_diamond",
                "id": "q1",
                "replicate": 0,
                "score": 1.0,
                **{field: False for field in protocol.RESULT_BOOL_FIELDS},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            protocol.validate_results(path, "gpqa_diamond", 0, {"q1"})
            del row["timeout"]
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "timeout"):
                protocol.validate_results(path, "gpqa_diamond", 0, {"q1"})

    def test_manifest_lock_cannot_be_reused_with_new_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            protocol.lock_config(run_dir, {"version": 1})
            with self.assertRaisesRegex(protocol.ProtocolError, "differs"):
                protocol.lock_config(run_dir, {"version": 2})


if __name__ == "__main__":
    unittest.main()
