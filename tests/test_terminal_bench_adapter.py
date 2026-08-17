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


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("terminal_bench", "eval/scripts/adapters/terminal_bench.py")
protocol = load_module("run_eval_protocol", "eval/scripts/run_eval_protocol.py")


def valid_pins() -> dict:
    return {
        "dataset": "terminal-bench/terminal-bench-2-1@2.1.0",
        "harness": "harbor@0.21.0",
        "verifier": "sha256:abc",
        "adapter": adapter.self_pin(),
    }


def trial(task, reward=None, exception=None, tokens=1200):
    row = {
        "task_name": task,
        "trial_uri": f"file:///jobs/{task}",
        "task_checksum": "deadbeef",
        "agent_result": {"n_output_tokens": tokens, "n_input_tokens": 10 * tokens},
    }
    if reward is not None:
        row["verifier_result"] = {"rewards": reward}
    if exception is not None:
        row["exception_info"] = exception
    return row


def write_pack(root: Path, names, filename="instruction.md"):
    for index, name in enumerate(names):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / filename).write_text(f"Task {index}: do the thing.", encoding="utf-8")
    return root


class PinTests(unittest.TestCase):
    def test_valid_pins(self) -> None:
        adapter.validate_pins(valid_pins())

    def test_placeholder_rejected(self) -> None:
        pins = dict(valid_pins(), dataset="REPLACE_WITH_TERMINAL_BENCH_REVISION")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_dataset_needs_a_version(self) -> None:
        pins = dict(valid_pins(), dataset="terminal-bench/terminal-bench-2-1")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)

    def test_edited_adapter_invalidates_the_pin(self) -> None:
        pins = dict(valid_pins(), adapter="sha256:" + "0" * 64)
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins)


class MaterializeTests(unittest.TestCase):
    def test_instructions_are_read_from_the_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_pack(Path(tmp), ["alpha", "beta"])
            instructions = adapter.read_task_instructions(Path(tmp))
        self.assertEqual(sorted(instructions), ["alpha", "beta"])
        self.assertIn("do the thing", instructions["alpha"])

    def test_missing_pack_says_how_to_get_it(self) -> None:
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.read_task_instructions(Path("/nonexistent/pack"))
        self.assertIn("harbor download", str(caught.exception))

    def test_pack_without_instruction_files_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "alpha").mkdir()
            with self.assertRaises(adapter.AdapterError):
                adapter.read_task_instructions(Path(tmp))

    def test_pilot_rows_carry_category_and_difficulty(self) -> None:
        # The runner rejects a pilot row missing either field.
        prompts, _ = adapter.materialize(
            {f"task{i}": "text" for i in range(40)}, adapter.PILOT_SUITE, 30
        )
        self.assertEqual(len(prompts), 30)
        for row in prompts:
            self.assertTrue(row["category"] and row["difficulty"])

    def test_pilot_larger_than_the_pack_is_refused(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.materialize({"a": "x"}, adapter.PILOT_SUITE, 30)

    def test_prompts_satisfy_the_runner(self) -> None:
        prompts, _ = adapter.materialize({"alpha": "do it", "beta": "do that"},
                                         adapter.SUITE, None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tb.jsonl"
            adapter.write_jsonl(path, prompts)
            self.assertEqual(protocol.validate_prompts(path, adapter.SUITE),
                             ["alpha", "beta"])


class TranslateTests(unittest.TestCase):
    def key(self, *names):
        return {name: {"task_name": name} for name in names}

    def test_rows_match_the_materialized_tasks(self) -> None:
        result = {"trial_results": [trial("alpha", {"reward": 1.0}),
                                    trial("beta", {"reward": 0.0})]}
        rows = adapter.translate(result, self.key("alpha", "beta"), adapter.SUITE, 0)
        self.assertEqual([r["id"] for r in rows], ["alpha", "beta"])
        self.assertEqual([r["score"] for r in rows], [1.0, 0.0])

    def test_best_of_attempts_is_taken(self) -> None:
        result = {"trial_results": [trial("alpha", {"reward": 0.0}),
                                    trial("alpha", {"reward": 1.0})]}
        rows = adapter.translate(result, self.key("alpha"), adapter.SUITE, 0)
        self.assertEqual(rows[0]["score"], 1.0)
        self.assertEqual(rows[0]["attempts_run"], 2)

    def test_agent_timeout_sets_the_timeout_flag(self) -> None:
        result = {"trial_results": [trial("alpha", None,
                                          {"exception_type": "AgentTimeoutError"})]}
        rows = adapter.translate(result, self.key("alpha"), adapter.SUITE, 0)
        self.assertTrue(rows[0]["timeout"])
        self.assertTrue(rows[0]["empty_answer"])
        self.assertEqual(rows[0]["score"], 0.0)

    def test_other_exceptions_are_recorded_without_the_timeout_flag(self) -> None:
        result = {"trial_results": [trial("alpha", {"reward": 0.0},
                                          {"exception_type": "EnvironmentBuildError"})]}
        rows = adapter.translate(result, self.key("alpha"), adapter.SUITE, 0)
        self.assertFalse(rows[0]["timeout"])
        self.assertEqual(rows[0]["exception_type"], "EnvironmentBuildError")

    def test_a_task_harbor_never_ran_is_fatal(self) -> None:
        # Silently scoring it zero would read as a model failure.
        result = {"trial_results": [trial("alpha", {"reward": 1.0})]}
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.translate(result, self.key("alpha", "beta"), adapter.SUITE, 0)
        self.assertIn("beta", str(caught.exception))

    def test_malformed_result_is_fatal(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.translate({}, self.key("alpha"), adapter.SUITE, 0)

    def test_rows_satisfy_the_runner_contract(self) -> None:
        result = {"trial_results": [trial("alpha", {"reward": 0.5})]}
        rows = adapter.translate(result, self.key("alpha"), adapter.SUITE, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            adapter.write_jsonl(path, rows)
            protocol.validate_results(path, adapter.SUITE, 0, {"alpha"})


class JobConfigTests(unittest.TestCase):
    def config(self, **overrides):
        args = SimpleNamespace(n_attempts=1, concurrency=4, timeout_multiplier=1.0,
                               environment="singularity")
        for key, value in overrides.items():
            setattr(args, key, value)
        return adapter.build_job_config(
            job_name="job", jobs_dir=Path("/jobs"), dataset="terminal-bench/terminal-bench-2-1",
            version="2.1.0", agent="hermes", model="openai/qwen38-eval",
            task_names=["beta", "alpha"], args=args,
            base_url="http://host:8000/v1", api_key="EMPTY",
        )

    def test_config_matches_harbor_field_names(self) -> None:
        config = self.config()
        self.assertEqual(config["environment"]["type"], "singularity")
        self.assertEqual(config["agents"][0]["name"], "hermes")
        self.assertEqual(config["agents"][0]["model_name"], "openai/qwen38-eval")
        self.assertEqual(config["datasets"][0]["version"], "2.1.0")
        self.assertEqual(config["n_concurrent_trials"], 4)

    def test_endpoint_reaches_the_agent_through_its_environment(self) -> None:
        env = self.config()["agents"][0]["env"]
        self.assertEqual(env["OPENAI_BASE_URL"], "http://host:8000/v1")

    def test_task_names_are_sorted_for_a_stable_config(self) -> None:
        self.assertEqual(self.config()["datasets"][0]["task_names"], ["alpha", "beta"])

    def test_config_validates_against_harbor_when_installed(self) -> None:
        try:
            from harbor.models.job.config import JobConfig
        except ImportError:
            self.skipTest("harbor is not installed in this environment")
        JobConfig.model_validate(self.config())


if __name__ == "__main__":
    unittest.main()
