"""The subset pin is the control this adapter exists to enforce.

Everything else it does, Terminal-Bench already does through the same Harbor
driver. What is new is that the task set is a sample, so the tests concentrate
on the ways a sample can quietly stop being the one that was pre-registered: a
pin that omits it, a pin that names a different one, and a task pack that only
holds part of it.
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
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


adapter = load_module("swebench_pro", "scripts/adapters/swebench_pro.py")
protocol = load_module("run_eval_protocol", "scripts/run_eval_protocol.py")
selector = load_module("swebenchpro_subset", "scripts/swebenchpro_subset.py")


def pin_of(names) -> str:
    return "sha256:" + hashlib.sha256("\n".join(names).encode()).hexdigest()


OTHER_PIN = "sha256:" + "2" * 64


def task(repo: str, index: int, style: str = "versioned") -> str:
    base = f"{index:040x}"
    if style == "versioned":
        return f"instance_{repo}-{base}-v{'c' * 40}"
    if style == "nan":
        return f"instance_{repo}-{base}-vnan"
    return f"instance_{repo}-{base}"


NAMES = [
    task("ansible__ansible", 1),
    task("ansible__ansible", 2, "nan"),
    task("flipt-io__flipt", 3, "bare"),
]


def subset_file(path: Path, names=None, **overrides) -> Path:
    chosen = list(names or NAMES)
    payload = {
        "dataset": "swebenchpro",
        "version": "1.0",
        "seed": 38027,
        "population": 731,
        "size": len(chosen),
        "subset_pin": pin_of(chosen),
        "task_names": chosen,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


PIN = pin_of(NAMES)


def valid_pins() -> dict:
    return {
        "dataset": f"swebenchpro@1.0+subset:{PIN}",
        "harness": "harbor@0.21.0",
        "verifier": "sha256:abc",
        "adapter": adapter.self_pin(),
    }


def instructions(names=None) -> dict:
    return {name: f"Fix the bug in {name}." for name in (names or NAMES)}


def trial(name, reward=None, tokens=1200):
    row = {
        "task_name": name,
        "trial_uri": f"file:///jobs/{name}",
        "task_checksum": "deadbeef",
        "agent_result": {"n_output_tokens": tokens, "n_input_tokens": 10 * tokens},
    }
    if reward is not None:
        row["verifier_result"] = {"rewards": reward}
    return row


class PinTests(unittest.TestCase):
    def test_a_complete_pin_is_accepted(self) -> None:
        adapter.validate_pins(valid_pins(), PIN)

    def test_a_pin_without_a_subset_is_refused(self) -> None:
        """Two arms could otherwise run different samples and still compare."""
        pins = dict(valid_pins(), dataset="swebenchpro@1.0")
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.validate_pins(pins, PIN)
        self.assertIn("subset", str(caught.exception))

    def test_a_pin_naming_another_subset_is_refused(self) -> None:
        pins = dict(valid_pins(), dataset=f"swebenchpro@1.0+subset:{OTHER_PIN}")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins, PIN)

    def test_another_dataset_is_refused(self) -> None:
        pins = dict(valid_pins(), dataset=f"swebench-verified@1.0+subset:{PIN}")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins, PIN)

    def test_a_malformed_subset_digest_is_refused(self) -> None:
        pins = dict(valid_pins(), dataset="swebenchpro@1.0+subset:nonsense")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins, None)

    def test_placeholders_are_refused(self) -> None:
        pins = dict(valid_pins(), harness="REPLACE_WITH_HARBOR_COMMIT")
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins, PIN)

    def test_an_edited_adapter_invalidates_the_pin(self) -> None:
        pins = dict(valid_pins(), adapter="sha256:" + "0" * 64)
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_pins(pins, PIN)

    def test_editing_the_shared_driver_invalidates_the_pin(self) -> None:
        """Harbor's translation lives in _harbor.py, so it has to be pinned too."""
        driver = ROOT / "scripts" / "adapters" / "_harbor.py"
        original = driver.read_bytes()
        before = adapter.self_pin()
        try:
            driver.write_bytes(original + b"\n# touched\n")
            self.assertNotEqual(adapter.self_pin(), before)
        finally:
            driver.write_bytes(original)
        self.assertEqual(adapter.self_pin(), before)


class RepoTests(unittest.TestCase):
    def test_every_name_shape_yields_a_repo(self) -> None:
        for style in ("versioned", "nan", "bare"):
            with self.subTest(style=style):
                self.assertEqual(
                    adapter.task_repo(task("ansible__ansible", 7, style)),
                    "ansible__ansible",
                )

    def test_the_adapter_and_the_selector_agree(self) -> None:
        """Two copies of this rule that disagreed would silently lose a stratum."""
        for style in ("versioned", "nan", "bare"):
            for repo in ("ansible__ansible", "flipt-io__flipt", "tutao__tutanota"):
                name = task(repo, 5, style)
                with self.subTest(name=name):
                    self.assertEqual(adapter.task_repo(name), selector.task_repo(name))

    def test_an_unreadable_name_is_an_error(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.task_repo("ansible__ansible-1234")


class TaskPackTests(unittest.TestCase):
    def test_a_missing_pack_names_this_suite_s_variable(self) -> None:
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.read_task_instructions(Path("/nonexistent/pack"), "SWEP_TASKS_DIR")
        message = str(caught.exception)
        self.assertIn("SWEP_TASKS_DIR", message)
        self.assertNotIn("TB_TASKS_DIR", message)


class SubsetLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_a_drawn_subset_loads(self) -> None:
        loaded = adapter.load_subset(subset_file(self.dir / "s.json"))
        self.assertEqual(loaded["task_names"], NAMES)

    def test_a_missing_file_says_how_to_draw_one(self) -> None:
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.load_subset(self.dir / "absent.json")
        self.assertIn("swebenchpro_subset.py", str(caught.exception))

    def test_a_subset_of_another_dataset_is_refused(self) -> None:
        path = subset_file(self.dir / "s.json", dataset="swebench-verified")
        with self.assertRaises(adapter.AdapterError):
            adapter.load_subset(path)

    def test_a_repeated_task_is_refused(self) -> None:
        path = subset_file(self.dir / "s.json", names=[NAMES[0], NAMES[0]])
        with self.assertRaises(adapter.AdapterError):
            adapter.load_subset(path)

    def test_a_tampered_task_list_is_refused(self) -> None:
        """The pin is recomputed, so editing names without it cannot pass."""
        path = subset_file(self.dir / "s.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["task_names"][0] = task("nodebb__nodebb", 42)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.load_subset(path)
        self.assertIn("does not match its own task_names", str(caught.exception))

    def test_a_subset_without_a_pin_is_refused(self) -> None:
        path = self.dir / "s.json"
        subset_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["subset_pin"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(adapter.AdapterError):
            adapter.load_subset(path)

    def test_the_real_drawn_subset_loads(self) -> None:
        loaded = adapter.load_subset(ROOT / adapter.DEFAULT_SUBSET)
        self.assertEqual(len(loaded["task_names"]), 300)
        self.assertEqual(loaded["population"], 731)


class MaterializeTests(unittest.TestCase):
    def subset(self, names=None) -> dict:
        return {"task_names": list(names or NAMES), "subset_pin": PIN}

    def test_prompts_follow_the_drawn_order(self) -> None:
        prompts, key = adapter.materialize(self.subset(), instructions())
        self.assertEqual([row["id"] for row in prompts], NAMES)
        self.assertEqual(list(key), NAMES)

    def test_rows_carry_the_repository_as_the_category(self) -> None:
        prompts, key = adapter.materialize(self.subset(), instructions())
        self.assertEqual(prompts[0]["category"], "ansible__ansible")
        self.assertEqual(prompts[2]["category"], "flipt-io__flipt")
        self.assertEqual(key[NAMES[2]]["category"], "flipt-io__flipt")

    def test_a_partial_task_pack_is_fatal(self) -> None:
        """Running what happened to download would be a different sample."""
        with self.assertRaises(adapter.AdapterError) as caught:
            adapter.materialize(self.subset(), instructions(NAMES[:2]))
        self.assertIn("missing 1", str(caught.exception))

    def test_extra_tasks_in_the_pack_are_ignored(self) -> None:
        extra = instructions(NAMES + [task("nodebb__nodebb", 9)])
        prompts, _ = adapter.materialize(self.subset(), extra)
        self.assertEqual(len(prompts), len(NAMES))

    def test_prompts_satisfy_the_runner(self) -> None:
        prompts, _ = adapter.materialize(self.subset(), instructions())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            adapter.write_jsonl(path, prompts)
            self.assertEqual(protocol.validate_prompts(path, adapter.SUITE), NAMES)


class TranslateTests(unittest.TestCase):
    def key(self):
        return {name: {"task_name": name, "category": adapter.task_repo(name)}
                for name in NAMES}

    def test_rows_carry_the_repository_through(self) -> None:
        result = {"trial_results": [trial(name, {"reward": 1.0}) for name in NAMES]}
        rows = adapter.translate(result, self.key(), adapter.SUITE, 0)
        self.assertEqual(
            [row["category"] for row in rows],
            ["ansible__ansible", "ansible__ansible", "flipt-io__flipt"],
        )

    def test_rows_satisfy_the_runner_contract(self) -> None:
        result = {"trial_results": [trial(name, {"reward": 0.5}) for name in NAMES]}
        rows = adapter.translate(result, self.key(), adapter.SUITE, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            adapter.write_jsonl(path, rows)
            protocol.validate_results(path, adapter.SUITE, 0, set(NAMES))

    def test_a_task_harbor_never_ran_is_fatal(self) -> None:
        result = {"trial_results": [trial(NAMES[0], {"reward": 1.0})]}
        with self.assertRaises(adapter.AdapterError):
            adapter.translate(result, self.key(), adapter.SUITE, 0)


class JobConfigTests(unittest.TestCase):
    def config(self):
        args = SimpleNamespace(n_attempts=1, concurrency=40, timeout_multiplier=1.0,
                               environment="singularity")
        return adapter.build_job_config(
            job_name="job", jobs_dir=Path("/jobs"), dataset="swebenchpro",
            version="1.0", agent="hermes", model="openai/qwen38-eval",
            task_names=list(reversed(NAMES)), args=args,
            base_url="http://host:8000/v1", api_key="EMPTY",
        )

    def test_the_dataset_pin_does_not_leak_into_the_job_config(self) -> None:
        """Harbor knows nothing about our subset digest; it takes name and version."""
        dataset = self.config()["datasets"][0]
        self.assertEqual(dataset["name"], "swebenchpro")
        self.assertEqual(dataset["version"], "1.0")
        self.assertEqual(dataset["task_names"], sorted(NAMES))

    def test_config_validates_against_harbor_when_installed(self) -> None:
        try:
            from harbor.models.job.config import JobConfig
        except ImportError:
            self.skipTest("harbor is not installed in this environment")
        JobConfig.model_validate(self.config())


class ParkedTests(unittest.TestCase):
    """The suite is not in the protocol, and that is deliberate."""

    def test_the_suite_is_not_registered(self) -> None:
        # Harbor's singularity backend cannot start these containers here, so
        # the adapter is kept but unwired. See eval/README.md.
        self.assertNotIn(adapter.SUITE, protocol.REQUIRED_SUITES)

    def test_the_adapter_still_works_for_the_day_it_is_wired_back(self) -> None:
        prompts, key = adapter.materialize(
            {"task_names": list(NAMES), "subset_pin": PIN}, instructions()
        )
        self.assertEqual(len(prompts), len(NAMES))
        self.assertEqual(len(key), len(NAMES))


if __name__ == "__main__":
    unittest.main()
