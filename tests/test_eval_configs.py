"""The shipped eval configs must be usable from a fresh checkout.

These files define what gets measured -- dataset revisions, token caps, request
timeouts and adapter pins. They used to exist only on the cluster, hand-edited,
so nothing tied a result to the measurement definition that produced it. Now
they are tracked, which only works if the deployment-specific paths in them
resolve from the environment rather than being spelled out.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_adapter_suite", ROOT / "eval" / "scripts" / "run_adapter_suite.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

CONFIGS = ("paired-1.json", "paired-2.json", "dry-run.json")
# Every variable the sbatch exports for the configs to consume.
ENVIRONMENT = {
    "EVAL_PYTHON": "/opt/venv/bin/python",
    "EVAL_CORPUS": "/data/haystack.txt",
    "EVAL_TOKENIZER": "/data/model",
}


class ShippedConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in ENVIRONMENT}
        os.environ.update(ENVIRONMENT)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def suites(self, name: str) -> list[str]:
        raw = json.loads((ROOT / "eval" / name).read_text(encoding="utf-8"))
        return [suite["name"] for suite in raw["suites"]]

    def test_every_config_loads_with_the_environment_the_sbatch_exports(self) -> None:
        for name in CONFIGS:
            for suite in self.suites(name):
                with self.subTest(config=name, suite=suite):
                    config = runner.load_config(ROOT / "eval" / name, suite)
                    entry = next(
                        item for item in config["suites"] if item["name"] == suite
                    )
                    for action in ("prepare", "run"):
                        command = entry[action]
                        self.assertEqual(command[0], ENVIRONMENT["EVAL_PYTHON"])
                        self.assertFalse(
                            any("${" in part for part in command),
                            f"{name}/{suite}/{action} left a variable unresolved",
                        )

    def test_no_cluster_paths_survive_in_the_tracked_files(self) -> None:
        # The whole point of tracking these is that they carry no deployment.
        for name in CONFIGS:
            text = (ROOT / "eval" / name).read_text(encoding="utf-8")
            with self.subTest(config=name):
                self.assertNotIn("/scratch/", text)
                self.assertNotIn("nicosuter", text)

    def test_an_unset_variable_stops_the_run(self) -> None:
        # Fail closed: exec'ing a file literally named "${EVAL_PYTHON}" would
        # fail later and less legibly, and on a config naming a corpus it could
        # silently measure the wrong thing.
        del os.environ["EVAL_PYTHON"]
        with self.assertRaises(runner.protocol.ProtocolError):
            runner.load_config(ROOT / "eval" / "paired-2.json", "gpqa_diamond")

    def test_paired_2_still_describes_the_campaign_we_ran(self) -> None:
        # A config edit that silently changed a cap would invalidate reuse of
        # every result scored under the old one.
        config = runner.load_config(ROOT / "eval" / "paired-2.json", "gpqa_diamond")
        entry = next(
            item for item in config["suites"] if item["name"] == "gpqa_diamond"
        )
        run = entry["run"]
        self.assertEqual(run[run.index("--max-tokens") + 1], "131072")
        self.assertEqual(run[run.index("--request-timeout") + 1], "5400")
        self.assertEqual(config["order_seed"], 38027)

    def test_every_suite_shares_one_token_cap(self) -> None:
        """One cap for every suite, in every config.

        Per-suite caps meant a suite's score partly measured its own budget: at
        65536, LiveCodeBench truncated 19% of baseline items and 27% of
        candidate items, every one of them at finish_reason=length with an empty
        answer, and the resulting 8-point 'regression' was entirely the cap.
        Wrong answers were 7 against 7.

        131072 is half the 262144 context, so the longest-prompt suites still
        fit their prompt beside it.
        """
        caps, timeouts = set(), set()
        for name in ("paired-1.json", "paired-2.json"):
            config = json.loads((ROOT / "eval" / name).read_text(encoding="utf-8"))
            for suite in config["suites"]:
                run = [str(part) for part in suite["run"]]
                if "--max-tokens" in run:
                    caps.add(run[run.index("--max-tokens") + 1])
                if "--request-timeout" in run:
                    timeouts.add(run[run.index("--request-timeout") + 1])
        self.assertEqual(caps, {"131072"}, f"suites disagree on the cap: {caps}")
        self.assertEqual(timeouts, {"5400"}, f"suites disagree on the timeout: {timeouts}")



class AdapterPinTests(unittest.TestCase):
    """Every config's adapter pin must match the adapter it points at.

    The pin exists to stop an adapter changing under a result set, and it works:
    it has refused a run twice. Both times the pin was simply stale, discovered
    on a cluster after a queue wait. It is a hash of files in this repository,
    so it can be checked here instead.
    """

    def adapter_for(self, suite: dict) -> Path | None:
        for part in suite.get("run", []):
            if str(part).endswith(".py"):
                return ROOT / str(part)
        return None

    def test_every_pin_matches_its_adapter(self) -> None:
        for name in CONFIGS:
            raw = json.loads((ROOT / "eval" / name).read_text(encoding="utf-8"))
            for suite in raw["suites"]:
                path = self.adapter_for(suite)
                if path is None or not path.is_file():
                    continue
                with self.subTest(config=name, suite=suite["name"]):
                    spec = importlib.util.spec_from_file_location(
                        f"pin_{suite['name']}", path
                    )
                    assert spec and spec.loader
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.assertEqual(
                        suite["pins"]["adapter"],
                        module.self_pin(),
                        f"{name}: {suite['name']} pin is stale; rerun "
                        f"`{path.relative_to(ROOT)} pin`",
                    )

if __name__ == "__main__":
    unittest.main()


import eval_suite  # noqa: E402  (eval/scripts is on sys.path above)

_PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "run_eval_protocol", ROOT / "eval" / "scripts" / "run_eval_protocol.py"
)
protocol = importlib.util.module_from_spec(_PROTOCOL_SPEC)
_PROTOCOL_SPEC.loader.exec_module(protocol)


class ProtocolCoverageTests(unittest.TestCase):
    """The suite set is derived from one versioned file, so this checks it is."""

    def test_the_runner_derives_its_required_set(self) -> None:
        self.assertEqual(protocol.REQUIRED_SUITES, eval_suite.names("v2"))

    def test_the_example_protocol_declares_a_version(self) -> None:
        config = json.loads(
            (ROOT / "eval" / "protocol.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["eval_suite"], "v1")
        self.assertEqual(
            {suite["name"] for suite in config["suites"]}, eval_suite.names("v2")
        )

    def test_the_earlier_batches_are_history_not_the_protocol(self) -> None:
        """paired-1 and paired-2 are batches that accreted; neither defines it."""
        for name in ("paired-1", "paired-2"):
            config = json.loads((ROOT / "eval" / f"{name}.json").read_text(encoding="utf-8"))
            names = {suite["name"] for suite in config["suites"]}
            with self.subTest(config=name):
                self.assertNotEqual(names, eval_suite.names("v2"))

    def test_every_required_suite_is_pinned(self) -> None:
        for suite in eval_suite.load("v1")["suites"]:
            with self.subTest(suite=suite["name"]):
                for field, value in suite["pins"].items():
                    self.assertNotIn("REPLACE_", value, field)


class NarrowExpansionTests(unittest.TestCase):
    """One suite's deployment variable must not fail the other six.

    RULER names ${EVAL_CORPUS} and ${EVAL_TOKENIZER}. Expanding the whole config
    meant a deployment that had not set those could not prepare bfcl_v4 either,
    and a prepare job failed all seven suites on one suite's missing variable.
    """

    def config(self, directory: Path) -> Path:
        path = directory / "config.json"
        path.write_text(json.dumps({
            "served_model_name": "openai/qwen38-eval",
            "order_seed": 1,
            "generation": {},
            "suites": [
                {"name": "plain", "replicates": 1, "pins": {"adapter": "a"},
                 "prepare": ["${EVAL_PYTHON}", "prepare"],
                 "run": ["${EVAL_PYTHON}", "run"]},
                {"name": "needs_corpus", "replicates": 1, "pins": {"adapter": "a"},
                 "prepare": ["${EVAL_PYTHON}", "prepare", "--corpus", "${EVAL_CORPUS}"],
                 "run": ["${EVAL_PYTHON}", "run"]},
            ],
        }), encoding="utf-8")
        return path

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        # Both are restored: EVAL_PYTHON used to be left pointing at an
        # interpreter that does not exist, which every later test running a
        # shell out of this process then inherited.
        self.previous = {key: os.environ.get(key) for key in ("EVAL_CORPUS", "EVAL_PYTHON")}
        os.environ.pop("EVAL_CORPUS", None)
        os.environ["EVAL_PYTHON"] = "/venv/bin/python"

    def tearDown(self) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_a_suite_is_held_only_to_the_variables_it_names(self) -> None:
        loaded = runner.load_config(self.config(self.dir), "plain")
        entry = next(s for s in loaded["suites"] if s["name"] == "plain")
        self.assertEqual(entry["prepare"][0], "/venv/bin/python")

    def test_the_suite_that_needs_it_still_fails_closed(self) -> None:
        with self.assertRaisesRegex(runner.protocol.ProtocolError, "EVAL_CORPUS"):
            runner.load_config(self.config(self.dir), "needs_corpus")

    def test_an_unknown_suite_still_lists_what_exists(self) -> None:
        with self.assertRaisesRegex(runner.protocol.ProtocolError, "plain"):
            runner.load_config(self.config(self.dir), "absent")


class EnvironmentContractTests(unittest.TestCase):
    """Where datasets cache is a property of the deployment, not of a job.

    Left unset, huggingface_hub caches under $HOME. That put benchmark data on a
    home filesystem while a large scratch cache went unused, and split the two so
    a dataset fetched by one path was invisible to another: a gated dataset
    warmed into the scratch cache still failed to load inside a batch job.
    """

    def source_env(self, **overrides):
        environment = {**os.environ, "RUN_BASE": "/scratch/example", **overrides}
        result = subprocess.run(
            ["bash", "-c", "source common/scripts/load_env.sh && echo \"$HF_HOME\""],
            capture_output=True, text=True, cwd=ROOT, env=environment,
        )
        return result.stdout.strip()

    def test_the_cache_defaults_under_the_run_base(self) -> None:
        self.assertEqual(self.source_env(HF_HOME=""), "/scratch/example/huggingface")

    def test_an_explicit_setting_still_wins(self) -> None:
        self.assertEqual(self.source_env(HF_HOME="/elsewhere/hf"), "/elsewhere/hf")

    def test_it_never_defaults_into_home(self) -> None:
        self.assertNotIn(os.path.expanduser("~"), self.source_env(HF_HOME=""))


class RulerItemBudgetTests(unittest.TestCase):
    """RULER's item count is a protocol decision, so state it out loud.

    RULER is the one suite whose item pool is synthesized rather than fixed, so
    `--items-per-task` is a dial and not a constraint. That makes it the only
    place the protocol's own rule -- "items buy precision that replicates do
    not" -- can actually be acted on, and it makes an accidental change to the
    dial invisible unless something asserts the resulting size.

    The count is tasks x lengths x items-per-task.
    """

    EXPECTED_ITEMS = 100

    def _ruler(self, config: str) -> dict:
        raw = json.loads((ROOT / "eval" / config).read_text())
        suites = raw["suites"]
        by_name = {e["name"]: e for e in suites} if isinstance(suites, list) else suites
        return by_name["ruler"]

    def _arg(self, argv: list, flag: str) -> str:
        return argv[argv.index(flag) + 1]

    @staticmethod
    def _ruler_module():
        spec = importlib.util.spec_from_file_location(
            "_ruler", ROOT / "eval" / "scripts" / "adapters" / "ruler.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_suite_size_is_what_the_protocol_says(self) -> None:
        ruler = self._ruler_module()
        prepare = self._ruler("eval-suite-v2.json")["prepare"]
        lengths = [int(x) for x in self._arg(prepare, "--lengths").split(",")]
        per_task = int(self._arg(prepare, "--items-per-task"))
        kept = ruler.parse_categories(self._arg(prepare, "--only"))
        plan = ruler.plan_categories(lengths, set(), kept=kept)
        self.assertEqual(len(plan) * per_task, self.EXPECTED_ITEMS)

    def test_every_kept_category_separated_some_checkpoint(self) -> None:
        """The suite is now defined by where the checkpoints measured differed.

        Sixteen categories returned an identical score on both arms for all five
        checkpoints and were dropped. These five moved: 128k/fwe and 32k/fwe on
        several, 32k/cwe on barry, 4k/cwe on fp8gdn, 128k/vt on philbert.
        """
        prepare = self._ruler("eval-suite-v2.json")["prepare"]
        kept = self._ruler_module().parse_categories(self._arg(prepare, "--only"))
        self.assertEqual(
            kept,
            {("4k", "cwe"), ("32k", "cwe"), ("32k", "fwe"), ("128k", "fwe"), ("128k", "vt")},
        )

    def test_the_cost_of_narrowing_is_written_down(self) -> None:
        """A suite that can no longer detect a failure must say so in the file.

        Every dropped niah_* category was the evidence that long-context
        retrieval survived quantization, which is the risk MODEL_CARD.md
        singles out. Losing that silently is the failure mode worth a test.
        """
        raw = json.loads((ROOT / "eval" / "eval-suite-v2.json").read_text())
        note = raw["rationale"]["ruler_categories"]
        self.assertIn("selection on the results", note)
        self.assertIn("long-context retrieval", note)


class WindowBudgetTests(unittest.TestCase):
    """Every v2 suite spends what the window leaves, not a fixed cap.

    A cap does not buy a shorter answer, it buys no answer. Nothing tells the
    model its budget -- the API does not carry it and the prompt does not
    mention it -- so it cannot spend against one: under 131072, nine of ten
    truncated MathArena items had spent the whole cap on reasoning and emitted
    zero answer tokens, and RULER truncated 14% of baseline items against 17% of
    candidate ones. Truncation lands on the arm that reasons longer, which is
    the arm under test, so the cap was measuring itself.

    `--max-tokens 0` omits the field, and the server then allows whatever the
    prompt leaves of max-model-len. That is the per-item budget: 258,217 tokens
    behind a 4k prompt and 131,112 behind a 131k one, decided per item by
    arithmetic rather than by one number chosen for all of them.

    One value still has to be shared. Per-suite caps meant a suite's score
    partly measured its own budget; "the rest of the window" is a single rule
    that happens to resolve differently per item.
    """

    def test_every_v2_suite_defers_to_the_window(self) -> None:
        raw = json.loads((ROOT / "eval" / "eval-suite-v2.json").read_text())
        for suite in raw["suites"]:
            run = [str(part) for part in suite["run"]]
            with self.subTest(suite=suite["name"]):
                self.assertIn("--max-tokens", run, "a suite with no cap flag is unreadable")
                self.assertEqual(
                    run[run.index("--max-tokens") + 1],
                    "0",
                    f"{suite['name']} still carries a fixed cap",
                )

    def test_the_reason_is_recorded_where_the_old_cap_was_justified(self) -> None:
        """token_budget claimed the cap was non-binding. It was not, on RULER."""
        raw = json.loads((ROOT / "eval" / "eval-suite-v2.json").read_text())
        note = raw["rationale"]["token_budget"]
        self.assertIn("0", note)
        self.assertIn("ruler", note.lower())
