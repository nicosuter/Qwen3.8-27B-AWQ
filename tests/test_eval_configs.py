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
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_adapter_suite", ROOT / "scripts" / "run_adapter_suite.py"
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


import eval_suite  # noqa: E402  (scripts/ is on sys.path above)

_PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "run_eval_protocol", ROOT / "scripts" / "run_eval_protocol.py"
)
protocol = importlib.util.module_from_spec(_PROTOCOL_SPEC)
_PROTOCOL_SPEC.loader.exec_module(protocol)


class ProtocolCoverageTests(unittest.TestCase):
    """The suite set is derived from one versioned file, so this checks it is."""

    def test_the_runner_derives_its_required_set(self) -> None:
        self.assertEqual(protocol.REQUIRED_SUITES, eval_suite.names("v1"))

    def test_the_example_protocol_declares_a_version(self) -> None:
        config = json.loads(
            (ROOT / "eval" / "protocol.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["eval_suite"], "v1")
        self.assertEqual(
            {suite["name"] for suite in config["suites"]}, eval_suite.names("v1")
        )

    def test_the_earlier_batches_are_history_not_the_protocol(self) -> None:
        """paired-1 and paired-2 are batches that accreted; neither defines it."""
        for name in ("paired-1", "paired-2"):
            config = json.loads((ROOT / "eval" / f"{name}.json").read_text(encoding="utf-8"))
            names = {suite["name"] for suite in config["suites"]}
            with self.subTest(config=name):
                self.assertNotEqual(names, eval_suite.names("v1"))

    def test_every_required_suite_is_pinned(self) -> None:
        for suite in eval_suite.load("v1")["suites"]:
            with self.subTest(suite=suite["name"]):
                for field, value in suite["pins"].items():
                    self.assertNotIn("REPLACE_", value, field)
