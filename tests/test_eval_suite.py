"""One file says what the protocol measures, so this checks nothing else does.

The failure this replaces was not exotic. The suite set lived in a Python
constant, in whichever `paired-N.json` a job pointed at, in a `PAIRED_SUITES`
string, and in whatever rows reached the comparator, and those four drifted:
RULER was required by the runner and missing from the config that ran everything
else, while `aa_lcr` and `aa_omniscience` were scored into a macro they were
never part of. Both directions are tested here.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import eval_suite  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "run_eval_protocol", ROOT / "scripts" / "run_eval_protocol.py"
)
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)


def write_suite(root: Path, version: str, names, declared=None, **overrides) -> Path:
    path = root / "eval" / f"eval-suite-{version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "eval_suite": declared or version,
        "suites": [{"name": n, "replicates": 1, "pins": {}} for n in names],
    }
    document.update(overrides)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class LoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_the_shipped_definition_loads(self) -> None:
        document = eval_suite.load("v1")
        self.assertEqual(document["eval_suite"], "v1")
        self.assertEqual(len(document["suites"]), 7)

    def test_a_file_that_disagrees_with_its_name_is_refused(self) -> None:
        """The whole point is that a name and its contents cannot diverge."""
        write_suite(self.root, "v2", ["a"], declared="v1")
        with self.assertRaises(eval_suite.EvalSuiteError) as caught:
            eval_suite.load("v2", root=self.root)
        self.assertIn("declares eval_suite", str(caught.exception))

    def test_a_missing_version_says_which_file(self) -> None:
        with self.assertRaises(eval_suite.EvalSuiteError) as caught:
            eval_suite.load("v9", root=self.root)
        self.assertIn("eval-suite-v9.json", str(caught.exception))

    def test_a_malformed_version_string_is_refused(self) -> None:
        for version in ("1", "latest", "v", "vX", "../v1"):
            with self.subTest(version=version), self.assertRaises(eval_suite.EvalSuiteError):
                eval_suite.path_for(version)

    def test_a_repeated_suite_is_refused(self) -> None:
        write_suite(self.root, "v3", ["a", "a"])
        with self.assertRaises(eval_suite.EvalSuiteError):
            eval_suite.load("v3", root=self.root)

    def test_a_suite_without_replicates_is_refused(self) -> None:
        path = write_suite(self.root, "v4", ["a"])
        document = json.loads(path.read_text(encoding="utf-8"))
        document["suites"][0]["replicates"] = 0
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(eval_suite.EvalSuiteError):
            eval_suite.load("v4", root=self.root)

    def test_an_empty_definition_is_refused(self) -> None:
        write_suite(self.root, "v5", [])
        with self.assertRaises(eval_suite.EvalSuiteError):
            eval_suite.load("v5", root=self.root)


class DerivationTests(unittest.TestCase):
    def test_the_runner_holds_no_suite_set_of_its_own(self) -> None:
        self.assertEqual(protocol.REQUIRED_SUITES, eval_suite.names("v1"))

    def test_every_suite_runs_once(self) -> None:
        self.assertEqual(set(eval_suite.replicates("v1").values()), {1})

    def test_the_parked_suites_are_recorded_with_reasons(self) -> None:
        """A suite that was considered and left out is a decision, not an absence."""
        parked = eval_suite.load("v1")["rationale"]["parked"]
        for name in ("terminal_bench_2_1", "swebench_pro_1_0"):
            with self.subTest(name=name):
                self.assertNotIn(name, eval_suite.names("v1"))
                self.assertGreater(len(parked[name]), 40)


class SelectionTests(unittest.TestCase):
    def test_a_batch_is_a_subset(self) -> None:
        self.assertEqual(eval_suite.select(["ruler", "bfcl_v4"]), ["bfcl_v4", "ruler"])

    def test_selection_follows_the_definition_order(self) -> None:
        chosen = eval_suite.select(["ruler", "gpqa_diamond", "bfcl_v4"])
        self.assertEqual(chosen, ["bfcl_v4", "gpqa_diamond", "ruler"])

    def test_a_suite_outside_the_protocol_is_refused(self) -> None:
        """aa_lcr reached a macro this way; naming it must now fail loudly."""
        with self.assertRaises(eval_suite.EvalSuiteError) as caught:
            eval_suite.select(["aa_lcr", "ruler"])
        self.assertIn("aa_lcr", str(caught.exception))
        self.assertIn("does not add to it", str(caught.exception))

    def test_selecting_everything_returns_the_whole_protocol(self) -> None:
        everything = sorted(eval_suite.names("v1"))
        self.assertEqual(sorted(eval_suite.select(everything)), everything)

    def test_missing_reports_what_a_result_set_lacks(self) -> None:
        present = eval_suite.names("v1") - {"ruler"}
        self.assertEqual(eval_suite.missing(present), ["ruler"])


class CommandTests(unittest.TestCase):
    def test_listing_succeeds(self) -> None:
        self.assertEqual(eval_suite.main([]), 0)

    def test_selecting_succeeds(self) -> None:
        self.assertEqual(eval_suite.main(["--select", "ruler"]), 0)


if __name__ == "__main__":
    unittest.main()
