import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = load_module("evalscope_bridge", "scripts/evalscope_bridge.py")

PIN = "cais/hle@" + "a" * 40


def review(sample_id, *, group_id=None, acc=1.0, text="q", target="18",
           extracted="18", value=None, main=None):
    return {
        "index": str(sample_id),
        "input": text,
        "target": target,
        "sample_score": {
            "score": {
                "value": {"acc": acc} if value is None else value,
                "extracted_prediction": extracted,
                "prediction": extracted,
                "main_score_name": main,
                "metadata": {},
            },
            "sample_id": sample_id,
            "group_id": sample_id if group_id is None else group_id,
            "sample_metadata": {},
        },
    }


class ConvertTests(unittest.TestCase):
    def test_a_review_becomes_a_result_row(self) -> None:
        rows = bridge.convert([review(0)], suite="hle", dataset_pin=PIN)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["suite"], "hle")
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["replicate"], 0)
        self.assertEqual(row["dataset_pin"], PIN)
        for field in bridge.BOOL_FIELDS:
            self.assertIn(field, row)

    def test_the_id_is_derived_from_the_item_not_its_position(self) -> None:
        # EvalScope's sample id is positional. A dataset that shifted would
        # otherwise join cleanly and compare different questions.
        same = bridge.convert([review(0, text="q1")], suite="s", dataset_pin=PIN)
        moved = bridge.convert([review(7, text="q1")], suite="s", dataset_pin=PIN)
        self.assertNotEqual(same[0]["id"], moved[0]["id"])
        self.assertEqual(same[0]["id"].split(":")[1], moved[0]["id"].split(":")[1])

    def test_two_arms_on_the_same_item_produce_the_same_id(self) -> None:
        a = bridge.convert([review(3, acc=1.0)], suite="s", dataset_pin=PIN)
        b = bridge.convert([review(3, acc=0.0)], suite="s", dataset_pin=PIN)
        self.assertEqual(a[0]["id"], b[0]["id"])
        self.assertNotEqual(a[0]["score"], b[0]["score"])

    def test_a_changed_question_changes_the_id(self) -> None:
        a = bridge.convert([review(3, text="q1")], suite="s", dataset_pin=PIN)
        b = bridge.convert([review(3, text="q2")], suite="s", dataset_pin=PIN)
        self.assertNotEqual(a[0]["id"], b[0]["id"])

    def test_completion_order_does_not_change_the_rows(self) -> None:
        # Reviews are written in completion order, not item order.
        forward = bridge.convert(
            [review(0, text="a"), review(1, text="b")], suite="s", dataset_pin=PIN
        )
        backward = bridge.convert(
            [review(1, text="b"), review(0, text="a")], suite="s", dataset_pin=PIN
        )
        self.assertEqual(
            sorted(r["id"] for r in forward), sorted(r["id"] for r in backward)
        )

    def test_repeats_become_replicates(self) -> None:
        records = [review(i, group_id=5, text="q") for i in (2, 0, 1)]
        rows = bridge.convert(records, suite="s", dataset_pin=PIN)
        self.assertEqual(sorted(r["replicate"] for r in rows), [0, 1, 2])
        self.assertEqual(len({r["id"] for r in rows}), 1, "one item, three draws")

    def test_an_empty_extraction_is_flagged(self) -> None:
        rows = bridge.convert([review(0, extracted="  ")], suite="s", dataset_pin=PIN)
        self.assertTrue(rows[0]["empty_answer"])
        rows = bridge.convert([review(0, extracted="18")], suite="s", dataset_pin=PIN)
        self.assertFalse(rows[0]["empty_answer"])

    def test_the_named_main_metric_wins(self) -> None:
        rec = review(0, value={"pass": 0.0, "acc": 1.0}, main="acc")
        self.assertEqual(bridge.convert([rec], suite="s", dataset_pin=PIN)[0]["score"], 1.0)

    def test_an_explicit_metric_overrides(self) -> None:
        rec = review(0, value={"acc": 1.0, "f1": 0.5})
        rows = bridge.convert([rec], suite="s", dataset_pin=PIN, metric="f1")
        self.assertEqual(rows[0]["score"], 0.5)

    def test_a_score_outside_the_unit_interval_is_refused(self) -> None:
        # Rescaling here would quietly change what a recovery ratio means.
        rec = review(0, value={"bleu": 42.0})
        with self.assertRaises(bridge.BridgeError):
            bridge.convert([rec], suite="s", dataset_pin=PIN)

    def test_a_missing_metric_is_refused(self) -> None:
        with self.assertRaises(bridge.BridgeError):
            bridge.convert([review(0)], suite="s", dataset_pin=PIN, metric="nope")

    def test_an_unpinned_dataset_is_refused(self) -> None:
        for pin in ("", "cais/hle", "cais/hle@main", "a" * 40):
            with self.subTest(pin=pin):
                with self.assertRaises(bridge.BridgeError):
                    bridge.convert([review(0)], suite="s", dataset_pin=pin)

    def test_a_row_without_a_sample_score_is_refused(self) -> None:
        with self.assertRaises(bridge.BridgeError):
            bridge.convert([{"input": "q", "target": "a"}], suite="s", dataset_pin=PIN)


class ComparatorContractTests(unittest.TestCase):
    """The rows have to survive compare_eval_results.py's own loader."""

    def test_rows_load_in_the_comparator(self) -> None:
        comparator = load_module("compare_eval", "scripts/compare_eval_results.py")
        rows = bridge.convert(
            [review(0, text="a", acc=1.0), review(1, text="b", acc=0.0)],
            suite="hle", dataset_pin=PIN,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            loaded = comparator.load_rows(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual({v["score"] for v in loaded.values()}, {0.0, 1.0})


class MaterializeTests(unittest.TestCase):
    def test_a_branch_is_not_a_pin(self) -> None:
        args = bridge.parse_args(
            ["materialize", "--repo", "cais/hle", "--revision", "main", "--into", "/tmp/x"]
        )
        with self.assertRaises(bridge.BridgeError):
            bridge.command_materialize(args)



class SuiteMapTests(unittest.TestCase):
    """The shipped port map has to stay usable and honest."""

    PATH = ROOT / "eval" / "evalscope-suites.json"

    def spec(self):
        return json.loads(self.PATH.read_text(encoding="utf-8"))

    def test_every_ported_suite_carries_a_pinned_revision(self) -> None:
        for entry in self.spec()["suites"]:
            if entry.get("ported"):
                with self.subTest(suite=entry["suite"]):
                    self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
                    self.assertIn("/", entry["repo"])

    def test_every_unported_suite_says_why(self) -> None:
        for entry in self.spec()["suites"]:
            if not entry.get("ported"):
                with self.subTest(suite=entry["suite"]):
                    self.assertTrue(entry.get("note"), "an exclusion needs a reason")

    def test_loading_an_unported_suite_is_refused(self) -> None:
        # Written against a synthetic entry rather than the shipped map: every
        # suite is ported now, and this guards the mechanism, not the contents.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suites.json"
            path.write_text(json.dumps({"suites": [
                {"suite": "held", "benchmark": "b", "repo": "o/n",
                 "revision": "a" * 40, "ported": False, "note": "why not"}
            ]}))
            with self.assertRaises(bridge.BridgeError) as caught:
                bridge.load_suite(path, "held")
        self.assertIn("why not", str(caught.exception))

    def test_a_ported_suite_without_a_pin_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suites.json"
            path.write_text(json.dumps({"suites": [
                {"suite": "loose", "benchmark": "b", "repo": "o/n",
                 "revision": "main", "ported": True}
            ]}))
            with self.assertRaises(bridge.BridgeError):
                bridge.load_suite(path, "loose")

    def test_loading_a_ported_suite_returns_its_pin(self) -> None:
        ported = [e["suite"] for e in self.spec()["suites"] if e.get("ported")]
        entry = bridge.load_suite(self.PATH, ported[0])
        self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")

    def test_an_unknown_suite_lists_what_exists(self) -> None:
        with self.assertRaises(bridge.BridgeError) as caught:
            bridge.load_suite(self.PATH, "not_a_suite")
        self.assertIn("mmlu_pro", str(caught.exception))

    def test_the_run_task_points_at_the_materialized_pin(self) -> None:
        import io, contextlib
        args = bridge.parse_args([
            "run", "--suite", "mmlu_pro", "--model", "m", "--api-url", "http://x/v1",
            "--work-dir", "/tmp/es", "--variant", "baseline", "--print-only",
        ])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(args)
        plan = json.loads(out.getvalue())
        task = plan["task"]
        self.assertEqual(task["dataset_hub"], "local")
        self.assertIn(plan["pin"].split("@")[1], task["dataset_args"]["mmlu_pro"]["dataset_id"])
        # The cap has to match the protocol's, or the comparison is confounded
        # by one side getting a different budget.
        self.assertEqual(task["generation_config"]["max_tokens"], 131072)

class BfclDatasetTests(unittest.TestCase):
    """Reshaping upstream BFCL into the layout EvalScope's adapter reads."""

    ROW = {"id": "simple_0", "question": [[{"role": "user", "content": "q"}]],
           "function": [{"name": "f", "description": "d",
                         "parameters": {"type": "dict", "properties": {}}}]}
    ANSWER = {"id": "simple_0", "ground_truth": [{"f": {}}]}

    @staticmethod
    def build_tools(functions):
        return [{"type": "function", "function": {"name": f["name"]}} for f in functions]

    def build(self, by_cat, answers):
        return bridge.bfcl_rows(by_cat, answers, self.build_tools)

    def test_prompt_and_key_are_merged_into_one_record(self) -> None:
        rows = self.build({"simple": [self.ROW]}, {"simple_0": self.ANSWER})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["subset"], "simple")
        self.assertFalse(row["multi_turn"])
        # Every structured field is a JSON string, which is what the adapter
        # json.loads back out in preprocess_row.
        for field in ("functions", "tools", "turns", "missed_functions",
                      "initial_config", "ground_truth"):
            with self.subTest(field=field):
                self.assertIsInstance(row[field], str)
                json.loads(row[field])
        self.assertEqual(json.loads(row["ground_truth"]), [{"f": {}}])

    def test_a_decision_only_category_needs_no_key(self) -> None:
        rows = self.build({"irrelevance": [dict(self.ROW, id="irrelevance_0")]}, {})
        self.assertEqual(json.loads(rows[0]["ground_truth"]), {})

    def test_an_unkeyed_item_is_dropped(self) -> None:
        with self.assertRaises(bridge.BridgeError):
            self.build({"simple": [self.ROW]}, {})

    def test_a_toolless_item_is_dropped(self) -> None:
        with self.assertRaises(bridge.BridgeError):
            self.build({"irrelevance": [dict(self.ROW, id="x", function=[])]}, {})

    def test_multi_turn_is_refused_rather_than_faked(self) -> None:
        # Those rows need an initial_config and the simulators' state.
        with self.assertRaises(bridge.BridgeError):
            self.build({"multi_turn_base": [self.ROW]}, {})

    def test_the_ast_categories_match_our_adapter(self) -> None:
        ours = load_module("bfcl_adapter", "scripts/adapters/bfcl.py")
        self.assertEqual(set(bridge.BFCL_AST_CATEGORIES), set(ours.CATEGORIES))
        self.assertEqual(set(bridge.BFCL_NO_GROUND_TRUTH), set(ours.NO_GROUND_TRUTH))

class DeferredReviewTests(unittest.TestCase):
    """A deferred review is an unscored item, never a failed one."""

    def deferred(self, sample_id=0):
        rec = review(sample_id)
        rec["sample_score"]["score"]["metadata"] = {
            "deferred": True, "execution_method": "deferred"
        }
        return rec

    def test_a_deferred_review_is_marked(self) -> None:
        rows = bridge.convert([self.deferred()], suite="lcb", dataset_pin=PIN)
        self.assertTrue(rows[0]["deferred"])

    def test_a_scored_review_is_not_marked(self) -> None:
        rows = bridge.convert([review(0)], suite="lcb", dataset_pin=PIN)
        self.assertNotIn("deferred", rows[0])

    def test_the_comparator_refuses_a_deferred_row(self) -> None:
        # The whole point: a generating pass that did not execute cannot be
        # mistaken for a suite that scored zero.
        comparator = load_module("compare_eval2", "scripts/compare_eval_results.py")
        rows = bridge.convert([self.deferred()], suite="lcb", dataset_pin=PIN)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError) as caught:
                comparator.load_rows(path)
        self.assertIn("deferred", str(caught.exception))

class GenerationSeedTests(unittest.TestCase):
    def plan(self):
        import io, contextlib
        args = bridge.parse_args([
            "run", "--suite", "mmlu_pro", "--model", "m", "--api-url", "http://x/v1",
            "--work-dir", "/tmp/es", "--variant", "baseline", "--print-only",
        ])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(args)
        return json.loads(out.getvalue())["task"]

    def test_no_per_request_seed_is_sent(self) -> None:
        """One seed on every request correlates the items' sampling noise.

        The bootstrap resamples items assuming they are independent draws; a
        shared uniform stream breaks that and narrows the interval below the
        truth. EvalScope never copies TaskConfig.seed into GenerateConfig, and
        we must not add it.
        """
        self.assertNotIn("seed", self.plan()["generation_config"])

    def test_the_task_seed_is_still_recorded(self) -> None:
        # Harmless and useful: it seeds the local RNG and any shuffling.
        self.assertIn("seed", self.plan())

try:
    from evalscope.api.registry import BENCHMARK_REGISTRY as _REGISTRY
except Exception:  # noqa: BLE001 - evalscope is only installed where runs happen
    _REGISTRY = None


@unittest.skipUnless(_REGISTRY, "evalscope is not installed in this environment")
class RegistryAgreementTests(unittest.TestCase):
    """The port map has to agree with the EvalScope actually installed.

    Confirmed against real reviews: a review's score.value keys are exactly the
    benchmark's declared metric_list, so the registry is the authority and this
    catches a version that renamed or moved one.
    """

    def spec(self):
        path = ROOT / "eval" / "evalscope-suites.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_every_named_benchmark_exists(self) -> None:
        for entry in self.spec()["suites"]:
            with self.subTest(suite=entry["suite"]):
                self.assertIn(entry["benchmark"], _REGISTRY)

    def test_every_metric_is_one_the_benchmark_declares(self) -> None:
        for entry in self.spec()["suites"]:
            metric = entry.get("metric")
            if not metric:
                continue
            declared = _REGISTRY[entry["benchmark"]].metric_list
            names = {m if isinstance(m, str) else next(iter(m)) for m in declared}
            with self.subTest(suite=entry["suite"]):
                self.assertIn(metric, names, f"{entry['benchmark']} declares {sorted(names)}")

class PinAgreementTests(unittest.TestCase):
    """Two arms on different dataset revisions is not a paired comparison."""

    def write(self, tmp, name, rows):
        path = Path(tmp) / name
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return path

    def compare(self, base_pin, cand_pin):
        comparator = load_module("compare_eval3", "scripts/compare_eval_results.py")
        a = bridge.convert([review(0, text="q", acc=1.0)], suite="s", dataset_pin=base_pin)
        b = bridge.convert([review(0, text="q", acc=1.0)], suite="s", dataset_pin=cand_pin)
        with tempfile.TemporaryDirectory() as tmp:
            baseline = comparator.load_rows(self.write(tmp, "b.jsonl", a))
            candidate = comparator.load_rows(self.write(tmp, "c.jsonl", b))
            return comparator, baseline, candidate

    def test_matching_pins_compare(self) -> None:
        comparator, baseline, candidate = self.compare(PIN, PIN)
        result = comparator.summarize(baseline, candidate, samples=8, seed=1)
        self.assertIn("suites", result)

    def test_differing_pins_are_refused(self) -> None:
        other = "cais/hle@" + "b" * 40
        comparator, baseline, candidate = self.compare(PIN, other)
        with self.assertRaises(ValueError) as caught:
            comparator.summarize(baseline, candidate, samples=8, seed=1)
        self.assertIn("different dataset revisions", str(caught.exception))

    def test_rows_without_a_pin_still_compare(self) -> None:
        # Our own adapters do not record one; absence must not break them.
        comparator = load_module("compare_eval4", "scripts/compare_eval_results.py")
        rows = [{"suite": "s", "id": "i", "replicate": 0, "score": 1.0}]
        with tempfile.TemporaryDirectory() as tmp:
            loaded = comparator.load_rows(self.write(tmp, "x.jsonl", rows))
        self.assertIsNone(next(iter(loaded.values()))["dataset_pin"])

class DatasetsRootTests(unittest.TestCase):
    """Lanes share one materialized copy; a per-lane root would re-download it."""

    def plan(self, extra_args=()):
        import io, contextlib
        args = bridge.parse_args([
            "run", "--suite", "mmlu_pro", "--model", "m", "--api-url", "http://x/v1",
            "--work-dir", "/tmp/lane-7", "--variant", "baseline", "--print-only",
            *extra_args,
        ])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(args)
        return json.loads(out.getvalue())

    def test_the_dataset_is_not_under_the_lane_work_dir(self) -> None:
        dataset = self.plan()["dataset"]
        self.assertNotIn("/tmp/lane-7", dataset)
        self.assertIn("eval-materialized/evalscope", dataset)

    def test_an_explicit_root_is_honoured(self) -> None:
        dataset = self.plan(["--datasets-root", "/shared/pins"])["dataset"]
        self.assertTrue(dataset.startswith("/shared/pins/"), dataset)

    def test_the_path_ends_in_the_pinned_revision(self) -> None:
        plan = self.plan()
        self.assertTrue(plan["dataset"].endswith(plan["pin"].split("@")[1]))

    def test_bfcl_reads_the_built_layout(self) -> None:
        import io, contextlib
        args = bridge.parse_args([
            "run", "--suite", "bfcl_v4", "--model", "m", "--api-url", "http://x/v1",
            "--work-dir", "/tmp/lane-7", "--variant", "baseline", "--print-only",
        ])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(args)
        # Built, not downloaded, so it does not live under the repo-named path.
        self.assertIn("/bfcl_v3/", json.loads(out.getvalue())["dataset"])

class ConcurrencyTests(unittest.TestCase):
    def plan(self, extra_args=()):
        import io, contextlib
        args = bridge.parse_args([
            "run", "--suite", "mmlu_pro", "--model", "m", "--api-url", "http://x/v1",
            "--work-dir", "/tmp/es", "--variant", "baseline", "--print-only",
            *extra_args,
        ])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(args)
        return json.loads(out.getvalue())["task"]

    def test_eval_batch_size_is_not_left_at_one(self) -> None:
        """EvalScope defaults it to 1, one request in flight.

        The smoke run showed a served suite crawling at a request per few
        seconds with the server otherwise idle; 12k items would take days.
        """
        self.assertGreater(self.plan()["eval_batch_size"], 1)

    def test_concurrency_is_passed_through(self) -> None:
        self.assertEqual(self.plan(["--concurrency", "128"])["eval_batch_size"], 128)

class DeferredTwoPassTests(unittest.TestCase):
    """Generate where it is not safe to execute; score where it is."""

    def plan(self, extra_args=()):
        import io, contextlib
        args = bridge.parse_args([
            "run", "--suite", "livecodebench_v6", "--model", "m",
            "--api-url", "http://x/v1", "--work-dir", "/tmp/es",
            "--variant", "baseline", "--print-only", *extra_args,
        ])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(args)
        return json.loads(out.getvalue())["task"]

    def params(self, task):
        return task["dataset_args"]["live_code_bench"].get("extra_params") or {}

    def test_the_generating_pass_does_not_execute(self) -> None:
        # The suites file pins execute=False for this one.
        self.assertFalse(self.params(self.plan())["execute"])

    def test_the_scoring_pass_reuses_predictions_and_executes(self) -> None:
        task = self.plan(["--use-cache", "/prev/run", "--rerun-review",
                          "--execute", "true"])
        self.assertEqual(task["use_cache"], "/prev/run")
        self.assertTrue(task["rerun_review"])
        self.assertTrue(self.params(task)["execute"])

    def test_the_plugin_is_named_so_the_run_loads_it(self) -> None:
        for suite in ("livecodebench_v6", "hle"):
            with self.subTest(suite=suite):
                entry = bridge.load_suite(ROOT / "eval" / "evalscope-suites.json", suite)
                self.assertTrue((ROOT / entry["plugin"]).is_file())

    def test_a_suite_without_a_plugin_is_unaffected(self) -> None:
        entry = bridge.load_suite(ROOT / "eval" / "evalscope-suites.json", "mmlu_pro")
        self.assertIsNone(entry.get("plugin"))

class SubsetQualifiedIdTests(unittest.TestCase):
    """Sample ids restart at zero in every subset."""

    def test_the_subset_qualifies_the_id(self) -> None:
        a = dict(review(0, text="q-cs"), _subset="computer science")
        b = dict(review(0, text="q-law"), _subset="law")
        rows = bridge.convert([a, b], suite="mmlu_pro", dataset_pin=PIN)
        ids = sorted(r["id"] for r in rows)
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(any(i.startswith("computer science/") for i in ids), ids)
        self.assertTrue(any(i.startswith("law/") for i in ids), ids)

    def test_the_same_item_in_both_arms_still_matches(self) -> None:
        rec = lambda acc: dict(review(0, text="q", acc=acc), _subset="law")
        a = bridge.convert([rec(1.0)], suite="s", dataset_pin=PIN)
        b = bridge.convert([rec(0.0)], suite="s", dataset_pin=PIN)
        self.assertEqual(a[0]["id"], b[0]["id"])

    def test_an_unqualified_record_still_works(self) -> None:
        rows = bridge.convert([review(0)], suite="s", dataset_pin=PIN)
        self.assertNotIn("/", rows[0]["id"])

class JudgeGuardTests(unittest.TestCase):
    """A judged suite must name its judge, or nothing runs.

    EvalScope's llm_judge defaults to Qwen/Qwen3-235B-A22B at
    https://api-inference.modelscope.cn/v1/. Left alone on a machine with a
    ModelScope token, HLE would send every reply to a third party and grade our
    gate with an unpinned remote model. It only failed loudly here because the
    cluster had no token.
    """

    BASE = ["run", "--model", "m", "--api-url", "http://x/v1",
            "--work-dir", "/tmp/es", "--variant", "baseline", "--print-only"]

    def plan(self, extra_args):
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(bridge.parse_args(self.BASE + list(extra_args)))
        return json.loads(out.getvalue())["task"]

    def test_a_scoring_pass_without_a_judge_is_refused(self) -> None:
        # Generating defers judging and needs none; turning judging on without
        # naming a judge is what must be refused.
        with self.assertRaises(bridge.BridgeError) as caught:
            self.plan(["--suite", "hle", "--execute", "true"])
        self.assertIn("modelscope", str(caught.exception).lower())

    def test_naming_a_judge_configures_it(self) -> None:
        task = self.plan(["--suite", "hle", "--execute", "true",
                          "--judge-model", "openai/gpt-oss-20b",
                          "--judge-api-url", "http://judge/v1"])
        self.assertEqual(task["judge_strategy"], "llm")
        self.assertEqual(task["judge_model_args"]["model_id"], "openai/gpt-oss-20b")
        self.assertEqual(task["judge_model_args"]["api_url"], "http://judge/v1")

    def test_half_a_judge_is_still_refused(self) -> None:
        for half in (["--judge-model", "m"], ["--judge-api-url", "http://j/v1"]):
            with self.subTest(half=half[0]):
                with self.assertRaises(bridge.BridgeError):
                    self.plan(["--suite", "hle", "--execute", "true", *half])

    def test_an_unjudged_suite_never_gets_judge_args(self) -> None:
        task = self.plan(["--suite", "mmlu_pro"])
        self.assertNotIn("judge_model_args", task)
        self.assertNotIn("judge_strategy", task)

    def test_naming_a_judge_for_an_unjudged_suite_is_refused(self) -> None:
        # Silently ignoring it would suggest a judge was in use when it was not.
        with self.assertRaises(bridge.BridgeError):
            self.plan(["--suite", "mmlu_pro", "--judge-model", "m",
                       "--judge-api-url", "http://j/v1"])

    def test_only_hle_is_marked_judged(self) -> None:
        spec = json.loads((ROOT / "eval" / "evalscope-suites.json").read_text())
        judged = {s["suite"] for s in spec["suites"] if s.get("judge_required")}
        self.assertEqual(judged, {"hle"})

class HleDeferredTests(unittest.TestCase):
    """HLE generates on the cluster before a judge has even been chosen."""

    BASE = ["run", "--model", "m", "--api-url", "http://x/v1",
            "--work-dir", "/tmp/es", "--variant", "baseline", "--print-only"]

    def plan(self, extra_args):
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(bridge.parse_args(self.BASE + list(extra_args)))
        return json.loads(out.getvalue())["task"]

    def test_generating_needs_no_judge(self) -> None:
        # Without deferral this is refused, because EvalScope would fall back to
        # a hosted third-party judge.
        task = self.plan(["--suite", "hle"])
        self.assertNotIn("judge_model_args", task)
        self.assertFalse(task["dataset_args"]["hle"]["extra_params"]["judge"])

    def test_scoring_turns_the_judge_on_and_reuses_predictions(self) -> None:
        task = self.plan(["--suite", "hle", "--use-cache", "/prev", "--rerun-review",
                          "--execute", "true", "--judge-model", "openai/gpt-oss-20b",
                          "--judge-api-url", "http://j/v1"])
        self.assertTrue(task["dataset_args"]["hle"]["extra_params"]["judge"])
        self.assertEqual(task["judge_model_args"]["model_id"], "openai/gpt-oss-20b")
        self.assertEqual(task["use_cache"], "/prev")
        self.assertTrue(task["rerun_review"])

    def test_each_suite_names_the_flag_its_adapter_reads(self) -> None:
        # LiveCodeBench gates execution, HLE gates its judge.
        spec = json.loads((ROOT / "eval" / "evalscope-suites.json").read_text())
        flags = {s["suite"]: s.get("defer_flag") for s in spec["suites"] if s.get("plugin")}
        self.assertEqual(flags, {"livecodebench_v6": "execute", "hle": "judge"})

    def test_execute_on_a_suite_with_no_flag_is_refused(self) -> None:
        with self.assertRaises(bridge.BridgeError):
            self.plan(["--suite", "mmlu_pro", "--execute", "false"])

class ResumeTests(unittest.TestCase):
    """A preempted lane must not lose what it already finished.

    EvalScope writes every prediction as it lands, so a killed lane has kept its
    work; it only needs pointing back at it. Our own adapters collect results in
    memory and write once at the end, so they lose the whole lane -- which for
    HLE at R=1 is about 31 hours of wall clock.
    """

    def plan(self, work_dir, extra_args=()):
        import io, contextlib
        args = bridge.parse_args([
            "run", "--suite", "mmlu_pro", "--model", "m", "--api-url", "http://x/v1",
            "--work-dir", str(work_dir), "--variant", "baseline", "--print-only",
            *extra_args,
        ])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bridge.command_run(args)
        text = out.getvalue()
        return json.loads(text[text.index("{"):])["task"]

    def partial(self, root):
        d = Path(root) / "runs" / "baseline" / "20260816_120000" / "predictions"
        d.mkdir(parents=True)
        return d

    def test_a_fresh_lane_does_not_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self.plan(Path(tmp) / "lane").get("use_cache"))

    def test_a_partial_lane_is_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp) / "lane"
            self.partial(lane)
            task = self.plan(lane)
            self.assertIsNotNone(task["use_cache"])
            # Reviews already computed stay computed; only predictions resume.
            self.assertFalse(task["rerun_review"])

    def test_no_resume_starts_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp) / "lane"
            self.partial(lane)
            self.assertIsNone(self.plan(lane, ["--no-resume"]).get("use_cache"))

    def test_an_explicit_cache_wins_over_autodetection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp) / "lane"
            self.partial(lane)
            task = self.plan(lane, ["--use-cache", "/elsewhere"])
            self.assertEqual(task["use_cache"], "/elsewhere")

    def test_a_directory_without_predictions_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp) / "lane"
            (lane / "runs" / "baseline" / "20260816_120000" / "logs").mkdir(parents=True)
            self.assertIsNone(self.plan(lane).get("use_cache"))


if __name__ == "__main__":
    unittest.main()
