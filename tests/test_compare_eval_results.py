import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "compare_eval_results", ROOT / "scripts" / "compare_eval_results.py"
)
assert SPEC and SPEC.loader
comparator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparator)

FLAGS = (
    "timeout",
    "empty_answer",
    "repetition_loop",
    "malformed_tool_call",
    "premature_final_answer",
    "context_failure",
)


def row(suite, item, replicate, score, must_pass=False, **flags):
    data = {
        "suite": suite,
        "id": item,
        "replicate": replicate,
        "score": score,
        "must_pass": must_pass,
    }
    data.update({flag: False for flag in FLAGS})
    data.update(flags)
    return data


def run_gate(baseline_rows, candidate_rows, *extra):
    tmp = tempfile.mkdtemp()
    base = Path(tmp) / "base.jsonl"
    cand = Path(tmp) / "cand.jsonl"
    out = Path(tmp) / "report.json"
    base.write_text("\n".join(json.dumps(r) for r in baseline_rows), encoding="utf-8")
    cand.write_text("\n".join(json.dumps(r) for r in candidate_rows), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_eval_results.py"),
         "--baseline", str(base), "--candidate", str(cand), "--output", str(out),
         "--bootstrap-samples", "800", *extra],
        capture_output=True, text=True,
    )
    return json.loads(out.read_text())["gate"], proc.stdout


def noisy_pair(suites, drops=None, seed=5, flags_for=None):
    """Paired runs where item difficulty is shared and sampling is independent."""
    rng = random.Random(seed)
    drops = drops or {}
    baseline, candidate = [], []
    for suite, items, reps in suites:
        for index in range(items):
            q = rng.uniform(0.55, 0.95)
            for replicate in range(reps):
                baseline.append(row(suite, f"{suite}-{index}", replicate, float(rng.random() < q)))
                drop = drops.get(suite, 0.0)
                extra = flags_for(suite, index) if flags_for else {}
                score = 0.0 if extra else float(rng.random() < max(0.0, q - drop))
                candidate.append(row(suite, f"{suite}-{index}", replicate, score, **extra))
    return baseline, candidate


SEVEN = [("bfcl_v4", 300, 1), ("terminal_bench_2_1", 90, 3), ("livecodebench_v6", 175, 4),
         ("gpqa_diamond", 198, 4), ("matharena_2026_06", 60, 4), ("multimodal", 400, 1),
         ("ruler", 210, 1)]


class MustPassTests(unittest.TestCase):
    def test_partial_credit_is_not_a_pass(self) -> None:
        # The old rule was score > 0, so a task dropping from full reward to a
        # sliver of partial credit counted as retained.
        base = [row("terminal_bench_2_1", f"t{i}", r, 1.0, must_pass=True)
                for i in range(40) for r in range(3)]
        cand = [row("terminal_bench_2_1", f"t{i}", r, 0.05, must_pass=True)
                for i in range(40) for r in range(3)]
        gate, _ = run_gate(base, cand)
        self.assertEqual(gate["must_pass_retention"], 0.0)
        self.assertTrue(gate["must_pass_failure"])
        self.assertFalse(gate["passed"])

    def test_retention_counts_items_not_replicate_rows(self) -> None:
        base = [row("terminal_bench_2_1", f"t{i}", r, 1.0, must_pass=True)
                for i in range(40) for r in range(3)]
        gate, _ = run_gate(base, list(base))
        self.assertEqual(gate["must_pass_baseline_passes"], 40)
        self.assertEqual(gate["must_pass_retention"], 1.0)

    def test_bank_the_baseline_never_passes_is_reported(self) -> None:
        base = [row("terminal_bench_2_1", f"t{i}", 0, 0.4, must_pass=True) for i in range(20)]
        gate, stdout = run_gate(base, list(base))
        self.assertTrue(gate["must_pass_unusable"])
        self.assertIn("must-pass bank unusable", stdout)


class FailureModeTests(unittest.TestCase):
    def test_regression_confined_to_one_suite_is_caught(self) -> None:
        # Pooling every row diluted a 15-point RULER context_failure regression
        # to 0.86 points, under the 1-point threshold.
        def flags(suite, index):
            return {"context_failure": True} if suite == "ruler" and index < 32 else {}

        base, cand = noisy_pair(SEVEN, flags_for=flags)
        gate, stdout = run_gate(base, cand)
        self.assertIn("context_failure", gate["failure_rate_failures"])
        self.assertEqual(gate["failure_rates"]["context_failure"]["worst_suite"], "ruler")
        self.assertIn("worst in ruler", stdout)


class QualityGateTests(unittest.TestCase):
    def test_noise_alone_passes(self) -> None:
        # The old per-suite point rule failed 60% of clean runs.
        base, cand = noisy_pair(SEVEN, seed=11)
        gate, _ = run_gate(base, cand)
        self.assertFalse(gate["macro_point_failure"])
        self.assertEqual(gate["suite_confident_failures"], [])
        self.assertTrue(gate["passed"])

    def test_broad_degradation_fails_on_the_macro(self) -> None:
        base, cand = noisy_pair(SEVEN, drops={s: 0.08 for s, *_ in SEVEN}, seed=12)
        gate, _ = run_gate(base, cand)
        self.assertTrue(gate["macro_point_failure"])
        self.assertFalse(gate["passed"])

    def test_one_catastrophic_suite_still_fails_the_run(self) -> None:
        # The macro would absorb it: 30 points on one of seven suites is 4.3.
        base, cand = noisy_pair(SEVEN, drops={"ruler": 0.30}, seed=13)
        gate, _ = run_gate(base, cand)
        self.assertIn("ruler", gate["suite_confident_failures"])
        self.assertFalse(gate["passed"])

    def test_small_suite_dip_is_a_review_flag_not_a_failure(self) -> None:
        # A 5-point dip on a 60-item suite: past the review line, but its
        # interval cannot rule out a drop smaller than the 5-point hard margin,
        # and the macro across two suites is only -2.5.
        base = [row("matharena_2026_06", f"m{i}", 0, 1.0) for i in range(60)]
        base += [row("gpqa_diamond", f"g{i}", 0, 1.0) for i in range(198)]
        cand = [row("matharena_2026_06", f"m{i}", 0, 0.0 if i < 3 else 1.0) for i in range(60)]
        cand += [row("gpqa_diamond", f"g{i}", 0, 1.0) for i in range(198)]
        gate, stdout = run_gate(base, cand)
        self.assertIn("matharena_2026_06", gate["suite_review_flags"])
        self.assertNotIn("matharena_2026_06", gate["suite_confident_failures"])
        self.assertIn("review (not a failure)", stdout)


class BaselineFloorTests(unittest.TestCase):
    def test_harness_broken_for_both_checkpoints_is_caught(self) -> None:
        # Everything else is paired, so both models scoring 40 looks perfect.
        base, cand = noisy_pair([("gpqa_diamond", 198, 4)], seed=15)
        broken_base = [dict(r, score=float(r["score"] and random.random() < 0.45)) for r in base]
        broken_cand = [dict(r, score=float(r["score"] and random.random() < 0.45)) for r in cand]
        gate, stdout = run_gate(broken_base, broken_cand, "--baseline-floor", "gpqa_diamond=0.80")
        self.assertIn("gpqa_diamond", gate["baseline_floor_failures"])
        self.assertFalse(gate["passed"])
        self.assertIn("may be broken for both", stdout)

    def test_healthy_baseline_clears_the_floor(self) -> None:
        base, cand = noisy_pair([("gpqa_diamond", 198, 4)], seed=16)
        gate, _ = run_gate(base, cand, "--baseline-floor", "gpqa_diamond=0.50")
        self.assertEqual(gate["baseline_floor_failures"], {})
        self.assertTrue(gate["passed"])

    def test_floor_for_a_missing_suite_fails_loudly(self) -> None:
        base, cand = noisy_pair([("gpqa_diamond", 20, 1)], seed=17)
        gate, stdout = run_gate(base, cand, "--baseline-floor", "livecodebench_v6=0.80")
        self.assertEqual(gate["baseline_floor_missing_suites"], ["livecodebench_v6"])
        self.assertFalse(gate["passed"])
        self.assertIn("no results for livecodebench_v6", stdout)

    def test_floor_syntax_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            comparator.parse_floors(["gpqa_diamond"])
        self.assertEqual(comparator.parse_floors(["a=0.5"]), {"a": 0.5})


if __name__ == "__main__":
    unittest.main()


class DeferredRowTests(unittest.TestCase):
    """A deferred zero means "not executed", and must never average as a failure."""

    def compare(self, baseline_rows, candidate_rows):
        tmp = tempfile.mkdtemp()
        base = Path(tmp) / "base.jsonl"
        cand = Path(tmp) / "cand.jsonl"
        out = Path(tmp) / "report.json"
        base.write_text("\n".join(json.dumps(r) for r in baseline_rows), encoding="utf-8")
        cand.write_text("\n".join(json.dumps(r) for r in candidate_rows), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "compare_eval_results.py"),
             "--baseline", str(base), "--candidate", str(cand), "--output", str(out)],
            capture_output=True, text=True,
        )

    def test_a_deferred_row_stops_the_comparison(self) -> None:
        baseline = [row("livecodebench_v6", f"i{i}", 0, 1.0) for i in range(4)]
        candidate = [row("livecodebench_v6", f"i{i}", 0, 1.0) for i in range(4)]
        candidate[2]["deferred"] = True
        candidate[2]["score"] = 0.0
        proc = self.compare(baseline, candidate)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("deferred", proc.stderr + proc.stdout)

    def test_scored_rows_carrying_deferred_false_are_fine(self) -> None:
        baseline = [row("livecodebench_v6", f"i{i}", 0, 1.0, deferred=False) for i in range(4)]
        candidate = [row("livecodebench_v6", f"i{i}", 0, 1.0, deferred=False) for i in range(4)]
        proc = self.compare(baseline, candidate)
        self.assertEqual(proc.returncode, 0, proc.stderr)
