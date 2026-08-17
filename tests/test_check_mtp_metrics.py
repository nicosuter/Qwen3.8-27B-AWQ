import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "check_mtp_metrics", ROOT / "common" / "scripts" / "check_mtp_metrics.py"
)
assert SPEC and SPEC.loader
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def snapshot(accepted: int, drafted: int, *, suffix: str = "_total") -> str:
    return (
        f'vllm:spec_decode_num_accepted_tokens{suffix}{{model_name="test"}} '
        f"{accepted}.0\n"
        f'vllm:spec_decode_num_draft_tokens{suffix}{{model_name="test"}} '
        f"{drafted}.0\n"
    )


class MtpMetricsTests(unittest.TestCase):
    def test_uses_counter_deltas(self) -> None:
        report = metrics.acceptance_report(snapshot(10, 20), snapshot(55, 120))
        self.assertEqual(report["accepted_draft_tokens"], 45)
        self.assertEqual(report["draft_tokens"], 100)
        self.assertEqual(report["acceptance_rate"], 0.45)
        self.assertTrue(report["passed"])

    def test_accepts_metric_names_without_total_suffix(self) -> None:
        report = metrics.acceptance_report(
            snapshot(0, 0, suffix=""), snapshot(4, 10, suffix="")
        )
        self.assertTrue(report["passed"])

    def test_rejects_missing_or_zero_draft_counters(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not expose"):
            metrics.acceptance_report("", "")
        with self.assertRaisesRegex(ValueError, "no draft tokens"):
            metrics.acceptance_report(snapshot(0, 0), snapshot(0, 0))

    def test_reports_acceptance_below_threshold(self) -> None:
        report = metrics.acceptance_report(snapshot(0, 0), snapshot(3, 10))
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
