import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "compare_mtp_results", ROOT / "scripts" / "compare_mtp_results.py"
)
assert SPEC and SPEC.loader
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def row(
    *,
    score: float = 1.0,
    failed: bool = False,
    elapsed_seconds: float = 1.0,
    output_tokens: int = 10,
    accepted_draft_tokens: int = 0,
    draft_tokens: int = 0,
) -> dict:
    return {
        "score": score,
        "failed": failed,
        "elapsed_seconds": elapsed_seconds,
        "output_tokens": output_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "draft_tokens": draft_tokens,
    }


class ComparisonTests(unittest.TestCase):
    def test_acceptance_is_weighted_by_drafted_tokens(self) -> None:
        disabled = {"short": row(), "long": row()}
        enabled = {
            "short": row(accepted_draft_tokens=1, draft_tokens=1),
            "long": row(accepted_draft_tokens=39, draft_tokens=99),
        }

        report = comparison.compare(disabled, enabled)

        self.assertEqual(report["accepted_draft_tokens"], 40)
        self.assertEqual(report["draft_tokens"], 100)
        self.assertEqual(report["acceptance_rate"], 0.40)
        self.assertTrue(report["gate"]["passed"])

    def test_server_delta_can_be_recorded_exactly_once(self) -> None:
        disabled = {"a": row(), "b": row()}
        enabled = {
            "a": row(accepted_draft_tokens=45, draft_tokens=100),
            "b": row(accepted_draft_tokens=0, draft_tokens=0),
        }

        report = comparison.compare(disabled, enabled)

        self.assertEqual(report["acceptance_rate"], 0.45)
        self.assertTrue(report["gate"]["passed"])

    def test_rejects_no_drafts_across_enabled_run(self) -> None:
        disabled = {"a": row()}
        enabled = {"a": row()}
        with self.assertRaisesRegex(ValueError, "no drafted tokens"):
            comparison.compare(disabled, enabled)

    def test_rejects_mismatched_request_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "item IDs differ"):
            comparison.compare(
                {"a": row()},
                {"b": row(accepted_draft_tokens=1, draft_tokens=1)},
            )

    def test_all_three_gates_fail_closed(self) -> None:
        disabled = {"a": row(score=1.0, failed=False)}
        enabled = {
            "a": row(
                score=0.5,
                failed=True,
                accepted_draft_tokens=1,
                draft_tokens=10,
            )
        }

        report = comparison.compare(disabled, enabled)

        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(
            report["gate"]["failures"], ["quality", "failures", "acceptance"]
        )

    def test_zero_output_does_not_create_non_json_speed_values(self) -> None:
        disabled = {"a": row(output_tokens=0)}
        enabled = {
            "a": row(
                output_tokens=0,
                accepted_draft_tokens=1,
                draft_tokens=1,
            )
        }

        report = comparison.compare(disabled, enabled)

        self.assertEqual(report["latency_derived_tokens_per_second"]["disabled"], 0)
        self.assertIsNone(report["latency_derived_speed_ratio"])
        json.dumps(report, allow_nan=False)


class LoadTests(unittest.TestCase):
    def write_rows(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / "results.jsonl"
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
        )
        return path

    def test_enabled_rows_require_explicit_acceptance_counters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = {
                "id": "a",
                "score": 1.0,
                "failed": False,
                "elapsed_seconds": 1.0,
                "output_tokens": 1,
            }
            path = self.write_rows(Path(temporary), [data])
            with self.assertRaisesRegex(ValueError, "accepted_draft_tokens"):
                comparison.load(path, mtp=True)

    def test_rejects_non_finite_elapsed_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = {
                "id": "a",
                "score": 1.0,
                "failed": False,
                "elapsed_seconds": math.inf,
                "output_tokens": 1,
            }
            path = self.write_rows(Path(temporary), [data])
            with self.assertRaisesRegex(ValueError, "invalid or duplicate metrics"):
                comparison.load(path, mtp=False)

    def test_rejects_missing_failure_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = {
                "id": "a",
                "score": 1.0,
                "elapsed_seconds": 1.0,
                "output_tokens": 1,
            }
            path = self.write_rows(Path(temporary), [data])
            with self.assertRaisesRegex(ValueError, "failed"):
                comparison.load(path, mtp=False)

    def test_rejects_coerced_counter_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = {
                "id": "a",
                "score": 1.0,
                "failed": False,
                "elapsed_seconds": 1.0,
                "output_tokens": 1,
                "accepted_draft_tokens": 0.5,
                "draft_tokens": 1,
            }
            path = self.write_rows(Path(temporary), [data])
            with self.assertRaisesRegex(ValueError, "counters must be integers"):
                comparison.load(path, mtp=True)


if __name__ == "__main__":
    unittest.main()
