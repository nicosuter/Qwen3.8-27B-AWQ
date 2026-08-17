import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class AuditOverlapTests(unittest.TestCase):
    def test_auto_field_includes_vision_user_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calibration = directory / "calibration.jsonl"
            evaluation = directory / "eval.jsonl"
            report = directory / "report.json"
            shared = "Which station is visible beside the red train on platform seven today"
            calibration.write_text(
                json.dumps(
                    {"kind": "vision", "user": shared + " in this photograph"}
                )
                + "\n",
                encoding="utf-8",
            )
            evaluation.write_text(
                json.dumps({"id": "vision-1", "text": shared}) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "eval" / "scripts" / "audit_eval_overlap.py"),
                    "--calibration",
                    str(calibration),
                    "--eval",
                    str(evaluation),
                    "--output",
                    str(report),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            matches = json.loads(report.read_text(encoding="utf-8"))["matches"]
            self.assertEqual(matches[0]["eval_id"], "vision-1")


if __name__ == "__main__":
    unittest.main()
