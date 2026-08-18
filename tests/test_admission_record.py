"""What a run reserved against, recorded beside what it scored.

Concurrency was deliberately not recorded as something that had to match across
arms, on the reasoning that it changes scheduling rather than the distribution.
That reasoning held only while a timeout could not be scored. It could: the
candidate arm of the int8 run was offered 384 against a baseline measured at 32,
and the difference came back as a 15-point RULER regression.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "scripts"))

_SPEC = importlib.util.spec_from_file_location(
    "run_adapter_suite", ROOT / "eval" / "scripts" / "run_adapter_suite.py"
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class AdmissionRecordTests(unittest.TestCase):
    def test_a_shared_broker_is_recorded_as_the_arbiter(self):
        record = runner.admission_record(
            {"EVAL_ADMISSION_SOCKET": "/run/admission.sock",
             "EVAL_ADMISSION_PRIORS": "eval/token-priors.json"}
        )
        self.assertEqual(record["mode"], "broker")
        self.assertEqual(record["priors"], "eval/token-priors.json")

    def test_a_process_local_budget_records_its_capacity(self):
        record = runner.admission_record({"EVAL_ADMISSION_TOKENS": "2960000"})
        self.assertEqual(record["mode"], "local")
        self.assertEqual(record["capacity_tokens"], 2960000)

    def test_no_admission_control_is_recorded_as_such_rather_than_omitted(self):
        """An absent field reads as an old run that did not record it. This has
        to say the run was unthrottled, because that is the thing that makes it
        incomparable with one that was."""
        record = runner.admission_record({})
        self.assertEqual(record["mode"], "none")
