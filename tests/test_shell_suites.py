"""Run the shell test suites under pytest, so they are not run by hand or never.

Two of the trickiest pieces of this repository are shell: the lane fan-out in
paired-suite-eval.sbatch, and the campaign machinery that decides what reaches
the queue. Both have test suites written in shell, because both are shell, and
until now neither ran unless somebody remembered to invoke them. The lane suite
had been passing for weeks while nothing checked that it still did.
"""

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUITES = (
    ("tests/test_paired_suite_lanes.sh", "eval/slurm/paired-suite-eval.sbatch"),
    ("tests/test_campaign_lib.sh", "eval/slurm/campaign-lib.sh"),
    ("tests/test_campaign.sh", "eval/slurm/campaign.sh"),
)


class ShellSuiteTests(unittest.TestCase):
    def test_shell_suites_pass(self) -> None:
        for suite, subject in SUITES:
            with self.subTest(suite=suite):
                completed = subprocess.run(
                    ["bash", str(ROOT / suite), str(ROOT / subject)],
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                )
                # The suites print one line per check; on failure the useful
                # detail is in that output, not in the exit status.
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"{suite} failed:\n{completed.stdout}\n{completed.stderr}",
                )
                self.assertIn("failed 0", completed.stdout)


if __name__ == "__main__":
    unittest.main()
