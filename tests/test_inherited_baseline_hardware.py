"""What device produced an inherited baseline, and what happens when nobody knows.

A candidate arm can inherit an already-scored baseline instead of buying the
same GPU hours again. That baseline carries the numerics of the device that
produced it, and below sm_89 the FP8 baseline's declared float8 activations are
served weight-only -- so a baseline measured there, paired against a candidate
served on an H200, is two different models under one name.

Nothing caught that. The taint report describes the device running the current
job, which says nothing about a baseline the job never loads, and it fired even
in candidate-only runs -- announcing a downgrade for weights that were not being
served. The check that replaces it reads the inherited run's own recorded
hardware and falls back to a hand-written attestation for runs predating the
field, because those record null, and null is not "whatever is serving now".
"""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTESTATIONS = ROOT / "eval" / "inherited-baseline-hardware.json"
SBATCH = ROOT / "eval" / "slurm" / "paired-suite-eval.sbatch"


def capability_of(name: str) -> str:
    """Run the sbatch's own resolver, so this cannot drift from what runs."""
    script = subprocess.run(
        ["awk", "/^capability_of\\(\\) \\{/,/^\\}/", str(SBATCH)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "case" in script, "capability_of no longer extracts; the definition moved"
    return subprocess.run(
        ["bash", "-c", f'{script}\ncapability_of "$1"', "_", name],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


class AttestationFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(ATTESTATIONS.read_text(encoding="utf-8"))

    def test_every_attested_run_resolves_to_a_capability(self) -> None:
        """An attestation naming a card nobody validated attests nothing."""
        for run, entry in self.data["runs"].items():
            with self.subTest(run=run):
                self.assertTrue(capability_of(entry["gpu"]), f"{entry['gpu']} is unknown")

    def test_every_attestation_says_where_it_came_from(self) -> None:
        """It is a claim about the past, so it has to carry its evidence."""
        for run, entry in self.data["runs"].items():
            with self.subTest(run=run):
                self.assertGreater(
                    len(entry.get("attested_by", "")), 40,
                    "attested_by must explain how the device is known",
                )

    def test_the_run_we_actually_inherit_from_is_covered(self) -> None:
        """The A100 baseline every A100 candidate is paired against.

        Its metadata records hardware null, so without this entry the check
        cannot tell it from a baseline of unknown origin -- and the whole point
        is that those two cases get different answers.
        """
        entry = self.data["runs"]["eval-suite-v1-cyan"]
        self.assertEqual(capability_of(entry["gpu"]), "8.0")
        self.assertLess(
            float(capability_of(entry["gpu"])), 8.9,
            "this baseline is served weight-only; the entry exists to say so",
        )


class SbatchGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SBATCH.read_text(encoding="utf-8")

    def test_the_taint_report_only_covers_served_arms(self) -> None:
        """Piping both arms in unconditionally is the bug this replaced."""
        self.assertNotIn(
            'printf \'%s\\n%s\\n\' "$BASELINE_INFO" "$CANDIDATE_INFO"', self.source,
            "the taint report is back to describing arms the job does not serve",
        )
        self.assertIn('if variant_requested baseline; then SERVED_INFO', self.source)
        self.assertIn('if variant_requested candidate; then SERVED_INFO', self.source)

    def test_an_unknown_or_mismatched_baseline_device_stops_the_run(self) -> None:
        """Both directions fail closed; the escape hatch has to be asked for.

        The silent direction is the one worth the test: a baseline measured on
        an A100 inherited into an H200 job produces no taint line at all, since
        the taint only ever describes the current device.
        """
        guard = self.source[self.source.index("INHERIT_GPU_INFO="):]
        guard = guard[: guard.index("\nfi\n")]
        self.assertIn("does not record what device produced it", guard)
        self.assertIn("was measured at capability", guard)
        self.assertEqual(
            guard.count("exit 1"), 2,
            "expected a refusal for an unknown device and one for a mismatch",
        )
        self.assertIn(
            'if [[ "${PAIRED_ALLOW_CROSS_HARDWARE_BASELINE:-0}" != "1" ]]', guard,
            "pairing across hardware has to be asked for, not defaulted into",
        )


if __name__ == "__main__":
    unittest.main()
