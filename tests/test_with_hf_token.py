"""The token wrapper is a control, so it is tested like one.

Its whole job is that a credential never reaches disk or a Slurm job record.
That is not something a comment can guarantee, so each property is asserted:
the token reaches the child, it does not leak into argv, and an attempt to hand
it to the batch system is refused rather than warned about.
"""

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "slurm" / "with-hf-token.sh"


def run(args, env=None, **kwargs):
    environment = dict(os.environ)
    environment.pop("HF_TOKEN", None)
    environment.pop("HUGGING_FACE_HUB_TOKEN", None)
    environment.update(env or {})
    return subprocess.run(
        [str(WRAPPER), *args],
        capture_output=True, text=True, env=environment, cwd=ROOT, **kwargs
    )


class RefusalTests(unittest.TestCase):
    """Slurm persists a job's environment, so the token must never get there."""

    SUBMITTERS = ("sbatch", "srun", "salloc", "sattach", "sbcast")

    def test_every_slurm_submitter_is_refused(self) -> None:
        for command in self.SUBMITTERS:
            with self.subTest(command=command):
                done = run(["--", command, "job.sbatch"], env={"HF_TOKEN": "secret"})
                self.assertEqual(done.returncode, 3)
                self.assertIn("refusing", done.stderr)

    def test_a_full_path_does_not_evade_the_refusal(self) -> None:
        done = run(["--", "/usr/bin/sbatch", "job.sbatch"], env={"HF_TOKEN": "secret"})
        self.assertEqual(done.returncode, 3)

    def test_the_refusal_says_what_to_do_instead(self) -> None:
        done = run(["--", "sbatch", "x"], env={"HF_TOKEN": "secret"})
        self.assertIn("materialize", done.stderr)

    def test_the_token_is_not_echoed_in_the_refusal(self) -> None:
        done = run(["--", "sbatch", "x"], env={"HF_TOKEN": "hunter2"})
        self.assertNotIn("hunter2", done.stderr + done.stdout)


class PassThroughTests(unittest.TestCase):
    def test_the_child_receives_the_token(self) -> None:
        done = run(["--", "bash", "-c", 'echo "$HF_TOKEN"'], env={"HF_TOKEN": "abc123"})
        self.assertEqual(done.stdout.strip(), "abc123")

    def test_both_variable_names_are_set(self) -> None:
        done = run(
            ["--", "bash", "-c", 'echo "$HUGGING_FACE_HUB_TOKEN"'],
            env={"HF_TOKEN": "abc123"},
        )
        self.assertEqual(done.stdout.strip(), "abc123")

    def test_implicit_persistence_is_disabled(self) -> None:
        # Otherwise huggingface_hub writes it to ~/.cache/huggingface/token.
        done = run(
            ["--", "bash", "-c", 'echo "$HF_HUB_DISABLE_IMPLICIT_TOKEN_PERSISTENCE"'],
            env={"HF_TOKEN": "abc123"},
        )
        self.assertEqual(done.stdout.strip(), "1")

    def test_the_token_is_never_an_argument(self) -> None:
        done = run(
            ["--", "bash", "-c", 'printf "%s\\n" "$@"', "_", "one", "two"],
            env={"HF_TOKEN": "abc123"},
        )
        self.assertEqual(done.stdout.split(), ["one", "two"])

    def test_the_exit_status_is_the_child_s(self) -> None:
        done = run(["--", "bash", "-c", "exit 7"], env={"HF_TOKEN": "abc123"})
        self.assertEqual(done.returncode, 7)

    def test_no_command_prints_usage(self) -> None:
        done = run([], env={"HF_TOKEN": "abc123"})
        self.assertEqual(done.returncode, 2)
        self.assertIn("with-hf-token.sh", done.stderr)

    def test_a_missing_token_never_hangs(self) -> None:
        """With no token and no terminal it must fail, not block on /dev/tty.

        A prompt that blocks would stall a non-interactive caller forever, so
        the timeout here is the assertion: exceeding it is the failure.
        """
        try:
            done = run(["--", "true"], stdin=subprocess.DEVNULL, timeout=20)
        except subprocess.TimeoutExpired:
            self.fail("blocked waiting for input with no terminal")
        # Either it refused for want of a token, or a terminal was available and
        # the caller supplied one. Both are fine; hanging is not.
        self.assertIn(done.returncode, (0, 1))


if __name__ == "__main__":
    unittest.main()
