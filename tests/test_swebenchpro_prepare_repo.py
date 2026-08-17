"""The repository prep that keeps a SWE-bench Pro rollout honest.

Built against a real repository rather than a mock, because every failure mode
here is a git reachability question and a fake would only test the fake. The
fixture reproduces the shape of the published images: a base commit, further
commits on the branch past it, a remote-tracking ref, and a tag -- the three
ways the upstream fix stays reachable after `git reset --hard`.
"""

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "eval" / "scripts" / "swebenchpro_prepare_repo.sh"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(repo)},
    )
    return result.stdout.strip()


class PrepareRepoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "app"
        self.repo.mkdir()
        git(self.repo, "init", "--quiet", "-b", "main")
        (self.repo / "file.txt").write_text("base\n")
        git(self.repo, "add", "file.txt")
        git(self.repo, "commit", "--quiet", "-m", "base state")
        self.base = git(self.repo, "rev-parse", "HEAD")

        # Everything after the base commit is the answer, in the three shapes
        # the published images actually carry it.
        (self.repo / "file.txt").write_text("fixed\n")
        git(self.repo, "commit", "--quiet", "-am", "fix the bug (#85280)")
        self.fix = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "tag", "v2.0")
        git(self.repo, "update-ref", "refs/remotes/origin/devel", self.fix)
        git(self.repo, "remote", "add", "origin", "https://example.invalid/repo.git")

    def prepare(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", str(SCRIPT), str(self.repo), self.base],
            capture_output=True, text=True,
        )

    def test_the_tree_is_moved_to_the_base_commit(self) -> None:
        """The published image is tagged at a different commit than the task."""
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.fix)
        self.assertEqual(self.prepare().returncode, 0)
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.base)
        self.assertEqual((self.repo / "file.txt").read_text(), "base\n")

    def test_the_fix_becomes_unreachable(self) -> None:
        self.assertEqual(self.prepare().returncode, 0)
        found = subprocess.run(
            ["git", "-C", str(self.repo), "cat-file", "-e", self.fix],
            capture_output=True,
        )
        self.assertNotEqual(found.returncode, 0, "the fix commit survived the strip")

    def test_git_log_all_no_longer_shows_the_fix(self) -> None:
        """The exploit is literally `git log --all`; this is that command."""
        self.assertIn("fix the bug", git(self.repo, "log", "--all", "--oneline"))
        self.assertEqual(self.prepare().returncode, 0)
        self.assertNotIn("fix the bug", git(self.repo, "log", "--all", "--oneline"))

    def test_remotes_and_tags_are_gone(self) -> None:
        self.assertEqual(self.prepare().returncode, 0)
        self.assertEqual(git(self.repo, "remote"), "")
        self.assertEqual(git(self.repo, "tag"), "")
        self.assertEqual(git(self.repo, "for-each-ref", "--format=%(refname)",
                             "refs/remotes"), "")

    def test_the_repository_still_works(self) -> None:
        """Deleting .git would be simpler and would break grading.

        The verifier applies a patch and diffs the result, so the repo has to
        survive the strip in working order.
        """
        self.assertEqual(self.prepare().returncode, 0)
        (self.repo / "file.txt").write_text("candidate edit\n")
        diff = git(self.repo, "diff")
        self.assertIn("candidate edit", diff)
        self.assertTrue((self.repo / ".git").is_dir())

    def test_it_reports_what_it_left(self) -> None:
        result = self.prepare()
        self.assertIn("no remotes, no tags", result.stdout)
        self.assertIn(self.base[:7], result.stdout)

    def test_an_unknown_base_commit_fails_rather_than_proceeding(self) -> None:
        """Silently preparing the wrong tree would be graded as a real result."""
        result = subprocess.run(
            ["sh", str(SCRIPT), str(self.repo), "0" * 40],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
