"""The checkpoint registry resolves to the paths the campaign actually binds.

A wrong revision here is not a crash: it serves a different snapshot under the
same served-model-name and reports the delta as though it measured the one the
run claims. So the layout it builds, and every entry it can build, are checked
against the registry rather than against a copy of it.
"""

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "checkpoints", ROOT / "eval" / "scripts" / "checkpoints.py"
)
assert spec and spec.loader
checkpoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoints)

RUN_BASE = Path("/scratch/test")
REGISTRY = json.loads((ROOT / "eval" / "checkpoints.json").read_text(encoding="utf-8"))


class RegistryTests(unittest.TestCase):
    def test_every_candidate_resolves(self) -> None:
        for name in checkpoints.names():
            with self.subTest(candidate=name):
                resolved = checkpoints.candidate(name, RUN_BASE)
                self.assertTrue(resolved["path"].startswith(str(RUN_BASE)))
                self.assertTrue(resolved["run_dir"].startswith(f"{RUN_BASE}/v2/"))

    def test_a_published_quantization_is_addressed_through_its_snapshot(self) -> None:
        resolved = checkpoints.candidate("cyankiwi", RUN_BASE)
        self.assertEqual(
            resolved["path"],
            f"{RUN_BASE}/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4"
            "/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea",
        )

    def test_one_of_ours_is_a_directory_under_run_base(self) -> None:
        self.assertEqual(
            checkpoints.candidate("bf16gdn", RUN_BASE)["path"], f"{RUN_BASE}/v2/model"
        )

    def test_the_baseline_is_bound_through_its_repository_root(self) -> None:
        """Not the snapshot: it is a farm of symlinks into ../../blobs."""
        resolved = checkpoints.baseline(RUN_BASE)
        self.assertEqual(
            resolved["repo"],
            f"{RUN_BASE}/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8",
        )
        self.assertEqual(resolved["revision"], REGISTRY["baseline"]["revision"])

    def test_hf_home_wins_over_the_default_cache_location(self) -> None:
        import os
        import unittest.mock

        with unittest.mock.patch.dict(os.environ, {"HF_HOME": "/elsewhere/hf"}):
            resolved = checkpoints.baseline(RUN_BASE)
        self.assertEqual(
            resolved["repo"], "/elsewhere/hf/hub/models--Qwen--Qwen3.8-27B-FP8"
        )

    def test_a_run_directory_is_derived_unless_the_entry_names_one(self) -> None:
        # philbert has no recorded run directory, so it gets the derived name;
        # cyankiwi's results already sit somewhere else and it records that.
        self.assertEqual(
            checkpoints.candidate("philbert", RUN_BASE, "v1")["run_dir"],
            f"{RUN_BASE}/v2/eval-suite-v1-philbert",
        )
        self.assertEqual(
            checkpoints.candidate("cyankiwi", RUN_BASE, "v1")["run_dir"],
            f"{RUN_BASE}/v2/eval-suite-v1-cyan",
        )

    def test_no_two_candidates_share_a_run_directory(self) -> None:
        """One run directory holds exactly one raw/candidate tree."""
        run_dirs = [
            checkpoints.candidate(name, RUN_BASE)["run_dir"]
            for name in checkpoints.names()
        ]
        self.assertEqual(len(set(run_dirs)), len(run_dirs), run_dirs)

    def test_an_unknown_name_says_what_is_known(self) -> None:
        with self.assertRaises(checkpoints.CheckpointError) as caught:
            checkpoints.candidate("nosuchthing", RUN_BASE)
        self.assertIn("cyankiwi", str(caught.exception))

    def test_an_entry_that_describes_nothing_is_refused(self) -> None:
        broken = ROOT / "tests" / "_broken_registry"
        (broken / "eval").mkdir(parents=True, exist_ok=True)
        try:
            (broken / "eval" / "checkpoints.json").write_text(
                json.dumps(
                    {
                        "baseline": REGISTRY["baseline"],
                        "candidates": {"vague": {"why": "no path, no repo"}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(checkpoints.CheckpointError):
                checkpoints.load(broken)
        finally:
            (broken / "eval" / "checkpoints.json").unlink(missing_ok=True)
            (broken / "eval").rmdir()
            broken.rmdir()


if __name__ == "__main__":
    unittest.main()
