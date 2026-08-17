import tempfile
import unittest
from pathlib import Path

from eval.scripts.prefetch_fp8_kernels import KERNELS, prefetch


class PrefetchFp8KernelsTests(unittest.TestCase):
    def test_downloads_each_pinned_kernel_without_importing_it(self) -> None:
        calls = []

        def resolve(repo_id: str, *, revision, version: int) -> str:
            self.assertIsNone(revision)
            calls.append(("resolve", repo_id, version))
            return f"{repo_id}-v{version}"

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)

            def install(repo_id: str, *, revision: str, validate_dependencies: bool):
                self.assertTrue(validate_dependencies)
                calls.append(("install", repo_id, revision))
                return path

            loaded = prefetch(resolve, install)

        self.assertEqual(
            calls,
            [
                ("resolve", "kernels-community/finegrained-fp8", 4),
                (
                    "install",
                    "kernels-community/finegrained-fp8",
                    "kernels-community/finegrained-fp8-v4",
                ),
            ],
        )
        self.assertEqual(len(loaded), len(KERNELS))

    def test_rejects_missing_cache_directory(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a directory"):
            prefetch(
                lambda *_args, **_kwargs: "revision",
                lambda *_args, **_kwargs: Path("does-not-exist"),
            )


if __name__ == "__main__":
    unittest.main()
