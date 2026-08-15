import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class RuntimeRequirementsTests(unittest.TestCase):
    def test_finegrained_fp8_kernel_version_is_pinned(self) -> None:
        requirements = {
            line.strip()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("kernels==0.16.0", requirements)


if __name__ == "__main__":
    unittest.main()
