"""Python embedded in shell scripts still has to be Python.

A batch job carries several small programs inline -- the reuse check, the
comparison rebuild, the taint report. They are not imported by anything, so
nothing type-checks or byte-compiles them, and a syntax error in one surfaces
only when the job runs. That is the worst place for it: the job has already been
queued, scheduled and allocated GPUs, and it fails on a line that never touches
the measurement.

It happened. A taint reporter written with escaped quotes inside an f-string
parsed as a shell string, then failed to parse as Python, and because it ran
unconditionally rather than only when there was a taint, it killed the job six
seconds in -- on every eval submitted until it was found.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Both forms in use share the marker: a `python - <<'PY'` heredoc, and the
# `python -c "$(cat <<'PY' ... PY)"` spelling used where stdin is a pipe.
BLOCK = re.compile(r"<<'PY'\n(.*?)\nPY\n", re.S)


def shell_files():
    return sorted(
        path
        for pattern in ("*.sh", "*.sbatch")
        for path in ROOT.rglob(pattern)
        if ".git" not in path.parts
    )


class InlinePythonTests(unittest.TestCase):
    def test_every_embedded_program_compiles(self) -> None:
        found = 0
        for path in shell_files():
            source = path.read_text(encoding="utf-8", errors="replace")
            for index, block in enumerate(BLOCK.findall(source)):
                found += 1
                name = f"{path.relative_to(ROOT)}#PY{index}"
                with self.subTest(block=name):
                    try:
                        compile(block, name, "exec")
                    except SyntaxError as error:
                        self.fail(f"{name} line {error.lineno}: {error.msg}")
        self.assertGreater(found, 3, "extraction found nothing; the marker changed")

    def test_the_heredoc_marker_is_quoted(self) -> None:
        """An unquoted <<PY lets the shell expand $... inside the program.

        The block would then be a different program than the one in the file,
        and usually a broken one, since these use $ for nothing.
        """
        for path in shell_files():
            source = path.read_text(encoding="utf-8", errors="replace")
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotRegex(source, r"<<PY\n", "unquoted heredoc marker")


if __name__ == "__main__":
    unittest.main()
