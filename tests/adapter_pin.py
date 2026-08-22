"""Print an adapter's pin, the way the adapters' own self_pin() computes it.

The reuse check in paired-suite-eval.sbatch recomputes this from the run command
to decide whether recorded rows were scored by the code now running. The shell
lane tests need the same number for their stub adapter, and a second hand-rolled
copy of the formula in a heredoc is how the two drift apart.
"""

import hashlib
import sys
from pathlib import Path


def pin(adapter: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted([adapter.resolve(), (adapter.parent / "_common.py").resolve()]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


if __name__ == "__main__":
    print(pin(Path(sys.argv[1])))
