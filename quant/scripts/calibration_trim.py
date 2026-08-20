"""Trim a truncated calibration row back to the last think block that closes.

Kept free of torch and transformers on purpose. quantize.py cannot be imported
without a CUDA-capable environment, so anything living there is only testable on
a GPU node -- which in practice means untested. This is the whole of the logic
and it is exact, because `<think>` and `</think>` are single tokens.

Why it exists: AWQ fits per-channel scales to the activations it observes, and
truncating at MAX_SEQ_LENGTH cuts long reasoning traces mid-block. On the
shipped calibration set 73.9% of the in-think token mass inside the window
belongs to a block that never terminates -- 2.8 tokens of reasoning that does
not conclude for every token that does. That asymmetry inflates the opening tag
and the interior while contributing nothing to the close, which is the shape
that moves a stop probability.

An empty `<think></think>` is a different thing and is left alone: both tags are
present and adjacent, so they are represented in proportion.
"""

from typing import Iterable


def trim_to_closed_think(ids: Iterable, *, open_id: int, close_id: int) -> list[int]:
    """Return `ids` cut back to just before an unterminated `<think>`.

    A prefix of the input, always -- never a rewrite. Rows whose only content is
    a dangling block trim away to nothing, and the caller decides whether a row
    that short is still worth keeping.
    """
    out = [int(token) for token in ids]
    depth = 0
    last_open = -1
    for index, token in enumerate(out):
        if token == open_id:
            if depth == 0:
                last_open = index
            depth += 1
        elif token == close_id and depth:
            depth -= 1
    if depth:
        return out[:last_open]
    return out
