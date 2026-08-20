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


# A row rendered with the chat template ends on the generation prompt, so it can
# close with a bare `<think>\n` carrying no reasoning at all. That is an artefact
# of rendering, not a trace cut mid-thought, and the trim removes it in two
# tokens. Anything longer than this is real reasoning that got cut.
TRIVIAL_DANGLE = 8


def closes_in_window(
    ids: Iterable, *, open_id: int, close_id: int, max_dangling: int = TRIVIAL_DANGLE
) -> bool:
    """True when no substantial reasoning is left unterminated in `ids`.

    Used while building the calibration set, so a sample whose reasoning does
    not fit is passed over and another is drawn. Without it the trim is the only
    defence, and trimming a row whose whole content is one oversized block
    leaves a prompt and spends the sample slot on nothing -- while selecting
    against the longest chains, which is the opposite of what a wide window is
    for.
    """
    # Materialised once: trim_to_closed_think consumes the argument, so counting
    # the original afterwards would read zero for a generator and wave through
    # every cut sample.
    tokens = [int(token) for token in ids]
    kept = trim_to_closed_think(tokens, open_id=open_id, close_id=close_id)
    return len(tokens) - len(kept) <= max_dangling


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
