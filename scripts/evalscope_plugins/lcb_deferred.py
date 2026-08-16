#!/usr/bin/env python3
"""Make EvalScope's LiveCodeBench stop at generation so scoring can run elsewhere.

Scoring this suite runs code the model wrote. EvalScope executes it in the local
environment by default -- its own benchmark description recommends a sandbox --
and it has no generate-only mode, so a run on the GPU cluster would execute
model-written code on the machine holding the checkpoints, the frozen item sets
and every result so far.

The split already exists in EvalScope, it is just not reachable for this suite:
`filter_prediction_cache` skips generation for any item whose prediction is
cached, and `--rerun-review` deletes the review cache and scores again. So the
only missing piece is a generating pass that does not execute. This adds it as
an `execute` extra param:

    cluster   extra_params={'execute': False}          predictions, no execution
    isolated  --use-cache <dir> --rerun-review
              extra_params={'execute': True}           executes, scores

Both passes must register under the name `live_code_bench`, because the
prediction cache path is `predictions/<model>/<benchmark>_<subset>.jsonl` and a
second registry entry would write somewhere the scoring pass never looks. So the
adapter is swapped on the existing metadata rather than registered alongside it.

Import this module before calling run_task.
"""

from typing import Any

from evalscope.api.metric import Score
from evalscope.api.registry import BENCHMARK_REGISTRY

DEFERRED = {"deferred": True, "execution_method": "deferred"}


def install() -> type:
    """Swap in the deferrable adapter. Idempotent."""
    meta = BENCHMARK_REGISTRY.get("live_code_bench")
    if meta is None:  # pragma: no cover - evalscope always registers it
        raise RuntimeError("live_code_bench is not registered; is evalscope installed?")
    base = meta.data_adapter
    if getattr(base, "_deferrable", False):
        return base

    class DeferrableLiveCodeBench(base):  # type: ignore[misc, valid-type]
        _deferrable = True

        def match_score(self, *args: Any, **kwargs: Any) -> Score:
            # Default True so an unconfigured run behaves exactly as upstream
            # does. Deferring is the thing you have to ask for.
            if self.extra_params.get("execute", True):
                return super().match_score(*args, **kwargs)
            # A zero here is not a failed item, it is an unscored one. The
            # marker is what tells the scoring pass, and our comparator, that
            # this row carries no verdict yet.
            return Score(value={"pass": 0.0}, metadata=dict(DEFERRED))

    meta.data_adapter = DeferrableLiveCodeBench
    BENCHMARK_REGISTRY["live_code_bench"] = meta
    return DeferrableLiveCodeBench


install()
