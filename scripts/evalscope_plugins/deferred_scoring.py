#!/usr/bin/env python3
"""Let a benchmark generate now and be scored somewhere else, later.

Two of our suites cannot be scored where they are generated. LiveCodeBench runs
code the model wrote, and the machine generating it holds the checkpoints and
every result so far. HLE calls an LLM judge, and EvalScope's default judge is a
hosted third-party endpoint, so a generating pass would ship every reply off the
cluster before anyone had chosen a judge.

EvalScope already has the two halves. `filter_prediction_cache` skips generation
for any item whose prediction is cached, and `--rerun-review` discards the
reviews and scores again. The only missing piece is a generating pass that does
not score, so this adds one:

    generate   extra_params={'<flag>': False}      predictions, nothing scored
    score      --use-cache <copy> --rerun-review
               extra_params={'<flag>': True}       scores, no model needed

Both passes must register under the benchmark's own name, because the prediction
cache path is `predictions/<model>/<benchmark>_<subset>.jsonl` and a second
registry entry would write where the scoring pass never looks. So the adapter is
swapped onto the existing metadata rather than registered alongside it.

The deferred score carries the benchmark's own metric name. A placeholder under
some other key would look like a missing metric to whatever reads it.
"""

from typing import Any

from evalscope.api.metric import Score
from evalscope.api.registry import BENCHMARK_REGISTRY

DEFERRED_METADATA = {"deferred": True, "execution_method": "deferred"}

# Which method actually scores, and the extra_param that turns it off. They
# differ per benchmark: LiveCodeBench executes in match_score, HLE calls its
# judge from llm_match_score.
TARGETS = {
    "live_code_bench": ("match_score", "execute"),
    "hle": ("llm_match_score", "judge"),
}


def metric_name(meta: Any) -> str:
    """The benchmark's own primary metric, so a deferred row parses like a real one."""
    for entry in meta.metric_list or []:
        return entry if isinstance(entry, str) else next(iter(entry))
    return "acc"


def install(benchmark: str) -> type:
    """Swap in a deferrable adapter for `benchmark`. Idempotent."""
    if benchmark not in TARGETS:
        raise RuntimeError(
            f"no deferral target known for {benchmark!r}; known: {sorted(TARGETS)}"
        )
    method, flag = TARGETS[benchmark]
    meta = BENCHMARK_REGISTRY.get(benchmark)
    if meta is None:  # pragma: no cover - evalscope registers all of these
        raise RuntimeError(f"{benchmark} is not registered; is evalscope installed?")

    base = meta.data_adapter
    if getattr(base, "_deferrable", False):
        return base
    if not hasattr(base, method):
        raise RuntimeError(f"{benchmark}'s adapter has no {method}(); evalscope changed")

    placeholder = metric_name(meta)
    original = getattr(base, method)

    def deferrable(self, *args: Any, **kwargs: Any) -> Score:
        # Default True so an unconfigured run behaves exactly as upstream does.
        # Deferring is the thing you have to ask for.
        if self.extra_params.get(flag, True):
            return original(self, *args, **kwargs)
        # A zero here is not a failed item, it is an unscored one. The marker is
        # what tells the scoring pass, and our comparator, that this row carries
        # no verdict yet.
        return Score(value={placeholder: 0.0}, metadata=dict(DEFERRED_METADATA))

    adapter = type(
        f"Deferrable{base.__name__}",
        (base,),
        {"_deferrable": True, method: deferrable},
    )
    meta.data_adapter = adapter
    BENCHMARK_REGISTRY[benchmark] = meta
    return adapter


def install_all() -> None:
    for name in TARGETS:
        if name in BENCHMARK_REGISTRY:
            install(name)


install_all()
