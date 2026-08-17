#!/usr/bin/env python3
"""What precision the Gated DeltaNet input projections are built at.

Separate from quantize.py because the decision is worth testing and quantize.py
cannot be imported without a GPU stack. Nothing here imports torch.

The projections in question are `linear_attn.in_proj_qkv` and `in_proj_z` -- the
query/key/value and gate inputs to the recurrent state update. `out_proj`, the
readout back into the residual stream, is not one of them: it is the analogue of
`self_attn.o_proj` and stays with the four-bit group in every mode.

Activations are BF16 in every mode and there is no flag for it. FP8 activation
quantization needs sm_89; the serving cards are sm_86, so vLLM's capability gate
drops it and serves the weights alone. Declaring it therefore bought nothing at
run time and cost two real things: the checkpoint described numerics no
deployment of ours performs, and the H200 evaluation measured that description
rather than the deployment. A mode nobody should pick is not a mode.

Three modes, and the differences between them are not academic:

`source` leaves them in BF16, which is what every third-party W4A16 release of
this model does with at least part of this path.

`int4` folds them into the four-bit group. This is philbert440's shape, which is
more aggressive than our FP8 build on these modules and measures less verbose
than it -- 1.04x median reasoning against our 1.13x -- so "quantizing the GDN
path causes the extra reasoning" is not what the evidence says.

`fp8-a16` quantizes the weights to FP8 and leaves activations in source
precision. This is what the serving hardware actually executes: on sm_86
`_is_fp8_w8a8` fails vLLM's capability gate and `CompressedTensorsW8A8Fp8` falls
back to `CompressedTensorsW8A16Fp8`, so the dynamic activation quantization is
dropped and only the weight-size saving remains. Building it explicitly makes
the checkpoint state what the deployment runs, instead of describing a
configuration that only exists on sm_89 and above.

`v2/model-fp8gdn` was built before this, with the activations quantized, and is
the checkpoint the scored Class A result belongs to. No mode reproduces it, on
purpose.
"""

import argparse
import re
import sys
from typing import Any

# Named explicitly rather than as "Linear" so the groups cannot overlap on a
# module, and so a projection that ends up in neither shows as a drop in the
# packed-tensor count rather than silently staying in source precision.
FOUR_BIT_TARGETS = (
    "re:.*mlp\\.(gate|up|down)_proj$",
    "re:.*self_attn\\.(q|k|v|o)_proj$",
    "re:.*linear_attn\\.out_proj$",
)
GDN_IN_PROJ_TARGETS = (
    "re:.*linear_attn\\.in_proj_qkv$",
    "re:.*linear_attn\\.in_proj_z$",
)
GDN_IN_PROJ_PRECISIONS = ("source", "int4", "fp8-a16")
# Retired rather than forgotten: the message a caller gets for it should say why
# it is gone and what replaces it, which "must be one of ..." does not.
RETIRED_PRECISIONS = {
    "fp8": "activations are BF16 in every mode now, and on sm_86 they always "
           "were; use fp8-a16 for the same served weights, honestly declared",
}
DEFAULT_GDN_IN_PROJ_PRECISION = "source"


# Every module that reads a given norm's output, in the layer types that have
# one. AWQ equalization divides the norm's weight by a per-channel scale and
# multiplies the same scale into the consumers, which is only function-preserving
# if *every* consumer gets it. Leaving a consumer in BF16 does not exempt it:
# folding a scale into a BF16 weight is a multiply, not a quantization, and
# skipping it changes the function before anything is quantized.
#
# This is why the GDN input projections are listed with in_proj_a and in_proj_b,
# which no mode ever quantizes. They share one normalized input with
# in_proj_qkv and in_proj_z, so an equalization that reached the first two and
# not the last two would silently rescale the decay and write-strength inputs.
NORM_CONSUMERS = {
    "input_layernorm": (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ),
    "post_attention_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
}


def unbalanced_norm_mappings(mappings: list[tuple[str, list[str]]]) -> list[str]:
    """AWQ mappings that smooth a norm without folding into all its consumers.

    Takes (smooth_pattern, balance_patterns) pairs as plain strings so the check
    needs no AWQ import. The patterns are evaluated against a canonical module
    path rather than searched for as substrings, because the mappings are
    written as suffix regexes -- `re:.*gate_proj$` covers `mlp.gate_proj`
    without containing it.
    """

    def covers(pattern: str, consumer: str) -> bool:
        name = f"model.language_model.layers.0.{consumer}"
        if pattern.startswith("re:"):
            return re.fullmatch(pattern[3:], name) is not None
        return name.endswith(pattern)

    problems = []
    for smooth, balances in mappings:
        for norm, consumers in NORM_CONSUMERS.items():
            if not covers(smooth, norm):
                continue
            missing = [
                c for c in consumers
                if not any(covers(pattern, c) for pattern in balances)
            ]
            if missing:
                problems.append(
                    f"{smooth} is smoothed but {', '.join(missing)} would not"
                    " receive the scale, which changes the function"
                )
    return problems


def gdn_in_proj_plan(precision: str) -> dict[str, Any]:
    """Resolve a precision to what the recipe has to do differently.

    `four_bit` is appended to the four-bit group's targets, `ignore` to the
    ignore list, and `own_group` names a preset for a group of their own. The
    caller strips that preset's activations; no mode keeps them.
    """
    if precision in RETIRED_PRECISIONS:
        raise ValueError(f"{precision!r} is no longer built: {RETIRED_PRECISIONS[precision]}")
    if precision not in GDN_IN_PROJ_PRECISIONS:
        raise ValueError(
            f"GDN_IN_PROJ_PRECISION must be one of {', '.join(GDN_IN_PROJ_PRECISIONS)}; got {precision!r}"
        )
    plan: dict[str, Any] = {"four_bit": (), "ignore": (), "own_group": None}
    if precision == "source":
        plan["ignore"] = GDN_IN_PROJ_TARGETS
    elif precision == "int4":
        # AWQ's mappings do not cover these projections, so folding them in here
        # quantizes them without the activation-aware rescaling the MLP and
        # attention paths get. That is also true of the third-party checkpoints
        # that quantize them to four bits, so it is the same shape, not a
        # shortcut -- but it is the reason this mode is not simply "int4 like
        # everything else".
        plan["four_bit"] = GDN_IN_PROJ_TARGETS
    else:
        # The preset carries dynamic FP8 activations; the recipe strips them,
        # which is the whole of "a16".
        plan["own_group"] = "FP8_BLOCK"
    return plan


def output_suffix(precision: str, algorithm: str = "awq") -> str:
    """The directory suffix a build writes to.

    Two builds that differ only in precision must not land in the same place:
    the algorithm and the GDN mode both leave config.json looking similar, and a
    checkpoint whose provenance is only in a job log is a checkpoint nobody can
    identify later.
    """
    if precision not in GDN_IN_PROJ_PRECISIONS:
        raise ValueError(f"unknown GDN in_proj precision {precision!r}")
    # -fp8gdn is deliberately not reachable: it holds the retired build.
    suffix = {"source": "", "int4": "-int4gdn", "fp8-a16": "-fp8gdn-a16"}[precision]
    if algorithm == "awq+gptq":
        suffix += "-gptq"
    return suffix


def main(argv: list[str] | None = None) -> int:
    """So the submitter asks this module rather than keeping its own copy.

    A shell script that repeats the mode list is a shell script that will accept
    a mode the job then rejects, an hour into a reservation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision", nargs="?", default=DEFAULT_GDN_IN_PROJ_PRECISION)
    parser.add_argument("--algorithm", default="awq")
    parser.add_argument("--suffix", action="store_true",
                        help="print the output directory suffix for this build")
    parser.add_argument("--modes", action="store_true", help="print the valid modes")
    args = parser.parse_args(argv)
    if args.modes:
        print(" ".join(GDN_IN_PROJ_PRECISIONS))
        return 0
    # An unset flag arrives as an empty argument rather than as no argument, so
    # the default lives here too. Letting the caller substitute it would put a
    # second copy of the default in a shell script.
    precision = args.precision or DEFAULT_GDN_IN_PROJ_PRECISION
    try:
        gdn_in_proj_plan(precision)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    print(output_suffix(precision, args.algorithm) if args.suffix else precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
