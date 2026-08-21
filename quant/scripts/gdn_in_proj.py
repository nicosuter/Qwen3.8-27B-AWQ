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

At four bits the ordering above reverses, because sixteen levels make the shape
of the grid matter more than its uniformity: e2m1 reconstructs these weights at
0.1098 against int4's 0.1219, a tenth better, and NVFP4 would do better still
since it blocks by 16 with an e4m3 scale rather than by 128. compressed-tensors
ships NVFP4A16, llm-compressor implements the tensor_group strategy it needs,
and the recipe change would be one line here.

It is servable on this hardware, which is worth stating because the class names
suggest otherwise. vLLM dispatches a weight-only NVFP4 config -- weights in
NVFP4 with input_quant None -- to CompressedTensorsW4A4Fp4(use_a16=True), and
that selects MarlinNvFp4LinearKernel, gated on device capability 75. Marlin
dequantizes to BF16 and runs an ordinary matmul, so no FP4 tensor core is
involved and the sm_86 serving cards and sm_90 evaluation cards both qualify.
The Blackwell requirement applies to the activation-quantized W4A4 path, not to
this one. What is missing off Blackwell is acceleration, not support.

Two things it is not. It is not a memory saving over int4: 4 + 8/16 bits a
weight against 4 + 16/128, because sixteen-element blocks carry eight times the
scale metadata. And it is not faster here, since both formats reach the same
Marlin dequantize-and-matmul path. What it buys is the tenth of reconstruction
error above, and a checkpoint that runs natively for anyone on Blackwell.

`int8` gives them eight bits with a group-128 scale, and is the default.

There is deliberately no FP8 mode beside it, and the reason is measurable on
these tensors rather than only arguable. Reconstruction error against the source
weights, every format given the same group-128 scale so the comparison is about
how the bits are spent inside a group:

    int8            (e0m7)   0.0067   1.00x
    fp8  e2m5                0.0066   0.98x
    fp8  e3m4                0.0127   1.88x
    fp8  e4m3  (standard)    0.0256   3.81x
    fp8  e5m2                0.0511   7.60x

Monotone in exponent bits: a group scale already carries the dynamic range, so
every bit spent on an exponent pays twice for it. e4m3 spends four of eight and
costs 3.81x the error. The limit of that trend -- no exponent bits, a uniform
grid -- is int8, and only e2m5 edges it, by 2%. No split is worth a format the
serving stack handles worse. The published literature agrees for the
block-scaled case: MXINT8 outperforms MXFP8 at block size 32, and FP8's real
advantage is in activations, whose range varies by orders of magnitude across
layers and which no mode here quantizes. vLLM routes int8 through
`CompressedTensorsWNA16`, minimum capability 75, so the serving cards run it
natively; that was checked in the container rather than assumed, since the FP8
path is exactly where assuming went wrong. Offering FP8 anyway would have kept a
mode whose only argument was that an earlier checkpoint used it, and rebuilding
that checkpoint is a job for the commit it came from.

`v2/model-fp8gdn` is that checkpoint: built before activations were settled,
with them quantized, and the one the scored Class A result belongs to. No mode
reproduces it and no mode writes to its directory.
"""

from __future__ import annotations

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
GDN_IN_PROJ_PRECISIONS = ("source", "int4", "nvfp4", "int8")
# Eight bits with a group scale, because these projections feed a recurrent
# state whose errors carry rather than being re-read each step. `source` remains
# the control the FP8 group is measured against and is still reachable by name.
DEFAULT_GDN_IN_PROJ_PRECISION = "int8"


# Every module that reads a given norm's output, per layer type. AWQ
# equalization divides the norm's weight by a per-channel scale and multiplies
# the same scale into the consumers, which is only function-preserving if
# *every* consumer gets it. Leaving a consumer in BF16 does not exempt it:
# folding a scale into a BF16 weight is a multiply, not a quantization, and
# skipping it changes the function before anything is quantized.
#
# This is why the GDN input projections are listed with in_proj_a and in_proj_b,
# which no mode ever quantizes. They share one normalized input with
# in_proj_qkv and in_proj_z, so an equalization that reached the first two and
# not the last two would silently rescale the decay and write-strength inputs.
#
# The two layer types are kept apart because input_layernorm feeds a different
# set in each, and a mapping written for one must not be judged against the
# other's consumers. The model is hybrid, so both are always present.
NORM_CONSUMERS = {
    "full_attention": {
        "input_layernorm": (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ),
        "post_attention_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
    },
    "linear_attention": {
        "input_layernorm": (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
        ),
        "post_attention_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
    },
}


def input_layernorm_pattern(indices: list[int]) -> str:
    """A suffix regex matching input_layernorm on exactly these layers.

    The two layer types need separate mappings because they consume the norm
    differently, and the only thing distinguishing them in a module path is the
    layer index, so the index set has to be spelled into the pattern.
    """
    if not indices:
        raise ValueError("no layers to smooth; the mapping would match nothing")
    return "re:.*layers\\.(" + "|".join(str(i) for i in indices) + ")\\.input_layernorm$"


def unbalanced_norm_mappings(
    mappings: list[tuple[str, list[str]]], layer_types: list[str]
) -> list[str]:
    """AWQ mappings that smooth a norm without folding into all its consumers.

    Takes (smooth_pattern, balance_patterns) pairs as plain strings so the check
    needs no AWQ import, plus the model's layer_types so each mapping is judged
    against the consumers the layers it actually matches have. Every layer index
    is tried rather than one representative per type: a mapping is written as a
    list of indices, so one that covers a type only partially would pass a
    representative check and fail on the layers it missed. Patterns are matched
    against the module path rather than searched for as substrings, because they
    are written as suffix regexes -- `re:.*gate_proj$` covers `mlp.gate_proj`
    without containing it.
    """

    def covers(pattern: str, index: int, consumer: str) -> bool:
        name = f"model.language_model.layers.{index}.{consumer}"
        if pattern.startswith("re:"):
            return re.fullmatch(pattern[3:], name) is not None
        return name.endswith(pattern)

    unknown = sorted(set(layer_types) - set(NORM_CONSUMERS))
    if unknown:
        raise ValueError(f"no consumer list for layer types: {', '.join(unknown)}")

    problems: list[str] = []
    seen = set()
    for smooth, balances in mappings:
        for index, kind in enumerate(layer_types):
            for norm, consumers in NORM_CONSUMERS[kind].items():
                if not covers(smooth, index, norm):
                    continue
                missing = tuple(
                    c for c in consumers
                    if not any(covers(pattern, index, c) for pattern in balances)
                )
                # One report per layer type, not per layer: 48 copies of the
                # same sentence says nothing 1 does not.
                if not missing or (smooth, kind, missing) in seen:
                    continue
                seen.add((smooth, kind, missing))
                problems.append(
                    f"{smooth} is smoothed on {kind} layers but"
                    f" {', '.join(missing)} would not receive the scale,"
                    " which changes the function"
                )
    return problems


def gdn_in_proj_plan(precision: str) -> dict[str, Any]:
    """Resolve a precision to what the recipe has to do differently.

    `four_bit` is appended to the four-bit group's targets, `ignore` to the
    ignore list, and `own_group` names a preset for a group of their own. The
    caller strips that preset's activations; no mode keeps them.
    """
    if precision not in GDN_IN_PROJ_PRECISIONS:
        raise ValueError(
            f"GDN_IN_PROJ_PRECISION must be one of {', '.join(GDN_IN_PROJ_PRECISIONS)}; got {precision!r}"
        )
    plan: dict[str, Any] = {"four_bit": (), "ignore": (), "own_group": None}
    if precision == "source":
        plan["ignore"] = GDN_IN_PROJ_TARGETS
    elif precision == "int4":
        # The same treatment the third-party checkpoints that quantize these
        # projections give them: four bits on the same grid as the rest of the
        # body, smoothed by the GDN input_layernorm mapping.
        plan["four_bit"] = GDN_IN_PROJ_TARGETS
    elif precision == "nvfp4":
        # Sixteen-element blocks with an e4m3 scale, which is why it beats a
        # uniform four-bit grid: at sixteen levels the shape of the grid matters
        # more than its evenness. Its own group rather than the four-bit one --
        # a different grid entirely, and a module cannot be in both.
        plan["own_group"] = "NVFP4A16"
    else:
        # W8A16 declares no activations at all. The caller strips them anyway,
        # so a preset added here later cannot reintroduce them by accident.
        plan["own_group"] = "W8A16"
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
    # Named for what varies, since the whole model is four-bit in every mode and
    # "-int4" alone would read as a claim about all of it. -fp8gdn stays
    # unreachable: it holds the build made before activations were settled.
    suffix = {
        "source": "",
        "int4": "-inproj-int4",
        "nvfp4": "-inproj-nvfp4",
        "int8": "-inproj-int8",
    }[precision]
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
