#!/usr/bin/env python3
"""What precision the Gated DeltaNet input projections are built at.

Separate from quantize.py because the decision is worth testing and quantize.py
cannot be imported without a GPU stack. Nothing here imports torch.

The projections in question are `linear_attn.in_proj_qkv` and `in_proj_z` -- the
query/key/value and gate inputs to the recurrent state update. `out_proj`, the
readout back into the residual stream, is not one of them: it is the analogue of
`self_attn.o_proj` and stays with the four-bit group in every mode.

Four modes, and the differences between them are not academic:

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

`fp8` is that with the activations quantized too, which is what Qwen's own FP8
release applies here and what our first build used. It is the only mode in the
set that quantizes an activation anywhere in the model, and it is measurably the
most verbose. On sm_86 it is also silently the same as `fp8-a16` at run time,
which is the strongest argument against choosing it by accident.
"""

import argparse
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
GDN_TARGETS = (
    "re:.*linear_attn\\.in_proj_qkv$",
    "re:.*linear_attn\\.in_proj_z$",
)
GDN_PRECISIONS = ("source", "int4", "fp8-a16", "fp8")
DEFAULT_GDN_PRECISION = "source"


def gdn_plan(precision: str) -> dict[str, Any]:
    """Resolve a precision to what the recipe has to do differently.

    `four_bit` is appended to the four-bit group's targets, `ignore` to the
    ignore list, and `own_group` names a preset for a group of their own.
    `quantize_activations` only means anything when there is one.
    """
    if precision not in GDN_PRECISIONS:
        raise ValueError(
            f"GDN_PRECISION must be one of {', '.join(GDN_PRECISIONS)}; got {precision!r}"
        )
    plan: dict[str, Any] = {
        "four_bit": (),
        "ignore": (),
        "own_group": None,
        "quantize_activations": False,
    }
    if precision == "source":
        plan["ignore"] = GDN_TARGETS
    elif precision == "int4":
        # AWQ's mappings do not cover these projections, so folding them in here
        # quantizes them without the activation-aware rescaling the MLP and
        # attention paths get. That is also true of the third-party checkpoints
        # that quantize them to four bits, so it is the same shape, not a
        # shortcut -- but it is the reason this mode is not simply "int4 like
        # everything else".
        plan["four_bit"] = GDN_TARGETS
    else:
        plan["own_group"] = "FP8_BLOCK"
        plan["quantize_activations"] = precision == "fp8"
    return plan


def output_suffix(precision: str, algorithm: str = "awq") -> str:
    """The directory suffix a build writes to.

    Two builds that differ only in precision must not land in the same place:
    the algorithm and the GDN mode both leave config.json looking similar, and a
    checkpoint whose provenance is only in a job log is a checkpoint nobody can
    identify later.
    """
    if precision not in GDN_PRECISIONS:
        raise ValueError(f"unknown GDN precision {precision!r}")
    suffix = {
        "source": "",
        "int4": "-int4gdn",
        "fp8-a16": "-fp8gdn-a16",
        # Unchanged: v2/model-fp8gdn already exists and has been scored.
        "fp8": "-fp8gdn",
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
    parser.add_argument("precision", nargs="?", default=DEFAULT_GDN_PRECISION)
    parser.add_argument("--algorithm", default="awq")
    parser.add_argument("--suffix", action="store_true",
                        help="print the output directory suffix for this build")
    parser.add_argument("--modes", action="store_true", help="print the valid modes")
    args = parser.parse_args(argv)
    if args.modes:
        print(" ".join(GDN_PRECISIONS))
        return 0
    # An unset flag arrives as an empty argument rather than as no argument, so
    # the default lives here too. Letting the caller substitute it would put a
    # second copy of the default in a shell script.
    precision = args.precision or DEFAULT_GDN_PRECISION
    try:
        gdn_plan(precision)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    print(output_suffix(precision, args.algorithm) if args.suffix else precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
