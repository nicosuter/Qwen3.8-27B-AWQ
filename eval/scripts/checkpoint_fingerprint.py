#!/usr/bin/env python3
"""Identify the checkpoint a run actually served.

Both variants are served under the same `--served-model-name`, deliberately, so
that no harness behavior can branch on which model is loaded. The consequence is
that nothing in a result file says which weights produced it, and several
checkpoints sit side by side under the same tree. This prints a descriptor to
record alongside the results so the answer is in the run, not in a job log that
gets overwritten.

The fingerprint hashes the safetensors index and config rather than the weights:
the index carries every shard name, its size and the full weight map, so two
different builds cannot collide, and it costs milliseconds instead of reading
tens of gigabytes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

FINGERPRINT_FILES = (
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    # Ours records the calibration seed and recipe; third-party checkpoints have
    # no equivalent, which is why it is optional rather than required.
    "run-metadata.json",
)

# Metadata alone identifies the build, not the bytes: requantizing with a
# different calibration seed yields the same tensor names, the same shard sizes
# and the same config, so the metadata hash would collide with the original --
# exactly the comparison a seed sensitivity study needs to tell apart. Sampling
# a little of each shard separates them. Every tensor differs between two
# calibration runs, so a slice from each shard is enough, and reading a few
# megabytes costs milliseconds where hashing 21GB costs half a minute.
SAMPLE_BYTES = 262144
SAMPLE_POINTS = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="checkpoint directory")
    parser.add_argument("--label", default="", help="variant name, recorded verbatim")
    # Optional, and absent means "unknown" rather than "fine": a descriptor
    # written without it records a null taint, which downstream can tell apart
    # from an empty one.
    parser.add_argument(
        "--compute-capability", type=float, default=None,
        help="the serving device's compute capability, e.g. 9.0; decides whether "
             "declared activation quantization will actually be performed",
    )
    return parser.parse_args(argv)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    used = []
    for name in FINGERPRINT_FILES:
        candidate = path / name
        if not candidate.is_file():
            continue
        digest.update(name.encode())
        digest.update(candidate.read_bytes())
        used.append(name)
    for shard in sorted(path.glob("*.safetensors")):
        size = shard.stat().st_size
        digest.update(shard.name.encode())
        digest.update(str(size).encode())
        with shard.open("rb") as handle:
            for point in range(SAMPLE_POINTS):
                offset = (size // SAMPLE_POINTS) * point
                handle.seek(offset)
                digest.update(handle.read(SAMPLE_BYTES))
        used.append(shard.name)
    if not used:
        raise SystemExit(f"{path}: nothing to fingerprint")
    return "sha256:" + digest.hexdigest()


# FP8 activation quantization needs sm_89. Below it vLLM's `_is_fp8_w8a8`
# capability gate fails and `CompressedTensorsW8A8Fp8` falls back to
# `CompressedTensorsW8A16Fp8`, which keeps the weight-size saving and drops the
# dynamic activation quantization entirely. Verified in vLLM 0.27.1.
#
# The fallback is silent, and it is silent in the direction that matters: the
# checkpoint declares activation quantization, the server does not perform it,
# and every result the run produces describes numerics the checkpoint does not
# have. A100 is sm_80 and the serving GPUs are sm_86, so the protocol's own A100
# lane has this property while the H200 lane does not -- which means two lanes
# scoring the same checkpoint are not measuring the same thing.
#
# Only the float8 case is listed because only it has been verified here. An int8
# activation scheme has its own gate and would need its own reading of the
# source before being asserted.
FLOAT8_ACTIVATION_MIN_CAPABILITY = 8.9


def declared_float8_activations(quantization: dict) -> list[str]:
    """Which parts of a checkpoint say their activations are FP8.

    Two schemas say it two ways, and reading only one of them is how the
    baseline came back clean while being the arm this matters most for.
    compressed-tensors puts `input_activations` on each config group;
    a native FP8 release has no groups at all and says `activation_scheme:
    dynamic` once, for the whole model.
    """
    named = [
        name
        for name, group in sorted((quantization.get("config_groups") or {}).items())
        if ((group.get("input_activations") or {}).get("type")) == "float"
    ]
    if named:
        return named
    if (
        quantization.get("quant_method") == "fp8"
        and quantization.get("activation_scheme") == "dynamic"
    ):
        return ["*"]
    return []


def activation_taint(quantization: dict, capability: float | None) -> list[dict] | None:
    """Declared activation quantization this device will not actually perform.

    Returns [] when everything declared will run, and None when the capability
    is unknown -- absent evidence is not evidence of a clean run, and the two
    have to stay distinguishable downstream.
    """
    if capability is None:
        return None
    if capability >= FLOAT8_ACTIVATION_MIN_CAPABILITY:
        return []
    return [
        {
            "group": name,
            "declared": "float8 activations",
            "served_as": "source precision, weight-only",
            "requires_capability": FLOAT8_ACTIVATION_MIN_CAPABILITY,
            "capability": capability,
        }
        for name in declared_float8_activations(quantization)
    ]


def describe(path: Path, label: str, capability: float | None = None) -> dict:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config") or {}
    groups = {}
    for name, group in (quantization.get("config_groups") or {}).items():
        weights = group.get("weights") or {}
        activations = group.get("input_activations") or {}
        groups[name] = {
            "type": weights.get("type"),
            "num_bits": weights.get("num_bits"),
            "symmetric": weights.get("symmetric"),
            "group_size": weights.get("group_size"),
            "strategy": weights.get("strategy"),
            "activations": activations.get("type"),
        }
    shards = sorted(path.glob("*.safetensors"))
    return {
        "label": label,
        "path": str(path),
        "fingerprint": fingerprint(path),
        "architectures": config.get("architectures"),
        "quantization_format": quantization.get("format"),
        "quantization_method": quantization.get("quant_method"),
        "config_groups": groups,
        # Rides on the descriptor rather than being computed separately, because
        # the descriptor is already exported as EVAL_CHECKPOINT_JSON and lands
        # in every suite's metadata. A taint recorded anywhere else would have
        # to be joined back to the arm it belongs to.
        "activation_taint": activation_taint(quantization, capability),
        "ignored_modules": len(quantization.get("ignore") or []),
        "shards": len(shards),
        "bytes_on_disk": sum(shard.stat().st_size for shard in shards),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.path / "config.json").is_file():
        print(f"{args.path}: no config.json", file=sys.stderr)
        return 1
    print(json.dumps(describe(args.path, args.label, args.compute_capability)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
