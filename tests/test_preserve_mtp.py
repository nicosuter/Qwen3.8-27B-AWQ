import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.preserve_mtp import (
    MTP_SHARD,
    QWEN38_MTP_KEYS,
    QWEN38_MTP_LINEAR_MODULES,
    QWEN38_MTP_SHAPES,
    merged_ignore,
    merged_index,
    mtp_linear_modules,
    mtp_weight_map,
    packed_weight_count,
    read_safetensors_header,
    repair_mtp_ignores,
    require_packed_export,
    preserve_mtp_weights,
    validate_mtp_artifact,
)


class PreserveMtpTests(unittest.TestCase):
    def test_extracts_only_top_level_mtp_keys(self) -> None:
        index = {
            "weight_map": {
                "model.layers.0.mtp_like.weight": "main.safetensors",
                "mtp.fc.weight": "source.safetensors",
                "mtp.norm.weight": "source.safetensors",
            }
        }
        self.assertEqual(
            mtp_weight_map(index),
            {
                "mtp.fc.weight": "source.safetensors",
                "mtp.norm.weight": "source.safetensors",
            },
        )

    def test_source_without_mtp_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, r"no top-level mtp\.\*"):
            mtp_weight_map({"weight_map": {"model.weight": "main.safetensors"}})

    def test_merge_is_non_mutating_and_updates_metadata(self) -> None:
        original = {
            "metadata": {"total_size": 100, "total_parameters": 20},
            "weight_map": {"model.weight_packed": "main.safetensors"},
        }
        merged = merged_index(
            original,
            {"mtp.fc.weight", "mtp.norm.weight"},
            mtp_numel=7,
            mtp_nbytes=14,
        )
        self.assertEqual(original["metadata"]["total_size"], 100)
        self.assertEqual(merged["metadata"]["total_size"], 114)
        self.assertEqual(merged["metadata"]["total_parameters"], 27)
        self.assertEqual(merged["weight_map"]["mtp.fc.weight"], MTP_SHARD)

    def test_merge_rejects_existing_mtp_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "already contains MTP"):
            merged_index(
                {"weight_map": {"mtp.fc.weight": "old.safetensors"}},
                {"mtp.fc.weight"},
                mtp_numel=1,
                mtp_nbytes=2,
            )

    def test_packed_export_gate(self) -> None:
        index = {
            "weight_map": {
                "a.weight_packed": "one.safetensors",
                "b.weight_packed": "two.safetensors",
                "b.weight_scale": "two.safetensors",
            }
        }
        self.assertEqual(packed_weight_count(index), 2)
        self.assertEqual(require_packed_export(index), 2)
        with self.assertRaisesRegex(RuntimeError, "compressed export failed"):
            require_packed_export({"weight_map": {"a.weight": "one.safetensors"}})

    def test_end_to_end_preservation_updates_shard_and_index_atomically(self) -> None:
        bf16 = object()

        class FakeTensor:
            dtype = bf16

            def __init__(self, values: tuple[int, ...], ndim: int = 1) -> None:
                self.values = values
                self.ndim = ndim
                self.shape = (len(values),) if ndim == 1 else (1, len(values))

            def clone(self):
                return FakeTensor(self.values, self.ndim)

            def numel(self) -> int:
                return len(self.values)

            def element_size(self) -> int:
                return 2

        tensor_files: dict[str, dict[str, FakeTensor]] = {}

        class FakeSafeOpen:
            def __init__(self, path, **_kwargs) -> None:
                self.tensors = tensor_files[str(path)]

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def keys(self):
                return self.tensors.keys()

            def get_tensor(self, name: str):
                return self.tensors[name]

        def fake_save_file(tensors, path, metadata=None) -> None:
            self.assertEqual(metadata["format"], "pt")
            header = {}
            offset = 0
            for name, tensor in tensors.items():
                size = tensor.numel() * tensor.element_size()
                header[name] = {
                    "dtype": "BF16",
                    "shape": list(tensor.shape),
                    "data_offsets": [offset, offset + size],
                }
                offset += size
            encoded = json.dumps(header).encode("utf-8")
            encoded += b" " * ((8 - len(encoded) % 8) % 8)
            Path(path).write_bytes(
                len(encoded).to_bytes(8, "little") + encoded + bytes(offset)
            )
            tensor_files[str(path)] = dict(tensors)

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.bfloat16 = bf16
        fake_torch.equal = lambda left, right: left.values == right.values
        fake_safetensors = types.ModuleType("safetensors")
        fake_safetensors.safe_open = FakeSafeOpen
        fake_safetensors_torch = types.ModuleType("safetensors.torch")
        fake_safetensors_torch.save_file = fake_save_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            source_index = root / "source-index.json"
            source_shard = root / "source.safetensors"
            source_index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "mtp.fc.weight": source_shard.name,
                            "mtp.norm.weight": source_shard.name,
                        }
                    }
                )
            )
            source_shard.write_bytes(b"source")
            tensor_files[str(source_shard)] = {
                "mtp.fc.weight": FakeTensor((1, 2, 3), ndim=2),
                "mtp.norm.weight": FakeTensor((4, 5)),
            }
            (output / "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {"mtp_num_hidden_layers": 1},
                        "quantization_config": {"ignore": ["lm_head", "model.visual"]},
                    }
                )
            )
            (output / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 10, "total_parameters": 5},
                        "weight_map": {
                            "model.layers.0.weight_packed": "model.safetensors"
                        },
                    }
                )
            )

            def download(_repo: str, _revision: str, filename: str) -> str:
                if filename == "model.safetensors.index.json":
                    return str(source_index)
                if filename == source_shard.name:
                    return str(source_shard)
                raise AssertionError(filename)

            modules = {
                "torch": fake_torch,
                "safetensors": fake_safetensors,
                "safetensors.torch": fake_safetensors_torch,
            }
            with patch.dict(sys.modules, modules):
                result = preserve_mtp_weights(
                    "source/model", "pinned", output, download=download
                )

            saved_index = json.loads(
                (output / "model.safetensors.index.json").read_text()
            )
            self.assertTrue((output / MTP_SHARD).is_file())
            self.assertEqual(result["mtp_parameters"], 2)
            self.assertEqual(result["mtp_numel"], 5)
            self.assertEqual(result["mtp_nbytes"], 10)
            self.assertEqual(result["packed_weights"], 1)
            self.assertEqual(result["mtp_ignored_modules"], ["mtp.fc"])
            self.assertEqual(saved_index["metadata"]["total_size"], 20)
            self.assertEqual(saved_index["metadata"]["total_parameters"], 10)
            self.assertEqual(saved_index["weight_map"]["mtp.fc.weight"], MTP_SHARD)
            saved_config = json.loads((output / "config.json").read_text())
            self.assertIn("mtp.fc", saved_config["quantization_config"]["ignore"])

    def test_derives_only_two_dimensional_mtp_linear_modules(self) -> None:
        self.assertEqual(
            mtp_linear_modules(
                {
                    "mtp.fc.weight": (10, 20),
                    "mtp.norm.weight": (20,),
                }
            ),
            ["mtp.fc"],
        )
        with self.assertRaisesRegex(ValueError, "not a top-level MTP tensor"):
            mtp_linear_modules({"model.fc.weight": (10, 20)})

    def test_adds_mtp_ignores_without_mutating_or_duplicating(self) -> None:
        original = {"quantization_config": {"ignore": ["lm_head", "mtp.fc"]}}
        updated, added = merged_ignore(original, ["mtp.fc", "mtp.layers.0.mlp.up_proj"])
        self.assertEqual(
            original["quantization_config"]["ignore"], ["lm_head", "mtp.fc"]
        )
        self.assertEqual(added, ["mtp.layers.0.mlp.up_proj"])
        self.assertEqual(
            updated["quantization_config"]["ignore"],
            ["lm_head", "mtp.fc", "mtp.layers.0.mlp.up_proj"],
        )


def write_safetensors(
    path: Path, tensors: dict[str, tuple[tuple[int, ...], str]]
) -> None:
    """Write a real (zero-filled) safetensors container for header-only tests."""
    header: dict[str, object] = {}
    offset = 0
    for name, (shape, dtype) in tensors.items():
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * 2
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(offset))


# The pinned Qwen/Qwen3.8-27B revision, read from the source shard header. Eight
# of the fifteen tensors are 2-D and therefore become Linear modules in vLLM.
PINNED_SOURCE_MTP = {
    "mtp.fc.weight": (5120, 10240),
    "mtp.layers.0.input_layernorm.weight": (5120,),
    "mtp.layers.0.mlp.down_proj.weight": (5120, 17408),
    "mtp.layers.0.mlp.gate_proj.weight": (17408, 5120),
    "mtp.layers.0.mlp.up_proj.weight": (17408, 5120),
    "mtp.layers.0.post_attention_layernorm.weight": (5120,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (1024, 5120),
    "mtp.layers.0.self_attn.o_proj.weight": (5120, 6144),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.q_proj.weight": (12288, 5120),
    "mtp.layers.0.self_attn.v_proj.weight": (1024, 5120),
    "mtp.norm.weight": (5120,),
    "mtp.pre_fc_norm_embedding.weight": (5120,),
    "mtp.pre_fc_norm_hidden.weight": (5120,),
}

EXPECTED_VLLM_IGNORES = [
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]


class MtpIgnoreContractTests(unittest.TestCase):
    def test_pinned_source_keyset_yields_exactly_the_vllm_linears(self) -> None:
        """The names vLLM checks against quantization_config.ignore.

        vLLM builds the head under prefix "mtp" and fuses q/k/v into qkv_proj and
        gate/up into gate_up_proj; should_ignore_layer maps those back and raises
        if the shards disagree, so all of q/k/v and both of gate/up must appear.
        """
        self.assertEqual(set(PINNED_SOURCE_MTP), QWEN38_MTP_KEYS)
        self.assertEqual(set(EXPECTED_VLLM_IGNORES), QWEN38_MTP_LINEAR_MODULES)
        self.assertEqual(mtp_linear_modules(PINNED_SOURCE_MTP), EXPECTED_VLLM_IGNORES)

    def test_norm_tensors_are_never_ignored(self) -> None:
        modules = mtp_linear_modules(PINNED_SOURCE_MTP)
        self.assertFalse([name for name in modules if name.endswith("norm")])

    def test_shape_table_and_module_constants_cannot_drift(self) -> None:
        """Tie the measured shard header to the hardcoded release constants.

        PINNED_SOURCE_MTP carries the shapes read from the pinned source shard;
        QWEN38_MTP_* are the names the release gates assert. Deriving one from
        the other here means a future revision cannot update one and leave the
        other silently stale.
        """
        self.assertEqual(set(PINNED_SOURCE_MTP), set(QWEN38_MTP_KEYS))
        self.assertEqual(PINNED_SOURCE_MTP, QWEN38_MTP_SHAPES)
        self.assertEqual(
            set(mtp_linear_modules(PINNED_SOURCE_MTP)), set(QWEN38_MTP_LINEAR_MODULES)
        )

    def test_ignore_set_covers_vllm_fused_runtime_modules(self) -> None:
        """Exercise vLLM's exact fused-ignore decision for Qwen3.5 MTP.

        Mirrors should_ignore_layer in vLLM's compressed-tensors utils: fused
        qkv/gate-up modules are ignored only when every unfused component has
        the same decision. The registered Qwen3_5MTP class has no top-level
        hf_to_vllm_mapper, so these construction-time names remain mtp.*.
        """
        fused_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        ignores = set(QWEN38_MTP_LINEAR_MODULES)

        def is_ignored(layer_name: str) -> bool:
            projection = layer_name.rsplit(".", 1)[-1]
            if projection not in fused_mapping:
                return layer_name in ignores
            components = {
                layer_name.removesuffix(projection) + component in ignores
                for component in fused_mapping[projection]
            }
            self.assertEqual(
                len(components),
                1,
                "vLLM rejects mixed ignore decisions within a fused module",
            )
            return components.pop()

        for runtime_name in (
            "mtp.fc",
            "mtp.layers.0.self_attn.qkv_proj",
            "mtp.layers.0.self_attn.o_proj",
            "mtp.layers.0.mlp.gate_up_proj",
            "mtp.layers.0.mlp.down_proj",
        ):
            self.assertTrue(is_ignored(runtime_name), runtime_name)
        self.assertFalse(is_ignored("model.language_model.layers.5.mlp.down_proj"))


class RepairMtpIgnoresTests(unittest.TestCase):
    def _checkpoint(self, root: Path, *, dtype: str = "BF16") -> Path:
        output = root / "output"
        output.mkdir()
        write_safetensors(
            output / MTP_SHARD,
            {name: (shape, dtype) for name, shape in PINNED_SOURCE_MTP.items()},
        )
        weight_map = {"model.layers.0.weight_packed": "model-00001.safetensors"}
        weight_map.update({name: MTP_SHARD for name in PINNED_SOURCE_MTP})
        (output / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 10}, "weight_map": weight_map})
        )
        (output / "config.json").write_text(
            json.dumps(
                {
                    "text_config": {"mtp_num_hidden_layers": 1},
                    "quantization_config": {"ignore": ["lm_head"]},
                }
            )
        )
        return output

    def test_reads_header_without_torch_and_rejects_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard.safetensors"
            write_safetensors(path, {"mtp.fc.weight": ((4, 8), "BF16")})
            header = read_safetensors_header(path)
            self.assertEqual(header["mtp.fc.weight"]["shape"], [4, 8])
            self.assertEqual(header["mtp.fc.weight"]["dtype"], "BF16")

            path.write_bytes((4096).to_bytes(8, "little") + b"{}")
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_safetensors_header(path)

    def test_adds_the_linear_exclusions_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._checkpoint(Path(temporary))
            result = repair_mtp_ignores(output)
            self.assertEqual(result["mtp_ignores_added"], EXPECTED_VLLM_IGNORES)
            ignore = json.loads((output / "config.json").read_text())[
                "quantization_config"
            ]["ignore"]
            self.assertEqual(ignore, ["lm_head"] + EXPECTED_VLLM_IGNORES)

            again = repair_mtp_ignores(output)
            self.assertEqual(again["mtp_ignores_added"], [])
            self.assertEqual(
                json.loads((output / "config.json").read_text())["quantization_config"][
                    "ignore"
                ],
                ignore,
            )

    def test_refuses_a_checkpoint_that_was_never_grafted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.layers.0.weight_packed": "a.st"}})
            )
            (output / "config.json").write_text(
                json.dumps({"quantization_config": {"ignore": []}})
            )
            with self.assertRaisesRegex(RuntimeError, "no mtp"):
                repair_mtp_ignores(output)

    def test_refuses_a_downcast_mtp_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._checkpoint(Path(temporary), dtype="F8_E4M3")
            with self.assertRaisesRegex(RuntimeError, "not BF16"):
                repair_mtp_ignores(output)

    def test_complete_artifact_contract_and_missing_ignore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._checkpoint(Path(temporary))
            repair_mtp_ignores(output)
            result = validate_mtp_artifact(
                output,
                expected_keys=QWEN38_MTP_KEYS,
                expected_modules=QWEN38_MTP_LINEAR_MODULES,
                expected_shapes=QWEN38_MTP_SHAPES,
            )
            self.assertEqual(result["packed_weights"], 1)
            self.assertEqual(result["mtp_parameters"], 15)
            self.assertEqual(
                set(result["mtp_ignored_modules"]), QWEN38_MTP_LINEAR_MODULES
            )

            config_path = output / "config.json"
            config = json.loads(config_path.read_text())
            config["quantization_config"]["ignore"].remove("mtp.fc")
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, "absent from.*ignore"):
                validate_mtp_artifact(
                    output,
                    expected_keys=QWEN38_MTP_KEYS,
                    expected_modules=QWEN38_MTP_LINEAR_MODULES,
                    expected_shapes=QWEN38_MTP_SHAPES,
                )

    def test_complete_artifact_contract_rejects_wrong_keyset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._checkpoint(Path(temporary))
            repair_mtp_ignores(output)
            index_path = output / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            del index["weight_map"]["mtp.norm.weight"]
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(RuntimeError, "keyset differs"):
                validate_mtp_artifact(
                    output,
                    expected_keys=QWEN38_MTP_KEYS,
                    expected_modules=QWEN38_MTP_LINEAR_MODULES,
                    expected_shapes=QWEN38_MTP_SHAPES,
                )


if __name__ == "__main__":
    unittest.main()
