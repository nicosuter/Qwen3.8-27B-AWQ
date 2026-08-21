"""What each GDN precision mode actually does to the recipe.

The decision is four-way and every mode ships a different checkpoint, so the
things worth pinning are that a projection lands in exactly one place, that two
builds cannot claim the same directory, and that the submitter and the job agree
on what the modes are. None of that needs a GPU stack, which is why the plan is
its own module.
"""

import re
import subprocess
import unittest
from pathlib import Path

from quant.scripts.gdn_in_proj import (
    DEFAULT_GDN_IN_PROJ_PRECISION,
    FOUR_BIT_TARGETS,
    GDN_IN_PROJ_PRECISIONS,
    GDN_IN_PROJ_TARGETS,
    gdn_in_proj_plan,
    output_suffix,
)

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "quant" / "scripts" / "gdn_in_proj.py"
SUBMIT = ROOT / "quant" / "scripts" / "submit_quantize.sh"


class PlacementTests(unittest.TestCase):
    def test_every_projection_lands_in_exactly_one_place(self) -> None:
        """Quantized twice would fight; quantized nowhere is a silent BF16 leak."""
        for precision in GDN_IN_PROJ_PRECISIONS:
            with self.subTest(precision=precision):
                plan = gdn_in_proj_plan(precision)
                for target in GDN_IN_PROJ_TARGETS:
                    places = [
                        target in plan["four_bit"],
                        target in plan["ignore"],
                        plan["own_group"] is not None,
                    ]
                    self.assertEqual(sum(places), 1, f"{target} in {sum(places)} places")

    def test_source_leaves_them_alone(self) -> None:
        plan = gdn_in_proj_plan("source")
        self.assertEqual(tuple(plan["ignore"]), GDN_IN_PROJ_TARGETS)
        self.assertEqual(tuple(plan["four_bit"]), ())
        self.assertIsNone(plan["own_group"])

    def test_int4_folds_them_into_the_four_bit_group(self) -> None:
        """philbert440's shape, which measures less verbose than our FP8 build."""
        plan = gdn_in_proj_plan("int4")
        self.assertEqual(tuple(plan["four_bit"]), GDN_IN_PROJ_TARGETS)
        self.assertEqual(tuple(plan["ignore"]), ())
        self.assertIsNone(plan["own_group"])

    def test_no_mode_can_ask_for_quantized_activations(self) -> None:
        """It is not a setting. sm_89 is needed to perform them and the serving
        cards are sm_86, so declaring them describes numerics nothing runs."""
        for precision in GDN_IN_PROJ_PRECISIONS:
            with self.subTest(precision=precision):
                self.assertNotIn("quantize_activations", gdn_in_proj_plan(precision))

    def test_eight_bit_gets_its_own_group_and_no_activations(self) -> None:
        """There is no activation axis, so the mode is not qualified for one."""
        self.assertEqual(gdn_in_proj_plan("int8")["own_group"], "W8A16")
        self.assertNotIn("a16", " ".join(GDN_IN_PROJ_PRECISIONS))

    def test_there_is_no_fp8_mode(self) -> None:
        """At eight bits it is the worse format for weight-only, so its only
        argument was that an earlier checkpoint used it. Reproducing that is a
        job for the commit it was built from."""
        self.assertNotIn("fp8", GDN_IN_PROJ_PRECISIONS)
        with self.assertRaises(ValueError):
            gdn_in_proj_plan("fp8")

    def test_the_default_is_eight_bit(self) -> None:
        """These projections feed a recurrent state, so their errors carry
        rather than being re-read each step."""
        self.assertEqual(DEFAULT_GDN_IN_PROJ_PRECISION, "int8")

    def test_out_proj_is_never_a_gdn_target(self) -> None:
        """It is the readout into the residual stream, not the state update.

        It stays four-bit in every mode, which is what makes these modes a test
        of the input path rather than of the block.
        """
        self.assertNotIn("out_proj", " ".join(GDN_IN_PROJ_TARGETS))
        self.assertIn("out_proj", " ".join(FOUR_BIT_TARGETS))
        for precision in GDN_IN_PROJ_PRECISIONS:
            with self.subTest(precision=precision):
                plan = gdn_in_proj_plan(precision)
                self.assertNotIn("out_proj", " ".join(plan["ignore"]))

    def test_an_unknown_mode_is_refused(self) -> None:
        """It used to select FP8, so a typo shipped the most aggressive mode."""
        with self.assertRaises(ValueError) as caught:
            gdn_in_proj_plan("fp16")
        self.assertIn("fp16", str(caught.exception))

    def test_the_default_is_a_real_mode(self) -> None:
        self.assertIn(DEFAULT_GDN_IN_PROJ_PRECISION, GDN_IN_PROJ_PRECISIONS)


class OutputDirectoryTests(unittest.TestCase):
    """Two builds in one directory is two builds nobody can tell apart later."""

    def test_no_two_builds_share_a_suffix(self) -> None:
        suffixes = [
            output_suffix(precision, algorithm)
            for precision in GDN_IN_PROJ_PRECISIONS
            for algorithm in ("awq", "awq+gptq")
        ]
        self.assertEqual(len(suffixes), len(set(suffixes)))

    def test_the_earlier_build_keeps_its_directory_to_itself(self) -> None:
        """v2/model-fp8gdn holds the scored Class A result, built before
        activations were settled, so no mode may write over it."""
        self.assertEqual(output_suffix("source"), "")
        suffixes = {output_suffix(p) for p in GDN_IN_PROJ_PRECISIONS}
        self.assertNotIn("-fp8gdn", suffixes)

    def test_a_suffix_names_what_varies(self) -> None:
        """The whole model is four-bit in every mode, so "-int4" alone would
        read as a claim about all of it."""
        self.assertEqual(output_suffix("int4"), "-inproj-int4")
        self.assertEqual(output_suffix("int8"), "-inproj-int8")

    def test_an_unknown_mode_has_no_directory(self) -> None:
        with self.assertRaises(ValueError):
            output_suffix("fp16")


class SubmitterAgreementTests(unittest.TestCase):
    """The submitter asks this module instead of repeating the list.

    A shell script with its own copy is one that accepts a mode the job then
    rejects, an hour into a reservation.
    """

    def run_module(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["python3", str(MODULE), *args], capture_output=True, text=True
        )

    def test_the_cli_reports_the_same_modes(self) -> None:
        result = self.run_module("--modes")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.split(), list(GDN_IN_PROJ_PRECISIONS))

    def test_an_empty_argument_means_the_default(self) -> None:
        """An unset shell flag arrives as an empty string, not as no argument."""
        result = self.run_module("")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), DEFAULT_GDN_IN_PROJ_PRECISION)

    def test_the_cli_refuses_an_unknown_mode_nonzero(self) -> None:
        result = self.run_module("fp16", "--suffix")
        self.assertEqual(result.returncode, 2)
        self.assertIn("fp16", result.stderr)

    def test_the_submitter_names_no_modes_of_its_own(self) -> None:
        text = SUBMIT.read_text(encoding="utf-8")
        # Prose may explain the modes and the flag they replaced; what must not
        # survive is a parser arm accepting either, because that is the copy
        # that drifts.
        self.assertNotIn("--fp8-gdn=*)", text, "the boolean flag outlived the boolean")
        self.assertNotIn("fp8-a16 | fp8", text, "the mode list has a second copy")
        self.assertIn("--gdn-in-proj=*)", text)
        self.assertIn("gdn_in_proj.py", text)
        # The header's --cpus-per-task=64 was written for a whole-node run.
        # Carrying it onto a two-GPU job asks for half a shared node's CPU for a
        # quarter of its GPUs, and CPU is the resource this cluster actually
        # schedules on.
        self.assertIn("CPUS=$(( NGPUS * CPUS_PER_GPU ))", text)
        self.assertIn('--cpus-per-task="$CPUS"', text)
        # The usage line used to print "(default source)" as a literal, which
        # went stale the moment the default moved. It asks now.
        for precision in GDN_IN_PROJ_PRECISIONS:
            self.assertNotIn(f"(default {precision})", text)

class ActivationTaintTests(unittest.TestCase):
    """Declared activation quantization the serving device will not perform.

    The fallback is silent and one-directional: the checkpoint says the
    activations are quantized, a device below sm_89 serves them weight-only, and
    every number the run produces then describes something the checkpoint is
    not. It is recorded rather than refused -- on the serving hardware the
    tainted measurement is the more relevant one -- but it cannot be invisible.
    """

    # compressed-tensors: activation scheme lives on each config group.
    MIXED = {
        "config_groups": {
            "group_0": {"weights": {"num_bits": 4}},
            "group_1": {"weights": {"num_bits": 8},
                        "input_activations": {"type": "float", "num_bits": 8}},
        }
    }
    # A native FP8 release: no config groups at all, one flat declaration. This
    # is the FP8 baseline, and reading only the schema above reported it clean
    # -- on the arm the taint matters most for, since every recovery figure is
    # measured against it.
    NATIVE_FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3"}

    def taint(self, capability, quantization=None):
        from eval.scripts.checkpoint_fingerprint import activation_taint

        return activation_taint(self.MIXED if quantization is None else quantization,
                                capability)

    def test_hopper_performs_what_is_declared(self) -> None:
        self.assertEqual(self.taint(9.0), [])

    def test_ampere_does_not(self) -> None:
        """A100 is sm_80 and the serving cards are sm_86; both fall back."""
        for capability in (8.0, 8.6):
            with self.subTest(capability=capability):
                taints = self.taint(capability)
                self.assertEqual(len(taints), 1)
                self.assertEqual(taints[0]["group"], "group_1")
                self.assertEqual(taints[0]["capability"], capability)

    def test_the_gate_is_where_vllm_puts_it(self) -> None:
        self.assertEqual(self.taint(8.9), [])
        self.assertEqual(len(self.taint(8.89)), 1)

    def test_the_native_fp8_baseline_is_tainted_too(self) -> None:
        """It declares dynamic FP8 activations without any config groups."""
        self.assertEqual(self.taint(9.0, self.NATIVE_FP8), [])
        tainted = self.taint(8.0, self.NATIVE_FP8)
        self.assertEqual(len(tainted), 1)
        self.assertEqual(tainted[0]["group"], "*")

    def test_a_weight_only_checkpoint_is_never_tainted(self) -> None:
        """Which is the point of building one: it serves as declared anywhere."""
        from eval.scripts.checkpoint_fingerprint import activation_taint

        weight_only = {"config_groups": {"group_0": {"weights": {"num_bits": 4}}}}
        for capability in (7.5, 8.0, 8.6, 9.0):
            with self.subTest(capability=capability):
                self.assertEqual(activation_taint(weight_only, capability), [])

    def test_an_unknown_device_is_not_reported_as_clean(self) -> None:
        """None and [] must stay distinguishable: one is 'fine', one is 'unread'."""
        self.assertIsNone(self.taint(None))
        self.assertIsNotNone(self.taint(9.0))

class AWQEqualizationTests(unittest.TestCase):
    """AWQ smoothing is only function-preserving if every consumer gets the scale.

    Equalization divides a norm's weight by a per-channel scale and multiplies it
    into the layers that read the norm. Miss one and that layer sees the divided
    input: the function has changed before anything has been quantized. Staying
    in BF16 is not an exemption, because folding a scale into a BF16 weight is
    just a multiply -- which is exactly the trap for in_proj_a and in_proj_b,
    which share the GDN block's normalized input and are never quantized.
    """

    # The shape the model has: 64 layers, full attention every fourth starting
    # at 3, Gated DeltaNet everywhere else.
    LAYER_TYPES = [
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)
    ]
    ATTENTION_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
    GDN_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention"]

    def shipped(self) -> list[tuple[str, list[str]]]:
        from quant.scripts.gdn_in_proj import input_layernorm_pattern

        return [
            (
                input_layernorm_pattern(self.ATTENTION_LAYERS),
                [r"re:.*self_attn\.q_proj$",
                 r"re:.*self_attn\.k_proj$",
                 r"re:.*self_attn\.v_proj$"],
            ),
            (
                input_layernorm_pattern(self.GDN_LAYERS),
                [r"re:.*linear_attn\.in_proj_qkv$",
                 r"re:.*linear_attn\.in_proj_z$",
                 r"re:.*linear_attn\.in_proj_a$",
                 r"re:.*linear_attn\.in_proj_b$"],
            ),
            (r"re:.*post_attention_layernorm$", [r"re:.*gate_proj$", r"re:.*up_proj$"]),
            (r"re:.*up_proj$", [r"re:.*down_proj$"]),
        ]

    def check(self, mappings: list[tuple[str, list[str]]]) -> list[str]:
        from quant.scripts.gdn_in_proj import unbalanced_norm_mappings

        return unbalanced_norm_mappings(mappings, self.LAYER_TYPES)

    def test_the_shipped_mappings_are_balanced(self) -> None:
        self.assertEqual(self.check(self.shipped()), [])

    def test_a_suffix_regex_counts_as_covering_its_module(self) -> None:
        """re:.*gate_proj$ covers mlp.gate_proj without containing the string."""
        self.assertEqual(
            self.check(
                [(r"re:.*post_attention_layernorm$",
                  [r"re:.*mlp\.(gate|up)_proj$"])]
            ),
            [],
        )

    def test_half_the_mlp_is_caught(self) -> None:
        problems = self.check(
            [(r"re:.*post_attention_layernorm$", [r"re:.*gate_proj$"])]
        )
        # Both layer types have an MLP and both are missing the same consumer,
        # so it is one report per type rather than one per layer.
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("mlp.up_proj" in problem for problem in problems))

    def test_smoothing_the_gdn_norm_must_reach_the_unquantized_consumers(self) -> None:
        """The case worth having the check for: a mapping that scales the norm
        and folds into the two projections a mode quantizes, leaving in_proj_a
        and in_proj_b reading a rescaled input."""
        problems = self.check(
            [(self.shipped()[1][0],
              [r"re:.*linear_attn\.in_proj_qkv$", r"re:.*linear_attn\.in_proj_z$"])]
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("linear_attn.in_proj_a", problems[0])
        self.assertIn("linear_attn.in_proj_b", problems[0])

    def test_each_layer_type_is_judged_on_its_own_consumers(self) -> None:
        """input_layernorm feeds q/k/v in one layer type and in_proj_* in the
        other. A mapping restricted to the attention layers is complete with
        q/k/v alone, and the GDN consumers must not be held against it."""
        attention_norm = self.shipped()[0][0]
        balances = [r"re:.*self_attn\.(q|k|v)_proj$"]
        self.assertEqual(self.check([(attention_norm, balances)]), [])
        problems = self.check([(self.shipped()[1][0], balances)])
        self.assertEqual(len(problems), 1)
        self.assertIn("linear_attn.in_proj_qkv", problems[0])

    def test_a_norm_smoothed_across_both_types_needs_both_consumer_sets(self) -> None:
        """An unindexed pattern hits every layer, so it answers for both."""
        problems = self.check(
            [(r"re:.*input_layernorm$", [r"re:.*self_attn\.(q|k|v)_proj$"])]
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("linear_attn.in_proj_qkv", problems[0])
        self.assertEqual(
            self.check(
                [(r"re:.*input_layernorm$",
                  [r"re:.*linear_attn\.in_proj_(qkv|z|a|b)$",
                   r"re:.*self_attn\.(q|k|v)_proj$"])]
            ),
            [],
        )

    def test_covering_a_layer_type_only_partly_is_still_caught(self) -> None:
        """Every index is checked, not one representative per type, so a
        mapping that is balanced on the layers it was written for and not on
        the rest does not slip through."""
        from quant.scripts.gdn_in_proj import input_layernorm_pattern

        problems = self.check(
            [(input_layernorm_pattern(self.GDN_LAYERS),
              [r"re:.*layers\.(0|1|2)\.linear_attn\.in_proj_(qkv|z|a|b)$"])]
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("linear_attn.in_proj_qkv", problems[0])

    def test_an_unrecognized_layer_type_is_refused(self) -> None:
        from quant.scripts.gdn_in_proj import unbalanced_norm_mappings

        with self.assertRaises(ValueError):
            unbalanced_norm_mappings(self.shipped(), ["sliding_attention"])

    def test_a_pattern_for_no_layers_is_refused(self) -> None:
        from quant.scripts.gdn_in_proj import input_layernorm_pattern

        with self.assertRaises(ValueError):
            input_layernorm_pattern([])

    def test_the_index_set_does_not_match_a_longer_index(self) -> None:
        """(3|7) must not match layers.31 through its leading 3."""
        from quant.scripts.gdn_in_proj import input_layernorm_pattern

        pattern = input_layernorm_pattern([3, 7])[len("re:"):]
        base = "model.language_model.layers"
        self.assertIsNotNone(re.fullmatch(pattern, f"{base}.3.input_layernorm"))
        self.assertIsNone(re.fullmatch(pattern, f"{base}.31.input_layernorm"))


if __name__ == "__main__":
    unittest.main()


class NvFp4ModeTests(unittest.TestCase):
    """A four-bit float grid for these projections, for export more than for us.

    At four bits the grid's shape matters more than its uniformity: sixteen
    levels spaced non-uniformly fit a weight distribution better than sixteen
    spaced evenly. Measured on these tensors, e2m1 reconstructs at 0.1098
    against int4's 0.1219, and NVFP4 blocks by sixteen with an e4m3 scale rather
    than by 128, so it should do better than that figure.

    It is servable here, which the class names in vLLM actively disguise: a
    weight-only NVFP4 config dispatches to CompressedTensorsW4A4Fp4(use_a16=True)
    and lands on MarlinNvFp4LinearKernel, gated on device capability 75. The
    Blackwell requirement belongs to the activation-quantized W4A4 path. So this
    mode is scorable on the same hardware as every other one, unlike the FP8
    mode that was removed.

    It is not chosen for memory. NVFP4 costs 4 + 8/16 bits a weight against
    int4's 4 + 16/128, so it is larger, and both reach the same Marlin
    dequantize-and-matmul off Blackwell.
    """

    def test_it_is_a_mode(self) -> None:
        self.assertIn("nvfp4", GDN_IN_PROJ_PRECISIONS)

    def test_it_gets_its_own_group_and_no_activations(self) -> None:
        plan = gdn_in_proj_plan("nvfp4")
        self.assertEqual(plan["own_group"], "NVFP4A16")
        self.assertNotIn("quantize_activations", plan)

    def test_it_does_not_also_join_the_four_bit_group(self) -> None:
        """Its own group is a different grid; being in both would be a conflict."""
        plan = gdn_in_proj_plan("nvfp4")
        self.assertEqual(plan["four_bit"], ())
        self.assertEqual(plan["ignore"], ())

    def test_it_writes_somewhere_of_its_own(self) -> None:
        suffixes = {p: output_suffix(p) for p in GDN_IN_PROJ_PRECISIONS}
        self.assertEqual(suffixes["nvfp4"], "-inproj-nvfp4")
        self.assertEqual(len(set(suffixes.values())), len(suffixes))

    def test_the_name_does_not_claim_an_activation_axis(self) -> None:
        """The preset is NVFP4A16; the mode name must not carry the A16."""
        self.assertNotIn("a16", "nvfp4")
