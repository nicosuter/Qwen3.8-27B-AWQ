"""What each GDN precision mode actually does to the recipe.

The decision is four-way and every mode ships a different checkpoint, so the
things worth pinning are that a projection lands in exactly one place, that two
builds cannot claim the same directory, and that the submitter and the job agree
on what the modes are. None of that needs a GPU stack, which is why the plan is
its own module.
"""

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

    def test_only_fp8_quantizes_activations(self) -> None:
        """The one mode that quantizes an activation anywhere in the model.

        On sm_86 it is also silently identical to fp8-a16 at run time, so the
        two must stay distinguishable everywhere else.
        """
        self.assertTrue(gdn_in_proj_plan("fp8")["quantize_activations"])
        self.assertFalse(gdn_in_proj_plan("fp8-a16")["quantize_activations"])
        self.assertEqual(gdn_in_proj_plan("fp8")["own_group"], gdn_in_proj_plan("fp8-a16")["own_group"])

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

    def test_the_scored_fp8_build_keeps_its_directory(self) -> None:
        """v2/model-fp8gdn exists and has a complete paired result against it."""
        self.assertEqual(output_suffix("fp8"), "-fp8gdn")
        self.assertEqual(output_suffix("source"), "")

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


if __name__ == "__main__":
    unittest.main()
