"""A pre-registration number that cannot be reproduced is not a record.

The rejection rates in EVAL.md come from this simulation, so what gets tested is
that it reproduces from its seed, that its parameters are the measured ones, and
that it moves in the directions the argument depends on: more suites means more
chances, a wider suite means a wider macro.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "simulate_gates", ROOT / "scripts" / "simulate_gates.py"
)
assert SPEC and SPEC.loader
sim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sim)

SETTINGS = dict(
    trials=4000,
    seed=38027,
    any_suite_drop=0.03,
    max_macro_drop=0.03,
    confident_drop=0.05,
    near_lossless=0.99,
    discordance_scale=1.0,
)


def run(names, **overrides):
    return sim.simulate(names, **{**SETTINGS, **overrides})


class ParameterTests(unittest.TestCase):
    def test_measured_suites_rescale_to_the_configuration_we_run(self) -> None:
        """GPQA was measured at R=4 and runs at R=1, so it must widen."""
        spec = sim.SUITES["gpqa_diamond"]
        error, modelled = sim.standard_error(spec, 1.0)
        self.assertFalse(modelled)
        measured = spec["half"] / sim.Z95
        self.assertGreater(error, measured)
        k = spec["k"]
        expected = measured * ((1 + k / 1) / (1 + k / 4)) ** 0.5
        self.assertAlmostEqual(error, expected, places=6)

    def test_a_suite_at_its_measured_configuration_does_not_move(self) -> None:
        spec = dict(sim.SUITES["ruler"], run_reps=1)
        error, _ = sim.standard_error(spec, 1.0)
        self.assertAlmostEqual(error, spec["half"] / sim.Z95, places=9)

    def test_unmeasured_suites_are_flagged_as_modelled(self) -> None:
        for name in ("livecodebench_v6",):
            with self.subTest(name=name):
                _, modelled = sim.standard_error(sim.SUITES[name], 1.0)
                self.assertTrue(modelled)

    def test_the_discordance_sweep_only_moves_modelled_suites(self) -> None:
        measured = sim.SUITES["bfcl_v4"]
        self.assertEqual(
            sim.standard_error(measured, 1.0), sim.standard_error(measured, 4.0)
        )
        low, _ = sim.standard_error(sim.SUITES["livecodebench_v6"], 1.0)
        high, _ = sim.standard_error(sim.SUITES["livecodebench_v6"], 4.0)
        self.assertAlmostEqual(high / low, 2.0, places=6)

    def test_the_parked_candidates_are_not_in_the_protocol(self) -> None:
        for name in ("swebench_pro_1_0", "terminal_bench_2_1"):
            with self.subTest(name=name):
                self.assertNotIn(name, sim.SUITES)

    def test_every_suite_in_the_protocol_has_parameters(self) -> None:
        protocol = importlib.util.spec_from_file_location(
            "run_eval_protocol", ROOT / "scripts" / "run_eval_protocol.py"
        )
        module = importlib.util.module_from_spec(protocol)
        protocol.loader.exec_module(module)
        self.assertEqual(set(sim.SUITES) - set(sim.CANDIDATE), module.REQUIRED_SUITES)


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = list(sim.SUITES)
        self.plus_one = self.protocol + list(sim.CANDIDATE)
        sim.SUITES.update(sim.CANDIDATE)
        self.addCleanup(lambda: [sim.SUITES.pop(n, None) for n in sim.CANDIDATE])

    def test_the_same_seed_reproduces_the_same_rates(self) -> None:
        self.assertEqual(run(self.protocol), run(self.protocol))

    def test_adding_a_suite_makes_the_any_suite_rule_fire_more(self) -> None:
        """This is the whole reason gates 1 and 2 replaced that rule."""
        self.assertGreater(
            run(self.plus_one)["any_suite_rule"], run(self.protocol)["any_suite_rule"]
        )

    def test_the_macro_rule_stays_far_below_the_any_suite_rule(self) -> None:
        report = run(self.protocol)
        self.assertLess(report["macro_rule"], report["any_suite_rule"] / 10)

    def test_a_wider_suite_widens_the_recovery_interval(self) -> None:
        narrow = run(self.protocol, discordance_scale=0.25)
        wide = run(self.protocol, discordance_scale=4.0)
        self.assertGreater(
            wide["recovery_geomean_95_halfwidth"],
            narrow["recovery_geomean_95_halfwidth"],
        )

    def test_the_null_recovery_geomean_interval_is_reported(self) -> None:
        report = run(self.protocol)
        self.assertGreater(report["recovery_geomean_95_halfwidth"], 0.0)

    def test_a_gentler_bar_is_failed_less_often_under_the_null(self) -> None:
        strict = run(self.protocol, near_lossless=0.99)
        loose = run(self.protocol, near_lossless=0.95)
        self.assertLess(loose["near_lossless_failure"], strict["near_lossless_failure"])

    def test_modelled_suites_are_named_in_the_report(self) -> None:
        self.assertEqual(
            run(self.protocol)["modelled"], ["livecodebench_v6", "mmmu_pro"],
        )


class CommandTests(unittest.TestCase):
    def test_the_json_form_reports_both_suite_counts(self) -> None:
        self.assertEqual(sim.main(["--trials", "500", "--json"]), 0)


if __name__ == "__main__":
    unittest.main()
