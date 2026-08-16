"""The subset is a pre-registration, so its properties are asserted, not assumed.

Three of them carry the whole argument for sampling at all: the draw reproduces
from the recorded seed, every prefix is repo-proportional so the size can move
without the sample changing identity, and a registry that has shifted under us
is refused rather than silently redrawn.
"""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "swebenchpro_subset", ROOT / "scripts" / "swebenchpro_subset.py"
)
subset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subset)

SHA = "a" * 40
OTHER = "b" * 40
# Roughly the real shape: a few large repos, one small one, and the three task
# name spellings the registry actually contains.
POPULATION = {
    "ansible__ansible": 96,
    "internetarchive__openlibrary": 91,
    "flipt-io__flipt": 85,
    "qutebrowser__qutebrowser": 79,
    "element-hq__element-web": 56,
    "tutao__tutanota": 20,
}


def task_name(repo: str, index: int, style: str = "versioned") -> str:
    base = f"{index:040x}"
    if style == "versioned":
        return f"instance_{repo}-{base}-v{OTHER}"
    if style == "nan":
        return f"instance_{repo}-{base}-vnan"
    return f"instance_{repo}-{base}"


def make_registry(path: Path, population: dict[str, int] | None = None) -> Path:
    styles = ("versioned", "nan", "bare")
    tasks = []
    for repo, count in (population or POPULATION).items():
        for index in range(count):
            tasks.append({"name": task_name(repo, index, styles[index % 3])})
    entries = [
        {"name": "terminal-bench", "version": "2.0", "tasks": [{"name": "unrelated"}]},
        {"name": "swebenchpro", "version": "1.0", "tasks": tasks},
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


class NameParsingTests(unittest.TestCase):
    def test_every_registry_name_shape_yields_a_repo(self) -> None:
        for style in ("versioned", "nan", "bare"):
            with self.subTest(style=style):
                self.assertEqual(
                    subset.task_repo(task_name("ansible__ansible", 3, style)),
                    "ansible__ansible",
                )

    def test_a_repo_name_containing_digits_survives(self) -> None:
        self.assertEqual(
            subset.task_repo(task_name("scaleapi__swe2bench", 1)), "scaleapi__swe2bench"
        )

    def test_an_unreadable_name_is_an_error_not_a_guess(self) -> None:
        with self.assertRaises(subset.SubsetError):
            subset.task_repo("ansible__ansible-1234")


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.path = make_registry(Path(self.tmp.name) / "registry.json")
        self.addCleanup(self.tmp.cleanup)

    def test_only_the_named_dataset_is_read(self) -> None:
        names, _ = subset.load_registry(self.path, "swebenchpro", "1.0")
        self.assertEqual(len(names), sum(POPULATION.values()))
        self.assertNotIn("unrelated", names)

    def test_the_digest_covers_the_file_bytes(self) -> None:
        _, first = subset.load_registry(self.path, "swebenchpro", "1.0")
        text = self.path.read_text(encoding="utf-8")
        self.path.write_text(text + " ", encoding="utf-8")
        _, second = subset.load_registry(self.path, "swebenchpro", "1.0")
        self.assertNotEqual(first, second)

    def test_a_missing_dataset_is_refused(self) -> None:
        with self.assertRaises(subset.SubsetError):
            subset.load_registry(self.path, "swebenchpro", "2.0")

    def test_duplicate_task_names_are_refused(self) -> None:
        entries = json.loads(self.path.read_text(encoding="utf-8"))
        entries[1]["tasks"].append(dict(entries[1]["tasks"][0]))
        self.path.write_text(json.dumps(entries), encoding="utf-8")
        with self.assertRaises(subset.SubsetError):
            subset.load_registry(self.path, "swebenchpro", "1.0")


class OrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.path = make_registry(Path(self.tmp.name) / "registry.json")
        self.names, self.digest = subset.load_registry(self.path, "swebenchpro", "1.0")
        self.addCleanup(self.tmp.cleanup)

    def draw(self, size: int, seed: int = subset.DEFAULT_SEED) -> list[str]:
        return subset.build(self.names, size, seed, self.digest)["task_names"]

    def test_the_ordering_covers_the_population_exactly_once(self) -> None:
        ordered = subset.nested_order(self.names, subset.DEFAULT_SEED)
        self.assertEqual(sorted(ordered), sorted(self.names))

    def test_the_same_seed_reproduces_the_same_draw(self) -> None:
        self.assertEqual(self.draw(300), self.draw(300))

    def test_a_different_seed_draws_a_different_sample(self) -> None:
        self.assertNotEqual(self.draw(300), self.draw(300, seed=subset.DEFAULT_SEED + 1))

    def test_a_smaller_size_is_a_prefix_of_a_larger_one(self) -> None:
        """The nesting property: resizing must not redraw the sample."""
        large = self.draw(360)
        for size in (24, 120, 240, 300):
            with self.subTest(size=size):
                self.assertEqual(self.draw(size), large[:size])

    def test_every_prefix_holds_the_repo_shares(self) -> None:
        total = len(self.names)
        for size in (24, 60, 120, 240, 300, 360):
            counts = subset.repo_counts(self.draw(size))
            for repo, population in POPULATION.items():
                expected = size * population / total
                # Largest-remainder apportionment cannot do better than one
                # whole item either way at any given cut.
                self.assertLessEqual(
                    abs(counts.get(repo, 0) - expected), 1.0, f"{repo} at n={size}"
                )

    def test_a_small_repo_is_not_dropped_entirely(self) -> None:
        counts = subset.repo_counts(self.draw(300))
        self.assertGreater(counts.get("tutao__tutanota", 0), 0)

    def test_the_pin_changes_with_the_sample(self) -> None:
        first = subset.build(self.names, 300, subset.DEFAULT_SEED, self.digest)
        second = subset.build(self.names, 301, subset.DEFAULT_SEED, self.digest)
        self.assertNotEqual(first["subset_pin"], second["subset_pin"])
        # The ordering is the same one, so its pin must not move with the cut.
        self.assertEqual(first["order_pin"], second["order_pin"])

    def test_sizes_outside_the_population_are_refused(self) -> None:
        for size in (0, -1, len(self.names) + 1):
            with self.subTest(size=size), self.assertRaises(subset.SubsetError):
                self.draw(size)

    def test_the_whole_population_can_be_taken(self) -> None:
        self.assertEqual(len(self.draw(len(self.names))), len(self.names))


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.registry = make_registry(self.dir / "registry.json")
        self.out = self.dir / "subset.json"
        self.addCleanup(self.tmp.cleanup)
        subset.main(
            ["select", "--registry", str(self.registry), "--size", "300",
             "--out", str(self.out)]
        )

    def test_select_writes_a_verifiable_subset(self) -> None:
        self.assertEqual(
            subset.main(
                ["verify", "--registry", str(self.registry), "--subset", str(self.out)]
            ),
            0,
        )

    def test_verify_refuses_a_registry_that_has_moved(self) -> None:
        entries = json.loads(self.registry.read_text(encoding="utf-8"))
        entries[1]["tasks"].append({"name": task_name("ansible__ansible", 999)})
        self.registry.write_text(json.dumps(entries), encoding="utf-8")
        with self.assertRaises(subset.SubsetError):
            subset.main(
                ["verify", "--registry", str(self.registry), "--subset", str(self.out)]
            )

    def test_verify_catches_an_edited_task_list(self) -> None:
        stored = json.loads(self.out.read_text(encoding="utf-8"))
        stored["task_names"][0] = stored["task_names"][-1]
        self.out.write_text(json.dumps(stored), encoding="utf-8")
        with self.assertRaises(subset.SubsetError):
            subset.main(
                ["verify", "--registry", str(self.registry), "--subset", str(self.out)]
            )

    def test_the_written_file_records_what_was_drawn(self) -> None:
        stored = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(stored["size"], 300)
        self.assertEqual(stored["population"], sum(POPULATION.values()))
        self.assertEqual(len(stored["task_names"]), 300)
        self.assertEqual(sum(stored["repo_counts"].values()), 300)

    def test_plan_reports_every_repo(self) -> None:
        self.assertEqual(
            subset.main(["plan", "--registry", str(self.registry), "--sizes", "120", "300"]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
