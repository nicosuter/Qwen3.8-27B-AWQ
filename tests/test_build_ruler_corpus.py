import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "build_ruler_corpus", ROOT / "scripts" / "build_ruler_corpus.py"
)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class NormalizeTests(unittest.TestCase):
    def test_line_endings_and_trailing_space_are_normalized(self) -> None:
        self.assertEqual(
            builder.normalize(["a  \r\nb\r", "c \n"]),
            "a\nb\n\nc\n",
        )

    def test_documents_are_separated_by_one_blank_line(self) -> None:
        self.assertEqual(builder.normalize(["one", "two", "three"]), "one\n\ntwo\n\nthree\n")

    def test_empty_documents_are_dropped(self) -> None:
        self.assertEqual(builder.normalize(["kept", "   ", ""]), "kept\n")

    def test_all_empty_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            builder.normalize(["", "  \n "])

    def test_unicode_is_composed_so_the_hash_is_stable(self) -> None:
        decomposed, composed = "café", "café"
        self.assertEqual(builder.normalize([decomposed]), builder.normalize([composed]))


class BuildTests(unittest.TestCase):
    def build(self, documents: list[str]) -> dict:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        args = SimpleNamespace(
            dataset="fixture/essays",
            revision="c" * 12,
            split="train",
            text_field="text",
            output=Path(tmp.name) / "haystack.txt",
        )
        return builder.build(args, loader=lambda *_: documents)

    def test_hash_is_reproducible_for_identical_input(self) -> None:
        first = self.build(["alpha", "beta"])
        second = self.build(["alpha", "beta"])
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertTrue(first["sha256"].startswith("sha256:"))

    def test_hash_changes_when_the_corpus_changes(self) -> None:
        self.assertNotEqual(
            self.build(["alpha", "beta"])["sha256"],
            self.build(["alpha", "gamma"])["sha256"],
        )

    def test_document_order_is_significant(self) -> None:
        self.assertNotEqual(
            self.build(["alpha", "beta"])["sha256"],
            self.build(["beta", "alpha"])["sha256"],
        )

    def test_output_file_matches_the_reported_hash(self) -> None:
        report = self.build(["alpha", "beta"])
        written = Path(report["output"]).read_text(encoding="utf-8")
        self.assertEqual(written, "alpha\n\nbeta\n")
        self.assertEqual(report["characters"], len(written))

    def test_missing_text_column_is_reported(self) -> None:
        def loader(*_):
            raise ValueError("fixture/essays has no 'text' column; got ['content']")

        with self.assertRaises(ValueError):
            builder.build(
                SimpleNamespace(
                    dataset="fixture/essays",
                    revision="c" * 12,
                    split="train",
                    text_field="text",
                    output=Path("unused"),
                ),
                loader=loader,
            )


if __name__ == "__main__":
    unittest.main()
