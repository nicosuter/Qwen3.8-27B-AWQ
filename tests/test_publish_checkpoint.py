import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "publish_checkpoint", ROOT / "quant" / "scripts" / "publish_checkpoint.py"
)
assert SPEC and SPEC.loader
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def make_artifact(tmp: Path, shards=("model-00001-of-00002.safetensors",
                                     "model-00002-of-00002.safetensors"),
                  extras=(), index_shards=None) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "config.json").write_text("{}", encoding="utf-8")
    for shard in shards:
        (tmp / shard).write_bytes(b"weights")
    referenced = index_shards if index_shards is not None else shards
    weight_map = {f"layer.{i}.weight": shard for i, shard in enumerate(referenced)}
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8")
    (tmp / "README.md").write_text("card", encoding="utf-8")
    for extra in extras:
        path = tmp / extra
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("junk", encoding="utf-8")
    return tmp


def args(path, **overrides):
    defaults = {"repo": "acme/model", "path": path, "message": "m", "delete": None,
                "execute": False, "skip_mtp_check": True}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class ArtifactCheckTests(unittest.TestCase):
    def test_healthy_artifact_reports_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = make_artifact(Path(tmp) / "a")
            report = publisher.check_artifact(path, skip_mtp=True)
        self.assertEqual(report["shards"], 2)
        self.assertEqual(report["tensors"], 2)

    def test_unreferenced_shard_is_refused(self) -> None:
        # This is the stale-weight failure, caught before it reaches the Hub.
        with tempfile.TemporaryDirectory() as tmp:
            path = make_artifact(
                Path(tmp) / "a",
                shards=("model-00001-of-00002.safetensors", "model-00001-of-00005.safetensors"),
                index_shards=("model-00001-of-00005.safetensors",),
            )
            with self.assertRaises(publisher.PublishError) as caught:
                publisher.check_artifact(path, skip_mtp=True)
        self.assertIn("not referenced by the index", str(caught.exception))

    def test_missing_shard_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = make_artifact(Path(tmp) / "a", shards=(),
                                 index_shards=("model-00001-of-00001.safetensors",))
            with self.assertRaises(publisher.PublishError):
                publisher.check_artifact(path, skip_mtp=True)

    def test_missing_config_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = make_artifact(Path(tmp) / "a")
            (path / "config.json").unlink()
            with self.assertRaises(publisher.PublishError):
                publisher.check_artifact(path, skip_mtp=True)


class LocalFileTests(unittest.TestCase):
    def test_local_state_is_never_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = make_artifact(Path(tmp) / "a",
                                 extras=(".omc/state/x.json", ".gitignore",
                                         "serve.log", "__pycache__/y.pyc"))
            names = publisher.local_files(path)
        self.assertNotIn(".gitignore", names)
        self.assertFalse([n for n in names if n.startswith(".omc/")])
        self.assertFalse([n for n in names if n.endswith((".log", ".pyc"))])
        self.assertIn("README.md", names)


class PlanTests(unittest.TestCase):
    def test_stale_shards_are_pruned(self) -> None:
        plan = publisher.plan_commit(
            ["model-00001-of-00005.safetensors", "config.json"],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
             "config.json", "README.md"],
            ("*.safetensors",),
        )
        self.assertEqual(plan["prune"],
                         ["model-00001-of-00002.safetensors",
                          "model-00002-of-00002.safetensors"])
        self.assertEqual(plan["upload"], ["model-00001-of-00005.safetensors"])
        self.assertEqual(plan["replace"], ["config.json"])

    def test_hub_only_files_are_left_alone(self) -> None:
        # A card maintained only on the Hub must survive a weights-only prune.
        plan = publisher.plan_commit(
            ["config.json"], ["config.json", "README.md", "old.safetensors"],
            ("*.safetensors",),
        )
        self.assertEqual(plan["prune"], ["old.safetensors"])
        self.assertEqual(plan["left_alone"], ["README.md"])


class PublishTests(unittest.TestCase):
    def run_publish(self, execute, remote, **overrides):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = make_artifact(Path(tmp) / "a",
                                 shards=("model-00001-of-00005.safetensors",))
            result = publisher.publish(
                args(path, execute=execute, **overrides),
                list_remote=lambda repo: remote,
                upload=lambda **kwargs: calls.append(kwargs),
            )
        return result, calls

    def test_dry_run_changes_nothing(self) -> None:
        result, calls = self.run_publish(False, ["model-00001-of-00002.safetensors"])
        self.assertFalse(result["executed"])
        self.assertEqual(calls, [])
        self.assertEqual(result["plan"]["prune"], ["model-00001-of-00002.safetensors"])

    def test_execute_passes_excludes_and_deletes(self) -> None:
        _, calls = self.run_publish(True, ["model-00001-of-00002.safetensors"])
        self.assertEqual(len(calls), 1)
        self.assertIn(".omc/*", calls[0]["ignore_patterns"])
        self.assertEqual(calls[0]["delete_patterns"], ["*.safetensors"])
        self.assertEqual(calls[0]["repo_type"], "model")

    def test_custom_delete_patterns_are_honoured(self) -> None:
        _, calls = self.run_publish(True, [], delete=["*.bin"])
        self.assertEqual(calls[0]["delete_patterns"], ["*.bin"])

    def test_a_new_repository_is_not_an_error(self) -> None:
        def missing(repo):
            raise RuntimeError("404")

        with tempfile.TemporaryDirectory() as tmp:
            path = make_artifact(Path(tmp) / "a", shards=("model-00001-of-00005.safetensors",))
            result = publisher.publish(args(path), list_remote=missing, upload=lambda **k: None)
        self.assertEqual(result["plan"]["prune"], [])

    def test_broken_artifact_never_reaches_upload(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = make_artifact(Path(tmp) / "a")
            (path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(publisher.PublishError):
                publisher.publish(args(path, execute=True), list_remote=lambda r: [],
                                  upload=lambda **k: calls.append(k))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
