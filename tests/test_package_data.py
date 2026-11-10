"""Release packages are exact, portable and reproducible without touching sources."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import package_data as package


def digest(content):
    return hashlib.sha256(content).hexdigest()


class PackageDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.root = self.workspace / "repository"
        self.root.mkdir()
        self.processed = self.root / "data/processed"
        self.processed.mkdir(parents=True)
        self.original = {
            "data/raw/scoreboards/game.json": b'{"source": true}\n',
            "data/raw/scoreboards/game.meta.json": b'{"status": 200, "path": "data/raw/scoreboards/game.json"}\n',
        }
        self.inventory = {"files": [], "status": "pending", "method": "filesystem rename"}
        for name, content in self.original.items():
            self.write(name, content)
            self.inventory["files"].append({"old_path": name, "path": name.replace("raw/scoreboards", "raw/espn/scoreboards"),
                                             "bytes": len(content), "sha256": digest(content)})
        self.inventory["bytes"] = sum(map(len, self.original.values()))
        self.write("data/processed/features.parquet", b"frozen fixture table")
        self.write("data/processed/feature_dictionary.json", b'{"features": ["example"]}\n')
        self.write(package.MODEL, b"frozen saved estimator")
        self.write("reports/tabpfn/dev/run.json", b'{"stage": "dev"}\n')
        self.manifest = {"schema_version": 1, "dataset_id": "a" * 64,
                         "tables": {"features": {"file": "features.parquet", "sha256": digest(b"frozen fixture table"), "rows": 1}},
                         "artifacts": {"feature_dictionary.json": digest(b'{"features": ["example"]}\n')},
                         "external_artifacts": {package.MODEL: digest(b"frozen saved estimator"),
                                                "reports/tabpfn/dev/run.json": digest(b'{"stage": "dev"}\n')},
                         "source_sha256": {"legacy.jsonl": "b" * 64}, "legacy_records": {"inputs": 1},
                         "publication": {"status": "complete", "raw_layout": "original"},
                         "remove_candidates": ["data/processed/old"], "cleanup_coverage": {"old": "new"},
                         "install_files": {"exports/model.joblib": package.MODEL}}
        self.save()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def save(self):
        content = (json.dumps(self.inventory, indent=2) + "\n").encode()
        self.write("data/processed/raw_migration.json", content)
        self.manifest["artifacts"]["raw_migration.json"] = digest(content)
        self.write("data/processed/manifest.json", (json.dumps(self.manifest, indent=2) + "\n").encode())

    def build(self, name="release", **kwargs):
        return package.package_dataset(self.root, f"data/releases/{name}", **kwargs)

    def members(self, name, archive):
        with tarfile.open(self.root / "data/releases" / name / archive, "r:gz") as stream:
            return {info.name: stream.extractfile(info).read() for info in stream}

    def test_reproducible_archives_and_manifest_exactly_exclude_unlisted_files(self):
        for name in ("data/processed/gradient_boosting/inputs.jsonl", "data/raw/other/unlisted.json",
                     "data/models/tabpfn/adaptation/checkpoint.bin", "data/.credentials", "reports/unlisted.json"):
            self.write(name, b"must not be in release")
        first = self.build("first")
        first_manifest = (self.root / "config/dataset_release.json").read_bytes()
        second = self.build("second")
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, (self.root / "config/dataset_release.json").read_bytes())
        for bundle in first["bundles"].values():
            for asset in bundle["archives"]:
                a = self.root / "data/releases/first" / asset["name"]
                b = self.root / "data/releases/second" / asset["name"]
                self.assertEqual(a.read_bytes(), b.read_bytes())
                self.assertEqual(digest(a.read_bytes()), asset["sha256"])
                with tarfile.open(a) as archive:
                    infos = archive.getmembers()
                    self.assertEqual(asset["files"], len(infos))
                    self.assertEqual(asset["unpacked_bytes"], sum(info.size for info in infos))
                    self.assertTrue(all(info.isfile() and not info.issym() for info in infos))
                    self.assertTrue(all((info.uid, info.gid, info.mode, info.mtime, info.uname, info.gname) == (0, 0, 0o644, 0, "", "") for info in infos))
        self.assertEqual(set(self.members("first", "nba-processed.tar.gz")), {
            "data/processed/features.parquet", "data/processed/feature_dictionary.json",
            "data/processed/raw_migration.json", "data/processed/manifest.json", package.MODEL,
            "reports/tabpfn/dev/run.json"})

    def test_portable_copies_preserve_identity_and_original_source_bytes(self):
        before = {name: (self.processed / name).read_bytes() for name in ("manifest.json", "raw_migration.json")}
        self.build()
        members = self.members("release", "nba-processed.tar.gz")
        portable = json.loads(members["data/processed/manifest.json"])
        inventory_bytes = members["data/processed/raw_migration.json"]
        self.assertEqual(portable["dataset_id"], self.manifest["dataset_id"])
        for key in ("tables", "source_sha256", "legacy_records", "external_artifacts"):
            self.assertEqual(portable[key], self.manifest[key])
        self.assertEqual(portable["publication"]["raw_layout"], "grouped")
        self.assertEqual(portable["cleanup"]["status"], "not_applicable")
        self.assertEqual(portable["distribution"], {"release": "nba-data-aaaaaaaaaaaa",
                         "source_manifest_sha256": digest(before["manifest.json"]), "raw_bundle_optional": True})
        self.assertFalse({"remove_candidates", "cleanup_coverage", "install_files"} & portable.keys())
        self.assertEqual(portable["artifacts"]["raw_migration.json"], digest(inventory_bytes))
        self.assertEqual(json.loads(inventory_bytes)["status"], "layout_map")
        self.assertEqual(json.loads(inventory_bytes)["files"], self.inventory["files"])
        self.assertEqual(before, {name: (self.processed / name).read_bytes() for name in before})
        raw = self.members("release", "nba-raw-001.tar.gz")
        self.assertEqual(raw, {row["path"]: self.original[row["old_path"]] for row in self.inventory["files"]})

    def test_raw_shards_use_unpacked_size_and_have_unique_members(self):
        limit = max(row["bytes"] for row in self.inventory["files"])
        result = self.build(raw_part_bytes=limit)
        archives = result["bundles"]["raw"]["archives"]
        self.assertEqual(len(archives), 2)
        self.assertTrue(all(asset["unpacked_bytes"] <= limit for asset in archives))
        names = [name for asset in archives for name in self.members("release", asset["name"])]
        self.assertEqual(set(names), {row["path"] for row in self.inventory["files"]})
        self.assertEqual(len(set(names)), len(names))

    def test_processed_and_raw_hash_tamper_abort_without_manifest_or_assets(self):
        for target in ("data/processed/features.parquet", "data/raw/scoreboards/game.json"):
            with self.subTest(target=target):
                source = self.root / target
                original = source.read_bytes()
                source.write_bytes(b"X" * len(original))
                with self.assertRaisesRegex(package.PackageError, "checksum"):
                    self.build()
                self.assertFalse((self.root / "config/dataset_release.json").exists())
                self.assertEqual(list((self.root / "data/releases/release").iterdir()), [])
                source.write_bytes(original)

    def test_raw_inventory_hash_and_size_tampering_abort(self):
        inventory_path = self.processed / "raw_migration.json"
        inventory_path.write_bytes(inventory_path.read_bytes() + b" ")
        with self.assertRaisesRegex(package.PackageError, "checksum"):
            self.build()
        self.save()
        self.inventory["files"][0]["bytes"] += 1
        self.save()
        with self.assertRaisesRegex(package.PackageError, "size mismatch"):
            self.build()

    def test_existing_assets_and_manifest_survive_failed_run(self):
        result = self.build()
        pin = (self.root / "config/dataset_release.json").read_bytes()
        archive = self.root / "data/releases/release/nba-processed.tar.gz"
        before = archive.read_bytes()
        with self.assertRaisesRegex(package.PackageError, "empty"):
            self.build()
        self.assertEqual(before, archive.read_bytes())
        self.write("data/raw/scoreboards/game.json", b"tampered")
        with self.assertRaises(package.PackageError):
            self.build("failed")
        self.assertEqual(pin, (self.root / "config/dataset_release.json").read_bytes())
        self.assertEqual(result["release"], "nba-data-aaaaaaaaaaaa")

    def test_grouped_layout_reads_canonical_paths(self):
        for row in self.inventory["files"]:
            path = self.root / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            (self.root / row["old_path"]).rename(path)
        self.manifest["publication"]["raw_layout"] = "grouped"
        self.save()
        self.build()
        self.assertEqual(len(self.members("release", "nba-raw-001.tar.gz")), 2)

    def test_archive_cdx_legacy_research_source_has_one_exact_mapping(self):
        content = b'{"captures": []}\n'
        self.write(package.ARCHIVE_CDX_SOURCE, content)
        self.inventory["files"].append({"old_path": package.ARCHIVE_CDX_SOURCE, "path": package.ARCHIVE_CDX_TARGET,
                                        "bytes": len(content), "sha256": digest(content)})
        self.inventory["bytes"] += len(content)
        self.save()
        self.build()
        self.assertEqual(self.members("release", "nba-raw-001.tar.gz")[package.ARCHIVE_CDX_TARGET], content)
        self.inventory["files"][-1]["path"] = "data/raw/notes/other.json"
        self.save()
        with self.assertRaisesRegex(package.PackageError, "Unknown raw source"):
            self.build("invalid")

    def test_data_symlink_allowed_but_source_and_output_symlinks_rejected(self):
        shared = self.workspace / "shared-data"
        (self.root / "data").rename(shared)
        (self.root / "data").symlink_to(shared, target_is_directory=True)
        self.build()
        outside = self.workspace / "outside"
        outside.mkdir()
        (shared / "releases/link").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(package.PackageError, "Symlink"):
            self.build("link")
        path = self.processed / "features.parquet"
        content = path.read_bytes()
        path.unlink()
        (outside / "table").write_bytes(content)
        path.symlink_to(outside / "table")
        with self.assertRaisesRegex(package.PackageError, "Symlink"):
            self.build("source-link")

    def test_unknown_path_escape_and_duplicate_members_rejected(self):
        original = deepcopy(self.manifest)
        self.manifest["external_artifacts"]["data/checkpoints/private.bin"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(package.PackageError, "Unknown external"):
            self.build()
        self.manifest = original
        self.manifest["tables"]["features"]["file"] = "../../outside.parquet"
        self.save()
        with self.assertRaisesRegex(package.PackageError, "Unsafe"):
            self.build()
        self.manifest["tables"]["features"]["file"] = "features.parquet"
        self.inventory["files"].append(dict(self.inventory["files"][0]))
        self.inventory["bytes"] += self.inventory["files"][0]["bytes"]
        self.save()
        with self.assertRaisesRegex(package.PackageError, "Duplicate"):
            self.build()
        with self.assertRaisesRegex(package.PackageError, "inside data/releases"):
            package.package_dataset(self.root, self.workspace / "wrong")

    def test_source_manifest_change_during_packaging_aborts(self):
        writer = package._write_archive
        def changing_writer(path, entries):
            result = writer(path, entries)
            with (self.processed / "manifest.json").open("ab") as stream:
                stream.write(b" ")
            return result
        with patch.object(package, "_write_archive", side_effect=changing_writer):
            with self.assertRaisesRegex(package.PackageError, "Source manifest changed"):
                self.build()
        self.assertEqual(list((self.root / "data/releases/release").iterdir()), [])

    def test_portable_release_installs_and_passes_shared_dataset_verify(self):
        # Use the actual migration fixture, preserving its split labels and lineage.
        sys.path.insert(0, str(Path(__file__).parent))
        from test_nba_dataset import fixture
        from nba_dataset import Dataset, publish_stage, stage_dataset
        from download_data import install_release
        repository = self.workspace / "real-schema"
        fixture(repository)
        raw = repository / "data/raw/scoreboards/source.json"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'{"original": true}\n')
        stage = repository / "stage"
        stage_dataset(repository, stage)
        publish_stage(repository, stage, relocate_raw=False)
        expected = Dataset(repository).verify()
        package.package_dataset(repository, "data/releases/release")
        clone = self.workspace / "fresh-clone"
        install_release(repository / "config/dataset_release.json", clone,
                        assets_dir=repository / "data/releases/release")
        self.assertEqual(Dataset(clone).verify(), expected)
        self.assertFalse((clone / "data/raw").exists())
        install_release(repository / "config/dataset_release.json", clone,
                        assets_dir=repository / "data/releases/release", include_raw=True)
        self.assertEqual((clone / "data/raw/espn/scoreboards/source.json").read_bytes(), raw.read_bytes())


if __name__ == "__main__":
    unittest.main()
