"""Release installation checks use tiny local archives and isolated clone roots."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from urllib.error import URLError
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_data as download


class DownloadDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.root = self.workspace / "clone"
        self.assets = self.workspace / "assets"
        self.assets.mkdir()
        self.manifest_path = self.workspace / "release.json"
        self.manifest = {
            "schema_version": 1, "release": "nba-data-test",
            "repository": "TommySini/cs230-nba-forecasting", "dataset_id": "frozen-dataset",
            "bundles": {},
        }
        self.processed = {"data/processed/features.parquet": b"frozen features",
                          "data/processed/manifest.json": b'{"dataset_id":"frozen-dataset"}\n',
                          "data/models/gradient_boosting/model.joblib": b"saved estimator",
                          "reports/tabpfn/dev/run.json": b'{"complete":true}\n'}
        self.raw = {"data/raw/espn/scoreboards/game.json": b'{"original":true}\n'}
        self.archive("processed", self.processed)
        self.archive("raw", self.raw)

    def archive(self, bundle, files, *, extras=(), append=False, name=None):
        name = name or f"nba-{bundle}.tar.gz"
        path = self.assets / name
        with tarfile.open(path, "w:gz") as archive:
            for filename, data in files.items():
                info = tarfile.TarInfo(filename)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            for info, data in extras:
                archive.addfile(info, io.BytesIO(data) if data is not None else None)
        regular_extras = [(info, data) for info, data in extras if info.isfile()]
        metadata = {"name": name,
                    "url": f"https://github.com/{self.manifest['repository']}/releases/download/{self.manifest['release']}/{name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                    "unpacked_bytes": sum(map(len, files.values())) + sum(info.size for info, _ in regular_extras),
                    "files": len(files) + len(regular_extras)}
        if append:
            self.manifest["bundles"][bundle]["archives"].append(metadata)
        else:
            self.manifest["bundles"][bundle] = {"description": bundle, "archives": [metadata]}
        self.save_manifest()
        return metadata

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def install(self, **kwargs):
        return download.install_release(self.manifest_path, self.root, assets_dir=self.assets, **kwargs)

    def assert_no_install(self):
        self.assertFalse((self.root / "data/processed").exists())
        self.assertFalse((self.root / "data/models").exists())
        self.assertFalse((self.root / "data/raw").exists())
        self.assertFalse((self.root / "reports").exists())
        self.assertEqual(list(self.root.rglob(".stage-*")), [])
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_local_processed_install_and_offline_idempotent_retry(self):
        result = self.install()
        self.assertEqual(result["files_installed"], len(self.processed))
        self.assertEqual(result["bundles"], ["processed"])
        self.assertEqual(result["dataset_id"], "frozen-dataset")
        self.assertEqual(result["bytes_installed"], sum(map(len, self.processed.values())))
        for name, data in self.processed.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        self.assertFalse((self.root / "data/raw").exists())
        before = {name: (self.root / name).stat().st_ino for name in self.processed}
        shutil.rmtree(self.assets)
        with patch.object(download, "urlopen", side_effect=AssertionError("Verified cache must be reused")):
            again = download.install_release(self.manifest_path, self.root)
        self.assertEqual(again["files_installed"], 0)
        self.assertEqual(again["files_already_present"], len(self.processed))
        self.assertEqual(before, {name: (self.root / name).stat().st_ino for name in self.processed})

    def test_include_raw_installs_optional_bundle(self):
        self.install()
        result = self.install(include_raw=True)
        self.assertEqual(result["files_installed"], len(self.raw))
        self.assertEqual(result["files_already_present"], len(self.processed))
        self.assertEqual(result["bundles"], ["processed", "raw"])
        for name, data in self.raw.items():
            self.assertEqual((self.root / name).read_bytes(), data)

    def test_cli_accepts_explicit_clone_root(self):
        process = subprocess.run([sys.executable, str(Path(download.__file__)), "--root", str(self.root),
                                  "--manifest", str(self.manifest_path), "--assets-dir", str(self.assets)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(process.stdout)["files_installed"], len(self.processed))

    def test_corrupt_bytes_or_checksum_fail_before_install(self):
        asset = self.assets / "nba-processed.tar.gz"
        original = asset.read_bytes()
        for corrupted in (original + b"extra", b"x" + original[1:]):
            with self.subTest(size=len(corrupted)):
                asset.write_bytes(corrupted)
                with self.assertRaisesRegex(download.DownloadError, "size.*mismatch"):
                    self.install()
                self.assert_no_install()

    def test_corrupt_optional_raw_cannot_partially_install_processed(self):
        (self.assets / "nba-raw.tar.gz").write_bytes(b"bad")
        with self.assertRaises(download.DownloadError):
            self.install(include_raw=True)
        self.assert_no_install()

    def test_incorrect_declared_file_count_or_unpacked_bytes(self):
        for field in ("files", "unpacked_bytes"):
            for delta in (-1, 1):
                with self.subTest(field=field, delta=delta):
                    metadata = self.manifest["bundles"]["processed"]["archives"][0]
                    metadata[field] += delta
                    self.save_manifest()
                    with self.assertRaisesRegex(download.DownloadError, "count or unpacked bytes"):
                        self.install()
                    self.assert_no_install()
                    metadata[field] -= delta

    def test_unsafe_member_paths_and_outside_bundle_scope(self):
        for name in ("../outside", "/absolute", "data/processed/../../outside", "./data/processed/file",
                     "data//processed/file", "data/processed/a\\b", "C:/outside", "scripts/run.py",
                     "data/raw/original.json", "data/models/other.joblib", "data/processed"):
            with self.subTest(name=name):
                self.archive("processed", {name: b"unsafe"})
                with self.assertRaises(download.DownloadError):
                    self.install()
                self.assert_no_install()
        self.assertFalse((self.workspace / "outside").exists())

    def test_links_and_special_entries_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(kind=kind):
                info = tarfile.TarInfo("data/processed/link")
                info.type, info.linkname = kind, "../../outside"
                self.archive("processed", {}, extras=[(info, None)])
                with self.assertRaisesRegex(download.DownloadError, "links and special"):
                    self.install()
                self.assert_no_install()

    def test_duplicate_members_within_and_across_archives(self):
        info = tarfile.TarInfo("data/processed/features.parquet")
        info.size = 1
        self.archive("processed", self.processed, extras=[(info, b"x")])
        with self.assertRaisesRegex(download.DownloadError, "Duplicate archive member"):
            self.install()
        self.assert_no_install()
        self.archive("processed", self.processed)
        self.archive("processed", {"data/processed/features.parquet": b"x"}, append=True, name="processed-two.tar.gz")
        with self.assertRaisesRegex(download.DownloadError, "Duplicate archive member"):
            self.install()
        self.assert_no_install()

    def test_case_equivalent_members_are_rejected_on_every_filesystem(self):
        self.archive("processed", {"data/processed/A": b"one", "data/processed/a": b"two"})
        with self.assertRaisesRegex(download.DownloadError, "Duplicate archive member"):
            self.install()
        self.assert_no_install()

    def test_directory_headers_are_allowed_but_not_links(self):
        parent = tarfile.TarInfo("data/processed/")
        parent.type = tarfile.DIRTYPE
        name = "data/processed/manifest.json"
        self.archive("processed", {name: self.processed[name]}, extras=[(parent, None)])
        self.assertEqual(self.install()["files_installed"], 1)

    def test_missing_processed_manifest_fails_before_install(self):
        files = {name: data for name, data in self.processed.items() if name != "data/processed/manifest.json"}
        self.archive("processed", files)
        with self.assertRaisesRegex(download.DownloadError, "must include.*manifest.json"):
            self.install()
        self.assert_no_install()

    def test_release_dataset_id_must_match_staged_manifest(self):
        self.manifest["dataset_id"] = "different-dataset"
        self.save_manifest()
        with self.assertRaisesRegex(download.DownloadError, "dataset_id does not match"):
            self.install()
        self.assert_no_install()

    def test_invalid_processed_manifest_fails_before_install(self):
        for data in (b"malformed JSON", b"[]", b"{}", b"\xff"):
            with self.subTest(data=data):
                self.archive("processed", {**self.processed, "data/processed/manifest.json": data})
                with self.assertRaises(download.DownloadError):
                    self.install()
                self.assert_no_install()

    def test_conflict_preflight_leaves_existing_files_and_no_partial_install(self):
        existing = self.root / "reports/tabpfn/dev/run.json"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"my work")
        with self.assertRaisesRegex(download.DownloadError, "conflicts"):
            self.install()
        self.assertEqual(existing.read_bytes(), b"my work")
        self.assertFalse((self.root / "data/processed").exists())
        self.assertFalse((self.root / "data/models").exists())

    def test_raw_conflict_also_prevents_processed_publication(self):
        existing = self.root / next(iter(self.raw))
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"my raw")
        with self.assertRaisesRegex(download.DownloadError, "conflicts"):
            self.install(include_raw=True)
        self.assertFalse((self.root / "data/processed").exists())
        self.assertEqual(existing.read_bytes(), b"my raw")

    def test_symlink_destination_and_ancestors_are_rejected(self):
        outside = self.workspace / "outside"
        outside.mkdir()
        for name in ("reports", "data/processed", "data/processed/features.parquet", "data/.downloads"):
            with self.subTest(name=name):
                shutil.rmtree(self.root, ignore_errors=True)
                link = self.root / name
                link.parent.mkdir(parents=True)
                if name.endswith("parquet"):
                    target = outside / "original"
                    target.write_bytes(self.processed[name])
                else:
                    target = outside
                    (outside / "nba-data-test").mkdir(exist_ok=True)
                link.symlink_to(target)
                before = sorted(str(path.relative_to(outside)) for path in outside.rglob("*"))
                with self.assertRaisesRegex(download.DownloadError, "(symlink|parent|file type)"):
                    self.install()
                self.assertTrue(link.is_symlink())
                self.assertEqual(before, sorted(str(path.relative_to(outside)) for path in outside.rglob("*")))

    def test_explicit_shared_data_root_symlink_is_supported(self):
        canonical = self.workspace / "canonical-data"
        canonical.mkdir()
        self.root.mkdir()
        (self.root / "data").symlink_to(canonical, target_is_directory=True)
        result = self.install()
        self.assertEqual(result["files_installed"], len(self.processed))
        self.assertTrue((self.root / "data").is_symlink())
        self.assertEqual((canonical / "processed/features.parquet").read_bytes(), self.processed["data/processed/features.parquet"])
        self.assertEqual((self.root / "reports/tabpfn/dev/run.json").read_bytes(), self.processed["reports/tabpfn/dev/run.json"])

    def test_publication_failure_rolls_back_only_new_files(self):
        name = "data/processed/features.parquet"
        existing = self.root / name
        existing.parent.mkdir(parents=True)
        existing.write_bytes(self.processed[name])
        unrelated = existing.parent / "my-analysis.txt"
        unrelated.write_bytes(b"keep")
        publish, calls = download._publish_file, []

        def fail_after_publication(*args):
            publish(*args)
            calls.append(args[1])
            if len(calls) == 2:
                raise OSError("injected failure after publication")

        with patch.object(download, "_publish_file", side_effect=fail_after_publication):
            with self.assertRaisesRegex(OSError, "injected failure"):
                self.install()
        self.assertEqual(existing.read_bytes(), self.processed[name])
        self.assertEqual(unrelated.read_bytes(), b"keep")
        for filename in self.processed:
            if filename != name:
                self.assertFalse((self.root / filename).exists())
        self.assertFalse((self.root / "data/models").exists())
        self.assertEqual(list(self.root.rglob("*.part")), [])
        self.assertEqual(self.install()["files_installed"], len(self.processed) - 1)

    def test_concurrent_new_file_is_never_overwritten_or_removed(self):
        publish, target = download._publish_file, (self.root / "data/processed/manifest.json").resolve()

        def concurrent_writer(source, destination, *rest):
            if destination == target:
                destination.write_bytes(b"concurrent work")
            publish(source, destination, *rest)

        with patch.object(download, "_publish_file", side_effect=concurrent_writer):
            with self.assertRaises(FileExistsError):
                self.install()
        self.assertEqual(target.read_bytes(), b"concurrent work")
        self.assertFalse((self.root / "data/models").exists())
        self.assertFalse((self.root / "data/processed/features.parquet").exists())

    def test_manifest_rejects_remote_redirect_sources_and_unsafe_asset_names(self):
        metadata = self.manifest["bundles"]["processed"]["archives"][0]
        for field, value in (("url", "https://example.com/unpinned.tar.gz"), ("name", "../escape.tar.gz"),
                             ("files", True), ("sha256", "invalid")):
            with self.subTest(field=field):
                original = metadata[field]
                metadata[field] = value
                self.save_manifest()
                with self.assertRaises(download.DownloadError):
                    self.install()
                metadata[field] = original
                self.assertFalse(self.root.exists())

    def test_manifest_is_published_after_all_other_files(self):
        publish, observed = download._publish_file, []

        def observe(source, destination, *rest):
            observed.append(destination)
            publish(source, destination, *rest)

        with patch.object(download, "_publish_file", side_effect=observe):
            self.install(include_raw=True)
        self.assertEqual(observed[-1], (self.root / "data/processed/manifest.json").resolve())

    def test_transient_download_failure_retries_without_partial_install(self):
        payload = (self.assets / "nba-processed.tar.gz").read_bytes()
        with patch.object(download, "urlopen", side_effect=[URLError("connection reset"), io.BytesIO(payload)]) as request:
            with patch.object(download.time, "sleep"):
                result = download.install_release(self.manifest_path, self.root)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["files_installed"], len(self.processed))
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_corrupted_cache_is_replaced_from_verified_source(self):
        result = self.install()
        cached = Path(result["cache"]) / "nba-processed.tar.gz"
        cached.write_bytes(b"interrupted cache write")
        self.assertEqual(self.install()["files_installed"], 0)
        self.assertEqual(cached.read_bytes(), (self.assets / "nba-processed.tar.gz").read_bytes())

    def test_successful_download_is_pinned_and_cached(self):
        payload = (self.assets / "nba-processed.tar.gz").read_bytes()
        with patch.object(download, "urlopen", return_value=io.BytesIO(payload)) as request:
            result = download.install_release(self.manifest_path, self.root)
        self.assertEqual(result["files_installed"], len(self.processed))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0].full_url, self.manifest["bundles"]["processed"]["archives"][0]["url"])
        self.assertEqual((Path(result["cache"]) / "nba-processed.tar.gz").read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
