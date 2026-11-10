#!/usr/bin/env python3
"""Package the frozen dataset as reproducible, checksum-pinned release assets.

This reads only the published manifest's inventory. It never recollects data,
changes the installed dataset, or uploads assets. Raw sources are optional for
users of the processed release and always extract into the grouped raw layout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024
RAW_PART_BYTES = 1_500_000_000
ASSET_LIMIT = 2 * 1024**3
MODEL = "data/models/gradient_boosting/model.joblib"
ARCHIVE_CDX_SOURCE = "data/research/archive_coverage/data/cdx_gamenotes_all.json"
ARCHIVE_CDX_TARGET = "data/raw/notes/archive_cdx/cdx_gamenotes_all.json"


class PackageError(ValueError):
    """An input or destination did not meet the release contract."""


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise PackageError(f"Invalid repository path: {value!r}")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or str(path) != value or any(part in (".", "..") or part.startswith(".") for part in path.parts):
        raise PackageError(f"Unsafe repository path: {value!r}")
    return path


def _path(root, relative, *, regular=False):
    """Allow the project's data-directory link, but no links below it."""
    parts = _relative(relative).parts
    current = root
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if regular:
                raise PackageError(f"Missing source file: {relative}") from None
            continue
        if stat.S_ISLNK(mode):
            if index == 0 and part == "data" and current.is_dir():
                continue
            raise PackageError(f"Symlink is not a release file or directory: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise PackageError(f"Non-directory ancestor: {relative}")
    if regular and not stat.S_ISREG(current.lstat().st_mode):
        raise PackageError(f"Expected a regular source file: {relative}")
    return current


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PackageError(f"Invalid SHA-256 for {name}")
    return value


@dataclass(frozen=True)
class Entry:
    name: str
    size: int
    sha256: str
    source: Path | None = None
    content: bytes | None = None


def _source(root, name, digest, *, local=None, size=None):
    _relative(name)
    path = _path(root, local or name, regular=True)
    observed = path.stat().st_size
    if size is not None and (type(size) is not int or size < 0 or observed != size):
        raise PackageError(f"Source size mismatch: {local or name}")
    return Entry(name, observed, _sha(digest, name), source=path)


def _memory(name, content):
    return Entry(name, len(content), hashlib.sha256(content).hexdigest(), content=content)


def _inventories(root, release):
    manifest_path = _path(root, "data/processed/manifest.json", regular=True)
    source_manifest = manifest_path.read_bytes()
    manifest = json.loads(source_manifest)
    dataset_id = _sha(manifest.get("dataset_id"), "dataset_id")
    if manifest.get("publication", {}).get("status") != "complete":
        raise PackageError("Package only a completely published dataset")
    layout = manifest["publication"].get("raw_layout")
    if layout not in ("original", "grouped"):
        raise PackageError("Unknown published raw layout")
    tables, artifacts = manifest.get("tables"), manifest.get("artifacts")
    external = manifest.get("external_artifacts", {})
    if not isinstance(tables, dict) or not tables or not isinstance(artifacts, dict) or not isinstance(external, dict):
        raise PackageError("Missing table or artifact inventory")
    inventory_path = _path(root, "data/processed/raw_migration.json", regular=True)
    source_inventory = inventory_path.read_bytes()
    if hashlib.sha256(source_inventory).hexdigest() != artifacts.get("raw_migration.json"):
        raise PackageError("Source checksum mismatch: data/processed/raw_migration.json")
    inventory = json.loads(source_inventory)
    records = inventory.get("files")
    if not isinstance(records, list) or not records:
        raise PackageError("Raw inventory must contain regular files")

    portable = deepcopy(manifest)
    for key in ("remove_candidates", "cleanup_coverage", "install_files"):
        portable.pop(key, None)
    portable["publication"]["raw_layout"] = "grouped"
    portable["distribution"] = {"release": release,
                                "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
                                "raw_bundle_optional": True}
    portable["cleanup"] = {"status": "not_applicable", "reason": "Fresh-clone release contains no legacy dataset copies."}
    portable_inventory = deepcopy(inventory)
    portable_inventory["status"] = "layout_map"
    portable_inventory["method"] = "Provenance layout map; optional raw release archives install exact source bytes at path."
    portable_inventory["raw_bundle_optional"] = True
    inventory_bytes = _json_bytes(portable_inventory)
    portable["artifacts"]["raw_migration.json"] = hashlib.sha256(inventory_bytes).hexdigest()
    processed = []
    for metadata in tables.values():
        filename = metadata["file"]
        if len(_relative(filename).parts) != 1 or not filename.endswith(".parquet"):
            raise PackageError(f"Unknown processed table path: {filename}")
        processed.append(_source(root, f"data/processed/{filename}", metadata["sha256"]))
    for filename, digest in artifacts.items():
        if len(_relative(filename).parts) != 1 or not filename.endswith(".json") or filename == "manifest.json":
            raise PackageError(f"Unknown processed artifact path: {filename}")
        if filename == "raw_migration.json":
            processed.append(_memory("data/processed/raw_migration.json", inventory_bytes))
        else:
            processed.append(_source(root, f"data/processed/{filename}", digest))
    for filename, digest in external.items():
        parts = _relative(filename).parts
        if filename != MODEL and (parts[0] != "reports" or len(parts) < 2 or
                                  PurePosixPath(filename).suffix not in (".json", ".csv", ".md", ".txt", ".pdf", ".png", ".svg", ".html")):
            raise PackageError(f"Unknown external artifact path: {filename}")
        processed.append(_source(root, filename, digest))
    processed.append(_memory("data/processed/manifest.json", _json_bytes(portable)))
    raw = []
    for record in records:
        name = record["path"]
        if _relative(name).parts[:2] != ("data", "raw"):
            raise PackageError(f"Unknown raw inventory path: {name}")
        local = record["old_path"] if layout == "original" else name
        if _relative(local).parts[:2] != ("data", "raw") and (local, name) != (ARCHIVE_CDX_SOURCE, ARCHIVE_CDX_TARGET):
            raise PackageError(f"Unknown raw source path: {local}")
        raw.append(_source(root, name, record["sha256"], local=local, size=record["bytes"]))
    if type(inventory.get("bytes")) is not int or sum(entry.size for entry in raw) != inventory["bytes"]:
        raise PackageError("Raw inventory byte total does not match its files")
    seen = set()
    for entry in processed + raw:
        if entry.name.casefold() in seen:
            raise PackageError(f"Duplicate archive member: {entry.name}")
        seen.add(entry.name.casefold())
    guards = [(manifest_path, source_manifest), (inventory_path, source_inventory)]
    return dataset_id, sorted(processed, key=lambda entry: entry.name), sorted(raw, key=lambda entry: entry.name), guards


class _CheckedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        content = self.stream.read(size)
        self.digest.update(content)
        self.size += len(content)
        return content


def _add_file(archive, entry):
    info = tarfile.TarInfo(entry.name)
    info.size, info.mode, info.mtime = entry.size, 0o644, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    if entry.source is None:
        archive.addfile(info, io.BytesIO(entry.content))
        return
    descriptor = os.open(entry.source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != entry.size:
            raise PackageError(f"Source size or type changed: {entry.name}")
        checked = _CheckedReader(stream)
        archive.addfile(info, checked)
        if checked.read(1) or checked.size != entry.size or checked.digest.hexdigest() != entry.sha256:
            raise PackageError(f"Source checksum mismatch: {entry.name}")
        after = os.fstat(stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise PackageError(f"Source changed while packaging: {entry.name}")


def _write_archive(path, entries):
    with path.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=3, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT,
                              bufsize=CHUNK, copybufsize=CHUNK) as archive:
                for entry in entries:
                    _add_file(archive, entry)
    size = path.stat().st_size
    if size >= ASSET_LIMIT:
        raise PackageError(f"Release asset must be smaller than 2 GiB: {path.name}")
    return {"sha256": _digest(path), "bytes": size,
            "unpacked_bytes": sum(entry.size for entry in entries), "files": len(entries)}


def _shards(entries, limit):
    if type(limit) is not int or not 0 < limit <= RAW_PART_BYTES:
        raise PackageError("Raw part size must be between 1 and 1,500,000,000 bytes")
    current, size = [], 0
    for entry in entries:
        if entry.size > limit:
            raise PackageError(f"Raw file exceeds the shard limit: {entry.name}")
        if current and size + entry.size > limit:
            yield current
            current, size = [], 0
        current.append(entry)
        size += entry.size
    if current:
        yield current


def _destination(root, output):
    candidate = Path(output)
    if not candidate.is_absolute():
        candidate = root / candidate
    # Absolute paths through either spelling of the permitted data link work.
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        try:
            relative = Path("data") / candidate.relative_to((root / "data").resolve())
        except ValueError:
            raise PackageError("Output must be inside data/releases") from None
    if relative.parts[:2] != ("data", "releases") or len(relative.parts) < 3:
        raise PackageError("Output must be a release directory inside data/releases")
    destination = _path(root, relative.as_posix())
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise PackageError("Output directory must be empty; existing packages are never overwritten")
    return destination


def package_dataset(root, output, *, manifest_path="config/dataset_release.json", release=None,
                    repository="TommySini/cs230-nba-forecasting", raw_part_bytes=RAW_PART_BYTES):
    """Build all assets, then atomically write the downloader manifest.

    On failure, only files created by this invocation are removed. Source
    manifests and installed source files are never modified.
    """
    root = Path(root).resolve()
    destination = _destination(root, output)
    requested_manifest = Path(manifest_path)
    if requested_manifest.is_absolute():
        try:
            requested_manifest = requested_manifest.relative_to(root)
        except ValueError:
            raise PackageError("Release manifest must be inside the repository") from None
    if requested_manifest.parts[0] != "config" or requested_manifest.suffix != ".json":
        raise PackageError("Release manifest must be a JSON file under config/")
    release_manifest = _path(root, requested_manifest.as_posix())
    slug = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    if not isinstance(repository, str) or not re.fullmatch(f"{slug}/{slug}", repository):
        raise PackageError("Invalid GitHub repository")
    if release is None:
        source = json.loads(_path(root, "data/processed/manifest.json", regular=True).read_bytes())
        release = "nba-data-" + _sha(source.get("dataset_id"), "dataset_id")[:12]
    if not isinstance(release, str) or not re.fullmatch(slug, release):
        raise PackageError("Invalid release name")
    dataset_id, processed, raw, guards = _inventories(root, release)
    groups = [("processed", "nba-processed.tar.gz", processed)]
    groups.extend(("raw", f"nba-raw-{index:03d}.tar.gz", rows)
                  for index, rows in enumerate(_shards(raw, raw_part_bytes), 1))
    result = {"schema_version": 1, "release": release, "repository": repository, "dataset_id": dataset_id,
              "bundles": {"processed": {"description": "Frozen shared Parquet tables, provenance, saved boosting model and run records.", "archives": []},
                          "raw": {"description": "Optional original source bodies and metadata, in canonical grouped paths; required for offline source replay.", "archives": []}}}
    destination.mkdir(parents=True, exist_ok=True)
    created = []
    pending = []
    try:
        for bundle, name, entries in groups:
            temporary = destination / (name + ".partial")
            final = destination / name
            if final.exists() or temporary.exists():
                raise PackageError(f"Refusing to overwrite release asset: {name}")
            created.append(temporary)
            metadata = _write_archive(temporary, entries)
            metadata.update(name=name, url=f"https://github.com/{repository}/releases/download/{quote(release, safe='')}/{name}")
            result["bundles"][bundle]["archives"].append(metadata)
            pending.append((temporary, final))
            print(f"Verified {name}: {metadata['files']:,} files, {metadata['bytes']:,} compressed bytes", flush=True)
        for path, original in guards:
            if path.read_bytes() != original:
                raise PackageError(f"Source manifest changed while packaging: {path.name}")
        for temporary, final in pending:
            # link() publishes without the overwrite semantics of rename().
            os.link(temporary, final)
            created.append(final)
            temporary.unlink()
        release_manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = release_manifest.with_name(release_manifest.name + ".partial")
        if temporary_manifest.exists():
            raise PackageError(f"Manifest staging file already exists: {temporary_manifest}")
        created.append(temporary_manifest)
        with temporary_manifest.open("xb") as stream:
            stream.write(_json_bytes(result))
        temporary_manifest.replace(release_manifest)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True, help="Empty directory under data/releases")
    parser.add_argument("--manifest", type=Path, default=Path("config/dataset_release.json"))
    parser.add_argument("--release", help="Default: nba-data-<dataset_id[:12]>")
    parser.add_argument("--repository", default="TommySini/cs230-nba-forecasting")
    args = parser.parse_args()
    result = package_dataset(args.root, args.output, manifest_path=args.manifest,
                             release=args.release, repository=args.repository)
    print(f"Packaged {result['release']}; download manifest: {args.manifest}")


if __name__ == "__main__":
    main()
