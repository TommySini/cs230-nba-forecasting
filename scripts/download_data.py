#!/usr/bin/env python3
"""Install the pinned dataset release; processed tables by default, raw with --include-raw.

Only the Python standard library is required. Archives are cached and verified,
then all files are staged and checked before any destination file is installed.
Existing files are never overwritten: matching bytes are reused, conflicts fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import unicodedata
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024
MODEL = "data/models/gradient_boosting/model.joblib"


class DownloadError(ValueError):
    """The release or destination failed a safety/integrity check."""


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            result.update(chunk)
    return result.hexdigest()


def _integer(value, name: str, minimum=0) -> int:
    if type(value) is not int or value < minimum:
        raise DownloadError(f"{name} must be an integer >= {minimum}")
    return value


def _read_manifest(path: Path, include_raw: bool):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise DownloadError("Expected dataset release schema_version 1")
    release, repository = manifest.get("release"), manifest.get("repository")
    slug = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    if not isinstance(release, str) or not re.fullmatch(slug, release):
        raise DownloadError("Invalid release name")
    if not isinstance(repository, str) or not re.fullmatch(f"{slug}/{slug}", repository):
        raise DownloadError("Invalid GitHub repository")
    if not isinstance(manifest.get("dataset_id"), str) or not manifest["dataset_id"]:
        raise DownloadError("Missing dataset_id")
    bundles = manifest.get("bundles")
    if not isinstance(bundles, dict):
        raise DownloadError("Missing bundles")
    selected = ["processed", "raw"] if include_raw else ["processed"]
    archives, seen = [], set()
    for bundle in selected:
        group = bundles.get(bundle)
        if not isinstance(group, dict) or not isinstance(group.get("archives"), list) or not group["archives"]:
            raise DownloadError(f"Missing {bundle} archives")
        for archive in group["archives"]:
            if not isinstance(archive, dict):
                raise DownloadError(f"Invalid archive in {bundle}")
            name = archive.get("name")
            if not isinstance(name, str) or not re.fullmatch(slug, name) or not name.endswith(".tar.gz"):
                raise DownloadError(f"Invalid archive name: {name!r}")
            if name.casefold() in seen:
                raise DownloadError(f"Duplicate archive name: {name}")
            seen.add(name.casefold())
            expected_url = f"https://github.com/{repository}/releases/download/{quote(release, safe='')}/{quote(name, safe='')}"
            if archive.get("url") != expected_url:
                raise DownloadError(f"Archive URL must identify this GitHub release: {name}")
            digest = archive.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise DownloadError(f"Invalid SHA256 for {name}")
            _integer(archive.get("bytes"), f"{name} bytes", 1)
            _integer(archive.get("unpacked_bytes"), f"{name} unpacked_bytes")
            _integer(archive.get("files"), f"{name} files")
            archives.append((bundle, archive))
    return manifest, selected, archives


def _regular(path: Path) -> bool:
    """lstat avoids accepting a symlink as an existing matching file."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(mode):
        raise DownloadError(f"Expected a regular file, found another file type: {path}")
    return True


def _directory(path: Path, created: list[Path] | None = None):
    """Create a directory one level at a time, rejecting symlink ancestors."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        _directory(path.parent, created)
        try:
            path.mkdir()
            if created is not None:
                created.append(path)
        except FileExistsError:
            pass
        mode = path.lstat().st_mode
    if not stat.S_ISDIR(mode):
        raise DownloadError(f"Expected a directory without symlinks: {path}")


def _parent_safe(path: Path, base: Path):
    """Check existing ancestors without creating destination directories."""
    current = base
    for part in path.relative_to(base).parts[:-1]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(mode):
            raise DownloadError(f"Unsafe destination parent: {current}")


def _data_root(root: Path) -> Path:
    data = root / "data"
    if data.is_symlink():
        # Worktrees intentionally share the canonical checkout's data directory.
        try:
            resolved = data.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise DownloadError(f"Invalid data directory link: {data}") from exc
        if not resolved.is_dir():
            raise DownloadError(f"Data directory link is not a directory: {data}")
        return resolved
    _directory(data)
    return data


def _cached_asset(archive: dict, cache: Path, assets_dir: Path | None) -> Path:
    destination = cache / archive["name"]
    if _regular(destination):
        if destination.stat().st_size == archive["bytes"] and _digest(destination) == archive["sha256"]:
            return destination
        # This path is the downloader's cache, never an installed user file.
        destination.unlink()
    attempts = 1 if assets_dir is not None else 3
    for attempt in range(attempts):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, prefix=".download-", suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                if assets_dir is not None:
                    source = (assets_dir / archive["name"]).open("rb")
                else:
                    request = Request(archive["url"], headers={"User-Agent": "cs230-nba-dataset/1", "Accept-Encoding": "identity"})
                    source = urlopen(request, timeout=60)
                total, digest = 0, hashlib.sha256()
                with source:
                    for chunk in iter(lambda: source.read(CHUNK), b""):
                        total += len(chunk)
                        if total > archive["bytes"]:
                            raise DownloadError(f"Archive size mismatch: {archive['name']}")
                        digest.update(chunk)
                        output.write(chunk)
                if total != archive["bytes"] or digest.hexdigest() != archive["sha256"]:
                    raise DownloadError(f"Archive size or SHA256 mismatch: {archive['name']}")
                output.flush()
                os.fsync(output.fileno())
            _regular(destination)  # Reject a symlink introduced while downloading.
            os.replace(temporary, destination)
            return destination
        except (URLError, TimeoutError, ConnectionError):
            if attempt + 1 == attempts:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise AssertionError("Unreachable download state")


def _member_path(member: tarfile.TarInfo, bundle: str) -> str:
    name = member.name.rstrip("/") if member.isdir() else member.name
    parts = name.split("/")
    if not name or "\\" in name or ":" in name or "\x00" in name or any(part in ("", ".", "..") for part in parts):
        raise DownloadError(f"Unsafe archive path: {member.name!r}")
    if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE) or member.issparse():
        raise DownloadError(f"Archive links and special entries are forbidden: {name}")
    prefixes = ("data/processed/", "reports/") if bundle == "processed" else ("data/raw/",)
    allowed = any(name.startswith(prefix) for prefix in prefixes)
    if bundle == "processed" and name == MODEL:
        allowed = True
    if member.isdir():
        parents = {"data", "data/processed", "data/models", "data/models/gradient_boosting", "reports"} if bundle == "processed" else {"data", "data/raw"}
        allowed = allowed or name in parents
        if name == MODEL:
            allowed = False
        if member.size != 0:
            raise DownloadError(f"Directory entry has a payload: {name}")
    if not allowed:
        raise DownloadError(f"Path outside {bundle} bundle scope: {name}")
    return name


def _stage_archive(path: Path, bundle: str, archive: dict, stage: Path, seen: set, files: dict):
    count, total = 0, 0
    with tarfile.open(path, mode="r:gz") as source:
        for member in source:
            name = _member_path(member, bundle)
            # APFS may compare case and Unicode spellings as the same path.
            collision_key = unicodedata.normalize("NFD", name).casefold()
            if collision_key in seen:
                raise DownloadError(f"Duplicate archive member: {name}")
            seen.add(collision_key)
            destination = stage / name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            count += 1
            total += member.size
            if member.size < 0 or count > archive["files"] or total > archive["unpacked_bytes"]:
                raise DownloadError(f"Archive exceeds declared file count or unpacked bytes: {archive['name']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest, copied = hashlib.sha256(), 0
            with source.extractfile(member) as contents, destination.open("xb") as output:
                for chunk in iter(lambda: contents.read(CHUNK), b""):
                    copied += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
            if copied != member.size:
                raise DownloadError(f"Truncated archive member: {name}")
            files[name] = {"sha256": digest.hexdigest(), "bytes": copied}
    if count != archive["files"] or total != archive["unpacked_bytes"]:
        raise DownloadError(f"Archive file count or unpacked bytes mismatch: {archive['name']}")


def _destination(name: str, root: Path, data: Path):
    base = data if name.startswith("data/") else root
    relative = name.removeprefix("data/") if base == data else name
    path = base / relative
    _parent_safe(path, base)
    return path, base


def _publish_file(source: Path, destination: Path, created: list, digest: str):
    """Publish complete bytes without replacing a concurrent user's file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".dataset-", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as contents:
                shutil.copyfileobj(contents, output, CHUNK)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o644)
        identity = temporary.stat()
        os.link(temporary, destination)  # Atomic, and fails if destination exists.
        created.append((destination, identity, digest))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_release(manifest_path: str | Path, root: str | Path = ROOT, *,
                    include_raw: bool = False, assets_dir: str | Path | None = None) -> dict:
    """Verify and install a release without overwriting any existing file.

    A failed install rolls back files it created. Successful archive downloads
    remain in data/.downloads/<release> for retry and offline reuse. Interrupted
    downloads restart; only verified complete archives enter the reusable cache.
    """
    manifest, selected, archives = _read_manifest(Path(manifest_path), include_raw)
    root = Path(root).expanduser().resolve()
    _directory(root)
    data = _data_root(root)
    cache = data / ".downloads" / manifest["release"]
    _parent_safe(cache / "entry", data)
    _directory(cache)
    local = Path(assets_dir) if assets_dir is not None else None
    installed, directories = [], []
    existing, installed_bytes = 0, 0
    with tempfile.TemporaryDirectory(dir=cache, prefix=".stage-") as temporary:
        stage, seen, files = Path(temporary), set(), {}
        for bundle, archive in archives:
            print(f"[{bundle}] Fetching/verifying {archive['name']} ({archive['bytes']:,} bytes)", file=sys.stderr, flush=True)
            asset = _cached_asset(archive, cache, local)
            print(f"[{bundle}] Checking/extracting {archive['name']} ({archive['files']:,} files)", file=sys.stderr, flush=True)
            _stage_archive(asset, bundle, archive, stage, seen, files)
        dataset_manifest = stage / "data/processed/manifest.json"
        if "data/processed/manifest.json" not in files:
            raise DownloadError("Processed archives must include data/processed/manifest.json")
        try:
            dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise DownloadError("Invalid data/processed/manifest.json") from exc
        if not isinstance(dataset, dict) or dataset.get("dataset_id") != manifest["dataset_id"]:
            raise DownloadError("Dataset manifest dataset_id does not match the release")
        pending = []
        # Preflight the entire selection, including raw, before publishing anything.
        for name, info in sorted(files.items()):
            destination, base = _destination(name, root, data)
            if _regular(destination):
                if destination.stat().st_size != info["bytes"] or _digest(destination) != info["sha256"]:
                    raise DownloadError(f"Existing file conflicts with the release: {destination}")
                existing += 1
            else:
                pending.append((name, destination, base, info))
        try:
            # Readers must not see a new dataset manifest before its files exist.
            pending.sort(key=lambda item: (item[0] == "data/processed/manifest.json", item[0]))
            for name, destination, base, info in pending:
                _parent_safe(destination, base)
                _directory(destination.parent, directories)
                _publish_file(stage / name, destination, installed, info["sha256"])
                installed_bytes += info["bytes"]
        except BaseException:
            for path, identity, digest in reversed(installed):
                try:
                    current = path.lstat()
                    if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino) and _digest(path) == digest:
                        path.unlink()
                except FileNotFoundError:
                    pass
            for path in reversed(directories):
                try:
                    path.rmdir()
                except OSError:
                    pass  # Do not remove a concurrently created user's file.
            raise
    return {"release": manifest["release"], "dataset_id": manifest["dataset_id"],
            "bundles": selected, "files_installed": len(installed),
            "files_already_present": existing, "bytes_installed": installed_bytes,
            "cache": str(cache)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/dataset_release.json")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to install into")
    parser.add_argument("--include-raw", action="store_true", help="Also install original source archives")
    parser.add_argument("--assets-dir", type=Path, help="Read downloaded archives from this directory instead of GitHub")
    args = parser.parse_args(argv)
    try:
        result = install_release(args.manifest, args.root, include_raw=args.include_raw, assets_dir=args.assets_dir)
    except (ValueError, OSError, tarfile.TarError) as exc:
        parser.exit(1, f"Dataset install failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
