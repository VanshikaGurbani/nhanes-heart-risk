"""Stage 1: download the public NHANES SAS transport (.XPT) files, with checksums.

Raw files are never modified. Each file's SHA-256 is recorded in manifest/nhanes_sha256.json
the first time it is downloaded. Later downloads and later runs verify against that manifest,
so a silently revised CDC file is caught instead of absorbed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from nhanes.config import CYCLES, MANIFEST_PATH, RAW_DIR, TABLES, file_stem, file_url


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict[str, dict], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download(url: str, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def fetch_all(raw_dir: Path = RAW_DIR, manifest_path: Path = MANIFEST_PATH) -> dict[str, dict]:
    """Download every needed file that is missing, then verify all against the manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    changed = False
    for cycle in CYCLES:
        for table in TABLES:
            name = f"{file_stem(table, cycle)}.xpt"
            dest = raw_dir / name
            if not dest.exists():
                _download(file_url(table, cycle), dest)
            digest = sha256_of(dest)
            entry = manifest.get(name)
            if entry is None:
                manifest[name] = {
                    "sha256": digest,
                    "bytes": dest.stat().st_size,
                    "url": file_url(table, cycle),
                }
                changed = True
            elif entry["sha256"] != digest:
                raise ChecksumError(
                    f"{name}: sha256 {digest[:12]} does not match manifest {entry['sha256'][:12]}. "
                    "Delete the file to re-download, or update the manifest on purpose."
                )
    if changed:
        save_manifest(manifest, manifest_path)
    return manifest
