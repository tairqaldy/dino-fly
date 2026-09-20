"""Pinned data sources with checksum enforcement.

URLs are pinned to commit SHAs (or DOI-backed file ids). SHA-256 values are recorded on the first download
(`flybrain download-data --record`), committed, and enforced on every later download or load.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from flybrain.config import data_dir

MANIFEST_PATH = Path(__file__).with_name("data_manifest.json")
_CHUNK = 1 << 20


@dataclass(frozen=True)
class DataFile:
    key: str
    path: str
    size: int
    sha256: str | None
    urls: tuple[str, ...]
    md5: str | None = None
    note: str = ""

    def local_path(self) -> Path:
        return data_dir() / self.path


def load_manifest() -> dict[str, DataFile]:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out: dict[str, DataFile] = {}
    for key, entry in raw["files"].items():
        out[key] = DataFile(
            key=key,
            path=entry["path"],
            size=int(entry["size"]),
            sha256=entry.get("sha256"),
            urls=tuple(entry["urls"]),
            md5=entry.get("md5"),
            note=entry.get("note", ""),
        )
    return out


def file_digests(path: Path) -> tuple[str, str, int]:
    """(sha256, md5, size) of a file, streamed."""
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


class ChecksumError(RuntimeError):
    pass


def verify(entry: DataFile, *, allow_unrecorded: bool = False) -> str:
    """Verify size, md5 (if published) and sha256 (if recorded). Returns the file's sha256."""
    path = entry.local_path()
    if not path.exists():
        raise FileNotFoundError(f"{entry.key}: {path} is missing — run `flybrain download-data`")
    sha, md5, size = file_digests(path)
    if size != entry.size:
        raise ChecksumError(f"{entry.key}: size {size} != expected {entry.size}")
    if entry.md5 and md5 != entry.md5:
        raise ChecksumError(f"{entry.key}: md5 {md5} != published {entry.md5}")
    if entry.sha256 is None:
        if not allow_unrecorded:
            raise ChecksumError(f"{entry.key}: no sha256 recorded in the manifest (run with --record once)")
    elif sha != entry.sha256:
        raise ChecksumError(f"{entry.key}: sha256 {sha} != recorded {entry.sha256}")
    return sha


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dino-fly/flybrain download-data"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=_CHUNK)
    tmp.replace(dest)


def record_sha256(key: str, sha256: str) -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["files"][key]["sha256"] = sha256
    MANIFEST_PATH.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def ensure(key: str, *, record: bool = False, log=print) -> Path:
    """Make sure one manifest file exists locally and passes verification; download it if needed."""
    entry = load_manifest()[key]
    path = entry.local_path()
    if not path.exists():
        last_error: Exception | None = None
        for url in entry.urls:
            try:
                log(f"[download] {key} <- {url}")
                _download(url, path)
                break
            except OSError as exc:  # network errors: try the mirror
                last_error = exc
                log(f"[download] failed: {exc}")
        else:
            raise RuntimeError(f"{key}: all download URLs failed") from last_error
    sha = verify(entry, allow_unrecorded=record)
    if entry.sha256 is None and record:
        record_sha256(key, sha)
        log(f"[record]   {key} sha256={sha}")
    else:
        log(f"[ok]       {key} sha256={sha[:16]}…")
    return path


def ensure_all(keys: list[str] | None = None, *, record: bool = False, log=print) -> dict[str, Path]:
    manifest = load_manifest()
    selected = keys or list(manifest)
    unknown = [k for k in selected if k not in manifest]
    if unknown:
        raise KeyError(f"unknown manifest keys: {unknown}; known: {sorted(manifest)}")
    return {key: ensure(key, record=record, log=log) for key in selected}
