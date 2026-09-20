"""Paths and run metadata shared by the package, the CLI and the experiments."""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
from pathlib import Path


def repo_root() -> Path:
    """Repository root = first parent that contains `pnpm-workspace.yaml`."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("could not locate the dino-fly repository root (pnpm-workspace.yaml not found)")


def data_dir() -> Path:
    """Where downloaded data lives: $DINOFLY_DATA_DIR or <repo>/data (git-ignored)."""
    override = os.environ.get("DINOFLY_DATA_DIR", "").strip()
    path = Path(override) if override else repo_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    path = repo_root() / "brain" / "experiments" / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = repo_root() / "brain" / "experiments" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = repo_root() / "docs" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_is_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def git_has_tag(tag: str) -> bool:
    return tag in _git("tag", "--list", tag).splitlines()


def run_metadata() -> dict:
    """Provenance block stamped into every results JSON (volatile fields live only here)."""
    meta: dict = {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["cuda"] = torch.version.cuda
    except ImportError:  # pragma: no cover - torch is always installed via the cpu/gpu extra
        pass
    return meta
