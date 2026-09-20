"""Shared plumbing for experiments: results I/O, held-out ledger, plotting defaults."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.config import figures_dir, git_commit, git_has_tag, git_is_dirty, results_dir, run_metadata


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def result_path(name: str) -> Path:
    return results_dir() / f"{name}.json"


def write_result(name: str, payload: dict) -> Path:
    """Write results/<name>.json. Volatile provenance goes under "meta" only (the report renderer ignores it)."""
    out = {"experiment": name, **payload, "meta": run_metadata()}
    path = result_path(name)
    path.write_text(json.dumps(out, indent=2, cls=_Encoder) + "\n", encoding="utf-8")
    print(f"[result] wrote {path}")
    return path


def load_result(name: str) -> dict | None:
    path = result_path(name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def figure_path(name: str) -> Path:
    return figures_dir() / f"{name}.png"


def require_preregistration(
    tag: str, experiment: str, conditions: list[str], *, allow_dirty: bool = False
) -> None:
    """Gate for anything that touches held-out seeds: clean tree, prereg tag present, ledger entry appended."""
    if not git_has_tag(tag):
        raise SystemExit(
            f"[prereg] git tag '{tag}' is missing — freeze the protocol and tag it before evaluating"
        )
    if git_is_dirty() and not allow_dirty:
        raise SystemExit(
            "[prereg] working tree is dirty — commit first, so the ledger points at the exact code"
        )
    entry = {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "experiment": experiment,
        "git_commit": git_commit(),
        "prereg_tag": tag,
        "conditions": conditions,
    }
    ledger = results_dir() / "heldout_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"[prereg] ledger entry appended: {entry}")


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )
    return plt


# one palette for every figure in the project
C_FLY = "#1f77b4"  # our model / intact fly
C_REF = "#222222"  # published reference
C_ALT = "#d95f02"  # second series
C_CTRL = "#7f7f7f"  # controls
C_BAD = "#c0392b"
