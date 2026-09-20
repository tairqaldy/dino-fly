from __future__ import annotations

import numpy as np
import pytest
import torch

from flybrain.connectome import Connectome, synthetic


@pytest.fixture(scope="session")
def synth() -> Connectome:
    return synthetic(n=1000, seed=0)


def tiny_connectome(edges: list[tuple[int, int, int]], n: int) -> Connectome:
    """Hand-built connectome from (pre, post, signed_count) triples."""
    edges = sorted(edges, key=lambda e: e[0])
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 3)
    return Connectome(
        name="tiny",
        root_ids=720_575_940_600_000_000 + np.arange(n, dtype=np.int64),
        pre=arr[:, 0].astype(np.int32),
        post=arr[:, 1].astype(np.int32),
        weight=arr[:, 2].astype(np.int32),
    )


def spikes_to_dense(spikes: np.ndarray, n_steps: int, n: int, column: int = 0) -> np.ndarray:
    dense = np.zeros((n_steps, n), dtype=bool)
    sel = spikes[spikes[:, 2] == column]
    dense[sel[:, 0], sel[:, 1]] = True
    return dense


def devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
