"""Spike-for-spike agreement with real Brian2.

The golden file is produced by `experiments/make_brian2_golden.py --synthetic` (needs the optional brian2 extra):
real Brian2 simulates the published equations on the synthetic connectome (autapses, duplicate edges, strong
negative weights, refractory-free driven neurons that also receive synaptic input) under a deterministic kick
schedule. Network, schedule and Brian2's spikes are all stored in the file, so this test needs neither Brian2 nor
a stable NumPy random stream.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.make_brian2_golden import GOLDEN
from flybrain.connectome import Connectome
from flybrain.lif import LIFNetwork, ScheduledDrive


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.fail(f"{GOLDEN} is missing — it must be committed")
    return np.load(GOLDEN)


def engine_spikes(golden, **kwargs) -> np.ndarray:
    conn = Connectome("golden", golden["root_ids"], golden["pre"], golden["post"], golden["weight"])
    n_steps = int(golden["n_steps"])
    stim = golden["stim"]
    table = np.zeros((n_steps, len(stim), 1), dtype=bool)
    table[golden["kick_steps"], golden["kick_which"], 0] = True
    net = LIFNetwork(conn, batch_size=1, dtype=torch.float64, **kwargs)
    net.set_drive(ScheduledDrive(stim, table, device=kwargs.get("device", "cpu")))
    spikes = net.run(n_steps, record="spikes").spikes[:, :2]
    return spikes[np.lexsort((spikes[:, 1], spikes[:, 0]))]


def test_golden_file_is_a_meaningful_test(golden):
    spikes = golden["spikes"]
    assert len(spikes) > 5000
    assert len(np.unique(spikes[:, 1])) > 500  # activity spreads through most of the recurrent network
    stim = set(golden["stim"].tolist())
    assert np.sum([n not in stim for n in spikes[:, 1]]) > 2000  # most spikes are synaptically evoked
    assert np.sum(golden["pre"] == golden["post"]) > 0  # autapses present
    assert np.sum(golden["weight"] < 0) > 0  # inhibition present


@pytest.mark.parametrize("chunk", [1, 18])
def test_engine_matches_brian2_spike_for_spike(golden, chunk):
    got = engine_spikes(golden, chunk_steps=chunk)
    assert np.array_equal(got, golden["spikes"].astype(np.int64))


@pytest.mark.gpu
def test_engine_matches_brian2_on_gpu(golden):
    got = engine_spikes(golden, device="cuda", use_cuda_graph=True)
    assert np.array_equal(got, golden["spikes"].astype(np.int64))
