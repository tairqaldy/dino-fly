"""The production engine must reproduce an independent dense reference simulator spike-for-spike."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import devices, spikes_to_dense
from reference_lif import simulate_reference

from flybrain import rng as crng
from flybrain.connectome import Connectome
from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive, ScheduledDrive

N_STEPS = 2500


def stim_neurons(synth: Connectome) -> np.ndarray:
    g = synth.meta["groups"]
    return np.array(g["SYN_GRN"] + g["LPLC2"] + g["LC4"], dtype=np.int64)


def test_matches_reference_with_scheduled_kicks(synth: Connectome):
    stim = stim_neurons(synth)
    rng = np.random.default_rng(11)
    table = rng.random((N_STEPS, len(stim))) < 0.012  # ~120 Hz
    ref = simulate_reference(synth, N_STEPS, stim_idx=stim, kick_table=table)

    net = LIFNetwork(synth, batch_size=1, dtype=torch.float64, chunk_steps=7)
    net.set_drive(ScheduledDrive(stim, table[:, :, None]))
    res = net.run(N_STEPS, record="spikes")
    got = spikes_to_dense(res.spikes, N_STEPS, synth.n)

    # the test is only meaningful if the network is really active, including the planted outputs
    g = synth.meta["groups"]
    assert ref.sum() > 3000
    assert ref[:, g["SYN_MN9"]].sum() > 5 and ref[:, g["DNp01"]].sum() > 5
    assert (ref.sum(axis=0) > 0).sum() > len(stim) + 20  # activity spreads beyond the driven neurons
    assert np.array_equal(got, ref)
    assert np.array_equal(res.counts[:, 0], ref.sum(axis=0))


@pytest.mark.parametrize("device", devices())
def test_matches_reference_with_poisson_drive(synth: Connectome, device: str):
    stim = stim_neurons(synth)
    p = LIFParams()
    rates = np.full(len(stim), 110.0)
    seeds = np.array([5, 2**40 + 9], dtype=np.int64)

    net = LIFNetwork(synth, batch_size=2, dtype=torch.float64, device=device, chunk_steps=18)
    drive = PoissonDrive(stim, batch_size=2, dt_ms=p.dt_ms, device=device)
    drive.set_rates(rates)
    drive.set_seeds(seeds)
    net.set_drive(drive)
    res = net.run(N_STEPS, record="spikes")

    p_u32 = np.full(len(stim), crng.rate_to_p_u32(110.0, p.dt_ms), dtype=np.int64)
    for col, seed in enumerate(seeds):
        ref = simulate_reference(synth, N_STEPS, stim_idx=stim, p_u32=p_u32, seed=int(seed))
        assert ref.sum() > 3000
        assert np.array_equal(spikes_to_dense(res.spikes, N_STEPS, synth.n, column=col), ref)


def test_matches_reference_with_ablation_and_refractory_targets(synth: Connectome):
    stim = stim_neurons(synth)
    gf = np.array(synth.meta["groups"]["DNp01"])
    table = np.random.default_rng(3).random((1500, len(stim))) < 0.02
    ref = simulate_reference(
        synth, 1500, stim_idx=stim, kick_table=table, ablate=gf, nonrefractory_targets=False
    )
    assert ref[:, gf].sum() == 0

    net = LIFNetwork(synth, batch_size=1, dtype=torch.float64)
    net.set_drive(ScheduledDrive(stim, table[:, :, None]), nonrefractory_targets=False)
    net.ablate(gf)
    res = net.run(1500, record="spikes")
    assert np.array_equal(spikes_to_dense(res.spikes, 1500, synth.n), ref)
