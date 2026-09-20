"""Timing and refractory semantics of the engine, checked case by case against Brian2's rules."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import spikes_to_dense, tiny_connectome

from flybrain.lif import LIFNetwork, LIFParams, ScheduledDrive

P = LIFParams()


def uval(net, neuron: int) -> float:
    return float(net.state_mv([neuron])[0][0, 0])


def gval(net, neuron: int) -> float:
    return float(net.state_mv([neuron])[1][0, 0])


def run_tiny(
    edges, n, kicks: dict[int, list[int]], n_steps, *, nonrefractory_targets=True, dtype=torch.float64, k=None
):
    """kicks: {neuron: [steps]} → (dense spikes [T, n], net)."""
    conn = tiny_connectome(edges, n)
    stim = np.array(sorted(kicks), dtype=np.int64)
    table = np.zeros((n_steps, len(stim), 1), dtype=bool)
    for j, neuron in enumerate(stim):
        table[kicks[int(neuron)], j, 0] = True
    net = LIFNetwork(conn, batch_size=1, dtype=dtype, chunk_steps=k)
    net.set_drive(ScheduledDrive(stim, table), nonrefractory_targets=nonrefractory_targets)
    res = net.run(n_steps, record="spikes")
    return spikes_to_dense(res.spikes, n_steps, n), net


def test_published_constants():
    assert P.ref_steps == 22 and P.delay_steps == 18
    assert P.kick_mv == pytest.approx(68.75)
    assert P.threshold_u == pytest.approx(7.0)


def test_rest_is_an_exact_fixed_point():
    conn = tiny_connectome([(0, 1, 5)], 3)
    net = LIFNetwork(conn, batch_size=2, dtype=torch.float32)
    res = net.run(500, record="counts")
    assert res.counts.sum() == 0
    assert torch.count_nonzero(net.u) == 0 and torch.count_nonzero(net.g) == 0
    assert net.n_active == 0  # nothing was ever touched


def test_kick_at_n_spikes_at_n_plus_1():
    spikes, _ = run_tiny([(0, 1, 1)], 2, {0: [10]}, 40)
    assert np.flatnonzero(spikes[:, 0]).tolist() == [11]


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 2e-6)])
def test_psp_matches_closed_form(dtype, tol):
    """Pre spike at step 1 → delivered at step 19 → u(19+j) = -(g0/3)(e^{-j dt/5} - e^{-j dt/20}) (u uses the OLD g)."""
    count = 10
    g0 = count * P.w_syn_mv
    for j in (0, 1, 7, 92, 400):
        _, net = run_tiny([(0, 1, count)], 2, {0: [0]}, 20 + j, dtype=dtype)
        expected = -(g0 / 3.0) * (math.exp(-j * 0.1 / 5.0) - math.exp(-j * 0.1 / 20.0))
        assert uval(net, 1) == pytest.approx(expected, abs=tol)
        assert gval(net, 1) == pytest.approx(g0 * math.exp(-j * 0.1 / 5.0), abs=tol * 10)
    # peak of the PSP is ~0.157 * g0 at ~9.2 ms
    _, net = run_tiny([(0, 1, count)], 2, {0: [0]}, 20 + 92, dtype=torch.float64)
    assert uval(net, 1) == pytest.approx(0.1575 * g0, rel=2e-3)


def test_earliest_postsynaptic_spike_is_m_plus_19():
    spikes, _ = run_tiny([(0, 1, 6000)], 2, {0: [4]}, 60)  # pre spikes at m = 5
    assert np.flatnonzero(spikes[:, 0]).tolist() == [5]
    assert np.flatnonzero(spikes[:, 1])[0] == 5 + 19


def test_refractory_is_22_steps_and_kicks_inside_it_are_dropped():
    # ref = 22 kept for the kicked neuron: spike at m = 11; kicks at m+1 … m+21 must vanish, m+22 is accepted.
    m = 11
    spikes, net = run_tiny([(0, 1, 1)], 2, {0: [10, *range(m + 1, m + 22)]}, 80, nonrefractory_targets=False)
    assert np.flatnonzero(spikes[:, 0]).tolist() == [m]
    assert uval(net, 0) == 0.0

    spikes, _ = run_tiny([(0, 1, 1)], 2, {0: [10, m + 21, m + 22]}, 80, nonrefractory_targets=False)
    assert np.flatnonzero(spikes[:, 0]).tolist() == [m, m + 23]


def test_poisson_targets_have_no_refractory_period():
    # kicks at n, n+1, n+2 → spikes at n+1 and n+3 (the kick landing in the spike step is erased by the reset)
    spikes, _ = run_tiny([(0, 1, 1)], 2, {0: [10, 11, 12]}, 40)
    assert np.flatnonzero(spikes[:, 0]).tolist() == [11, 13]


def test_synaptic_input_during_refractory_is_dropped_not_banked():
    # neuron 1 spikes at m = 31 (kicked at 30, ref = 22). Neuron 0 spikes at 21 → its input lands at 39 = m+8: dropped.
    edges = [(0, 1, 50)]
    spikes, net = run_tiny(edges, 2, {0: [20], 1: [30]}, 60, nonrefractory_targets=False)
    assert np.flatnonzero(spikes[:, 1]).tolist() == [31]
    assert gval(net, 1) == 0.0 and uval(net, 1) == 0.0
    # control: same input outside the refractory window is integrated
    spikes, net = run_tiny(edges, 2, {0: [20]}, 60, nonrefractory_targets=False)
    assert gval(net, 1) > 0.0


def test_autapse_input_arrives_inside_refractory_and_is_dropped():
    # spike at m → own input at m+18 < m+22 → dropped, so an autapse can never re-excite its neuron by itself
    spikes, net = run_tiny([(0, 0, 6000)], 1, {0: [5]}, 200, nonrefractory_targets=False)
    assert np.flatnonzero(spikes[:, 0]).tolist() == [6]
    assert gval(net, 0) == 0.0


def test_duplicate_edges_are_summed():
    a, _ = run_tiny([(0, 1, 3000), (0, 1, 3000)], 2, {0: [4]}, 60)
    b, _ = run_tiny([(0, 1, 6000)], 2, {0: [4]}, 60)
    assert np.array_equal(a, b)


def test_inhibition_cancels_excitation_exactly():
    spikes, net = run_tiny([(0, 2, 6000), (1, 2, -6000)], 3, {0: [4], 1: [4]}, 80)
    assert not spikes[:, 2].any()
    assert gval(net, 2) == 0.0  # integer accumulation: +6000 - 6000 == 0 exactly


def test_ablated_neuron_cannot_spike_but_still_integrates():
    conn = tiny_connectome([(0, 1, 6000)], 2)
    table = np.zeros((60, 1, 1), dtype=bool)
    table[4, 0, 0] = True
    net = LIFNetwork(conn, batch_size=1, dtype=torch.float64)
    net.set_drive(ScheduledDrive(np.array([0]), table))
    net.ablate(np.array([1]))
    res = net.run(60, record="counts")
    assert res.counts[1, 0] == 0 and res.counts[0, 0] == 1
    assert uval(net, 1) > P.threshold_u
    net.ablate(None)
    net.reset()
    assert net.run(60, record="counts").counts[1, 0] == 1


def test_watch_reports_counts_and_first_spike_step():
    conn = tiny_connectome([(0, 1, 6000)], 2)
    table = np.zeros((100, 1, 2), dtype=bool)
    table[4, 0, 0] = True  # column 0 only
    net = LIFNetwork(conn, batch_size=2, dtype=torch.float64, chunk_steps=10)
    net.set_drive(ScheduledDrive(np.array([0]), table))
    res = net.run(100, record=None, watch=np.array([1, 0]))
    assert res.watch_counts.tolist() == [[1, 0], [1, 0]]
    assert res.watch_first_step.tolist() == [[24, -1], [5, -1]]


def test_engine_rejects_bad_chunk_and_unsupported_reset_potential():
    conn = tiny_connectome([(0, 1, 1)], 2)
    with pytest.raises(ValueError):
        LIFNetwork(conn, chunk_steps=19)
    with pytest.raises(NotImplementedError):
        LIFNetwork(conn, LIFParams(v_reset_mv=-60.0))
