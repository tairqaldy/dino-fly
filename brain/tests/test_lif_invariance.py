"""Results must not depend on chunk size, batch size, column recycling, device or CUDA graphs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import spikes_to_dense

from flybrain.connectome import Connectome
from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

P = LIFParams()


def make(
    synth: Connectome,
    seeds,
    *,
    k=None,
    device="cpu",
    dtype=torch.float32,
    rate=120.0,
    graph=False,
    max_events=4_000_000,
    active_set=True,
):
    g = synth.meta["groups"]
    stim = np.array(g["SYN_GRN"] + g["LPLC2"] + g["LC4"], dtype=np.int64)
    b = len(seeds)
    net = LIFNetwork(
        synth,
        batch_size=b,
        dtype=dtype,
        device=device,
        chunk_steps=k,
        use_cuda_graph=graph,
        max_events_per_slice=max_events,
        active_set=active_set,
    )
    drive = PoissonDrive(stim, batch_size=b, dt_ms=P.dt_ms, device=device)
    drive.set_rates(np.full(len(stim), rate))
    drive.set_seeds(np.asarray(seeds, dtype=np.int64))
    net.set_drive(drive)
    return net, drive


def dense(res, n_steps, n, col=0):
    return spikes_to_dense(res.spikes, n_steps, n, column=col)


def test_chunk_size_does_not_matter(synth: Connectome):
    n_steps = 12 * 36 + 5  # many ring lengths, not a multiple of any chunk size → exercises short last chunks
    base = None
    for k in (1, 5, 10, 18):
        net, _ = make(synth, [7], k=k)
        got = dense(net.run(n_steps, record="spikes"), n_steps, synth.n)
        assert got.sum() > 100
        base = got if base is None else base
        assert np.array_equal(got, base), f"chunk size {k} changed the result"


def test_run_can_be_split_into_arbitrary_calls(synth: Connectome):
    net, _ = make(synth, [7], k=10)
    whole = dense(net.run(400, record="spikes"), 400, synth.n)
    net2, _ = make(synth, [7], k=10)
    parts = []
    for n_steps in (3, 97, 100, 1, 199):
        res = net2.run(n_steps, record="spikes")
        parts.append(res.spikes)
    got = spikes_to_dense(np.concatenate(parts), 400, synth.n)
    assert np.array_equal(got, whole)


def test_columns_are_independent_of_batch_size(synth: Connectome):
    seeds = [3, 4, 2**35 + 1]
    net, _ = make(synth, seeds, k=10)
    res = net.run(600, record="spikes")
    for col, seed in enumerate(seeds):
        solo, _ = make(synth, [seed], k=18)
        assert np.array_equal(
            dense(res, 600, synth.n, col), dense(solo.run(600, record="spikes"), 600, synth.n)
        )
    assert not np.array_equal(dense(res, 600, synth.n, 0), dense(res, 600, synth.n, 1))


def test_masked_reset_recycles_a_column_like_a_fresh_trial(synth: Connectome):
    net, drive = make(synth, [3, 4], k=10)
    net.run(250, record=None)
    net.reset(columns=np.array([1]))
    drive.set_seeds(np.array([99]), columns=np.array([1]))
    res = net.run(500, record="spikes")
    res.spikes[:, 0] -= 250

    fresh, _ = make(synth, [99], k=10)
    assert np.array_equal(dense(res, 500, synth.n, 1), dense(fresh.run(500, record="spikes"), 500, synth.n))
    # column 0 was not disturbed by resetting column 1
    cont, _ = make(synth, [3], k=10)
    cont.run(250, record=None)
    assert np.array_equal(
        dense(res, 500, synth.n, 0), spikes_shift(cont.run(500, record="spikes"), 250, 500, synth.n)
    )


def spikes_shift(res, shift, n_steps, n):
    res.spikes[:, 0] -= shift
    return dense(res, n_steps, n)


def test_event_slicing_does_not_change_results(synth: Connectome):
    a, _ = make(synth, [1, 2], k=18)
    b, _ = make(synth, [1, 2], k=18, max_events=50)  # force many tiny slices
    ra, rb = a.run(300, record="spikes"), b.run(300, record="spikes")
    assert ra.spikes.shape[0] > 200
    assert np.array_equal(ra.spikes, rb.spikes)


def test_counts_only_recording_matches_spike_recording(synth: Connectome):
    a, _ = make(synth, [1, 2], k=10)
    b, _ = make(synth, [1, 2], k=10)
    ra, rb = a.run(300, record="spikes"), b.run(300, record="counts")
    assert rb.spikes is None
    assert np.array_equal(ra.counts, rb.counts)
    assert ra.counts.sum() == ra.spikes.shape[0]
    assert np.allclose(rb.rates_hz(), rb.counts / 0.03)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gpu_matches_cpu_bit_for_bit(synth: Connectome, dtype):
    cpu, _ = make(synth, [1, 2, 3], k=10, dtype=dtype)
    gpu, _ = make(synth, [1, 2, 3], k=10, dtype=dtype, device="cuda")
    rc, rg = cpu.run(1500, record="spikes"), gpu.run(1500, record="spikes")
    assert rc.spikes.shape[0] > 1000
    assert np.array_equal(rc.spikes, rg.spikes)
    assert cpu.n_active == gpu.n_active
    assert torch.equal(cpu.u, gpu.u.cpu()) and torch.equal(cpu.g, gpu.g.cpu())


@pytest.mark.gpu
def test_cuda_graph_matches_eager_bit_for_bit(synth: Connectome):
    eager, _ = make(synth, [1, 2, 3], k=10, device="cuda")
    graph, _ = make(synth, [1, 2, 3], k=10, device="cuda", graph=True)
    re, rg = eager.run(1505, record="spikes"), graph.run(1505, record="spikes")
    assert re.spikes.shape[0] > 1000
    assert np.array_equal(re.spikes, rg.spikes)
    assert torch.equal(eager.u, graph.u) and torch.equal(eager.g, graph.g)


def test_active_set_is_exact(synth: Connectome):
    """Updating only neurons that ever received input gives the same spikes as updating all of them."""
    dense, _ = make(synth, [1, 2, 3], k=10, active_set=False)
    sparse, _ = make(synth, [1, 2, 3], k=10, active_set=True)
    assert dense.n_active == synth.n and sparse.n_active == 70  # only the driven neurons so far
    rd, rs = dense.run(1500, record="spikes"), sparse.run(1500, record="spikes")
    assert rd.spikes.shape[0] > 1000
    assert np.array_equal(rd.spikes, rs.spikes)
    assert 70 < sparse.n_active <= synth.n
    probe = np.arange(synth.n)
    for a, b in zip(dense.state_mv(probe), sparse.state_mv(probe), strict=True):
        assert np.array_equal(a, b)
    # a full reset shrinks the active set again and the next run is reproducible
    sparse.reset()
    assert sparse.n_active == 70
    sparse._drive.set_seeds(np.array([1, 2, 3]))
    again = sparse.run(1500, record="spikes").spikes
    again[:, 0] -= 1500  # spike steps are global
    assert np.array_equal(again, rs.spikes)


def test_active_set_with_small_buckets_and_watch(synth: Connectome, monkeypatch):
    import flybrain.lif as lif

    monkeypatch.setattr(lif, "SLOT_BUCKET", 16)  # force many capacity changes
    gf = np.array(synth.meta["groups"]["DNp01"])
    a, _ = make(synth, [5, 6], k=7, active_set=True)
    b, _ = make(synth, [5, 6], k=7, active_set=False)
    ra, rb = a.run(900, record="spikes", watch=gf), b.run(900, record="spikes", watch=gf)
    assert np.array_equal(ra.spikes, rb.spikes)
    assert np.array_equal(ra.watch_counts, rb.watch_counts) and ra.watch_counts.sum() > 0
    assert np.array_equal(ra.watch_first_step, rb.watch_first_step)
