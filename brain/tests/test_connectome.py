from __future__ import annotations

import numpy as np
import pytest

from flybrain.connectome import Connectome, load_connectome, synthetic


def test_synthetic_is_deterministic_and_well_formed(synth: Connectome):
    again = synthetic(n=1000, seed=0)
    assert np.array_equal(synth.pre, again.pre)
    assert np.array_equal(synth.post, again.post)
    assert np.array_equal(synth.weight, again.weight)
    other = synthetic(n=1000, seed=1)
    assert not np.array_equal(synth.post[:1000], other.post[:1000])

    stats = synth.stats()
    assert stats["neurons"] == 1000
    assert stats["self_loops"] >= 12  # planted autapses
    assert stats["duplicate_edges"] >= 1  # planted duplicates
    assert stats["negative_edges"] > 0
    assert np.all(np.diff(synth.pre) >= 0)
    assert load_connectome("synthetic", n=1000, seed=0).n == 1000


def test_root_ids_stay_int64_beyond_2_53(synth: Connectome):
    assert synth.root_ids.dtype == np.int64
    assert synth.root_ids.min() > 2**53  # would be corrupted by any float / JSON-number round trip
    idx = np.array([0, 17, 999])
    ids = synth.ids_of(idx)
    assert np.array_equal(synth.index_of(ids), idx)
    assert np.array_equal(synth.index_of([int(x) for x in ids]), idx)
    with pytest.raises(KeyError):
        synth.index_of([int(ids[0]) + 1])
    assert synth.has_ids([int(ids[0]), 5]).tolist() == [True, False]


def test_dale_law_sign_per_presynaptic_neuron(synth: Connectome):
    for neuron in np.unique(synth.pre)[:200]:
        w = synth.weight[synth.pre == neuron]
        assert np.all(w > 0) or np.all(w < 0)


def test_out_adjacency_matches_dense_matrix(synth: Connectome):
    ptr, post, weight = synth.out_adjacency()
    assert ptr[0] == 0 and ptr[-1] == synth.n_edges
    dense = np.zeros((synth.n, synth.n), dtype=np.int64)
    for pre in range(synth.n):
        for e in range(ptr[pre], ptr[pre + 1]):
            dense[post[e], pre] += weight[e]
    assert np.array_equal(dense, synth.dense_weight_matrix())


def test_rejects_float_ids_and_unsorted_edges():
    ids = np.arange(3, dtype=np.int64)
    e = np.zeros(1, dtype=np.int32)
    with pytest.raises(TypeError):
        Connectome("x", ids.astype(np.float64), e, e, e)
    with pytest.raises(ValueError):
        Connectome(
            "x",
            ids,
            np.array([1, 0], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            np.array([1, 1], dtype=np.int32),
        )
    with pytest.raises(ValueError):
        Connectome("x", ids, np.array([5], dtype=np.int32), e, e)


def test_silence_removes_only_outgoing_edges(synth: Connectome):
    target = int(synth.pre[100])
    silenced = synth.silence([target])
    assert not np.any(silenced.pre == target)
    assert np.sum(silenced.post == target) == np.sum((synth.post == target) & (synth.pre != target))
    assert silenced.n == synth.n


def test_edge_mask_and_subgraph(synth: Connectome):
    groups = synth.meta["groups"]
    mask = synth.edge_mask(pre_idx=groups["LC4"] + groups["LPLC2"], post_idx=groups["DNp01"])
    assert mask.sum() > 0
    sub = synth.with_edges(mask, suffix="direct")
    assert set(np.unique(sub.post)) <= set(groups["DNp01"])
    assert set(np.unique(sub.pre)) <= set(groups["LC4"] + groups["LPLC2"])


def test_shuffle_preserves_degrees_weights_and_signs(synth: Connectome):
    sh = synth.shuffled(seed=3)
    assert np.array_equal(sh.pre, synth.pre)  # out-degree per presynaptic neuron
    assert np.array_equal(
        sh.weight, synth.weight
    )  # outgoing weight multiset + sign stay with the presynaptic neuron
    assert np.array_equal(np.bincount(sh.post, minlength=synth.n), np.bincount(synth.post, minlength=synth.n))
    assert not np.array_equal(sh.post, synth.post)
    assert np.array_equal(sh.post, synth.shuffled(seed=3).post)  # deterministic
    assert not np.array_equal(sh.post, synth.shuffled(seed=4).post)


def test_shuffle_with_preserved_edges(synth: Connectome):
    groups = synth.meta["groups"]
    keep = synth.edge_mask(pre_idx=groups["LC4"] + groups["LPLC2"]) | synth.edge_mask(
        post_idx=groups["DNp01"]
    )
    sh = synth.shuffled(seed=5, preserve=keep)
    assert np.array_equal(sh.post[keep], synth.post[keep])
    assert not np.array_equal(sh.post[~keep], synth.post[~keep])
    assert np.array_equal(np.bincount(sh.post, minlength=synth.n), np.bincount(synth.post, minlength=synth.n))
