from __future__ import annotations

import numpy as np
import torch

from flybrain import rng as crng


def test_mul32_and_mix32_stay_in_32_bits_and_match_python_ints():
    xs = np.array([0, 1, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 123456789], dtype=np.int64)
    for c in (0x7FEB352D, 0x846CA68B):
        got = crng._mul32(xs, c)
        assert got.tolist() == [(int(x) * c) & 0xFFFFFFFF for x in xs]
    mixed = crng.mix32(xs)
    assert mixed.min() >= 0 and mixed.max() <= 0xFFFFFFFF
    assert (
        len(np.unique(crng.mix32(np.arange(200_000, dtype=np.int64)))) == 200_000
    )  # bijection → no collisions


def test_numpy_and_torch_agree():
    neurons = np.arange(50, dtype=np.int64).reshape(-1, 1)
    steps = np.arange(1000, dtype=np.int64).reshape(1, -1)
    lo, hi = crng.split_seed(2**40 + 12345)
    a = crng.draw_u32(crng.stream_key(neurons, lo, hi), steps)
    b = crng.draw_u32(crng.stream_key(torch.as_tensor(neurons), lo, hi), torch.as_tensor(steps)).numpy()
    assert np.array_equal(a, b)


def test_bernoulli_rate_and_independence():
    neurons = np.arange(64, dtype=np.int64).reshape(-1, 1)
    steps = np.arange(100_000, dtype=np.int64).reshape(1, -1)
    lo, hi = crng.split_seed(7)
    u = crng.draw_u32(crng.stream_key(neurons, lo, hi), steps)
    p = 0.01
    kicks = u < crng.rate_to_p_u32(100.0, 0.1)
    n = kicks.size
    assert abs(kicks.mean() - p) < 5 * np.sqrt(p * (1 - p) / n)
    # per-stream rates are all plausible
    per = kicks.mean(axis=1)
    assert np.all(np.abs(per - p) < 6 * np.sqrt(p * (1 - p) / kicks.shape[1]))
    # uniformity of the raw values and absence of correlation between streams / lags
    x = u.astype(np.float64) / 2**32
    assert abs(x.mean() - 0.5) < 1e-3
    assert abs(np.corrcoef(x[0], x[1])[0, 1]) < 0.02
    assert abs(np.corrcoef(x[0, :-1], x[0, 1:])[0, 1]) < 0.02
    # different trial seeds give different streams
    u2 = crng.draw_u32(crng.stream_key(neurons, *crng.split_seed(8)), steps)
    assert (u == u2).mean() < 1e-3


def test_rate_to_p_u32():
    assert crng.rate_to_p_u32(0.0, 0.1) == 0
    assert crng.rate_to_p_u32(100.0, 0.1) == int(0.01 * 2**32)
    assert crng.rate_to_p_u32(1e9, 0.1) == 2**32  # saturates at probability 1
