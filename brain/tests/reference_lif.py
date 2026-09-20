"""Independent reference simulator used only by tests.

Dense, per-step, single-column NumPy implementation of the *literal* Brian2 semantics of the published model
(explicit `not_refractory` masking of the ODE, the threshold and every write; no chunking, no ring buffer, no
(0, 0) shortcut). The production engine in `flybrain/lif.py` must reproduce its spikes exactly.
"""

from __future__ import annotations

import numpy as np

from flybrain import rng as crng
from flybrain.connectome import Connectome
from flybrain.lif import LIFParams


def simulate_reference(
    conn: Connectome,
    n_steps: int,
    *,
    params: LIFParams | None = None,
    stim_idx: np.ndarray | None = None,
    kick_table: np.ndarray | None = None,  # bool [T, n_stim]
    p_u32: np.ndarray | None = None,  # int64 [n_stim]
    seed: int = 0,
    ablate: np.ndarray | None = None,
    nonrefractory_targets: bool = True,
    dtype=np.float64,
) -> np.ndarray:
    """Returns spikes as a bool array [n_steps, N]."""
    p = params or LIFParams()
    em, es, c1 = (dtype(x) for x in p.propagator())
    th, w_syn, kick_mv = dtype(p.threshold_u), dtype(p.w_syn_mv), dtype(p.kick_mv)
    n, delay = conn.n, p.delay_steps
    w_int = conn.dense_weight_matrix()  # [post, pre]

    u = np.zeros(n, dtype=dtype)
    g = np.zeros(n, dtype=dtype)
    last = np.full(n, -(10**9), dtype=np.int64)
    ref = np.full(n, p.ref_steps, dtype=np.int64)
    can_spike = np.ones(n, dtype=bool)
    if ablate is not None:
        can_spike[ablate] = False
    stim = np.zeros(0, dtype=np.int64) if stim_idx is None else np.asarray(stim_idx, dtype=np.int64)
    if nonrefractory_targets:
        ref[stim] = 0
    if p_u32 is not None:
        lo, hi = crng.split_seed(seed)
        key = crng.stream_key(stim, lo, hi)

    spikes = np.zeros((n_steps, n), dtype=bool)
    for step in range(n_steps):
        nr = (step - last) >= ref
        u_new = u * em + g * c1
        g_new = g * es
        u = np.where(nr, u_new, u)
        g = np.where(nr, g_new, g)

        spk = (u > th) & nr & can_spike
        nr_now = nr & ~spk  # Brian2's thresholder marks spiking neurons refractory immediately

        if step - delay >= 0:
            inp = w_int @ spikes[step - delay].astype(np.int64)
            g = np.where(nr_now, g + inp.astype(dtype) * w_syn, g)

        if stim.size:
            if kick_table is not None:
                kick = kick_table[step] if step < len(kick_table) else np.zeros(stim.size, dtype=bool)
            elif p_u32 is not None:
                kick = crng.draw_u32(key, np.int64(step)) < p_u32
            else:
                kick = np.zeros(stim.size, dtype=bool)
            live = kick & nr_now[stim]
            u[stim] = np.where(live, u[stim] + kick_mv, u[stim])

        u[spk] = 0
        g[spk] = 0
        last[spk] = step
        spikes[step] = spk
    return spikes
