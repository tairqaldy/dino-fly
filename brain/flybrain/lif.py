"""Batched leaky integrate-and-fire simulation of a connectome — our own implementation of the published
whole-brain model (Shiu et al., Nature 634:210-219, 2024), written to be *Brian2-faithful*.

Model (verified against the authors' MIT-licensed Brian2 code and against Brian2's source):

    dv/dt = (v_0 - v + g) / tau_m          (unless refractory)
    dg/dt = -g / tau_syn                   (unless refractory)
    spike:  v > v_th (and not refractory)   reset:  v = v_rst, g = 0
    synapse: g_post += w after a fixed delay,  w = signed synapse count * w_syn
    Poisson drive: Bernoulli(rate*dt) kicks of w_syn*f_poi = 68.75 mV straight into v; driven neurons have
    refractory period 0 (as in the published code).

Brian2 semantics reproduced here:
  * dt = 0.1 ms (Brian2 default; the paper code never sets it); integration is exact (`method='linear'`).
  * Order within a step: integrate → threshold → deliver delayed input + Poisson kicks → reset.
    A kick at step n therefore produces a spike at step n+1; a spike at step m is delivered at m+18 and can
    cause a postsynaptic spike at m+19 at the earliest.
  * `(unless refractory)` makes v and g *read-only* while refractory: input and kicks that arrive during the
    refractory period are DROPPED, not banked. A neuron that spiked at step m is refractory for m+1 … m+21 and
    integrates again at m+22:  not_refractory(n) = (n - m) >= 22.

Implementation notes:
  * State is neuron-major `[N, B]`; B independent brains (trials / games) share one connectome.
  * We store u = v - v_0. Because v_rst == v_0 (asserted) and the reset zeroes g, a refractory neuron sits at
    exactly (u, g) = (0, 0), so only *delivery and kicks* need the refractory mask.
  * Event-driven propagation in chunks: spikes emitted inside a chunk of K <= delay steps cannot affect that
    chunk, so K steps run as pure elementwise work, followed by ONE `nonzero` and a scatter of signed integer
    synapse counts into a ring buffer at (step + delay). Integer accumulation is order-independent, hence
    deterministic on GPU.
  * All per-step ops use device-side step counters (no host sync), so the same op sequence can be replayed
    as a CUDA graph (`use_cuda_graph=True`). Arithmetic uses plain IEEE mul/add kernels (no fused a*b+c), so
    CPU, GPU-eager and CUDA-graph paths agree bit-for-bit.
  * Poisson drive uses counter-based random numbers (`flybrain.rng`): results do not depend on B, K or device.
  * Exact active set: a neuron at rest that never received input is a fixed point of the dynamics (u = g = 0,
    no spike, ever), and in this model >99 % of the brain is in that state at any time. State rows are therefore
    *slots* that are assigned to neurons the first time they receive input (or are driven), and every per-step
    kernel runs on the prefix of assigned slots only. `active_set=False` pre-assigns slot i = neuron i (dense
    mode). Both modes are the same code path and give identical spikes (tests/test_lif_invariance.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from flybrain import rng as crng
from flybrain.connectome import Connectome


@dataclass(frozen=True)
class LIFParams:
    """Published parameters (Shiu et al. 2024, `model.py` default_params)."""

    dt_ms: float = 0.1
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_th_mv: float = -45.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    t_ref_ms: float = 2.2
    t_delay_ms: float = 1.8
    w_syn_mv: float = 0.275
    poisson_factor: float = 250.0

    @property
    def ref_steps(self) -> int:
        return round(self.t_ref_ms / self.dt_ms)

    @property
    def delay_steps(self) -> int:
        return round(self.t_delay_ms / self.dt_ms)

    @property
    def kick_mv(self) -> float:
        return self.w_syn_mv * self.poisson_factor

    @property
    def threshold_u(self) -> float:
        return self.v_th_mv - self.v_rest_mv

    def propagator(self) -> tuple[float, float, float]:
        """(em, es, c1) of the exact update  u' = u*em + g*c1,  g' = g*es  (float64)."""
        em = math.exp(-self.dt_ms / self.tau_m_ms)
        es = math.exp(-self.dt_ms / self.tau_syn_ms)
        c1 = self.tau_syn_ms / (self.tau_syn_ms - self.tau_m_ms) * (es - em)
        return em, es, c1


# ----------------------------------------------------------------------------- drives
class PoissonDrive:
    """Bernoulli(rate·dt) kicks for a fixed set of neurons, with per-column rates and per-column trial seeds."""

    def __init__(
        self, neuron_idx: np.ndarray, batch_size: int, dt_ms: float, device: torch.device | str = "cpu"
    ):
        self.neuron_idx = np.asarray(neuron_idx, dtype=np.int64)
        if len(np.unique(self.neuron_idx)) != len(self.neuron_idx):
            raise ValueError("driven neuron indices must be unique")
        self.batch_size = batch_size
        self.dt_ms = dt_ms
        self.device = torch.device(device)
        n = len(self.neuron_idx)
        self._idx_t = torch.as_tensor(self.neuron_idx, device=self.device).view(n, 1)
        self.p_u32 = torch.zeros(n, batch_size, dtype=torch.int64, device=self.device)
        self.key = torch.zeros(n, batch_size, dtype=torch.int64, device=self.device)
        self.set_seeds(np.arange(batch_size, dtype=np.int64))

    def set_seeds(self, seeds: np.ndarray, columns: np.ndarray | None = None) -> None:
        """Trial seed per column (all columns, or only `columns`)."""
        seeds = np.asarray(seeds, dtype=np.uint64)
        cols = np.arange(self.batch_size) if columns is None else np.asarray(columns, dtype=np.int64)
        if seeds.shape != cols.shape:
            raise ValueError("one seed per selected column required")
        lo = torch.as_tensor((seeds & np.uint64(crng.M32)).astype(np.int64), device=self.device).view(1, -1)
        hi = torch.as_tensor((seeds >> np.uint64(32)).astype(np.int64), device=self.device).view(1, -1)
        self.key[:, torch.as_tensor(cols, device=self.device)] = crng.stream_key(self._idx_t, lo, hi)

    def set_rates(self, rates_hz: np.ndarray | torch.Tensor) -> None:
        """Rates in Hz, shape [n_stim] (same for all columns) or [n_stim, B]."""
        r = torch.as_tensor(rates_hz, dtype=torch.float64, device=self.device)
        if r.ndim == 1:
            r = r.view(-1, 1).expand(-1, self.batch_size)
        if r.shape != self.p_u32.shape:
            raise ValueError(f"rates shape {tuple(r.shape)} != {tuple(self.p_u32.shape)}")
        if bool((r < 0).any()):
            raise ValueError("negative rate")
        p = torch.clamp(r * (self.dt_ms * 1e-3) * 4294967296.0, max=4294967296.0)
        self.p_u32.copy_(p.to(torch.int64))

    def kicks(self, rel_steps: torch.Tensor) -> torch.Tensor:
        """rel_steps: int64 [K, 1, B] trial-relative step numbers → bool [K, n_stim, B]."""
        return crng.draw_u32(self.key.unsqueeze(0), rel_steps) < self.p_u32.unsqueeze(0)


class ScheduledDrive:
    """Deterministic kicks from an explicit table (used by tests and the Brian2 cross-check)."""

    def __init__(self, neuron_idx: np.ndarray, table: np.ndarray, device: torch.device | str = "cpu"):
        self.neuron_idx = np.asarray(neuron_idx, dtype=np.int64)
        table = np.asarray(table, dtype=bool)  # [T, n_stim, B]
        if table.ndim != 3 or table.shape[1] != len(self.neuron_idx):
            raise ValueError("table must be [T, n_stim, B]")
        self.table = torch.as_tensor(table, device=device)
        self.batch_size = table.shape[2]

    def kicks(self, rel_steps: torch.Tensor) -> torch.Tensor:
        t = rel_steps[:, 0, :]  # [K, B]
        n_t = self.table.shape[0]
        valid = (t >= 0) & (t < n_t)
        cols = torch.arange(self.batch_size, device=self.table.device).view(1, -1).expand_as(t)
        out = self.table[t.clamp(0, n_t - 1), :, cols]  # [K, B, n_stim]
        return (out & valid.unsqueeze(-1)).permute(0, 2, 1)


# ----------------------------------------------------------------------------- results
@dataclass
class RunResult:
    n_steps: int
    dt_ms: float
    counts: np.ndarray | None = None  # int64 [N, B] spikes per neuron and column during this run
    spikes: np.ndarray | None = None  # int64 [M, 3] rows (global step, neuron index, column)
    watch_counts: np.ndarray | None = None  # int64 [n_watch, B]
    watch_first_step: np.ndarray | None = (
        None  # int64 [n_watch, B], run-relative step of first spike, -1 if none
    )
    meta: dict = field(default_factory=dict)

    def rates_hz(self) -> np.ndarray:
        """Per-neuron, per-column firing rate over the run."""
        if self.counts is None:
            raise ValueError("run was not recorded with counts")
        return self.counts / (self.n_steps * self.dt_ms * 1e-3)


# ----------------------------------------------------------------------------- engine
SLOT_BUCKET = 4096  # active-set capacity grows in buckets so CUDA graphs are re-captured rarely


class _Views:
    """Prefix views [:cap] of the state tensors (rebuilt whenever the active-set capacity changes)."""

    def __init__(self, net: LIFNetwork, cap: int) -> None:
        self.cap = cap
        self.u = net.u[:cap]
        self.g = net.g[:cap]
        self.refr_until = net._refr_until[:cap]
        self.ref_steps = net._ref_steps_s[:cap]
        self.ref_tmp = net._ref_tmp[:cap]
        self.can_spike = net._can_spike_s[:cap]
        self.ring = net._ring[:, :cap]
        self.inp = net._inp[:, :cap]
        self.tmp = net._tmp[:cap]
        self.nr = net._nr[:cap]
        self.spk = net._spk_buf[:, :cap]


class _Plastic:
    """Book-keeping of the plastic synapse group (see LIFNetwork.set_plastic)."""

    edge_idx: np.ndarray
    plastic_id: torch.Tensor  # [E] → index into gains, -1 for frozen edges
    n_post: int
    post_pos: torch.Tensor  # [P] position of each plastic edge's target within post_neurons
    post_neurons: torch.Tensor
    post_slots: torch.Tensor
    gains: torch.Tensor  # [P]
    ring: torch.Tensor  # [L, n_post, B] float
    inp: torch.Tensor


class LIFNetwork:
    def __init__(
        self,
        conn: Connectome,
        params: LIFParams | None = None,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        chunk_steps: int | None = None,
        use_cuda_graph: bool = False,
        active_set: bool = True,
        max_events_per_slice: int = 4_000_000,
    ) -> None:
        self.params = params or LIFParams()
        p = self.params
        if p.v_reset_mv != p.v_rest_mv:
            # The (0, 0) fixed point of refractory / untouched neurons needs this.
            raise NotImplementedError("engine assumes v_reset == v_rest (true for the published model)")
        self.conn = conn
        self.n = conn.n
        self.b = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.delay = p.delay_steps
        self.k = int(chunk_steps or self.delay)
        if not 1 <= self.k <= self.delay:
            raise ValueError(f"chunk_steps must be in [1, {self.delay}] (spikes must not affect their own chunk)")
        self.ring_len = self.delay + self.k
        self.max_events = int(max_events_per_slice)
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self.active_set = bool(active_set)

        dev, n, b = self.device, self.n, self.b
        ptr, post, weight = conn.out_adjacency()
        self._out_ptr = torch.as_tensor(ptr, device=dev)
        self._out_deg = torch.as_tensor(np.diff(ptr), device=dev)
        self._out_post = torch.as_tensor(post.astype(np.int64), device=dev)
        self._out_w = torch.as_tensor(weight, device=dev)

        em, es, c1 = p.propagator()
        self._em, self._es, self._c1 = em, es, c1
        self._th = p.threshold_u
        self._w_syn = p.w_syn_mv
        self._kick_mv = p.kick_mv

        # slot-indexed state (row s belongs to neuron _neuron_of_slot[s])
        self.u = torch.zeros(n, b, dtype=dtype, device=dev)
        self.g = torch.zeros(n, b, dtype=dtype, device=dev)
        self._refr_until = torch.zeros(n, b, dtype=torch.int32, device=dev)
        self._ref_steps_s = torch.full((n, 1), p.ref_steps, dtype=torch.int32, device=dev)
        self._ref_tmp = torch.zeros(n, 1, dtype=torch.int32, device=dev)
        self._can_spike_s = torch.ones(n, 1, dtype=torch.bool, device=dev)
        self._ring = torch.zeros(self.ring_len, n, b, dtype=torch.int32, device=dev)
        self._inp = torch.zeros(1, n, b, dtype=torch.int32, device=dev)
        self._tmp = torch.zeros(n, b, dtype=dtype, device=dev)
        self._nr = torch.zeros(n, b, dtype=torch.bool, device=dev)
        self._spk_buf = torch.zeros(self.k, n, b, dtype=torch.bool, device=dev)
        self._step_i32 = torch.zeros(1, dtype=torch.int32, device=dev)
        self._slot = torch.zeros(1, dtype=torch.int64, device=dev)

        # neuron-indexed parameters and bookkeeping
        self._ref_steps_n = torch.full((n,), p.ref_steps, dtype=torch.int32, device=dev)
        self._can_spike_n = torch.ones(n, dtype=torch.bool, device=dev)
        self._has_ablation = False
        self._slot_of = torch.full((n,), -1, dtype=torch.int64, device=dev)
        self._neuron_of_slot = torch.zeros(n, dtype=torch.int64, device=dev)
        self._n_act = 0
        self._v = _Views(self, 0)

        self.step = 0  # global step counter (host side)
        self._col_start = torch.zeros(b, dtype=torch.int64, device=dev)  # step at which each column's trial began
        self.spike_counts = torch.zeros(n, b, dtype=torch.int64, device=dev)  # neuron-indexed

        self._drive: PoissonDrive | ScheduledDrive | None = None
        self._stim_neurons: torch.Tensor | None = None
        self._stim_slots: torch.Tensor | None = None
        self._kick_buf: torch.Tensor | None = None
        self._plastic: _Plastic | None = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._init_active_set()

    # ------------------------------------------------------------------ active set
    @property
    def n_active(self) -> int:
        return self._n_act

    def _init_active_set(self) -> None:
        self._slot_of.fill_(-1)
        self._n_act = 0
        if not self.active_set:
            self._activate(torch.arange(self.n, device=self.device))
        else:
            if self._stim_neurons is not None:
                self._activate(self._stim_neurons)
            if self._plastic is not None:
                post = self._plastic.post_neurons
                self._activate(post[self._slot_of[post] < 0])
            self._set_capacity()
        if self._plastic is not None:
            self._plastic.post_slots = self._slot_of[self._plastic.post_neurons]

    def _activate(self, neurons: torch.Tensor) -> None:
        """Give slots to `neurons` (unique, all currently without a slot). Their state rows are at rest (zero)."""
        m = int(neurons.numel())
        if m:
            slots = torch.arange(self._n_act, self._n_act + m, device=self.device)
            self._slot_of[neurons] = slots
            self._neuron_of_slot[slots] = neurons
            self._ref_steps_s[slots, 0] = self._ref_steps_n[neurons]
            self._can_spike_s[slots, 0] = self._can_spike_n[neurons]
            self._n_act += m
        self._set_capacity()

    def _set_capacity(self) -> None:
        cap = min(self.n, -(-max(self._n_act, 1) // SLOT_BUCKET) * SLOT_BUCKET)
        if cap != self._v.cap:
            self._v = _Views(self, cap)
            self._graphs.clear()

    def _refresh_slot_params(self) -> None:
        if self._n_act:
            neurons = self._neuron_of_slot[: self._n_act]
            self._ref_steps_s[: self._n_act, 0] = self._ref_steps_n[neurons]
            self._can_spike_s[: self._n_act, 0] = self._can_spike_n[neurons]
        self._graphs.clear()

    def _slots(self, neurons: torch.Tensor) -> torch.Tensor:
        return self._slot_of[neurons]

    # ------------------------------------------------------------------ configuration
    def set_drive(self, drive: PoissonDrive | ScheduledDrive | None, *, nonrefractory_targets: bool = True) -> None:
        """Attach a drive. Published convention: Poisson-driven neurons have refractory period 0."""
        self._ref_steps_n.fill_(self.params.ref_steps)
        self._drive = drive
        if drive is None:
            self._stim_neurons = self._stim_slots = self._kick_buf = None
            self._refresh_slot_params()
            return
        if drive.batch_size != self.b:
            raise ValueError("drive batch size mismatch")
        self._stim_neurons = torch.as_tensor(drive.neuron_idx, device=self.device)
        if nonrefractory_targets:
            self._ref_steps_n[self._stim_neurons] = 0
        fresh = self._stim_neurons[self._slot_of[self._stim_neurons] < 0]
        self._activate(fresh)
        self._stim_slots = self._slot_of[self._stim_neurons]
        self._kick_buf = torch.zeros(self.k, len(drive.neuron_idx), self.b, dtype=torch.bool, device=self.device)
        self._refresh_slot_params()

    def set_plastic(self, edge_mask: np.ndarray | None, gains: torch.Tensor | None = None) -> torch.Tensor | None:
        """Make the masked edges plastic: their weight becomes w0 · gain (gain tensor shared by all columns).

        Plastic input travels through a small float ring buffer that only covers the postsynaptic neurons of the
        plastic edges (the MBONs). With all gains = 1 the result is identical to the frozen network. Returns the
        gain tensor [n_plastic] (float, on device) so that a learning rule can update it in place.
        """
        self._graphs.clear()
        if edge_mask is None:
            self._plastic = None
            return None
        edge_idx = np.flatnonzero(np.asarray(edge_mask, dtype=bool))
        post_neurons = np.unique(self.conn.post[edge_idx]).astype(np.int64)
        dev = self.device
        plastic_id = torch.full((self.conn.n_edges,), -1, dtype=torch.int64, device=dev)
        plastic_id[torch.as_tensor(edge_idx, device=dev)] = torch.arange(len(edge_idx), device=dev)
        post_t = torch.as_tensor(post_neurons, device=dev)
        self._activate(post_t[self._slot_of[post_t] < 0])
        pl = _Plastic()
        pl.edge_idx = edge_idx
        pl.plastic_id = plastic_id
        pl.n_post = len(post_neurons)
        pl.post_pos = torch.as_tensor(np.searchsorted(post_neurons, self.conn.post[edge_idx]), device=dev)
        pl.post_neurons = post_t
        pl.post_slots = self._slot_of[post_t]
        pl.gains = torch.ones(len(edge_idx), dtype=self.dtype, device=dev) if gains is None else gains.to(dev, self.dtype)
        pl.ring = torch.zeros(self.ring_len, pl.n_post, self.b, dtype=self.dtype, device=dev)
        pl.inp = torch.zeros(1, pl.n_post, self.b, dtype=self.dtype, device=dev)
        self._plastic = pl
        return pl.gains

    def ablate(self, neuron_idx: np.ndarray | None) -> None:
        """Neurons that cannot spike (Kir2.1-like silencing). `None` clears the ablation."""
        self._can_spike_n.fill_(True)
        self._has_ablation = neuron_idx is not None and len(neuron_idx) > 0
        if self._has_ablation:
            self._can_spike_n[torch.as_tensor(np.asarray(neuron_idx, dtype=np.int64), device=self.device)] = False
        self._refresh_slot_params()

    def reset(self, columns: np.ndarray | None = None) -> None:
        """Reset state, in-flight input and trial clock of the given columns (default: all columns)."""
        if columns is None:
            self.u.zero_()
            self.g.zero_()
            self._refr_until.zero_()
            self._ring.zero_()
            if self._plastic is not None:
                self._plastic.ring.zero_()
            self.spike_counts.zero_()
            self._col_start.fill_(self.step)
            self._init_active_set()  # shrink back to the driven neurons
            if self._stim_neurons is not None:
                self._stim_slots = self._slot_of[self._stim_neurons]
            self._refresh_slot_params()
            return
        cols = torch.as_tensor(np.asarray(columns, dtype=np.int64), device=self.device)
        if cols.numel() == 0:
            return
        cap = self._v.cap
        self.u[:cap, cols] = 0
        self.g[:cap, cols] = 0
        self._refr_until[:cap, cols] = 0
        self._ring[:, :cap, cols] = 0
        if self._plastic is not None:
            self._plastic.ring[:, :, cols] = 0
        self.spike_counts[:, cols] = 0
        self._col_start[cols] = self.step

    # ------------------------------------------------------------------ the model
    def _step_ops(self, k: int) -> None:
        """One dt on the active slots. Only device-side tensors → identical in eager mode and inside a CUDA graph."""
        v = self._v
        u, g, tmp, nr = v.u, v.g, v.tmp, v.nr
        # 1) integrate (exact propagator; plain IEEE mul/add — no fused multiply-add patterns)
        torch.mul(g, self._c1, out=tmp)
        u.mul_(self._em)
        u.add_(tmp)
        g.mul_(self._es)
        # 2) threshold
        spk = v.spk[k]
        torch.gt(u, self._th, out=spk)
        if self._has_ablation:
            spk.logical_and_(v.can_spike)
        # 3) deliver delayed input + Poisson kicks; both are dropped for refractory neurons
        torch.le(v.refr_until, self._step_i32, out=nr)
        torch.index_select(v.ring, 0, self._slot, out=v.inp)
        v.ring.index_fill_(0, self._slot, 0)
        v.inp.mul_(nr)
        tmp.copy_(v.inp[0])
        tmp.mul_(self._w_syn)
        g.add_(tmp)
        if self._plastic is not None:  # float path for the (few) plastic synapses: KC→MBON
            pl = self._plastic
            torch.index_select(pl.ring, 0, self._slot, out=pl.inp)
            pl.ring.index_fill_(0, self._slot, 0.0)
            g.index_add_(0, pl.post_slots, pl.inp[0].mul(nr.index_select(0, pl.post_slots)).mul_(self._w_syn))
        if self._stim_slots is not None:
            live = self._kick_buf[k] & nr.index_select(0, self._stim_slots)
            u.index_add_(0, self._stim_slots, live.to(self.dtype).mul_(self._kick_mv))
        # 4) reset (erases anything delivered to a neuron in the step it spiked = Brian2's drop)
        u.masked_fill_(spk, 0.0)
        g.masked_fill_(spk, 0.0)
        torch.add(v.ref_steps, self._step_i32, out=v.ref_tmp)
        torch.where(spk, v.ref_tmp, v.refr_until, out=v.refr_until)
        # 5) advance device-side clock
        self._step_i32.add_(1)
        self._slot.add_(1)
        self._slot.remainder_(self.ring_len)

    def _run_chunk_ops(self, k_len: int) -> None:
        for k in range(k_len):
            self._step_ops(k)

    def _run_chunk(self, k_len: int) -> None:
        if not self.use_cuda_graph:
            self._run_chunk_ops(k_len)
            return
        graph = self._graphs.get(k_len)
        if graph is None:
            # Warm-up on a side stream, restore state, capture (standard CUDA-graph recipe).
            saved = [t.clone() for t in self._graph_state()]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._run_chunk_ops(k_len)
            torch.cuda.current_stream().wait_stream(side)
            self._restore(saved)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._run_chunk_ops(k_len)
            self._restore(saved)
            self._graphs[k_len] = graph
        graph.replay()

    def _graph_state(self) -> list[torch.Tensor]:
        v = self._v
        return [v.u, v.g, v.refr_until, v.ring, self._step_i32, self._slot, v.spk]

    def _restore(self, saved: list[torch.Tensor]) -> None:
        for dst, src in zip(self._graph_state(), saved, strict=True):
            dst.copy_(src)

    def _propagate(self, k_len: int, step0: int, record_spikes: list | None, count: bool) -> None:
        idx = self._v.spk[:k_len].nonzero()  # [M, 3] = (k, slot, column); the one host sync per chunk
        m = idx.shape[0]
        if m == 0:
            return
        kk, bb = idx[:, 0], idx[:, 2]
        nn = self._neuron_of_slot[idx[:, 1]]
        if count:
            self.spike_counts.index_put_((nn, bb), torch.ones_like(nn), accumulate=True)
        if record_spikes is not None:
            record_spikes.append(torch.stack([kk + step0, nn, bb], dim=1).cpu())
        deg = self._out_deg[nn]
        cum = deg.cumsum(0)
        total = int(cum[-1])
        if total == 0:
            return
        if total <= self.max_events:
            bounds = [(0, m)]
        else:  # rare: slice a burst so the expansion cannot exhaust memory
            cum_cpu = cum.cpu().numpy()
            bounds, lo = [], 0
            while lo < m:
                base = cum_cpu[lo - 1] if lo else 0
                hi = int(np.searchsorted(cum_cpu, base + self.max_events, side="right"))
                hi = min(max(hi, lo + 1), m)
                bounds.append((lo, hi))
                lo = hi
        ring_flat = self._ring.view(-1)
        for lo, hi in bounds:
            d = deg[lo:hi]
            n_ev = int(cum[hi - 1] - (cum[lo - 1] if lo else 0))
            if n_ev == 0:
                continue
            rep = torch.repeat_interleave(torch.arange(hi - lo, device=self.device), d, output_size=n_ev)
            first = (d.cumsum(0) - d)[rep]
            edge = self._out_ptr[nn[lo:hi]][rep] + (torch.arange(n_ev, device=self.device) - first)
            targets = self._out_post[edge]
            tslot = self._slot_of[targets]
            if self.active_set:
                untouched = tslot < 0
                if bool(untouched.any()):
                    self._activate(torch.unique(targets[untouched]))
                    tslot = self._slot_of[targets]
            ring_slot = (step0 + kk[lo:hi][rep] + self.delay) % self.ring_len
            cols = bb[lo:hi][rep]
            weights = self._out_w[edge]
            if self._plastic is not None:
                pl = self._plastic
                pid = pl.plastic_id[edge]
                is_pl = pid >= 0
                if bool(is_pl.any()):
                    pe = pid[is_pl]
                    flat_f = (ring_slot[is_pl] * pl.n_post + pl.post_pos[pe]) * self.b + cols[is_pl]
                    pl.ring.view(-1).index_add_(0, flat_f, weights[is_pl].to(self.dtype) * pl.gains[pe])
                    weights = weights.masked_fill(is_pl, 0)
            flat = (ring_slot * self.n + tslot) * self.b + cols
            ring_flat.index_add_(0, flat, weights)

    # ------------------------------------------------------------------ running
    @torch.no_grad()
    def run(self, n_steps: int, *, record: str | None = "counts", watch: np.ndarray | None = None) -> RunResult:
        """Advance all columns by n_steps.

        record: None | "counts" (per-neuron spike counts) | "spikes" (also every spike as (step, neuron, column)).
        watch:  neuron indices whose spike count and first-spike step (run-relative) are returned per column.
        """
        if record not in (None, "counts", "spikes"):
            raise ValueError("record must be None, 'counts' or 'spikes'")
        if record is not None:
            self.spike_counts.zero_()
        spike_list: list | None = [] if record == "spikes" else None
        watch_t = None if watch is None else torch.as_tensor(np.asarray(watch, dtype=np.int64), device=self.device)
        if watch_t is not None:
            w_counts = torch.zeros(len(watch_t), self.b, dtype=torch.int64, device=self.device)
            w_first = torch.full((len(watch_t), self.b), -1, dtype=torch.int64, device=self.device)

        start = self.step
        done = 0
        while done < n_steps:
            k_len = min(self.k, n_steps - done)
            step0 = self.step
            self._step_i32.fill_(step0)
            self._slot.fill_(step0 % self.ring_len)
            if self._drive is not None:
                rel = torch.arange(step0, step0 + k_len, device=self.device).view(-1, 1, 1) - self._col_start.view(1, 1, -1)
                self._kick_buf[:k_len] = self._drive.kicks(rel)
            self._run_chunk(k_len)
            self.step += k_len
            if watch_t is not None:  # read before propagation may re-bucket the views
                ws = self._slot_of[watch_t]
                w = self._v.spk[:k_len].index_select(1, ws.clamp(min=0)) & (ws >= 0).view(1, -1, 1)
                w_counts += w.sum(0)
                first_k = w.to(torch.int8).argmax(0) + (step0 - start)
                w_first = torch.where(w.any(0) & (w_first < 0), first_k, w_first)
            self._propagate(k_len, step0, spike_list, count=record is not None)
            done += k_len

        result = RunResult(n_steps=n_steps, dt_ms=self.params.dt_ms)
        if record is not None:
            result.counts = self.spike_counts.cpu().numpy()
        if spike_list is not None:
            spikes = torch.cat(spike_list).numpy() if spike_list else np.zeros((0, 3), dtype=np.int64)
            # canonical order (step, neuron, column): slot numbering must not leak into results
            result.spikes = spikes[np.lexsort((spikes[:, 2], spikes[:, 1], spikes[:, 0]))]
        if watch_t is not None:
            result.watch_counts = w_counts.cpu().numpy()
            result.watch_first_step = w_first.cpu().numpy()
        result.meta["n_active"] = self._n_act
        return result

    def state_mv(self, neuron_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(u, g) in mV of selected neurons, [n, B] each (neuron-indexed view of the slot-indexed state)."""
        idx = torch.as_tensor(np.asarray(neuron_idx, dtype=np.int64), device=self.device)
        slots = self._slot_of[idx]
        ok = (slots >= 0).view(-1, 1)
        u = torch.where(ok, self.u.index_select(0, slots.clamp(min=0)), torch.zeros((), dtype=self.dtype, device=self.device))
        g = torch.where(ok, self.g.index_select(0, slots.clamp(min=0)), torch.zeros((), dtype=self.dtype, device=self.device))
        return u.cpu().numpy(), g.cpu().numpy()

    def membrane_mv(self, neuron_idx: np.ndarray) -> np.ndarray:
        """Current membrane potential (mV) of selected neurons, [n, B]."""
        return self.state_mv(neuron_idx)[0] + self.params.v_rest_mv
