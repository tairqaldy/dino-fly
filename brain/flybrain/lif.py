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
        max_events_per_slice: int = 4_000_000,
    ) -> None:
        self.params = params or LIFParams()
        p = self.params
        if p.v_reset_mv != p.v_rest_mv:
            # The (0, 0) fixed point of refractory neurons — and with it the mask simplification — needs this.
            raise NotImplementedError("engine assumes v_reset == v_rest (true for the published model)")
        self.conn = conn
        self.n = conn.n
        self.b = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.delay = p.delay_steps
        self.k = int(chunk_steps or self.delay)
        if not 1 <= self.k <= self.delay:
            raise ValueError(
                f"chunk_steps must be in [1, {self.delay}] (spikes must not affect their own chunk)"
            )
        self.ring_len = self.delay + self.k
        self.max_events = int(max_events_per_slice)
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"

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

        self.u = torch.zeros(n, b, dtype=dtype, device=dev)
        self.g = torch.zeros(n, b, dtype=dtype, device=dev)
        self._refr_until = torch.zeros(n, b, dtype=torch.int32, device=dev)
        self._ref_steps = torch.full((n, 1), p.ref_steps, dtype=torch.int32, device=dev)
        self._ref_tmp = torch.zeros(n, 1, dtype=torch.int32, device=dev)
        self._can_spike = torch.ones(n, 1, dtype=torch.bool, device=dev)
        self._has_ablation = False
        self._ring = torch.zeros(self.ring_len, n, b, dtype=torch.int32, device=dev)
        self._inp = torch.zeros(1, n, b, dtype=torch.int32, device=dev)
        self._tmp = torch.zeros(n, b, dtype=dtype, device=dev)
        self._nr = torch.zeros(n, b, dtype=torch.bool, device=dev)
        self._spk_buf = torch.zeros(self.k, n, b, dtype=torch.bool, device=dev)
        self._step_i32 = torch.zeros(1, dtype=torch.int32, device=dev)
        self._slot = torch.zeros(1, dtype=torch.int64, device=dev)

        self.step = 0  # global step counter (host side)
        self._col_start = torch.zeros(
            b, dtype=torch.int64, device=dev
        )  # step at which each column's trial began
        self.spike_counts = torch.zeros(n, b, dtype=torch.int64, device=dev)

        self._drive: PoissonDrive | ScheduledDrive | None = None
        self._stim_idx: torch.Tensor | None = None
        self._kick_buf: torch.Tensor | None = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    # ------------------------------------------------------------------ configuration
    def set_drive(
        self, drive: PoissonDrive | ScheduledDrive | None, *, nonrefractory_targets: bool = True
    ) -> None:
        """Attach a drive. Published convention: Poisson-driven neurons have refractory period 0."""
        self._ref_steps.fill_(self.params.ref_steps)
        self._drive = drive
        self._graphs.clear()
        if drive is None:
            self._stim_idx, self._kick_buf = None, None
            return
        if drive.batch_size != self.b:
            raise ValueError("drive batch size mismatch")
        self._stim_idx = torch.as_tensor(drive.neuron_idx, device=self.device)
        self._kick_buf = torch.zeros(
            self.k, len(drive.neuron_idx), self.b, dtype=torch.bool, device=self.device
        )
        if nonrefractory_targets:
            self._ref_steps[self._stim_idx] = 0

    def ablate(self, neuron_idx: np.ndarray | None) -> None:
        """Neurons that cannot spike (Kir2.1-like silencing). `None` clears the ablation."""
        self._can_spike.fill_(True)
        self._has_ablation = neuron_idx is not None and len(neuron_idx) > 0
        if self._has_ablation:
            self._can_spike[torch.as_tensor(np.asarray(neuron_idx, dtype=np.int64), device=self.device)] = (
                False
            )
        self._graphs.clear()

    def reset(self, columns: np.ndarray | None = None) -> None:
        """Reset state, in-flight input and trial clock of the given columns (default: all)."""
        if columns is None:
            self.u.zero_()
            self.g.zero_()
            self._refr_until.zero_()
            self._ring.zero_()
            self.spike_counts.zero_()
            self._col_start.fill_(self.step)
            return
        cols = torch.as_tensor(np.asarray(columns, dtype=np.int64), device=self.device)
        if cols.numel() == 0:
            return
        self.u[:, cols] = 0
        self.g[:, cols] = 0
        self._refr_until[:, cols] = 0
        self._ring[:, :, cols] = 0
        self.spike_counts[:, cols] = 0
        self._col_start[cols] = self.step

    # ------------------------------------------------------------------ the model
    def _step_ops(self, k: int) -> None:
        """One dt. Only device-side tensors → identical in eager mode and inside a CUDA graph."""
        u, g, tmp, nr = self.u, self.g, self._tmp, self._nr
        # 1) integrate (exact propagator; plain IEEE mul/add — no fused multiply-add patterns)
        torch.mul(g, self._c1, out=tmp)
        u.mul_(self._em)
        u.add_(tmp)
        g.mul_(self._es)
        # 2) threshold
        spk = self._spk_buf[k]
        torch.gt(u, self._th, out=spk)
        if self._has_ablation:
            spk.logical_and_(self._can_spike)
        # 3) deliver delayed input + Poisson kicks; both are dropped for refractory neurons
        torch.le(self._refr_until, self._step_i32, out=nr)
        torch.index_select(self._ring, 0, self._slot, out=self._inp)
        self._ring.index_fill_(0, self._slot, 0)
        self._inp.mul_(nr)
        tmp.copy_(self._inp[0])
        tmp.mul_(self._w_syn)
        g.add_(tmp)
        if self._stim_idx is not None:
            live = self._kick_buf[k] & nr.index_select(0, self._stim_idx)
            u.index_add_(0, self._stim_idx, live.to(self.dtype).mul_(self._kick_mv))
        # 4) reset (erases anything delivered to a neuron in the step it spiked = Brian2's drop)
        u.masked_fill_(spk, 0.0)
        g.masked_fill_(spk, 0.0)
        torch.add(self._ref_steps, self._step_i32, out=self._ref_tmp)
        torch.where(spk, self._ref_tmp, self._refr_until, out=self._refr_until)
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
            # Warm-up on a side stream, then restore state, then capture (standard CUDA-graph recipe).
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
            self._restore(saved)  # capture does not execute, but keep state pristine regardless
            self._graphs[k_len] = graph
        graph.replay()

    def _graph_state(self) -> list[torch.Tensor]:
        return [self.u, self.g, self._refr_until, self._ring, self._step_i32, self._slot, self._spk_buf]

    def _restore(self, saved: list[torch.Tensor]) -> None:
        for dst, src in zip(self._graph_state(), saved, strict=True):
            dst.copy_(src)

    def _propagate(self, k_len: int, step0: int, record_spikes: list | None, count: bool) -> None:
        idx = self._spk_buf[:k_len].nonzero()  # [M, 3] = (k, neuron, column); the one host sync per chunk
        m = idx.shape[0]
        if m == 0:
            return
        kk, nn, bb = idx[:, 0], idx[:, 1], idx[:, 2]
        if count:
            self.spike_counts.index_put_((nn, bb), torch.ones_like(nn), accumulate=True)
        if record_spikes is not None:
            rec = idx.clone()
            rec[:, 0] += step0
            record_spikes.append(rec.cpu())
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
                hi = max(hi, lo + 1)
                bounds.append((lo, min(hi, m)))
                lo = min(hi, m)
        ring_flat = self._ring.view(-1)
        for lo, hi in bounds:
            d = deg[lo:hi]
            n_ev = int(cum[hi - 1] - (cum[lo - 1] if lo else 0))
            if n_ev == 0:
                continue
            rep = torch.repeat_interleave(torch.arange(hi - lo, device=self.device), d, output_size=n_ev)
            first = (d.cumsum(0) - d)[rep]
            edge = self._out_ptr[nn[lo:hi]][rep] + (torch.arange(n_ev, device=self.device) - first)
            slot = (step0 + kk[lo:hi][rep] + self.delay) % self.ring_len
            flat = (slot * self.n + self._out_post[edge]) * self.b + bb[lo:hi][rep]
            ring_flat.index_add_(0, flat, self._out_w[edge])

    # ------------------------------------------------------------------ running
    @torch.no_grad()
    def run(
        self,
        n_steps: int,
        *,
        record: str | None = "counts",
        watch: np.ndarray | None = None,
    ) -> RunResult:
        """Advance all columns by n_steps.

        record: None | "counts" (per-neuron spike counts) | "spikes" (also every spike as (step, neuron, column)).
        watch:  neuron indices whose spike count and first-spike step (run-relative) are returned per column.
        """
        if record not in (None, "counts", "spikes"):
            raise ValueError("record must be None, 'counts' or 'spikes'")
        if record is not None:
            self.spike_counts.zero_()
        spike_list: list | None = [] if record == "spikes" else None
        watch_t = (
            None if watch is None else torch.as_tensor(np.asarray(watch, dtype=np.int64), device=self.device)
        )
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
                rel = torch.arange(step0, step0 + k_len, device=self.device).view(
                    -1, 1, 1
                ) - self._col_start.view(1, 1, -1)
                self._kick_buf[:k_len] = self._drive.kicks(rel)
            self._run_chunk(k_len)
            self.step += k_len
            self._propagate(k_len, step0, spike_list, count=record is not None)
            if watch_t is not None:
                w = self._spk_buf[:k_len].index_select(1, watch_t)  # [k_len, n_watch, B]
                w_counts += w.sum(0)
                first_k = w.to(torch.int8).argmax(0) + (step0 - start)
                new = w.any(0) & (w_first < 0)
                w_first = torch.where(new, first_k, w_first)
            done += k_len

        result = RunResult(n_steps=n_steps, dt_ms=self.params.dt_ms)
        if record is not None:
            result.counts = self.spike_counts.cpu().numpy()
        if spike_list is not None:
            result.spikes = torch.cat(spike_list).numpy() if spike_list else np.zeros((0, 3), dtype=np.int64)
        if watch_t is not None:
            result.watch_counts = w_counts.cpu().numpy()
            result.watch_first_step = w_first.cpu().numpy()
        return result

    def membrane_mv(self, neuron_idx: np.ndarray) -> np.ndarray:
        """Current membrane potential (mV) of selected neurons, [n, B]."""
        idx = torch.as_tensor(np.asarray(neuron_idx, dtype=np.int64), device=self.device)
        return (self.u.index_select(0, idx) + self.params.v_rest_mv).cpu().numpy()
