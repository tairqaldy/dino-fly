"""Dopamine-gated plasticity at Kenyon cell → MBON synapses — the ONLY learning mechanism in dino-fly.

Three-factor local rule, evaluated once per game frame (the traces are ~100x slower than a frame):

    eligibility   e_km ← e_km · exp(-Δt/τ_e) + pre_k · (1 + post_m)       per plastic synapse k→m and brain column
    dopamine      DA_m = Σ_d  c_dm · spikes_d                              per MBON m, from the DANs that synapse onto it
    update        g_km ← clip(g_km − η · Σ_columns DA_m · e_km , g_min, g_max)       effective weight = w0 · g

`pre_k` / `post_m` are spike counts of the frame (post as a low-pass trace). Dopamine *depresses* synapses that were
recently active, as in the fly mushroom body; whether an event helps or hurts a behaviour is decided by which
dopaminergic neurons the reward / punishment transducer drives and by where they project — i.e. by the connectome:
`c_dm` is the (row-normalised) DAN→MBON synapse count, a proxy for compartment membership (synapse locations are not
part of the connectivity table we use; documented limitation).

Assumptions that are ours, not the fly's (see docs/DECISIONS.md): the additive `1 +` makes the rule work when the
MBON is silent (postsynaptic spiking is not required for MB plasticity; Hige et al. 2015) while still rewarding
pre/post coincidence; gains are shared by all brain columns of a batch ("one fly living B lives in parallel").
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from flybrain.connectome import Connectome


@dataclass(frozen=True)
class PlasticityParams:
    eta: float = 2e-4
    tau_e_ms: float = 1500.0
    tau_post_ms: float = 100.0
    g_min: float = 0.0
    g_max: float = 2.0
    use_post: bool = True


class KcMbonPlasticity:
    def __init__(self, conn: Connectome, net, kc: np.ndarray, mbon: np.ndarray, dan: np.ndarray,
                 params: PlasticityParams | None = None, frame_ms: float = 10.0) -> None:
        self.params = params or PlasticityParams()
        self.net, self.dev, self.dtype = net, net.device, net.dtype
        # sorted, because positions are looked up with searchsorted (PAM + PPL1 arrive concatenated, i.e. unsorted)
        self.kc, self.mbon, self.dan = (np.sort(np.asarray(x, dtype=np.int64)) for x in (kc, mbon, dan))
        mask = conn.edge_mask(pre_idx=self.kc, post_idx=self.mbon)
        self.edge_mask = mask
        self.gains = net.set_plastic(mask)
        edges = np.flatnonzero(mask)
        self.n_plastic = len(edges)
        dev = self.dev
        self.pre_pos = torch.as_tensor(np.searchsorted(self.kc, conn.pre[edges]), device=dev)
        self.post_pos = torch.as_tensor(np.searchsorted(self.mbon, conn.post[edges]), device=dev)
        # DAN → MBON "compartment" matrix from synapse counts, each MBON row normalised to sum 1
        c = np.zeros((len(self.mbon), len(self.dan)), dtype=np.float64)
        dm = conn.edge_mask(pre_idx=self.dan, post_idx=self.mbon)
        np.add.at(c, (np.searchsorted(self.mbon, conn.post[dm]), np.searchsorted(self.dan, conn.pre[dm])), np.abs(conn.weight[dm]))
        row = c.sum(axis=1, keepdims=True)
        self.c_dm = torch.as_tensor(np.divide(c, row, out=np.zeros_like(c), where=row > 0), dtype=self.dtype, device=dev)
        self.watch = np.concatenate([self.kc, self.mbon, self.dan])
        self._sl = (slice(0, len(self.kc)), slice(len(self.kc), len(self.kc) + len(self.mbon)), slice(len(self.kc) + len(self.mbon), None))
        self.decay_e = math.exp(-frame_ms / self.params.tau_e_ms)
        self.decay_post = math.exp(-frame_ms / self.params.tau_post_ms)
        self.e = torch.zeros(self.n_plastic, net.b, dtype=self.dtype, device=dev)
        self.post_trace = torch.zeros(len(self.mbon), net.b, dtype=self.dtype, device=dev)
        self.enabled = True
        self.total_update = 0.0
        self.n_updates = 0

    def describe(self) -> dict:
        return {"n_plastic_synapses": self.n_plastic, "n_kc": len(self.kc), "n_mbon": len(self.mbon), "n_dan": len(self.dan),
                "mbons_with_dan_input": int((self.c_dm.sum(dim=1) > 0).sum()), "params": asdict(self.params)}

    def reset_columns(self, columns: np.ndarray | None = None) -> None:
        if columns is None:
            self.e.zero_()
            self.post_trace.zero_()
        else:
            cols = torch.as_tensor(np.asarray(columns, dtype=np.int64), device=self.dev)
            self.e[:, cols] = 0
            self.post_trace[:, cols] = 0

    @torch.no_grad()
    def frame(self, watch_counts: np.ndarray | torch.Tensor, *, da_override: torch.Tensor | None = None) -> float:
        """Advance traces by one frame and apply the dopamine-gated update. Returns Σ|Δg| of this frame."""
        counts = torch.as_tensor(watch_counts, device=self.dev).to(self.dtype)
        kc, mbon, dan = (counts[s] for s in self._sl)
        post = 1.0 + self.post_trace if self.params.use_post else torch.ones_like(self.post_trace)
        self.e.mul_(self.decay_e).add_(kc.index_select(0, self.pre_pos) * post.index_select(0, self.post_pos))
        self.post_trace.mul_(self.decay_post).add_(mbon)
        if not self.enabled:
            return 0.0
        da = self.c_dm @ dan if da_override is None else da_override  # [n_mbon, B]
        if not bool((da > 0).any()):
            return 0.0
        delta = -self.params.eta * (da.index_select(0, self.post_pos) * self.e).sum(dim=1)
        before = self.gains.clone()
        self.gains.add_(delta).clamp_(self.params.g_min, self.params.g_max)
        change = float((self.gains - before).abs().sum())
        self.total_update += change
        self.n_updates += 1
        return change

    # ---- checkpoints: a generation IS its gain vector
    def state(self) -> dict:
        return {"gains": self.gains.detach().cpu().numpy().astype(np.float16), "edge_idx": np.flatnonzero(self.edge_mask).astype(np.int64)}

    def load(self, gains: np.ndarray) -> None:
        self.gains.copy_(torch.as_tensor(gains.astype(np.float32), device=self.dev).to(self.dtype))

    def randomise_like(self, reference: np.ndarray, seed: int) -> None:
        """Ablation: keep the *distribution* of learned gains but assign them to random synapses."""
        rng = np.random.default_rng(seed)
        self.load(rng.permutation(reference.astype(np.float32)))
