"""Connectome container, loaders and null models.

A connectome here is a list of directed edges (pre → post) with a **signed integer synapse count**
(`Excitatory x Connectivity` in the Shiu et al. files: sign from the presynaptic neuron's predicted
neurotransmitter, magnitude = number of synapses). Weights are frozen exactly as reconstructed.

FlyWire root IDs are ~7.2e17 (> 2^53): they are kept as int64 end-to-end and must never pass through
floats, JSON numbers or JavaScript numbers.

Loaders are registered by name so that a second dataset (MaleCNS, Phase 8) can be dropped in.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from flybrain import data_manifest

EXPECTED = {
    # name: (neurons, edge rows)  — verified against the published files
    "flywire783": (138_639, 15_091_983),
    "flywire630": (127_400, None),
}


@dataclass(frozen=True)
class Connectome:
    name: str
    root_ids: np.ndarray  # int64 [N], index = neuron index used everywhere else
    pre: np.ndarray  # int32 [E], sorted ascending (presynaptic-sorted edge list)
    post: np.ndarray  # int32 [E]
    weight: np.ndarray  # int32 [E], signed synapse count
    annotations: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.root_ids.dtype != np.int64:
            raise TypeError("root_ids must be int64 (never float: IDs exceed 2^53)")
        for name in ("pre", "post", "weight"):
            arr = getattr(self, name)
            if arr.dtype != np.int32:
                raise TypeError(f"{name} must be int32")
            if arr.shape != self.pre.shape:
                raise ValueError("pre, post and weight must have the same length")
        if len(np.unique(self.root_ids)) != len(self.root_ids):
            raise ValueError("root_ids must be unique")
        if self.n_edges:
            if (
                self.pre.min() < 0
                or self.pre.max() >= self.n
                or self.post.min() < 0
                or self.post.max() >= self.n
            ):
                raise ValueError("edge index out of range")
            if np.any(np.diff(self.pre) < 0):
                raise ValueError("edges must be sorted by presynaptic index")

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return int(self.root_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.pre.shape[0])

    def index_of(self, ids: Iterable[int]) -> np.ndarray:
        """Neuron indices for FlyWire root IDs (raises KeyError listing any ID that is not in this connectome)."""
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        order = np.argsort(self.root_ids, kind="stable")
        pos = np.searchsorted(self.root_ids, ids_arr, sorter=order)
        pos = np.clip(pos, 0, self.n - 1)
        idx = order[pos]
        missing = ids_arr[self.root_ids[idx] != ids_arr]
        if missing.size:
            raise KeyError(
                f"{missing.size} root id(s) not in connectome '{self.name}': {missing[:5].tolist()}…"
            )
        return idx.astype(np.int64)

    def has_ids(self, ids: Iterable[int]) -> np.ndarray:
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        return np.isin(ids_arr, self.root_ids)

    def ids_of(self, idx: Iterable[int]) -> np.ndarray:
        return self.root_ids[np.asarray(list(idx), dtype=np.int64)]

    def out_adjacency(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Presynaptic-sorted CSR: (ptr int64 [N+1], post int32 [E], weight int32 [E])."""
        counts = np.bincount(self.pre, minlength=self.n).astype(np.int64)
        ptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])
        return ptr, self.post, self.weight

    def dense_weight_matrix(self) -> np.ndarray:
        """W[post, pre] as int64, duplicates summed. Only for small (test) connectomes."""
        if self.n > 5000:
            raise ValueError("dense matrix is for small connectomes only")
        w = np.zeros((self.n, self.n), dtype=np.int64)
        np.add.at(w, (self.post, self.pre), self.weight.astype(np.int64))
        return w

    def stats(self) -> dict:
        key = self.pre.astype(np.int64) * self.n + self.post.astype(np.int64)
        in_abs = np.bincount(self.post, weights=np.abs(self.weight).astype(np.float64), minlength=self.n)
        return {
            "name": self.name,
            "neurons": self.n,
            "edges": self.n_edges,
            "synapses": int(np.abs(self.weight).astype(np.int64).sum()),
            "self_loops": int(np.sum(self.pre == self.post)),
            "duplicate_edges": int(self.n_edges - np.unique(key).size),
            "negative_edges": int(np.sum(self.weight < 0)),
            "zero_edges": int(np.sum(self.weight == 0)),
            "max_abs_in_strength": int(in_abs.max()) if self.n_edges else 0,
            "max_abs_weight": int(np.abs(self.weight).max()) if self.n_edges else 0,
        }

    # ------------------------------------------------------------ edge selection
    def edge_mask(
        self, pre_idx: Iterable[int] | None = None, post_idx: Iterable[int] | None = None
    ) -> np.ndarray:
        """Boolean mask over edges with pre ∈ pre_idx (if given) and post ∈ post_idx (if given)."""
        mask = np.ones(self.n_edges, dtype=bool)
        if pre_idx is not None:
            sel = np.zeros(self.n, dtype=bool)
            sel[np.asarray(list(pre_idx), dtype=np.int64)] = True
            mask &= sel[self.pre]
        if post_idx is not None:
            sel = np.zeros(self.n, dtype=bool)
            sel[np.asarray(list(post_idx), dtype=np.int64)] = True
            mask &= sel[self.post]
        return mask

    def with_edges(self, keep: np.ndarray, *, suffix: str) -> Connectome:
        """Sub-connectome keeping only edges where `keep` is True (neurons and indices unchanged)."""
        keep = np.asarray(keep, dtype=bool)
        return replace(
            self,
            name=f"{self.name}:{suffix}",
            pre=self.pre[keep],
            post=self.post[keep],
            weight=self.weight[keep],
            meta={**self.meta, "derived": suffix},
        )

    def silence(self, idx: Iterable[int]) -> Connectome:
        """Published silencing (Shiu et al.): remove all *outgoing* synapses of the given neurons.

        The neuron still receives input and can still spike — it just has no effect downstream.
        """
        return self.with_edges(~self.edge_mask(pre_idx=idx), suffix="silenced")

    # ---------------------------------------------------------------- null models
    def shuffled(self, seed: int, *, preserve: np.ndarray | None = None) -> Connectome:
        """Degree-preserving shuffle: permute the postsynaptic column across edges.

        Preserved exactly: every presynaptic neuron's out-degree, outgoing weight multiset and sign
        (Dale's law), and every postsynaptic neuron's in-degree (number of incoming edges).
        Destroyed: who is connected to whom. Duplicate pre→post pairs may appear; the engine sums them.
        Edges where `preserve` is True are left untouched (stricter null models).
        """
        rng = np.random.default_rng(seed)
        post = self.post.copy()
        movable = np.ones(self.n_edges, dtype=bool) if preserve is None else ~np.asarray(preserve, dtype=bool)
        where = np.flatnonzero(movable)
        post[where] = post[where][rng.permutation(where.size)]
        tag = f"shuffle{seed}" + ("" if preserve is None else "+preserved")
        return replace(self, name=f"{self.name}:{tag}", post=post, meta={**self.meta, "derived": tag})


# ----------------------------------------------------------------------- loaders
_LOADERS: dict[str, Callable[..., Connectome]] = {}


def register_loader(name: str) -> Callable[[Callable[..., Connectome]], Callable[..., Connectome]]:
    def deco(fn: Callable[..., Connectome]) -> Callable[..., Connectome]:
        _LOADERS[name] = fn
        return fn

    return deco


def load_connectome(name: str, **kwargs) -> Connectome:
    if name not in _LOADERS:
        raise KeyError(f"unknown connectome '{name}'; known: {sorted(_LOADERS)}")
    return _LOADERS[name](**kwargs)


def _load_flywire(version: str) -> Connectome:
    import pyarrow.parquet as pq

    name = f"flywire{version}"
    comp_path = data_manifest.ensure(f"{name}_completeness", log=lambda _msg: None)
    conn_path = data_manifest.ensure(f"{name}_connectivity", log=lambda _msg: None)

    comp = pd.read_csv(comp_path, index_col=0)
    root_ids = comp.index.to_numpy()
    if root_ids.dtype != np.int64:
        raise TypeError(f"{comp_path.name}: root ids parsed as {root_ids.dtype}, expected int64")

    cols = [
        "Presynaptic_ID",
        "Postsynaptic_ID",
        "Presynaptic_Index",
        "Postsynaptic_Index",
        "Excitatory x Connectivity",
    ]
    table = pq.read_table(conn_path, columns=cols)
    pre_id = table.column(cols[0]).to_numpy()
    post_id = table.column(cols[1]).to_numpy()
    pre = table.column(cols[2]).to_numpy()
    post = table.column(cols[3]).to_numpy()
    weight = table.column(cols[4]).to_numpy()

    # The file's own index columns must agree with the completeness row order (this is THE neuron index).
    if not (np.array_equal(root_ids[pre], pre_id) and np.array_equal(root_ids[post], post_id)):
        raise ValueError(f"{name}: parquet index columns disagree with the completeness CSV row order")
    if np.abs(weight).max() > np.iinfo(np.int32).max:
        raise OverflowError("synapse count does not fit int32")

    if np.any(np.diff(pre) < 0):
        order = np.argsort(pre, kind="stable")
        pre, post, weight = pre[order], post[order], weight[order]

    conn = Connectome(
        name=name,
        root_ids=root_ids.astype(np.int64),
        pre=pre.astype(np.int32),
        post=post.astype(np.int32),
        weight=weight.astype(np.int32),
        meta={
            "source": "Shiu et al. 2024 preparation of FlyWire materialization " + version,
            "sha256": {
                "completeness": data_manifest.load_manifest()[f"{name}_completeness"].sha256,
                "connectivity": data_manifest.load_manifest()[f"{name}_connectivity"].sha256,
            },
        },
    )
    exp_n, exp_e = EXPECTED[name]
    if conn.n != exp_n or (exp_e is not None and conn.n_edges != exp_e):
        raise ValueError(f"{name}: got N={conn.n}, E={conn.n_edges}; expected N={exp_n}, E={exp_e}")
    return conn


@register_loader("flywire783")
def load_flywire783() -> Connectome:
    return _load_flywire("783")


@register_loader("flywire630")
def load_flywire630() -> Connectome:
    return _load_flywire("630")


# --------------------------------------------------------------------- synthetic
SYNTH_ID_BASE = 720_575_940_600_000_000  # looks like a FlyWire root id (> 2^53) on purpose


@register_loader("synthetic")
def synthetic(n: int = 1000, seed: int = 0) -> Connectome:
    """Small FlyWire-like connectome for CPU tests and CI.

    Random sparse wiring with Dale's-law signs, autapses, duplicate edges and strong negative weights, plus
    planted pathways mirroring the circuits we use on the real brain (so every code path runs without data):

      SYN_GRN (20) → SYN_IN (40) → SYN_MN9 (2)              "sugar → proboscis motor neuron"
      LPLC2 (30), LC4 (20) → DNp01 / Giant Fiber (2)        "looming → escape", with an inhibitory LC4 side path
    """
    if n < 200:
        raise ValueError("synthetic connectome needs n >= 200")
    rng = np.random.default_rng(seed)
    root_ids = SYNTH_ID_BASE + np.arange(n, dtype=np.int64) * 1009 + 17

    groups = {
        "SYN_GRN": np.arange(0, 20),
        "SYN_IN": np.arange(20, 60),
        "SYN_MN9": np.arange(60, 62),
        "LPLC2": np.arange(62, 92),
        "LC4": np.arange(92, 112),
        "SYN_LCINH": np.arange(112, 122),
        "DNp01": np.arange(122, 124),
    }
    planted = np.concatenate(list(groups.values()))
    inhibitory = rng.random(n) < 0.3
    inhibitory[planted] = False
    inhibitory[groups["SYN_LCINH"]] = True

    pre_l: list[np.ndarray] = []
    post_l: list[np.ndarray] = []
    cnt_l: list[np.ndarray] = []

    def connect(src: np.ndarray, dst: np.ndarray, p: float, lo: int, hi: int) -> None:
        mask = rng.random((src.size, dst.size)) < p
        s, d = np.nonzero(mask)
        pre_l.append(src[s])
        post_l.append(dst[d])
        cnt_l.append(rng.integers(lo, hi + 1, size=s.size))

    # background: every neuron projects to ~15 random targets; most edges are weak (1..12 synapses), some are
    # strong (15..60) so that activity really spreads through the recurrent network in tests
    out_deg = rng.poisson(15, size=n) + 1
    bg_pre = np.repeat(np.arange(n), out_deg)
    pre_l.append(bg_pre)
    post_l.append(rng.integers(0, n, size=bg_pre.size))
    weak = np.minimum(rng.geometric(0.35, size=bg_pre.size), 12)
    strong = rng.integers(15, 61, size=bg_pre.size)
    cnt_l.append(np.where(rng.random(bg_pre.size) < 0.15, strong, weak))
    # the planted interneurons and outputs also broadcast into the background
    connect(groups["SYN_IN"], np.arange(124, n), 0.03, 20, 60)
    connect(groups["DNp01"], np.arange(124, n), 0.05, 20, 60)

    connect(groups["SYN_GRN"], groups["SYN_IN"], 0.5, 4, 14)
    connect(groups["SYN_IN"], groups["SYN_MN9"], 0.6, 3, 10)
    connect(groups["SYN_GRN"], groups["SYN_MN9"], 0.3, 1, 4)
    connect(groups["LPLC2"], groups["DNp01"], 0.9, 2, 8)
    connect(groups["LC4"], groups["DNp01"], 0.9, 4, 12)
    connect(groups["LC4"], groups["SYN_LCINH"], 0.5, 3, 9)
    connect(groups["SYN_LCINH"], groups["DNp01"], 0.8, 5, 40)  # strong negative weights

    # autapses and explicit duplicate edges (the engine must sum duplicates and drop autapse input correctly)
    auto = rng.choice(n, size=12, replace=False)
    pre_l.append(auto)
    post_l.append(auto)
    cnt_l.append(rng.integers(5, 30, size=auto.size))
    dup = rng.choice(bg_pre.size, size=60, replace=False)
    pre_l.append(pre_l[0][dup])
    post_l.append(post_l[0][dup])
    cnt_l.append(rng.integers(1, 6, size=dup.size))

    pre = np.concatenate(pre_l).astype(np.int64)
    post = np.concatenate(post_l).astype(np.int64)
    count = np.concatenate(cnt_l).astype(np.int64)
    weight = np.where(inhibitory[pre], -count, count)
    order = np.argsort(pre, kind="stable")

    cell_type = np.full(n, "", dtype=object)
    for name, members in groups.items():
        cell_type[members] = name
    side = np.where(np.arange(n) % 2 == 0, "left", "right")
    annotations = pd.DataFrame(
        {
            "root_id": pd.array(root_ids, dtype="Int64"),
            "super_class": np.where(np.isin(np.arange(n), groups["DNp01"]), "descending", "central"),
            "cell_class": "",
            "cell_type": cell_type,
            "hemibrain_type": np.where(cell_type == "DNp01", "Giant Fiber", ""),
            "side": side,
            "top_nt": np.where(inhibitory, "gaba", "acetylcholine"),
        }
    )
    return Connectome(
        name=f"synthetic{n}-seed{seed}",
        root_ids=root_ids,
        pre=pre[order].astype(np.int32),
        post=post[order].astype(np.int32),
        weight=weight[order].astype(np.int32),
        annotations=annotations,
        meta={"source": "synthetic", "seed": seed, "groups": {k: v.tolist() for k, v in groups.items()}},
    )
