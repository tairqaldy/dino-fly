"""Coarse visual context → the visual projection neurons that synapse onto Kenyon cells. Fixed and hand-written.

The mushroom body can only associate outcomes with situations it is told about. In the connectome 265 visual
projection neurons (aMe12, MTe32, MTe30, LTe25, … — everything of super_class `visual_projection` with a direct synapse
onto a Kenyon cell; LC4 / LPLC2 have none) provide that visual input. The situation is described by

    obstacle class {small cactus, large cactus, pterodactyl}   and   its angular size θ (proximity)

and nothing is driven when no obstacle is in view. Codes (the 265 neurons are sorted by root ID):

    one_of_9     9 disjoint equal groups = class × proximity bin {θ<15°, 15–30°, ≥30°}; one group active.
    random_half  the same 9 contexts, each a fixed pseudo-random half of the neurons (seeded, overlapping — like odours,
                 which activate overlapping sets of glomeruli).
    class_only   one pseudo-random half per obstacle class; proximity is carried by the rate alone.

Rate: `rate_hz` while in view, or — with `ramp_deg` > 0 — rising with the obstacle's angular size,
rate_hz · min(θ / ramp_deg, 1), so that an obstacle entering the screen does not switch on a full-strength volley.

This is a deliberately crude, easily falsifiable stand-in for real visual features (DECISIONS.md D13). Code, ramp and
rate are fixed by `experiments/mb_drive.py` from what the drive does to the brain — it must make MBONs respond without
firing the Giant Fiber by itself — never by game score.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

CONTEXT_VERSION = 3
N_CLASSES, N_PROXIMITY = 3, 3
N_CONTEXTS = N_CLASSES * N_PROXIMITY
PROXIMITY_EDGES_DEG = (15.0, 30.0)
CODE_SEED = 783
CODES = ("one_of_9", "random_half", "class_only")


@dataclass(frozen=True)
class ContextParams:
    version: int = CONTEXT_VERSION
    code: str = "class_only"
    rate_hz: float = 100.0
    ramp_deg: float = 30.0
    min_kc_fraction: float = 0.05  # 0 → all 265 KC-projecting visual neurons
    # "vpn": the context drives visual projection neurons (rule 3 of the project, the intended design).
    # "kc":  DEVIATION (DECISIONS.md D13): the context drives the visual Kenyon cells themselves, one synapse further
    #        in, because every vpn-level drive either fires the Giant Fiber or abolishes its looming response.
    level: str = "vpn"


def kc_projecting_vpns(conn, vpn_idx: np.ndarray, kc_idx: np.ndarray, min_kc_fraction: float = 0.0) -> np.ndarray:
    """Visual projection neurons with at least one direct synapse onto a Kenyon cell, sorted by root ID.

    `min_kc_fraction` keeps only neurons that send at least that share of their output synapses to Kenyon cells: at
    0.05 these are 47 of the 265 neurons (aMe12, MTe32, MTe30, LTe25, …) carrying 82 % of all visual-projection → KC
    synapses; the other 218 mostly talk to the rest of the brain.
    """
    mask = conn.edge_mask(pre_idx=vpn_idx, post_idx=kc_idx)
    pre = np.unique(conn.pre[mask])
    if min_kc_fraction > 0:
        w = np.abs(conn.weight).astype(np.int64)
        total = np.bincount(conn.pre, weights=w, minlength=len(conn.root_ids))
        to_kc = np.bincount(conn.pre[mask], weights=w[mask], minlength=len(conn.root_ids))
        pre = pre[to_kc[pre] >= min_kc_fraction * total[pre]]
    return pre[np.argsort(conn.root_ids[pre])]


def visual_kenyon_cells(conn, vpn_idx: np.ndarray, kc_idx: np.ndarray, min_kc_fraction: float = 0.05) -> np.ndarray:
    """Kenyon cells with a direct synapse from a (dedicated) KC-projecting visual neuron, sorted by root ID."""
    pre = kc_projecting_vpns(conn, vpn_idx, kc_idx, min_kc_fraction)
    kcs = np.unique(conn.post[conn.edge_mask(pre_idx=pre, post_idx=kc_idx)])
    return kcs[np.argsort(conn.root_ids[kcs])]


def context_neurons(conn, vpn_idx: np.ndarray, kc_idx: np.ndarray, params: ContextParams) -> np.ndarray:
    """The neurons the context transducer drives, for either level."""
    if params.level == "kc":
        return visual_kenyon_cells(conn, vpn_idx, kc_idx, params.min_kc_fraction)
    if params.level == "vpn":
        return kc_projecting_vpns(conn, vpn_idx, kc_idx, params.min_kc_fraction)
    raise ValueError(f"unknown context level {params.level!r}")


def context_index(obstacle_type: int, theta_deg: float) -> int:
    """Class × proximity bin, 0..8."""
    proximity = int(theta_deg >= PROXIMITY_EDGES_DEG[0]) + int(theta_deg >= PROXIMITY_EDGES_DEG[1])
    return obstacle_type * N_PROXIMITY + proximity


def group_of(n_neurons: int) -> np.ndarray:
    """one_of_9: group id (0..8) of each context neuron — contiguous equal blocks in root-ID order."""
    return (np.arange(n_neurons) * N_CONTEXTS) // n_neurons


@lru_cache(maxsize=8)
def membership(n_neurons: int, code: str = "random_half") -> np.ndarray:
    """Boolean [n_neurons, 9]: which context neurons are active in which class × proximity context."""
    if code == "one_of_9":
        m = group_of(n_neurons)[:, None] == np.arange(N_CONTEXTS)[None, :]
    elif code in ("random_half", "class_only"):
        m = np.random.default_rng(CODE_SEED).random((n_neurons, N_CONTEXTS)) < 0.5
        if code == "class_only":  # the class's first column, whatever the proximity
            m = np.repeat(m[:, ::N_PROXIMITY], N_PROXIMITY, axis=1)
    else:
        raise ValueError(f"unknown context code {code!r}")
    m.setflags(write=False)
    return m


def context_rates(n_neurons: int, obstacle_type: int | None, theta_deg: float, params: ContextParams) -> np.ndarray:
    rates = np.zeros(n_neurons)
    if obstacle_type is not None:
        scale = min(theta_deg / params.ramp_deg, 1.0) if params.ramp_deg > 0 else 1.0
        rates[membership(n_neurons, params.code)[:, context_index(obstacle_type, theta_deg)]] = params.rate_hz * max(scale, 0.0)
    return rates
