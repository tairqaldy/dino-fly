"""Coarse visual context → the visual projection neurons that synapse onto Kenyon cells. Fixed and hand-written.

The mushroom body can only associate outcomes with situations it is told about. In the connectome 265 visual
projection neurons (aMe12, MTe32, MTe30, LTe25, … — everything of super_class `visual_projection` with a direct synapse
onto a Kenyon cell; LC4 / LPLC2 have none) provide that visual input. The transducer sorts them by root ID, splits
them into 9 equal groups and activates exactly one group per frame:

    context = obstacle class {small cactus, large cactus, pterodactyl} × proximity {far θ<15°, mid 15–30°, near ≥30°}

No group is active when nothing is in view. This is a deliberately crude, easily falsifiable stand-in for real visual
features (DECISIONS.md D13); the active rate is fixed by a sparseness criterion, not by game score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTEXT_VERSION = 1
N_CLASSES, N_PROXIMITY = 3, 3
PROXIMITY_EDGES_DEG = (15.0, 30.0)


@dataclass(frozen=True)
class ContextParams:
    version: int = CONTEXT_VERSION
    rate_hz: float = 60.0


def kc_projecting_vpns(conn, vpn_idx: np.ndarray, kc_idx: np.ndarray) -> np.ndarray:
    """Visual projection neurons with at least one direct synapse onto a Kenyon cell, sorted by root ID."""
    mask = conn.edge_mask(pre_idx=vpn_idx, post_idx=kc_idx)
    pre = np.unique(conn.pre[mask])
    return pre[np.argsort(conn.root_ids[pre])]


def context_index(obstacle_type: int, theta_deg: float) -> int:
    proximity = int(theta_deg >= PROXIMITY_EDGES_DEG[0]) + int(theta_deg >= PROXIMITY_EDGES_DEG[1])
    return obstacle_type * N_PROXIMITY + proximity


def group_of(n_neurons: int) -> np.ndarray:
    """Group id (0..8) of each context neuron: contiguous equal blocks in root-ID order."""
    return (np.arange(n_neurons) * (N_CLASSES * N_PROXIMITY)) // n_neurons


def context_rates(n_neurons: int, active_group: int | None, params: ContextParams) -> np.ndarray:
    rates = np.zeros(n_neurons)
    if active_group is not None:
        rates[group_of(n_neurons) == active_group] = params.rate_hz
    return rates
