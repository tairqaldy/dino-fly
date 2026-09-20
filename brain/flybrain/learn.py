"""Learning loop glue: context drive, dopamine bursts and the KC→MBON plasticity rule around `play_games`.

A *generation* is a gain vector (one float per plastic synapse). Training = playing TRAIN seeds with plasticity on;
evaluation = playing the fixed held-out seeds with plasticity off. Nothing else in the brain ever changes.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from flybrain.plasticity import KcMbonPlasticity, PlasticityParams
from flybrain.transducer.context import ContextParams, context_index, context_rates
from flybrain.transducer.dopamine import DopamineChannel, DopamineParams, timing_magnitude


class Learner:
    """Plugs into `play_games(..., learner=...)`. Drive order after LC4/LPLC2: [context VPNs, PAM, PPL1]."""

    def __init__(self, conn, net, *, kc, mbon, pam, ppl1, context_vpns, plasticity: PlasticityParams | None = None,
                 dopamine: DopamineParams | None = None, context: ContextParams | None = None, frame_ms: float = 10.0,
                 da_mode: str = "normal", da_seed: int = 0) -> None:
        if da_mode not in ("normal", "none", "shuffled"):
            raise ValueError("da_mode must be normal | none | shuffled")
        self.context_vpns, self.pam, self.ppl1 = (np.asarray(x, dtype=np.int64) for x in (context_vpns, pam, ppl1))
        self.extra_neurons = np.concatenate([self.context_vpns, self.pam, self.ppl1])
        self.rule = KcMbonPlasticity(conn, net, kc, mbon, np.concatenate([self.pam, self.ppl1]), plasticity, frame_ms)
        self.watch = self.rule.watch
        self.dopamine, self.context = dopamine or DopamineParams(), context or ContextParams()
        self.b = net.b
        self.reward = [DopamineChannel(self.dopamine) for _ in range(self.b)]
        self.punish = [DopamineChannel(self.dopamine) for _ in range(self.b)]
        self.da_mode = da_mode
        self._rng = np.random.default_rng(da_seed)
        self.rewards = self.punishments = 0

    def describe(self) -> dict:
        return self.rule.describe() | {"dopamine": asdict(self.dopamine), "context": asdict(self.context), "da_mode": self.da_mode,
                                       "n_context_vpns": len(self.context_vpns), "n_pam": len(self.pam), "n_ppl1": len(self.ppl1)}

    # ---- hooks called by play_games
    def start_game(self, col: int) -> None:
        self.rule.reset_columns(np.array([col]))
        self.reward[col], self.punish[col] = DopamineChannel(self.dopamine), DopamineChannel(self.dopamine)

    def extra_rates(self, games, views, tails) -> np.ndarray:
        n_ctx, n_pam, n_ppl1 = len(self.context_vpns), len(self.pam), len(self.ppl1)
        out = np.zeros((n_ctx + n_pam + n_ppl1, self.b))
        for col, g in enumerate(games):
            if g is None:
                continue
            view = views[col]
            if col not in tails and view is not None and view.obstacle_index >= 0:
                o = g.state.obstacles[view.obstacle_index]
                out[:n_ctx, col] = context_rates(n_ctx, context_index(o.type, view.theta_deg), self.context)
            # shuffled-DA ablation: a similar number of bursts, delivered at random moments
            if self.da_mode == "shuffled" and self._rng.random() < self._shuffle_p:
                (self.reward if self._rng.random() < 0.5 else self.punish)[col].trigger(1.0)
            out[n_ctx : n_ctx + n_pam, col] = self.reward[col].rate()
            out[n_ctx + n_pam :, col] = self.punish[col].rate()
        return out

    _shuffle_p = 1.0 / 150.0  # ≈ one burst per 150 frames, close to the event rate of the naive fly

    def frame(self, watch_counts: np.ndarray) -> None:
        self.rule.frame(watch_counts)

    def on_cleared(self, col: int, approach) -> None:
        if self.da_mode == "normal":
            jumped_at = approach.frames_to_collision_at_jump if approach is not None else None
            self.reward[col].trigger(timing_magnitude(jumped_at, self.dopamine))
            self.rewards += 1

    def on_crash(self, col: int) -> int:
        if self.da_mode == "normal":
            self.punish[col].trigger(1.0)
            self.punishments += 1
        return self.dopamine.burst_frames + 5  # frames the crashed game stays in its column (punishment tail)
