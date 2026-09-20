"""Learning loop glue: context drive, dopamine bursts and the KC→MBON plasticity rule around `play_games`.

A *generation* is a gain vector (one float per plastic synapse). Training = playing TRAIN seeds with plasticity on;
evaluation = playing the fixed held-out seeds with plasticity off. Nothing else in the brain ever changes.

Two hypotheses about how the learned synapses reach behaviour (docs/RESEARCH.md):

    H1  only through the real wiring MBON → … → Giant Fiber. Nothing is added.
    H2  a documented MODEL ASSUMPTION, not connectome: learned changes of mushroom-body output shift the gain of the
        looming pathway (`SensoryGainParams`). The mapping is fixed and hand-written; the only learned quantities are
        still the KC→MBON gains, changed only by the dopamine-gated rule.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from flybrain.plasticity import KcMbonPlasticity, PlasticityParams
from flybrain.transducer.context import N_CONTEXTS, ContextParams, context_index, context_rates
from flybrain.transducer.dopamine import DopamineChannel, DopamineParams, timing_magnitude


KC_VOLLEY_SPIKES_PER_FRAME = 300  # the context drive alone produces ≈ 80 Kenyon-cell spikes per 10-ms frame


@dataclass(frozen=True)
class SensoryGainParams:
    """H2. With A = low-passed summed firing of the avoidance-type (+) and approach-type (−) MBONs and Ā = what a naive
    brain shows in the same sequence of contexts (same filter):

        looming gain = G · 2^clip(κ · (A₊/Ā₊ − A₋/Ā₋), −κ, +κ)

    so a naive brain plays at exactly G on average, abolishing the approach-type response doubles the gain (κ = 1
    octave) and abolishing the avoidance-type response halves it. Valence follows Aso et al. 2014 (eLife 3:e04580):
    glutamatergic MBONs promote avoidance, GABAergic and cholinergic MBONs promote approach.
    """

    tau_ms: float = 300.0
    max_octaves: float = 1.0
    min_baseline_hz: float = 1.0  # below this naive population rate a term is ignored (no division by ~0)


class Learner:
    """Plugs into `play_games(..., learner=...)`. Drive order after LC4/LPLC2: [context VPNs, PAM, PPL1]."""

    def __init__(self, conn, net, *, kc, mbon, pam, ppl1, context_vpns, plasticity: PlasticityParams | None = None,
                 dopamine: DopamineParams | None = None, context: ContextParams | None = None, frame_ms: float = 10.0,
                 da_mode: str = "normal", da_seed: int = 0, sensory_gain: SensoryGainParams | None = None,
                 mbon_valence: np.ndarray | None = None, naive_mbon_hz: np.ndarray | None = None) -> None:
        if da_mode not in ("normal", "none", "shuffled"):
            raise ValueError("da_mode must be normal | none | shuffled")
        self.context_vpns, self.pam, self.ppl1 = (np.asarray(x, dtype=np.int64) for x in (context_vpns, pam, ppl1))
        self.extra_neurons = np.concatenate([self.context_vpns, self.pam, self.ppl1])
        self.rule = KcMbonPlasticity(conn, net, kc, mbon, np.concatenate([self.pam, self.ppl1]), plasticity, frame_ms)
        self.watch = self.rule.watch
        self.dopamine, self.context = dopamine or DopamineParams(), context or ContextParams()
        self.b = net.b
        self.frame_ms = frame_ms
        self.reward = [DopamineChannel(self.dopamine) for _ in range(self.b)]
        self.punish = [DopamineChannel(self.dopamine) for _ in range(self.b)]
        self.da_mode = da_mode
        self._rng = np.random.default_rng(da_seed)
        self.rewards = self.punishments = self.shuffled_bursts = 0
        self.spike_totals = {"kc": 0, "mbon": 0, "dan": 0, "frames": 0, "kc_volley_column_frames": 0}
        # ---- H2 (optional)
        self.sensory_gain_params = sensory_gain
        if sensory_gain is not None:
            if mbon_valence is None or naive_mbon_hz is None:
                raise ValueError("H2 needs mbon_valence (per MBON, in the order of rule.mbon) and naive_mbon_hz [9, 2]")
            v = np.asarray(mbon_valence)
            if v.shape != self.rule.mbon.shape or np.asarray(naive_mbon_hz).shape != (N_CONTEXTS, 2):
                raise ValueError("mbon_valence / naive_mbon_hz have the wrong shape")
            self._pop = np.stack([v > 0, v < 0])  # [2, n_mbon]: avoidance-type, approach-type
            self.naive_mbon_hz = np.asarray(naive_mbon_hz, dtype=np.float64)
            self._decay = math.exp(-frame_ms / sensory_gain.tau_ms)
            self._act = np.zeros((2, self.b))
            self._base = np.zeros((2, self.b))
            self._ctx_now = np.full(self.b, -1)
            self.gain_log_sum = 0.0  # Σ log2(gain) over (frame, column) with an obstacle in view → mean modulation
            self.gain_log_n = 0

    def describe(self) -> dict:
        out = self.rule.describe() | {"dopamine": asdict(self.dopamine), "context": asdict(self.context), "da_mode": self.da_mode,
                                      "n_context_vpns": len(self.context_vpns), "n_pam": len(self.pam), "n_ppl1": len(self.ppl1)}
        if self.sensory_gain_params is not None:
            out |= {"sensory_gain": asdict(self.sensory_gain_params), "n_avoidance_mbons": int(self._pop[0].sum()),
                    "n_approach_mbons": int(self._pop[1].sum())}
        return out

    # ---- hooks called by play_games
    def start_game(self, col: int) -> None:
        self.rule.reset_columns(np.array([col]))
        self.reward[col], self.punish[col] = DopamineChannel(self.dopamine), DopamineChannel(self.dopamine)
        if self.sensory_gain_params is not None:
            self._act[:, col] = 0.0
            self._base[:, col] = 0.0
            self._ctx_now[col] = -1

    def extra_rates(self, games, views, tails) -> np.ndarray:
        n_ctx, n_pam, n_ppl1 = len(self.context_vpns), len(self.pam), len(self.ppl1)
        out = np.zeros((n_ctx + n_pam + n_ppl1, self.b))
        for col, g in enumerate(games):
            if g is None:
                continue
            view = views[col]
            ctx = -1
            if col not in tails and view is not None and view.obstacle_index >= 0:
                o = g.state.obstacles[view.obstacle_index]
                ctx = context_index(o.type, view.theta_deg)
                out[:n_ctx, col] = context_rates(n_ctx, o.type, view.theta_deg, self.context)
            if self.sensory_gain_params is not None:
                self._ctx_now[col] = ctx
            # shuffled-DA ablation: a similar number of bursts, delivered at random moments
            if self.da_mode == "shuffled" and self._rng.random() < self._shuffle_p:
                (self.reward if self._rng.random() < 0.5 else self.punish)[col].trigger(1.0)
                self.shuffled_bursts += 1
            out[n_ctx : n_ctx + n_pam, col] = self.reward[col].rate()
            out[n_ctx + n_pam :, col] = self.punish[col].rate()
        return out

    _shuffle_p = 1.0 / 150.0  # ≈ one burst per 150 frames, close to the event rate of the naive fly

    def frame(self, watch_counts: np.ndarray) -> None:
        self.rule.frame(watch_counts)
        counts = np.asarray(watch_counts)
        kc, mbon, dan = (counts[s] for s in self.rule._sl)
        t = self.spike_totals
        t["kc"] += int(kc.sum())
        t["mbon"] += int(mbon.sum())
        t["dan"] += int(dan.sum())
        t["frames"] += 1
        # self-sustained volleys of hundreds of Kenyon cells (seen a few frames after some crashes; docs/RESEARCH.md)
        t["kc_volley_column_frames"] += int((kc.sum(axis=0) > KC_VOLLEY_SPIKES_PER_FRAME).sum())
        if self.sensory_gain_params is not None:
            d = self._decay
            hz = (self._pop.astype(np.float64) @ mbon) * (1000.0 / self.frame_ms)  # [2, B] summed population rate
            self._act = self._act * d + hz * (1.0 - d)
            naive = np.where(self._ctx_now >= 0, self.naive_mbon_hz[np.maximum(self._ctx_now, 0)].T, 0.0)  # [2, B]
            self._base = self._base * d + naive * (1.0 - d)

    def sensory_gain(self) -> np.ndarray | None:
        """H2: multiplicative factor on the looming rates of every column (None under H1)."""
        p = self.sensory_gain_params
        if p is None:
            return None
        ok = self._base >= p.min_baseline_hz
        rel = np.where(ok, self._act / np.where(ok, self._base, 1.0), 1.0)
        octaves = np.clip(p.max_octaves * (rel[0] - rel[1]), -p.max_octaves, p.max_octaves)
        seen = self._ctx_now >= 0
        self.gain_log_sum += float(octaves[seen].sum())
        self.gain_log_n += int(seen.sum())
        return np.power(2.0, octaves)

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
