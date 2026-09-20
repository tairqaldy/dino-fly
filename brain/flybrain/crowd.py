"""Crowd teaching: humans teach the fly — through dopamine only, never through supervised labels.

1. Observational replay. A stored human run is replayed frame by frame; the fly *watches* it: the game states of the
   human's run are fed through the same sensory transducers into the brain, the fly's Giant Fiber responds (its jumps
   do not steer the replayed game). Whenever the GF fires within ±`window` frames of a human jump that led to a
   cleared obstacle → reward (PAM burst). Whenever the GF fires although the human did not jump there and the human
   survived that obstacle → punishment (PPL1 burst). What the dopamine changes is left to the plasticity rule.
2. Death curriculum. Seeds on which many humans die are over-sampled in the fly's own training games.

Both are ablated against solo training on equal compute in `experiments/crowd_teaching.py`.

Known problem, found in Phase 4 and not yet resolved here: the punishment of (1) is a PPL1 burst *inside a running
game*, and in this model such bursts ignite self-sustained Kenyon-cell volleys (`experiments/kc_volley.py`, DECISIONS
D20). Before the replay is used on real human runs, check `learner.spike_totals["kc_volley_column_frames"]`; if volleys
appear, the punishment has to move (e.g. to the end of the replayed run) — a decision to log, not to make silently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from flybrain import dino_core as dc
from flybrain.play import BIO_MS_PER_FRAME
from flybrain.transducer.looming import LoomingParams, obstacle_view, population_rates


@dataclass(frozen=True)
class HumanRun:
    seed: int
    actions: tuple[tuple[int, int], ...]
    frames: int
    score: int


def human_jump_frames(run: HumanRun) -> tuple[list[int], list[int], list[dc.GameState]]:
    """(frames at which the human started a jump that was followed by a cleared obstacle, all jump frames, states)."""
    states: list[dc.GameState] = []
    dc.replay(run.seed, list(run.actions), run.frames, on_frame=states.append)
    jumps = [s.frame for prev, s in zip([dc.create_initial_state(run.seed), *states], states, strict=False) if s.jumps > prev.jumps]
    good = []
    for f in jumps:
        cleared_at_jump = states[f - 1].cleared
        horizon = states[min(f - 1 + 60, len(states) - 1)]
        if horizon.cleared > cleared_at_jump and not any(s.crashed for s in states[f - 1 : f + 59]):
            good.append(f)
    return good, jumps, states


def death_curriculum(runs: list[HumanRun], n: int, rng: np.random.Generator) -> list[int]:
    """Sample n training seeds, weighting every seed by (1 + number of human deaths on it)."""
    deaths = Counter(r.seed for r in runs)
    seeds = sorted(deaths)
    if not seeds:
        return []
    w = np.array([1 + deaths[s] for s in seeds], dtype=np.float64)
    return [int(s) for s in rng.choice(seeds, size=n, p=w / w.sum())]


def observational_replay(net, drive, n_lc4: int, n_lplc2: int, gf_watch: np.ndarray, runs: list[HumanRun], learner, *,
                         looming: LoomingParams, window: int = 6, bio_ms_per_frame: float = BIO_MS_PER_FRAME) -> dict:
    """Let the fly watch human runs (one run per brain column, processed in batches). Returns event counts."""
    b = net.b
    steps = round(bio_ms_per_frame / net.params.dt_ms)
    watch = np.concatenate([gf_watch, learner.watch])
    stats = {"runs": 0, "frames": 0, "rewards": 0, "punishments": 0, "gf_frames": 0}
    for start in range(0, len(runs), b):
        batch = runs[start : start + b]
        prepared = [human_jump_frames(r) for r in batch]
        net.reset()
        learner.rule.reset_columns()
        drive.set_seeds(np.array([r.seed for r in batch] + [0] * (b - len(batch)), dtype=np.uint64))
        for col in range(b):
            learner.start_game(col)
        longest = max(len(p[2]) for p in prepared)
        finished: set[int] = set(range(len(batch), b))
        for f in range(longest):
            lc4_rate, lplc2_rate = np.zeros(b), np.zeros(b)
            games, views = [None] * b, [None] * b
            for col, (_good, _jumps, states) in enumerate(prepared):
                if f < len(states) and not states[f].crashed:
                    view = obstacle_view(states[f], bio_ms_per_frame)
                    lc4, lplc2 = population_rates(view.theta_deg, view.theta_dot_deg_s, looming)
                    lc4_rate[col], lplc2_rate[col] = float(lc4), float(lplc2)
                    games[col], views[col] = _Shim(states[f]), view
                elif col not in finished:
                    # the run is over: silence its column — an idle brain must neither linger nor teach (play.py does the same)
                    finished.add(col)
                    net.reset(columns=np.array([col]))
            drive.set_rates(np.concatenate([np.tile(lc4_rate, (n_lc4, 1)), np.tile(lplc2_rate, (n_lplc2, 1)), learner.extra_rates(games, views, {})], axis=0))
            res = net.run(steps, record=None, watch=watch)
            seen = np.array(res.watch_counts[len(gf_watch) :])
            seen[:, sorted(finished)] = 0
            learner.frame(seen)
            gf = res.watch_counts[: len(gf_watch)].sum(axis=0)
            for col, (good, jumps, states) in enumerate(prepared):
                if col in finished or gf[col] == 0:
                    continue
                frame = states[f].frame
                stats["gf_frames"] += 1
                if any(abs(frame - j) <= window for j in good):
                    learner.reward[col].trigger(1.0)
                    stats["rewards"] += 1
                elif not any(abs(frame - j) <= window for j in jumps) and not any(s.crashed for s in states[f : f + 60]):
                    learner.punish[col].trigger(1.0)
                    stats["punishments"] += 1
            stats["frames"] += sum(f < len(p[2]) for p in prepared)
        stats["runs"] += len(batch)
    return stats


class _Shim:
    """Just enough of play._Game for Learner.extra_rates (which only reads `.state`)."""

    def __init__(self, state: dc.GameState) -> None:
        self.state = state
