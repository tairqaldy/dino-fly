"""Closed loop: many independent (game, brain) pairs as columns of one batched LIF simulation.

Per 60 Hz game frame:  game state → looming transducer → LPLC2 / LC4 Poisson rates → whole-brain LIF for
`bio_ms_per_frame` of biological time → Giant Fiber spikes → motor transducer → JUMP key → game step.

Results are independent of batching: every game uses the Poisson stream key (noise_seed, game_seed) and a trial
clock that restarts when its column is recycled.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from flybrain import dino_core as dc
from flybrain.transducer.looming import LoomingParams, obstacle_view, population_rates
from flybrain.transducer.motor import JumpMotor, MotorParams

BIO_MS_PER_FRAME = 10.0
MAX_FRAMES = 10_000
OBSTACLE_NAMES = ("CACTUS_SMALL", "CACTUS_LARGE", "PTERODACTYL")


def stream_key(noise_seed: int, game_seed: int) -> int:
    """64-bit Poisson stream key of one game."""
    return ((int(noise_seed) & 0xFFFFFFFF) << 32) | (int(game_seed) & 0xFFFFFFFF)


@dataclass
class Approach:
    """One obstacle coming into view: what the fly did about it."""

    obstacle_type: int
    size: int
    y_bottom_px: int
    speed_at_entry: int
    frame_in_view: int
    gf_spikes: int = 0
    frames_to_first_gf: int | None = None
    distance_at_first_gf_px: float | None = None
    theta_at_first_gf_deg: float | None = None
    jumped: bool = False
    frames_to_collision_at_jump: float | None = None
    outcome: str = "open"  # cleared | crashed | open


@dataclass
class GameResult:
    seed: int
    noise_seed: int
    frames: int
    score: int
    cleared: int
    crashed: bool
    censored: bool
    death_type: int
    jumps: int
    gf_spikes: int
    gf_spikes_ignored: int
    jumps_without_obstacle_in_view: int
    actions: list[tuple[int, int]] = field(default_factory=list)
    approaches: list[Approach] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


class _Game:
    def __init__(self, seed: int, noise_seed: int, motor: MotorParams) -> None:
        self.seed, self.noise_seed = seed, noise_seed
        self.state = dc.create_initial_state(seed)
        self.motor = JumpMotor(motor)
        self.jump_key = False
        self.bits = 0
        self.actions: list[tuple[int, int]] = []
        self.approaches: list[Approach] = []
        self.current: tuple[int, Approach] | None = None  # (spawn identity, approach)
        self.gf_spikes = 0
        self.false_jumps = 0

    def result(self, max_frames: int) -> GameResult:
        s = self.state
        if self.current is not None:
            self.current[1].outcome = "crashed" if s.crashed else "open"
        return GameResult(
            seed=self.seed,
            noise_seed=self.noise_seed,
            frames=s.frame,
            score=dc.score(s),
            cleared=s.cleared,
            crashed=s.crashed,
            censored=not s.crashed and s.frame >= max_frames,
            death_type=s.death_type,
            jumps=s.jumps,
            gf_spikes=self.gf_spikes,
            gf_spikes_ignored=self.motor.ignored,
            jumps_without_obstacle_in_view=self.false_jumps,
            actions=self.actions,
            approaches=self.approaches,
        )


def play_games(
    net,
    drive,
    n_lc4: int,
    n_lplc2: int,
    gf_watch: np.ndarray,
    seeds: Sequence[int],
    *,
    looming: LoomingParams,
    noise_seed: int = 0,
    motor: MotorParams | None = None,
    bio_ms_per_frame: float = BIO_MS_PER_FRAME,
    max_frames: int = MAX_FRAMES,
    on_result: Callable[[GameResult], None] | None = None,
    on_frame: Callable[[int, list], None] | None = None,
    learner=None,
) -> list[GameResult]:
    """Play every seed once. `drive` must drive [LC4 neurons…, LPLC2 neurons…] in that order; `gf_watch` = GF indices.

    With a `learner` (flybrain.learn.Learner) the drive continues with [context VPNs…, PAM…, PPL1…], the learner sees
    the per-frame spike counts of its watched neurons, is told about cleared obstacles and crashes, and a crashed game
    stays in its column for a short "punishment tail" during which only dopaminergic neurons are driven.
    """
    motor = motor or MotorParams()
    b = net.b
    watch = gf_watch if learner is None else np.concatenate([gf_watch, learner.watch])
    n_watch_gf = len(gf_watch)
    tails: dict[int, int] = {}
    steps_per_frame = round(bio_ms_per_frame / net.params.dt_ms)
    queue = list(seeds)
    games: list[_Game | None] = [None] * b
    results: list[GameResult] = []

    def start(col: int) -> None:
        if not queue:
            games[col] = None
            net.reset(columns=np.array([col]))  # an idle column must be silent: nothing of its last game may linger
            return
        seed = queue.pop(0)
        games[col] = _Game(seed, noise_seed, motor)
        net.reset(columns=np.array([col]))
        if learner is not None:
            learner.start_game(col)
        drive.set_seeds(np.array([stream_key(noise_seed, seed)], dtype=np.uint64), columns=np.array([col]))

    net.reset()
    for col in range(b):
        start(col)

    lc4_rate = np.zeros(b)
    lplc2_rate = np.zeros(b)
    frame_no = 0
    while any(g is not None for g in games):
        views = []
        for col, g in enumerate(games):
            if g is None:
                lc4_rate[col] = lplc2_rate[col] = 0.0
                views.append(None)
                continue
            view = obstacle_view(g.state, bio_ms_per_frame)
            views.append(view)
            if col in tails:  # punishment tail: the game is over, the senses are off, only dopamine is delivered
                lc4_rate[col] = lplc2_rate[col] = 0.0
                continue
            lc4, lplc2 = population_rates(view.theta_deg, view.theta_dot_deg_s, looming)
            lc4_rate[col], lplc2_rate[col] = float(lc4), float(lplc2)
        rates = [np.tile(lc4_rate, (n_lc4, 1)), np.tile(lplc2_rate, (n_lplc2, 1))]
        if learner is not None:
            rates.append(learner.extra_rates(games, views, tails))
            gain = learner.sensory_gain()  # H2 only (a documented model assumption); None under H1
            if gain is not None:
                rates[0] = rates[0] * gain
                rates[1] = rates[1] * gain
        drive.set_rates(np.concatenate(rates, axis=0))
        res = net.run(steps_per_frame, record=None, watch=watch)
        gf = res.watch_counts[:n_watch_gf].sum(axis=0)
        if learner is not None:
            seen = np.array(res.watch_counts[n_watch_gf:])
            seen[:, [g is None for g in games]] = 0  # columns without a game never teach
            learner.frame(seen)

        for col, g in enumerate(games):
            if g is None:
                continue
            if col in tails:
                tails[col] -= 1
                if tails[col] <= 0:
                    del tails[col]
                    result = g.result(max_frames)
                    results.append(result)
                    if on_result is not None:
                        on_result(result)
                    start(col)
                continue
            view, s = views[col], g.state
            n_gf = int(gf[col])
            g.gf_spikes += n_gf
            # --- bookkeeping per approach (identity of an obstacle = its gap draw + type, stable while it lives)
            if view.obstacle_index >= 0:
                o = s.obstacles[view.obstacle_index]
                ident = hash((o.type, o.gap, o.size, o.y_bottom_px))
                if g.current is None or g.current[0] != ident:
                    if g.current is not None:
                        g.current[1].outcome = "cleared"
                    g.current = (ident, Approach(o.type, o.size, o.y_bottom_px, s.speed, s.frame))
                    g.approaches.append(g.current[1])
                ap = g.current[1]
                if n_gf:
                    if ap.gf_spikes == 0:
                        ap.frames_to_first_gf = s.frame - ap.frame_in_view
                        ap.distance_at_first_gf_px = view.distance_px
                        ap.theta_at_first_gf_deg = view.theta_deg
                    ap.gf_spikes += n_gf
            elif g.current is not None:
                g.current[1].outcome = "cleared"
                g.current = None

            was_pressed = g.jump_key
            g.jump_key = g.motor.update(n_gf, airborne=s.jumping)
            if g.jump_key and not was_pressed:
                if view.obstacle_index < 0:
                    g.false_jumps += 1
                elif g.current is not None and not g.current[1].jumped:
                    o = s.obstacles[view.obstacle_index]
                    closing = (s.speed + o.speed_offset) / dc.FPX
                    front_gap = o.x / dc.FPX - (dc.D["x"] + dc.D["width"])
                    g.current[1].jumped = True
                    g.current[1].frames_to_collision_at_jump = front_gap / closing
            bits = 1 if g.jump_key else 0
            if bits != g.bits:
                g.actions.append((s.frame, bits))
                g.bits = bits
            g.state = dc.step(s, g.jump_key, False)
            if learner is not None:
                if g.state.cleared > s.cleared:
                    learner.on_cleared(col, g.current[1] if g.current is not None else None)
                if g.state.crashed:
                    tails[col] = learner.on_crash(col)
                    continue

            if g.state.crashed or g.state.frame >= max_frames:
                result = g.result(max_frames)
                results.append(result)
                if on_result is not None:
                    on_result(result)
                start(col)
        frame_no += 1
        if on_frame is not None:
            on_frame(frame_no, games)
    return results
