"""Fixed transducers and the closed game loop, on the synthetic connectome (CPU)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from flybrain import dino_core as dc
from flybrain import neurons
from flybrain.connectome import Connectome
from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive
from flybrain.play import GameResult, play_games, stream_key
from flybrain.transducer.looming import (
    EYE_OFFSET_PX,
    NOTHING_IN_VIEW,
    LoomingParams,
    looming_theta_deg,
    looming_theta_dot_deg_s,
    obstacle_view,
    population_rates,
    time_to_collision_ms,
)
from flybrain.transducer.motor import JumpMotor, MotorParams

PARAMS = LoomingParams(version=0, gain_hz=20.0)
PLAY_PARAMS = LoomingParams(version=0, gain_hz=150.0)  # the synthetic GF needs a strong drive


# ------------------------------------------------------------------------------------------------ looming stimulus
def test_looming_formula_and_its_derivative():
    for rv in (10.0, 40.0, 80.0):
        assert looming_theta_deg(rv, rv) == pytest.approx(90.0)  # at t = r/v the disk subtends 90°
        t = np.linspace(5 * rv, 0.6 * rv, 400)
        theta = np.radians(looming_theta_deg(t, rv))
        numeric = -np.gradient(theta, t) * 1e3  # rad/s (t counts down)
        analytic = np.radians(looming_theta_dot_deg_s(t, rv))
        assert np.allclose(numeric[2:-2], analytic[2:-2], rtol=2e-3)
        assert np.all(np.diff(looming_theta_deg(t, rv)) > 0)
        assert time_to_collision_ms(float(looming_theta_deg(3 * rv, rv)), rv) == pytest.approx(3 * rv)
    # θ depends on t only through t / (r/v)
    assert looming_theta_deg(30.0, 10.0) == pytest.approx(looming_theta_deg(240.0, 80.0))


def test_rate_functions_are_bounded_gated_and_shaped_as_documented():
    theta = np.linspace(0, 120, 200)
    lc4, lplc2 = population_rates(theta, np.full_like(theta, 500.0), PARAMS)
    assert lplc2.max() == pytest.approx(PARAMS.gain_hz, rel=1e-3)
    assert theta[np.argmax(lplc2)] == pytest.approx(42.0, abs=0.5)
    assert np.allclose(lc4, PARAMS.gain_hz * 500.0 / 3000.0)
    vel = np.array([-100.0, 0.0, 1500.0, 3000.0, 9000.0])
    lc4, lplc2 = population_rates(np.full(5, 42.0), vel, PARAMS)
    assert lc4.tolist() == pytest.approx([0.0, 0.0, 10.0, 20.0, 20.0])  # linear, capped, zero when not expanding
    assert lplc2.tolist() == pytest.approx([0.0, 0.0, 20.0, 20.0, 20.0])  # expansion-gated
    assert np.all(lc4 >= 0) and np.all(lplc2 >= 0)


# ------------------------------------------------------------------------------------------------ game geometry
def state_with(obstacles, **kw) -> dc.GameState:
    return replace(dc.with_obstacles(dc.create_initial_state(1), tuple(obstacles)), **kw)


def cactus(x_px: float, **kw) -> dc.Obstacle:
    base = dict(type=1, x=int(x_px * 1000), y_bottom_px=0, size=1, width_px=25, height_px=50, gap=10**6,
                speed_offset=0, following_created=True, passed=False)
    return dc.Obstacle(**{**base, **kw})


def test_view_geometry_matches_hand_computation():
    eye_x, eye_y = 50 + EYE_OFFSET_PX[0], EYE_OFFSET_PX[1]
    s = state_with([cactus(eye_x + 100.0)])
    view = obstacle_view(s, 10.0)
    theta = math.degrees(math.atan2(50 - eye_y, 100.0) - math.atan2(0 - eye_y, 100.0))
    assert view.obstacle_index == 0 and view.distance_px == pytest.approx(100.0)
    assert view.theta_deg == pytest.approx(theta)
    # analytic θ̇ equals the finite difference of θ along the approach (6 px per 10 ms frame)
    near = obstacle_view(state_with([cactus(eye_x + 100.0 - 0.006)]), 10.0)
    assert view.theta_dot_deg_s == pytest.approx((near.theta_deg - view.theta_deg) / 1e-5, rel=1e-3)
    assert view.tau_ms == pytest.approx(1e3 * view.theta_deg / view.theta_dot_deg_s)
    # biological time scaling: a longer biological frame means slower apparent expansion
    assert obstacle_view(s, 20.0).theta_dot_deg_s == pytest.approx(view.theta_dot_deg_s / 2)


def test_view_ignores_offscreen_and_passed_obstacles_and_tracks_the_nearest():
    assert obstacle_view(state_with([]), 10.0) is NOTHING_IN_VIEW
    assert obstacle_view(state_with([cactus(650.0)]), 10.0) is NOTHING_IN_VIEW  # not on screen yet
    assert obstacle_view(state_with([cactus(60.0)]), 10.0) is NOTHING_IN_VIEW  # already behind the eye
    view = obstacle_view(state_with([cactus(60.0), cactus(300.0), cactus(500.0)]), 10.0)
    assert view.obstacle_index == 1
    assert NOTHING_IN_VIEW.tau_ms == float("inf")


def test_no_spurious_expansion_burst_when_the_nearest_obstacle_changes():
    """θ̇ is analytic per obstacle: a new obstacle entering the view starts with a small θ̇, not a jump artefact."""
    far = obstacle_view(state_with([cactus(599.0)]), 10.0)
    assert 0 < far.theta_dot_deg_s < 10.0
    jumping = obstacle_view(state_with([cactus(300.0)], dino_y=60_000), 10.0)
    ducking = obstacle_view(state_with([cactus(300.0)], ducking=True), 10.0)
    assert jumping.theta_deg != ducking.theta_deg  # eye height matters


# ------------------------------------------------------------------------------------------------ motor
def test_motor_presses_next_frame_holds_and_ignores_spikes_in_flight():
    m = JumpMotor(MotorParams(hold_frames=3))
    assert [m.update(0, False)] == [False]
    pressed = [m.update(1, False)] + [m.update(1, True) for _ in range(4)]
    assert pressed == [True, True, True, False, False]
    assert m.triggered == 1 and m.ignored == 4
    assert m.update(2, False) is True and m.ignored == 5  # two spikes in one frame = one jump


def test_motor_delay():
    m = JumpMotor(MotorParams(hold_frames=2, delay_frames=2))
    assert [m.update(1, False), m.update(0, False), m.update(0, False), m.update(0, False)] == [False, False, True, True]


# ------------------------------------------------------------------------------------------------ closed loop
def make(synth: Connectome, b: int, *, ablate_gf: bool = False):
    p = LIFParams()
    lc4, lplc2, gf = (neurons.indices(synth, n) for n in ("LC4", "LPLC2", "GF"))
    net = LIFNetwork(synth, p, batch_size=b, chunk_steps=10)
    drive = PoissonDrive(np.concatenate([lc4, lplc2]), b, p.dt_ms)
    net.set_drive(drive)
    if ablate_gf:
        net.ablate(gf)
    return net, drive, len(lc4), len(lplc2), gf


def play(synth, b, seeds, **kw) -> dict[int, GameResult]:
    net, drive, n_lc4, n_lplc2, gf = make(synth, b, ablate_gf=kw.pop("ablate_gf", False))
    results = play_games(net, drive, n_lc4, n_lplc2, gf, seeds, looming=PLAY_PARAMS, max_frames=400, **kw)
    assert sorted(r.seed for r in results) == sorted(seeds)
    return {r.seed: r for r in results}


def test_synthetic_neuron_sets_resolve_like_the_real_ones(synth: Connectome):
    assert len(neurons.indices(synth, "GF")) == 2
    assert len(neurons.indices(synth, "LC4")) == 20 and len(neurons.indices(synth, "LPLC2")) == 30


def test_closed_loop_is_deterministic_and_independent_of_batching(synth: Connectome):
    seeds = [1, 2, 3, 4, 5]
    solo = play(synth, 1, seeds)
    batched = play(synth, 3, seeds)  # columns get recycled: 5 games on 3 columns
    for seed in seeds:
        a, b = solo[seed], batched[seed]
        assert (a.frames, a.score, a.jumps, a.gf_spikes, a.actions) == (b.frames, b.score, b.jumps, b.gf_spikes, b.actions)
    assert sum(r.jumps for r in solo.values()) > 0  # the planted looming → GF pathway really drives jumps
    assert any(ap.gf_spikes > 0 for r in solo.values() for ap in r.approaches)
    # the recorded action log replays to the same game in the pure engine
    r = solo[1]
    assert dc.replay(1, r.actions, 400).frame == r.frames
    assert stream_key(1, 2) != stream_key(2, 1)


def test_noise_seed_changes_the_run_and_gf_ablation_stops_all_jumps(synth: Connectome):
    base = play(synth, 2, [1, 2])
    other = play(synth, 2, [1, 2], noise_seed=7)
    assert any(base[s].actions != other[s].actions for s in (1, 2))
    silenced = play(synth, 2, [1, 2], ablate_gf=True)
    assert all(r.jumps == 0 and r.gf_spikes == 0 for r in silenced.values())
    never = dc.replay(1, [], 400)
    assert silenced[1].frames == never.frame and silenced[1].death_type == never.death_type
    assert silenced[1].to_json()["approaches"][0]["outcome"] == "crashed"
