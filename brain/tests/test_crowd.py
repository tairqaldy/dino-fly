"""Crowd teaching mechanics on the synthetic connectome. The "human" here is a scripted test player."""

from __future__ import annotations

import numpy as np
from test_plasticity import make_learning_setup

from flybrain import dino_core as dc
from flybrain.connectome import Connectome
from flybrain.crowd import HumanRun, death_curriculum, human_jump_frames, observational_replay
from flybrain.transducer.looming import LoomingParams


def scripted_human(seed: int, lead_frames: int = 7, max_frames: int = 900) -> HumanRun:
    state, bits, actions = dc.create_initial_state(seed), 0, []
    while not state.crashed and state.frame < max_frames:
        ahead = [o for o in state.obstacles if o.x + o.width_px * dc.FPX >= dc.D["x"] * dc.FPX]
        jump = bool(ahead) and not state.jumping and ahead[0].x - (dc.D["x"] + dc.D["width"]) * dc.FPX <= state.speed * lead_frames
        b = int(jump or state.jumping)
        if b != bits:
            actions.append((state.frame, b))
            bits = b
        state = dc.step(state, b == 1, False)
    return HumanRun(seed=seed, actions=tuple(actions), frames=state.frame, score=dc.score(state))


def test_human_jump_frames_separates_successful_jumps():
    run = scripted_human(11)
    good, jumps, states = human_jump_frames(run)
    assert len(states) == run.frames and len(jumps) >= 3
    assert set(good) <= set(jumps) and len(good) >= 2  # this player clears several obstacles
    never = HumanRun(seed=11, actions=(), frames=400, score=0)
    assert human_jump_frames(never)[:2] == ([], [])


def test_death_curriculum_prefers_seeds_where_humans_die():
    runs = [HumanRun(5, (), 300, 40)] * 30 + [HumanRun(6, (), 300, 40)]
    picks = death_curriculum(runs, 2000, np.random.default_rng(0))
    assert set(picks) == {5, 6}
    assert picks.count(5) > 5 * picks.count(6)
    assert death_curriculum([], 10, np.random.default_rng(0)) == []


def test_observational_replay_delivers_dopamine_and_changes_gains(synth: Connectome):
    net, drive, learner, n_lc4, n_lplc2, gf = make_learning_setup(synth, 2, "none")  # no game-driven dopamine
    runs = [scripted_human(s) for s in (11, 12, 13)]
    stats = observational_replay(net, drive, n_lc4, n_lplc2, gf, runs, learner, looming=LoomingParams(version=0, gain_hz=150.0))
    assert stats["runs"] == 3 and stats["frames"] == sum(r.frames for r in runs)
    assert stats["gf_frames"] > 0 and stats["rewards"] + stats["punishments"] > 0
    assert float((learner.rule.gains - 1).abs().sum()) > 0  # imitation through dopamine, not through labels
