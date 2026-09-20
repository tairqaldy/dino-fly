"""KC→MBON plasticity: the only learning mechanism. Synthetic connectome, CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import spikes_to_dense

from flybrain import neurons
from flybrain.connectome import Connectome
from flybrain.learn import Learner
from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive
from flybrain.plasticity import KcMbonPlasticity, PlasticityParams
from flybrain.play import play_games
from flybrain.transducer.context import context_index, context_rates, group_of, kc_projecting_vpns
from flybrain.transducer.dopamine import DopamineChannel, DopamineParams, timing_magnitude
from flybrain.transducer.looming import LoomingParams

P = LIFParams()


def idx(synth: Connectome) -> dict[str, np.ndarray]:
    g = synth.meta["groups"]
    return {k: np.array(g[v]) for k, v in dict(kc="KCg", mbon="MBON01", pam="PAM01", ppl1="PPL101", vpn="aMe12").items()}


def drive_vpns(synth, net, b, rate=150.0, seed=3):
    i = idx(synth)
    drive = PoissonDrive(i["vpn"], b, P.dt_ms)
    drive.set_rates(np.full(len(i["vpn"]), rate))
    drive.set_seeds(np.arange(b) + seed)
    net.set_drive(drive)
    return drive


def test_plastic_path_with_unit_gains_equals_the_frozen_network(synth: Connectome):
    i = idx(synth)
    frozen = LIFNetwork(synth, batch_size=2, dtype=torch.float64)
    plastic = LIFNetwork(synth, batch_size=2, dtype=torch.float64)
    gains = plastic.set_plastic(synth.edge_mask(pre_idx=i["kc"], post_idx=i["mbon"]))
    assert gains is not None and gains.numel() > 100 and bool((gains == 1).all())
    drive_vpns(synth, frozen, 2)
    drive_vpns(synth, plastic, 2)
    a, b = frozen.run(1500, record="spikes"), plastic.run(1500, record="spikes")
    assert a.counts[i["mbon"]].sum() > 20  # the miniature mushroom body is really active
    assert np.array_equal(a.spikes, b.spikes)


def test_gains_scale_the_kc_to_mbon_drive(synth: Connectome):
    i = idx(synth)
    rates = []
    for gain in (1.0, 0.5, 0.0):
        net = LIFNetwork(synth, batch_size=1, dtype=torch.float64)
        g = net.set_plastic(synth.edge_mask(pre_idx=i["kc"], post_idx=i["mbon"]))
        g.fill_(gain)
        drive_vpns(synth, net, 1)
        res = net.run(2500, record="spikes")
        rates.append(int(spikes_to_dense(res.spikes, 2500, synth.n)[:, i["mbon"]].sum()))
    assert rates[0] > rates[1] > rates[2] == 0 or (rates[0] > rates[1] >= rates[2] and rates[2] < rates[0] / 3)


def test_three_factor_rule(synth: Connectome):
    i = idx(synth)
    net = LIFNetwork(synth, batch_size=2)
    dan = np.concatenate([i["pam"], i["ppl1"]])
    rule = KcMbonPlasticity(synth, net, i["kc"], i["mbon"], dan, PlasticityParams(eta=0.01), frame_ms=10.0)
    n_kc, n_mbon = len(i["kc"]), len(i["mbon"])
    counts = np.zeros((n_kc + n_mbon + len(dan), 2))
    counts[:10, 0] = 3  # ten KCs fire in column 0 only
    assert rule.frame(counts) == 0.0  # eligibility without dopamine changes nothing
    assert float(rule.e[:, 1].abs().sum()) == 0.0 and float(rule.e[:, 0].sum()) > 0
    counts[:] = 0
    counts[n_kc + n_mbon : n_kc + n_mbon + len(i["pam"]), 0] = 5  # PAM burst
    changed = rule.frame(counts)
    assert changed > 0
    g = rule.gains.cpu().numpy()
    pre, post = rule.pre_pos.cpu().numpy(), rule.post_pos.cpu().numpy()
    depressed = g < 1
    assert depressed.any() and np.all(pre[depressed] < 10)  # only synapses of the active KCs …
    assert np.all(post[depressed] < 3)  # … onto MBONs of the compartment PAM innervates (MBON01[:3])
    assert g.min() >= 0 and g.max() <= 1.0  # dopamine only depresses
    for _ in range(2000):  # clipping
        rule.e.fill_(50.0)
        rule.frame(counts)
    assert float(rule.gains.min()) == 0.0
    state = rule.state()
    assert state["gains"].dtype == np.float16 and len(state["gains"]) == rule.n_plastic
    rule.randomise_like(state["gains"], seed=1)
    assert sorted(rule.gains.cpu().numpy().tolist()) == sorted(state["gains"].astype(np.float32).tolist())
    rule.enabled = False
    rule.load(np.ones(rule.n_plastic))
    rule.e.fill_(50.0)
    assert rule.frame(counts) == 0.0 and float(rule.gains.min()) == 1.0
    rule.reset_columns()
    assert float(rule.e.abs().sum()) == 0.0


def test_dopamine_and_context_transducers():
    p = DopamineParams()
    assert timing_magnitude(8.0, p) == 1.0 and timing_magnitude(None, p) == 0.25
    assert 0.25 < timing_magnitude(11.0, p) < 1.0 and timing_magnitude(40.0, p) == 0.25
    ch = DopamineChannel(p)
    assert ch.rate() == 0.0
    ch.trigger(0.5)
    assert [ch.rate() for _ in range(p.burst_frames + 1)] == [50.0] * p.burst_frames + [0.0]
    assert [context_index(0, 5), context_index(0, 20), context_index(2, 45)] == [0, 1, 8]
    groups = group_of(265)
    assert groups.min() == 0 and groups.max() == 8 and np.all(np.diff(groups) >= 0)
    assert abs(np.bincount(groups).max() - np.bincount(groups).min()) <= 1
    from flybrain.transducer.context import ContextParams, membership

    r = context_rates(265, 1, 20.0, ContextParams(rate_hz=60.0, code="one_of_9", ramp_deg=0.0))  # class 1, mid → context 4
    assert set(np.unique(r)) == {0.0, 60.0} and np.all((r > 0) == (groups == 4))
    assert context_rates(265, None, 0.0, ContextParams()).sum() == 0
    # overlapping codes: every context drives a fixed, different, roughly half-sized subset
    m = membership(265, "random_half")
    assert m.shape == (265, 9) and np.array_equal(m, membership(265, "random_half"))
    assert np.all(np.abs(m.mean(axis=0) - 0.5) < 0.12) and len({c.tobytes() for c in m.T}) == 9
    r = context_rates(265, 1, 20.0, ContextParams(code="random_half", rate_hz=80.0, ramp_deg=0.0))
    assert np.all((r > 0) == m[:, 4]) and set(np.unique(r)) == {0.0, 80.0}
    # class_only: one code per obstacle class; the rate ramps with angular size up to ramp_deg
    mc = membership(265, "class_only")
    assert all(np.array_equal(mc[:, c * 3], mc[:, c * 3 + k]) for c in range(3) for k in range(3)) and len({c.tobytes() for c in mc.T}) == 3
    pr = ContextParams(code="class_only", rate_hz=100.0, ramp_deg=30.0)
    assert context_rates(265, 2, 15.0, pr).max() == 50.0 and context_rates(265, 2, 45.0, pr).max() == 100.0
    assert np.array_equal(context_rates(265, 2, 5.0, pr) > 0, context_rates(265, 2, 45.0, pr) > 0)


def test_context_neurons_at_both_levels(synth: Connectome):
    from flybrain.transducer.context import ContextParams, context_neurons, visual_kenyon_cells

    i = idx(synth)
    every = kc_projecting_vpns(synth, i["vpn"], i["kc"])
    dedicated = kc_projecting_vpns(synth, i["vpn"], i["kc"], min_kc_fraction=0.05)
    assert set(dedicated) <= set(every) <= set(i["vpn"]) and len(every) > 5
    assert np.all(np.diff(synth.root_ids[every]) > 0)  # sorted by root ID → the code is reproducible
    kcs = visual_kenyon_cells(synth, i["vpn"], i["kc"], 0.0)
    assert 0 < len(kcs) <= len(i["kc"]) and set(kcs) <= set(i["kc"]) and np.all(np.diff(synth.root_ids[kcs]) > 0)
    assert np.array_equal(context_neurons(synth, i["vpn"], i["kc"], ContextParams(level="kc", min_kc_fraction=0.0)), kcs)
    assert np.array_equal(context_neurons(synth, i["vpn"], i["kc"], ContextParams(level="vpn", min_kc_fraction=0.0)), every)
    with pytest.raises(ValueError):
        context_neurons(synth, i["vpn"], i["kc"], ContextParams(level="retina"))


def make_learning_setup(synth: Connectome, b: int, da_mode: str, context=None):
    i = idx(synth)
    lc4, lplc2, gf = (neurons.indices(synth, n) for n in ("LC4", "LPLC2", "GF"))
    vpn_all = np.flatnonzero((synth.annotations["super_class"] == "visual_projection").to_numpy())
    assert set(i["vpn"]) < set(vpn_all)  # LPLC2 / LC4 are visual projection neurons too …
    ctx = kc_projecting_vpns(synth, np.setdiff1d(vpn_all, np.concatenate([lc4, lplc2])), i["kc"])  # … but never context
    assert set(ctx) <= set(i["vpn"]) and len(ctx) > 5
    net = LIFNetwork(synth, P, batch_size=b, chunk_steps=10)
    learner = Learner(synth, net, kc=i["kc"], mbon=i["mbon"], pam=i["pam"], ppl1=i["ppl1"], context_vpns=ctx,
                      plasticity=PlasticityParams(eta=0.02), da_mode=da_mode, context=context)
    drive = PoissonDrive(np.concatenate([lc4, lplc2, learner.extra_neurons]), b, P.dt_ms)
    net.set_drive(drive)
    return net, drive, learner, len(lc4), len(lplc2), gf


@pytest.mark.parametrize("da_mode,expect_change", [("normal", True), ("none", False), ("shuffled", True)])
def test_learning_loop_changes_gains_only_through_dopamine(synth: Connectome, da_mode: str, expect_change: bool):
    net, drive, learner, n_lc4, n_lplc2, gf = make_learning_setup(synth, 3, da_mode)
    if da_mode == "none":
        # "no dopamine" = the dopaminergic neurons cannot spike: the network itself may drive them (endogenous dopamine),
        # and the rule listens to their spikes whoever caused them
        net.ablate(np.concatenate([learner.pam, learner.ppl1]))
    results = play_games(net, drive, n_lc4, n_lplc2, gf, [1, 2, 3, 4], looming=LoomingParams(version=0, gain_hz=150.0),
                         max_frames=350, learner=learner)
    assert len(results) == 4
    gains = learner.rule.gains.cpu().numpy()
    assert (np.abs(gains - 1).max() > 0) == expect_change
    assert gains.min() >= 0 and gains.max() <= 2
    if da_mode == "normal":
        assert learner.punishments == sum(r.crashed for r in results) and learner.punishments > 0
        assert learner.describe()["n_plastic_synapses"] == learner.rule.n_plastic
    assert learner.spike_totals["frames"] > 0 and learner.sensory_gain() is None  # H1: nothing touches the senses


def test_reward_is_scaled_by_the_jump_that_cleared_the_obstacle(synth: Connectome):
    """By the time an obstacle counts as passed the view has moved on to the next one; the reward must still be scaled
    by the timing of the jump over the obstacle that was cleared (it used to get the minimum magnitude)."""
    net, drive, learner, n_lc4, n_lplc2, gf = make_learning_setup(synth, 3, "normal")
    rewarded, original = [], learner.on_cleared
    learner.on_cleared = lambda col, approach: (rewarded.append(approach), original(col, approach))[1]
    play_games(net, drive, n_lc4, n_lplc2, gf, [1, 2, 3, 4, 5, 6], looming=LoomingParams(version=0, gain_hz=150.0), max_frames=900, learner=learner)
    assert len(rewarded) >= 1 and learner.rewards == len(rewarded)
    assert all(a is not None and a.jumped and a.frames_to_collision_at_jump is not None for a in rewarded)


def test_columns_without_a_game_are_silent_and_never_teach(synth: Connectome):
    """Found on the real connectome: a finished column kept a self-sustained Kenyon-cell volley going and, through
    endogenous dopamine, rewrote most synapses while it sat idle. Idle columns are reset and masked out."""
    net, drive, learner, n_lc4, n_lplc2, gf = make_learning_setup(synth, 3, "normal")
    seen = {"idle_spikes": 0, "idle_frames": 0, "busy_spikes": 0}
    original_rates, original_frame = learner.extra_rates, learner.frame

    def extra_rates(games, views, tails):
        seen["idle"] = [g is None for g in games]
        return original_rates(games, views, tails)

    def frame(watch_counts):
        counts = np.asarray(watch_counts)
        seen["idle_spikes"] += int(counts[:, seen["idle"]].sum())
        seen["idle_frames"] += int(sum(seen["idle"]))
        seen["busy_spikes"] += int(counts.sum())
        original_frame(watch_counts)

    learner.extra_rates, learner.frame = extra_rates, frame
    play_games(net, drive, n_lc4, n_lplc2, gf, [1, 2, 3, 4], looming=LoomingParams(version=0, gain_hz=150.0), max_frames=350, learner=learner)
    assert seen["idle_frames"] > 0 and seen["busy_spikes"] > 0  # 4 games in 3 columns: columns do go idle …
    assert seen["idle_spikes"] == 0  # … and then contribute nothing
    u, g = net.state_mv(np.arange(synth.n))  # after the last game every column is idle → the whole brain is at rest
    assert float(np.abs(u).max()) == 0.0 and float(np.abs(g).max()) == 0.0


def make_h2_learner(synth: Connectome, b: int = 2):
    from flybrain.learn import SensoryGainParams

    i = idx(synth)
    net = LIFNetwork(synth, P, batch_size=b, chunk_steps=10)
    n_mbon = len(i["mbon"])
    valence = np.where(np.arange(n_mbon) < n_mbon // 2, 1, -1)  # first half avoidance-type, second half approach-type
    naive = np.tile([[40.0, 60.0]], (9, 1))  # summed Hz per population, identical in all 9 contexts
    learner = Learner(synth, net, kc=i["kc"], mbon=i["mbon"], pam=i["pam"], ppl1=i["ppl1"], context_vpns=i["vpn"],
                      sensory_gain=SensoryGainParams(tau_ms=50.0), mbon_valence=valence, naive_mbon_hz=naive)
    return net, learner, valence, naive


def run_h2_frames(learner, valence, hz_avoid: float, hz_approach: float, ctx: int, frames: int = 400, seed: int = 0) -> float:
    """Feed Poisson MBON spike counts with the given summed population rates; return the mean log2 gain of column 0."""
    rng = np.random.default_rng(seed)
    n_kc, n_mbon, n_dan = len(learner.rule.kc), len(learner.rule.mbon), len(learner.rule.dan)
    octaves = []
    for _ in range(frames):
        counts = np.zeros((n_kc + n_mbon + n_dan, learner.b))
        for sign, hz in ((1, hz_avoid), (-1, hz_approach)):
            rows = n_kc + np.flatnonzero(valence == sign)
            counts[rows] = rng.poisson(hz * 0.010 / len(rows), size=(len(rows), learner.b))
        learner._ctx_now[:] = ctx
        learner.frame(counts)
        octaves.append(np.log2(learner.sensory_gain()[0]))
    return float(np.mean(octaves[100:]))


def test_h2_sensory_gain_is_neutral_for_a_naive_brain_and_bounded(synth: Connectome):
    _net, learner, valence, _ = make_h2_learner(synth)
    assert abs(run_h2_frames(learner, valence, 40.0, 60.0, ctx=4)) < 0.1  # naive MBON output → gain G on average
    learner.start_game(0)
    # (the clip at ±1 octave makes the noisy mean fall a little short of ±1)
    assert run_h2_frames(learner, valence, 40.0, 0.0, ctx=4) > 0.7  # approach-type response abolished → gain ≈ doubles
    learner.start_game(0)
    assert run_h2_frames(learner, valence, 0.0, 60.0, ctx=4) < -0.7  # avoidance-type response abolished → gain ≈ halves
    learner.start_game(0)
    assert run_h2_frames(learner, valence, 0.0, 0.0, ctx=-1) == 0.0  # nothing in view → no baseline → no modulation
    g = learner.sensory_gain()
    assert g.shape == (learner.b,) and np.all((g >= 0.5) & (g <= 2.0))
    with pytest.raises(ValueError):
        i = idx(synth)
        from flybrain.learn import SensoryGainParams

        Learner(synth, _net, kc=i["kc"], mbon=i["mbon"], pam=i["pam"], ppl1=i["ppl1"], context_vpns=i["vpn"], sensory_gain=SensoryGainParams())


def test_h2_gain_scales_the_looming_drive_in_the_play_loop(synth: Connectome):
    from flybrain.transducer.context import ContextParams

    # context off: in the small synthetic net the context volley alone can fire the GF
    net, drive, learner, n_lc4, n_lplc2, gf = make_learning_setup(synth, 2, "none", ContextParams(rate_hz=0.0))
    learner.sensory_gain = lambda: np.zeros(2)  # looming pathway switched off through the H2 hook …
    blind = play_games(net, drive, n_lc4, n_lplc2, gf, [1, 2], looming=LoomingParams(version=0, gain_hz=150.0), max_frames=350, learner=learner)
    learner.sensory_gain = lambda: None
    seeing = play_games(net, drive, n_lc4, n_lplc2, gf, [1, 2], looming=LoomingParams(version=0, gain_hz=150.0), max_frames=350, learner=learner)
    assert sum(r.jumps for r in blind) < sum(r.jumps for r in seeing)  # … removes the looming-evoked jumps
