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
    from flybrain.transducer.context import ContextParams

    r = context_rates(265, 4, ContextParams())
    assert set(np.unique(r)) == {0.0, 60.0} and np.all((r > 0) == (groups == 4))
    assert context_rates(265, None, ContextParams()).sum() == 0


def make_learning_setup(synth: Connectome, b: int, da_mode: str):
    i = idx(synth)
    lc4, lplc2, gf = (neurons.indices(synth, n) for n in ("LC4", "LPLC2", "GF"))
    vpn_all = np.flatnonzero((synth.annotations["super_class"] == "visual_projection").to_numpy())
    assert set(i["vpn"]) < set(vpn_all)  # LPLC2 / LC4 are visual projection neurons too …
    ctx = kc_projecting_vpns(synth, np.setdiff1d(vpn_all, np.concatenate([lc4, lplc2])), i["kc"])  # … but never context
    assert set(ctx) <= set(i["vpn"]) and len(ctx) > 5
    net = LIFNetwork(synth, P, batch_size=b, chunk_steps=10)
    learner = Learner(synth, net, kc=i["kc"], mbon=i["mbon"], pam=i["pam"], ppl1=i["ppl1"], context_vpns=ctx,
                      plasticity=PlasticityParams(eta=0.02), da_mode=da_mode)
    drive = PoissonDrive(np.concatenate([lc4, lplc2, learner.extra_neurons]), b, P.dt_ms)
    net.set_drive(drive)
    return net, drive, learner, len(lc4), len(lplc2), gf


@pytest.mark.parametrize("da_mode,expect_change", [("normal", True), ("none", False), ("shuffled", True)])
def test_learning_loop_changes_gains_only_through_dopamine(synth: Connectome, da_mode: str, expect_change: bool):
    net, drive, learner, n_lc4, n_lplc2, gf = make_learning_setup(synth, 3, da_mode)
    results = play_games(net, drive, n_lc4, n_lplc2, gf, [1, 2, 3, 4], looming=LoomingParams(version=0, gain_hz=150.0),
                         max_frames=350, learner=learner)
    assert len(results) == 4
    gains = learner.rule.gains.cpu().numpy()
    assert (np.abs(gains - 1).max() > 0) == expect_change
    assert gains.min() >= 0 and gains.max() <= 2
    if da_mode == "normal":
        assert learner.punishments == sum(r.crashed for r in results) and learner.punishments > 0
        assert learner.describe()["n_plastic_synapses"] == learner.rule.n_plastic
