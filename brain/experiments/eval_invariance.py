"""Technical check (DEV seeds): does an evaluation depend on anything but the gain vector?

In the learning experiments generation 0 is evaluated on a fresh network, later generations on a network that has
just played hundreds of training games. If the held-out score of generation 1 differs from generation 0 we want to be
sure the difference comes from the learned synapses and not from the network's history (active-set order, leftover
state, random streams). Protocol, H1 set-up, 32 DEV seeds:

    A  fresh network, naive gains                          → evaluate
    B  same network after 64 training games (gains moved)  → evaluate
    C  same network, gains put back to exactly 1           → evaluate     must equal A game by game
    D  fresh network, gains of B loaded                    → evaluate     must equal B game by game
    E  fresh DENSE network (no active set), naive gains    → evaluate     must equal A: the lazily grown active set is exact
    F / G  Phase-1 set-up (no learner, no plastic path): active set vs. dense → must be equal

    uv run --no-sync python -u -m experiments.eval_invariance
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments import da_burst, learning, mb_drive
from experiments.common import load_result, write_result
from flybrain import seeds as seedsets
from flybrain.learn import Learner
from flybrain.plasticity import PlasticityParams
from flybrain.play import play_games
from flybrain.transducer.context import context_neurons
from flybrain.transducer.looming import FROZEN

NAME = "eval_invariance"
BATCH = 32
TRAIN_OFFSET = 870_000  # TRAIN seeds used by nothing else


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import pandas as pd

    from flybrain import data_manifest, neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome("flywire783")
    ix = {n: neurons.indices(conn, n) for n in ("LC4", "LPLC2", "GF", "KC", "MBON", "PAM", "PPL1")}
    ann = pd.read_csv(data_manifest.ensure("flywire_annotations", log=lambda _m: None), sep="\t", dtype={"root_id": "Int64"},
                      usecols=["root_id", "super_class"], low_memory=False)
    vpn_ids = ann.loc[ann["super_class"] == "visual_projection", "root_id"].astype("int64").to_numpy()
    vpn = np.setdiff1d(conn.index_of(vpn_ids[np.isin(vpn_ids, conn.root_ids)]), np.concatenate([ix["LC4"], ix["LPLC2"]]))
    context = mb_drive.chosen_context()
    ctx = context_neurons(conn, vpn, ix["KC"], context)
    eta = learning.calibrated_eta(load_result("learning_pilot"))
    p = LIFParams()
    dev_seeds = list(seedsets.DEV_SEEDS[:BATCH])

    def build(active_set: bool = True):
        net = LIFNetwork(conn, p, batch_size=BATCH, device=args.device, chunk_steps=10, active_set=active_set)
        learner = Learner(conn, net, kc=ix["KC"], mbon=ix["MBON"], pam=ix["PAM"], ppl1=ix["PPL1"], context_vpns=ctx,
                          plasticity=PlasticityParams(eta=eta), dopamine=da_burst.chosen_dopamine(), context=context)
        drive = PoissonDrive(np.concatenate([ix["LC4"], ix["LPLC2"], learner.extra_neurons]), BATCH, p.dt_ms, device=args.device)
        net.set_drive(drive)
        return net, drive, learner

    def play(net, drive, learner, seeds, noise_seed):
        return play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"], list(seeds), looming=FROZEN, noise_seed=noise_seed, learner=learner)

    def evaluate(net, drive, learner) -> dict[int, int]:
        mode, learner.da_mode, learner.rule.enabled = learner.da_mode, "none", False
        games = play(net, drive, learner, dev_seeds, 0)
        learner.da_mode, learner.rule.enabled = mode, True
        return {g.seed: g.score for g in games}

    net, drive, learner = build()
    a = evaluate(net, drive, learner)
    play(net, drive, learner, seedsets.train_seeds(TRAIN_OFFSET, 64), 1)
    trained = learner.rule.gains.detach().cpu().numpy().copy()
    b = evaluate(net, drive, learner)
    learner.rule.load(np.ones_like(trained))
    c = evaluate(net, drive, learner)
    del net, drive, learner
    net, drive, learner = build()
    learner.rule.load(trained)
    d = evaluate(net, drive, learner)
    del net, drive, learner
    net, drive, learner = build(active_set=False)
    e = evaluate(net, drive, learner)
    del net, drive, learner

    def phase1(active_set: bool) -> dict[int, int]:
        net = LIFNetwork(conn, p, batch_size=BATCH, device=args.device, chunk_steps=10, active_set=active_set)
        drive = PoissonDrive(np.concatenate([ix["LC4"], ix["LPLC2"]]), BATCH, p.dt_ms, device=args.device)
        net.set_drive(drive)
        games = play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"], dev_seeds, looming=FROZEN, noise_seed=0)
        return {g.seed: g.score for g in games}

    f, g = phase1(True), phase1(False)

    def same(x, y):
        return int(sum(x[s] == y[s] for s in dev_seeds))

    named = (("A_fresh_naive", a), ("B_trained", b), ("C_same_net_gains_reset", c), ("D_fresh_net_trained_gains", d), ("E_fresh_dense_naive", e),
             ("F_phase1_active_set", f), ("G_phase1_dense", g))
    result = {"protocol": {"seeds": f"DEV_SEEDS[:{BATCH}]", "training_games": 64, "eta": eta},
              "mean_score": {k: float(np.mean(list(v.values()))) for k, v in named},
              "games_identical": {"C_vs_A": same(c, a), "D_vs_B": same(d, b), "B_vs_A": same(b, a), "E_vs_A": same(e, a), "E_vs_C": same(e, c), "G_vs_F": same(g, f)},
              "n_games": len(dev_seeds), "fraction_of_gains_changed_by_training": float((np.abs(trained - 1) > 1e-3).mean())}
    print(result, flush=True)
    write_result(NAME, result)
    return 0


def render_report(r: dict) -> str:
    n, g, m = r["n_games"], r["games_identical"], r["mean_score"]
    ok = all(g[k] == n for k in ("C_vs_A", "D_vs_B", "E_vs_A", "G_vs_F"))
    return "\n".join([
        f"{r['protocol']['seeds']}, H1 set-up, {r['protocol']['training_games']} training games between A and B; identical = same final score, game by game.",
        "",
        "| comparison | must be identical because | games identical | mean scores |",
        "|---|---|---:|---|",
        f"| C vs. A | same gains (all 1), used vs. fresh network | {g['C_vs_A']} / {n} | {m['C_same_net_gains_reset']:.1f} vs. {m['A_fresh_naive']:.1f} |",
        f"| D vs. B | same trained gains, fresh vs. used network | {g['D_vs_B']} / {n} | {m['D_fresh_net_trained_gains']:.1f} vs. {m['B_trained']:.1f} |",
        f"| E vs. A | dense engine vs. lazily grown active set, naive gains | {g['E_vs_A']} / {n} | {m['E_fresh_dense_naive']:.1f} vs. {m['A_fresh_naive']:.1f} |",
        f"| G vs. F | the same for the Phase-1 set-up (no learner, no plastic path) | {g['G_vs_F']} / {n} | {m['G_phase1_dense']:.1f} vs. {m['F_phase1_active_set']:.1f} |",
        f"| B vs. A | (not required) trained vs. naive gains | {g['B_vs_A']} / {n} | {m['B_trained']:.1f} vs. {m['A_fresh_naive']:.1f} |",
        "",
        ("**All required identities hold: an evaluation depends on the gain vector only.**" if ok else
         "**Not all required identities hold — evaluations depend on the network's history.** Differences between generations in the learning tables "
         "must be read against this."),
        "", "Source: `results/eval_invariance.json`.",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
