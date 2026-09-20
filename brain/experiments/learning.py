"""Phase 4: can the fly learn to play better with its own learning mechanism?

Only KC→MBON gains change (dopamine-gated three-factor rule, `flybrain/plasticity.py`). Everything else is frozen.

Per condition: generation 0 = naive gains (all 1). Each generation plays `GAMES_PER_GEN` TRAIN seeds with plasticity ON
(reward = obstacle cleared → PAM burst, scaled by jump timing; punishment = crash → PPL1 burst; coarse visual context →
KC-projecting visual neurons), then is evaluated with plasticity OFF on HELDOUT_100 — the same 100 seeds and the same
Poisson noise for every generation and every condition, never used for training.

Conditions:  normal | shuffled_da (bursts at random times) | random_plasticity (the final learned gains of `normal`,
randomly re-assigned to synapses) | no_da (no dopamine → gains cannot change; equals generation 0 by construction,
evaluated once as a check).

Pre-declared context calibration (not score-based): the context rate is the smallest of {20, 40, 60, 100, 150} Hz at
which ≥ 10 % of the Kenyon cells that receive direct visual input fire at least once in 500 ms.

    uv run --no-sync python -u -m experiments.learning --generations 8
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np

from experiments.common import C_ALT, C_CTRL, C_FLY, figure_path, load_result, setup_matplotlib, write_result
from experiments.naive_play import summarise
from flybrain import seeds as seedsets
from flybrain.config import repo_root
from flybrain.learn import Learner
from flybrain.plasticity import PlasticityParams
from flybrain.play import BIO_MS_PER_FRAME, play_games
from flybrain.transducer.context import ContextParams, kc_projecting_vpns
from flybrain.transducer.looming import FROZEN

NAME = "learning"
GAMES_PER_GEN = 128
BATCH = 32
EVAL_NOISE_SEED = 0
CONTEXT_RATES = (20.0, 40.0, 60.0, 100.0, 150.0)


def checkpoint_dir():
    path = repo_root() / "brain" / "checkpoints" / NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        plot(load_result(NAME))
        return 0

    import pandas as pd
    import torch

    from flybrain import data_manifest, neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome("flywire783")
    ix = {n: neurons.indices(conn, n) for n in ("LC4", "LPLC2", "GF", "KC", "MBON", "PAM", "PPL1")}
    ann = pd.read_csv(data_manifest.ensure("flywire_annotations", log=lambda _m: None), sep="\t", dtype={"root_id": "Int64"},
                      usecols=["root_id", "super_class"], low_memory=False)
    vpn_ids = ann.loc[ann["super_class"] == "visual_projection", "root_id"].astype("int64").to_numpy()
    vpn = conn.index_of(vpn_ids[np.isin(vpn_ids, conn.root_ids)])
    ctx = kc_projecting_vpns(conn, np.setdiff1d(vpn, np.concatenate([ix["LC4"], ix["LPLC2"]])), ix["KC"])
    recipient_kc = np.unique(conn.post[conn.edge_mask(pre_idx=ctx, post_idx=ix["KC"])])
    p = LIFParams()

    # ---- context calibration (biological sparseness criterion, no game involved)
    cal = []
    net = LIFNetwork(conn, p, batch_size=9, device=args.device, chunk_steps=10)
    drive = PoissonDrive(ctx, 9, p.dt_ms, device=args.device)
    net.set_drive(drive)
    from flybrain.transducer.context import group_of

    groups = group_of(len(ctx))
    for rate in CONTEXT_RATES:
        net.reset()
        drive.set_seeds(np.arange(9) + 31)
        drive.set_rates(np.stack([np.where(groups == g, rate, 0.0) for g in range(9)], axis=1))
        counts = net.run(5000, record="counts").counts
        frac = float((counts[recipient_kc] > 0).any(axis=1).mean())
        cal.append({"rate_hz": rate, "fraction_of_recipient_kcs_active": frac, "kc_spikes": int(counts[ix["KC"]].sum()), "mbon_spikes": int(counts[ix["MBON"]].sum())})
        print(f"[context] {rate:5.0f} Hz → {frac:.2%} of {len(recipient_kc)} recipient KCs fire; MBON spikes {cal[-1]['mbon_spikes']}", flush=True)
    chosen = next((c["rate_hz"] for c in cal if c["fraction_of_recipient_kcs_active"] >= 0.10), CONTEXT_RATES[-1])
    context = ContextParams(rate_hz=chosen)
    del net, drive
    torch.cuda.empty_cache()

    def build(da_mode: str):
        net = LIFNetwork(conn, p, batch_size=BATCH, device=args.device, chunk_steps=10)
        learner = Learner(conn, net, kc=ix["KC"], mbon=ix["MBON"], pam=ix["PAM"], ppl1=ix["PPL1"], context_vpns=ctx,
                          plasticity=PlasticityParams(), context=context, da_mode=da_mode)
        drive = PoissonDrive(np.concatenate([ix["LC4"], ix["LPLC2"], learner.extra_neurons]), BATCH, p.dt_ms, device=args.device)
        net.set_drive(drive)
        return net, drive, learner

    def evaluate(net, drive, learner) -> dict:
        learner.rule.enabled = False
        rows = play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"], list(seedsets.HELDOUT_100), looming=FROZEN,
                          noise_seed=EVAL_NOISE_SEED, learner=replace_da(learner, "none"))
        learner.rule.enabled = True
        s = summarise([r.to_json() for r in rows])
        return {k: s[k] for k in ("score_stats", "cleared_stats", "jumps_per_game", "p_jump_per_approach", "p_cleared_given_jump",
                                  "frames_to_collision_at_jump_median", "false_jump_rate", "reaction_latency_bio_ms_median", "score")}

    def replace_da(learner, mode):
        learner.da_mode_saved = learner.da_mode
        learner.da_mode = mode
        return learner

    result: dict = {"protocol": {"games_per_generation": GAMES_PER_GEN, "batch": BATCH, "heldout": "HELDOUT_100", "eval_noise_seed": EVAL_NOISE_SEED,
                                 "bio_ms_per_frame": BIO_MS_PER_FRAME, "transducer_gain_hz": FROZEN.gain_hz, "n_context_vpns": len(ctx),
                                 "n_recipient_kcs": len(recipient_kc)},
                    "context_calibration": {"sweep": cal, "chosen_rate_hz": chosen, "rule": "smallest rate with >= 10% of recipient KCs active in 500 ms"},
                    "conditions": {}}
    final_gains = None
    for cond in ("normal", "shuffled_da"):
        net, drive, learner = build("normal" if cond == "normal" else "shuffled")
        result.setdefault("plasticity", learner.describe())
        curve = []
        train_offset = 0 if cond == "normal" else 500_000
        for gen in range(args.generations + 1):
            t0 = time.time()
            if gen > 0:
                learner.da_mode = "normal" if cond == "normal" else "shuffled"
                play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"],
                           list(seedsets.train_seeds(train_offset + (gen - 1) * GAMES_PER_GEN, GAMES_PER_GEN)), looming=FROZEN, noise_seed=gen, learner=learner)
            gains = learner.rule.gains.detach().cpu().numpy()
            ev = evaluate(net, drive, learner)
            np.savez_compressed(checkpoint_dir() / f"{cond}_gen{gen:02d}.npz", **learner.rule.state())
            curve.append({"generation": gen, "heldout": ev, "gain_mean": float(gains.mean()), "gain_min": float(gains.min()),
                          "fraction_of_gains_changed": float((np.abs(gains - 1) > 1e-3).mean()), "sum_abs_gain_change": float(np.abs(gains - 1).sum()),
                          "rewards_so_far": learner.rewards, "punishments_so_far": learner.punishments, "wall_s": time.time() - t0})
            print(f"[{cond}] gen {gen}: held-out score {ev['score_stats']['mean']:.1f} (median {ev['score_stats']['median']:.0f}), cleared "
                  f"{ev['cleared_stats']['mean']:.2f}; gains changed {curve[-1]['fraction_of_gains_changed']:.2%}, Σ|Δg| {curve[-1]['sum_abs_gain_change']:.1f}; "
                  f"{curve[-1]['wall_s']:.0f} s", flush=True)
        result["conditions"][cond] = curve
        if cond == "normal":
            final_gains = learner.rule.gains.detach().cpu().numpy().copy()
        del net, drive, learner
        torch.cuda.empty_cache()

    net, drive, learner = build("none")
    learner.rule.randomise_like(final_gains, seed=7)
    result["conditions"]["random_plasticity"] = [{"generation": args.generations, "heldout": evaluate(net, drive, learner)}]
    learner.rule.load(np.ones_like(final_gains))
    play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"], list(seedsets.train_seeds(900_000, BATCH)), looming=FROZEN, noise_seed=1, learner=learner)
    result["conditions"]["no_da"] = [{"generation": 1, "heldout": evaluate(net, drive, learner),
                                      "sum_abs_gain_change": float((learner.rule.gains - 1).abs().sum())}]
    write_result(NAME, result)
    (checkpoint_dir() / "README.json").write_text(json.dumps({"note": "gain vectors (float16) per generation; git-ignored"}) + "\n", encoding="utf-8")
    plot(load_result(NAME))
    return 0


def plot(r: dict | None) -> None:
    if r is None:
        raise SystemExit("no results yet")
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for cond, color in (("normal", C_FLY), ("shuffled_da", C_ALT)):
        curve = r["conditions"].get(cond, [])
        gens = [c["generation"] for c in curve]
        mean = [c["heldout"]["score_stats"]["mean"] for c in curve]
        lo = [c["heldout"]["score_stats"]["mean_ci95"][0] for c in curve]
        hi = [c["heldout"]["score_stats"]["mean_ci95"][1] for c in curve]
        axes[0].plot(gens, mean, marker="o", ms=3, color=color, label=cond.replace("_", " "))
        axes[0].fill_between(gens, lo, hi, color=color, alpha=0.15)
        axes[1].plot(gens, [c["sum_abs_gain_change"] for c in curve], marker="o", ms=3, color=color, label=cond.replace("_", " "))
    for cond, marker in (("random_plasticity", "x"), ("no_da", "+")):
        for c in r["conditions"].get(cond, []):
            axes[0].scatter([c["generation"]], [c["heldout"]["score_stats"]["mean"]], marker=marker, color=C_CTRL, s=40, label=cond.replace("_", " "))
    axes[0].set(xlabel="generation", ylabel="held-out score (mean, 95% CI; same 100 seeds)", title="A  learning curve")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="generation", ylabel="Σ |gain − 1| over plastic synapses", title="B  how much the KC→MBON synapses changed")
    fig.tight_layout()
    fig.savefig(figure_path(NAME))
    plt.close(fig)


def render_report(r: dict) -> str:
    pl, cal = r["plasticity"], r["context_calibration"]
    normal = r["conditions"]["normal"]
    first, last = normal[0], normal[-1]
    lines = [
        f"Plastic set: **{pl['n_plastic_synapses']:,} KC→MBON connections** ({pl['n_kc']:,} KCs, {pl['n_mbon']} MBONs; {pl['mbons_with_dan_input']} MBONs receive "
        f"PAM/PPL1 input = our compartment proxy). Rule parameters: η = {pl['params']['eta']}, τ_e = {pl['params']['tau_e_ms']:.0f} ms, gains ∈ [{pl['params']['g_min']}, {pl['params']['g_max']}]. "
        f"Context: {r['protocol']['n_context_vpns']} KC-projecting visual neurons → {r['protocol']['n_recipient_kcs']} KCs; rate {cal['chosen_rate_hz']:.0f} Hz ({cal['rule']}).",
        "",
        "| condition | generation | held-out score mean [95% CI] | median | obstacles cleared | Σ|Δg| | gains changed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cond in ("normal", "shuffled_da", "random_plasticity", "no_da"):
        for c in r["conditions"].get(cond, []):
            st = c["heldout"]["score_stats"]
            lines.append(f"| {cond} | {c['generation']} | {st['mean']:.1f} [{st['mean_ci95'][0]:.1f}, {st['mean_ci95'][1]:.1f}] | {st['median']:.0f} | "
                         f"{c['heldout']['cleared_stats']['mean']:.2f} | {c.get('sum_abs_gain_change', float('nan')):.1f} | "
                         f"{c.get('fraction_of_gains_changed', float('nan')):.1%} |")
    diff = last["heldout"]["score_stats"]["mean"] - first["heldout"]["score_stats"]["mean"]
    a, b = np.array(last["heldout"]["score"]), np.array(first["heldout"]["score"])
    from flybrain.stats import paired_comparison

    pc = paired_comparison(a, b)
    lines += [
        "",
        f"**Generation {last['generation']} vs. generation 0 on the same held-out seeds and noise:** Δ score = {diff:+.1f} "
        f"(paired 95% CI [{pc['mean_diff_ci95'][0]:+.1f}, {pc['mean_diff_ci95'][1]:+.1f}], Wilcoxon p = {pc['wilcoxon_p']:.2g}). "
        f"Rewards delivered: {last['rewards_so_far']}, punishments: {last['punishments_so_far']}.",
        "",
        "![learning curve](figures/learning.png)",
        "",
        "Source: `results/learning.json`; gain vectors per generation in `brain/checkpoints/learning/` (not committed).",
    ]
    return "\n".join(lines)


_ = replace

if __name__ == "__main__":
    raise SystemExit(main())
