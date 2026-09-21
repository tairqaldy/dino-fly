"""Phase 4: can the fly learn to play better with its own learning mechanism?

Only KC→MBON gains change (dopamine-gated three-factor rule, `flybrain/plasticity.py`). Everything else is frozen.

Hypotheses (one run each, `--hypothesis`):
    h1  the learned synapses act only through the real wiring MBON → … → Giant Fiber. Nothing is added.
    h2  model assumption (NOT connectome): learned changes of mushroom-body output shift the looming gain within
        [G/2, 2G] (`flybrain.learn.SensoryGainParams`; valence by transmitter after Aso et al. 2014). The naive fly plays
        at G. Still no learned layer: the only learned quantities are the KC→MBON gains.

Protocol — fixed before any held-out game (git tag `prereg-phase4-<hypothesis>`, ledger entry):
  * generation 0 = naive gains (all 1). Each of 5 generations plays GAMES_PER_GEN = 64 fresh TRAIN seeds with
    plasticity ON (obstacle cleared → PAM burst scaled by jump timing; crash → PPL1 burst; visual context as fixed by
    `experiments/mb_drive.py`). Evaluation is with plasticity OFF on HELDOUT_100 — same seeds, same Poisson noise,
    every time — before training, after generations 1 and 3 and after the last one (compute budget: one night).
  * conditions: normal | shuffled_da (reward bursts at random moments — about as many as the naive fly earns, random
    magnitude — instead of after a cleared obstacle; punishments stay after the crash, because a PPL1 burst inside a
    running game ignites Kenyon-cell volleys in this model; evaluated after the last generation) | random_plasticity (final gains of `normal`, randomly re-assigned to synapses) | no_da (no dopamine:
    gains must stay exactly 1, which makes it identical to generation 0 — asserted, not re-played).
  * primary endpoint: held-out score, last generation vs. generation 0, paired by seed (bootstrap CI, Wilcoxon).
    "Learning" is claimed only if that difference is positive with a CI excluding 0 AND the last generation also beats
    shuffled_da (paired, CI excluding 0). Anything else is reported as "no learning effect".
  * context code and rate come from `experiments/mb_drive.py` (MBON-response rule, no game involved).
  * the learning rate comes from `--pilot` (below) through a synaptic criterion — never from a score.

Pilot (`--pilot`; DEV seeds for evaluation, TRAIN seeds ≥ 800,000 that are never reused; writes
`results/learning_pilot*.json`, which is not rendered into RESEARCH.md): plays PILOT_GAMES games at η₀ and sets
    η = η₀ · 0.5 / D,   D = mean depression (1 − g) of the 1 % most depressed plastic synapses, scaled to one generation,
i.e. the fastest-learning synapses lose half their weight in the first generation and the rule's range is not
exhausted at once. The pilot also smoke-tests every code path of the full run.

    uv run --no-sync python -u -m experiments.learning --hypothesis h1 --pilot
    uv run --no-sync python -u -m experiments.learning --hypothesis h1
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments import da_burst, mb_drive
from experiments.common import (
    C_ALT,
    C_CTRL,
    C_FLY,
    figure_path,
    load_result,
    require_preregistration,
    setup_matplotlib,
    write_result,
)
from experiments.naive_play import summarise
from flybrain import seeds as seedsets
from flybrain.config import repo_root
from flybrain.learn import Learner, SensoryGainParams
from flybrain.plasticity import PlasticityParams
from flybrain.play import BIO_MS_PER_FRAME, play_games
from flybrain.stats import paired_comparison
from flybrain.transducer.context import N_CONTEXTS, context_neurons
from flybrain.transducer.looming import FROZEN

NAMES = {"h1": "learning", "h2": "learning_h2"}
GAMES_PER_GEN = 64
BATCH = 32
EVAL_NOISE_SEED = 0
PILOT_ETA0 = 1e-7  # small enough that no synapse reaches g_min during the pilot
PILOT_GAMES = 32
PILOT_EVAL_SEEDS = 32
PILOT_TRAIN_OFFSET = 800_000
TARGET_TOP1_DEPRESSION_PER_GENERATION = 0.5
KEEP = ("score_stats", "cleared_stats", "jumps_per_game", "p_jump_per_approach", "p_cleared_given_jump",
        "frames_to_collision_at_jump_median", "false_jump_rate", "reaction_latency_bio_ms_median", "score")


def checkpoint_dir(name: str):
    path = repo_root() / "brain" / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def gain_stats(gains: np.ndarray) -> dict:
    dep = 1.0 - gains
    k = max(1, len(dep) // 100)
    top = np.sort(dep)[-k:]
    return {"gain_mean": float(gains.mean()), "gain_min": float(gains.min()), "gain_max": float(gains.max()),
            "fraction_of_gains_changed": float((np.abs(dep) > 1e-3).mean()), "fraction_at_g_min": float((gains <= 1e-6).mean()),
            "sum_abs_gain_change": float(np.abs(dep).sum()), "top1pct_mean_depression": float(top.mean())}


def calibrated_eta(pilot: dict) -> float:
    """η from the pilot's synaptic measurement (depression is linear in η and in the number of games while nothing clips)."""
    last = pilot["conditions"]["normal"][-1]
    if last["fraction_at_g_min"] > 0 or last["top1pct_mean_depression"] <= 0:
        raise SystemExit("the pilot cannot calibrate η (synapses clipped or nothing changed) — change PILOT_ETA0 and re-run it")
    per_generation = last["top1pct_mean_depression"] * GAMES_PER_GEN / pilot["protocol"]["games_per_generation"]
    return float(f"{pilot['protocol']['eta'] * TARGET_TOP1_DEPRESSION_PER_GENERATION / per_generation:.3g}")


def evaluated(gen: int, generations: int) -> bool:
    """`normal` is evaluated on the held-out seeds before training, after odd generations and after the last one."""
    return gen == 0 or gen % 2 == 1 or gen == generations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypothesis", choices=("h1", "h2"), default="h1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    name = NAMES[args.hypothesis] + ("_pilot" if args.pilot else "")
    if args.plot_only:
        plot(load_result(name), name)
        return 0

    import pandas as pd
    import torch

    from flybrain import data_manifest, neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conditions = ["normal", "shuffled_da", "random_plasticity", "no_da"]
    if args.pilot:
        eta, generations, games_per_gen = PILOT_ETA0, 1, PILOT_GAMES
        eval_seeds, train_base = list(seedsets.DEV_SEEDS[:PILOT_EVAL_SEEDS]), PILOT_TRAIN_OFFSET
    else:
        pilot = load_result(NAMES[args.hypothesis] + "_pilot")
        if pilot is None:
            raise SystemExit("run the pilot first (--pilot): it fixes the learning rate")
        eta, generations, games_per_gen = calibrated_eta(pilot), args.generations, GAMES_PER_GEN
        eval_seeds, train_base = list(seedsets.HELDOUT_100), 0
        require_preregistration(f"prereg-phase4-{args.hypothesis}", name, conditions)

    conn = load_connectome("flywire783")
    ix = {n: neurons.indices(conn, n) for n in ("LC4", "LPLC2", "GF", "KC", "MBON", "PAM", "PPL1")}
    ann = pd.read_csv(data_manifest.ensure("flywire_annotations", log=lambda _m: None), sep="\t", dtype={"root_id": "Int64"},
                      usecols=["root_id", "super_class", "top_nt"], low_memory=False)
    vpn_ids = ann.loc[ann["super_class"] == "visual_projection", "root_id"].astype("int64").to_numpy()
    vpn = conn.index_of(vpn_ids[np.isin(vpn_ids, conn.root_ids)])
    context = mb_drive.chosen_context()
    vpn = np.setdiff1d(vpn, np.concatenate([ix["LC4"], ix["LPLC2"]]))
    ctx = context_neurons(conn, vpn, ix["KC"], context)
    recipient_kc = ctx if context.level == "kc" else np.unique(conn.post[conn.edge_mask(pre_idx=ctx, post_idx=ix["KC"])])
    p = LIFParams()
    plasticity = PlasticityParams(eta=eta)

    # ---- H2 only: MBON valence by transmitter, and what a naive brain's MBONs do in each context (no game involved)
    h2: dict = {}
    if args.hypothesis == "h2":
        mbon_sorted = np.sort(ix["MBON"])
        nt_of = dict(zip(ann["root_id"].astype("int64"), ann["top_nt"].fillna(""), strict=True))
        nt = np.array([nt_of.get(int(conn.root_ids[i]), "") for i in mbon_sorted])
        valence = np.where(nt == "glutamate", 1, np.where(np.isin(nt, ("gaba", "acetylcholine")), -1, 0))
        # What do a naive brain's MBONs do in each class × proximity bin? Measured while the naive fly really plays (no
        # dopamine, no plasticity, no gain modulation) on TRAIN seeds ≥ 850,000 that nothing else uses — a first version
        # replayed never-jump approaches instead and under-estimated the rates during play, so the naive gain was not G.
        net = LIFNetwork(conn, p, batch_size=BATCH, device=args.device, chunk_steps=10)
        naive = Learner(conn, net, kc=ix["KC"], mbon=ix["MBON"], pam=ix["PAM"], ppl1=ix["PPL1"], context_vpns=ctx, plasticity=plasticity,
                        dopamine=da_burst.chosen_dopamine(), context=context, da_mode="none")
        naive.rule.enabled = False
        drive = PoissonDrive(np.concatenate([ix["LC4"], ix["LPLC2"], naive.extra_neurons]), BATCH, p.dt_ms, device=args.device)
        net.set_drive(drive)
        spikes, frames, inner = np.zeros((N_CONTEXTS, 2)), np.zeros(N_CONTEXTS), naive.frame

        def record(watch_counts) -> None:
            mb = np.asarray(watch_counts)[naive.rule._sl[1]]  # [n_mbon, B], rows in the order of np.sort(MBON)
            for col in np.flatnonzero(naive._ctx_now >= 0):
                spikes[naive._ctx_now[col]] += [mb[valence > 0, col].sum(), mb[valence < 0, col].sum()]
                frames[naive._ctx_now[col]] += 1
            inner(watch_counts)

        naive.frame = record
        play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"], list(seedsets.train_seeds(850_000, BATCH)), looming=FROZEN, noise_seed=99, learner=naive)
        naive_hz = spikes / np.maximum(frames, 1)[:, None] * (1000.0 / BIO_MS_PER_FRAME)  # [9, 2]
        for k in np.flatnonzero(frames == 0):  # a bin that never occurred (pterodactyls are rare): mean of the other classes, same proximity
            same = [j for j in range(N_CONTEXTS) if j % 3 == k % 3 and frames[j] > 0]
            naive_hz[k] = naive_hz[same].mean(axis=0) if same else 0.0
        h2 = {"sensory_gain": SensoryGainParams(), "mbon_valence": valence, "naive_mbon_hz": naive_hz}
        print(f"[h2] avoidance-type MBONs {int((valence > 0).sum())}, approach-type {int((valence < 0).sum())}; naive summed rates per context "
              f"(Hz) avoidance {np.round(naive_hz[:, 0], 1).tolist()} approach {np.round(naive_hz[:, 1], 1).tolist()}; frames per bin {frames.astype(int).tolist()}", flush=True)
        del net, drive, naive
        torch.cuda.empty_cache()

    def build(da_mode: str):
        net = LIFNetwork(conn, p, batch_size=BATCH, device=args.device, chunk_steps=10)
        learner = Learner(conn, net, kc=ix["KC"], mbon=ix["MBON"], pam=ix["PAM"], ppl1=ix["PPL1"], context_vpns=ctx,
                          plasticity=plasticity, dopamine=da_burst.chosen_dopamine(), context=context, da_mode=da_mode, **h2)
        drive = PoissonDrive(np.concatenate([ix["LC4"], ix["LPLC2"], learner.extra_neurons]), BATCH, p.dt_ms, device=args.device)
        net.set_drive(drive)
        return net, drive, learner

    def play(net, drive, learner, seeds, noise_seed):
        return play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"], list(seeds), looming=FROZEN, noise_seed=noise_seed, learner=learner)

    def evaluate(net, drive, learner) -> dict:
        mode, learner.da_mode, learner.rule.enabled = learner.da_mode, "none", False
        if args.hypothesis == "h2":
            learner.gain_log_sum, learner.gain_log_n = 0.0, 0
        rows = play(net, drive, learner, eval_seeds, EVAL_NOISE_SEED)
        learner.da_mode, learner.rule.enabled = mode, True
        s = summarise([r.to_json() for r in rows])
        out = {k: s[k] for k in KEEP}
        if args.hypothesis == "h2":
            out["mean_gain_octaves_with_obstacle_in_view"] = learner.gain_log_sum / max(1, learner.gain_log_n)
            learner.gain_log_sum, learner.gain_log_n = 0.0, 0
        return out

    result: dict = {"hypothesis": args.hypothesis, "pilot": args.pilot,
                    "protocol": {"games_per_generation": games_per_gen, "generations": generations, "batch": BATCH,
                                 "eval_seeds": "DEV_SEEDS[:32]" if args.pilot else "HELDOUT_100", "eval_noise_seed": EVAL_NOISE_SEED,
                                 "bio_ms_per_frame": BIO_MS_PER_FRAME, "transducer_gain_hz": FROZEN.gain_hz, "n_context_vpns": len(ctx),
                                 "n_recipient_kcs": len(recipient_kc), "context_level": context.level, "context_code": context.code,
                                 "context_rate_hz": context.rate_hz, "context_ramp_deg": context.ramp_deg, "eta": eta},
                    "conditions": {}}
    if h2:
        result["h2"] = {"naive_mbon_hz_avoidance_approach_per_context": h2["naive_mbon_hz"].tolist(),
                        "n_avoidance_mbons": int((h2["mbon_valence"] > 0).sum()), "n_approach_mbons": int((h2["mbon_valence"] < 0).sum())}
    final_gains = None
    # the pilot measures how fast synapses change in the normal condition, nothing else: it never plays an evaluation game
    for cond in ("normal",) if args.pilot else ("normal", "shuffled_da"):
        net, drive, learner = build("normal" if cond == "normal" else "shuffled")
        result.setdefault("plasticity", learner.describe())
        curve = []
        offset = train_base + (0 if cond == "normal" else 400_000)
        for gen in range(generations + 1):
            t0 = time.time()
            if gen > 0:
                play(net, drive, learner, seedsets.train_seeds(offset + (gen - 1) * games_per_gen, games_per_gen), gen)
            gains = learner.rule.gains.detach().cpu().numpy()
            entry = {"generation": gen, **gain_stats(gains), "rewards_so_far": learner.rewards, "punishments_so_far": learner.punishments,
                     "shuffled_bursts_so_far": learner.shuffled_bursts, "spike_totals_so_far": dict(learner.spike_totals)}
            if args.hypothesis == "h2" and gen > 0:
                # mean gain shift while this generation trained (in the pilot, where the synapses barely move, this is the
                # neutrality check of the H2 mapping: a naive brain must play at G, i.e. ≈ 0 octaves)
                entry["mean_gain_octaves_while_training"] = learner.gain_log_sum / max(1, learner.gain_log_n)
                learner.gain_log_sum, learner.gain_log_n = 0.0, 0
            # shuffled_da: generation 0 is the same brain as normal generation 0; only its last generation is evaluated
            if not args.pilot and ((cond == "normal" and evaluated(gen, generations)) or gen == generations):
                entry["heldout"] = evaluate(net, drive, learner)
            if args.pilot and args.hypothesis == "h2" and gen == 0:
                # neutrality check of the H2 mapping on DEV seeds: a naive brain must play at G (|shift| ≤ 0.05 octaves).
                # Only the gain shift is kept — the pilot does not look at scores.
                entry["naive_gain_octaves_dev"] = evaluate(net, drive, learner)["mean_gain_octaves_with_obstacle_in_view"]
                print(f"[pilot] naive H2 brain on DEV seeds: mean gain shift {entry['naive_gain_octaves_dev']:+.3f} octaves", flush=True)
            np.savez_compressed(checkpoint_dir(name) / f"{cond}_gen{gen:02d}.npz", **learner.rule.state())
            entry["wall_s"] = time.time() - t0
            curve.append(entry)
            ev = entry.get("heldout")
            print(f"[{cond}] gen {gen}: " + (f"score {ev['score_stats']['mean']:.1f} (median {ev['score_stats']['median']:.0f}), cleared "
                  f"{ev['cleared_stats']['mean']:.2f}; " if ev else "") + f"gains changed {entry['fraction_of_gains_changed']:.2%}, at g_min "
                  f"{entry['fraction_at_g_min']:.2%}, top-1% depression {entry['top1pct_mean_depression']:.3f}; spikes {entry['spike_totals_so_far']}; "
                  f"R {learner.rewards} P {learner.punishments} S {learner.shuffled_bursts}"
                  + (f"; gain while training {entry['mean_gain_octaves_while_training']:+.3f} oct" if "mean_gain_octaves_while_training" in entry else "")
                  + f"; {entry['wall_s']:.0f} s", flush=True)
            # not resumable, but a crash late in the run must not lose the generations already measured
            (checkpoint_dir(name) / "partial.json").write_text(json.dumps(result | {"conditions": result["conditions"] | {cond: curve}}), encoding="utf-8")
        result["conditions"][cond] = curve
        if cond == "normal":
            final_gains = learner.rule.gains.detach().cpu().numpy().copy()
        del net, drive, learner
        torch.cuda.empty_cache()

    if args.pilot:
        write_result(name, result)
        print(f"[pilot] η₀ = {PILOT_ETA0:g} → calibrated η = {calibrated_eta(load_result(name)):g} for {GAMES_PER_GEN} games per generation", flush=True)
        return 0

    net, drive, learner = build("none")
    learner.rule.randomise_like(final_gains, seed=7)
    result["conditions"]["random_plasticity"] = [{"generation": generations, "heldout": evaluate(net, drive, learner)}]
    learner.rule.load(np.ones_like(final_gains))
    # no dopamine = PAM / PPL1 cannot spike (the rule listens to their spikes, whoever caused them)
    net.ablate(np.concatenate([ix["PAM"], ix["PPL1"]]))
    play(net, drive, learner, seedsets.train_seeds(train_base + 790_000, BATCH), 1)
    unchanged = float((learner.rule.gains - 1).abs().sum())
    result["conditions"]["no_da"] = [{"generation": 0, "sum_abs_gain_change": unchanged, "training_games": BATCH,
                                      "note": "dopaminergic neurons silenced; unchanged gains ⇒ the same brain as generation 0 (not re-played)"}]
    write_result(name, result)
    (checkpoint_dir(name) / "README.json").write_text(json.dumps({"note": "gain vectors (float16) per generation; git-ignored"}) + "\n", encoding="utf-8")
    plot(load_result(name), name)
    if unchanged != 0.0:
        raise SystemExit(f"no_da changed the gains by {unchanged} — the rule must be inert without dopamine (result written, look at it)")
    return 0


def plot(r: dict | None, name: str) -> None:
    if r is None:
        raise SystemExit("no results yet")
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    normal = [c for c in r["conditions"]["normal"] if "heldout" in c]
    gens = [c["generation"] for c in normal]
    st = [c["heldout"]["score_stats"] for c in normal]
    axes[0].plot(gens, [s["mean"] for s in st], marker="o", ms=3, color=C_FLY, label="normal")
    axes[0].fill_between(gens, [s["mean_ci95"][0] for s in st], [s["mean_ci95"][1] for s in st], color=C_FLY, alpha=0.15)
    for cond, marker, color in (("shuffled_da", "s", C_ALT), ("random_plasticity", "x", C_CTRL)):
        for c in r["conditions"].get(cond, []):
            if "heldout" in c:
                s = c["heldout"]["score_stats"]
                axes[0].errorbar([c["generation"] + (0.08 if cond == "shuffled_da" else 0.16)], [s["mean"]],
                                 yerr=[[s["mean"] - s["mean_ci95"][0]], [s["mean_ci95"][1] - s["mean"]]], marker=marker, color=color, ms=5, capsize=2,
                                 label=cond.replace("_", " "))
    axes[0].set(xlabel="generation", ylabel="held-out score (mean, 95% CI; same 100 seeds)", title=f"A  learning curve ({r['hypothesis'].upper()})")
    axes[0].legend(fontsize=7)
    for cond, color in (("normal", C_FLY), ("shuffled_da", C_ALT)):
        curve = r["conditions"].get(cond, [])
        axes[1].plot([c["generation"] for c in curve], [c["top1pct_mean_depression"] for c in curve], marker="o", ms=3, color=color, label=f"{cond.replace('_', ' ')}: top 1 %")
        axes[1].plot([c["generation"] for c in curve], [c["fraction_of_gains_changed"] for c in curve], marker="o", ms=3, ls=":", color=color,
                     label=f"{cond.replace('_', ' ')}: fraction changed")
    axes[1].set(xlabel="generation", ylabel="depression (1 − gain) / fraction of synapses", title="B  how much the KC→MBON synapses changed")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path(name))
    plt.close(fig)


def render_report(r: dict) -> str:
    pl, pr = r["plasticity"], r["protocol"]
    name = NAMES[r["hypothesis"]]
    normal = [c for c in r["conditions"]["normal"] if "heldout" in c]
    first, last = normal[0], normal[-1]
    lines = [
        f"Plastic set: **{pl['n_plastic_synapses']:,} KC→MBON connections** ({pl['n_kc']:,} KCs, {pl['n_mbon']} MBONs; {pl['mbons_with_dan_input']} MBONs receive "
        f"PAM/PPL1 input = our compartment proxy). η = {pr['eta']:g} (synaptic pilot criterion), τ_e = {pl['params']['tau_e_ms']:.0f} ms, gains ∈ "
        f"[{pl['params']['g_min']}, {pl['params']['g_max']}]. Context: "
        + (f"delivered directly to {pr['n_recipient_kcs']} visual Kenyon cells (deviation, DECISIONS.md D13)" if pr.get("context_level") == "kc"
           else f"{pr['n_context_vpns']} KC-projecting visual neurons → {pr['n_recipient_kcs']} KCs")
        + f", code `{pr['context_code']}`, {pr['context_rate_hz']:.0f} Hz"
        + (f", ramp {pr['context_ramp_deg']:.0f}°" if pr["context_ramp_deg"] > 0 else "") + " (fixed by the context diagnostic above). "
        f"{pr['generations']} generations × {pr['games_per_generation']} training games.",
    ]
    if r["hypothesis"] == "h2":
        h = r["h2"]
        lines += ["", f"H2 mapping: {h['n_avoidance_mbons']} glutamatergic (avoidance-type) and {h['n_approach_mbons']} GABAergic / cholinergic (approach-type) "
                      f"MBONs; looming gain = G · 2^clip(A₊/Ā₊ − A₋/Ā₋, −1, 1), τ = {pl['sensory_gain']['tau_ms']:.0f} ms."]
    lines += [
        "",
        "| condition | generation | held-out score mean [95% CI] | median | obstacles cleared | P(jump / approach) | synapses changed | top-1 % depression |"
        + (" mean gain shift (octaves) |" if r["hypothesis"] == "h2" else ""),
        "|---|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if r["hypothesis"] == "h2" else ""),
    ]
    for cond in ("normal", "shuffled_da", "random_plasticity"):
        for c in r["conditions"].get(cond, []):
            if "heldout" not in c:
                continue
            h, st = c["heldout"], c["heldout"]["score_stats"]
            changed = f"{c['fraction_of_gains_changed']:.1%}" if "fraction_of_gains_changed" in c else "as normal, permuted"
            top = f"{c['top1pct_mean_depression']:.2f}" if "top1pct_mean_depression" in c else "—"
            lines.append(f"| {cond} | {c['generation']} | {st['mean']:.1f} [{st['mean_ci95'][0]:.1f}, {st['mean_ci95'][1]:.1f}] | {st['median']:.0f} | "
                         f"{h['cleared_stats']['mean']:.2f} | {h['p_jump_per_approach']:.2f} | {changed} | {top} |"
                         + (f" {h['mean_gain_octaves_with_obstacle_in_view']:+.2f} |" if r["hypothesis"] == "h2" else ""))
    nd = r["conditions"]["no_da"][0]
    lines += ["", f"no_da: after {nd['training_games']} training games without dopamine Σ|Δg| = {nd['sum_abs_gain_change']:g} — the rule is inert without "
                  "dopamine, so this brain is generation 0."]
    a, b = np.array(last["heldout"]["score"]), np.array(first["heldout"]["score"])
    pc = paired_comparison(a, b)
    lines += ["", f"**Generation {last['generation']} vs. generation 0 (same held-out seeds and noise):** Δ score = {a.mean() - b.mean():+.1f} "
                  f"(paired 95% CI [{pc['mean_diff_ci95'][0]:+.1f}, {pc['mean_diff_ci95'][1]:+.1f}], Wilcoxon p = {pc['wilcoxon_p']:.2g}). "
                  f"Dopamine events during training: {last['rewards_so_far']} rewards, {last['punishments_so_far']} punishments."]
    same = [f"generation {c['generation']}: {int((np.array(c['heldout']['score']) == a).sum())}" for c in normal[:-1]]
    same += [f"{cond}: {int((np.array(c['heldout']['score']) == a).sum())}" for cond in ("shuffled_da", "random_plasticity")
             for c in r["conditions"].get(cond, []) if "heldout" in c]
    lines.append(f"Held-out games with exactly the same score as in generation {last['generation']} (of {len(a)}): " + "; ".join(same) + ".")
    verdicts = [pc["mean_diff_ci95"][0] > 0]
    for cond in ("shuffled_da", "random_plasticity"):
        ctrl = [c for c in r["conditions"].get(cond, []) if "heldout" in c]
        if ctrl:
            pcc = paired_comparison(a, np.array(ctrl[-1]["heldout"]["score"]))
            lines.append(f"Last generation vs. {cond}: Δ = {a.mean() - np.mean(ctrl[-1]['heldout']['score']):+.1f} "
                         f"(paired 95% CI [{pcc['mean_diff_ci95'][0]:+.1f}, {pcc['mean_diff_ci95'][1]:+.1f}], Wilcoxon p = {pcc['wilcoxon_p']:.2g}).")
            if cond == "shuffled_da":
                verdicts.append(pcc["mean_diff_ci95"][0] > 0)
    lines += ["", "**Pre-declared verdict: " + ("learning effect** (better than generation 0 and better than shuffled dopamine)." if all(verdicts) else
                                                "no learning effect** (the criteria — better than generation 0 *and* better than shuffled dopamine, both with a "
                                                "95% CI excluding 0 — are not met)."),
              "", f"![learning curve](figures/{name}.png)", "",
              f"Source: `results/{name}.json`; gain vectors per generation in `brain/checkpoints/{name}/` (not committed)."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
