"""Phase 0 correctness experiment: reproduce the published sugar → MN9 result with our own LIF engine.

V1  connectome v630 (the one the paper used): 21 sugar GRNs driven at 10…200 Hz, 30 trials x 1 s, readout = MN9
    firing rate (mean of per-trial rates, as in the paper's `utils.get_rate`). Compared with the paper's Fig. 1d
    source data. Pre-registered pass criterion (DECISIONS.md F1): |ours - 65.7| <= 3 Hz at 100 Hz and every point
    within max(3 Hz, 3 * combined SE).
V2  v630, whole network: per-neuron firing rates at 100 / 200 Hz vs. the published example spike files.
V3  v783 (what dino-fly uses): same protocol, our own baseline numbers (nothing is published for 783).

    uv run --no-sync python -m experiments.sugar_mn9            # full run (~10 min on an RTX 5060 Laptop)
    uv run --no-sync python -m experiments.sugar_mn9 --plot-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common import C_ALT, C_FLY, C_REF, figure_path, load_result, setup_matplotlib, write_result

NAME = "sugar_mn9"
N_TRIALS = 30
T_RUN_STEPS = 10_000  # 1 s at dt = 0.1 ms
RATES_HZ = list(range(10, 201, 10))
PRECISE_TRIALS = 300  # extra precision for the 100 Hz point (10 batches of 30)
TOLERANCE_HZ = 3.0
REFERENCE = Path(__file__).resolve().parents[1] / "flybrain" / "reference" / "shiu2024_fig1d.json"
# Brian2-CPU ground truth for sugar @200 Hz on v783 reported in eonsystemspbc/fly-brain data/benchmark-results.csv
# (t_run = 100 s, n_run = 1: 1,704,969 spikes). Quoted as a sanity reference for total network activity.
FLYBRAIN_783_SPIKES_PER_S_200HZ = 17_050


def trial_seed(version: str, rate: int, trial: int) -> int:
    return int(version) * 1_000_000_000 + rate * 100_000 + trial


def simulate(version: str, device: str, rates: list[int], n_trials: int, precise_trials: int) -> dict:
    import torch

    from flybrain import neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome(f"flywire{version}")
    grn_set = getattr(neurons, f"SUGAR_GRN_{version}")
    mn9_set = getattr(neurons, f"MN9_{version}")
    grn, mn9 = conn.index_of(grn_set.ids), conn.index_of(mn9_set.ids)
    p = LIFParams()
    net = LIFNetwork(conn, p, batch_size=n_trials, device=device, dtype=torch.float32)
    drive = PoissonDrive(grn, n_trials, p.dt_ms, device=device)
    net.set_drive(drive)

    out: dict = {
        "connectome": conn.name,
        "connectome_stats": conn.stats(),
        "data_sha256": conn.meta["sha256"],
        "grn_ids": [str(i) for i in grn_set.ids],
        "mn9_ids": [str(i) for i in mn9_set.ids],
        "grn_rate_hz": rates,
        "mn9_published": {"rate_hz": [], "std_hz": []},
        "mn9_partner": {"rate_hz": [], "std_hz": []},
        "grn_mean_rate_hz": [],
        "total_spikes_per_trial": [],
        "active_neurons": [],
        "per_neuron_rate_hz": {},
    }

    def run(rate: int, seed_offset: int = 0) -> np.ndarray:
        net.reset()
        drive.set_rates(np.full(len(grn), float(rate)))
        drive.set_seeds(np.array([trial_seed(version, rate, seed_offset + t) for t in range(n_trials)]))
        return net.run(T_RUN_STEPS, record="counts").counts  # [N, trials]; 1 s → counts are rates in Hz

    for rate in rates:
        counts = run(rate)
        for key, idx in (("mn9_published", mn9[0]), ("mn9_partner", mn9[1])):
            out[key]["rate_hz"].append(float(counts[idx].mean()))
            out[key]["std_hz"].append(
                float(counts[idx].std())
            )  # population s.d. over trials, as in the paper
        out["grn_mean_rate_hz"].append(float(counts[grn].mean()))
        out["total_spikes_per_trial"].append(float(counts.sum() / n_trials))
        out["active_neurons"].append(int((counts.sum(axis=1) > 0).sum()))
        if rate in (100, 200):
            mean_rate = counts.mean(axis=1)
            active = np.flatnonzero(mean_rate > 0)
            out["per_neuron_rate_hz"][str(rate)] = {
                "flywire_id": [str(i) for i in conn.root_ids[active]],
                "rate_hz": mean_rate[active].round(4).tolist(),
            }
        print(
            f"[{conn.name}] {rate:3d} Hz: MN9 {out['mn9_published']['rate_hz'][-1]:6.2f} / "
            f"{out['mn9_partner']['rate_hz'][-1]:6.2f} Hz, {out['total_spikes_per_trial'][-1]:8.0f} spikes/trial, "
            f"{out['active_neurons'][-1]} active"
        )

    if precise_trials and 100 in rates:
        per_trial = np.concatenate(
            [run(100, seed_offset=1000 + k * n_trials)[mn9] for k in range(precise_trials // n_trials)],
            axis=1,
        )
        out["precise_100hz"] = {
            "n_trials": int(per_trial.shape[1]),
            "mn9_published_mean_hz": float(per_trial[0].mean()),
            "mn9_published_std_hz": float(per_trial[0].std()),
            "mn9_published_se_hz": float(per_trial[0].std(ddof=1) / np.sqrt(per_trial.shape[1])),
            "mn9_partner_mean_hz": float(per_trial[1].mean()),
        }
        print(f"[{conn.name}] precise 100 Hz: {out['precise_100hz']}")
    return out


def compare_with_published_curve(ours: dict, ref: dict) -> dict:
    res: dict = {"tolerance_hz": TOLERANCE_HZ, "neurons": {}}
    for key in ("mn9_published", "mn9_partner"):
        o, o_sd = np.array(ours[key]["rate_hz"]), np.array(ours[key]["std_hz"])
        r, r_sd = np.array(ref["neurons"][key]["rate_hz"]), np.array(ref["neurons"][key]["std_hz"])
        se = np.sqrt(o_sd**2 / N_TRIALS + r_sd**2 / ref["n_trials"])
        allowed = np.maximum(TOLERANCE_HZ, 3 * se)
        diff = o - r
        res["neurons"][key] = {
            "diff_hz": diff.round(3).tolist(),
            "allowed_hz": allowed.round(3).tolist(),
            "within": (np.abs(diff) <= allowed).tolist(),
            "rmse_hz": float(np.sqrt(np.mean(diff**2))),
            "max_abs_diff_hz": float(np.abs(diff).max()),
        }
    i100 = ours["grn_rate_hz"].index(100)
    d100 = res["neurons"]["mn9_published"]["diff_hz"][i100]
    res["at_100hz"] = {
        "ours_hz": ours["mn9_published"]["rate_hz"][i100],
        "published_hz": 65.7,
        "diff_hz": d100,
    }
    res["pass_100hz"] = bool(abs(d100) <= TOLERANCE_HZ)
    res["pass_all_points"] = bool(all(res["neurons"]["mn9_published"]["within"]))
    res["pass"] = bool(res["pass_100hz"] and res["pass_all_points"])
    return res


def compare_with_published_spikes(ours: dict) -> dict:
    """Per-neuron firing rates vs. the published example spike files (v630; 30 x 1 s at 100 and 200 Hz)."""
    import pandas as pd

    from flybrain import data_manifest

    out = {}
    for rate, key in ((100, "shiu_example_sugar_100hz"), (200, "shiu_example_sugar_200hz")):
        df = pd.read_parquet(data_manifest.ensure(key, log=lambda _m: None), columns=["trial", "flywire_id"])
        pub = (df.groupby("flywire_id").size() / N_TRIALS).rename("pub")
        mine = pd.Series(
            ours["per_neuron_rate_hz"][str(rate)]["rate_hz"],
            index=pd.Index(
                [int(i) for i in ours["per_neuron_rate_hz"][str(rate)]["flywire_id"]], dtype="int64"
            ),
            name="ours",
        )
        both = pd.concat([pub, mine], axis=1).fillna(0.0)
        a_pub, a_ours = set(pub.index), set(mine.index)
        strong = both[(both["pub"] >= 5) | (both["ours"] >= 5)]
        mn9_pub = {str(i): float(pub.get(int(i), 0.0)) for i in ours["mn9_ids"]}
        out[str(rate)] = {
            "published_active": len(a_pub),
            "ours_active": len(a_ours),
            "jaccard_active": len(a_pub & a_ours) / len(a_pub | a_ours),
            "pearson_r": float(np.corrcoef(both["pub"], both["ours"])[0, 1]),
            "pearson_r_log1p": float(np.corrcoef(np.log1p(both["pub"]), np.log1p(both["ours"]))[0, 1]),
            "median_abs_diff_hz_neurons_ge_5hz": float((strong["ours"] - strong["pub"]).abs().median()),
            "published_total_spikes_per_trial": float(len(df) / N_TRIALS),
            "ours_total_spikes_per_trial": float(
                ours["total_spikes_per_trial"][ours["grn_rate_hz"].index(rate)]
            ),
            "published_mn9_hz_this_file": mn9_pub,
            "scatter": {"pub": both["pub"].round(3).tolist(), "ours": both["ours"].round(3).tolist()},
        }
    return out


def plot(result: dict) -> None:
    plt = setup_matplotlib()
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    v630, v783 = result["v630"], result["v783"]
    x = v630["grn_rate_hz"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    for key, color, label in (
        ("mn9_published", C_FLY, "MN9 (paper's readout)"),
        ("mn9_partner", C_ALT, "MN9 partner"),
    ):
        ax.errorbar(
            x,
            ref["neurons"][key]["rate_hz"],
            yerr=ref["neurons"][key]["std_hz"],
            color=C_REF,
            lw=1,
            marker="o",
            ms=3,
            capsize=2,
            alpha=0.75 if key == "mn9_published" else 0.4,
            label="published (Brian2)" if key == "mn9_published" else None,
        )
        ax.errorbar(
            np.array(x) + 1.5,
            v630[key]["rate_hz"],
            yerr=v630[key]["std_hz"],
            color=color,
            lw=1.2,
            marker="s",
            ms=3,
            capsize=2,
            label=f"ours: {label}",
        )
    ax.set(
        xlabel="sugar GRN Poisson rate (Hz)",
        ylabel="firing rate (Hz), mean ± s.d. over 30 trials",
        title="A  connectome v630: ours vs. published Fig. 1d",
    )
    ax.legend(loc="upper left")

    ax = axes[1]
    sc = result["v630_vs_published_spikes"]["100"]
    lim = max(max(sc["scatter"]["pub"]), max(sc["scatter"]["ours"])) * 1.05
    ax.plot([0, lim], [0, lim], color=C_REF, lw=0.8, ls="--")
    ax.scatter(sc["scatter"]["pub"], sc["scatter"]["ours"], s=8, color=C_FLY, alpha=0.6, linewidths=0)
    ax.set(
        xlabel="published rate (Hz)",
        ylabel="our rate (Hz)",
        xlim=(0, lim),
        ylim=(0, lim),
        title=f"B  every active neuron @100 Hz  (r = {sc['pearson_r']:.4f}, Jaccard = {sc['jaccard_active']:.2f})",
    )
    ax.set_aspect("equal")

    ax = axes[2]
    ax.plot(
        x,
        ref["neurons"]["mn9_published"]["rate_hz"],
        color=C_REF,
        lw=1,
        marker="o",
        ms=3,
        alpha=0.6,
        label="published, v630",
    )
    ax.errorbar(
        x,
        v783["mn9_published"]["rate_hz"],
        yerr=v783["mn9_published"]["std_hz"],
        color=C_FLY,
        lw=1.2,
        marker="s",
        ms=3,
        capsize=2,
        label="ours, v783: MN9",
    )
    ax.errorbar(
        x,
        v783["mn9_partner"]["rate_hz"],
        yerr=v783["mn9_partner"]["std_hz"],
        color=C_ALT,
        lw=1.2,
        marker="s",
        ms=3,
        capsize=2,
        label="ours, v783: MN9 partner",
    )
    ax.set(
        xlabel="sugar GRN Poisson rate (Hz)",
        ylabel="firing rate (Hz)",
        title="C  connectome v783 (used by dino-fly)",
    )
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(figure_path(NAME))
    plt.close(fig)
    print(f"[figure] wrote {figure_path(NAME)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument(
        "--quick", action="store_true", help="4 rates, no precise run (smoke test; does not write results)"
    )
    args = ap.parse_args()

    if args.plot_only:
        result = load_result(NAME)
        if result is None:
            raise SystemExit("no results yet")
        plot(result)
        return 0

    rates = [50, 100, 150, 200] if args.quick else RATES_HZ
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    v630 = simulate("630", args.device, rates, N_TRIALS, 0 if args.quick else PRECISE_TRIALS)
    if args.quick:
        i = rates.index(100)
        print("quick check @100 Hz:", v630["mn9_published"]["rate_hz"][i], "published 65.7")
        return 0
    v783 = simulate("783", args.device, rates, N_TRIALS, PRECISE_TRIALS)
    spikes_cmp = compare_with_published_spikes(v630)
    result = {
        "protocol": {
            "n_trials": N_TRIALS,
            "t_run_s": 1.0,
            "dt_ms": 0.1,
            "dtype": "float32",
            "rates_hz": rates,
        },
        "criterion": compare_with_published_curve(v630, ref),
        "v630": {k: v for k, v in v630.items() if k != "per_neuron_rate_hz"},
        "v630_vs_published_spikes": spikes_cmp,
        "v783": {k: v for k, v in v783.items() if k != "per_neuron_rate_hz"},
        "v783_reference": {
            "flybrain_brian2_cpu_spikes_per_s_at_200hz": FLYBRAIN_783_SPIKES_PER_S_200HZ,
            "ours_spikes_per_s_at_200hz": v783["total_spikes_per_trial"][rates.index(200)],
        },
    }
    write_result(NAME, result)
    plot(result)
    c = result["criterion"]
    print(
        f"[criterion] 100 Hz: ours {c['at_100hz']['ours_hz']:.2f} vs 65.7 → diff {c['at_100hz']['diff_hz']:+.2f} Hz; "
        f"pass_100hz={c['pass_100hz']} pass_all_points={c['pass_all_points']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
