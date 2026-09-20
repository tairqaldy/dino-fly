"""Phase 4 calibration: can the mushroom body reach the escape circuit at all?

KC→MBON plasticity can only change behaviour through the real wiring MBON → … → Giant Fiber (there is no direct
MBON→GF synapse in the connectome). For each of the 35 MBON cell types we drive all its neurons at 100 Hz during the
standard looming stimulus (r/v = 40 ms, frozen transducer) and measure how the GF response changes against the
looming-only baseline; we also drive each type alone. Output: the "MBON → GF influence map".

    uv run --no-sync python -u -m experiments.mbon_influence
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.common import C_BAD, C_CTRL, C_FLY, figure_path, load_result, setup_matplotlib, write_result
from experiments.looming_gf import BASELINE_MS, CHUNK, DT_MS, stimulus
from flybrain.transducer.looming import FROZEN, population_rates

NAME = "mbon_influence"
RV_MS = 40.0
N_TRIALS = 40
MBON_RATE_HZ = 100.0


def run_block(net, drive, n_lc4, n_lplc2, n_extra, gf, *, looming: bool, extra_rate: float, seed_base: int) -> dict:
    t, theta, vel = stimulus(RV_MS)
    net.reset()
    drive.set_seeds(np.arange(net.b) + seed_base)
    first = np.full((len(gf), net.b), -1, dtype=np.int64)
    counts = np.zeros((len(gf), net.b), dtype=np.int64)
    for ms in range(len(t)):
        lc4, lplc2 = population_rates(theta[ms], vel[ms], FROZEN) if looming else (0.0, 0.0)
        on = extra_rate if ms >= BASELINE_MS else 0.0
        drive.set_rates(np.concatenate([np.full(n_lc4, float(lc4)), np.full(n_lplc2, float(lplc2)), np.full(n_extra, on)]))
        res = net.run(CHUNK, record=None, watch=gf)
        new = (first < 0) & (res.watch_first_step >= 0)
        first = np.where(new, res.watch_first_step + ms * CHUNK, first)
        counts += res.watch_counts
    any_first = np.where(first >= 0, first, np.iinfo(np.int64).max).min(axis=0)
    spiked = any_first < np.iinfo(np.int64).max
    return {
        "p_spike": float(spiked.mean()),
        "gf_spikes_per_trial": float(counts.sum(axis=0).mean()),
        "first_spike_ms_after_onset_median": float(np.median(any_first[spiked]) * DT_MS - BASELINE_MS) if spiked.any() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        plot(load_result(NAME))
        return 0

    import pandas as pd

    from flybrain import data_manifest, neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome("flywire783")
    ann = pd.read_csv(data_manifest.ensure("flywire_annotations", log=lambda _m: None), sep="\t",
                      dtype={"root_id": "Int64"}, usecols=["root_id", "cell_type", "top_nt"], low_memory=False)
    mbon = ann[ann["cell_type"].str.startswith("MBON", na=False)]
    lc4, lplc2, gf = (neurons.indices(conn, n) for n in ("LC4", "LPLC2", "GF"))
    p = LIFParams()
    net = LIFNetwork(conn, p, batch_size=N_TRIALS, device=args.device, chunk_steps=CHUNK)

    def block(extra: np.ndarray, **kw) -> dict:
        drive = PoissonDrive(np.concatenate([lc4, lplc2, extra]), N_TRIALS, p.dt_ms, device=args.device)
        net.set_drive(drive)
        return run_block(net, drive, len(lc4), len(lplc2), len(extra), gf, **kw)

    none = np.zeros(0, dtype=np.int64)
    baselines = [block(none, looming=True, extra_rate=0.0, seed_base=1000 * (k + 1)) for k in range(3)]
    base_p = float(np.mean([b["p_spike"] for b in baselines]))
    print(f"[baseline] P(GF spike) = {[b['p_spike'] for b in baselines]} (three independent noise sets)", flush=True)
    rows = []
    for cell_type, grp in sorted(mbon.groupby("cell_type"), key=lambda kv: kv[0]):
        idx = conn.index_of(grp["root_id"].astype("int64"))
        with_loom = block(idx, looming=True, extra_rate=MBON_RATE_HZ, seed_base=1000)  # same noise as baseline 0
        alone = block(idx, looming=False, extra_rate=MBON_RATE_HZ, seed_base=1000)
        rows.append({"cell_type": cell_type, "n_neurons": len(idx), "top_nt": grp["top_nt"].mode().iat[0] if grp["top_nt"].notna().any() else "",
                     "with_looming": with_loom, "alone": alone, "delta_p_spike": with_loom["p_spike"] - baselines[0]["p_spike"]})
        print(f"[{cell_type:>10}] n={len(idx)} ΔP={rows[-1]['delta_p_spike']:+.2f} P_alone={alone['p_spike']:.2f}", flush=True)

    noise_band = float(max(abs(b["p_spike"] - base_p) for b in baselines) + 2 * np.sqrt(base_p * (1 - base_p) / N_TRIALS))
    write_result(NAME, {
        "protocol": {"rv_ms": RV_MS, "n_trials": N_TRIALS, "mbon_rate_hz": MBON_RATE_HZ, "transducer_gain_hz": FROZEN.gain_hz,
                     "connectome": conn.name, "direct_mbon_to_gf_synapses": int(np.abs(conn.weight[conn.edge_mask(pre_idx=conn.index_of(mbon['root_id'].astype('int64')), post_idx=gf)]).sum())},
        "baselines": baselines, "baseline_p_spike": base_p, "noise_band_delta_p": noise_band, "mbon_types": rows,
        "n_types_outside_noise_band": int(sum(abs(r["delta_p_spike"]) > noise_band for r in rows)),
        "n_types_driving_gf_alone": int(sum(r["alone"]["p_spike"] > 0 for r in rows)),
    })
    plot(load_result(NAME))
    return 0


def plot(r: dict | None) -> None:
    if r is None:
        raise SystemExit("no results yet")
    plt = setup_matplotlib()
    rows = sorted(r["mbon_types"], key=lambda x: x["delta_p_spike"])
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(rows))
    colors = [C_BAD if abs(v["delta_p_spike"]) > r["noise_band_delta_p"] else C_CTRL for v in rows]
    ax.bar(x, [v["delta_p_spike"] for v in rows], color=colors)
    ax.axhspan(-r["noise_band_delta_p"], r["noise_band_delta_p"], color=C_FLY, alpha=0.12, label="noise band (looming only, other noise seeds)")
    ax.set_xticks(x, [v["cell_type"] for v in rows], rotation=90, fontsize=7)
    ax.set(ylabel="Δ P(GF spike) when this MBON type fires at 100 Hz", title=f"MBON → Giant Fiber influence map (baseline P = {r['baseline_p_spike']:.2f}, r/v = 40 ms)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(NAME))
    plt.close(fig)


def render_report(r: dict) -> str:
    rows = sorted(r["mbon_types"], key=lambda x: -abs(x["delta_p_spike"]))
    pr = r["protocol"]
    lines = [
        f"There are **{pr['direct_mbon_to_gf_synapses']} direct MBON → GF synapses** in the connectome, so any influence is polysynaptic. "
        f"Protocol: looming r/v = {pr['rv_ms']:.0f} ms through the frozen transducer (G = {pr['transducer_gain_hz']:g} Hz), {pr['n_trials']} trials; "
        f"each MBON type driven at {pr['mbon_rate_hz']:.0f} Hz. Baseline P(GF spike) = {r['baseline_p_spike']:.2f}; "
        f"a change is called real only outside ±{r['noise_band_delta_p']:.2f} (spread across noise seeds + 2 s.e.).",
        "",
        f"**{r['n_types_outside_noise_band']} of {len(rows)} MBON types** move the GF response outside the noise band; "
        f"{r['n_types_driving_gf_alone']} can make the GF fire on their own.",
        "",
        "| MBON type | neurons | predicted transmitter | ΔP(GF spike) with looming | P(GF spike) alone |",
        "|---|---:|---|---:|---:|",
    ]
    for v in rows[:10]:
        lines.append(f"| {v['cell_type']} | {v['n_neurons']} | {v['top_nt']} | {v['delta_p_spike']:+.2f} | {v['alone']['p_spike']:.2f} |")
    lines += ["", "(ten largest effects shown; all 35 types are in the JSON)", "", "![MBON → GF influence](figures/mbon_influence.png)", "", "Source: `results/mbon_influence.json`."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
