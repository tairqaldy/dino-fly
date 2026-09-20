"""Phase 1: does a looming stimulus, delivered through LPLC2 / LC4, reach the Giant Fiber in the whole-brain model?

Standard protocol of the fly looming literature (von Reyn 2017, Ache 2019, Jang 2023): a disk with size-to-speed
ratio r/v ∈ {10, 20, 40, 80} ms grows from 10° to 63° and then holds. The fixed transducer
(`flybrain/transducer/looming.py`) turns (θ, θ̇) into Poisson rates of all LPLC2 and LC4 neurons (both sides).

Measured:  P(GF spike), first-spike time and angular size at the first GF spike vs. r/v, for a log-spaced sweep of
the transducer's one free parameter G; LC4-only / LPLC2-only drive; GF's rank among all descending neurons; the same
drive delivered to LPLC1 / LC6 (wrong cell types); and a G-insensitive check: with GF spiking disabled, the angular
size at which the GF membrane potential peaks, per r/v.

Calibration (pre-declared, DECISIONS.md F3): G* = the sweep value whose median first-spike angular size (over all
trials that spiked, all r/v) is closest to 42°, among values with overall P(spike) ≥ 0.5. Fallbacks: GF never
fires → report, no calibration; GF fires at stimulus onset for every G → smallest G with P(spike) ≥ 0.5 at r/v = 40 ms.

    uv run --no-sync python -u -m experiments.looming_gf
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace

import numpy as np

from experiments.common import (
    C_ALT,
    C_CTRL,
    C_FLY,
    C_REF,
    figure_path,
    load_result,
    setup_matplotlib,
    write_result,
)
from flybrain.transducer.looming import (
    LoomingParams,
    looming_theta_deg,
    looming_theta_dot_deg_s,
    population_rates,
    time_to_collision_ms,
)

NAME = "looming_gf"
RV_MS = (10.0, 20.0, 40.0, 80.0)
THETA_START, THETA_END = 10.0, 63.0
BASELINE_MS, HOLD_MS = 50.0, 100.0
N_TRIALS = 20
G_SWEEP_HZ = (2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
TARGET_SIZE_DEG = 42.0
CHUNK = 10  # steps per rate update (1 ms)
DT_MS = 0.1


def stimulus(rv_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """θ(t), θ̇(t) sampled every ms from baseline start to end of hold. Returns (t_ms from trial start, θ, θ̇)."""
    t0, t1 = time_to_collision_ms(THETA_START, rv_ms), time_to_collision_ms(THETA_END, rv_ms)
    n_loom = int(np.ceil(t0 - t1))
    t = np.arange(int(BASELINE_MS) + n_loom + int(HOLD_MS), dtype=np.float64)
    ttc = t0 - (t - BASELINE_MS)
    theta = np.where(t < BASELINE_MS, THETA_START, looming_theta_deg(np.maximum(ttc, t1), rv_ms))
    vel = np.where((t >= BASELINE_MS) & (ttc > t1), looming_theta_dot_deg_s(ttc, rv_ms), 0.0)
    return t, theta, vel


def run_condition(net, drive, pops: dict, params: LoomingParams, *, which: str, seed_base: int, watch, sample_vm=None):
    """One batch: columns = r/v × trials. `which` ∈ both | lc4 | lplc2 | swap. Returns per-column measurements."""
    rvs = np.repeat(RV_MS, N_TRIALS)
    stims = {rv: stimulus(rv) for rv in RV_MS}
    n_ms = max(len(s[0]) for s in stims.values())
    b = len(rvs)
    theta = np.full((n_ms, b), THETA_END)
    vel = np.zeros((n_ms, b))
    for col, rv in enumerate(rvs):
        _, th, ve = stims[rv]
        theta[: len(th), col], vel[: len(ve), col] = th, ve

    net.reset()
    drive.set_seeds(np.arange(b) + seed_base)
    first = np.full((len(watch), b), -1, dtype=np.int64)
    counts = np.zeros((len(watch), b), dtype=np.int64)
    vm = [] if sample_vm is not None else None
    n_a, n_b = len(pops["a"]), len(pops["b"])
    for ms in range(n_ms):
        lc4, lplc2 = population_rates(theta[ms], vel[ms], params)
        if which == "lc4":
            lplc2 = np.zeros_like(lplc2)
        elif which == "lplc2":
            lc4 = np.zeros_like(lc4)
        rates = np.concatenate([np.tile(lc4, (n_a, 1)), np.tile(lplc2, (n_b, 1))], axis=0)
        drive.set_rates(rates)
        res = net.run(CHUNK, record=None, watch=watch)
        new = (first < 0) & (res.watch_first_step >= 0)
        first = np.where(new, res.watch_first_step + ms * CHUNK, first)
        counts += res.watch_counts
        if vm is not None:
            vm.append(net.membrane_mv(sample_vm).mean(axis=0))
    out = {"rv_ms": rvs, "first_step": first, "counts": counts, "theta": theta, "n_ms": n_ms}
    if vm is not None:
        out["vm"] = np.asarray(vm)  # [n_ms, B]
    return out


def summarise(cond: dict, n_gf: int = 2) -> dict:
    """Per r/v: P(spike), first-spike latency / time-to-collision / angular size (either GF)."""
    first = cond["first_step"][:n_gf]
    any_first = np.where(first >= 0, first, np.iinfo(np.int64).max).min(axis=0)
    spiked = any_first < np.iinfo(np.int64).max
    rows, sizes_all = [], []
    for rv in RV_MS:
        cols = np.flatnonzero(cond["rv_ms"] == rv)
        sp = spiked[cols]
        t_ms = any_first[cols][sp] * DT_MS
        onset_ms = t_ms - BASELINE_MS
        ms_idx = np.clip((t_ms).astype(int), 0, cond["n_ms"] - 1)
        size = cond["theta"][ms_idx, cols[sp]]
        ttc = time_to_collision_ms(THETA_START, rv) - onset_ms
        sizes_all.extend(size.tolist())
        rows.append(
            {
                "rv_ms": rv,
                "p_spike": float(sp.mean()),
                "gf_spikes_per_trial": float(cond["counts"][:n_gf, cols].sum(axis=0).mean()),
                "latency_from_onset_ms_median": float(np.median(onset_ms)) if sp.any() else None,
                "time_to_collision_ms_median": float(np.median(ttc)) if sp.any() else None,
                "size_at_first_spike_deg_median": float(np.median(size)) if sp.any() else None,
                "size_at_first_spike_deg_iqr": [float(np.percentile(size, 25)), float(np.percentile(size, 75))]
                if sp.any()
                else None,
                "spiked_before_onset": int((onset_ms < 0).sum()),
            }
        )
    return {
        "per_rv": rows,
        "p_spike_overall": float(spiked.mean()),
        "size_at_first_spike_deg_median_overall": float(np.median(sizes_all)) if sizes_all else None,
    }


def calibrate(sweep: list[dict]) -> dict:
    ok = [s for s in sweep if s["p_spike_overall"] >= 0.5 and s["size_at_first_spike_deg_median_overall"] is not None]
    if not any(s["p_spike_overall"] > 0 for s in sweep):
        return {"status": "gf_never_fires", "gain_hz": None}
    if ok:
        best = min(ok, key=lambda s: abs(s["size_at_first_spike_deg_median_overall"] - TARGET_SIZE_DEG))
        return {
            "status": "calibrated",
            "rule": f"median first-spike size closest to {TARGET_SIZE_DEG} deg among G with P(spike) >= 0.5",
            "gain_hz": best["gain_hz"],
            "median_size_deg": best["size_at_first_spike_deg_median_overall"],
            "p_spike_overall": best["p_spike_overall"],
        }
    rv40 = [s for s in sweep if next(r for r in s["per_rv"] if r["rv_ms"] == 40.0)["p_spike"] >= 0.5]
    if rv40:
        best = min(rv40, key=lambda s: s["gain_hz"])
        return {"status": "fallback_smallest_gain_with_p>=0.5_at_rv40", "gain_hz": best["gain_hz"]}
    return {"status": "no_gain_qualifies", "gain_hz": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        plot(load_result(NAME))
        return 0

    from flybrain import neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome("flywire783")
    lc4, lplc2, gf = (neurons.indices(conn, n) for n in ("LC4", "LPLC2", "GF"))
    dn_all = neurons.indices(conn, "DN_ALL")
    watch = np.concatenate([gf, dn_all[~np.isin(dn_all, gf)]])
    p = LIFParams()
    b = len(RV_MS) * N_TRIALS
    net = LIFNetwork(conn, p, batch_size=b, device=args.device, chunk_steps=CHUNK)
    drive = PoissonDrive(np.concatenate([lc4, lplc2]), b, p.dt_ms, device=args.device)
    net.set_drive(drive)
    pops = {"a": lc4, "b": lplc2}
    base = LoomingParams(version=1, gain_hz=1.0)

    sweep = []
    for k, gain in enumerate(G_SWEEP_HZ):
        cond = run_condition(net, drive, pops, replace(base, gain_hz=gain), which="both", seed_base=10_000 * (k + 1), watch=watch)
        s = summarise(cond) | {"gain_hz": gain}
        sweep.append(s)
        print(f"[sweep] G={gain:6.1f} Hz  P(spike)={s['p_spike_overall']:.2f}  median size={s['size_at_first_spike_deg_median_overall']}", flush=True)
    cal = calibrate(sweep)
    print("[calibration]", cal, flush=True)

    result: dict = {
        "protocol": {
            "rv_ms": list(RV_MS), "theta_start_deg": THETA_START, "theta_end_deg": THETA_END, "baseline_ms": BASELINE_MS,
            "hold_ms": HOLD_MS, "n_trials_per_rv": N_TRIALS, "dt_ms": DT_MS, "rate_update_ms": CHUNK * DT_MS,
            "target_size_deg": TARGET_SIZE_DEG, "connectome": conn.name, "transducer_forms": asdict(base) | {"gain_hz": "swept"},
            "populations": {"LC4": len(lc4), "LPLC2": len(lplc2), "GF": len(gf)},
        },
        "gain_sweep": sweep,
        "calibration": cal,
    }

    if cal["gain_hz"] is not None:
        star = replace(base, gain_hz=cal["gain_hz"])
        for which in ("lc4", "lplc2"):
            result[f"{which}_only"] = summarise(run_condition(net, drive, pops, star, which=which, seed_base=777_000, watch=watch))
            print(f"[{which} only] P(spike)={result[f'{which}_only']['p_spike_overall']:.2f}", flush=True)
        for sigma in (10.0, 25.0):
            key = f"sigma_{sigma:.0f}"
            result[key] = summarise(run_condition(net, drive, pops, replace(star, size_sigma_deg=sigma), which="both", seed_base=778_000, watch=watch))

        # specificity 1: GF's rank among all descending neurons (same run as the calibrated condition)
        cond = run_condition(net, drive, pops, star, which="both", seed_base=779_000, watch=watch)
        p_dn = (cond["counts"] > 0).mean(axis=1)
        order = np.argsort(-p_dn, kind="stable")
        ids = conn.root_ids[watch]
        result["calibrated"] = summarise(cond)
        result["dn_ranking"] = {
            "n_descending": len(watch),
            "n_responding": int((p_dn > 0).sum()),
            "gf_p_spike": [float(p_dn[0]), float(p_dn[1])],
            "gf_ranks": [int(np.flatnonzero(order == 0)[0]) + 1, int(np.flatnonzero(order == 1)[0]) + 1],
            "top10": [{"flywire_id": str(ids[i]), "p_spike": float(p_dn[i]), "mean_spikes": float(cond["counts"][i].mean())} for i in order[:10]],
        }

        # G-insensitive check: GF cannot spike → at which angular size does its membrane potential peak?
        net.ablate(gf)
        cond = run_condition(net, drive, pops, star, which="both", seed_base=780_000, watch=watch, sample_vm=gf)
        net.ablate(None)
        vm_rows = []
        for rv in RV_MS:
            cols = np.flatnonzero(cond["rv_ms"] == rv)
            mean_vm = cond["vm"][:, cols].mean(axis=1)
            k_peak = int(np.argmax(mean_vm))
            vm_rows.append({"rv_ms": rv, "peak_vm_mv": float(mean_vm[k_peak]), "size_at_vm_peak_deg": float(cond["theta"][k_peak, cols[0]]),
                            "vm_trace_mv": mean_vm.round(3).tolist(), "theta_trace_deg": cond["theta"][:, cols[0]].round(2).tolist()})
        result["vm_no_spike"] = vm_rows

        # specificity 2: identical drive into the wrong visual projection neurons
        swap_a, swap_b = neurons.indices(conn, "LC6"), neurons.indices(conn, "LPLC1")
        drive2 = PoissonDrive(np.concatenate([swap_a, swap_b]), b, p.dt_ms, device=args.device)
        net.set_drive(drive2)
        result["swap_lc6_lplc1"] = summarise(run_condition(net, drive2, {"a": swap_a, "b": swap_b}, star, which="both", seed_base=781_000, watch=watch))
        print(f"[swap] P(spike)={result['swap_lc6_lplc1']['p_spike_overall']:.2f}", flush=True)

    write_result(NAME, result)
    plot(result)
    return 0


def plot(result: dict) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    colors = {10.0: "#08306b", 20.0: "#2171b5", 40.0: "#6baed6", 80.0: "#bdd7e7"}

    ax = axes[0]
    for rv in RV_MS:
        t, theta, _ = stimulus(rv)
        ax.plot(t - BASELINE_MS, theta, color=colors[rv], label=f"r/v = {rv:.0f} ms")
    ax.axhline(TARGET_SIZE_DEG, color=C_REF, lw=0.8, ls=":")
    ax.set(xlabel="time from looming onset (ms)", ylabel="angular size θ (deg)", title="A  stimulus (10° → 63°, then hold)")
    ax.legend()

    ax = axes[1]
    gains = [s["gain_hz"] for s in result["gain_sweep"]]
    for rv in RV_MS:
        ax.plot(gains, [next(r for r in s["per_rv"] if r["rv_ms"] == rv)["p_spike"] for s in result["gain_sweep"]],
                marker="o", ms=3, color=colors[rv], label=f"{rv:.0f} ms")
    ax.set(xscale="log", xlabel="transducer gain G (Hz, peak LC4 / LPLC2 rate)", ylabel="P(≥1 GF spike)", title="B  GF spike probability", ylim=(-0.03, 1.03))
    g_star = result["calibration"].get("gain_hz")
    if g_star:
        ax.axvline(g_star, color=C_ALT, lw=1, ls="--", label=f"G* = {g_star:g} Hz")
    ax.legend(title="r/v", fontsize=7)

    ax = axes[2]
    for rv in RV_MS:
        ys = [next(r for r in s["per_rv"] if r["rv_ms"] == rv)["size_at_first_spike_deg_median"] for s in result["gain_sweep"]]
        ax.plot(gains, [np.nan if y is None else y for y in ys], marker="o", ms=3, color=colors[rv])
    ax.axhline(TARGET_SIZE_DEG, color=C_REF, lw=0.8, ls=":", label="42° target")
    if g_star:
        ax.axvline(g_star, color=C_ALT, lw=1, ls="--")
    ax.set(xscale="log", xlabel="transducer gain G (Hz)", ylabel="θ at first GF spike (deg, median)", title="C  size at first spike vs. G")
    ax.legend(fontsize=7)

    ax = axes[3]
    if "calibrated" in result:
        labels, vals = [], []
        for key, label in (("calibrated", "LC4+LPLC2"), ("lc4_only", "LC4 only"), ("lplc2_only", "LPLC2 only"), ("swap_lc6_lplc1", "LC6+LPLC1\n(wrong cells)")):
            labels.append(label)
            vals.append([r["p_spike"] for r in result[key]["per_rv"]])
        x = np.arange(len(labels))
        for j, rv in enumerate(RV_MS):
            ax.bar(x + (j - 1.5) * 0.2, [v[j] for v in vals], width=0.2, color=colors[rv], label=f"{rv:.0f} ms")
        ax.set_xticks(x, labels)
        ax.set(ylabel="P(≥1 GF spike)", title=f"D  which input drives the GF (G* = {g_star:g} Hz)", ylim=(0, 1.05))
    else:
        ax.text(0.5, 0.5, "no calibration possible", ha="center", va="center", color=C_CTRL)
        ax.set_axis_off()
    for a in axes[:3]:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(NAME))
    plt.close(fig)
    print(f"[figure] wrote {figure_path(NAME)}")
    _ = C_FLY


def render_report(r: dict) -> str:
    cal = r["calibration"]
    pr = r["protocol"]
    lines = [
        f"Protocol: r/v ∈ {{{', '.join(f'{x:.0f}' for x in pr['rv_ms'])}}} ms, {pr['theta_start_deg']:.0f}° → {pr['theta_end_deg']:.0f}°, "
        f"{pr['n_trials_per_rv']} trials each, all {pr['populations']['LC4']} LC4 and {pr['populations']['LPLC2']} LPLC2 neurons driven "
        f"(both sides), connectome {pr['connectome']}, dt {pr['dt_ms']} ms.",
        "",
        "| G (Hz) | P(GF spike) overall | median θ at first spike (deg) | P(spike) per r/v = 10 / 20 / 40 / 80 ms |",
        "|---:|---:|---:|---|",
    ]
    for s in r["gain_sweep"]:
        size = s["size_at_first_spike_deg_median_overall"]
        lines.append(
            f"| {s['gain_hz']:g} | {s['p_spike_overall']:.2f} | {'–' if size is None else f'{size:.1f}'} | "
            + " / ".join(f"{x['p_spike']:.2f}" for x in s["per_rv"])
            + " |"
        )
    lines.append("")
    if cal["gain_hz"] is None:
        lines.append(f"**Calibration: {cal['status']}.** No transducer gain could be fixed by the pre-declared rule.")
        return "\n".join(lines)
    lines.append(
        f"**Calibration ({cal['status']}): G\\* = {cal['gain_hz']:g} Hz** — median angular size at the first GF spike "
        f"{cal.get('median_size_deg', float('nan')):.1f}° (target {pr['target_size_deg']:.0f}°), P(spike) = {cal.get('p_spike_overall', float('nan')):.2f}. "
        "This value is frozen as `TRANSDUCER_VERSION = 1` before any game is played."
    )
    lines += ["", "At G\\*:", "", "| r/v (ms) | P(spike) | GF spikes / trial | first spike: ms after onset | ms before collision | θ at first spike, median [IQR] |", "|---:|---:|---:|---:|---:|---|"]
    for x in r["calibrated"]["per_rv"]:
        if x["size_at_first_spike_deg_median"] is None:
            lines.append(f"| {x['rv_ms']:.0f} | {x['p_spike']:.2f} | {x['gf_spikes_per_trial']:.1f} | – | – | – |")
            continue
        iqr = x["size_at_first_spike_deg_iqr"]
        lines.append(
            f"| {x['rv_ms']:.0f} | {x['p_spike']:.2f} | {x['gf_spikes_per_trial']:.1f} | {x['latency_from_onset_ms_median']:.0f} | "
            f"{x['time_to_collision_ms_median']:.0f} | {x['size_at_first_spike_deg_median']:.1f}° [{iqr[0]:.1f}, {iqr[1]:.1f}] |"
        )
    dn = r["dn_ranking"]
    lines += [
        "",
        f"- **Which input matters:** LC4 only → P(spike) {r['lc4_only']['p_spike_overall']:.2f}; LPLC2 only → "
        f"{r['lplc2_only']['p_spike_overall']:.2f}; both → {r['calibrated']['p_spike_overall']:.2f}.",
        f"- **Wrong cell types (same drive into LC6 + LPLC1):** P(GF spike) = {r['swap_lc6_lplc1']['p_spike_overall']:.2f}.",
        f"- **Specificity of the readout:** of {dn['n_descending']:,} descending neurons, {dn['n_responding']} fire at all under this "
        f"drive; the two GFs rank {dn['gf_ranks'][0]} and {dn['gf_ranks'][1]} by spike probability "
        f"(P = {dn['gf_p_spike'][0]:.2f}, {dn['gf_p_spike'][1]:.2f}).",
        "- **Size-threshold behaviour (GF spiking disabled):** angular size at the peak of the GF membrane potential: "
        + ", ".join(f"{v['size_at_vm_peak_deg']:.0f}° (r/v {v['rv_ms']:.0f} ms)" for v in r["vm_no_spike"])
        + ". In real flies the GF response peaks at a roughly constant angular size across r/v (Ache et al. 2019).",
        f"- **Sensitivity to our choice σ = 15°:** median size at first spike {r['sigma_10']['size_at_first_spike_deg_median_overall']:.1f}° "
        f"(σ = 10°) and {r['sigma_25']['size_at_first_spike_deg_median_overall']:.1f}° (σ = 25°).",
        "",
        "![looming → GF](figures/looming_gf.png)",
        "",
        "Source: `results/looming_gf.json`.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
