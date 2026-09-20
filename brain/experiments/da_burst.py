"""Phase 4 diagnostic: which dopamine burst can the model take without igniting the mushroom body?

Found while piloting the learning experiment: the published LIF model treats dopamine like any excitatory fast
transmitter and has no adaptation, and the PAM / PPL1 neurons are wired recurrently with Kenyon cells. A strong
"punishment" burst (16 PPL1 neurons, 100 Hz, 100 ms) ignites self-sustained firing of about a third of all Kenyon
cells that never stops — and that seizure, not the game, then drives the plasticity rule.

No game is involved. Each configuration is one brain column: cluster {PAM, PPL1} × rate × duration, with the visual
context off or on (the context fixed by `experiments/mb_drive.py`, large cactus at θ = 40°). 200 ms baseline, the
burst, then 600 ms; the context (if on) ends 400 ms after the burst started, so the last 200 ms are undriven.

Rule, written before the results were seen: a configuration *ignites* if Kenyon cells still fire in the undriven last
200 ms (more than 10 KC spikes per 10-ms frame; a quiet brain has 0). The dopamine transducer uses, for both clusters,
the strongest burst (highest rate, then longest duration) of the grid that ignites in no column.

    uv run --no-sync python -u -m experiments.da_burst
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from experiments import mb_drive
from experiments.common import load_result, write_result
from flybrain.play import BIO_MS_PER_FRAME
from flybrain.transducer.context import context_neurons, context_rates
from flybrain.transducer.dopamine import DopamineParams

NAME = "da_burst"
CLUSTERS = ("PAM", "PPL1")
RATES_HZ = (10.0, 25.0, 50.0, 100.0)
DURATIONS_FRAMES = (5, 10)
PRE, CONTEXT_OFF_AFTER, TOTAL = 20, 40, 80  # frames: burst starts at PRE; context ends PRE + 40; run ends at TOTAL
IGNITION_KC_SPIKES_PER_FRAME = 10.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import pandas as pd

    from flybrain import data_manifest, neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome("flywire783")
    ix = {n: neurons.indices(conn, n) for n in ("LC4", "LPLC2", "KC", "MBON", "PAM", "PPL1")}
    ann = pd.read_csv(data_manifest.ensure("flywire_annotations", log=lambda _m: None), sep="\t", dtype={"root_id": "Int64"},
                      usecols=["root_id", "super_class"], low_memory=False)
    vpn_ids = ann.loc[ann["super_class"] == "visual_projection", "root_id"].astype("int64").to_numpy()
    vpn = np.setdiff1d(conn.index_of(vpn_ids[np.isin(vpn_ids, conn.root_ids)]), np.concatenate([ix["LC4"], ix["LPLC2"]]))
    context = mb_drive.chosen_context()
    ctx = context_neurons(conn, vpn, ix["KC"], context)

    configs = list(itertools.product(CLUSTERS, RATES_HZ, DURATIONS_FRAMES, (False, True)))
    b = len(configs)
    p = LIFParams()
    net = LIFNetwork(conn, p, batch_size=b, device=args.device, chunk_steps=10)
    drive = PoissonDrive(np.concatenate([ctx, ix["PAM"], ix["PPL1"]]), b, p.dt_ms, device=args.device)
    net.set_drive(drive)
    drive.set_seeds(np.arange(b) + 11)
    n_ctx, n_pam = len(ctx), len(ix["PAM"])
    watch = np.concatenate([ix["KC"], ix["MBON"], ix["PAM"], ix["PPL1"]])
    cuts = np.cumsum([len(ix["KC"]), len(ix["MBON"]), len(ix["PAM"])])
    ctx_rates = context_rates(n_ctx, 1, 40.0, context)
    kc_t, kc_active_t, mbon_t, dan_t = (np.zeros((TOTAL, b)) for _ in range(4))
    for f in range(TOTAL):
        rates = np.zeros((n_ctx + n_pam + len(ix["PPL1"]), b))
        for col, (cluster, rate, frames, with_context) in enumerate(configs):
            if with_context and f < PRE + CONTEXT_OFF_AFTER:
                rates[:n_ctx, col] = ctx_rates
            if PRE <= f < PRE + frames:
                rows = slice(n_ctx, n_ctx + n_pam) if cluster == "PAM" else slice(n_ctx + n_pam, None)
                rates[rows, col] = rate
        drive.set_rates(rates)
        kc, mbon, pam, ppl1 = np.split(net.run(round(BIO_MS_PER_FRAME / p.dt_ms), record=None, watch=watch).watch_counts, cuts)
        kc_t[f], kc_active_t[f], mbon_t[f], dan_t[f] = kc.sum(axis=0), (kc > 0).sum(axis=0), mbon.sum(axis=0), pam.sum(axis=0) + ppl1.sum(axis=0)

    rows = []
    for col, (cluster, rate, frames, with_context) in enumerate(configs):
        late = slice(TOTAL - 20, TOTAL)
        row = {"cluster": cluster, "rate_hz": rate, "burst_frames": frames, "context": with_context,
               "kc_spikes_per_frame_before": float(kc_t[10:PRE, col].mean()),
               "kc_spikes_per_frame_during": float(kc_t[PRE : PRE + frames, col].mean()),
               "kc_spikes_per_frame_last_200ms": float(kc_t[late, col].mean()),
               "kcs_active_per_frame_last_200ms": float(kc_active_t[late, col].mean()),
               "mbon_spikes_per_frame_last_200ms": float(mbon_t[late, col].mean()),
               "dan_spikes_per_frame_during": float(dan_t[PRE : PRE + frames, col].mean()),
               "dan_spikes_per_frame_last_200ms": float(dan_t[late, col].mean())}
        row["ignites"] = bool(row["kc_spikes_per_frame_last_200ms"] > IGNITION_KC_SPIKES_PER_FRAME)
        rows.append(row)
        print(f"[{cluster:>4} {rate:3.0f} Hz × {frames:2d} frames, context {'on ' if with_context else 'off'}] KC spikes/frame before "
              f"{row['kc_spikes_per_frame_before']:6.1f}, during {row['kc_spikes_per_frame_during']:7.1f}, last 200 ms "
              f"{row['kc_spikes_per_frame_last_200ms']:7.1f} ({row['kcs_active_per_frame_last_200ms']:.0f} KCs); DAN spikes/frame during "
              f"{row['dan_spikes_per_frame_during']:6.1f}, last {row['dan_spikes_per_frame_last_200ms']:6.1f} → {'IGNITES' if row['ignites'] else 'ok'}", flush=True)
    safe = [(rate, frames) for rate, frames in itertools.product(RATES_HZ, DURATIONS_FRAMES)
            if not any(r["ignites"] for r in rows if r["rate_hz"] == rate and r["burst_frames"] == frames)]
    chosen = max(safe, default=None)
    write_result(NAME, {"protocol": {"context": {"level": context.level, "code": context.code, "rate_hz": context.rate_hz}, "pre_frames": PRE,
                                     "context_off_after_frames": CONTEXT_OFF_AFTER, "total_frames": TOTAL,
                                     "ignition_kc_spikes_per_frame": IGNITION_KC_SPIKES_PER_FRAME, "n_pam": n_pam, "n_ppl1": len(ix["PPL1"])},
                        "rows": rows, "chosen": None if chosen is None else {"rate_hz": chosen[0], "burst_frames": chosen[1]}})
    print(f"[chosen] {chosen}", flush=True)
    return 0


def chosen_dopamine() -> DopamineParams:
    r = load_result(NAME)
    if r is None or r.get("chosen") is None:
        raise SystemExit("run experiments.da_burst first: no dopamine burst of the grid leaves the mushroom body quiet")
    return DopamineParams(rate_hz=float(r["chosen"]["rate_hz"]), burst_frames=int(r["chosen"]["burst_frames"]))


def render_report(r: dict) -> str:
    pr = r["protocol"]
    lines = [
        f"One brain column per configuration; {pr['n_pam']} PAM or {pr['n_ppl1']} PPL1 neurons driven; 200 ms baseline, burst, 600 ms after; "
        f"the last 200 ms are undriven. *Ignites* = more than {pr['ignition_kc_spikes_per_frame']:g} Kenyon-cell spikes per 10-ms frame in those last 200 ms.",
        "",
        "| cluster | rate (Hz) | duration (ms) | context | KC spikes / frame: before | during burst | last 200 ms | DAN spikes / frame, last 200 ms | ignites |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in r["rows"]:
        lines.append(f"| {row['cluster']} | {row['rate_hz']:.0f} | {row['burst_frames'] * 10} | {'on' if row['context'] else 'off'} | "
                     f"{row['kc_spikes_per_frame_before']:.1f} | {row['kc_spikes_per_frame_during']:.1f} | {row['kc_spikes_per_frame_last_200ms']:.1f} | "
                     f"{row['dan_spikes_per_frame_last_200ms']:.1f} | {'**yes**' if row['ignites'] else 'no'} |")
    c = r["chosen"]
    lines += ["", ("**No burst of the grid is safe.**" if c is None else
                   f"**Chosen by the pre-written rule: {c['rate_hz']:.0f} Hz for {c['burst_frames'] * 10} ms** (both clusters)."),
              "", "Source: `results/da_burst.json`."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
