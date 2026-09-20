"""Phase 4 diagnostic: can visual context reach the mushroom body without disturbing the escape reflex?

KC→MBON plasticity can only matter if the context drive makes Kenyon cells fire and those make MBONs fire — the
published LIF model has no spontaneous activity, so a silent MBON stays silent however its input synapses change. But
the 265 visual projection neurons that carry the context are real neurons with many other targets: driving them can
fire the Giant Fiber directly (the dino would jump at the mere sight of an obstacle) and can make dopaminergic
neurons fire by themselves ("endogenous dopamine", which gates plasticity regardless of the game's outcome).

No game score is involved. Nine never-jumping dinos (DEV seeds 1–9) run into their first obstacle; the sequence of
views (obstacle class, angular size θ, expansion rate) of each run is replayed into one brain column, for every
context configuration = code × ramp × peak rate (`flybrain/transducer/context.py`):
    context only          → GF spikes (must be none), dopaminergic spikes, MBON and KC activity
    looming + context     → is the looming-evoked GF spike still there, at the same time before the collision?
    looming only          → the reference reflex

Selection rule, written before the results were seen: among configurations with (1) no context-evoked GF spike in
any column, (2) P(looming-evoked GF spike) ≥ reference − 0.2 and (3) median spike time within ±3 frames of the
reference, take the one with the highest mean rate of the visually reachable MBONs during the last 300 ms before the
collision; ties → lower peak rate. If that rate is below MBON_CRITERION_HZ the mushroom body is called *unreachable*
for a learning experiment under this transducer family.

    uv run --no-sync python -u -m experiments.mb_drive
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from experiments.common import load_result, write_result
from flybrain import dino_core
from flybrain.play import BIO_MS_PER_FRAME
from flybrain.transducer.context import (
    CODE_SEED,
    ContextParams,
    context_neurons,
    context_rates,
    kc_projecting_vpns,
)
from flybrain.transducer.looming import FROZEN, obstacle_view, population_rates

NAME = "mb_drive"
SEEDS = tuple(range(1, 10))  # DEV seeds
RAMPS_DEG = (0.0, 30.0)
# (minimum share of a neuron's output that goes to Kenyon cells, codes, peak rates): all 265 KC-projecting visual
# neurons, or only the 47 / 30 that are "dedicated" mushroom-body inputs (context.kc_projecting_vpns)
FAMILIES = {
    "vpn": (
        (0.05, ("class_only",), (100.0, 200.0, 300.0)),
        (0.10, ("class_only",), (100.0, 200.0, 300.0)),
        (0.0, ("one_of_9", "random_half", "class_only"), (100.0, 200.0)),
    ),
    # --level kc (DEVIATION, DECISIONS.md D13): drive the visual Kenyon cells themselves; a Poisson "kick" is a spike,
    # so these are Kenyon-cell firing rates
    "kc": ((0.05, ("class_only",), (5.0, 10.0, 20.0, 40.0)),),
}
LAST_FRAMES = 30  # "the last 300 ms before the collision"
MBON_CRITERION_HZ = 1.0
MAX_P_DROP, MAX_SHIFT_FRAMES = 0.2, 3.0


def never_jump_views(seed: int) -> list[tuple[int, float, float]]:
    """(obstacle type, θ°, θ̇ °/s) for every frame from the first sighting of an obstacle to the crash."""
    state, views = dino_core.create_initial_state(seed), []
    while not state.crashed:
        view = obstacle_view(state, BIO_MS_PER_FRAME)
        if view.obstacle_index >= 0:
            views.append((state.obstacles[view.obstacle_index].type, float(view.theta_deg), float(view.theta_dot_deg_s)))
        elif views:
            raise RuntimeError("the first obstacle left the view before the crash")
        state = dino_core.step(state, False, False)
    return views


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--level", choices=tuple(FAMILIES), default="vpn")
    args = ap.parse_args()
    level, families = args.level, FAMILIES[args.level]

    import pandas as pd

    from flybrain import data_manifest, neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome("flywire783")
    ix = {n: neurons.indices(conn, n) for n in ("LC4", "LPLC2", "KC", "MBON", "GF", "PAM", "PPL1")}
    ann = pd.read_csv(data_manifest.ensure("flywire_annotations", log=lambda _m: None), sep="\t", dtype={"root_id": "Int64"},
                      usecols=["root_id", "super_class", "cell_type"], low_memory=False)
    vpn_ids = ann.loc[ann["super_class"] == "visual_projection", "root_id"].astype("int64").to_numpy()
    vpn = conn.index_of(vpn_ids[np.isin(vpn_ids, conn.root_ids)])
    vpn = np.setdiff1d(vpn, np.concatenate([ix["LC4"], ix["LPLC2"]]))
    subsets = {frac: context_neurons(conn, vpn, ix["KC"], ContextParams(level=level, min_kc_fraction=frac)) for frac, _codes, _rates in families}
    ctx = np.unique(np.concatenate(list(subsets.values())))  # every neuron any configuration drives
    recipient_kc = np.unique(conn.post[conn.edge_mask(pre_idx=kc_projecting_vpns(conn, vpn, ix["KC"]), post_idx=ix["KC"])])
    type_of = dict(zip(ann["root_id"].astype("int64"), ann["cell_type"].fillna("?"), strict=True))
    mbon_type = np.array([type_of.get(int(conn.root_ids[i]), "?") for i in ix["MBON"]])
    visual_mbon = np.isin(ix["MBON"], np.unique(conn.post[conn.edge_mask(pre_idx=recipient_kc, post_idx=ix["MBON"])]))

    runs = [never_jump_views(s) for s in SEEDS]
    b, n_frames = len(runs), max(len(r) for r in runs)
    classes = [r[0][0] for r in runs]
    print(f"[approach] {b} never-jump runs, {min(len(r) for r in runs)}–{n_frames} frames in view, obstacle classes {classes}", flush=True)

    p = LIFParams()
    steps = round(BIO_MS_PER_FRAME / p.dt_ms)
    net = LIFNetwork(conn, p, batch_size=b, device=args.device, chunk_steps=10)
    drive = PoissonDrive(np.concatenate([ix["LC4"], ix["LPLC2"], ctx]), b, p.dt_ms, device=args.device)
    net.set_drive(drive)
    n_lc4, n_lplc2, n_ctx = len(ix["LC4"]), len(ix["LPLC2"]), len(ctx)
    watch = np.concatenate([ix["GF"], ix["KC"], ix["MBON"], ix["PAM"], ix["PPL1"]])
    cuts = np.cumsum([len(ix[k]) for k in ("GF", "KC", "MBON", "PAM")])

    def replay(context: ContextParams | None, looming: bool) -> dict:
        net.reset()
        drive.set_seeds(np.arange(b) + 31)
        first_gf = np.full(b, -1)
        kc_seen = np.zeros((len(ix["KC"]), b), dtype=bool)
        mbon_last = np.zeros((len(ix["MBON"]), b))
        mbon_all = np.zeros((len(ix["MBON"]), b))
        gf_n = dan_n = 0
        for f in range(n_frames):
            rates = np.zeros((n_lc4 + n_lplc2 + n_ctx, b))
            for col, run in enumerate(runs):
                if f >= len(run):
                    continue
                typ, theta, theta_dot = run[f]
                if looming:
                    lc4, lplc2 = population_rates(theta, theta_dot, FROZEN)
                    rates[:n_lc4, col], rates[n_lc4 : n_lc4 + n_lplc2, col] = float(lc4), float(lplc2)
                if context is not None:
                    member = np.isin(ctx, subsets[context.min_kc_fraction])
                    rates[n_lc4 + n_lplc2 :, col][member] = context_rates(int(member.sum()), typ, theta, context)
            drive.set_rates(rates)
            gf, kc, mbon, pam, ppl1 = np.split(net.run(steps, record=None, watch=watch).watch_counts, cuts)
            alive = np.array([f < len(run) for run in runs])
            gf_any = (gf.sum(axis=0) > 0) & alive
            first_gf = np.where((first_gf < 0) & gf_any, [len(run) - f for run in runs], first_gf)  # frames before the crash
            gf_n += int(gf[:, alive].sum())
            dan_n += int(pam[:, alive].sum() + ppl1[:, alive].sum())
            kc_seen |= (kc > 0) & alive
            mbon_all += mbon * alive
            mbon_last += mbon * np.array([len(run) - LAST_FRAMES <= f < len(run) for run in runs])
        per_type = {t: float(mbon_last[mbon_type == t].sum() / ((mbon_type == t).sum() * b * LAST_FRAMES * BIO_MS_PER_FRAME / 1000))
                    for t in np.unique(mbon_type)}
        spiked = first_gf >= 0
        return {
            "gf_spikes": gf_n, "columns_with_gf_spike": int(spiked.sum()), "p_gf_spike": float(spiked.mean()),
            "gf_first_spike_frames_before_crash_median": float(np.median(first_gf[spiked])) if spiked.any() else None,
            "dan_spikes_per_approach": dan_n / b,
            "kcs_active_per_approach": float(kc_seen.sum(axis=0).mean()),
            "mbons_active_per_approach": float((mbon_all > 0).sum(axis=0).mean()),
            "visual_mbon_rate_hz_last_300ms": float(mbon_last[visual_mbon].sum() / (visual_mbon.sum() * b * LAST_FRAMES * BIO_MS_PER_FRAME / 1000)),
            "top_mbon_types_hz_last_300ms": [[t, round(v, 2)] for t, v in sorted(per_type.items(), key=lambda kv: -kv[1])[:4]],
            "n_active_neurons": int(net.n_active),
        }

    reference = replay(None, looming=True)
    print(f"[reference] looming only: P(GF spike) {reference['p_gf_spike']:.2f}, first spike {reference['gf_first_spike_frames_before_crash_median']} frames "
          f"before the crash, endogenous DAN spikes {reference['dan_spikes_per_approach']:.1f}", flush=True)
    rows = []
    configs = [(frac, code, ramp, rate) for frac, codes, rates in families for code, ramp, rate in itertools.product(codes, RAMPS_DEG, rates)]
    for frac, code, ramp, rate in configs:
        context = ContextParams(code=code, rate_hz=rate, ramp_deg=ramp, min_kc_fraction=frac, level=level)
        alone, both = replay(context, looming=False), replay(context, looming=True)
        shift = (None if both["gf_first_spike_frames_before_crash_median"] is None or reference["gf_first_spike_frames_before_crash_median"] is None
                 else both["gf_first_spike_frames_before_crash_median"] - reference["gf_first_spike_frames_before_crash_median"])
        ok = (alone["gf_spikes"] == 0 and both["p_gf_spike"] >= reference["p_gf_spike"] - MAX_P_DROP
              and shift is not None and abs(shift) <= MAX_SHIFT_FRAMES)
        rows.append({"min_kc_fraction": frac, "n_neurons": len(subsets[frac]), "code": code, "ramp_deg": ramp, "rate_hz": rate, "context_only": alone,
                     "looming_and_context": both, "gf_spike_time_shift_frames": shift, "reflex_preserved": bool(ok)})
        print(f"[{len(subsets[frac]):3d} neurons {code:>11} ramp {ramp:2.0f}° {rate:3.0f} Hz] context only: GF spikes {alone['gf_spikes']} ({alone['columns_with_gf_spike']}/{b} columns), "
              f"DAN {alone['dan_spikes_per_approach']:.0f}/approach, KCs {alone['kcs_active_per_approach']:.0f}, visual MBONs "
              f"{alone['visual_mbon_rate_hz_last_300ms']:.2f} Hz {alone['top_mbon_types_hz_last_300ms'][:2]} | with looming: P {both['p_gf_spike']:.2f}, "
              f"shift {shift} → {'ok' if ok else 'reflex disturbed'}", flush=True)
    passing = [r for r in rows if r["reflex_preserved"]]
    best = max(passing, key=lambda r: (r["context_only"]["visual_mbon_rate_hz_last_300ms"], -r["rate_hz"]), default=None)
    chosen = None if best is None else {k: best[k] for k in ("min_kc_fraction", "n_neurons", "code", "ramp_deg", "rate_hz")} | {
        "level": level, "visual_mbon_rate_hz_last_300ms": best["context_only"]["visual_mbon_rate_hz_last_300ms"],
        "mushroom_body_reachable": best["context_only"]["visual_mbon_rate_hz_last_300ms"] >= MBON_CRITERION_HZ}
    write_result(result_name(level), {"level": level, "protocol": {"seeds": list(SEEDS), "obstacle_classes": classes, "frames_in_view": [len(r) for r in runs], "code_seed": CODE_SEED,
                                     "n_context_vpns": n_ctx, "n_recipient_kcs": len(recipient_kc), "n_visual_mbons": int(visual_mbon.sum()),
                                     "mbon_criterion_hz": MBON_CRITERION_HZ, "max_p_drop": MAX_P_DROP, "max_shift_frames": MAX_SHIFT_FRAMES,
                                     "transducer_gain_hz": FROZEN.gain_hz},
                        "reference_looming_only": reference, "rows": rows, "chosen": chosen})
    print(f"[chosen] {chosen}", flush=True)
    return 0


def result_name(level: str) -> str:
    return NAME if level == "vpn" else f"{NAME}_{level}"


def chosen_context() -> ContextParams:
    """The context configuration fixed by the rule above (read from the committed results; no game score involved).

    The intended level (visual projection neurons) wins if any of its configurations preserves the reflex; otherwise the
    Kenyon-cell level is used — a deviation that every result stamps into its protocol.
    """
    for level in FAMILIES:
        r = load_result(result_name(level))
        if r is not None and r.get("chosen") is not None:
            c = r["chosen"]
            return ContextParams(code=c["code"], rate_hz=float(c["rate_hz"]), ramp_deg=float(c["ramp_deg"]),
                                 min_kc_fraction=float(c["min_kc_fraction"]), level=level)
    raise SystemExit("run experiments.mb_drive (both levels) first: no context configuration preserves the escape reflex")


def render_report(r: dict) -> str:
    pr, ref = r["protocol"], r["reference_looming_only"]
    intro = (f"{len(pr['seeds'])} never-jumping dinos run into their first obstacle; the views are replayed into a naive brain. 265 visual projection "
             f"neurons synapse directly onto {pr['n_recipient_kcs']} Kenyon cells, which reach {pr['n_visual_mbons']} of the 96 MBONs. "
             if r.get("level", "vpn") == "vpn" else
             "Same protocol, but the context drives the visual Kenyon cells themselves (rates are Kenyon-cell firing rates). ")
    lines = [
        intro + f"Reference (looming only): P(GF spike) = {ref['p_gf_spike']:.2f}, first spike "
        f"{ref['gf_first_spike_frames_before_crash_median']:.0f} frames before the crash, {ref['dan_spikes_per_approach']:.1f} dopaminergic spikes per approach.",
        "",
        "| context neurons | code | ramp | peak rate (Hz) | context alone: GF spikes | dopaminergic spikes / approach | KCs active | visually reachable "
        "MBONs, last 300 ms (Hz) | with looming: P(GF spike) | spike-time shift (frames) | reflex preserved |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in r["rows"]:
        a, both, shift = row["context_only"], row["looming_and_context"], row["gf_spike_time_shift_frames"]
        ramp = "—" if row["ramp_deg"] == 0 else f"{row['ramp_deg']:.0f}°"
        lines.append(f"| {row['n_neurons']} | {row['code']} | {ramp} | {row['rate_hz']:.0f} | {a['gf_spikes']} | "
                     f"{a['dan_spikes_per_approach']:.0f} | {a['kcs_active_per_approach']:.0f} | {a['visual_mbon_rate_hz_last_300ms']:.2f} | "
                     f"{both['p_gf_spike']:.2f} | {'—' if shift is None else f'{shift:+.0f}'} | {'yes' if row['reflex_preserved'] else 'no'} |")
    c = r["chosen"]
    if c is None:
        verdict = "**No configuration preserves the reflex.**"
    else:
        verdict = (f"**Chosen by the pre-written rule: {c['n_neurons']} neurons, `{c['code']}`, ramp {c['ramp_deg']:.0f}°, {c['rate_hz']:.0f} Hz** — visually reachable MBONs fire "
                   f"{c['visual_mbon_rate_hz_last_300ms']:.2f} Hz before the collision, "
                   + ("above" if c["mushroom_body_reachable"] else "**below**") + f" the {pr['mbon_criterion_hz']:g} Hz criterion.")
    lines += ["", verdict, "", f"Source: `results/{result_name(r.get('level', 'vpn'))}.json`."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
