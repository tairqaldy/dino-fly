"""Sensitivity analysis (DEV seeds only): what would a different looming gain buy the naive fly?

H2 lets mushroom-body output shift the looming gain within [G/2, 2G]. Before looking at any H2 learning result this
script measures what that range is worth at best: the naive, frozen fly (no context, no learning) plays the same 32 DEV
seeds with the transducer gain set to G·2^k for k ∈ {−1, −½, 0, +½, +1}. Nothing here is used to choose a parameter —
G stays frozen at the value fixed in Phase 1 by a biological criterion — and DEV seeds are never headline numbers.

    uv run --no-sync python -u -m experiments.gain_sensitivity
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from experiments.common import write_result
from experiments.naive_play import summarise
from flybrain import seeds as seedsets
from flybrain.play import play_games
from flybrain.transducer.looming import FROZEN

NAME = "gain_sensitivity"
OCTAVES = (-1.0, -0.5, 0.0, 0.5, 1.0)
N_SEEDS = 32
KEEP = ("score_stats", "cleared_stats", "jumps_per_game", "p_jump_per_approach", "p_cleared_given_jump", "frames_to_collision_at_jump_median", "false_jump_rate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from flybrain import neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome("flywire783")
    ix = {n: neurons.indices(conn, n) for n in ("LC4", "LPLC2", "GF")}
    p = LIFParams()
    net = LIFNetwork(conn, p, batch_size=N_SEEDS, device=args.device, chunk_steps=10)
    drive = PoissonDrive(np.concatenate([ix["LC4"], ix["LPLC2"]]), N_SEEDS, p.dt_ms, device=args.device)
    net.set_drive(drive)
    rows = []
    for k in OCTAVES:
        looming = replace(FROZEN, version=0, gain_hz=FROZEN.gain_hz * 2.0**k)  # version 0 = exploratory, not the frozen transducer
        games = play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"], list(seedsets.DEV_SEEDS[:N_SEEDS]), looming=looming, noise_seed=0)
        s = summarise([g.to_json() for g in games])
        rows.append({"octaves": k, "gain_hz": looming.gain_hz} | {key: s[key] for key in KEEP})
        print(f"[gain] {k:+.1f} octaves ({looming.gain_hz:.2f} Hz): score {s['score_stats']['mean']:.1f} [{s['score_stats']['mean_ci95'][0]:.1f}, "
              f"{s['score_stats']['mean_ci95'][1]:.1f}], P(jump/approach) {s['p_jump_per_approach']:.2f}, jump {s['frames_to_collision_at_jump_median']:.1f} "
              f"frames before collision, P(cleared | jump) {s['p_cleared_given_jump']:.2f}", flush=True)
    write_result(NAME, {"protocol": {"seeds": f"DEV_SEEDS[:{N_SEEDS}]", "frozen_gain_hz": FROZEN.gain_hz, "noise_seed": 0}, "rows": rows})
    return 0


def render_report(r: dict) -> str:
    lines = [
        f"Naive frozen fly, {r['protocol']['seeds']}, no context, no learning; the frozen gain is {r['protocol']['frozen_gain_hz']:g} Hz. "
        "DEV seeds: a sensitivity analysis, not a headline result.",
        "",
        "| gain | score mean [95% CI] | P(jump / approach) | jump, frames before collision (median) | P(cleared given jump) | share of jumps with nothing in view |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in r["rows"]:
        st = row["score_stats"]
        lines.append(f"| G · 2^{row['octaves']:+g} = {row['gain_hz']:.2f} Hz | {st['mean']:.1f} [{st['mean_ci95'][0]:.1f}, {st['mean_ci95'][1]:.1f}] | "
                     f"{row['p_jump_per_approach']:.2f} | {row['frames_to_collision_at_jump_median']:.1f} | {row['p_cleared_given_jump']:.2f} | "
                     f"{row['false_jump_rate']:.2f} |")
    lines += ["", "Source: `results/gain_sensitivity.json`."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
