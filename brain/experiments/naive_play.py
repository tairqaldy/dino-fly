"""Phase 1: how does the naive fly play — and how much of that is the brain?

Every condition plays the same HELDOUT_200 seeds (obstacle sequences do not depend on actions → paired design) with
common random numbers (same Poisson stream keys). Transducer and engine are frozen; nothing is tuned here.

Brain conditions (whole-brain LIF, connectome v783 unless stated):
  intact                 the naive fly (= M2, full connectome)
  gf_ablated             control (c): Giant Fiber cannot spike (Kir2.1-like)
  gf_output_zeroed       published-style silencing (GF's outgoing synapses removed; it still spikes)
  m1_monosynaptic        attribution ladder M1: ONLY the direct LC4/LPLC2 → GF synapses exist
  m3_no_direct           attribution ladder M3: everything EXCEPT the direct LC4/LPLC2 → GF synapses
  shuffle_global_XX      control (a): degree-preserving shuffle of the whole connectome, 20 realisations
  shuffle_preserve_XX    stricter null: shuffle everything except edges out of LC4/LPLC2 and into the GF, 5 realisations
  intact_noise1/2        same fly, different Poisson noise (game-seed vs. noise-seed variance)
  intact_bio16.7         secondary condition: 16.7 ms of biological time per frame (real time) instead of 10 ms
No-brain conditions:
  never_jump, oracle     floor and scripted ceiling (labelled non-fly baselines)
  m0_threshold           attribution ladder M0: jump when the transducer's summed output crosses a threshold that is
                         calibrated by the same biological criterion as G (42° on the standalone looming protocol)
  random_matched         control (b): random jumps, rate matched to the intact fly per speed bin
  yoked                  control (b'): the intact fly's own key presses from a different seed

    uv run --no-sync python -u -m experiments.naive_play --budget-hours 6
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments.common import (
    C_ALT,
    C_BAD,
    C_CTRL,
    C_FLY,
    C_REF,
    figure_path,
    load_result,
    require_preregistration,
    setup_matplotlib,
    write_result,
)
from flybrain import dino_core as dc
from flybrain import seeds as seedsets
from flybrain.config import cache_dir
from flybrain.play import BIO_MS_PER_FRAME, MAX_FRAMES, GameResult, play_games
from flybrain.stats import bootstrap_mean_ci, describe, holm, paired_comparison, survival_curve
from flybrain.transducer.looming import FROZEN, TRANSDUCER_VERSION, obstacle_view, population_rates
from flybrain.transducer.motor import JumpMotor, MotorParams

NAME = "naive_play"
PREREG_TAG = "prereg-phase1"
N_SHUFFLE_GLOBAL, N_SHUFFLE_PRESERVE = 20, 5
BATCH = 64


# ------------------------------------------------------------------------------------------------ caching
def cache_path(experiment: str, condition: str):
    path = cache_dir() / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{condition}.jsonl"


def load_cached(experiment: str, condition: str) -> dict[int, dict]:
    path = cache_path(experiment, condition)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["seed"]: r for r in rows}


def append_cached(experiment: str, condition: str, result: GameResult) -> None:
    with cache_path(experiment, condition).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_json()) + "\n")


# ------------------------------------------------------------------------------------------------ no-brain agents
def run_policy(seed: int, policy, max_frames: int = MAX_FRAMES) -> dict:
    """policy(state) → (jump, duck). Returns a GameResult-shaped dict."""
    state, bits, actions = dc.create_initial_state(seed), 0, []
    while not state.crashed and state.frame < max_frames:
        jump, duck = policy(state)
        b = int(jump) + 2 * int(duck)
        if b != bits:
            actions.append((state.frame, b))
            bits = b
        state = dc.step(state, jump, duck)
    return {
        "seed": seed, "frames": state.frame, "score": dc.score(state), "cleared": state.cleared, "crashed": state.crashed,
        "censored": not state.crashed, "death_type": state.death_type, "jumps": state.jumps, "actions": actions,
    }


def oracle_policy(s: dc.GameState) -> tuple[bool, bool]:
    """Python twin of the scripted look-ahead player in packages/dino-core/scripts/policies.ts (non-fly baseline)."""
    fpx, d = dc.FPX, dc.D
    dino_back, dino_front = d["x"] * fpx, (d["x"] + d["width"]) * fpx
    ahead = [o for o in s.obstacles if o.x + o.width_px * fpx >= dino_back]
    if not ahead:
        return False, s.jumping
    o = ahead[0]
    rel, dist = s.speed + o.speed_offset, o.x - dino_front
    if s.jumping:
        clear_below = dist > rel * 9
        return (not clear_below), clear_below
    if o.type == 2 and o.y_bottom_px >= 50:
        return False, False
    if o.type == 2 and o.y_bottom_px == 25:
        return False, dist < rel * 14
    return dist <= rel * (7 if o.height_px >= 50 else 6), False


def threshold_policy_factory(threshold: float, n_lc4: int, n_lplc2: int, bio_ms: float):
    """M0: the transducer's summed output (expected VPN spikes/s at unit gain) against a fixed threshold."""
    unit = type(FROZEN)(version=FROZEN.version, gain_hz=1.0, size_peak_deg=FROZEN.size_peak_deg,
                        size_sigma_deg=FROZEN.size_sigma_deg, velocity_sat_deg_s=FROZEN.velocity_sat_deg_s)

    def make():
        motor = JumpMotor(MotorParams())

        def policy(s: dc.GameState) -> tuple[bool, bool]:
            v = obstacle_view(s, bio_ms)
            lc4, lplc2 = population_rates(v.theta_deg, v.theta_dot_deg_s, unit)
            drive = n_lc4 * float(lc4) + n_lplc2 * float(lplc2)
            return motor.update(1 if drive >= threshold else 0, airborne=s.jumping), False

        return policy

    return make, unit


def calibrate_m0_threshold(n_lc4: int, n_lplc2: int, target_deg: float = 42.0) -> dict:
    """Same biological criterion as for G: median angular size at threshold crossing over r/v ∈ {10,20,40,80} ms."""
    from experiments.looming_gf import RV_MS, stimulus

    unit = type(FROZEN)(version=FROZEN.version, gain_hz=1.0)
    traces = []
    for rv in RV_MS:
        _, theta, vel = stimulus(rv)
        lc4, lplc2 = population_rates(theta, vel, unit)
        traces.append((theta, n_lc4 * lc4 + n_lplc2 * lplc2))
    best = None
    for thr in np.geomspace(1.0, max(t[1].max() for t in traces), 600):
        sizes = [float(theta[np.argmax(s >= thr)]) for theta, s in traces if (s >= thr).any()]
        if len(sizes) < len(traces):
            continue
        err = abs(float(np.median(sizes)) - target_deg)
        if best is None or err < best[0]:
            best = (err, float(thr), sizes)
    return {"threshold": best[1], "sizes_deg_per_rv": best[2], "median_size_deg": float(np.median(best[2]))}


# ------------------------------------------------------------------------------------------------ summaries
def summarise(rows: list[dict], *, bio_ms: float = BIO_MS_PER_FRAME) -> dict:
    rows = sorted(rows, key=lambda r: r["seed"])
    arr = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    out = {
        "n_games": len(rows),
        "seeds": [r["seed"] for r in rows],
        "score": arr("score").tolist(),
        "cleared": arr("cleared").tolist(),
        "frames": arr("frames").tolist(),
        "censored": int(arr("censored").sum()),
        "score_stats": describe(arr("score")) | {"mean_ci95": bootstrap_mean_ci(arr("score"))},
        "cleared_stats": describe(arr("cleared")) | {"mean_ci95": bootstrap_mean_ci(arr("cleared"))},
        "jumps_per_game": float(arr("jumps").mean()),
        "death_type_counts": {dc.C["obstacles"][t]["name"]: int((arr("death_type") == t).sum()) for t in range(3)},
        "survival": survival_curve(arr("frames"), arr("crashed"), MAX_FRAMES),
    }
    if "gf_spikes" in rows[0]:
        aps = [a for r in rows for a in r["approaches"]]
        resolved = [a for a in aps if a["outcome"] != "open"]
        jumped = [a for a in resolved if a["jumped"]]
        with_gf = [a for a in resolved if a["gf_spikes"] > 0]
        total_frames = float(arr("frames").sum())
        out |= {
            "gf_spikes_per_game": float(arr("gf_spikes").mean()),
            "gf_spikes_per_bio_s": float(arr("gf_spikes").sum() / (total_frames * bio_ms * 1e-3)),
            "gf_spikes_ignored_per_game": float(arr("gf_spikes_ignored").mean()),
            "jumps_without_obstacle_in_view": int(arr("jumps_without_obstacle_in_view").sum()),
            "false_jump_rate": float(arr("jumps_without_obstacle_in_view").sum() / max(arr("jumps").sum(), 1)),
            "approaches": len(resolved),
            "p_gf_spike_per_approach": len(with_gf) / max(len(resolved), 1),
            "p_jump_per_approach": len(jumped) / max(len(resolved), 1),
            "p_cleared_per_approach": sum(a["outcome"] == "cleared" for a in resolved) / max(len(resolved), 1),
            "p_cleared_given_jump": sum(a["outcome"] == "cleared" for a in jumped) / max(len(jumped), 1),
            "gf_spikes_per_approach": float(np.mean([a["gf_spikes"] for a in resolved])) if resolved else 0.0,
            "reaction_latency_bio_ms_median": float(np.median([a["frames_to_first_gf"] for a in with_gf]) * bio_ms) if with_gf else None,
            "theta_at_first_gf_deg_median": float(np.median([a["theta_at_first_gf_deg"] for a in with_gf])) if with_gf else None,
            "distance_at_first_gf_px_median": float(np.median([a["distance_at_first_gf_px"] for a in with_gf])) if with_gf else None,
            "frames_to_collision_at_jump": [round(a["frames_to_collision_at_jump"], 2) for a in jumped][:4000],
            "frames_to_collision_at_jump_median": float(np.median([a["frames_to_collision_at_jump"] for a in jumped])) if jumped else None,
            "cleared_by_type": {
                dc.C["obstacles"][t]["name"]: {
                    "approaches": sum(a["obstacle_type"] == t for a in resolved),
                    "p_cleared": float(np.mean([a["outcome"] == "cleared" for a in resolved if a["obstacle_type"] == t]))
                    if any(a["obstacle_type"] == t for a in resolved) else None,
                }
                for t in range(3)
            },
        }
    return out


def jump_rate_by_speed_bin(intact_rows: list[dict]) -> dict[int, float]:
    grounded, starts = {}, {}
    for r in intact_rows:
        log = [tuple(a) for a in r["actions"]]
        prev = {"jumps": 0}

        def on_frame(s, prev=prev):
            b = s.speed // 1000
            if s.jumps > prev["jumps"]:
                starts[b] = starts.get(b, 0) + 1
                prev["jumps"] = s.jumps
            if not s.jumping:
                grounded[b] = grounded.get(b, 0) + 1

        dc.replay(r["seed"], log, MAX_FRAMES, on_frame=on_frame)
    return {b: starts.get(b, 0) / n for b, n in grounded.items() if n}


def random_matched_policy_factory(rates: dict[int, float], seed: int):
    rng = np.random.default_rng(seed)
    motor = JumpMotor(MotorParams())

    def policy(s: dc.GameState) -> tuple[bool, bool]:
        p = rates.get(s.speed // 1000, 0.0)
        fire = (not s.jumping) and rng.random() < p
        return motor.update(1 if fire else 0, airborne=s.jumping), False

    return policy


# ------------------------------------------------------------------------------------------------ main
def brain_conditions(conn, lc4, lplc2, gf) -> list[tuple[str, dict]]:
    vpn = np.concatenate([lc4, lplc2])
    direct = conn.edge_mask(pre_idx=vpn, post_idx=gf)
    keep_for_strict = conn.edge_mask(pre_idx=vpn) | conn.edge_mask(post_idx=gf)
    conds: list[tuple[str, dict]] = [
        ("intact", {}),
        ("gf_ablated", {"ablate": gf}),
        ("m1_monosynaptic", {"conn": lambda: conn.with_edges(direct, suffix="direct-only")}),
    ]
    conds += [(f"shuffle_global_{i:02d}", {"conn": (lambda i=i: conn.shuffled(1000 + i))}) for i in range(N_SHUFFLE_GLOBAL)]
    conds += [
        ("m3_no_direct", {"conn": lambda: conn.with_edges(~direct, suffix="no-direct")}),
        ("gf_output_zeroed", {"conn": lambda: conn.silence(gf)}),
    ]
    conds += [
        (f"shuffle_preserve_{i:02d}", {"conn": (lambda i=i: conn.shuffled(2000 + i, preserve=keep_for_strict))})
        for i in range(N_SHUFFLE_PRESERVE)
    ]
    conds += [("intact_noise1", {"noise_seed": 1}), ("intact_noise2", {"noise_seed": 2}), ("intact_bio16.7", {"bio_ms": 16.7})]
    return conds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-hours", type=float, default=6.0)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--dev", action="store_true", help="DEV seeds, first 32 only, no pre-registration (pipeline check)")
    ap.add_argument("--only", nargs="*", help="run only these brain conditions")
    args = ap.parse_args()
    if args.plot_only:
        plot(load_result(NAME))
        return 0
    if np.isnan(FROZEN.gain_hz):
        raise SystemExit("transducer gain is not frozen yet — run experiments.looming_gf and set FROZEN")

    experiment = NAME + ("_dev" if args.dev else "")
    seeds = list(seedsets.DEV_SEEDS[:32]) if args.dev else list(seedsets.HELDOUT_200)
    if not args.dev:
        require_preregistration(PREREG_TAG, NAME, ["all conditions listed in the module docstring"], allow_dirty=True)

    import torch

    from flybrain import neurons
    from flybrain.connectome import load_connectome
    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    conn = load_connectome("flywire783")
    lc4, lplc2, gf = (neurons.indices(conn, n) for n in ("LC4", "LPLC2", "GF"))
    p = LIFParams()
    t_start = time.time()
    summaries: dict[str, dict] = {}
    skipped: list[str] = []

    for cond, spec in brain_conditions(conn, lc4, lplc2, gf):
        if args.only and cond not in args.only:
            continue
        done = load_cached(experiment, cond)
        todo = [s for s in seeds if s not in done]
        if todo and (time.time() - t_start) / 3600 > args.budget_hours:
            skipped.append(cond)
            continue
        bio_ms = spec.get("bio_ms", BIO_MS_PER_FRAME)
        if todo:
            c = spec["conn"]() if "conn" in spec else conn
            net = LIFNetwork(c, p, batch_size=min(BATCH, len(todo)), device=args.device, chunk_steps=10)
            drive = PoissonDrive(np.concatenate([lc4, lplc2]), net.b, p.dt_ms, device=args.device)
            net.set_drive(drive)
            if "ablate" in spec:
                net.ablate(spec["ablate"])
            t0 = time.time()
            play_games(
                net, drive, len(lc4), len(lplc2), gf, todo, looming=FROZEN, noise_seed=spec.get("noise_seed", 0),
                bio_ms_per_frame=bio_ms, on_result=lambda r, cond=cond: append_cached(experiment, cond, r),
            )
            print(f"[{cond}] {len(todo)} games in {time.time() - t0:.0f} s (active slots: {net.n_active})", flush=True)
            del net, drive
            torch.cuda.empty_cache()
            done = load_cached(experiment, cond)
        summaries[cond] = summarise([done[s] for s in seeds], bio_ms=bio_ms)
        st = summaries[cond]
        print(f"[{cond}] score mean {st['score_stats']['mean']:.1f} median {st['score_stats']['median']:.0f} | cleared mean "
              f"{st['cleared_stats']['mean']:.2f} | jumps/game {st['jumps_per_game']:.1f}", flush=True)

    # ---- no-brain conditions
    summaries["never_jump"] = summarise([run_policy(s, lambda _s: (False, False)) for s in seeds])
    summaries["oracle"] = summarise([run_policy(s, oracle_policy) for s in seeds])
    m0 = calibrate_m0_threshold(len(lc4), len(lplc2))
    make_m0, _ = threshold_policy_factory(m0["threshold"], len(lc4), len(lplc2), BIO_MS_PER_FRAME)
    summaries["m0_threshold"] = summarise([run_policy(s, make_m0()) for s in seeds]) | {"calibration": m0}
    if "intact" in summaries:
        intact_rows = [load_cached(experiment, "intact")[s] for s in seeds]
        rates = jump_rate_by_speed_bin(intact_rows)
        summaries["random_matched"] = summarise([run_policy(s, random_matched_policy_factory(rates, 50_000 + s)) for s in seeds])
        summaries["random_matched"]["jump_rate_per_grounded_frame_by_speed_bin"] = {str(k): v for k, v in sorted(rates.items())}
        yoked = []
        for i, s in enumerate(seeds):
            donor = intact_rows[(i + 1) % len(seeds)]
            final = dc.replay(s, [tuple(a) for a in donor["actions"]], MAX_FRAMES)
            yoked.append({"seed": s, "frames": final.frame, "score": dc.score(final), "cleared": final.cleared,
                          "crashed": final.crashed, "censored": not final.crashed, "death_type": final.death_type, "jumps": final.jumps})
        summaries["yoked"] = summarise(yoked)

    # ---- paired statistics against the intact fly
    comparisons: dict[str, dict] = {}
    if "intact" in summaries:
        base = summaries["intact"]
        for cond, st in summaries.items():
            if cond == "intact" or cond.startswith("shuffle_"):
                continue
            comparisons[cond] = {
                "cleared": paired_comparison(np.array(base["cleared"]), np.array(st["cleared"])),
                "score": paired_comparison(np.array(base["score"]), np.array(st["score"])),
            }
        adj = holm({c: v["cleared"]["wilcoxon_p"] for c, v in comparisons.items()})
        for c, padj in adj.items():
            comparisons[c]["cleared"]["wilcoxon_p_holm"] = padj
    shuffles = {}
    for fam in ("shuffle_global", "shuffle_preserve"):
        members = [st for c, st in summaries.items() if c.startswith(fam)]
        if members:
            means = [m["cleared_stats"]["mean"] for m in members]
            shuffles[fam] = {
                "n_realisations": len(members),
                "cleared_mean_per_realisation": means,
                "score_mean_per_realisation": [m["score_stats"]["mean"] for m in members],
                "jumps_per_game_per_realisation": [m["jumps_per_game"] for m in members],
                "intact_exceeds_all": bool("intact" in summaries and summaries["intact"]["cleared_stats"]["mean"] > max(means)),
            }

    compact = {c: {k: v for k, v in st.items() if k != "seeds"} for c, st in summaries.items()}
    write_result(
        experiment,
        {
            "protocol": {
                "seeds": "DEV_SEEDS[:32]" if args.dev else "HELDOUT_200", "n_seeds": len(seeds), "max_frames": MAX_FRAMES,
                "bio_ms_per_frame": BIO_MS_PER_FRAME, "dt_ms": p.dt_ms, "engine_version": dc.ENGINE_VERSION,
                "transducer_version": TRANSDUCER_VERSION, "transducer": FROZEN.__dict__, "motor": MotorParams().__dict__,
                "connectome": conn.name, "prereg_tag": None if args.dev else PREREG_TAG,
            },
            "conditions": compact,
            "comparisons_vs_intact": comparisons,
            "shuffle_families": shuffles,
            "not_yet_measured": skipped,
            "wall_hours": (time.time() - t_start) / 3600,
        },
    )
    if not args.dev:
        plot(load_result(NAME))
    return 0


# ------------------------------------------------------------------------------------------------ figure + report
LADDER = [("never_jump", "never jump"), ("m0_threshold", "M0 transducer\n+ threshold"), ("m1_monosynaptic", "M1 direct\nsynapses only"),
          ("intact", "M2 full\nconnectome"), ("m3_no_direct", "M3 full minus\ndirect"), ("oracle", "scripted\noracle")]


def plot(result: dict | None) -> None:
    if result is None:
        raise SystemExit("no results yet")
    plt = setup_matplotlib()
    conds = result["conditions"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    ax = axes[0]
    show = [("intact", C_FLY, "intact fly"), ("m0_threshold", C_ALT, "M0 threshold (no brain)"), ("random_matched", C_CTRL, "random, matched rate"),
            ("gf_ablated", C_BAD, "GF ablated"), ("oracle", C_REF, "scripted oracle")]
    for key, color, label in show:
        if key in conds:
            x = np.sort(conds[key]["score"])
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, label=label)
    ax.set(xscale="symlog", xlabel="score (held-out seeds)", ylabel="cumulative fraction of games", title="A  score distributions")
    ax.legend(fontsize=7)

    ax = axes[1]
    for key, color, label in show:
        if key in conds:
            s = conds[key]["survival"]
            ax.plot(s["frame"], s["survival"], color=color, label=label)
    ax.set(xlabel="game frame", ylabel="fraction of games still alive", title="B  survival", xscale="symlog")

    ax = axes[2]
    names, means, los, his = [], [], [], []
    for key, label in LADDER:
        if key in conds:
            st = conds[key]["cleared_stats"]
            names.append(label)
            means.append(st["mean"])
            los.append(st["mean"] - st["mean_ci95"][0])
            his.append(st["mean_ci95"][1] - st["mean"])
    colors = [C_CTRL, C_ALT, "#6baed6", C_FLY, "#9ecae1", C_REF][: len(names)]
    ax.bar(range(len(names)), means, yerr=[los, his], color=colors, capsize=3)
    for fam, marker in (("shuffle_global", "x"), ("shuffle_preserve", "+")):
        if fam in result["shuffle_families"]:
            ys = result["shuffle_families"][fam]["cleared_mean_per_realisation"]
            ax.scatter(np.full(len(ys), len(names) - 0.5 + (0.0 if fam == "shuffle_global" else 0.25)), ys, marker=marker, color=C_BAD, s=18, label=fam.replace("_", " "))
    ax.set_xticks(range(len(names)), [n.replace("\n", " ") for n in names], fontsize=7, rotation=28, ha="right")
    ax.set(ylabel="obstacles cleared per game (mean, 95% CI)", title="C  attribution ladder", yscale="symlog")
    ax.legend(fontsize=7)

    ax = axes[3]
    if "intact" in conds and conds["intact"].get("frames_to_collision_at_jump"):
        ax.hist(conds["intact"]["frames_to_collision_at_jump"], bins=40, color=C_FLY, alpha=0.85)
        ax.axvspan(6, 10, color=C_REF, alpha=0.12, label="window in which a jump clears a cactus at speed 6")
        ax.set(xlabel="frames to collision when the fly jumps", ylabel="jumps", title="D  jump timing of the intact fly")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path(NAME))
    plt.close(fig)
    print(f"[figure] wrote {figure_path(NAME)}")


def render_report(r: dict) -> str:
    conds, pr = r["conditions"], r["protocol"]
    lines = [
        f"Protocol: {pr['n_seeds']} held-out seeds ({pr['seeds']}), engine v{pr['engine_version']}, transducer v{pr['transducer_version']} "
        f"(G = {pr['transducer']['gain_hz']:g} Hz), {pr['bio_ms_per_frame']:g} ms biological time per frame, dt {pr['dt_ms']} ms, games capped at "
        f"{pr['max_frames']:,} frames, pre-registration tag `{pr['prereg_tag']}`. All comparisons are paired by seed.",
        "",
        "| condition | score mean [95% CI] | score median | max | obstacles cleared (mean) | jumps / game | capped |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["intact", "gf_ablated", "gf_output_zeroed", "random_matched", "yoked", "never_jump", "m0_threshold", "m1_monosynaptic",
             "m3_no_direct", "intact_noise1", "intact_noise2", "intact_bio16.7", "oracle"]
    for key in order:
        if key not in conds:
            continue
        st, cl = conds[key]["score_stats"], conds[key]["cleared_stats"]
        lines.append(f"| `{key}` | {st['mean']:.1f} [{st['mean_ci95'][0]:.1f}, {st['mean_ci95'][1]:.1f}] | {st['median']:.0f} | {st['max']:.0f} | "
                     f"{cl['mean']:.2f} | {conds[key]['jumps_per_game']:.1f} | {conds[key]['censored']} |")
    for fam, label in (("shuffle_global", "degree-preserving shuffle of the whole connectome"), ("shuffle_preserve", "shuffle preserving LC4/LPLC2-out and GF-in edges")):
        if fam in r["shuffle_families"]:
            f = r["shuffle_families"][fam]
            m = np.array(f["cleared_mean_per_realisation"])
            lines.append(f"| `{fam}` × {f['n_realisations']} ({label}) | score mean per realisation {np.mean(f['score_mean_per_realisation']):.1f} "
                         f"(range {min(f['score_mean_per_realisation']):.1f}–{max(f['score_mean_per_realisation']):.1f}) | | | {m.mean():.2f} "
                         f"(range {m.min():.2f}–{m.max():.2f}) | {np.mean(f['jumps_per_game_per_realisation']):.1f} | |")
    if "intact" in conds:
        i = conds["intact"]
        lines += [
            "",
            f"**The intact fly in numbers:** GF fires {i['gf_spikes_per_bio_s']:.1f} spikes per biological second "
            f"({i['gf_spikes_per_approach']:.1f} per obstacle approach; P(≥1 GF spike per approach) = {i['p_gf_spike_per_approach']:.2f}); "
            f"P(jump per approach) = {i['p_jump_per_approach']:.2f}; P(cleared | jumped) = {i['p_cleared_given_jump']:.2f}; "
            f"median reaction latency {i['reaction_latency_bio_ms_median']:.0f} ms (biological) from obstacle entering view to first GF spike, "
            f"at θ = {i['theta_at_first_gf_deg_median']:.1f}° / {i['distance_at_first_gf_px_median']:.0f} px; median jump "
            f"{i['frames_to_collision_at_jump_median']:.1f} frames before collision; jumps with nothing in view: {i['false_jump_rate']:.1%}. "
            "Deaths by obstacle: " + ", ".join(f"{k} {v}" for k, v in i["death_type_counts"].items()) + ".",
            "",
            "**Paired comparisons against the intact fly (obstacles cleared; Wilcoxon signed-rank, Holm-corrected):**",
            "",
            "| condition | mean difference intact − condition [95% CI] | p (Holm) | rank-biserial r |",
            "|---|---:|---:|---:|",
        ]
        for key, c in r["comparisons_vs_intact"].items():
            cc = c["cleared"]
            lines.append(f"| `{key}` | {cc['mean_diff']:+.2f} [{cc['mean_diff_ci95'][0]:+.2f}, {cc['mean_diff_ci95'][1]:+.2f}] | {cc.get('wilcoxon_p_holm', cc['wilcoxon_p']):.2g} | {cc['rank_biserial']:+.2f} |")
    if r["not_yet_measured"]:
        lines += ["", "Not yet measured (compute budget): " + ", ".join(f"`{c}`" for c in r["not_yet_measured"]) + "."]
    lines += ["", "![naive play](figures/naive_play.png)", "", "Source: `results/naive_play.json` (per-game records incl. action logs are cached locally, not committed)."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
