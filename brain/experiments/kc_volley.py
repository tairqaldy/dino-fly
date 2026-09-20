"""Phase 4 diagnostic: self-sustained Kenyon-cell volleys after crashes.

While piloting the learning experiment we saw that, a few frames after some crashes, hundreds of Kenyon cells start
firing together although nothing drives them. This script measures how often that happens, where, and whether it
is a property of the game state (follows the seed) or an artefact of the batched engine (follows the column).

Protocol: 16 never-reused TRAIN seeds (1,800,001 …) played in 8 brain columns with the full learning set-up (looming,
visual context, reward / punishment bursts) but with plasticity switched off; per column and frame we count Kenyon-cell
spikes. A *volley* frame has more than KC_VOLLEY_SPIKES_PER_FRAME spikes (the context alone gives ≈ 80). The first 8
seeds are then played again in reversed column order: every onset must reappear at the same game frame of the same
seed.

    uv run --no-sync python -u -m experiments.kc_volley
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments import da_burst, mb_drive
from experiments.common import write_result
from flybrain.learn import KC_VOLLEY_SPIKES_PER_FRAME, Learner
from flybrain.play import play_games
from flybrain.transducer.context import context_neurons
from flybrain.transducer.looming import FROZEN

NAME = "kc_volley"
SEEDS = tuple(range(1_800_001, 1_800_017))
BATCH = 8
MAX_FRAMES = 700


class Recorder(Learner):
    """A learner that never learns; it only records what the mushroom body does in every column and frame."""

    def begin(self) -> None:
        self.rows: list[dict] = []
        self.rule.enabled = False

    def extra_rates(self, games, views, tails):
        out = super().extra_rates(games, views, tails)
        self._now = {"seed": [g.seed if g else -1 for g in games], "game_frame": [g.state.frame if g else -1 for g in games],
                     "tail": [c in tails for c in range(self.b)]}
        return out

    def frame(self, watch_counts) -> None:
        super().frame(watch_counts)
        kc, _mbon, dan = (np.asarray(watch_counts)[s] for s in self.rule._sl)
        self.rows.append(self._now | {"kc_spikes": kc.sum(axis=0), "kcs_active": (kc > 0).sum(axis=0), "dan_spikes": dan.sum(axis=0)})


def onsets(rows: list[dict]) -> list[dict]:
    kc = np.array([r["kc_spikes"] for r in rows])
    hot = kc > KC_VOLLEY_SPIKES_PER_FRAME
    found = []
    for col in range(kc.shape[1]):
        for t in np.flatnonzero(hot[1:, col] & ~hot[:-1, col]) + 1:
            crash_frame = next(i for i in range(t, -1, -1) if not rows[i]["tail"][col]) + 1 if rows[t]["tail"][col] else None
            found.append({"seed": int(rows[t]["seed"][col]), "game_frame": int(rows[t]["game_frame"][col]), "in_punishment_tail": bool(rows[t]["tail"][col]),
                          "frames_after_crash": None if crash_frame is None else int(t - crash_frame),
                          "kcs_active_next_frame": int(rows[min(t + 1, len(rows) - 1)]["kcs_active"][col])})
    return sorted(found, key=lambda o: (o["seed"], o["game_frame"]))


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
    p = LIFParams()
    net = LIFNetwork(conn, p, batch_size=BATCH, device=args.device, chunk_steps=10)
    rec = Recorder(conn, net, kc=ix["KC"], mbon=ix["MBON"], pam=ix["PAM"], ppl1=ix["PPL1"], context_vpns=ctx,
                   dopamine=da_burst.chosen_dopamine(), context=context)
    drive = PoissonDrive(np.concatenate([ix["LC4"], ix["LPLC2"], rec.extra_neurons]), BATCH, p.dt_ms, device=args.device)
    net.set_drive(drive)

    def play(seeds, max_frames):
        rec.begin()
        games = play_games(net, drive, len(ix["LC4"]), len(ix["LPLC2"]), ix["GF"], list(seeds), looming=FROZEN, noise_seed=1, learner=rec, max_frames=max_frames)
        return games, rec.rows

    games, rows = play(SEEDS, MAX_FRAMES)
    found = onsets(rows)
    kc = np.array([r["kc_spikes"] for r in rows])
    dan = np.array([r["dan_spikes"] for r in rows])
    active = np.array([r["kcs_active"] for r in rows])
    tail = np.array([r["tail"] for r in rows])
    busy = np.array([[s >= 0 for s in r["seed"]] for r in rows])
    hot = kc > KC_VOLLEY_SPIKES_PER_FRAME
    print(f"[volley] {len(games)} games, {sum(g.crashed for g in games)} crashes, {len(found)} volley onsets, {int(hot.sum())} volley column-frames of "
          f"{int(busy.sum())}; inside punishment tails: {int((hot & tail).sum())}", flush=True)
    for o in found:
        print("   ", o, flush=True)

    # column invariance: the first 8 seeds again, in reversed column order, only as long as needed
    first = [o for o in found if o["seed"] in SEEDS[:BATCH]]
    horizon = max((o["game_frame"] for o in first), default=0) + 40
    _, rows_rev = play(tuple(reversed(SEEDS[:BATCH])), horizon)
    again = [o for o in onsets(rows_rev) if o["seed"] in SEEDS[:BATCH]]
    key = lambda o: (o["seed"], o["game_frame"])  # noqa: E731
    expected = sorted(key(o) for o in first if o["game_frame"] < horizon - 20)
    same = expected == sorted(key(o) for o in again if o["game_frame"] < horizon - 20)
    print(f"[volley] reversed column order: onsets {sorted(key(o) for o in again)} vs {expected} → {'identical' if same else 'DIFFERENT'}", flush=True)

    write_result(NAME, {
        "protocol": {"seeds": list(SEEDS), "batch": BATCH, "max_frames": MAX_FRAMES, "volley_threshold_kc_spikes_per_frame": KC_VOLLEY_SPIKES_PER_FRAME,
                     "n_kc": len(ix["KC"]), "context": {"level": context.level, "code": context.code, "rate_hz": context.rate_hz}},
        "games": len(games), "crashes": int(sum(g.crashed for g in games)), "onsets": found,
        "column_frames": int(busy.sum()), "volley_column_frames": int(hot.sum()), "volley_column_frames_in_punishment_tail": int((hot & tail).sum()),
        "kc_spikes_per_frame_with_context_median": float(np.median(kc[busy & ~hot & ~tail & (kc > 0)])),
        "kcs_active_per_volley_frame_median": float(np.median(active[hot])) if hot.any() else None,
        "dan_spikes_per_frame_in_volley_median": float(np.median(dan[hot])) if hot.any() else None,
        "onsets_follow_the_seed_when_columns_are_reversed": bool(same)})
    return 0


def render_report(r: dict) -> str:
    pr = r["protocol"]
    n = len(r["onsets"])
    after = [o["frames_after_crash"] for o in r["onsets"] if o["frames_after_crash"] is not None]
    lines = [
        f"{r['games']} games ({r['crashes']} crashes) with looming, visual context and dopamine bursts, plasticity off. A volley frame has more than "
        f"{pr['volley_threshold_kc_spikes_per_frame']} Kenyon-cell spikes; with the context alone the median is {r['kc_spikes_per_frame_with_context_median']:.0f}.",
        "",
        f"- **{n} volley onsets in {r['crashes']} crashes**; {r['volley_column_frames']} volley frames of {r['column_frames']:,} column-frames, "
        f"{r['volley_column_frames_in_punishment_tail']} of them inside the punishment tail after a crash"
        + (f" (onset {min(after)}–{max(after)} frames after the crash)." if after else "."),
    ]
    if n:
        lines.append(f"- In a volley frame a median of **{r['kcs_active_per_volley_frame_median']:.0f} of {pr['n_kc']:,} Kenyon cells** fire and dopaminergic "
                     f"neurons fire {r['dan_spikes_per_frame_in_volley_median']:.0f} spikes per frame on their own.")
    lines += [f"- Replaying the first {pr['batch']} seeds in reversed column order: onsets "
              + ("**reappear at the same frame of the same game** — a property of the game state, not of the batched engine."
                 if r["onsets_follow_the_seed_when_columns_are_reversed"] else "**do not reappear identically** — this needs investigation."),
              "", "Source: `results/kc_volley.json`."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
