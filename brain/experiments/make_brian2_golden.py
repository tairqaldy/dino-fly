"""Cross-check of our LIF engine against *real Brian2* — the simulator the published model was written in.

The published equations (Shiu et al. 2024, MIT) are expressed in Brian2 below; Poisson input is replaced by a
deterministic kick schedule (SpikeGeneratorGroup + delay-0 synapse `v += 68.75 mV`, which runs in the same
scheduling slot and under the same refractory write-protection as Brian2's PoissonInput), so both simulators see
bit-identical input and can be compared spike for spike. A firing-rate comparison cannot see off-by-one errors in
delay / refractory handling; this can.

Needs the optional `brian2` extra (local validation only):

    uv sync --extra gpu --extra brian2
    uv run --no-sync python -m experiments.make_brian2_golden --synthetic         # writes the golden file used by CI
    uv run --no-sync python -m experiments.make_brian2_golden --real flywire630   # spike-exact check on the real brain
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain import rng as crng
from flybrain.config import results_dir, run_metadata
from flybrain.connectome import Connectome, load_connectome, synthetic
from flybrain.lif import LIFParams

GOLDEN = Path(__file__).resolve().parents[1] / "flybrain" / "data" / "brian2_golden_spikes.npz"


def kick_table(stim: np.ndarray, n_steps: int, rate_hz: float, seed: int, dt_ms: float) -> np.ndarray:
    """bool [n_steps, n_stim] from the engine's own counter-based RNG (so the engine can reproduce it natively)."""
    lo, hi = crng.split_seed(seed)
    key = crng.stream_key(stim.reshape(1, -1).astype(np.int64), lo, hi)
    steps = np.arange(n_steps, dtype=np.int64).reshape(-1, 1)
    return crng.draw_u32(key, steps) < crng.rate_to_p_u32(rate_hz, dt_ms)


def run_brian2(conn: Connectome, stim: np.ndarray, table: np.ndarray, p: LIFParams) -> np.ndarray:
    """Returns spikes as int64 [M, 2] rows (step, neuron index), sorted."""
    import brian2 as b2

    b2.prefs.codegen.target = "numpy"  # no compiler needed; semantics are identical to the C++ targets
    b2.start_scope()
    b2.defaultclock.dt = p.dt_ms * b2.ms
    ns = {
        "v_0": p.v_rest_mv * b2.mV,
        "v_rst": p.v_reset_mv * b2.mV,
        "v_th": p.v_th_mv * b2.mV,
        "t_mbr": p.tau_m_ms * b2.ms,
        "tau": p.tau_syn_ms * b2.ms,
    }
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neu = b2.NeuronGroup(
        conn.n, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=ns
    )
    neu.v = ns["v_0"]
    neu.g = 0 * b2.mV
    neu.rfc = p.t_ref_ms * b2.ms
    neu.rfc[stim] = 0 * b2.ms  # published convention: Poisson targets have no refractory period

    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_delay_ms * b2.ms)
    syn.connect(i=conn.pre.astype(np.int64), j=conn.post.astype(np.int64))
    syn.w = conn.weight.astype(np.float64) * p.w_syn_mv * b2.mV

    steps, which = np.nonzero(table)
    gen = b2.SpikeGeneratorGroup(len(stim), which, steps * p.dt_ms * b2.ms)
    kick = b2.Synapses(gen, neu, on_pre=f"v += {p.kick_mv} * mV")
    kick.connect(i=np.arange(len(stim)), j=stim.astype(np.int64))

    mon = b2.SpikeMonitor(neu)
    net = b2.Network(neu, syn, gen, kick, mon)
    net.run(table.shape[0] * p.dt_ms * b2.ms)
    step = np.rint(np.asarray(mon.t / b2.ms) / p.dt_ms).astype(np.int64)
    out = np.stack([step, np.asarray(mon.i, dtype=np.int64)], axis=1)
    return out[np.lexsort((out[:, 1], out[:, 0]))]


def run_engine(conn: Connectome, stim: np.ndarray, table: np.ndarray, p: LIFParams, device: str) -> np.ndarray:
    import torch

    from flybrain.lif import LIFNetwork, ScheduledDrive

    net = LIFNetwork(conn, p, batch_size=1, dtype=torch.float64, device=device)
    net.set_drive(ScheduledDrive(stim, table[:, :, None], device=device))
    spikes = net.run(table.shape[0], record="spikes").spikes[:, :2]
    return spikes[np.lexsort((spikes[:, 1], spikes[:, 0]))]


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    same = a.shape == b.shape and bool(np.array_equal(a, b))
    sa, sb = {tuple(r) for r in a.tolist()}, {tuple(r) for r in b.tolist()}
    diff = sorted(sa ^ sb)
    return {
        "identical": same,
        "brian2_spikes": len(a),
        "engine_spikes": len(b),
        "mismatched_spikes": len(diff),
        "first_mismatch_step": diff[0][0] if diff else None,
    }


def synthetic_golden() -> int:
    p = LIFParams()
    conn = synthetic(n=1000, seed=0)
    g = conn.meta["groups"]
    stim = np.array(g["SYN_GRN"] + g["LPLC2"] + g["LC4"], dtype=np.int64)
    table = kick_table(stim, 3000, 120.0, seed=2024, dt_ms=p.dt_ms)
    spikes = run_brian2(conn, stim, table, p)
    active = len(np.unique(spikes[:, 1]))
    print(f"[brian2] synthetic: {len(spikes)} spikes, {active} active neurons")
    report = compare(spikes, run_engine(conn, stim, table, p, "cpu"))
    print("[compare]", report)
    # Everything needed to replay the comparison is stored, so CI depends on neither Brian2 nor RNG stream stability.
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GOLDEN,
        root_ids=conn.root_ids,
        pre=conn.pre,
        post=conn.post,
        weight=conn.weight,
        stim=stim,
        kick_steps=np.nonzero(table)[0].astype(np.int32),
        kick_which=np.nonzero(table)[1].astype(np.int32),
        n_steps=np.int64(table.shape[0]),
        spikes=spikes.astype(np.int32),
    )
    print(f"wrote {GOLDEN} ({GOLDEN.stat().st_size / 1e3:.0f} kB)")
    return 0 if report["identical"] else 1


def real_crosscheck(name: str, trials: int, n_steps: int, rate_hz: float, device: str) -> int:
    from flybrain import neurons

    p = LIFParams()
    conn = load_connectome(name)
    grn = neurons.SUGAR_GRN_630 if name == "flywire630" else neurons.SUGAR_GRN_783
    mn9 = neurons.MN9_630 if name == "flywire630" else neurons.MN9_783
    stim, mn9_idx = conn.index_of(grn.ids), conn.index_of(mn9.ids)
    rows = []
    for trial in range(trials):
        table = kick_table(stim, n_steps, rate_hz, seed=7_000 + trial, dt_ms=p.dt_ms)
        t0 = time.time()
        ref = run_brian2(conn, stim, table, p)
        t_b2 = time.time() - t0
        t0 = time.time()
        got = run_engine(conn, stim, table, p, device)
        row = compare(ref, got) | {
            "trial": trial,
            "brian2_wall_s": round(t_b2, 1),
            "engine_wall_s": round(time.time() - t0, 1),
            "mn9_spikes_brian2": [int(np.sum(ref[:, 1] == i)) for i in mn9_idx],
            "mn9_spikes_engine": [int(np.sum(got[:, 1] == i)) for i in mn9_idx],
        }
        print(row)
        rows.append(row)
    out = {
        "experiment": "brian2_crosscheck",
        "connectome": name,
        "grn_rate_hz": rate_hz,
        "n_steps": n_steps,
        "dtype": "float64",
        "all_identical": all(r["identical"] for r in rows),
        "trials": rows,
        "meta": run_metadata(),
    }
    path = results_dir() / f"brian2_crosscheck_{name}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0 if out["all_identical"] else 1


def native_rerun(name: str, trials: int, n_steps: int, rate_hz: float) -> int:
    """Re-run the *published* stochastic protocol in real Brian2 (its own PoissonInput + RNG) on this machine.

    This measures how much the published 30-trial MN9 estimate varies from run to run — the yardstick for judging
    the difference between our engine's estimate and the published number.
    """
    import brian2 as b2

    from flybrain import neurons

    p = LIFParams()
    conn = load_connectome(name)
    grn = neurons.SUGAR_GRN_630 if name == "flywire630" else neurons.SUGAR_GRN_783
    mn9 = neurons.MN9_630 if name == "flywire630" else neurons.MN9_783
    stim, mn9_idx = conn.index_of(grn.ids), conn.index_of(mn9.ids)

    b2.prefs.codegen.target = "numpy"
    b2.start_scope()
    b2.defaultclock.dt = p.dt_ms * b2.ms
    ns = {"v_0": p.v_rest_mv * b2.mV, "v_rst": p.v_reset_mv * b2.mV, "v_th": p.v_th_mv * b2.mV,
          "t_mbr": p.tau_m_ms * b2.ms, "tau": p.tau_syn_ms * b2.ms}
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neu = b2.NeuronGroup(conn.n, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                         refractory="rfc", namespace=ns)
    neu.v = ns["v_0"]
    neu.g = 0 * b2.mV
    neu.rfc = p.t_ref_ms * b2.ms
    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_delay_ms * b2.ms)
    syn.connect(i=conn.pre.astype(np.int64), j=conn.post.astype(np.int64))
    syn.w = conn.weight.astype(np.float64) * p.w_syn_mv * b2.mV
    inputs = []
    for i in stim:
        inputs.append(b2.PoissonInput(target=neu[int(i) : int(i) + 1], target_var="v", N=1, rate=rate_hz * b2.Hz,
                                      weight=p.kick_mv * b2.mV))
        neu.rfc[int(i)] = 0 * b2.ms
    mon = b2.SpikeMonitor(neu)
    net = b2.Network(neu, syn, mon, *inputs)
    net.store()
    rates, totals = [], []
    for trial in range(trials):
        net.restore()
        b2.seed(90_000 + trial)
        net.run(n_steps * p.dt_ms * b2.ms)
        idx = np.asarray(mon.i)
        rates.append([float(np.sum(idx == m)) / (n_steps * p.dt_ms * 1e-3) for m in mn9_idx])
        totals.append(int(len(idx)))
        print(f"[brian2 native] trial {trial}: MN9 {rates[-1]}, {totals[-1]} spikes", flush=True)
    r = np.asarray(rates)
    out = {
        "experiment": "brian2_native_rerun",
        "connectome": name,
        "grn_rate_hz": rate_hz,
        "n_trials": trials,
        "mn9_published_mean_hz": float(r[:, 0].mean()),
        "mn9_published_std_hz": float(r[:, 0].std()),
        "mn9_published_se_hz": float(r[:, 0].std(ddof=1) / np.sqrt(trials)),
        "mn9_partner_mean_hz": float(r[:, 1].mean()),
        "mn9_partner_std_hz": float(r[:, 1].std()),
        "total_spikes_per_trial": float(np.mean(totals)),
        "per_trial_mn9_hz": r.tolist(),
        "brian2_version": b2.__version__,
        "meta": run_metadata(),
    }
    path = results_dir() / f"brian2_native_rerun_{name}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}: MN9 = {out['mn9_published_mean_hz']:.2f} ± {out['mn9_published_se_hz']:.2f} (SE) Hz")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="(re)generate the golden spike file used by CI")
    ap.add_argument("--real", metavar="CONNECTOME", help="spike-exact cross-check on flywire630 / flywire783")
    ap.add_argument("--native", metavar="CONNECTOME", help="re-run the published stochastic protocol in Brian2")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    code = 0
    if args.synthetic:
        code |= synthetic_golden()
    if args.real:
        code |= real_crosscheck(args.real, args.trials, args.steps, args.rate, args.device)
    if args.native:
        code |= native_rerun(args.native, args.trials, args.steps, args.rate)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
