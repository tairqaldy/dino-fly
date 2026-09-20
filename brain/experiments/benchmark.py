"""Phase 0 benchmark: biological seconds per wall-clock second of the whole-brain LIF engine on this GPU.

    uv run --no-sync python -m experiments.benchmark
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.common import write_result

NAME = "benchmark"
WARMUP_STEPS = 360
MEASURE_STEPS = 2_000


def bench(conn, grn, *, batch: int, rate_hz: float, graph: bool, chunk: int, device: str) -> dict:
    import torch

    from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

    p = LIFParams()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = LIFNetwork(conn, p, batch_size=batch, device=device, chunk_steps=chunk, use_cuda_graph=graph)
    drive = PoissonDrive(grn, batch, p.dt_ms, device=device)
    drive.set_rates(np.full(len(grn), rate_hz))
    drive.set_seeds(np.arange(batch) + 4242)
    net.set_drive(drive)
    net.run(WARMUP_STEPS, record=None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = net.run(MEASURE_STEPS, record="counts")
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    bio = MEASURE_STEPS * p.dt_ms * 1e-3
    row = {
        "path": "cuda-graph" if graph else "eager",
        "batch": batch,
        "chunk_steps": chunk,
        "drive": "rest" if rate_hz == 0 else f"sugar GRNs {rate_hz:.0f} Hz",
        "ms_per_step": wall / MEASURE_STEPS * 1e3,
        "realtime_per_brain": bio / wall,
        "realtime_aggregate": bio * batch / wall,
        "spikes_per_bio_s_per_brain": float(res.counts.sum() / batch / bio),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    print(row, flush=True)
    del net, drive
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--connectome", default="flywire783")
    args = ap.parse_args()

    import torch

    from flybrain import neurons
    from flybrain.connectome import load_connectome

    conn = load_connectome(args.connectome)
    grn = conn.index_of(neurons.SUGAR_GRN_783.ids if args.connectome == "flywire783" else neurons.SUGAR_GRN_630.ids)
    rows = []
    for graph in (False, True):
        for batch in (1, 8, 32, 64):
            for rate in (0.0, 100.0):
                rows.append(bench(conn, grn, batch=batch, rate_hz=rate, graph=graph, chunk=18, device=args.device))
    for chunk in (1, 10):  # effect of chunk size (eager, B = 32, driven)
        rows.append(bench(conn, grn, batch=32, rate_hz=100.0, graph=False, chunk=chunk, device=args.device))
    write_result(
        NAME,
        {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "connectome": conn.name,
            "neurons": conn.n,
            "edges": conn.n_edges,
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "rows": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
