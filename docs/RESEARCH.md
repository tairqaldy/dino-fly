# Research log: can a fruit-fly brain play Chrome Dino?

Every number in this document is rendered from a JSON file written by a script in `brain/experiments/`
(`python -m experiments.report` fills the blocks between `BEGIN/END` markers; CI fails if they are stale).
Anything without a number is **not yet measured**.

## Rules of the game

1. Connectome weights are frozen as reconstructed (FlyWire 783); 2. the only learning mechanism (Phase 4+) is
dopamine-gated plasticity at KC→MBON synapses; 3. input only via identified sensory pathways, output only via
identified descending neurons, through fixed hand-written transducers; 4. failures are results.

## Phase 0 — Is our engine the published model?

**Question.** Does our PyTorch LIF engine reproduce the published whole-brain model (Shiu et al. 2024)?

**Method.** (i) Semantic exactness: spike-for-spike agreement with real Brian2 on a synthetic network with
deterministic input (covers the off-by-one cases a firing-rate comparison cannot see). (ii) The paper's own
experiment: 21 labellar sugar GRNs driven with Poisson input at 10–200 Hz, 30 trials × 1 s, read out at the MN9
proboscis motor neuron, on the connectome version the paper used (v630), compared with the paper's Fig. 1d source
data; plus per-neuron firing rates across the whole network against the published example spike files.
(iii) The same protocol on v783, the connectome everything else in this project uses (no published reference exists).

<!-- BEGIN:sugar_mn9 -->
**Result 1 — spike-exact equivalence with Brian2.**
On the real v630 connectome (127,400 neurons, 14.7 M connections), real Brian2 and our engine were fed the same kick schedule (100 Hz on the 21 sugar GRNs, 10,000 steps, float64): **identical spike trains, 0 mismatches** in 2 trials (10,269, 9,317 spikes). Brian2 (numpy target) needed 17.8 s per trial, our engine 3.4 s. The same check on a synthetic 1k-neuron network with autapses, duplicate edges and strong inhibition is a unit test in CI (`tests/test_lif_brian2_golden.py`). Source: `results/brian2_crosscheck_flywire630.json`.

**Result 2 — the published sugar → MN9 curve (connectome v630, 30 trials × 1 s).**

| GRN rate (Hz) | MN9 published | MN9 ours | diff | MN9 partner published | MN9 partner ours |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 |
| 40 | 4.8 | 5.2 | +0.3 | 3.7 | 4.0 |
| 50 | 19.4 | 18.4 | -1.0 | 16.0 | 14.6 |
| 60 | 36.4 | 37.3 | +0.9 | 27.1 | 28.4 |
| 80 | 58.1 | 58.8 | +0.6 | 42.6 | 44.6 |
| 100 | 65.7 | 68.6 | +2.9 | 49.7 | 49.4 |
| 120 | 75.0 | 74.9 | -0.1 | 52.0 | 53.9 |
| 140 | 80.9 | 81.1 | +0.2 | 56.4 | 56.2 |
| 150 | 83.7 | 83.1 | -0.6 | 60.1 | 57.3 |
| 160 | 84.6 | 85.2 | +0.6 | 59.5 | 60.6 |
| 180 | 89.2 | 88.5 | -0.7 | 62.5 | 59.5 |
| 200 | 93.2 | 94.0 | +0.8 | 62.9 | 63.9 |

RMSE over the 20-point curve: 0.89 Hz (MN9), 1.31 Hz (partner); largest deviation 2.93 Hz; 20/20 points within the pre-registered band.
Pre-registered criterion at 100 Hz (|ours − 65.7| ≤ 3 Hz): ours 68.63 Hz, diff +2.93 Hz → **PASS**.
Context for that number: the paper's Fig. 1d run gives 65.7 Hz and the authors' own second published 100 Hz run (`results/example/sugarR_100Hz.parquet`) gives 67.03 Hz; re-running the published stochastic protocol in real Brian2 on this machine (30 trials) gives 67.73 ± 0.68 Hz (mean ± SE); our engine with 300 trials gives 67.20 ± 0.28 Hz. Given Result 1, remaining differences are sampling noise of 30-trial estimates, not model differences.

**Result 3 — every neuron, not just MN9 (v630 vs. the published example spike files).**

| GRN rate | active neurons (published / ours) | Jaccard | Pearson r of rates | spikes per trial (published / ours) |
|---:|---:|---:|---:|---:|
| 100 Hz | 404 / 409 | 0.936 | 0.9997 | 9,636 / 9,681 |
| 200 Hz | 448 / 454 | 0.961 | 0.9998 | 17,052 / 17,016 |

**Result 4 — connectome v783, the one dino-fly uses (no published reference exists).**
MN9 at 100 Hz sugar drive: **68.2 Hz** (partner 50.8 Hz); at 200 Hz: 92.4 Hz. Network-wide 16,848 spikes per simulated second at 200 Hz across 448 active neurons of 138,639 (a third-party Brian2-CPU benchmark of the same condition reports ≈ 17,050). 

![sugar → MN9](figures/sugar_mn9.png)

Sources: `results/sugar_mn9.json`, `flybrain/reference/shiu2024_fig1d.json` (script-extracted from doi:10.17617/3.CZODIW).
<!-- END:sugar_mn9 -->

## Phase 0 — How fast is it?

<!-- BEGIN:benchmark -->
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; torch 2.14.0+cu130; connectome flywire783 (138,639 neurons, 15,091,983 connections); dt = 0.1 ms; float32.

| engine path | brains (B) | drive | ms / step | biological s per wall s (per brain) | … (all brains) | peak VRAM (GB) |
|---|---:|---|---:|---:|---:|---:|
| eager | 1 | rest | 0.332 | 0.301 | 0.30 | 0.21 |
| eager | 1 | sugar GRNs 100 Hz | 0.373 | 0.268 | 0.27 | 0.21 |
| eager | 8 | rest | 0.355 | 0.281 | 2.25 | 0.40 |
| eager | 8 | sugar GRNs 100 Hz | 0.410 | 0.244 | 1.95 | 0.40 |
| eager | 32 | rest | 1.503 | 0.067 | 2.13 | 1.04 |
| eager | 32 | sugar GRNs 100 Hz | 1.630 | 0.061 | 1.96 | 1.04 |
| eager | 64 | rest | 3.573 | 0.028 | 1.79 | 1.88 |
| eager | 64 | sugar GRNs 100 Hz | 3.847 | 0.026 | 1.66 | 1.90 |
| cuda-graph | 1 | rest | 0.167 | 0.599 | 0.60 | 0.24 |
| cuda-graph | 1 | sugar GRNs 100 Hz | 0.262 | 0.382 | 0.38 | 0.24 |
| cuda-graph | 8 | rest | 0.343 | 0.292 | 2.34 | 0.59 |
| cuda-graph | 8 | sugar GRNs 100 Hz | 0.452 | 0.221 | 1.77 | 0.59 |
| cuda-graph | 32 | rest | 1.468 | 0.068 | 2.18 | 1.81 |
| cuda-graph | 32 | sugar GRNs 100 Hz | 1.727 | 0.058 | 1.85 | 1.81 |
| cuda-graph | 64 | rest | 3.569 | 0.028 | 1.79 | 3.43 |
| cuda-graph | 64 | sugar GRNs 100 Hz | 3.631 | 0.028 | 1.76 | 3.43 |
| eager | 32 | sugar GRNs 100 Hz | 2.788 | 0.036 | 1.15 | 0.66 |
| eager | 32 | sugar GRNs 100 Hz | 1.728 | 0.058 | 1.85 | 0.86 |

Source: `results/benchmark.json`.
<!-- END:benchmark -->

## Phase 1 — Does looming reach the Giant Fiber?

<!-- BEGIN:looming_gf -->
Protocol: r/v ∈ {10, 20, 40, 80} ms, 10° → 63°, 20 trials each, all 104 LC4 and 210 LPLC2 neurons driven (both sides), connectome flywire783, dt 0.1 ms.

| G (Hz) | P(GF spike) overall | median θ at first spike (deg) | P(spike) per r/v = 10 / 20 / 40 / 80 ms |
|---:|---:|---:|---|
| 2 | 0.40 | 49.4 | 0.05 / 0.30 / 0.40 / 0.85 |
| 3 | 0.70 | 40.3 | 0.35 / 0.60 / 0.90 / 0.95 |
| 5 | 0.94 | 29.3 | 0.80 / 0.95 / 1.00 / 1.00 |
| 7 | 0.99 | 26.3 | 0.95 / 1.00 / 1.00 / 1.00 |
| 10 | 1.00 | 18.7 | 1.00 / 1.00 / 1.00 / 1.00 |
| 15 | 1.00 | 15.6 | 1.00 / 1.00 / 1.00 / 1.00 |
| 20 | 1.00 | 12.9 | 1.00 / 1.00 / 1.00 / 1.00 |
| 30 | 1.00 | 11.9 | 1.00 / 1.00 / 1.00 / 1.00 |
| 50 | 1.00 | 11.4 | 1.00 / 1.00 / 1.00 / 1.00 |
| 75 | 1.00 | 11.1 | 1.00 / 1.00 / 1.00 / 1.00 |
| 100 | 1.00 | 10.8 | 1.00 / 1.00 / 1.00 / 1.00 |
| 150 | 1.00 | 10.6 | 1.00 / 1.00 / 1.00 / 1.00 |
| 200 | 1.00 | 10.6 | 1.00 / 1.00 / 1.00 / 1.00 |

**Calibration (calibrated): G\* = 3 Hz** — median angular size at the first GF spike 40.3° (target 42°), P(spike) = 0.70. This value is frozen as `TRANSDUCER_VERSION = 1` before any game is played.

At G\*:

| r/v (ms) | P(spike) | GF spikes / trial | first spike: ms after onset | ms before collision | θ at first spike, median [IQR] |
|---:|---:|---:|---:|---:|---|
| 10 | 0.30 | 0.3 | 117 | -2 | 63.0° [63.0, 63.0] |
| 20 | 0.45 | 0.5 | 202 | 27 | 63.0° [58.7, 63.0] |
| 40 | 0.95 | 1.1 | 348 | 109 | 40.2° [35.7, 55.4] |
| 80 | 1.00 | 2.2 | 646 | 269 | 33.1° [28.4, 35.0] |

- **Which input matters:** LC4 only → P(spike) 0.00; LPLC2 only → 0.66; both → 0.68.
- **Wrong cell types (same drive into LC6 + LPLC1):** P(GF spike) = 0.05.
- **Specificity of the readout:** of 1,299 descending neurons, 18 fire at all under this drive; the two GFs rank 6 and 1 by spike probability (P = 0.17, 0.68).
- **Size-threshold behaviour (GF spiking disabled):** angular size at the peak of the GF membrane potential: 63° (r/v 10 ms), 63° (r/v 20 ms), 47° (r/v 40 ms), 44° (r/v 80 ms). In real flies the GF response peaks at a roughly constant angular size across r/v (Ache et al. 2019).
- **Sensitivity to our choice σ = 15°:** median size at first spike 45.6° (σ = 10°) and 19.3° (σ = 25°).

![looming → GF](figures/looming_gf.png)

Source: `results/looming_gf.json`.
<!-- END:looming_gf -->

## Phase 1 — How does the naive fly play?

<!-- BEGIN:naive_play -->
not yet measured
<!-- END:naive_play -->

## Phase 4 — Can the mushroom body reach the escape circuit?

**Question.** KC→MBON plasticity can only matter for the game if MBON activity influences the Giant Fiber through
the real wiring. Does it, in this model?

<!-- BEGIN:mbon_influence -->
not yet measured
<!-- END:mbon_influence -->

## Phase 4 — Does the fly learn?

**Hypothesis H1.** Dopamine-gated depression of KC→MBON synapses (reward = cleared obstacle → PAM, punishment = crash →
PPL1), with a coarse visual context delivered to Kenyon cells, improves the held-out score over generations.
**What would falsify it:** no improvement beyond the shuffled-dopamine and random-plasticity ablations on the same
held-out seeds. A null result is reported as such.

<!-- BEGIN:learning -->
not yet measured
<!-- END:learning -->

## Phase 5 — Does teaching by humans help?

Two mechanisms are implemented (`flybrain/crowd.py`): observational replay of stored human runs with dopamine as
the only teaching signal, and a curriculum of seeds on which many humans die. Both need validated human runs from
the public leaderboard.

<!-- BEGIN:crowd_teaching -->
not yet measured
<!-- END:crowd_teaching -->
