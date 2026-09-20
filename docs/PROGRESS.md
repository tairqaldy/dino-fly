# Progress log

Newest entries at the bottom. Each entry: what was done, what was measured, what is next. Every number quoted here
comes from `brain/experiments/results/*.json` (rendered in `docs/RESEARCH.md`).

## Phase 0 — Skeleton & sanity · done (2026-09-20/21)

**Done.** Monorepo (pnpm + uv), conventions (`CLAUDE.md`), CI, licences; pinned data manifest with SHA-256 enforcement
(FlyWire 630 + 783 as prepared by Shiu et al., published example spikes, annotation table v3.1.0; MD5s match the
Edmond archive); int64-safe connectome loader with null models and a synthetic 1k connectome; our own Brian2-faithful,
batched, event-driven LIF engine with counter-based Poisson RNG and an exact active set.

**Measured.**
- Spike-for-spike identity with real Brian2: synthetic net (CI golden test) and the **real v630 connectome**
  (2 trials × 1 s, 10,269 + 9,317 spikes, 0 mismatches).
- Published sugar → MN9 curve (v630): RMSE 0.89 Hz over 20 points; at 100 Hz ours 68.6 vs. published 65.7 Hz → inside
  the pre-registered ±3 Hz. 300 trials of ours: 67.2 ± 0.3; native Brian2 here (30 trials): 67.7 ± 0.7; the authors'
  second published run: 67.0. Per-neuron rates vs. published spikes: r = 0.9997.
- v783 (no published reference): MN9 = 68.2 Hz at 100 Hz drive; 16.8k spikes/s network-wide at 200 Hz.
- Benchmark (RTX 5060 Laptop, dense engine): 0.30× real time for one brain (0.60× with CUDA graphs), ≈ 2 brain-seconds
  per wall second for 8–64 brains. The later active-set engine played 200 games (≈ 600 frames each) in ≈ 160 s.

## Phase 1 — Innate escape · done (2026-09-21)

**Done.** Neuron-set registry + generated `NEURONS.md` (incl. measured VPN→GF wiring: 374–431 LC4 and 458–622 LPLC2
synapses per Giant Fiber in the model's connectivity vs. 2,442 / 1,366 in manual tracing); looming → GF experiment;
transducer gain frozen by the pre-declared biological rule (**G\* = 3 Hz**); deterministic game engine in TS (100 %
branch coverage) + byte-identical Python port; transducers, batched closed loop; pre-registration tag `prereg-phase1`;
held-out evaluation with controls and attribution ladder.

**Measured (200 held-out seeds, paired).** Intact fly 83.3 [77.6, 89.3] vs. never-jump 40.0; GF ablated 40.0;
rate-matched random 42.0; yoked 52.6; 20 global + 5 strict degree-preserving shuffles all 40.0. Attribution ladder:
transducer + threshold without a brain (M0) 281.5; direct VPN→GF synapses only (M1) 40.0; full connectome (M2) 83.3;
full minus direct synapses (M3) 83.2. With 16.7 ms of fly time per frame instead of 10: 153.4.

**What it means.** The wiring is necessary and specific, the jump is carried by *indirect* pathways, and the brain
currently acts as a noisy threshold on a transducer that alone would play better. Open questions for Tair are in
`docs/DECISIONS.md` (notably D8: which `BIO_MS_PER_FRAME` should be the headline).

## Phases 2–3 — Dashboard, API · done locally (2026-09-21)

Protocol package (JSON Schema + TS/pydantic mirrors, shared examples), pixel renderer, web app (play vs. ghost, live
brain map, leaderboard, research, lab), brain worker (verified live: schema-valid feed at ≈ 24 fps while sharing the
GPU with an experiment, browser pages render without console errors), API with replay-validated submissions, Postgres
schema + migrations (verified against a local Postgres), R2/disk blob store. Web app deployed to GitHub Pages.
**Not done:** Railway API deploy fails at build scheduling without a log (image builds and runs locally) — see
`docs/DEPLOY.md`; Cloudflare Pages/R2/Turnstile need credentials that are not on this machine.

## Phase 4 — Learning · in progress

Plastic KC→MBON path in the engine (identical to the frozen network at gain 1), three-factor rule, dopamine and
context transducers, generations + held-out evaluation, ablations; MBON → GF influence calibration.
Results: see `docs/RESEARCH.md` (sections stay "not yet measured" until their JSON exists).

## Phase 5 — Crowd teaching · implemented, not yet measured

Observational replay through dopamine and the death curriculum are implemented and unit-tested on the synthetic
connectome; the ablation needs real human runs from the leaderboard (`GET /api/runs/human`).

## Phase 6 — Hardware · written, not yet tested on hardware

ESP32 stats display + XIAO ESP32S3 MJPEG camera firmware, laptop pose pipeline (detector unit-tested). Nothing has
been compiled with PlatformIO or flashed yet.

## Next

1. Tair: answer the open decisions (D1, D3, D4, D8) and unblock the Railway deploy.
2. Read Phase 4 results critically; if KC→MBON plasticity cannot reach the escape circuit, test the second hypothesis
   from the brief (dopamine-gated sensory gain, as a documented model assumption) — not started.
3. Flash the firmware; wire pose events into the Play page; collect human runs → Phase 5 ablation.
4. Phase 7/8: ducking via a calibrated descending neuron, pixel-based motion front end, MaleCNS loader, v1.0 write-up.
