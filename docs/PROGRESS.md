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

## Phase 4 — Learning · measured (2026-09-21), see RESEARCH.md for the verdicts

**Done.** Plastic KC→MBON path in the engine (identical to the frozen network at gain 1), three-factor rule, dopamine
and context transducers, generations + held-out evaluation with ablations, two hypotheses (H1: real wiring only;
H2: mushroom-body output sets the looming gain — a documented model assumption), pre-registration tags
`prereg-phase4-h1` / `-h2`, and five diagnostics that had to come first.

**What the diagnostics found (all in RESEARCH.md, none involves a game score).**
- MBON → GF influence map: no direct synapse; 10 of 35 MBON types suppress the Giant Fiber, none excites it.
- Visual context cannot enter where the brief wanted it: all 24 configurations that drive KC-projecting *visual
  projection neurons* either fire the Giant Fiber themselves or abolish its looming response. Delivered directly to
  the 388 visual Kenyon cells the context leaves the reflex untouched and makes MBONs fire (mostly MBON27 / 09 / 32 —
  types without influence on the GF). This deviation from rule 3 is flagged as DECISIONS.md D13.
- The published model can ignite: a few frames after some crashes roughly a third of all Kenyon cells fire together and
  keep going. Our batched play loop let finished columns sit in that state and teach — fixed, regression-tested,
  counted in every result. PAM / PPL1 bursts alone do not ignite it; PPL1 bursts inside a running game do, which is why
  the shuffled-dopamine control shuffles only the reward (D20).
- A reward-bookkeeping bug (reward magnitude read from the *next* obstacle) was found and fixed before any held-out
  learning game.
- Doubling the looming gain by hand roughly doubles the naive DEV score (88.7 → 174–190), so H2 has room to show an
  effect if the mushroom body moves the gain the right way.

**Learning results.** Rendered into RESEARCH.md from `results/learning.json` (H1) and `results/learning_h2.json` (H2)
as soon as each run finishes; anything still missing there reads "not yet measured".

## Phase 5 — Crowd teaching · implemented, not yet measured

Observational replay through dopamine and the death curriculum are implemented and unit-tested on the synthetic
connectome; the ablation needs real human runs from the leaderboard (`GET /api/runs/human`).

## Phase 6 — Hardware · compiles, not yet tested on hardware

ESP32 stats display + XIAO ESP32S3 MJPEG camera firmware, laptop pose pipeline (detector unit-tested). Both firmware
projects **build** with PlatformIO Core 6.2.0 (`uvx --from platformio pio run -d firmware/<project>`, 2026-09-21, no
warnings in our code): stats display 1,018,901 bytes of flash (77.7 % of the default partition) and 48,876 bytes of RAM;
pose camera 757,501 bytes (22.7 %) and 47,296 bytes. Nothing has been flashed or run on a board yet.

## Next

1. Tair: answer the open decisions — D13 (context delivered at the Kenyon cells: accept the deviation?), D17 (is this
   the H2 you meant?), D8 (which `BIO_MS_PER_FRAME` is the headline), D1, D3, D4 — and unblock the Railway deploy (D15).
2. Flash the firmware (it compiles; pins are untested), try body control with a real camera, collect human runs with
   the local API → Phase 5 ablation (first check the replay's mid-game punishment for Kenyon-cell volleys).
3. The honest fix for Phase 4's context problem is a pixel-based motion front end (Phase 7): let the fly *see* the
   game through its own optic lobes instead of hand-written context codes.
4. Understand the post-crash Kenyon-cell volleys (which loop sustains them; does APL inhibition fail?) — a finding
   about the published model worth a short note of its own.
5. Phase 7/8: ducking via a calibrated descending neuron, MaleCNS loader, v1.0 write-up.

## How to try everything locally

The commands are in [`DEPLOY.md` § 1](DEPLOY.md) (Postgres in Docker → API → `flybrain worker` → web dev server; use Git
Bash or adapt `export` to PowerShell's `$env:`). Optional body control:
`uv run --no-sync --with mediapipe --with opencv-python python brain/pose/pose_input.py --source 0`, then tick
"play with your body" on the Play page.
