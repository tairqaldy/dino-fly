# dino-fly

**A real fruit-fly brain plays Chrome Dino.** The complete adult *Drosophila* connectome (FlyWire, ~139k neurons,
~15M connections) runs as a spiking neural network on a GPU. An approaching cactus is shown to the fly the way
neuroscientists show looming stimuli to real flies; the fly's own escape circuit (looming detectors LPLC2/LC4 →
Giant Fiber) triggers the jump.

**What makes this project different:** there is **no neural network on top of the brain**. No YOLO, no policy net,
no trained readout. Connectome weights are frozen as reconstructed. The only learning mechanism (Phase 4+) is the
fly's own: dopamine-gated plasticity at Kenyon cell → mushroom body output neuron synapses. The game→neurons and
neurons→action mappings are fixed, hand-written and documented. Every number below is produced by a script in
this repo, and failed experiments are reported as results.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Skeleton, connectome loader, our own LIF engine, reproduction of the published sugar→MN9 experiment, GPU benchmark | in progress |
| 1 | Verified neuron sets, looming→Giant Fiber experiment, deterministic game engine, naive fly vs. controls | not started |
| 2–8 | Dashboard, API + leaderboard, learning, crowd teaching, ESP32 hardware, write-up | not started |

Results so far: **not yet measured.** (This section is filled from `brain/experiments/results/*.json`; see `docs/RESEARCH.md`.)

## How it relates to other "fly brain plays X" projects

Most demos put a trained readout (an MLP, a CEM-optimised linear layer, PPO…) between the connectome and the game,
or use a small hand-picked circuit. dino-fly simulates the whole FlyWire brain with the published, validated
LIF model of Shiu et al. (2024), trains nothing in Phases 0–1, and quantifies with an *attribution ladder*
(transducer-only → monosynaptic pathway → full connectome) how much of the behaviour the brain actually contributes.

## Run it

```bash
# Python side (needs uv; GPU optional for tests, required for whole-brain experiments)
cd brain
uv sync --extra gpu          # or: --extra cpu
uv run pytest                # CPU tests on a synthetic 1k-neuron connectome
uv run flybrain download-data
uv run python -m experiments.sugar_mn9

# TypeScript side (deterministic game engine)
pnpm install && pnpm test
```

## Credits and citation

Built on the FlyWire connectome (Dorkenwald et al. 2024; Schlegel et al. 2024; CC BY 4.0) and the whole-brain model of
Shiu et al., *Nature* 634:210–219 (2024). See `CITATION.cff` and `THIRD_PARTY_NOTICES.md`. Code: MIT © Tair Kaldybayev.
