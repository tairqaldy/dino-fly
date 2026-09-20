# Progress log

Newest entries at the bottom. Each entry: what was done, what was measured, what is next. Numbers quoted here come
from `brain/experiments/results/*.json`.

## Phase 0 — Skeleton & sanity (in progress)

**Done so far**
- Monorepo skeleton (pnpm + uv), conventions (`CLAUDE.md`), CI, licences and third-party notices.
- Pinned data manifest with SHA-256 enforcement; FlyWire 630 + 783 (Shiu et al. preparation), the published example
  spike files and the FlyWire annotation table v3.1.0 download and verify (incl. the MD5s published by the Edmond archive).
- `flybrain.connectome`: int64-safe loader with invariants, published silencing, degree-preserving null models,
  synthetic 1k-neuron connectome for CPU tests.
- `flybrain.lif`: our own Brian2-faithful, batched, event-driven LIF engine with counter-based Poisson RNG.
  Verified spike-for-spike against an independent dense reference simulator; invariant to chunk size, batch size,
  column recycling; CPU = GPU = CUDA-graph bit-for-bit.
- `packages/dino-core` first slice: integer-only mulberry32 + floor-division helpers, 100 % covered.

**Measured:** not yet measured (correctness experiment and benchmark are next).
