# CLAUDE.md — conventions for working in dino-fly

dino-fly: the complete FlyWire *Drosophila* connectome, run as a spiking LIF network on a GPU, plays
Chrome Dino. Author: Tair Kaldybayev (@tairqaldy). License: MIT. Language for code, commits, docs: English.

## The one rule (never break it)

1. Connectome weights stay **frozen exactly as reconstructed**, except at synapses that are plastic in the real animal.
2. The **only** learning mechanism is dopamine-gated three-factor plasticity at KC→MBON synapses. No backprop,
   no gradient descent, no policy network, no trained readout that "decodes" the brain.
3. Sensory input enters only through identified sensory pathways; motor output leaves only through identified
   descending/motor neurons. Game→neurons and neurons→action mappings are **fixed, hand-written, documented
   transducers**, never learned.
4. Every claim in README/RESEARCH is backed by a number produced by a script in this repo. A failed experiment
   is reported as a result. "The fly could not learn X" is a valid outcome.

If you catch yourself adding a learned layer between brain and game to make something work: stop and write it
up in `docs/DECISIONS.md` instead.

## Honesty protocol (enforced in code)

- **Transducer frozen before first game contact.** Forms, ratios, eye position, motor rule, `BIO_MS_PER_FRAME` and the
  single gain `G` are fixed from biology/literature, never from game score. `TRANSDUCER_VERSION` / `ENGINE_VERSION`
  and all parameters are stamped into every results JSON. Changing either bumps the version and invalidates results.
- **Pre-registration by git.** Held-out evaluation scripts refuse a dirty tree, require the `prereg-*` tag, and
  append to `brain/experiments/results/heldout_ledger.jsonl`. Analysis choices go into the forking-paths log in
  `docs/DECISIONS.md`, marked before/after seeing data.
- **Seed hygiene.** `flybrain/seeds.py`: `HELDOUT_100 ⊂ HELDOUT_200`, `DEV_SEEDS`, `TRAIN` — disjoint, unit-tested.
  Held-out seeds are only touched by final evaluation scripts. Never evaluate on training seeds.
- **Numbers only from scripts.** Each experiment is `brain/experiments/<name>.py` with fixed seeds, writing
  `brain/experiments/results/<name>.json` + `docs/figures/<name>.png`. RESEARCH.md tables are rendered from those JSONs
  (`python -m experiments.report`); CI fails when out of sync. No mock data, no placeholder numbers: unmeasured = "not yet measured".
- **Attack our own claims.** Controls and the attribution ladder (transducer-only → monosynaptic → full connectome)
  are part of every headline result.

## Layout

```
brain/        Python (uv): flybrain package (connectome, lif, rng, neurons, transducers, dino_core port, play), experiments/, tests/
packages/     TS (pnpm): dino-core = deterministic game engine (SOURCE OF TRUTH), later dino-render, protocol
apps/         later: web (Vite+React), api (Node on Railway)
firmware/     later: PlatformIO projects
docs/         ARCHITECTURE, RESEARCH, NEURONS (generated), DECISIONS (ADR + open questions), PROGRESS, figures/
scripts/      thin shell wrappers over the Python CLI
data/         git-ignored downloads (connectome, annotations)
```

## Commands

```bash
# Python (from brain/)
uv sync --extra gpu            # local GPU box (CUDA 13 wheels);  CI uses: uv sync --extra cpu
uv run pytest                  # CPU tests on the synthetic 1k connectome (what CI runs)
uv run pytest -m "gpu or data" # needs GPU + downloaded data
uv run ruff check . && uv run ruff format --check .
uv run flybrain download-data  # pinned URLs + SHA-256
uv run python -m experiments.<name>      # writes results/<name>.json + docs/figures/<name>.png
uv run python -m experiments.report --check
uv run flybrain neurons-doc --check

# TypeScript (from repo root)
pnpm install && pnpm test && pnpm lint
pnpm --filter @dino-fly/dino-core gen-fixtures   # regenerates golden fixtures (CI diffs them)
```

## Conventions

- Python 3.12, `uv`, `ruff`, `pytest`, type hints, pydantic for messages. TS strict, `biome`, `vitest`. Conventional commits.
- Every neuron set is defined **once** in `flybrain/neurons.py` with its annotation query, expected count and citation;
  `docs/NEURONS.md` is generated from it. Verify cell-type names against the annotation table before use
  (`cell_class` is empty for LC/LPLC/DN types — query `cell_type`; GF is `DNp01` / hemibrain_type `Giant Fiber`).
- FlyWire root IDs are int64 (~7.2e17 > 2^53). Never let them become floats, JS numbers or JSON numbers — use
  int64 / strings. IDs differ between materializations 630 and 783.
- Never commit data files > 10 MB, checkpoints, secrets or `.env`. Commit `.env.example`.
- GPU-only code stays isolated; CI (CPU) runs everything on the synthetic connectome (`pytest` markers `gpu`, `data`, `brian2`).
- The canonical LIF engine is the eager PyTorch path (plain IEEE ops, Brian2-faithful semantics). Any fast path must pass
  the bitwise eager-vs-fast preflight before an experiment uses it. Headline numbers use dt = 0.1 ms.
- The TS game engine is the source of truth; `flybrain/dino_core.py` must match its golden fixtures hash-for-hash.
  Physics code uses only `idiv`/`imod` on integers — no floats, no raw `/` `%` `Math.round` `|0`. Constants live once in
  `packages/dino-core/constants.json`.
- When biology is uncertain, say so in the code comment and the doc, and choose the option that is easiest to falsify.
- Small PR-sized commits per phase step; append to `docs/PROGRESS.md` as you go. Each phase ends with tests green,
  a PROGRESS entry and a commit. Open questions for Tair go to `docs/DECISIONS.md` with a recommended default.
- Third-party code: `eonsystemspbc/fly-brain` is GPL — read it, never copy it. Shiu et al.'s original code is MIT.

## Environment notes (Tair's laptop)

Windows 11 native (no WSL Ubuntu distro), RTX 5060 Laptop 8 GB (Blackwell → needs CUDA ≥ 12.8 wheels; we use cu130),
uv-managed CPython 3.12, Node 24 + pnpm 10. No Triton on Windows by default: do not depend on `torch.compile`.
Long GPU jobs run in the background with resumable per-seed caches under `brain/experiments/cache/` (git-ignored).

- Always `uv run --no-sync …` inside `brain/` (a plain `uv run` re-syncs without the torch extra and uninstalls torch).
  Experiments: `PYTHONUTF8=1 uv run --no-sync python -u -m experiments.<name>`; logs go to `brain/experiments/logs/`.
- One GPU experiment at a time. Two concurrent runs halve the speed and overwrite each other's checkpoints — check
  for a running `experiments.*` process before launching.
- Firmware builds without installing PlatformIO: `uvx --from platformio pio run -d firmware/<project>`.

## Lessons that cost a night (Phase 4)

- **Check the manipulation before the experiment.** Any new drive into the brain gets a no-game diagnostic first
  (`experiments/mb_drive.py`, `da_burst.py`, `kc_volley.py`): does it reach its target, does it leave the looming
  reflex alone, does it make dopaminergic neurons or the Giant Fiber fire by itself? The reflex sits at threshold
  (G = 3 Hz); almost everything disturbs it.
- **The published model can ignite.** Dopamine is a fast excitatory transmitter in it and nothing adapts: after some
  crashes ~1/3 of all Kenyon cells fire in self-sustained volleys. Never leave a brain column running after its game
  is over, never let idle columns reach the learning rule, never deliver PPL1 bursts inside a running game.
- **Learning rates come from a synaptic criterion in a pilot, never from a score**; pilots use DEV seeds and TRAIN
  seeds ≥ 800,000 and never play an evaluation game.
- Held-out learning runs need their own tag (`prereg-phase4-h1`, `-h2`, …) and a clean tree; commit the pilot JSON
  first, because it fixes η.
