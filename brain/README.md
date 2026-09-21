# flybrain

The Python half of [dino-fly](../README.md): the FlyWire *Drosophila* connectome run as a spiking LIF network that
plays Chrome Dino.

- `flybrain/` — connectome loader, the Brian2-faithful event-driven LIF engine, the fixed sensory/motor transducers,
  KC→MBON plasticity, the byte-identical port of the TypeScript game engine, and the live worker.
- `experiments/` — every number in the research log: each script writes `experiments/results/<name>.json` and a figure,
  and `docs/RESEARCH.md` is rendered from those files.
- `tests/` — CPU tests on a synthetic 1k-neuron connectome; the `gpu`, `data` and `brian2` markers are skipped in CI.

```bash
uv sync --extra gpu                  # or --extra cpu
uv run --no-sync pytest
uv run --no-sync flybrain download-data
uv run --no-sync flybrain worker     # the fly plays live on ws://localhost:8765
```

`Dockerfile` builds this package as a CPU service (the hosted brain on Railway); see `docs/DEPLOY.md`.
