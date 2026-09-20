# Third-party notices

dino-fly's own code is MIT-licensed (see `LICENSE`). It builds on the following data, models and references.
No third-party data files are committed to this repository; they are downloaded by `flybrain download-data`
from pinned URLs with recorded SHA-256 checksums (`brain/flybrain/data_manifest.py`).

## Data

| What | Source | License / terms | How we use it |
|---|---|---|---|
| FlyWire connectome, materializations 630 and 783 (completeness CSV + connectivity parquet as prepared by Shiu et al.) | <https://github.com/philshiu/Drosophila_brain_model> (commit `91bdd1e`), mirrored byte-identically in <https://github.com/eonsystemspbc/fly-brain> (commit `a3db62f`), archived at <https://doi.org/10.17617/3.CZODIW> | FlyWire data: CC BY 4.0. Credit: FlyWire Consortium — Dorkenwald et al., *Nature* 2024 (doi:10.1038/s41586-024-07558-y); neurotransmitter predictions: Eckstein et al., *Cell* 2024 | Downloaded at run time; never redistributed |
| FlyWire neuron annotations (`Supplemental_file1_neuron_annotations.tsv`, v3.1.0) | <https://github.com/flyconnectome/flywire_annotations> (commit `8587524c`) | The repository carries **no license file**; the table was published as supplemental data of Schlegel et al., *Nature* 2024 (CC BY 4.0 article). Cite Schlegel et al. 2024, Dorkenwald et al. 2024, Matsliah et al. 2024, Berg et al. 2025 | Downloaded at run time; never redistributed. We commit only small derived lists of FlyWire root IDs for the neuron sets we use (`brain/flybrain/data/neuron_sets.lock.json`), with attribution. See `docs/DECISIONS.md` D3 |
| Published reference results of the Shiu et al. model (Fig. 1d rates, example spike outputs) | <https://doi.org/10.17617/3.CZODIW> and the GitHub repo above | MIT (Copyright (c) 2023 Philip Shiu and Nico Spiller) | A 20-point reference curve is committed as `brain/flybrain/reference/shiu2024_fig1d.json` with provenance; a script re-extracts it from the archive |

## Model

- **Shiu et al. 2024 whole-brain LIF model** — Shiu, P. K. et al. "A Drosophila computational brain model reveals
  sensorimotor processing", *Nature* 634:210–219 (2024), doi:10.1038/s41586-024-07763-9. Original code MIT
  (Copyright (c) 2023 Philip Shiu and Nico Spiller). `brain/flybrain/lif.py` is our own re-implementation of the
  published model; `brain/experiments/make_brian2_golden.py` expresses the same published equations in Brian2 for
  validation.
- **`eonsystemspbc/fly-brain`** is GPL-2.0-or-later. We read it to understand data layout and pitfalls.
  **No code from it is copied into this repository.**
- **Brian2** (CeCILL 2.1) is used only as an optional, local validation dependency; its scheduling and refractory
  semantics are the reference our engine is tested against.

## Game

- **Chromium "Dino" game** (`components/neterror/resources/dino_game/`), Copyright The Chromium Authors, BSD-3-Clause.
  We consulted it for physics constants and feel. `packages/dino-core` is our own implementation with our own
  (future) pixel art; no Chromium code or sprites are included.

## Literature used for the fixed transducers

Cited in `docs/NEURONS.md` and `docs/RESEARCH.md`: von Reyn et al. 2014 (*Nat Neurosci*), Klapoetke et al. 2017 (*Nature*),
von Reyn et al. 2017 (*Neuron*), Ache et al. 2019 (*Curr Biol*), Jang et al. 2023 (*J Exp Biol*), Gabbiani et al. 1999 (*J Neurosci*).
