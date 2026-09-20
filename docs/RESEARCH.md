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
not yet measured
<!-- END:sugar_mn9 -->

## Phase 0 — How fast is it?

<!-- BEGIN:benchmark -->
not yet measured
<!-- END:benchmark -->

## Phase 1 — Does looming reach the Giant Fiber?

<!-- BEGIN:looming_gf -->
not yet measured
<!-- END:looming_gf -->

## Phase 1 — How does the naive fly play?

<!-- BEGIN:naive_play -->
not yet measured
<!-- END:naive_play -->
