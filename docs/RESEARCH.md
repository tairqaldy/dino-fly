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
Protocol: 200 held-out seeds (HELDOUT_200), engine v1, transducer v1 (G = 3 Hz), 10 ms biological time per frame, dt 0.1 ms, games capped at 10,000 frames, pre-registration tag `prereg-phase1`. All comparisons are paired by seed.

| condition | score mean [95% CI] | score median | max | obstacles cleared (mean) | jumps / game | capped |
|---|---:|---:|---:|---:|---:|---:|
| `intact` | 83.3 [77.6, 89.3] | 71 | 312 | 4.51 | 4.6 | 0 |
| `gf_ablated` | 40.0 [40.0, 40.0] | 40 | 40 | 0.00 | 0.0 | 0 |
| `gf_output_zeroed` | 83.3 [77.6, 89.3] | 71 | 312 | 4.51 | 4.6 | 0 |
| `random_matched` | 42.0 [41.4, 42.6] | 40 | 58 | 0.21 | 2.5 | 0 |
| `yoked` | 52.6 [51.3, 54.0] | 50 | 110 | 1.35 | 1.6 | 0 |
| `never_jump` | 40.0 [40.0, 40.0] | 40 | 40 | 0.00 | 0.0 | 0 |
| `m0_threshold` | 281.5 [255.9, 307.8] | 245 | 849 | 21.52 | 21.7 | 0 |
| `m1_monosynaptic` | 40.0 [40.0, 40.0] | 40 | 40 | 0.00 | 0.0 | 0 |
| `m3_no_direct` | 83.2 [77.5, 89.2] | 70 | 312 | 4.50 | 4.6 | 0 |
| `intact_noise1` | 83.7 [78.3, 89.2] | 72 | 220 | 4.64 | 4.7 | 0 |
| `intact_noise2` | 84.2 [77.8, 91.1] | 69 | 270 | 4.63 | 4.7 | 0 |
| `intact_bio16.7` | 153.4 [139.7, 167.8] | 120 | 549 | 11.14 | 11.3 | 0 |
| `oracle` | 2637.0 [2637.0, 2637.0] | 2637 | 2637 | 175.09 | 144.9 | 200 |
| `shuffle_global` × 20 (degree-preserving shuffle of the whole connectome) | score mean per realisation 40.0 (range 40.0–40.0) | | | 0.00 (range 0.00–0.00) | 0.0 | |
| `shuffle_preserve` × 5 (shuffle preserving LC4/LPLC2-out and GF-in edges) | score mean per realisation 40.0 (range 40.0–40.0) | | | 0.00 (range 0.00–0.00) | 0.0 | |

**The intact fly in numbers:** GF fires 1.0 spikes per biological second (0.9 per obstacle approach; P(≥1 GF spike per approach) = 0.83); P(jump per approach) = 0.82; P(cleared | jumped) = 0.99; median reaction latency 510 ms (biological) from obstacle entering view to first GF spike, at θ = 42.0° / 47 px; median jump 6.3 frames before collision; jumps with nothing in view: 0.0%. Deaths by obstacle: CACTUS_SMALL 114, CACTUS_LARGE 86, PTERODACTYL 0.

**Paired comparisons against the intact fly (obstacles cleared; Wilcoxon signed-rank, Holm-corrected):**

| condition | mean difference intact − condition [95% CI] | p (Holm) | rank-biserial r |
|---|---:|---:|---:|
| `gf_ablated` | +4.51 [+3.94, +5.09] | 2.1e-29 | +1.00 |
| `m1_monosynaptic` | +4.51 [+3.94, +5.09] | 2.1e-29 | +1.00 |
| `m3_no_direct` | +0.01 [-0.06, +0.10] | 1 | +0.00 |
| `gf_output_zeroed` | +0.00 [+0.00, +0.00] | 1 | +0.00 |
| `intact_noise1` | -0.13 [-0.93, +0.69] | 1 | -0.02 |
| `intact_noise2` | -0.12 [-0.98, +0.74] | 1 | -0.02 |
| `intact_bio16.7` | -6.63 [-8.08, -5.25] | 3.5e-14 | -0.66 |
| `never_jump` | +4.51 [+3.94, +5.09] | 2.1e-29 | +1.00 |
| `oracle` | -170.57 [-171.25, -169.88] | 1.6e-33 | -1.00 |
| `m0_threshold` | -17.00 [-19.02, -15.05] | 1.9e-29 | -0.97 |
| `random_matched` | +4.30 [+3.73, +4.88] | 2.3e-28 | +0.99 |
| `yoked` | +3.16 [+2.58, +3.75] | 1.8e-19 | +0.83 |

![naive play](figures/naive_play.png)

Source: `results/naive_play.json` (per-game records incl. action logs are cached locally, not committed).
<!-- END:naive_play -->

## Phase 4 — Can the mushroom body reach the escape circuit?

**Question.** KC→MBON plasticity can only matter for the game if MBON activity influences the Giant Fiber through
the real wiring. Does it, in this model?

<!-- BEGIN:mbon_influence -->
There are **0 direct MBON → GF synapses** in the connectome, so any influence is polysynaptic. Protocol: looming r/v = 40 ms through the frozen transducer (G = 3 Hz), 40 trials; each MBON type driven at 100 Hz. Baseline P(GF spike) = 0.87; a change is called real only outside ±0.20 (spread across noise seeds + 2 s.e.).

**10 of 35 MBON types** move the GF response outside the noise band; 0 can make the GF fire on their own.

| MBON type | neurons | predicted transmitter | ΔP(GF spike) with looming | P(GF spike) alone |
|---|---:|---|---:|---:|
| MBON12 | 4 | acetylcholine | -0.70 | 0.00 |
| MBON05 | 2 | acetylcholine | -0.68 | 0.00 |
| MBON18 | 2 | acetylcholine | -0.62 | 0.00 |
| MBON14 | 4 | acetylcholine | -0.53 | 0.00 |
| MBON35 | 2 | acetylcholine | -0.48 | 0.00 |
| MBON16 | 2 | acetylcholine | -0.43 | 0.00 |
| MBON20 | 2 | gaba | -0.43 | 0.00 |
| MBON13 | 2 | acetylcholine | -0.40 | 0.00 |
| MBON28 | 2 | acetylcholine | -0.38 | 0.00 |
| MBON26 | 2 | acetylcholine | -0.35 | 0.00 |

(ten largest effects shown; all 35 types are in the JSON)

![MBON → GF influence](figures/mbon_influence.png)

Source: `results/mbon_influence.json`.
<!-- END:mbon_influence -->

## Phase 4 — Can visual context reach the mushroom body without disturbing the reflex?

**Question.** The mushroom body can only associate outcomes with situations it is told about. The brief's design is
to encode a coarse visual context (obstacle class × proximity) in the visual projection neurons that synapse onto
Kenyon cells. Two things must hold for a learning experiment to mean anything: the context must make MBONs fire
(the published model has no spontaneous activity, so a silent MBON stays silent however its synapses change), and
it must leave the innate reflex alone — those visual neurons are real neurons with many other targets.
The selection rule was written into `experiments/mb_drive.py` before its results existed; no game score is involved.

<!-- BEGIN:mb_drive -->
not yet measured
<!-- END:mb_drive -->

**Reading.** Driving all 265 KC-projecting visual neurons fires the Giant Fiber by itself — the dino would jump at
the mere sight of an obstacle, up to ~70 frames too early. Restricting the drive to the 47 (or 30) neurons that are dedicated
mushroom-body inputs avoids that but *abolishes* the looming-evoked GF spike, at rates where MBONs are still nearly
silent — so the suppression does not run through the mushroom body and learning could not undo it. In this model the
escape reflex sits at threshold (the frozen transducer gain is only 3 Hz), and no visual-projection-level context
drive that we tried leaves it intact. Strong drives also make dopaminergic neurons fire from network activity alone
(table: up to several hundred spikes per approach; a full punishment burst of the transducer is 16 PPL1 neurons ×
100 Hz × 100 ms ≈ 160 spikes), which would gate plasticity regardless of what happens in the game.

**Deviation (DECISIONS.md D13), flagged for review.** To test the learning hypotheses at all, the context is delivered
one synapse further in: directly to the 388 Kenyon cells that receive input from the dedicated visual neurons. This
bends the project's rule 3 (sensory input only through identified sensory pathways); it adds no learned component.
The same diagnostic and the same rule:

<!-- BEGIN:mb_drive_kc -->
not yet measured
<!-- END:mb_drive_kc -->

**Reading.** A Kenyon-cell-level context leaves the reflex untouched and makes MBONs fire — mostly MBON27, MBON09 and
MBON32, which the influence map above found to have *no* effect on the Giant Fiber. So the prediction for H1, stated
before the learning experiment ran: the visually driven part of the mushroom body does not talk to the escape
circuit in this model, and changing its synapses should change nothing.

## Phase 4 — Can the model take a dopamine burst?

Early pilots of the learning experiment (DEV seeds) showed implausibly fast, non-specific depression of most
KC→MBON synapses, whatever the learning rate. The cause was not the rule but the network: a few frames after some
crashes a large fraction of all Kenyon cells start firing together although nothing drives them, and the state sustains
itself (the published model treats dopamine like a fast excitatory transmitter and has no adaptation, and PAM / PPL1
neurons are wired recurrently with Kenyon cells and MBONs). Measured:

<!-- BEGIN:kc_volley -->
not yet measured
<!-- END:kc_volley -->

Our play loop used to leave a finished game's brain column running until the whole batch was done — with a volley
going and endogenous dopamine gating plasticity on everything. Two consequences: (1) the loop now silences a column
as soon as its game is over and idle columns never reach the learning rule (regression test
`test_columns_without_a_game_are_silent_and_never_teach`); what remains is confined to the ≤ 15-frame punishment tail
after a crash and is counted in every learning result (`kc_volley_column_frames`). (2) We checked whether the
reward / punishment bursts themselves ignite the mushroom body:

<!-- BEGIN:da_burst -->
not yet measured
<!-- END:da_burst -->

**Reading.** Neither a PAM nor a PPL1 burst alone ignites the mushroom body, with or without the visual context, so
the transducer keeps the strongest burst of the grid. The volleys after crashes need the state a real game leaves
behind (they follow the game seed, not the batch column — the engine is column-invariant); we did not dissect the
mechanism further. They are a property of the published model that anyone building plasticity on top of it should
know about.

## Phase 4 — Does the fly learn? H1: only through the real wiring

**Hypothesis H1.** Dopamine-gated depression of KC→MBON synapses (reward = cleared obstacle → PAM burst scaled by jump
timing, punishment = crash → PPL1 burst) improves the held-out score over generations, acting only through the real
wiring MBON → … → Giant Fiber. **What would falsify it:** no improvement over generation 0, or no advantage over the
shuffled-dopamine ablation, on the same held-out seeds (criteria and analysis fixed in the docstring of
`experiments/learning.py`, git tag `prereg-phase4-h1`, ledger entry). A null result is reported as such.

The learning rate is not tuned on any score: a pilot (DEV seeds for evaluation, training seeds that are never reused)
sets η so that the 1 % fastest-learning synapses lose half their weight in the first generation.
"No dopamine" means the dopaminergic neurons cannot spike: the rule listens to PAM / PPL1 spikes whoever caused them,
and the network itself makes them fire a little (endogenous dopamine).

<!-- BEGIN:learning -->
not yet measured
<!-- END:learning -->

## Phase 4 — Second hypothesis, H2: mushroom-body output sets the looming gain

The brief foresees this step: *"If the influence is too weak to change jump timing, the honest result is 'plasticity
at KC→MBON does not reach the escape circuit strongly enough in this model' — report it, then test the second
hypothesis: plasticity modulates the sensory gain (documented as a model assumption, still no learned layer)."*

**Model assumption (not connectome).** Learned changes of mushroom-body output shift the gain of the looming pathway:
with A₊ / A₋ the low-passed (τ = 300 ms) summed firing of avoidance-type / approach-type MBONs and Ā the same
quantities in a naive brain in the same contexts,

    looming gain = G · 2^clip(A₊/Ā₊ − A₋/Ā₋, −1, +1)

so a naive brain plays at exactly G, abolishing the approach-type response doubles the gain and abolishing the
avoidance-type response halves it. Valence follows Aso et al. 2014 (eLife 3:e04580): glutamatergic MBONs promote
avoidance, GABAergic and cholinergic MBONs approach; transmitters are FlyWire's predictions. The mapping is fixed and
hand-written; the only learned quantities remain the KC→MBON gains, changed only by the dopamine-gated rule. Same
protocol, criteria and ablations as H1 (git tag `prereg-phase4-h2`).

<!-- BEGIN:learning_h2 -->
not yet measured
<!-- END:learning_h2 -->

## Phase 5 — Does teaching by humans help?

Two mechanisms are implemented (`flybrain/crowd.py`): observational replay of stored human runs with dopamine as
the only teaching signal, and a curriculum of seeds on which many humans die. Both need validated human runs from
the public leaderboard.

<!-- BEGIN:crowd_teaching -->
not yet measured
<!-- END:crowd_teaching -->
