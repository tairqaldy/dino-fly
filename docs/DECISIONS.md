# Decisions

ADR-style log. Each entry: context → decision (the default we proceed with) → status. **Open** entries are
questions for Tair; work continues with the stated default until he overrides it.

The second half of this file is the **forking-paths log**: every analysis choice, and whether it was made
before or after seeing the data it affects.

---

## D1 — Run the brain natively on Windows (not WSL2) · open

The brief assumes "laptop (WSL2, NVIDIA)". The machine has no Ubuntu WSL distro (only `docker-desktop`); PyTorch
ships CUDA 13 wheels for native Windows and the RTX 5060 (Blackwell) works with them.
**Default:** native Windows now, no system changes; all code stays cross-platform (paths, LF line endings, no
Triton dependency) so WSL2/Linux works unchanged later. **Ask:** do you want a WSL2 Ubuntu setup anyway (e.g. for
`torch.compile`)?

## D2 — Validate on connectome v630, operate on v783 · decided

Every published reference number of Shiu et al. 2024 (MN9 vs. sugar-GRN rate, example spike files) was produced
with FlyWire materialization **630** (127,400 neurons). Nothing is published for 783. Root IDs differ between the
two. **Decision:** the correctness test reproduces the v630 results with our engine; all dino-fly work then uses
v783 and reports our own v783 numbers next to it.

## D3 — Annotation table: version pin and missing license · open

`flyconnectome/flywire_annotations` has **no license file**. `main` is v3.1.0 (783 root IDs, MaleCNS-cross-checked
types); the Schlegel et al. 2024 *Nature* version is tag v2.1.0.
**Default:** pin v3.1.0 by commit `8587524c`; download at run time, never redistribute the table; commit only small
derived lists of root IDs for the neuron sets we use, with citation. **Ask:** OK, or prefer v2.1.0 / asking the
authors about licensing before the public release?

## D4 — The looming transducer has one genuinely free parameter (rate scale `G`) · open

No LC4/LPLC2 firing rates in Hz are published (the data are calcium imaging and GF membrane potential). The
functional forms come from Ache et al. 2019 (LC4 ∝ angular velocity, LPLC2 Gaussian in angular size), but the
absolute rate scale cannot be taken from the literature.
**Default:** one global gain `G`, fixed by a biological criterion decided *before* any game is played (see
forking-paths log), never by game score; sensitivity to `G` is reported openly on DEV seeds.
The 42° size-tuning peak is second-hand so far (primary paper is paywalled): to be verified or labelled "our choice".

## D5 — Poisson-driven neurons have no refractory period · decided

In the published code every `PoissonInput` target gets `rfc = 0 ms`. **Decision:** keep this convention for all
driven neurons (sugar GRNs, LPLC2, LC4) so that "rate r" means the same thing as in the published model.

## D6 — MVP visual input is non-retinotopic · decided (limitation)

LPLC2/LC4 populations of both hemispheres are driven uniformly by a parameterised looming stimulus; LC4 and LPLC2
get equal peak rates so that the *connectome* sets their relative weight at the GF. A pixel-based motion-detector
front end is Phase 7.

## D7 — Game engine deviations from the brief · decided

(a) The PRNG state lives inside `GameState`, so the signature is `step(state, input)` rather than
`step(state, input, rng)` — same determinism, simpler replay/serialisation. (b) Own integer collision boxes; Phase 2
pixel art is drawn to fit them. (c) Chromium's per-frame pixel rounding and `deltaTime` scaling are deliberately not
replicated; the update order and the max-jump-height clamp are. (d) `packages/dino-core` is built in Phase 1
(not Phase 2) because Phase 1 requires golden tests of the Python port against TS fixtures.
`ENGINE_VERSION` is frozen before any experiment; changing physics bumps it and invalidates results.

## D8 — `BIO_MS_PER_FRAME` · open

The brief's default (10 ms of biological time per 60 Hz frame) makes the fly's world 1.67× faster than real time,
which scales angular velocity (the LC4 channel) by the same factor. Real time would be ≈ 16.7 ms.
**Default:** 10 ms as specified, as the primary condition; ≈ 16.7 ms reported as a secondary condition.
**Ask:** which one should be the headline?

## D9 — Compute bounds · decided

Games are capped at `MAX_FRAMES = 10,000` (≈ 167 s of game time; capped runs are reported as censored). Phase 1
experiments run in a fixed priority order within a ≈ 6–8 h GPU budget; whatever is not reached is reported as
"not yet measured". Headline numbers always use dt = 0.1 ms.

## D10 — What "GF silenced" means · decided

The published silencing (zero the neuron's *outgoing* weights) does not stop the GF from spiking, and our motor
readout *is* the GF spike. **Decision:** the control is "GF cannot spike" (Kir2.1-like ablation); the published
output-zeroing variant is also run and reported.

## D11 — Throughput expectation · decided

The brief expects the PyTorch backend to run "well above real time". The reference PyTorch implementation
(`eonsystemspbc/fly-brain`) measures 0.10–0.18× real time on an RTX 4070. We built an event-driven engine instead
and report measured numbers (see `docs/PROGRESS.md`); fast paths must pass a bitwise preflight against the
canonical eager engine.

## D12 — Pushing to the public repository · decided

Commits are pushed to `origin/main` at the end of each phase (approved with the Phase 0/1 plan).

## D13 — Visual context for the mushroom body: delivered at the Kenyon cells (deviation from rule 3) · open

KC→MBON plasticity needs Kenyon-cell activity that depends on the situation. LC4 / LPLC2 have no synapse onto any
Kenyon cell; 265 other visual projection neurons (aMe12, MTe32, MTe30, LTe25, …) do, and the brief asks for the
context to enter there. We tried (`experiments/mb_drive.py`, no game involved, selection rule written first):
9 disjoint groups, overlapping random-half codes, one code per obstacle class with a rate that ramps with angular
size; all 265 neurons or only the 47 / 30 that send ≥ 5 % / ≥ 10 % of their output to Kenyon cells. **Every**
configuration either fires the Giant Fiber by itself or abolishes its looming response (the reflex sits at
threshold: G\* = 3 Hz), usually while MBONs are still almost silent.
**Default (deviation):** the context drives the 388 visual Kenyon cells directly (`ContextParams.level = "kc"`, one
code per obstacle class, rate fixed by the same rule). This bends rule 3 ("sensory input only through identified
sensory pathways") by one synapse; it adds nothing learned, and every result stamps `context_level` into its
protocol. The visual-projection level stays implemented and is preferred automatically if a configuration ever
passes the diagnostic. **Ask:** accept this deviation, or treat "no usable visual route into the mushroom body in
this model" as the final Phase 4 answer? (A pixel-based front end, Phase 7, is the real fix.)

## D14 — Plasticity rule details that are ours · open

(a) Eligibility uses `pre · (1 + post)` so that learning also works when the MBON is silent (postsynaptic spiking is
not required for MB plasticity, Hige et al. 2015) while still favouring coincidence; `use_post=False` gives the pure
pre × dopamine rule. (b) The dopamine signal per MBON is the synapse-count-weighted sum of PAM/PPL1 spikes onto that
MBON — a proxy for compartment membership, because the connectivity table has no synapse locations. (c) All brain
columns of a training batch share one gain vector ("one fly living 32 lives in parallel"); updates are summed.
(d) Traces and updates are evaluated once per game frame (10 ms), the traces being ≥ 100 ms.

## D15 — Deployment targets · open

Web: deployed to GitHub Pages because no Cloudflare credentials exist on the dev machine; Cloudflare Pages remains
the documented target. API: Railway project, Postgres and variables are set up, but every `railway up` fails at
"scheduling build" without a build log although the image builds and runs locally. **Ask:** please look at the
build-log link in the Railway dashboard (plan limit?).

## D16 — Reproducibility of fast paths · decided

The canonical engine is the eager PyTorch path, bit-identical on CPU, GPU and under CUDA graphs, and spike-identical
to Brian2. The active-set optimisation is exact (same spikes as dense; unit-tested). A fused-kernel path was
prototyped with torch's jiterator but not adopted: it was not faster than eager at B = 64 on this GPU and its
fused multiply-add breaks bitwise equality. The plastic (KC→MBON) path accumulates floats, so learning runs are
deterministic on CPU but only statistically reproducible on GPU.

## D17 — Second hypothesis (H2): mushroom-body output sets the looming gain · open

Foreseen by the brief ("plasticity modulates the sensory gain — documented as a model assumption, still no learned
layer"). **Default:** looming gain = G · 2^clip(A₊/Ā₊ − A₋/Ā₋, −1, 1), with A the low-passed (τ = 300 ms) summed
firing of avoidance-type (glutamatergic) / approach-type (GABAergic, cholinergic) MBONs and Ā the naive brain's values
for the same class × proximity bin (`flybrain.learn.SensoryGainParams`). Choices that are ours: valence by predicted
transmitter (Aso et al. 2014), the ±1-octave range, τ, normalising by the naive response so that a naive brain plays
at exactly G (otherwise "learning" could merely undo a handicap we introduced). All were fixed before any H2 game.
Two things the pilot showed (`results/learning_h2_pilot.json`; gain shifts only, no scores): (a) the naive response
has to be measured while the naive fly really plays — a first calibration on replayed never-jump approaches was not
neutral; with the play-based baseline the naive brain's mean shift on DEV seeds is +0.01 octaves. (b) While
*training*, reward bursts excite MBONs directly (dopaminergic synapses are excitatory in the model), which lowers the
gain by ≈ 0.18 octaves on average during training games; evaluation has no bursts and is unaffected. In this
connectome the glutamatergic (avoidance-type) MBONs are practically silent under the visual context, so the mapping
is effectively one-sided: less approach-type firing → higher gain. **Ask:** is this the H2 you had in mind?

## D18 — "No dopamine" means silenced dopaminergic neurons · decided

The plasticity rule listens to PAM / PPL1 spikes, whoever caused them. The network itself makes them fire a little
(endogenous dopamine), so "the transducer delivers no bursts" is not "no dopamine". The no-DA ablation therefore
silences PAM / PPL1 (cannot spike); the script asserts that the gains stay exactly 1. Endogenous dopamine is left in
place in all other conditions — it is the connectome's own.

## D20 — What "shuffled dopamine" shuffles · decided

First version: as many bursts as the naive fly earns, at random moments, randomly PAM or PPL1. In the pilot the PPL1
bursts that landed inside a running game ignited self-sustained Kenyon-cell volleys (1,384 volley column-frames vs.
17 in the normal condition; `results/learning_pilot.json` at commit `76c98e6`), so that control would have tested
"seizures are bad", not "contingency matters". **Decision:** the control shuffles the *reward*: PAM bursts at random
moments (≈ 1 per 120 frames, magnitude uniform in [0.25, 1]) instead of after a cleared obstacle; the punishment stays
after the crash, where it is in every condition. Decided on DEV / training seeds, before any held-out learning game.

## D19 — Learning-rate calibration and training budget · decided

η is not given by the brief and no published number maps onto this model. It is set by a pilot through a synaptic
criterion (the 1 % fastest-learning synapses lose half their weight in the first generation) on DEV / never-reused
training seeds; scores are printed by the pilot but not used. Budget: 5 generations × 64 games, evaluation on the
100 forever-held-out seeds before training and after generations 1, 3 and 5 (normal) or after the last one (shuffled
dopamine), chosen to fit both hypotheses into one night of GPU time (≈ 2 brain-seconds per wall second; training
games cost about twice as much wall time as evaluation games in the pilot).

---

# Forking-paths log

| # | Choice | Made | Before / after seeing the affected data |
|---|---|---|---|
| F1 | Correctness criterion for the engine: \|ours − 65.7 Hz\| ≤ 3 Hz at 100 Hz and every point of the published 20-point curve within max(3 Hz, 3·SE) | Phase 0 plan | before |
| F2 | Seed sets (`HELDOUT_100 ⊂ HELDOUT_200`, DEV, TRAIN ranges) | Phase 0 | before |
| F3 | Transducer forms and constants: LC4 ∝ θ̇ capped at 3000°/s; LPLC2 Gaussian in θ, μ = 42°, σ = 15° (our choice); equal peak rates; expansion gating; bilateral uniform drive; eye at (38, 40) px of the sprite; `BIO_MS_PER_FRAME` = 10; motor: GF spike → full jump next frame (10-frame key hold), either GF | Phase 1 plan + `looming.py` / `motor.py` commits | before any game was played |
| F4 | Calibration rule for the single free gain: G\* = sweep value whose median first-GF-spike angular size over r/v ∈ {10,20,40,80} ms is closest to 42°, among G with P(spike) ≥ 0.5; pre-declared fallbacks | `experiments/looming_gf.py` docstring, committed before it ran | before |
| F5 | Outcome of F4: **G\* = 3 Hz** (median size 40.3°, P(spike) 0.70). Frozen as `TRANSDUCER_VERSION = 1` | `results/looming_gf.json` | — (result) |
| F6 | Pipeline check of the closed loop on 32 DEV seeds with the frozen transducer (intact, GF-ablated, M1) — *DEV seeds only*; no parameter was changed afterwards | `results/naive_play_dev.json` | after F5, before any held-out game |
| F7 | Naive-play analysis plan: conditions, 200 held-out seeds paired by seed, common random numbers, primary endpoint = obstacles cleared (and score), games capped at 10,000 frames, 20 global + 5 strict shuffles compared as distributions of per-realisation means, paired bootstrap CIs, Wilcoxon signed-rank with Holm correction, rank-biserial effect size; fixed priority order under a compute budget | `experiments/naive_play.py` docstring + git tag `prereg-phase1` | before |
| F8 | First context transducer (9 disjoint groups of the 265 KC-projecting visual neurons; rate by "≥ 10 % of recipient KCs respond" → 100 Hz) | `context.py` v1, D13 (first version) | before any data; **abandoned** after F9 |
| F9 | Manipulation checks of the context drive, no game involved: static presentation showed silent MBONs for F8; an overlapping code at 200 Hz made MBONs fire. A DEV-seed pilot with that drive (32 games) showed the naive fly jumping a median of 78 frames early (score at the never-jump floor) and mostly endogenous dopamine → the diagnostic was rebuilt around real approach sequences with a written selection rule (no context-evoked GF spike; P(GF spike) ≥ reference − 0.2; spike time within ± 3 frames; then maximal MBON rate) | `experiments/mb_drive.py` docstring | rule written **after** seeing the static diagnostic and one DEV pilot, **before** the approach-based results; no held-out game had been played with any learner |
| F10 | Outcome of F9: no visual-projection-level configuration passes (24 tried); Kenyon-cell-level context passes → 388 KCs, one code per obstacle class, 40 Hz, no ramp (D13 deviation) | `results/mb_drive.json`, `results/mb_drive_kc.json` | — (result) |
| F11 | Learning protocol: H1 / H2 definitions, conditions, 5 × 64 training games, evaluation schedule, primary endpoint (last generation vs. generation 0, paired) **and** the requirement to beat shuffled dopamine, verdict wording, η by the synaptic pilot criterion, "no dopamine" = silenced DANs | `experiments/learning.py` docstring + git tags `prereg-phase4-h1`, `prereg-phase4-h2` | before any held-out learning game. Seen beforehand: DEV pilots (scores printed, not used for any choice) |
| F12 | H2 mapping (valence by transmitter, ± 1 octave, τ = 300 ms, normalisation by the naive response) | `flybrain/learn.py`, D17 | before any H2 game |
