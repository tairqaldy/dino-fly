"""Every neuron set dino-fly touches is defined here, once.

Two kinds of sets:
  * `PublishedIds` — literal FlyWire root IDs taken from a publication (with the materialization they belong to).
  * `NeuronSet`   — an annotation query (cell type etc.) resolved against the pinned FlyWire annotation table.

`docs/NEURONS.md` is generated from this module (`flybrain neurons-doc`). Root IDs are int64 > 2^53: keep them as
Python ints / int64 / strings, never floats or JSON numbers. IDs differ between materializations 630 and 783.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublishedIds:
    name: str
    materialization: str  # "630" | "783"
    ids: tuple[int, ...]
    citation: str
    rationale: str


_SHIU = "Shiu et al., Nature 634:210-219 (2024), doi:10.1038/s41586-024-07763-9; example.ipynb / figures.ipynb (MIT)"

# The 21 labellar sugar-sensing gustatory receptor neurons of the paper's example (`neu_sugar`), materialization 630.
SUGAR_GRN_630 = PublishedIds(
    name="SUGAR_GRN_630",
    materialization="630",
    ids=(
        720575940624963786,
        720575940630233916,
        720575940637568838,
        720575940638202345,
        720575940617000768,
        720575940630797113,
        720575940632889389,
        720575940621754367,
        720575940621502051,
        720575940640649691,
        720575940639332736,
        720575940616885538,
        720575940639198653,
        720575940620900446,
        720575940617937543,
        720575940632425919,
        720575940633143833,
        720575940612670570,
        720575940628853239,
        720575940629176663,
        720575940611875570,
    ),
    citation=_SHIU,
    rationale="Input population of the published sugar → MN9 experiment, our engine-correctness test.",
)

# Same neurons in materialization 783: 20 IDs are unchanged, one root ID was superseded
# (…620900446 → …639259967; the new ID carries the same annotation as the others: gustatory / sugar/water / LB3).
SUGAR_GRN_783 = PublishedIds(
    name="SUGAR_GRN_783",
    materialization="783",
    ids=tuple(720575940639259967 if i == 720575940620900446 else i for i in SUGAR_GRN_630.ids),
    citation=_SHIU + "; 630→783 remap of one ID as used by the 783 benchmarks of eonsystemspbc/fly-brain",
    rationale="Sugar GRNs on the connectome dino-fly actually uses.",
)

# MN9 proboscis motor neurons. The paper reads out …660219265 (the one contralateral to the stimulated GRNs).
# Note: the paper's notebooks call it "left", the FlyWire annotation table says side=right (FAFB mirror issue);
# we therefore name them by role, not by side.
MN9_630 = PublishedIds(
    name="MN9_630",
    materialization="630",
    ids=(720575940660219265, 720575940645521262),
    citation=_SHIU,
    rationale="Readout of the published experiment: (published MN9, its bilateral partner).",
)
MN9_783 = PublishedIds(
    name="MN9_783",
    materialization="783",
    ids=(720575940660219265, 720575940618238523),
    citation=_SHIU + "; partner ID in 783 cross-checked in the annotation table (motor / ingestion_motor_neuron / CB0701, nerve PhN)",
    rationale="Same two motor neurons in materialization 783 (the partner's root ID changed).",
)

PUBLISHED: tuple[PublishedIds, ...] = (SUGAR_GRN_630, SUGAR_GRN_783, MN9_630, MN9_783)
