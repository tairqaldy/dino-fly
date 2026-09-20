"""Every neuron set dino-fly touches is defined here, once.

Two kinds of sets:
  * `PublishedIds` — literal FlyWire root IDs taken from a publication (with the materialization they belong to).
  * `NeuronSet`   — an annotation query (cell type etc.) resolved against the pinned FlyWire annotation table.

`docs/NEURONS.md` is generated from this module (`flybrain neurons-doc`). Root IDs are int64 > 2^53: keep them as
Python ints / int64 / strings, never floats or JSON numbers. IDs differ between materializations 630 and 783.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


# ------------------------------------------------------------------------------ annotation-driven sets
ANNOTATION_SOURCE = (
    "flyconnectome/flywire_annotations v3.1.0 (commit 8587524c), Supplemental_file1_neuron_annotations.tsv; "
    "Schlegel et al., Nature 2024, doi:10.1038/s41586-024-07686-5"
)


@dataclass(frozen=True)
class NeuronSet:
    """A set of neurons defined by a pandas query on the FlyWire annotation table (materialization 783)."""

    name: str
    query: str
    expected_count: int
    role: str  # "sensory drive" | "motor readout" | "candidate" | "inventory"
    citation: str
    rationale: str


_LOOMING = (
    "Klapoetke et al., Nature 551:237 (2017); von Reyn et al., Neuron 94:1190 (2017); "
    "Ache et al., Curr Biol 29:1073 (2019)"
)
_MB = "Li et al., eLife 9:e62576 (2020)"
_DN_CANDIDATE = "Descending neuron in the looming/takeoff neighbourhood of the GF; alternative readout candidate."

SETS: tuple[NeuronSet, ...] = (
    NeuronSet(
        "LPLC2",
        "cell_type == 'LPLC2'",
        210,
        "sensory drive",
        _LOOMING,
        "Looming-selective visual projection neurons (radial-motion opponency); encode the angular SIZE of an "
        "approaching object and synapse directly onto the Giant Fiber. Driven by the looming transducer.",
    ),
    NeuronSet(
        "LC4",
        "cell_type == 'LC4'",
        104,
        "sensory drive",
        _LOOMING,
        "Visual projection neurons encoding the angular VELOCITY of a looming object; direct input to the Giant "
        "Fiber. Driven by the looming transducer.",
    ),
    NeuronSet(
        "GF",
        "cell_type == 'DNp01'",
        2,
        "motor readout",
        "von Reyn et al., Nat Neurosci 17:962 (2014); annotated hemibrain_type 'Giant Fiber'",
        "Giant Fiber descending neurons (DNp01), one per side. A single GF spike triggers the short-mode escape "
        "takeoff; in dino-fly a GF spike is the JUMP command. The table's neurotransmitter prediction for the GF is "
        "low-confidence and differs between sides (ACh 0.32 / Glu 0.27); we only read its spikes.",
    ),
    NeuronSet(
        "LPLC1",
        "cell_type == 'LPLC1'",
        140,
        "candidate",
        _LOOMING,
        "Looming-responsive VPN without direct GF input; specificity control (same drive, wrong cell type).",
    ),
    NeuronSet(
        "LC6",
        "cell_type == 'LC6'",
        125,
        "candidate",
        "Wu et al., eLife 5:e21022 (2016)",
        "Looming-responsive VPN linked to takeoff when activated; specificity control and alternative pathway.",
    ),
    NeuronSet("DNp02", "cell_type == 'DNp02'", 2, "candidate", _LOOMING, _DN_CANDIDATE),
    NeuronSet("DNp04", "cell_type == 'DNp04'", 2, "candidate", _LOOMING, _DN_CANDIDATE),
    NeuronSet("DNp06", "cell_type == 'DNp06'", 2, "candidate", _LOOMING, _DN_CANDIDATE),
    NeuronSet("DNp11", "cell_type == 'DNp11'", 2, "candidate", _LOOMING, _DN_CANDIDATE),
    NeuronSet(
        "DN_ALL",
        "super_class == 'descending'",
        1303,
        "inventory",
        ANNOTATION_SOURCE,
        "All annotated descending neurons: used to rank the GF's response to looming drive against every other "
        "possible motor readout (specificity analysis), never as a learned readout.",
    ),
    NeuronSet(
        "KC",
        "cell_class == 'Kenyon_Cell'",
        5177,
        "inventory",
        _MB,
        "Kenyon cells of the mushroom body (Phase 4: presynaptic side of the only plastic synapses).",
    ),
    NeuronSet(
        "MBON",
        "cell_type.str.startswith('MBON', na=False)",
        96,
        "inventory",
        _MB,
        "Mushroom body output neurons (Phase 4: postsynaptic side of the plastic KC→MBON synapses).",
    ),
    NeuronSet(
        "PAM",
        "cell_type.str.startswith('PAM', na=False)",
        307,
        "inventory",
        _MB,
        "PAM-cluster dopaminergic neurons (reward teaching signal, Phase 4).",
    ),
    NeuronSet(
        "PPL1",
        "cell_type.str.startswith('PPL1', na=False)",
        16,
        "inventory",
        _MB,
        "PPL1-cluster dopaminergic neurons (punishment teaching signal, Phase 4).",
    ),
)

SET_BY_NAME = {s.name: s for s in SETS}
LOCK_PATH = Path(__file__).with_name("data") / "neuron_sets.lock.json"


def resolve_in_table(neuron_set: NeuronSet, annotations) -> np.ndarray:
    """Root IDs (int64, sorted) matching the set's query in an annotation DataFrame."""
    sub = annotations.query(neuron_set.query, engine="python")
    return np.sort(sub["root_id"].astype("int64").to_numpy())


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def ids(name: str) -> np.ndarray:
    """FlyWire-783 root IDs of a set, from the committed lock file (no annotation table needed at run time)."""
    return np.array([int(i) for i in load_lock()["sets"][name]["ids"]], dtype=np.int64)


def indices(conn, name: str) -> np.ndarray:
    """Neuron indices of a set in `conn`. Synthetic connectomes resolve the same query on their own annotations."""
    if conn.annotations is not None:
        return conn.index_of(resolve_in_table(SET_BY_NAME[name], conn.annotations))
    all_ids = ids(name)
    return conn.index_of(all_ids[np.isin(all_ids, conn.root_ids)])
