"""Seed sets. Disjoint by construction; `tests/test_seeds.py` enforces it.

- HELDOUT_100: the forever-evaluation set. The same 100 seeds for every generation, forever.
- HELDOUT_200: HELDOUT_100 plus 100 more; used for the Phase 1 naive-play report.
- DEV_SEEDS:   debugging and sensitivity analyses. Never reported as headline numbers.
- TRAIN:       everything the fly may train on (Phase 4+). Never evaluate on training seeds.

Held-out seeds may only be touched by final evaluation scripts, which refuse a dirty git tree and append to
`brain/experiments/results/heldout_ledger.jsonl`.
"""

from __future__ import annotations

HELDOUT_BASE = 9_000_000
HELDOUT_100: tuple[int, ...] = tuple(range(HELDOUT_BASE, HELDOUT_BASE + 100))
HELDOUT_200: tuple[int, ...] = tuple(range(HELDOUT_BASE, HELDOUT_BASE + 200))

DEV_SEEDS: tuple[int, ...] = tuple(range(1, 101))

TRAIN_START = 1_000_000
TRAIN_STOP = 9_000_000  # exclusive


def is_train_seed(seed: int) -> bool:
    return TRAIN_START <= seed < TRAIN_STOP


def train_seeds(start_offset: int, count: int) -> tuple[int, ...]:
    lo = TRAIN_START + start_offset
    hi = lo + count
    if hi > TRAIN_STOP:
        raise ValueError("training seed range exhausted")
    return tuple(range(lo, hi))
