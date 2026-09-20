from __future__ import annotations

import re

import pytest

from flybrain import seeds
from flybrain.data_manifest import load_manifest


def test_seed_sets_are_disjoint_and_nested():
    assert len(seeds.HELDOUT_100) == 100 and len(seeds.HELDOUT_200) == 200
    assert set(seeds.HELDOUT_100) < set(seeds.HELDOUT_200)
    assert not set(seeds.HELDOUT_200) & set(seeds.DEV_SEEDS)
    assert not any(seeds.is_train_seed(s) for s in seeds.HELDOUT_200 + seeds.DEV_SEEDS)
    train = seeds.train_seeds(0, 1000)
    assert all(seeds.is_train_seed(s) for s in train)
    assert not set(train) & (set(seeds.HELDOUT_200) | set(seeds.DEV_SEEDS))
    with pytest.raises(ValueError):
        seeds.train_seeds(seeds.TRAIN_STOP, 1)


def test_manifest_urls_are_pinned_and_checksummed():
    manifest = load_manifest()
    assert {"flywire783_connectivity", "flywire630_connectivity", "flywire_annotations"} <= set(manifest)
    for key, entry in manifest.items():
        assert entry.size > 0
        assert entry.sha256 and re.fullmatch(r"[0-9a-f]{64}", entry.sha256), f"{key}: sha256 not recorded"
        for url in entry.urls:
            assert url.startswith("https://")
            pinned = re.search(r"/[0-9a-f]{40}/", url) or "edmond.mpg.de/api/access/datafile/" in url
            assert pinned, f"{key}: URL is not pinned to a commit or archive file id: {url}"
