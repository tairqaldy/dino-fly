"""The Python port of the game engine must reproduce the TypeScript engine's golden fixtures byte for byte."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flybrain import dino_core as dc
from flybrain.config import repo_root

FIXTURES = repo_root() / "packages" / "dino-core" / "fixtures"
TS_CONSTANTS = repo_root() / "packages" / "dino-core" / "constants.json"


def _lf(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def test_constants_copy_is_identical_to_the_typescript_source():
    assert _lf(dc.CONSTANTS_PATH) == _lf(TS_CONSTANTS), "run: cp packages/dino-core/constants.json brain/flybrain/data/dino_constants.json"


def test_prng_matches_cross_language_vectors():
    vectors = json.loads((FIXTURES / "rng_vectors.json").read_text(encoding="utf-8"))
    assert len(vectors["seeds"]) >= 9
    for v in vectors["seeds"]:
        state = dc.seed_to_state(v["seed"])
        for expected in v["u32"]:
            state, value = dc.rng_next_u32(state)
            assert value == expected
        for expected in v["randInt"]["values"]:
            state, value = dc.rng_rand_int(state, v["randInt"]["lo"], v["randInt"]["hi"])
            assert value == expected


@pytest.fixture(scope="module")
def game_runs() -> dict:
    return json.loads((FIXTURES / "game_runs.json").read_text(encoding="utf-8"))


def test_fixture_file_matches_this_engine_version(game_runs):
    import hashlib

    assert game_runs["engineVersion"] == dc.ENGINE_VERSION
    assert game_runs["constantsSha256"] == hashlib.sha256(_lf(TS_CONSTANTS).encode("utf-8")).hexdigest()
    assert len(game_runs["runs"]) >= 10


def test_every_frame_of_every_golden_run_matches(game_runs):
    chars = game_runs["hashChars"]
    every = game_runs["checkpointEvery"]
    total_frames = 0
    for run in game_runs["runs"]:
        expected = run["hashes"]
        checkpoints = {cp[0]: cp for cp in run["checkpoints"]}
        frames: list[dc.GameState] = []
        dc.replay(run["seed"], [tuple(a) for a in run["actions"]], run["maxFrames"], on_frame=frames.append)
        assert len(frames) * chars == len(expected), f"{run['name']}: run length differs"
        for i, state in enumerate(frames):
            got = dc.state_hash(state, chars)
            if got != expected[i * chars : (i + 1) * chars]:
                near = max((f for f in checkpoints if f <= state.frame), default=None)
                pytest.fail(
                    f"{run['name']}: first divergence at frame {state.frame}\n  python: {dc.canonical_string(state)}\n"
                    f"  nearest TS checkpoint (frame {near}): {checkpoints.get(near)}"
                )
            if state.frame % every == 0 or state.crashed:
                assert dc.state_to_ints(state) == checkpoints[state.frame]
        final, summary = frames[-1], run["summary"]
        assert (final.frame, dc.score(final), final.crashed, final.death_type) == (
            summary["frames"],
            summary["score"],
            summary["crashed"],
            summary["deathType"],
        )
        assert (final.jumps, final.ducks, final.cleared) == (summary["jumps"], summary["ducks"], summary["cleared"])
        total_frames += len(frames)
    assert total_frames > 30_000  # incl. oracle runs that reach max speed, pterodactyls and cactus groups


def test_port_is_pure_and_rejects_bad_logs():
    s0 = dc.create_initial_state(3)
    s1 = dc.step(s0, True, False)
    assert s0.frame == 0 and s1.frame == 1 and s1.jumping
    crashed = dc.replay(1, [], 2000)
    assert crashed.crashed and dc.step(crashed, True, False) is crashed
    with pytest.raises(ValueError):
        dc.replay(1, [(5, 1), (5, 0)], 10)
    with pytest.raises(ValueError):
        dc.bits_to_input(4)
