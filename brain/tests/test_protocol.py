"""Python protocol models vs. the shared JSON Schema and examples (source of truth: packages/protocol)."""

from __future__ import annotations

import json

import jsonschema
import pytest

from flybrain import protocol as proto
from flybrain.config import repo_root

PKG = repo_root() / "packages" / "protocol"
SCHEMA = json.loads((PKG / "schemas" / "messages.schema.json").read_text(encoding="utf-8"))
EXAMPLES = json.loads((PKG / "examples" / "messages.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def test_python_models_cover_every_worker_side_message_type():
    schema_types = {alt["properties"]["type"]["const"] for alt in SCHEMA["oneOf"]}
    assert set(proto.PAYLOADS) == schema_types - {"leaderboard.update"}  # produced and consumed by the API + web only


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["type"] for e in EXAMPLES])
def test_shared_examples_validate_and_round_trip(example):
    VALIDATOR.validate(example)
    if example["type"] not in proto.PAYLOADS:
        return
    kind, payload = proto.parse(example)
    again = proto.envelope(kind, payload, ts=example["ts"])
    VALIDATOR.validate(again)
    assert again == example


def test_malformed_messages_are_rejected():
    for bad in (
        {"type": "fly.status", "v": 2, "ts": 1, "payload": {"online": True}},
        {"type": "nope", "v": 1, "ts": 1, "payload": {}},
        {"type": "fly.status", "v": 1, "ts": "now", "payload": {"online": True}},
    ):
        with pytest.raises(ValueError):
            proto.parse(bad)
    with pytest.raises(ValueError):
        proto.parse({"type": "dopamine.event", "v": 1, "ts": 1, "payload": {"kind": "bribe", "magnitude": 1, "source": "game"}})
    with pytest.raises(TypeError):
        proto.envelope("fly.status", proto.HumanInput(source="pose", action="jump"))


def test_worker_commands_and_dopamine_queue():
    from flybrain.worker import GROUPS, LiveFly

    sent = []
    fly = LiveFly(sent.append, device="cpu")
    fly.command(proto.LabCommand(command="stop"))
    assert not fly.running.is_set()
    fly.command(proto.LabCommand(command="start"))
    fly.command(proto.LabCommand(command="set_seed", value=77))
    fly.command(proto.LabCommand(command="punish", value=2))
    assert fly.running.is_set() and fly.seed == 77
    assert fly.pending_dopamine[0] == proto.DopamineEvent(kind="punish", magnitude=2, source="human_button")
    assert "GF" in GROUPS and len(set(GROUPS)) == len(GROUPS)
