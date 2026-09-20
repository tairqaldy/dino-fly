"""Realtime protocol v1 — pydantic mirror of `packages/protocol/schemas/messages.schema.json` (the source of truth).

`tests/test_protocol.py` validates the shared examples and everything this module produces against that schema.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = 1


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FlyHello(_Payload):
    workerId: str
    connectome: str
    neurons: int
    engineVersion: int
    transducerVersion: int
    generation: int
    bioMsPerFrame: float
    gpu: str | None = None
    groups: list[str]


class FlyFrame(_Payload):
    seed: int
    state: list[int]
    gf: int
    jump: bool
    theta: float
    thetaDot: float
    rates: tuple[float, float]
    activity: list[int]
    dopamine: list[float] | None = None


class FlyRunEnd(_Payload):
    seed: int
    generation: int
    score: int
    frames: int
    cleared: int
    deathType: int
    jumps: int
    actions: list[tuple[int, int]]


class LearningPoint(_Payload):
    generation: int
    heldoutMean: float


class FlyStats(_Payload):
    generation: int
    gamesPlayed: int
    bestScore: int
    meanScoreRecent: float
    totalJumps: int
    totalDeaths: int | None = None
    realtimeFactor: float | None = None
    learningCurve: list[LearningPoint] = Field(default_factory=list)


class FlyStatus(_Payload):
    online: bool
    generation: int | None = None


class DopamineEvent(_Payload):
    kind: Literal["reward", "punish"]
    magnitude: float = Field(ge=0)
    source: Literal["game", "human_button", "hardware"]
    seed: int | None = None
    frame: int | None = None


class HumanInput(_Payload):
    source: Literal["keyboard", "touch", "pose"]
    action: Literal["jump", "duck", "release"]


class LabCommand(_Payload):
    command: Literal["start", "stop", "set_seed", "set_generation", "reward", "punish"]
    value: float | None = None


PAYLOADS: dict[str, type[_Payload]] = {
    "fly.hello": FlyHello,
    "fly.frame": FlyFrame,
    "fly.run_end": FlyRunEnd,
    "fly.stats": FlyStats,
    "fly.status": FlyStatus,
    "dopamine.event": DopamineEvent,
    "human.input": HumanInput,
    "lab.command": LabCommand,
}


def envelope(msg_type: str, payload: _Payload, ts: float | None = None) -> dict:
    """JSON-ready envelope {type, v, ts, payload}; None-valued optional fields are omitted."""
    if not isinstance(payload, PAYLOADS[msg_type]):
        raise TypeError(f"{msg_type} needs a {PAYLOADS[msg_type].__name__}")
    return {
        "type": msg_type,
        "v": PROTOCOL_VERSION,
        "ts": ts if ts is not None else time.time() * 1e3,
        "payload": payload.model_dump(mode="json", exclude_none=True),
    }


def parse(message: dict) -> tuple[str, _Payload]:
    """Validate an incoming envelope; raises ValueError / pydantic.ValidationError when malformed."""
    if message.get("v") != PROTOCOL_VERSION or not isinstance(message.get("ts"), int | float):
        raise ValueError("bad envelope")
    msg_type = message.get("type")
    if msg_type not in PAYLOADS:
        raise ValueError(f"unknown message type {msg_type!r}")
    return msg_type, PAYLOADS[msg_type].model_validate(message.get("payload"))
