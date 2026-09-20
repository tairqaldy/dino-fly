"""Body-movement input: the jump/duck detector (pure logic; camera + MediaPipe are not needed)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jsonschema

from flybrain.config import repo_root

spec = importlib.util.spec_from_file_location("pose_input", Path(__file__).resolve().parents[1] / "pose" / "pose_input.py")
pose_input = importlib.util.module_from_spec(spec)
sys.modules["pose_input"] = pose_input  # dataclasses need the module to be importable by name
spec.loader.exec_module(pose_input)


def feed(det, samples):
    return [e for e in (det.update(pose_input.PoseSample(t, hip, nose, nose + 0.1)) for t, hip, nose in samples) if e]


def test_calibrates_then_detects_jump_and_duck_in_body_units():
    det = pose_input.ActionDetector(calibration_s=1.0)
    still = [(0.1 * i, 0.60, 0.20) for i in range(12)]  # nose-to-hip = 0.40 of the image
    assert feed(det, still) == [] and det.baseline is not None
    events = feed(det, [(2.0, 0.59, 0.20), (2.1, 0.52, 0.12), (2.2, 0.60, 0.20), (2.3, 0.72, 0.36), (2.4, 0.60, 0.20)])
    assert events == ["jump", "release", "duck", "release"]
    assert feed(det, [(2.5, 0.60, 0.21)]) == []  # small wobble is ignored


def test_emits_schema_valid_protocol_messages():
    schema = json.loads((repo_root() / "packages/protocol/schemas/messages.schema.json").read_text(encoding="utf-8"))
    for action in ("jump", "duck", "release"):
        jsonschema.Draft202012Validator(schema).validate(json.loads(pose_input.envelope(action)))
