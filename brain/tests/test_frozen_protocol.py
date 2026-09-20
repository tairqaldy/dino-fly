"""The frozen transducer must be exactly what the committed calibration experiment produced."""

from __future__ import annotations

import json

from flybrain.config import results_dir
from flybrain.play import BIO_MS_PER_FRAME, MAX_FRAMES
from flybrain.transducer.looming import FROZEN, TRANSDUCER_VERSION
from flybrain.transducer.motor import MOTOR_VERSION, MotorParams


def test_frozen_gain_is_the_committed_calibration_result():
    result = json.loads((results_dir() / "looming_gf.json").read_text(encoding="utf-8"))
    cal = result["calibration"]
    assert cal["status"] == "calibrated"
    assert cal["gain_hz"] == FROZEN.gain_hz
    forms = result["protocol"]["transducer_forms"]
    assert (FROZEN.size_peak_deg, FROZEN.size_sigma_deg, FROZEN.velocity_sat_deg_s) == (
        forms["size_peak_deg"],
        forms["size_sigma_deg"],
        forms["velocity_sat_deg_s"],
    )
    assert result["protocol"]["target_size_deg"] == 42.0


def test_protocol_constants_are_the_preregistered_ones():
    assert (TRANSDUCER_VERSION, MOTOR_VERSION) == (1, 1)
    assert FROZEN.version == TRANSDUCER_VERSION
    assert (BIO_MS_PER_FRAME, MAX_FRAMES) == (10.0, 10_000)
    assert MotorParams() == MotorParams(version=1, hold_frames=10, delay_frames=0)
