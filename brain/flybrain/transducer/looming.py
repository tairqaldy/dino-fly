"""Looming transducer: (angular size θ, angular velocity θ̇) of an approaching object → Poisson rates of the
LPLC2 and LC4 visual projection neuron populations.

This is the *parameterised looming stimulus* neuroscientists present to real flies, expressed as the drive of the
two cell types that carry it to the Giant Fiber. It is fixed and hand-written; nothing is learned or tuned on
game score.

Functional forms (Ache et al., Curr Biol 2019: "a model summing a linear function of angular velocity (provided by
LC4) and a Gaussian function of angular size (provided by LPLC2) replicates GF looming response dynamics"):

    r_LC4(θ̇)      = G · min(θ̇ / θ̇_sat, 1)                     for θ̇ > 0, else 0
    r_LPLC2(θ, θ̇) = G · exp(-(θ - μ)² / (2σ²))                 for θ̇ > 0 (expansion only), else 0

Constants and their status (see docs/DECISIONS.md D4, D6 and the forking-paths log):
    μ = 42°        peak of the size tuning — as restated from Ache et al. 2019 by secondary sources (primary paywalled)
    σ = 15°        OUR CHOICE (not found in the literature); sensitivity reported in experiments/looming_gf.py
    θ̇_sat = 3000°/s OUR CHOICE: LC4 reaches its peak rate at the fastest standard looming stimulus
                   (r/v = 10 ms reaches ≈ 3100°/s at 63°); linear below that
    G              the one genuinely free parameter: no LC4/LPLC2 firing rates in Hz are published. Both populations
                   share the same peak rate G, so their relative weight at the GF is set by the connectome.
                   G is calibrated by a biological criterion in experiments/looming_gf.py, then frozen below.
Both hemispheres are driven identically (no retinotopy in the MVP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoomingParams:
    version: int
    gain_hz: float
    size_peak_deg: float = 42.0
    size_sigma_deg: float = 15.0
    velocity_sat_deg_s: float = 3000.0


# Frozen after the calibration in experiments/looming_gf.py (results/looming_gf.json → "calibration").
# tests/test_looming.py asserts that this constant matches the committed calibration result.
TRANSDUCER_VERSION = 1
FROZEN = LoomingParams(version=TRANSDUCER_VERSION, gain_hz=float("nan"))


def looming_theta_deg(t_to_collision_ms: np.ndarray, rv_ms: float) -> np.ndarray:
    """Angular size of an object with size-to-speed ratio r/v, `t` ms before collision: θ = 2·atan((r/v)/t)."""
    t = np.maximum(np.asarray(t_to_collision_ms, dtype=np.float64), 1e-9)
    return np.degrees(2.0 * np.arctan(rv_ms / t))


def looming_theta_dot_deg_s(t_to_collision_ms: np.ndarray, rv_ms: float) -> np.ndarray:
    """θ̇ = 2(r/v) / (t² + (r/v)²), in deg/s."""
    t = np.asarray(t_to_collision_ms, dtype=np.float64)
    return np.degrees(2.0 * rv_ms / (t**2 + rv_ms**2)) * 1e3


def time_to_collision_ms(theta_deg: float, rv_ms: float) -> float:
    return rv_ms / np.tan(np.radians(theta_deg) / 2.0)


# ------------------------------------------------------------------------------------------ game geometry
# The dino's eye, as an offset (px) from the sprite's left edge and from its feet. Fixed points inside the head
# collision box of packages/dino-core/constants.json (running) and near the front of the ducking sprite.
EYE_OFFSET_PX = (38.0, 40.0)
EYE_OFFSET_DUCK_PX = (52.0, 18.0)


@dataclass(frozen=True)
class View:
    """What the dino sees of the nearest obstacle ahead. `obstacle_index` is -1 when nothing is in view."""

    obstacle_index: int
    distance_px: float
    theta_deg: float
    theta_dot_deg_s: float

    @property
    def tau_ms(self) -> float:
        """Time to collision estimate τ = θ / θ̇ (ms of biological time); inf when not expanding."""
        return 1e3 * self.theta_deg / self.theta_dot_deg_s if self.theta_dot_deg_s > 0 else float("inf")


NOTHING_IN_VIEW = View(-1, float("inf"), 0.0, 0.0)


def obstacle_view(state, bio_ms_per_frame: float) -> View:
    """Angular size θ and analytic expansion rate θ̇ of the nearest on-screen obstacle ahead of the eye.

    Pure 2-D side-view geometry, no free scale: θ is the angle subtended at the eye by the obstacle's vertical
    extent [y_bottom, y_top] at horizontal distance d; θ̇ = (∂θ/∂d)·ḋ with ḋ = -(ground speed + the obstacle's own
    speed), converted to biological time. θ̇ is analytic per tracked obstacle — a frame-to-frame finite difference
    would produce a spurious burst of "expansion" whenever the nearest obstacle changes.
    """
    from flybrain.dino_core import FPX, C, D

    ex, ey = EYE_OFFSET_DUCK_PX if state.ducking else EYE_OFFSET_PX
    eye_x = D["x"] + ex
    eye_y = state.dino_y / FPX + ey
    width = C["world"]["width"]
    for i, o in enumerate(state.obstacles):
        x_left = o.x / FPX
        if x_left <= eye_x or x_left >= width:
            continue
        d = max(x_left - eye_x, 1.0)
        a = o.y_bottom_px + o.height_px - eye_y
        b = o.y_bottom_px - eye_y
        theta = np.arctan2(a, d) - np.arctan2(b, d)
        closing_px_per_frame = (state.speed + o.speed_offset) / FPX
        dtheta_dd = -a / (d * d + a * a) + b / (d * d + b * b)
        theta_dot = -dtheta_dd * closing_px_per_frame * (1000.0 / bio_ms_per_frame)  # rad per biological second
        return View(i, d, float(np.degrees(theta)), float(np.degrees(theta_dot)))
    return NOTHING_IN_VIEW


def population_rates(
    theta_deg: np.ndarray, theta_dot_deg_s: np.ndarray, params: LoomingParams
) -> tuple[np.ndarray, np.ndarray]:
    """(LC4 rate, LPLC2 rate) in Hz for arrays of θ (deg) and θ̇ (deg/s)."""
    theta = np.asarray(theta_deg, dtype=np.float64)
    vel = np.asarray(theta_dot_deg_s, dtype=np.float64)
    expanding = vel > 0
    lc4 = params.gain_hz * np.clip(vel / params.velocity_sat_deg_s, 0.0, 1.0)
    lplc2 = params.gain_hz * np.exp(-((theta - params.size_peak_deg) ** 2) / (2.0 * params.size_sigma_deg**2))
    return np.where(expanding, lc4, 0.0), np.where(expanding, lplc2, 0.0)
