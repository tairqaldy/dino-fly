"""Reward / punishment transducer: game events → Poisson bursts in dopaminergic neurons. Fixed and hand-written.

    obstacle cleared  → reward      → PAM cluster burst        (magnitude scaled by how well-timed the jump was)
    collision         → punishment  → PPL1 cluster burst
    manual buttons    → the same two pathways (source = human_button / hardware)

A burst is `burst_frames` game frames of Poisson drive at `rate_hz · magnitude`. What the dopamine then does is up to
the connectome (which MBON compartments those neurons innervate) and the plasticity rule.
"""

from __future__ import annotations

from dataclasses import dataclass

DOPAMINE_VERSION = 1


@dataclass(frozen=True)
class DopamineParams:
    version: int = DOPAMINE_VERSION
    rate_hz: float = 100.0
    burst_frames: int = 10
    # a jump is "well timed" when it starts this many frames before the collision would happen
    ideal_frames_to_collision: float = 8.0
    timing_tolerance_frames: float = 6.0


def timing_magnitude(frames_to_collision_at_jump: float | None, p: DopamineParams) -> float:
    """1.0 for a perfectly timed jump, falling linearly to 0.25 at the tolerance and beyond."""
    if frames_to_collision_at_jump is None:
        return 0.25
    err = abs(frames_to_collision_at_jump - p.ideal_frames_to_collision) / p.timing_tolerance_frames
    return max(0.25, 1.0 - 0.75 * min(err, 1.0))


class DopamineChannel:
    """Per-game burst state for one cluster (PAM or PPL1)."""

    def __init__(self, params: DopamineParams | None = None) -> None:
        self.params = params or DopamineParams()
        self.frames_left = 0
        self.magnitude = 0.0
        self.events = 0

    def trigger(self, magnitude: float = 1.0) -> None:
        self.frames_left = self.params.burst_frames
        self.magnitude = max(self.magnitude if self.frames_left else 0.0, magnitude)
        self.events += 1

    def rate(self) -> float:
        """Poisson rate (Hz) of the cluster for the coming frame; call once per frame."""
        if self.frames_left <= 0:
            self.magnitude = 0.0
            return 0.0
        self.frames_left -= 1
        return self.params.rate_hz * self.magnitude
