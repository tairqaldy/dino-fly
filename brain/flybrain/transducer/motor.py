"""Motor transducer: Giant Fiber spikes → JUMP. Fixed, hand-written, nothing decoded or learned.

In the animal a single Giant Fiber spike drives the tergotrochanteral "jump" muscle and triggers the short-mode
escape takeoff. Here a GF spike (either side) during frame f presses JUMP from frame f+1 on. The key is held for
`hold_frames` so that every GF-triggered jump is the same full jump (in the game, the maximum-height clamp engages at
frame 8; releasing after that changes nothing). GF spikes while the dino is airborne or the key is still held are
ignored but counted — the real fly cannot take off twice either.
"""

from __future__ import annotations

from dataclasses import dataclass

MOTOR_VERSION = 1


@dataclass(frozen=True)
class MotorParams:
    version: int = MOTOR_VERSION
    hold_frames: int = 10
    delay_frames: int = 0  # constant extra latency between GF spike and key press (0 = next frame)


class JumpMotor:
    """Per-game motor state."""

    def __init__(self, params: MotorParams | None = None) -> None:
        self.params = params or MotorParams()
        self.hold_left = 0
        self.pending: list[int] = []  # countdowns of delayed jump commands
        self.triggered = 0
        self.ignored = 0

    def update(self, gf_spikes: int, airborne: bool) -> bool:
        """Call once per frame with the number of GF spikes of that frame; returns the JUMP key state for the next step."""
        if gf_spikes > 0:
            if airborne or self.hold_left > 0 or self.pending:
                self.ignored += gf_spikes
            else:
                self.pending.append(self.params.delay_frames)
                self.ignored += gf_spikes - 1
        if self.pending:
            if self.pending[0] <= 0:
                self.pending.pop(0)
                if not airborne and self.hold_left == 0:
                    self.hold_left = self.params.hold_frames
                    self.triggered += 1
            else:
                self.pending[0] -= 1
        pressed = self.hold_left > 0
        if pressed:
            self.hold_left -= 1
        return pressed
