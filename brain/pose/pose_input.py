"""Play with your body: MJPEG stream (XIAO ESP32S3 Sense or any webcam) → MediaPipe Pose → JUMP / DUCK.

    uv run --no-sync --with mediapipe --with opencv-python python brain/pose/pose_input.py --source http://<xiao-ip>:81/stream
    uv run --no-sync --with mediapipe --with opencv-python python brain/pose/pose_input.py --source 0      # laptop webcam

Stand still for the first 2 s (baseline). JUMP = hips rise quickly above the baseline; DUCK = nose drops well below
its baseline. Actions are emitted as `human.input` protocol messages on a local WebSocket (ws://localhost:8766) that
the web app's Play page can connect to; the same messages a keyboard produces. Thresholds are in body-height units,
so they do not depend on distance to the camera.

NOT YET TESTED WITH THE XIAO CAMERA (no hardware attached when this was written); the detector logic below is
unit-tested in brain/tests/test_pose.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass


@dataclass
class PoseSample:
    t: float
    hip_y: float  # image coordinates, 0 = top, 1 = bottom
    nose_y: float
    shoulder_y: float


class ActionDetector:
    """Turns a stream of pose samples into 'jump' / 'duck' / 'release' events."""

    def __init__(self, calibration_s: float = 2.0, jump_rise: float = 0.12, duck_drop: float = 0.30) -> None:
        self.calibration_s, self.jump_rise, self.duck_drop = calibration_s, jump_rise, duck_drop
        self._calib: list[PoseSample] = []
        self.baseline: PoseSample | None = None
        self.state = "release"

    def update(self, s: PoseSample) -> str | None:
        if self.baseline is None:
            self._calib.append(s)
            if s.t - self._calib[0].t >= self.calibration_s:
                n = len(self._calib)
                self.baseline = PoseSample(s.t, sum(x.hip_y for x in self._calib) / n, sum(x.nose_y for x in self._calib) / n,
                                           sum(x.shoulder_y for x in self._calib) / n)
            return None
        body = max(self.baseline.hip_y - self.baseline.nose_y, 1e-3)  # nose-to-hip distance = one "body unit"
        rise = (self.baseline.hip_y - s.hip_y) / body
        drop = (s.nose_y - self.baseline.nose_y) / body
        new = "jump" if rise > self.jump_rise else "duck" if drop > self.duck_drop else "release"
        if new != self.state:
            self.state = new
            return new
        return None


def envelope(action: str) -> str:
    return json.dumps({"type": "human.input", "v": 1, "ts": time.time() * 1e3, "payload": {"source": "pose", "action": action}})


async def main_async(source: str, port: int) -> None:  # pragma: no cover - needs a camera + mediapipe
    import cv2
    import mediapipe as mp
    import websockets

    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    pose = mp.solutions.pose.Pose(model_complexity=0)
    lm = mp.solutions.pose.PoseLandmark
    det = ActionDetector()
    async with websockets.serve(handler, "127.0.0.1", port):
        print(f"[pose] human.input events on ws://127.0.0.1:{port} — stand still for {det.calibration_s:.0f} s to calibrate")
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.05)
                continue
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                p = res.pose_landmarks.landmark
                sample = PoseSample(time.time(), (p[lm.LEFT_HIP].y + p[lm.RIGHT_HIP].y) / 2, p[lm.NOSE].y,
                                    (p[lm.LEFT_SHOULDER].y + p[lm.RIGHT_SHOULDER].y) / 2)
                action = det.update(sample)
                if action:
                    print("[pose]", action)
                    websockets.broadcast(clients, envelope(action))
            await asyncio.sleep(0)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    asyncio.run(main_async(args.source, args.port))
