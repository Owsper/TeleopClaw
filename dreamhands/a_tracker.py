"""Adapter: A's tracker HandState (common/types.py) -> the control-side HandState.

A publishes pixels in the mirrored 640x480 frame; control wants normalized coords:
- pos x = palm centre / width, y = 1 - palm centre / height (up = up),
  z = apparent palm span, corrected for tilt (larger = closer to the camera). Palm centre = mean of the
  wrist and the four knuckles, steadier than any single landmark.
- A's pinch is 0 = touching .. 1 = apart; ours is the opposite, so pinch = 1 - A.pinch.
- A's `stop` (still open palm for 1 s) -> open_palm, the runner disengages on it.
"""
from __future__ import annotations

import numpy as np

from .types import HandState

PALM_IDX = [0, 5, 9, 13, 17]
PALM_REF = 200.0  # px; wrist->middle-knuckle is ~60-140 px at arm's length


def from_a(h, width: int = 640, height: int = 480, *, wrist_origin: bool = False,
           side: str = "right") -> HandState:
    if not h.valid:
        return HandState(t=h.t, present=False, handedness=side)
    px = np.asarray(h.px, float)
    c = px[0] if wrist_origin else px[PALM_IDX].mean(0)
    reach = np.linalg.norm(px - px[0], axis=1)
    extension = (reach[[8, 12, 16, 20]] - reach[[6, 10, 14, 18]]) / max(h.palm, 1)
    v_sign = (np.all(extension[:2] > 0.30) and np.all(extension[2:] < 0.20)
              and np.linalg.norm(px[8] - px[12]) > 0.25 * h.palm and h.pinch > 0.6)
    span = h.palm if h.palm_span is None else h.palm_span
    depth = span / width * (640.0 / PALM_REF)
    pos = np.array([c[0] / width, 1.0 - c[1] / height, depth])
    return HandState(t=h.t, present=True, confidence=1.0, pos=pos,
                     pinch=float(1.0 - h.pinch), fist=bool(h.fist),
                     open_palm=bool(h.stop), handedness=side, v_sign=bool(v_sign))


class ATracker:
    """Wraps A's HandTracker (webcam + MediaPipe) behind the same read() as MouseTracker."""

    def __init__(self, camera_index: int = 0, pose_model: str | None = None, both_hands: bool = False):
        from hand_tracker import HandTracker  # A's module at the repo root
        self.tracker = HandTracker(camera_index=camera_index, pose_model=pose_model, both_hands=both_hands).start()

    def read(self) -> HandState:
        return from_a(self.tracker.latest())

    def close(self) -> None:
        self.tracker.close()
