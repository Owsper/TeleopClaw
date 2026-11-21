"""Adapter: A's tracker HandState (common/types.py) -> the control-side HandState.

A publishes pixels in the mirrored 640x480 frame; control wants normalized coords:
- pos x = palm centre / width, y = 1 - palm centre / height (up = up),
  z = palm size / PALM_REF (bigger palm = closer to the camera). Palm centre = mean of the
  wrist and the four knuckles, steadier than any single landmark.
- A's pinch is 0 = touching .. 1 = apart; ours is the opposite, so pinch = 1 - A.pinch.
- A's `stop` (still open palm for 1 s) -> open_palm, the runner disengages on it.
"""
from __future__ import annotations

import numpy as np

from .types import HandState

PALM_IDX = [0, 5, 9, 13, 17]
PALM_REF = 200.0  # px; wrist->middle-knuckle is ~60-140 px at arm's length


def from_a(h, width: int = 640, height: int = 480) -> HandState:
    if not h.valid:
        return HandState(t=h.t, present=False)
    c = np.asarray(h.px, float)[PALM_IDX].mean(0)
    pos = np.array([c[0] / width, 1.0 - c[1] / height, h.palm / PALM_REF])
    return HandState(t=h.t, present=True, confidence=1.0, pos=pos,
                     pinch=float(1.0 - h.pinch), fist=bool(h.fist),
                     open_palm=bool(h.stop), handedness="right")


class ATracker:
    """Wraps A's HandTracker (webcam + MediaPipe) behind the same read() as MouseTracker."""

    def __init__(self, camera_index: int = 0):
        from hand_tracker import HandTracker  # A's module at the repo root
        self.tracker = HandTracker(camera_index=camera_index).start()

    def read(self) -> HandState:
        return from_a(self.tracker.latest())

    def close(self) -> None:
        self.tracker.close()
