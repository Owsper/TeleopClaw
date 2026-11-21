"""Robust geometry-based 1-hand gestures, normalized by palm size."""
import numpy as np
from common.types import HandState


class Gestures:
    def __init__(self):
        self.reset()

    def reset(self):
        self._fist = False
        self._fist_enter_frames = 0
        self._fist_exit_frames = 0
        self._open_since = None
        self._open_anchor = None
        self._filtered_pinch = None

    def update(self, px: np.ndarray, t: float) -> HandState:
        px = np.asarray(px, dtype=np.float32)
        if px.shape != (21, 2):
            raise ValueError("Expected 21 hand landmarks with (x, y) positions")

        # Pixel distance wrist(0) -> middle-knuckle(9); approx camera-depth proxy.
        palm = float(np.linalg.norm(px[9] - px[0]))
        if palm < 10:  # too tiny/far away to track safely
            self.reset()
            return HandState.invalid(t)

        # Pinch is scaled so hand closer to the lens doesn't change the ratio.
        thumb_index = float(np.linalg.norm(px[4] - px[8]) / palm)
        raw_pinch = float(np.clip((thumb_index - 0.25) / 0.95, 0.0, 1.0))
        if self._filtered_pinch is None:
            self._filtered_pinch = raw_pinch
        else:
            self._filtered_pinch = 0.65 * self._filtered_pinch + 0.35 * raw_pinch

        # Compare tip-to-wrist vs PIP-to-wrist for index, middle, ring, pinky.
        # A curled finger has a tip no farther from wrist than its middle joint.
        tips = (8, 12, 16, 20)
        pips = (6, 10, 14, 18)
        margin = np.array([
            (np.linalg.norm(px[tip] - px[0]) -
             np.linalg.norm(px[pip] - px[0])) / palm
            for tip, pip in zip(tips, pips)
        ])

        # Hysteresis + frame debounce: pinch should not look like fist.
        if not self._fist:
            self._fist_enter_frames = (
                self._fist_enter_frames + 1 if np.all(margin < 0.25) else 0
            )
            if self._fist_enter_frames >= 5:
                self._fist = True
                self._fist_exit_frames = 0
        else:
            self._fist_exit_frames = (
                self._fist_exit_frames + 1 if np.any(margin > 0.40) else 0
            )
            if self._fist_exit_frames >= 3:
                self._fist = False
                self._fist_enter_frames = 0

        # Stop gesture: all fingers extended + wrist roughly stationary for 1s.
        is_open = bool(np.all(margin > 0.30) and not self._fist)
        wrist = px[0]
        if not is_open:
            self._open_since = None
            self._open_anchor = None
        elif (self._open_anchor is None or
              np.linalg.norm(wrist - self._open_anchor) > 0.15 * palm):
            self._open_since = t
            self._open_anchor = wrist.copy()
        stop = bool(self._open_since is not None and t - self._open_since >= 1.0)

        return HandState(True, t, px, palm, float(self._filtered_pinch),
                         self._fist, stop)
