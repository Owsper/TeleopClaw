"""Mouse stand-in for A's hand tracker: publishes the same HandState.

In the OpenCV window: move = x/y, wheel = depth z, hold left button = pinch (close gripper),
hold right button = fist (clutch). Leaving the window = hand lost.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .types import HandState


class MouseTracker:
    def __init__(self, window: str, width: int = 640, height: int = 360):
        self.w, self.h = width, height
        self.x, self.y, self.z = 0.5, 0.5, 0.5
        self.pinch = self.fist = False
        self.inside = False
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._cb)

    def _cb(self, event, x, y, flags, _):
        self.inside = 0 <= x < self.w and 0 <= y < self.h
        self.x, self.y = x / self.w, 1.0 - y / self.h
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pinch = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.pinch = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.fist = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.fist = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.z = float(np.clip(self.z + (0.03 if flags > 0 else -0.03), 0.0, 1.0))

    def read(self) -> HandState:
        # the mouse only reports on motion; a still mouse inside the window counts as fresh
        return HandState(t=time.monotonic(), present=self.inside,
                         confidence=1.0 if self.inside else 0.0,
                         pos=np.array([self.x, self.y, self.z]), pinch=float(self.pinch),
                         fist=self.fist, handedness="right")
