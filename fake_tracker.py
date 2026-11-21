"""Simple interface stub for TeleopClaw control integration before webcam is ready."""
import time
import numpy as np
from common.types import HandState


class FakeTracker:
    def __init__(self):
        self.pinch = 1.0
        self.fist = False
        self.stop = False
        self.valid = True
        self.px = np.zeros((21, 2), dtype=np.float32)
        self.px[0] = [320, 300]
        self.px[9] = [320, 240]

    def latest(self):
        return HandState(
            self.valid, time.monotonic(), self.px.copy(),
            60.0, self.pinch, self.fist, self.stop,
        )
