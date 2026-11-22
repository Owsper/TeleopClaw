from dataclasses import dataclass
import numpy as np


@dataclass
class HandState:
    """TeleopClaw vision-to-control contract. All positions are in the *mirrored* camera image."""
    valid: bool
    t: float                 # time.monotonic(), seconds, frame capture time
    px: np.ndarray           # (21, 2), image pixels, float32
    palm: float              # wrist -> middle MCP (pixel distance)
    pinch: float             # 0 = thumb/index touching; 1 = separated
    fist: bool               # debounced clutch gesture
    stop: bool               # still open palm for >= 1 second

    palm_span: float | None = None  # median 3D wrist-to-knuckle span in pixel units

    @classmethod
    def invalid(cls, t: float = 0.0) -> "HandState":
        return cls(False, t, np.zeros((21, 2), dtype=np.float32),
                   0.0, 1.0, False, False)
