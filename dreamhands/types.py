"""Shared team interfaces. DRAFT v0 (Person B): tell the team before changing fields.

Coordinate conventions
- HandState.pos: normalized camera coords. x: 0 = image left .. 1 = image right (as the
  operator sees the mirrored preview), y: 0 = bottom .. 1 = top, z: relative depth,
  larger = hand closer to the camera (unitless, e.g. palm size).
- Robot vectors are 14-D, right arm first (YAM dataset layout):
  right_arm[0:6], right_grip[6], left_arm[7:13], left_grip[13]. Gripper 1 = open, 0 = closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HandState:
    """What A's tracker publishes (latest frame)."""
    t: float = 0.0                       # time.monotonic() of the frame
    present: bool = False                # a hand is detected
    confidence: float = 0.0              # 0..1
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # see module doc
    pinch: float = 0.0                   # 0 = open .. 1 = thumb/index touching
    fist: bool = False                   # clutch gesture
    open_palm: bool = False              # optional stop gesture
    handedness: str = "right"            # "right" | "left"


@dataclass
class Tick:
    """What the runner emits every sim step (20 Hz) to on_tick listeners (C, D)."""
    step: int                            # step counter since start
    t: float                             # time.monotonic()
    state: np.ndarray                    # (14,) measured joints+grip, right-first
    action: np.ndarray                   # (14,) commanded targets, right-first
    images: dict | None                  # {"top","left","right"} RGB 360x640, or None
    hand: HandState
    mode: str                            # DISARMED | READY | ENGAGED | HOLD | DREAM
    tcp: np.ndarray = field(default_factory=lambda: np.zeros(3))  # right TCP world pos (m)
    flags: dict = field(default_factory=dict)  # ik_reject, workspace_edge, tracking_lost, ...
    events: list = field(default_factory=list)  # keyboard events: "record", "next", "discard"


def to14(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Pack two 7-D arm vectors into the 14-D right-first layout."""
    return np.r_[np.asarray(right, float), np.asarray(left, float)]


def from14(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 14-D right-first into (right7, left7)."""
    v = np.asarray(v, float)
    return v[:7].copy(), v[7:].copy()
