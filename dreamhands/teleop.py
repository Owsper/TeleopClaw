"""Hand -> TCP target mapping: state machine, clutch-relative motion, filters, workspace.

Pure logic, no sim: feed it HandState + flags each tick, get a TCP target and gripper.

States
- DISARMED: nothing is commanded from the hand (runner keeps the last command).
- READY:    armed, arm parked; waiting for engage.
- ENGAGED:  TCP follows the hand, relative to where both were at engage time.
- HOLD:     hand lost while engaged; TCP frozen. Hand back -> ENGAGED (re-anchored).

Clutch-relative: on every (re)engage we store the hand and TCP anchors, so taking the
hand away or making a fist (clutch) and coming back never makes the arm jump.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import HandState

DISARMED, READY, ENGAGED, HOLD = "DISARMED", "READY", "ENGAGED", "HOLD"

# hand delta (x right, y up, z closer to camera) -> world delta. Tuned at 15:30 (axes step).
# Default assumes the operator faces the same way the arm reaches (-y): right = -x world,
# up = +z, pushing the hand toward the camera = forward = -y.
DEFAULT_AXES = np.array([[-1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])


@dataclass
class Workspace:
    lo: np.ndarray = field(default_factory=lambda: np.array([-0.35, -0.10, 0.78]))
    hi: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.15, 0.88]))
    edge: float = 0.01  # flag workspace_edge when closer than this to a wall (m)

    def clamp(self, p: np.ndarray) -> tuple[np.ndarray, bool]:
        c = np.clip(p, self.lo, self.hi)
        near = bool(np.any(c - self.lo < self.edge) or np.any(self.hi - c < self.edge))
        return c, near


@dataclass
class TeleopConfig:
    axes: np.ndarray = field(default_factory=lambda: DEFAULT_AXES.copy())
    scale: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.3]))  # m per unit
    ema: float = 0.35            # weight of the new sample (1 = no smoothing)
    deadband: float = 0.003      # m; smaller target changes are ignored (kills jitter)
    max_speed: float = 0.5       # m/s TCP rate limit
    lost_timeout: float = 0.10   # s without a fresh confident frame -> HOLD
    min_conf: float = 0.5
    pinch_close: float = 0.6     # hysteresis on pinch -> gripper
    pinch_open: float = 0.4
    precision: float = 0.3       # scale multiplier in precision mode


class Teleop:
    def __init__(self, ready_tcp, workspace: Workspace | None = None,
                 cfg: TeleopConfig | None = None):
        self.cfg = cfg or TeleopConfig()
        self.ws = workspace or Workspace()
        self.ready_tcp = np.asarray(ready_tcp, float)
        self.mode = DISARMED
        self.target = self.ready_tcp.copy()   # last commanded TCP
        self.grip = 1.0                       # 1 open, 0 closed
        self.precision = False
        self._hand0 = None                    # hand anchor (filtered)
        self._tcp0 = None                     # TCP anchor
        self._filt = None                     # EMA state of the hand
        self.flags: dict = {}

    # --- mode control -------------------------------------------------------------
    def arm(self, tcp=None) -> None:
        self.mode = READY
        self.target = np.asarray(self.ready_tcp if tcp is None else tcp, float).copy()
        self.grip = 1.0
        self._hand0 = None

    def disarm(self) -> None:
        self.mode = DISARMED
        self._hand0 = None

    def rebase(self, tcp) -> None:
        """Take-over: continue from wherever the arm actually is now."""
        self.target = np.asarray(tcp, float).copy()
        self._hand0 = None
        if self.mode == ENGAGED:
            self.mode = HOLD  # re-anchors on the next fresh frame

    def set_precision(self, on: bool) -> None:
        if on != self.precision:
            self.precision = on
            self._hand0 = None  # re-anchor so the scale change doesn't jump

    # --- per tick -------------------------------------------------------------------
    def _fresh(self, hand: HandState, now: float) -> bool:
        return (hand.present and hand.confidence >= self.cfg.min_conf
                and now - hand.t <= self.cfg.lost_timeout)

    def update(self, hand: HandState, engage: bool, now: float, dt: float = 0.05):
        """Returns (tcp_target, grip). engage = operator wants the arm to follow."""
        c = self.cfg
        fresh = self._fresh(hand, now)
        self.flags = {"tracking_lost": not fresh, "workspace_edge": False}
        if self.mode == DISARMED:
            return self.target, self.grip

        if fresh:
            raw = np.asarray(hand.pos, float)
            self._filt = raw if self._filt is None else c.ema * raw + (1 - c.ema) * self._filt
        else:
            self._filt = None  # don't smear stale samples into the next engage

        if self.mode == READY and engage and fresh:
            self.mode = ENGAGED
            self._hand0 = None
        elif self.mode in (ENGAGED, HOLD) and not engage:
            self.mode = READY
            self._hand0 = None
        elif self.mode == ENGAGED and not fresh:
            self.mode = HOLD
            self._hand0 = None
        elif self.mode == HOLD and fresh:
            self.mode = ENGAGED

        if self.mode != ENGAGED:
            return self.target, self.grip

        if self._hand0 is None:  # (re)anchor: zero motion on this tick
            self._hand0, self._tcp0 = self._filt.copy(), self.target.copy()
            return self.target, self.grip

        k = c.precision if self.precision else 1.0
        want = self._tcp0 + k * (c.axes @ (c.scale * (self._filt - self._hand0)))
        want, edge = self.ws.clamp(want)
        step = want - self.target
        n = float(np.linalg.norm(step))
        if n >= c.deadband:
            vmax = c.max_speed * dt
            if n > vmax:
                step *= vmax / n
            self.target = self.target + step
        self.flags["workspace_edge"] = edge

        if hand.pinch >= c.pinch_close:
            self.grip = 0.0
        elif hand.pinch <= c.pinch_open:
            self.grip = 1.0
        return self.target, self.grip
