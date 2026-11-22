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

# hand delta (x right, y up, z closer to camera) -> world delta; columns = world dirs.
# Matches the `top` HUD camera (behind the arms, looking to -y): image-right = -x world,
# up = +z, pushing the hand toward the webcam = forward = -y. The runner re-derives this
# from the camera pose (runner.camera_axes), this is the fallback.
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
    # m of TCP per unit of normalized hand motion. x: a comfortable half-frame sweep (~0.5)
    # covers half the 0.5 m workspace width; y: the workspace is only 0.10 m tall, so vertical
    # is gentler for the grasp approach; z: depth from palm size is the noisiest axis.
    scale: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.3, 0.3]))
    direct_hand: bool = False  # wrist x/y and calibrated apparent palm size
    gesture_planes: bool = False  # V switches between tabletop aim and vertical motion
    keyboard_axes: bool = False  # webcam controls x and grip, keys control depth and height
    ema: float = 0.35            # weight of the new sample (1 = no smoothing)
    deadband: float = 0.003      # m; smaller target changes are ignored (kills jitter)
    max_speed: float = 0.5       # m/s TCP rate limit
    lost_timeout: float = 0.10   # s without a fresh confident frame -> HOLD
    min_conf: float = 0.5
    pinch_close: float = 0.6     # hysteresis on pinch -> gripper
    pinch_open: float = 0.4
    precision: float = 0.3       # scale multiplier in precision mode
    # auto precision (only while the gripper is open: once it holds something, lift fast)
    auto_precision: bool = True
    near_on: float = 0.05        # m TCP-object distance: enter precision ..
    near_off: float = 0.07       # .. leave it (hysteresis, no flicker at the boundary)
    prepinch_on: float = 0.25    # fingers start closing (below pinch_close) -> precision
    prepinch_off: float = 0.15


    @property
    def fruit_feedback(self) -> bool:
        return self.direct_hand or self.gesture_planes or self.keyboard_axes


class Teleop:
    def __init__(self, ready_tcp, workspace: Workspace | None = None,
                 cfg: TeleopConfig | None = None):
        self.cfg = cfg or TeleopConfig()
        self.ws = workspace or Workspace()
        self.ready_tcp = np.asarray(ready_tcp, float)
        self.mode = DISARMED
        self.target = self.ready_tcp.copy()   # last commanded TCP
        self.grip = 1.0                       # 1 open, 0 closed
        self.precision = False                # manual toggle (key p)
        self._near = False                    # auto triggers (hysteresis state)
        self._prepinch = False
        self._was_precise = False             # effective precision on the last tick
        self._distance_ratio = 1.0
        self._hand0 = None                    # hand anchor (filtered)
        self._tcp0 = None                     # TCP anchor
        self._filt = None                     # EMA state of the hand
        self.height_mode = False
        self._v_since = None
        self._v_switched = False
        self.flags: dict = {}

    # --- mode control -------------------------------------------------------------
    def arm(self, tcp=None) -> None:
        self.mode = READY
        self.height_mode = False
        self._v_since = None
        self._v_switched = False
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

    def reject_motion(self, previous_target: np.ndarray) -> None:
        """At a reach limit, let the next hand movement start at the accepted pose."""
        self.target = previous_target
        self._tcp0 = previous_target.copy()
        self._hand0 = self._filt.copy()
        self._distance_ratio = 1.0

    def set_precision(self, on: bool) -> None:
        self.precision = on  # update() re-anchors when the effective scale changes

    @property
    def precise(self) -> bool:
        """Effective precision: manual toggle, or auto near an object / pre-pinch."""
        auto = self.cfg.auto_precision and self.grip > 0.5 and (self._near or self._prepinch)
        return self.precision or auto

    def _auto_triggers(self, hand: HandState, near_dist: float | None) -> None:
        c = self.cfg
        if near_dist is None:
            self._near = False
        else:
            self._near = near_dist < (c.near_off if self._near else c.near_on)
        self._prepinch = hand.pinch >= (c.prepinch_off if self._prepinch else c.prepinch_on)

    # --- per tick -------------------------------------------------------------------
    def fresh(self, hand: HandState, now: float) -> bool:
        return (hand.present and hand.confidence >= self.cfg.min_conf
                and now - hand.t <= self.cfg.lost_timeout
                and (not self.cfg.direct_hand or (np.all(np.isfinite(hand.pos)) and hand.pos[2] > 0)))

    def update(self, hand: HandState, engage: bool, now: float, dt: float = 0.05,
               near_dist: float | None = None, keys: str = ""):
        """Returns (tcp_target, grip). engage = operator wants the arm to follow.

        near_dist: distance (m) from the TCP to the closest graspable object, if known.
        """
        c = self.cfg
        fresh = self.fresh(hand, now)
        if fresh:
            self._auto_triggers(hand, near_dist)
        self.flags = {"tracking_lost": not fresh, "workspace_edge": False,
                      "precision": self.precise}
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

        if c.gesture_planes:
            if self.mode == ENGAGED and hand.v_sign:
                if self._v_since is None:
                    self._v_since = hand.t
                if not self._v_switched and hand.t - self._v_since >= 0.35:
                    self.height_mode = not self.height_mode
                    self._v_switched = True
                self._hand0 = None
                return self.target, self.grip
            self._v_since = None
            self._v_switched = False

        if self.mode != ENGAGED:
            return self.target, self.grip

        # gripper first: anchoring below must never delay a grasp by a tick
        was_open = self.grip > 0.5
        if hand.pinch >= c.pinch_close:
            self.grip = 0.0
        elif hand.pinch <= c.pinch_open:
            self.grip = 1.0
        if c.fruit_feedback and was_open and self.grip < 0.5:
            self._hand0 = None  # closing fingers can shift the measured palm centre
        if self.precise != self._was_precise:  # depends on grip (auto only while open)
            self._was_precise = self.precise
            self._hand0 = None  # re-anchor so the scale change doesn't jump
        self.flags["precision"] = self.precise

        if self._hand0 is None:  # (re)anchor: zero motion on this tick
            self._filt = raw.copy()  # discard motion accumulated before the anchor
            self._distance_ratio = 1.0
            self._hand0, self._tcp0 = self._filt.copy(), self.target.copy()
            return self.target, self.grip

        k = c.precision if self.precise else 1.0
        motion = self._filt - self._hand0
        if c.direct_hand:
            # Apparent palm span varies inversely with distance. Calibrate at
            # each anchor and absorb small variations in the monocular estimate.
            ratio = self._hand0[2] / self._filt[2]
            self._distance_ratio = float(np.clip(self._distance_ratio, ratio - 0.025, ratio + 0.025))
            # An off-centre wrist projects farther from the image centre as it
            # approaches the lens, even with no physical x/y movement.
            motion[:2] = (self._filt[:2] - 0.5) * self._distance_ratio - (self._hand0[:2] - 0.5)
            motion[2] = 1.0 - self._distance_ratio
        delta = c.scale * motion
        if c.gesture_planes:
            if self.height_mode:
                delta = np.array([0.0, delta[1], 0.0])
            else:
                delta = np.array([delta[0], 0.0, delta[1]])
        if c.keyboard_axes:
            delta[1:] = 0
            if self.grip < 0.5 and ("k" in keys or "j" in keys):
                # A keyed lift/lower keeps the current sideways command. Rebase
                # that axis so wrist motion on this tick cannot be replayed later.
                self._filt[0] = self._hand0[0] = raw[0]
                side = c.axes[:, 0]
                self._tcp0 += side * np.dot(self.target - self._tcp0, side)
                delta[0] = 0
            # Each key event moves the persistent anchor 5 mm. Clamp below
            # discards excess at the workspace walls, even on repeated keys.
            depth = int("w" in keys) - int("s" in keys)
            height = int("k" in keys) - int("j" in keys)
            self._tcp0 += 0.005 * (depth * c.axes[:, 2] + height * c.axes[:, 1])
        want = self._tcp0 + k * (c.axes @ delta)
        clamped, edge = self.ws.clamp(want)
        if c.fruit_feedback:
            limits = ("sideways", "depth", "height")
            self.flags["blocked_axes"] = [name for name, excess in zip(limits, want - clamped)
                                          if abs(excess) > 1e-6]
        # Discard travel beyond each wall so reversing the hand moves the arm
        # back immediately, while motion parallel to the wall stays continuous.
        self._tcp0 += clamped - want
        want = clamped
        step = want - self.target
        n = float(np.linalg.norm(step))
        if n >= c.deadband:
            vmax = c.max_speed * dt
            if n > vmax:
                step *= vmax / n
            self.target = self.target + step
        self.flags["workspace_edge"] = edge
        return self.target, self.grip
