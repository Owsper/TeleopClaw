"""TeleopRunner: one call per 50 ms tick -> teleop -> IK -> sim -> Tick to listeners.

The runner never talks to the Dream API (C owns that). Listeners get a Tick every step;
images are attached only on steps where `record_images` asks for them (even steps by default)
because rendering all three cameras costs ~120 ms, more than the 50 ms budget.
"""
from __future__ import annotations

import time
from typing import Callable

import mujoco
import numpy as np

from .ik import TOP_DOWN, ArmIK
from .sim import HOME, Sim
from .teleop import DISARMED, Teleop
from .types import HandState, Tick, to14

READY_TCP = np.array([-0.05, 0.05, 0.88])
KEY_EVENTS = {"r": "record", "n": "next", "x": "discard"}


def camera_axes(sim: Sim, camera: str = "top") -> np.ndarray:
    """Hand->world axes so motion matches what the operator sees on the HUD camera.

    hand x (right)  -> image-right of the camera, flattened onto the table plane
    hand y (up)     -> world +z
    hand z (closer) -> camera look direction flattened onto the table (push = forward)
    Columns are world directions of hand x, y, z.
    """
    m, d = sim.model, sim.data
    mujoco.mj_kinematics(m, d)
    R = d.cam_xmat[m.camera(camera).id].reshape(3, 3)

    def flat(v):
        v = np.array([v[0], v[1], 0.0])
        return v / np.linalg.norm(v)

    return np.column_stack([flat(R[:, 0]), [0.0, 0.0, 1.0], flat(-R[:, 2])])


class TeleopRunner:
    def __init__(self, sim: Sim | None = None, teleop: Teleop | None = None,
                 record_images: bool = False, image_every: int = 2):
        self.sim = sim or Sim()
        self.ik = ArmIK(self.sim.model)
        if teleop is None:
            teleop = Teleop(READY_TCP)
            teleop.cfg.axes = camera_axes(self.sim, "top")
        self.teleop = teleop
        self.record_images, self.image_every = record_images, image_every
        self.on_tick: list[Callable[[Tick], None]] = []
        self.left = HOME.copy()
        self.q = self.sim.state()["right"][:6].copy()  # commanded right joints
        self.grip = 1.0
        self.step_i = 0
        self.engage = False
        self._ready_q = self.ik.solve_global(READY_TCP, TOP_DOWN).q

    # --- helpers ----------------------------------------------------------------------
    def tcp(self) -> np.ndarray:
        return self.sim.data.site("right_tcp").xpos.copy()

    def right_cmd(self) -> np.ndarray:
        return np.r_[self.q, self.grip]

    def go_ready(self, seconds: float = 2.0, hz: float = 20.0) -> float:
        """Interpolate joints to the top-down ready pose, arm the teleop, return pos_err (m)."""
        q0 = self.sim.state()["right"][:6].copy()
        n = max(1, int(seconds * hz))
        for i in range(1, n + 1):
            a = 0.5 - 0.5 * np.cos(np.pi * i / n)  # smooth start/stop
            self.q = (1 - a) * q0 + a * self._ready_q
            self.sim.step(self.left, np.r_[self.q, 1.0])
        for _ in range(10):  # settle
            self.sim.step(self.left, np.r_[self.q, 1.0])
        self.grip = 1.0
        self.teleop.arm(READY_TCP)
        err = float(np.linalg.norm(self.tcp() - READY_TCP))
        print(f"go_ready: pos_err {err * 1000:.1f} mm")
        return err

    def take_over(self) -> None:
        """Hand control back to the human mid-episode: start from measured joints."""
        meas = self.sim.state()["right"]
        self.q, self.grip = meas[:6].copy(), float(meas[6])
        self.teleop.rebase(self.tcp())

    # --- one tick ---------------------------------------------------------------------
    def tick(self, hand: HandState, keys: str = "", now: float | None = None) -> Tick:
        t0 = time.perf_counter()
        now = time.monotonic() if now is None else now
        events = [KEY_EVENTS[k] for k in keys if k in KEY_EVENTS]
        flags = {"ik_reject": False}

        if hand.open_palm:  # A's stop gesture: drop engage, operator re-engages with f
            self.engage = False
        engage = self.engage and not hand.fist  # fist = clutch (lift the mouse)
        target, grip = self.teleop.update(hand, engage, now)
        flags.update(self.teleop.flags)
        if self.teleop.mode != DISARMED:
            res, reject = self.ik.step(target, TOP_DOWN, self.q)
            if reject or not np.all(np.isfinite(res.q)):
                flags["ik_reject"] = True
            else:
                self.q = res.q
                flags["ik_err_mm"] = round(res.pos_err * 1000, 1)
            self.grip = grip

        state = self.sim.step(self.left, self.right_cmd())
        images = None
        if self.record_images and self.step_i % self.image_every == 0:
            images = self.sim.images()
        flags["step_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        tick = Tick(step=self.step_i, t=now,
                    state=to14(state["right"], state["left"]),
                    action=to14(self.right_cmd(), self.left),
                    images=images, hand=hand, mode=self.teleop.mode,
                    tcp=self.tcp(), flags=flags, events=events)
        for fn in self.on_tick:
            fn(tick)
        self.step_i += 1
        return tick
