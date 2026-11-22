"""TeleopRunner: one call per 50 ms tick -> teleop -> IK -> sim -> Tick to listeners.

The runner never talks to the Dream API (C owns that). Listeners get a Tick every step, in
step order; images are attached only on steps where `record_images` asks for them (even
steps by default). Rendering all three cameras costs ~100 ms, more than the 50 ms budget, so
by default a background thread renders them (render_worker.py) and a tick that carries
images reaches listeners up to MAX_LAG steps late. Ticks with keyboard events, and
`flush()`, wait until everything before them is delivered, so listener state (C's recorder)
is current whenever the caller looks at it.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Callable

import mujoco
import numpy as np

from .ik import TOP_DOWN, ArmIK
from .sim import HOME, Sim
from .teleop import DISARMED, ENGAGED, Teleop
from .types import HandState, Tick, to14

READY_TCP = np.array([-0.05, 0.05, 0.88])
APPLE_HOVER_Z = 0.85  # reachable above the stock apple at the workspace edge
APPROACH_STEP = 0.006  # maximum TCP movement per tick, in meters
KEY_EVENTS = {"r": "record", "n": "next", "x": "discard"}
MAX_LAG = 4  # steps a tick may wait for its images before the loop blocks on them
OBJECTS = ("apple", "banana", "orange", "water_bottle", "bowl", "plate")


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


def default_async() -> bool:
    """Background rendering only where it is known to work: Windows (the recording laptop).

    macOS: GLFW contexts must live on the main thread. Linux headless EGL: contexts on
    worker threads abort the process in CI (SIGABRT, can't be caught). Override with
    DREAMHANDS_ASYNC_RENDER=1 / 0.
    """
    env = os.environ.get("DREAMHANDS_ASYNC_RENDER")
    if env is not None:
        return env == "1"
    return sys.platform == "win32"


class TeleopRunner:
    def __init__(self, sim: Sim | None = None, teleop: Teleop | None = None,
                 record_images: bool = False, image_every: int = 2,
                 async_images: bool | None = None, prefer_pinch: bool = True,
                 keyboard_axes: bool = False, gesture_planes: bool = False, direct_hand: bool = False):
        self.sim = sim or Sim()
        self.ik = ArmIK(self.sim.model)
        if teleop is None:
            ready = np.r_[READY_TCP[:2], APPLE_HOVER_Z] if gesture_planes or direct_hand else READY_TCP
            teleop = Teleop(ready)
            teleop.cfg.axes = camera_axes(self.sim, "top")
        if keyboard_axes or gesture_planes or direct_hand:
            teleop.cfg.direct_hand = direct_hand
            teleop.cfg.gesture_planes = gesture_planes
            teleop.cfg.keyboard_axes = keyboard_axes
            if gesture_planes or direct_hand:
                teleop.cfg.scale[1] = 0.5
            if direct_hand:
                teleop.cfg.scale[2] = 0.4
            teleop.cfg.auto_precision = False
            teleop.cfg.scale[0] = 0.85  # quarter-frame sweep reaches the stock apple
        self.teleop = teleop
        self.record_images, self.image_every = record_images, image_every
        self.async_images = default_async() if async_images is None else async_images
        self._worker = None
        self._queue: deque[Tick] = deque()   # ticks not yet delivered, in step order
        self._waiting: set[int] = set()      # steps whose images are still rendering
        self._done: dict[int, dict] = {}     # finished renders not yet attached
        self.last_images: dict | None = None  # newest delivered images (HUD)
        self.on_tick: list[Callable[[Tick], None]] = []
        self.left = HOME.copy()
        self.q = self.sim.state()["right"][:6].copy()  # commanded right joints
        self.grip = 1.0
        self.step_i = 0
        self.engage = False
        self.prefer_pinch = prefer_pinch
        self.approach_message = ""
        self._approach: deque[np.ndarray] = deque()
        self._approach_deadline = 0
        self._approach_requires_hand = True
        self._palm_was_open = False
        self._ready_q = self.ik.solve_global(teleop.ready_tcp, TOP_DOWN).q
        names = {self.sim.model.body(i).name for i in range(self.sim.model.nbody)}
        self.objects = [o for o in OBJECTS if o in names]  # auto-precision targets

    # --- helpers ----------------------------------------------------------------------
    def tcp(self) -> np.ndarray:
        return self.sim.data.site("right_tcp").xpos.copy()

    def right_cmd(self) -> np.ndarray:
        return np.r_[self.q, self.grip]

    # --- image delivery -------------------------------------------------------------
    def _render_worker(self):
        if self._worker is None and self.async_images and self.sim.can_render:
            from .render_worker import RenderWorker
            self._worker = RenderWorker(self.sim.model)
            if not self._worker.ok:
                self._worker, self.async_images = None, False
        return self._worker

    def _deliver(self, block: bool = False) -> None:
        """Emit queued ticks in order; block = wait for every pending render."""
        if self._worker is not None:
            self._done.update(self._worker.poll())
        while self._queue:
            t = self._queue[0]
            if t.step in self._waiting:
                while t.step not in self._done:
                    if not block and self.step_i - t.step < MAX_LAG:
                        return
                    step, imgs = self._worker.wait()
                    self._done[step] = imgs
                t.images = self._done.pop(t.step)
                self._waiting.discard(t.step)
            if t.images is not None:
                self.last_images = t.images
            self._queue.popleft()
            for fn in self.on_tick:
                fn(t)

    def warm_images(self) -> None:
        """Start the render threads now (~0.2 s) instead of on the first recorded tick."""
        self._render_worker()

    def flush(self) -> None:
        """Deliver every tick produced so far (waits for in-flight renders)."""
        self._deliver(block=True)

    def close(self) -> None:
        self.flush()
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def near_dist(self) -> float | None:
        """Distance (m) from the right TCP to the closest scene object, None if none."""
        if not self.objects:
            return None
        tcp = self.tcp()
        return min(float(np.linalg.norm(self.sim.data.body(o).xpos - tcp)) for o in self.objects)

    def object_between_jaws(self) -> str | None:
        """Return a scene object with active contacts on both right fingertips."""
        model = self.sim.model
        left, right = model.body("right_tip_left").id, model.body("right_tip_right").id
        touched = {left: set(), right: set()}
        for contact in self.sim.data.contact:
            if contact.efc_address < 0:
                continue
            a, b = model.geom_bodyid[contact.geom]
            if a in touched:
                touched[a].add(b)
            if b in touched:
                touched[b].add(a)
        shared = touched[left] & touched[right]
        return next((name for name in self.objects if model.body(name).id in shared), None)

    def go_ready(self, seconds: float = 2.0, hz: float = 20.0) -> float:
        """Interpolate joints to the top-down ready pose, arm the teleop, return pos_err (m)."""
        self._approach.clear()
        self.approach_message = ""
        q0 = self.sim.state()["right"][:6].copy()
        n = max(1, int(seconds * hz))
        for i in range(1, n + 1):
            a = 0.5 - 0.5 * np.cos(np.pi * i / n)  # smooth start/stop
            self.q = (1 - a) * q0 + a * self._ready_q
            self.sim.step(self.left, np.r_[self.q, 1.0])
        for _ in range(10):  # settle
            self.sim.step(self.left, np.r_[self.q, 1.0])
        self.grip = 1.0
        self.teleop.arm()
        err = float(np.linalg.norm(self.tcp() - self.teleop.ready_tcp))
        print(f"go_ready: pos_err {err * 1000:.1f} mm")
        return err

    def take_over(self) -> None:
        """Hand control back to the human mid-episode: start from measured joints."""
        if self.approaching:
            self.cancel_apple_approach()
        meas = self.sim.state()["right"]
        self.q, self.grip = meas[:6].copy(), float(meas[6])
        self.teleop.rebase(self.tcp())

    @property
    def approaching(self) -> bool:
        return bool(self._approach)

    def start_apple_approach(self, *, require_hand: bool = True) -> None:
        """Plan a short motion to the current apple position with the claw open."""
        if self.approaching:
            return
        if self.teleop.mode == DISARMED or self.grip < 0.5:
            self.approach_message = "Ready the arm and open the claw first"
            return
        apple = self.sim.data.body("apple").xpos
        goal = np.array([*apple[:2], APPLE_HOVER_Z])
        clamped, _ = self.teleop.ws.clamp(goal)
        if not np.all(np.isfinite(goal)) or np.linalg.norm(goal - clamped) > 0.002 or apple[2] < 0.76:
            self.approach_message = "Apple outside the approach area"
            return
        start, _ = self.ik.fk(self.q)
        if start[2] < clamped[2]:
            self._approach.append(np.array([*start[:2], clamped[2]]))
        self._approach.append(clamped)
        self._approach_deadline = self.step_i + 100
        self._approach_requires_hand = require_hand
        self.engage = True
        self.teleop.rebase(start)
        self.approach_message = "Moving above apple. F cancels."

    def cancel_apple_approach(self, message: str = "Approach canceled - F for manual control") -> None:
        self._approach.clear()
        self.engage = False
        self.approach_message = message

    def _step_approach(self, flags: dict) -> None:
        if self.step_i >= self._approach_deadline:
            self.cancel_apple_approach("Approach timed out - F for manual control")
            return
        goal = self._approach[0]
        current = self.teleop.target
        delta = goal - current
        distance = np.linalg.norm(delta)
        target = goal if distance <= APPROACH_STEP else current + delta * (APPROACH_STEP / distance)
        res, reject = self.ik.step(target, TOP_DOWN, self.q)
        if reject or not res.ok or not np.all(np.isfinite(res.q)):
            flags["ik_reject"] = True
            self.cancel_apple_approach("Approach blocked - F for manual control")
            return
        self.q = res.q
        self.teleop.rebase(target)
        flags["ik_err_mm"] = round(res.pos_err * 1000, 1)
        if distance <= APPROACH_STEP and np.linalg.norm(self.tcp() - goal) <= 0.006:
            self._approach.popleft()
            if not self.approaching:
                self.engage = self._approach_requires_hand
                self.approach_message = ("Manual control. Adjust, lower, then pinch."
                                         if self.engage else "Above apple. F resumes hand control.")

    # --- one tick ---------------------------------------------------------------------
    def tick(self, hand: HandState, keys: str = "", now: float | None = None) -> Tick:
        t0 = time.perf_counter()
        now = time.monotonic() if now is None else now
        events = [KEY_EVENTS[k] for k in keys if k in KEY_EVENTS]
        flags = {"ik_reject": False}

        guiding = self.approaching
        palm_stop = hand.open_palm and not self._palm_was_open
        if hand.present:
            self._palm_was_open = hand.open_palm
        # A sustained open palm can be part of the approach cue, so only a new
        # palm stop in manual mode drops engage. F can resume after a stop.
        if palm_stop and not guiding and self.teleop.fresh(hand, now):
            if self.teleop.cfg.gesture_planes or self.teleop.cfg.direct_hand:
                if self.teleop.mode == DISARMED:
                    self.go_ready()
                    self.engage = True
            elif not self.teleop.cfg.keyboard_axes:
                self.engage = False
        # During manual webcam control a strong pinch wins even on its first
        # frame. The tracker can also report a fist when the other fingers curl.
        pinch_grasp = (self.prefer_pinch and not guiding
                       and (hand.pinch >= self.teleop.cfg.pinch_close
                            or (self.grip < 0.5 and hand.pinch > self.teleop.cfg.pinch_open)))
        clutch = hand.fist and not pinch_grasp
        if "f" in keys:
            if guiding:
                self.engage = False  # the approach handles cancellation below
            else:
                # Pause only if the arm is actually following. A missing hand
                # or clutch can leave engage requested while motion is paused.
                following = (self.engage and self.teleop.mode == ENGAGED and not clutch
                             and self.teleop.fresh(hand, now))
                self.engage = not following
                self.approach_message = ""
        flags["clutch"] = clutch
        engage = self.engage and not clutch
        previous_target = self.teleop.target
        target, grip = self.teleop.update(hand, engage and not guiding, now, near_dist=self.near_dist(), keys=keys)
        flags.update(self.teleop.flags)
        if guiding:
            if flags["tracking_lost"] and self._approach_requires_hand:
                self.cancel_apple_approach("Hand lost. F resumes, G moves above apple.")
            elif not engage or self.teleop.mode == DISARMED:
                self.cancel_apple_approach()
            else:
                flags["assisted_approach"] = True
                self._step_approach(flags)
        elif self.teleop.mode != DISARMED:
            if self.teleop.mode == ENGAGED and not np.array_equal(previous_target, target):
                # A Cartesian step near a reach limit may require a large joint jump.
                # Try progressively shorter moves from the last accepted command.
                for fraction in (1.0, 0.5, 0.25, 0.125):
                    candidate = previous_target + fraction * (target - previous_target)
                    res, reject = self.ik.step(candidate, TOP_DOWN, self.q)
                    if res.ok and not reject and np.all(np.isfinite(res.q)):
                        self.q = res.q
                        self.teleop.target = candidate
                        flags["ik_err_mm"] = round(res.pos_err * 1000, 1)
                        break
                else:
                    self.teleop.reject_motion(previous_target)
                    flags["ik_reject"] = True
            self.grip = grip

        state = self.sim.step(self.left, self.right_cmd())
        if self.teleop.cfg.fruit_feedback and self.grip < 0.5:
            flags["between_jaws"] = self.object_between_jaws()
        images, pending = None, False
        if self.record_images and self.step_i % self.image_every == 0:
            worker = self._render_worker()
            if worker is not None:
                worker.submit(self.step_i, self.sim.data)  # state right after this step
                pending = True
            else:
                images = self.sim.images()
        flags["step_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        tick = Tick(step=self.step_i, t=now,
                    state=to14(state["right"], state["left"]),
                    action=to14(self.right_cmd(), self.left),
                    images=images, hand=hand, mode=self.teleop.mode,
                    tcp=self.tcp(), flags=flags, events=events)
        if pending:
            self._waiting.add(self.step_i)
        self._queue.append(tick)
        self.step_i += 1
        self._deliver(block=bool(events))
        return tick
