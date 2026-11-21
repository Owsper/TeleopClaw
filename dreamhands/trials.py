"""Apple pick trials for the 17:00 go/no-go (>= 5 of 10).

Referee rule (ours; the dreamlabs package ships none): a trial succeeds when the apple is
lifted >= LIFT_M above where it started and stays there for HOLD_S, while the right gripper
is commanded closed. It fails on timeout or when the apple falls off the table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ik import TOP_DOWN
from .types import Tick

LIFT_M = 0.04
HOLD_S = 1.0
TIMEOUT_S = 60.0
APPROACH_Z = 0.86   # hover above the apple (ready z 0.88 is out of reach at the apple)
GRASP_Z = 0.80      # TCP height at the apple's equator (apple centre z ~0.792)


@dataclass
class Verdict:
    done: bool = False
    success: bool = False
    reason: str = ""
    lift: float = 0.0


class Referee:
    """Feed it every Tick (runner.on_tick); reads the apple straight from the sim."""

    def __init__(self, sim, body: str = "apple", dt: float = 0.05):
        self.sim, self.body, self.dt = sim, body, dt
        self.reset()

    def reset(self) -> None:
        self.z0 = float(self.sim.data.body(self.body).xpos[2])
        self.t = 0.0
        self.held = 0.0
        self.verdict = Verdict()

    def __call__(self, tick: Tick) -> Verdict:
        v = self.verdict
        if v.done:
            return v
        self.t += self.dt
        z = float(self.sim.data.body(self.body).xpos[2])
        v.lift = z - self.z0
        closed = tick.action[6] < 0.5
        self.held = self.held + self.dt if (v.lift >= LIFT_M and closed) else 0.0
        if self.held >= HOLD_S:
            v.done, v.success, v.reason = True, True, f"lifted {v.lift * 100:.1f} cm"
        elif z < self.z0 - 0.05:
            v.done, v.reason = True, "apple fell off the table"
        elif self.t >= TIMEOUT_S:
            v.done, v.reason = True, "timeout"
        return v


def scripted_pick(runner, referee: Referee, noise: float = 0.0, rng=None) -> Verdict:
    """Feasibility baseline: drive the same IK/sim path as teleop, no human.

    noise = std (m) of the XY aim error, to see how much slop the grasp tolerates.
    """
    rng = np.random.default_rng(rng)
    apple = runner.sim.data.body(referee.body).xpos.copy()
    xy = apple[:2] + rng.normal(0, noise, 2) if noise else apple[:2]
    cur = runner.tcp()

    def go(p, grip, steps):
        nonlocal cur
        p = np.asarray(p, float)
        for i in range(1, steps + 1):
            tgt = cur + (p - cur) * min(1.0, i / max(1, steps - 5))
            res, rej = runner.ik.step(tgt, TOP_DOWN, runner.q)
            if not rej:
                runner.q = res.q
            runner.grip = grip
            st = runner.sim.step(runner.left, runner.right_cmd())
            tick = Tick(step=0, t=0.0, state=np.r_[st["right"], st["left"]],
                        action=np.r_[runner.right_cmd(), runner.left], images=None,
                        hand=None, mode="SCRIPT", tcp=runner.tcp())
            if referee(tick).done:
                return True
        cur = p
        return False

    plan = [(np.r_[xy, APPROACH_Z], 1.0, 40), (np.r_[xy, GRASP_Z], 1.0, 30),
            (np.r_[xy, GRASP_Z], 0.0, 15), (np.r_[xy, APPROACH_Z], 0.0, 30),
            (np.r_[xy, APPROACH_Z], 0.0, 40)]
    for p, g, n in plan:
        if go(p, g, n):
            break
    return referee.verdict
