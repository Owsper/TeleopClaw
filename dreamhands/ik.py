"""Damped-least-squares IK for the right YAM arm on site `right_tcp`.

Works on a private MjData copy, so it never disturbs the running sim.
Gripper approach axis = site +z (measured on the real scene).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# site frame with +z pointing straight down (gripper looking at the table)
TOP_DOWN = np.array([[0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0]])


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    rot_err: float
    ok: bool


def _rot_err(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Axis-angle error vector (world frame) taking R_cur to R_des."""
    q_cur, q_des, q_err, q_inv = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q_cur, np.ascontiguousarray(R_cur).reshape(-1))
    mujoco.mju_mat2Quat(q_des, np.ascontiguousarray(R_des).reshape(-1))
    mujoco.mju_negQuat(q_inv, q_cur)
    mujoco.mju_mulQuat(q_err, q_des, q_inv)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q_err, 1.0)
    return v


class ArmIK:
    def __init__(self, model: mujoco.MjModel, side: str = "right", site: str | None = None,
                 damping: float = 0.05, rot_weight: float = 0.3, max_jump: float = 0.25,
                 free_yaw: bool = True):
        self.m = model
        self.free_yaw = free_yaw
        self.d = mujoco.MjData(model)
        self.site = model.site(site or f"{side}_tcp").id
        jids = [model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = np.array([model.jnt_range[j][0] for j in jids])
        self.hi = np.array([model.jnt_range[j][1] for j in jids])
        self.damping, self.rot_weight, self.max_jump = damping, rot_weight, max_jump
        self._jp, self._jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # required before mj_jacSite
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, pos: np.ndarray, rot: np.ndarray | None, q0: np.ndarray, iters: int = 30,
              tol: float = 1e-3) -> IKResult:
        q = np.clip(np.asarray(q0, float).copy(), self.lo, self.hi)
        w = self.rot_weight
        for _ in range(iters):
            p, R = self.fk(q)
            ep = pos - p
            er = self._orient_err(R, rot)
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
                break
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            J, e = self._jp[:, self.dadr], ep
            if rot is not None:
                Jr = self._jr[:, self.dadr]
                if self.free_yaw:  # don't constrain rotation about the approach axis
                    z = rot[:, 2]
                    Jr = (np.eye(3) - np.outer(z, z)) @ Jr
                J = np.vstack([J, w * Jr])
                e = np.r_[ep, w * er]
            JJt = J @ J.T + self.damping ** 2 * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, R = self.fk(q)
        pe = float(np.linalg.norm(pos - p))
        re = float(np.linalg.norm(self._orient_err(R, rot)))
        return IKResult(q, pe, re, pe < 5e-3 and re < 0.05)

    def _orient_err(self, R: np.ndarray, rot: np.ndarray | None) -> np.ndarray:
        if rot is None:
            return np.zeros(3)
        if self.free_yaw:  # only align the approach (site z) axis
            return np.cross(R[:, 2], rot[:, 2])
        return _rot_err(R, rot)

    def step(self, pos, rot, q_cur, iters: int = 15) -> tuple[IKResult, bool]:
        """Teleop tick: solve from the current command; reject jumps > max_jump rad."""
        res = self.solve(pos, rot, q_cur, iters=iters)
        reject = bool(np.max(np.abs(res.q - q_cur)) > self.max_jump)
        return res, reject

    def solve_global(self, pos, rot, seeds: int = 24, rng: int = 0) -> IKResult:
        """Multi-seed solve for startup (no continuity constraint)."""
        g = np.random.default_rng(rng)
        best = None
        cands = [np.r_[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]] + [g.uniform(self.lo, self.hi) for _ in range(seeds)]
        for q0 in cands:
            r = self.solve(pos, rot, q0, iters=100)
            score = r.pos_err + 0.05 * r.rot_err
            if best is None or score < best[0]:
                best = (score, r)
            if r.ok:
                break
        return best[1]
