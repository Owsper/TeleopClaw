import numpy as np

from dreamhands.ik import TOP_DOWN, ArmIK
from dreamhands.sim import HOME, Sim

READY_TCP = np.array([-0.05, 0.05, 0.88])
APPLE_GRASP = np.array([0.15, 0.105, 0.80])


def _margin(ik, q):
    return float(np.min(np.minimum(q - ik.lo, ik.hi - q)))


def test_ready_pose_top_down_with_margin():
    sim = Sim(render=False)
    ik = ArmIK(sim.model)
    r = ik.solve_global(READY_TCP, TOP_DOWN)
    assert r.ok and _margin(ik, r.q) > 0.1
    for _ in range(60):
        sim.step(HOME, np.r_[r.q, 1.0])
    assert np.linalg.norm(sim.data.site("right_tcp").xpos - READY_TCP) < 0.01
    sim.close()


def test_apple_reachable_top_down():
    sim = Sim(render=False)
    ik = ArmIK(sim.model)
    assert ik.solve_global(APPLE_GRASP, TOP_DOWN).ok
    sim.close()


def test_teleop_sweep_tracks_without_rejects():
    sim = Sim(render=False)
    ik = ArmIK(sim.model)
    q = ik.solve_global(READY_TCP, TOP_DOWN).q
    worst, rejects = 0.0, 0
    for i in range(200):
        p = READY_TCP + np.array([0.15 * np.sin(i / 20), 0.05 * np.sin(i / 13), -0.04 * (1 - np.cos(i / 15))])
        res, rej = ik.step(p, TOP_DOWN, q)
        rejects += rej
        if not rej:
            q = res.q
        worst = max(worst, res.pos_err)
    assert rejects == 0 and worst < 0.005
    sim.close()
