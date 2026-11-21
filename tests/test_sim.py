import numpy as np

from dreamhands.sim import HOME, Sim


def test_step_moves_right_arm():
    sim = Sim(render=False)
    ready = np.r_[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    for _ in range(20):
        sim.step(HOME, ready)
    q = sim.state()["right"]
    assert q.shape == (7,)
    assert abs(q[1] - 1.0) < 0.05 and abs(q[2] - 1.0) < 0.05
    sim.close()


def test_render_is_optional():
    sim = Sim()
    imgs = sim.render()
    # either real 360x640 RGB, or None when the machine has no GL context
    assert imgs is None or imgs["top"].shape == (360, 640, 3)
    sim.close()
