import time

import numpy as np
import pytest

from dreamhands.runner import TeleopRunner
from dreamhands.sim import Sim
from dreamhands.takeover import DREAM, FIST_S, HUMAN, DryPolicy, Takeover
from dreamhands.teleop import DISARMED
from dreamhands.types import HandState


def _hand(fist=False, palm=False, x=0.5):
    return HandState(t=time.monotonic(), present=True, confidence=1.0,
                     pos=np.array([x, 0.5, 0.5]), fist=fist, open_palm=palm)


class Spy(DryPolicy):
    def __init__(self, sim):
        super().__init__(sim, rtt=0.0)
        self.resets = 0

    def reset_episode(self):
        self.resets += 1
        super().reset_episode()


def _setup(rec=None):
    sim = Sim(render=True)
    if sim.images() is None:
        pytest.skip("no GL on this machine")
    r = TeleopRunner(sim, async_images=False)
    pol = Spy(sim)
    return r, pol, Takeover(r, pol, rec=rec)


def _run_dream(to, n):
    for _ in range(n):
        to.step(_hand())
        if to.ask.busy:
            to.ask.th.join()


def test_dream_moves_the_arm_after_the_window_fills():
    r, pol, to = _setup()
    q0 = r.sim.state()["right"][:6].copy()
    _run_dream(to, 80)
    assert to.mode == DREAM and pol.resets == 1 and to.fed
    assert np.abs(r.sim.state()["right"][:6] - q0).max() > 0.05


def test_fist_takes_over_without_a_jump_and_palm_gives_back():
    r, pol, to = _setup()
    _run_dream(to, 80)
    n = round(FIST_S / 0.05)
    for i in range(n - 1):
        to.step(_hand(fist=True))
        assert to.mode == DREAM, f"switched early at {i}"
        if to.ask.busy:
            to.ask.th.join()
    tcp = r.tcp()  # where Dream-1 left the arm
    to.step(_hand(fist=True))
    assert to.mode == HUMAN and r.engage and r.teleop.mode != DISARMED
    assert np.linalg.norm(r.tcp() - tcp) < 0.01  # no jump on take-over
    for _ in range(5):  # fist = clutch: holding it keeps the arm where it is
        to.step(_hand(fist=True))
    assert np.linalg.norm(r.tcp() - tcp) < 0.01
    to.step(_hand(palm=True))
    assert to.mode == DREAM and pol.resets == 2 and not to.fed


def test_keys_switch_and_human_segment_is_recorded():
    from dreamhands.recorder import Recorder
    eps = []
    rec = Recorder("pick up the apple", on_done=eps.append)
    r, pol, to = _setup(rec)
    _run_dream(to, 20)
    to.step(_hand(), "h")
    assert to.mode == HUMAN and rec.ep is not None
    for i in range(40):  # 2 s of human ticks
        to.step(_hand(x=0.5 + 0.002 * i))
    to.step(_hand(), "g")
    assert to.mode == DREAM and rec.ep is None
    assert len(eps) == 1 and eps[0].dur >= 1.6 and eps[0].n > 0
