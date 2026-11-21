import time

from common.types import HandState as AHand
from dreamhands.a_tracker import from_a
from dreamhands.runner import TeleopRunner
from dreamhands.sim import Sim
from dreamhands.teleop import ENGAGED, HOLD
from fake_tracker import FakeTracker  # A's stub


def _at(a, t):
    s = a.latest()
    s.t = t
    return s


def test_from_a_maps_axes_and_inverts_pinch():
    a = FakeTracker()
    a.pinch = 0.1  # A: almost touching
    h = from_a(a.latest())
    assert h.present and h.pinch > 0.8
    assert 0 < h.pos[0] < 1 and 0 < h.pos[1] < 1
    a.px[:, 0] += 64  # hand moves right in the mirrored image
    a.px[:, 1] -= 48  # and up
    h2 = from_a(a.latest())
    assert h2.pos[0] > h.pos[0] + 0.05 and h2.pos[1] > h.pos[1] + 0.05
    assert not from_a(AHand.invalid(time.monotonic())).present


def test_a_stream_drives_arm_and_holds_on_loss():
    r = TeleopRunner(Sim(render=False))
    r.go_ready()
    r.engage = True
    a = FakeTracker()
    t = 0.0
    for _ in range(3):
        t += 0.05
        r.tick(from_a(_at(a, t)), now=t)
    assert r.teleop.mode == ENGAGED
    for _ in range(20):
        t += 0.05
        a.px[:, 0] += 3  # move right
        r.tick(from_a(_at(a, t)), now=t)
    assert r.tcp()[0] < -0.06  # right on the top view = -x world
    t += 0.05
    r.tick(from_a(AHand.invalid(t)), now=t)  # A reports invalid
    assert r.teleop.mode == HOLD
    a.stop = True
    t += 0.05
    r.tick(from_a(_at(a, t)), now=t)
    assert not r.engage  # stop gesture disengages
    r.sim.close()
