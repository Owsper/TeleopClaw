import time

import numpy as np
import pytest

from gestures import Gestures

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

@pytest.mark.parametrize("direct", [False, True])
def test_recentering_then_sliding_left_with_palm_tilt_keeps_depth(direct):
    tracker = Gestures()
    runner = TeleopRunner(Sim(render=False), direct_hand=direct)
    # Hand in pixel units relative to its wrist, fingers pointed up.
    palm = np.zeros((21, 3))
    for base, x in zip((5, 9, 13, 17), (-30, -10, 10, 30)):
        palm[base:base + 4, 0] = x
        palm[base:base + 4, 1] = [-75, -95, -115, -135]
    palm[4] = [-65, -55, 0]

    def sample(i, x, angle, scale=1, keys=""):
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
        xyz = scale * (palm @ rotation.T)
        px = xyz[:, :2] + [x, 290]
        now = i * 0.05
        h = from_a(tracker.update(px, now, xyz[:, 2]), wrist_origin=direct)
        return runner.tick(h, keys=keys, now=now)

    try:
        runner.go_ready()
        sample(0, 220, 0, keys="f")
        sample(1, 220, 0, keys="f")
        start = sample(2, 320, 0.85, keys="f")
        for i in range(1, 17):
            end = sample(2 + i, 320 - 4 * i, 0.85 * (1 - i / 16))
        assert end.tcp[0] > start.tcp[0] + 0.025  # left in the mirrored HUD
        assert abs(end.tcp[1] - start.tcp[1]) < 0.003  # forward/back on table
        for i in range(19, 32):
            closer = sample(i, 256, 0, scale=1 + (i - 18) * 0.025)
        assert closer.tcp[1] < end.tcp[1] - 0.01
    finally:
        runner.sim.close()


def test_one_hand_orange_pick_from_landmarks_without_keys():
    from dreamhands.trials import Referee

    r = TeleopRunner(Sim(render=False), gesture_planes=True)
    tracker = Gestures()
    points = np.zeros((21, 2))
    for base, x in zip((5, 9, 13, 17), (-34, -8, 16, 32)):
        points[base:base + 4, 0] = x
        points[base:base + 4, 1] = [-70, -90, -110, -134]
    points[4] = [-85, -60]
    now = 0.0

    def frame(dx=0, dy=0, gesture="open"):
        nonlocal now
        now += 0.05
        px = points + [320 + dx, 250 + dy]
        if gesture == "V":
            px[[16, 20], 1] = px[[13, 17], 1]
        elif gesture == "pinch":
            px[4] = px[8]
        return r.tick(from_a(tracker.update(px, now)), now=now)

    try:
        for _ in range(25):
            frame()
        assert r.engage and not r.teleop.height_mode
        ref = Referee(r.sim, body="orange")
        # Physical hand movement in pixels in the mirrored 640 x 480 frame.
        x, y = -38, 77
        for i in range(1, 21):
            frame(x * i / 20, y * i / 20)
        for _ in range(12):
            frame(x, y)
        anchor = r.teleop.target.copy()
        for _ in range(20):
            frame(x, y, "V")
        assert r.teleop.height_mode
        np.testing.assert_array_equal(anchor, r.teleop.target)
        frame(x, y)
        for i in range(1, 25):
            frame(x, y + 48 * i / 24)
        for _ in range(20):
            closed = frame(x, y + 48, "pinch")
        assert closed.flags["between_jaws"] == "orange"
        for i in range(1, 49):
            lift = frame(x + i, y + 48 - min(i, 30) * 2, "pinch")
            ref(lift)
        assert ref.verdict.success
        np.testing.assert_allclose(r.teleop.target[:2], anchor[:2])
        for _ in range(12):
            frame(x, y)
        assert r.grip == 1.0
    finally:
        r.close()
        r.sim.close()

def test_direct_hand_pick_with_simultaneous_xyz_and_no_mode_switch():
    from dreamhands.trials import Referee

    runner = TeleopRunner(Sim(render=False), direct_hand=True)
    tracker = Gestures()
    points = np.zeros((21, 2))
    for base, x in zip((5, 9, 13, 17), (-34, -8, 16, 32)):
        points[base:base + 4, 0] = x
        points[base:base + 4, 1] = [-70, -90, -110, -134]
    points[4] = [-85, -60]
    now = 0.0

    def move(dx=0, dy=0, distance=1, pinch=False):
        nonlocal now
        now += 0.05
        # Pinhole projection around image centre, wrist initially off centre.
        px = (points + [60 + dx, 10 + dy]) / distance + [320, 240]
        if pinch:
            px[4] = px[8]
        sample = from_a(tracker.update(px, now), wrist_origin=True)
        return runner.tick(sample, now=now)

    try:
        for _ in range(24):
            move()
        assert runner.engage
        ref = Referee(runner.sim, body="orange")
        for i in range(1, 25):
            f = i / 24
            move(-38 * f, 10 * f, 1 + 0.225 * f)
        for _ in range(15):
            move(-38, 10, 1.225)
        hover = runner.tcp().copy()
        assert abs(hover[0]) < 0.006
        assert abs(hover[1] - 0.13) < 0.006
        assert hover[2] < 0.845
        for i in range(1, 26):
            move(-38, 10 + 38 * i / 25, 1.225)
        for _ in range(12):
            move(-38, 48, 1.225)
        assert np.linalg.norm(runner.tcp()[:2] - hover[:2]) < 0.005
        for _ in range(16):
            closed = move(-38, 48, 1.225, pinch=True)
        assert closed.flags["between_jaws"] == "orange"
        for i in range(55):
            tick = move(-38, 48 - min(i, 32) * 2, 1.225, pinch=True)
            ref(tick)
        assert ref.verdict.success
        for _ in range(12):
            move(-38, -16, 1.225)
        assert runner.grip == 1
    finally:
        runner.close()
        runner.sim.close()
