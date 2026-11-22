from types import SimpleNamespace

import numpy as np

from dreamhands import ui
from dreamhands.runner import TeleopRunner
from dreamhands.sim import Sim
from dreamhands.trials import Referee
from dreamhands.types import HandState
from gestures import Gestures
from hand_tracker import hand_states


def test_mirrored_hand_labels_survive_detection_reordering_and_duplicates():
    filters = {side: Gestures() for side in ("left", "right")}

    def detection(label, x):
        points = [SimpleNamespace(x=x, y=0.6, z=0) for _ in range(21)]
        points[9].y = 0.4
        return [SimpleNamespace(category_name=label, score=0.98)], points

    def read(detections):
        labels, points = zip(*detections)
        result = SimpleNamespace(handedness=labels, hand_landmarks=points)
        return hand_states(result, filters, 640, 480, 1.0, both_hands=True)

    right = detection("Left", 0.25)  # crossed hands in the mirrored frame
    left = detection("Right", 0.75)
    for detections in ([left, right], [right, left]):
        states = read(detections)
        assert states["left"].valid and states["right"].valid
        assert states["left"].px[0, 0] == 480
        assert states["right"].px[0, 0] == 160
    assert not read([left])["right"].valid
    ambiguous = read([left, left])
    assert not any(h.valid for h in ambiguous.values())


def test_two_arms_independent_pick_hold_and_reacquire():
    runner = TeleopRunner(Sim(render=False), direct_hand=True, bimanual=True)
    now = 0.0

    def frame(x=0, y=0, distance=1, pinch=0, right_x=0, left_present=True):
        nonlocal now
        now += 0.05
        def hand(side, position, grip, present=True):
            return HandState(t=now, present=present, confidence=1, handedness=side,
                             pos=np.array(position), pinch=grip, open_palm=now < 1)
        left = hand("left", (0.5 + x / distance, 0.5 + y / distance, 0.5 / distance),
                    pinch, left_present)
        right = hand("right", (0.5 + right_x, 0.5, 0.5), 0)
        return runner.tick(right, now=now, left_hand=left)

    try:
        t0 = runner.sim.data.time
        for _ in range(65):
            frame()
        np.testing.assert_allclose(runner.sim.data.time - t0, 65 * 0.05, atol=1e-7)
        for arm in (runner, runner.left_arm):
            np.testing.assert_allclose(arm.tcp(), arm.teleop.ready_tcp, atol=0.003)

        ref = Referee(runner.sim)
        # Move the left wrist toward the stock apple while steering the right away.
        x = -0.03 / 0.85
        distance = 1 + 0.055 / 0.4 + 0.025
        for i in range(1, 31):
            f = i / 30
            frame(x * f, distance=1 + (distance - 1) * f, right_x=0.09 * f)
        for i in range(1, 37):
            frame(x, -0.101 * min(i / 24, 1), distance, right_x=0.09)
        for _ in range(20):
            closed = frame(x, -0.101, distance, 1, right_x=0.09)
        assert closed.action[6] == 1 and closed.action[13] == 0
        assert closed.flags["left"]["between_jaws"] == "apple"
        for i in range(1, 80):
            lifted = frame(x, -0.101 + 0.15 * min(i / 35, 1), distance, 1, right_x=0.09)
            ref(lifted)
        assert ref.verdict.success

        for _ in range(14):
            lost = frame(right_x=0.15, left_present=False)
            np.testing.assert_array_equal(lost.action[7:], lifted.action[7:])
        assert lost.flags["left"]["tracking_lost"] and not lost.flags["right"]["tracking_lost"]
        assert np.linalg.norm(lost.action[:6] - lifted.action[:6]) > 0.01
        reacquired = frame(x=0.24, y=0.22, pinch=1, right_x=0.15)
        np.testing.assert_array_equal(reacquired.action[7:], lost.action[7:])
    finally:
        runner.close()
        runner.sim.close()


def test_left_can_start_alone_and_pause_while_right_stays_parked():
    r = TeleopRunner(Sim(render=False), direct_hand=True, bimanual=True)
    try:
        parked = r.right_cmd()
        for i in range(65):
            t = i * 0.05
            left = HandState(t=t, present=True, confidence=1, pos=np.full(3, 0.5),
                             open_palm=True, handedness="left")
            tick = r.tick(HandState(), now=t, left_hand=left)
            np.testing.assert_array_equal(tick.action[:7], parked)
        assert r.left_arm.engage and not r.engage
        assert not tick.flags["tracking_lost"]
        labels = [text for text, _ in ui.status(tick, r, now=t)]
        assert "LEFT ENGAGED  pinch 0.00" in labels
        assert "RIGHT --" in labels
        for command, engaged in [("f", False), ("f", True)]:
            left.t += 0.05
            tick = r.tick(HandState(), keys=command, now=left.t, left_hand=left)
            assert r.left_arm.engage == engaged
            np.testing.assert_array_equal(tick.action[:7], parked)
    finally:
        r.close()
        r.sim.close()
