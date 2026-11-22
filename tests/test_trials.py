import numpy as np

from dreamhands.runner import TeleopRunner
from dreamhands.sim import Sim
from dreamhands.trials import LIFT_M, Referee, scripted_pick
from dreamhands.types import HandState


def test_scripted_pick_lifts_apple():
    """The grasp is physically feasible through our IK/sim path (go/no-go baseline)."""
    r = TeleopRunner(Sim(render=False))
    r.go_ready()
    v = scripted_pick(r, Referee(r.sim))
    assert v.success, v
    r.sim.close()


def test_referee_does_not_pass_untouched_apple():
    r = TeleopRunner(Sim(render=False))
    r.go_ready()
    ref = Referee(r.sim)
    for i in range(40):
        hand = HandState(t=float(i), present=True, confidence=1.0, pos=np.full(3, 0.5))
        v = ref(r.tick(hand, now=float(i)))
    assert not v.done and abs(v.lift) < 0.005
    r.sim.close()


def test_apple_lifts_when_webcam_pinch_also_looks_like_fist():
    sim = Sim(render=False)
    runner = TeleopRunner(sim)
    try:
        runner.go_ready()
        runner.start_apple_approach(require_hand=False)
        for i in range(100):
            runner.tick(HandState(), now=i * 0.05)
            if not runner.approaching:
                break
        assert not runner.approaching
        runner.engage = True
        now = i * 0.05

        def move_hand(y, pinch=0.0, fist=False, keys=""):
            nonlocal now
            now += 0.05
            hand = HandState(t=now, present=True, confidence=1.0,
                             pos=np.array([0.5, y, 0.5]), pinch=pinch, fist=fist)
            return runner.tick(hand, keys=keys, now=now)

        for step in range(115):
            move_hand(0.6 - min(step, 100) * 0.005)
        start_height = sim.data.body("apple").xpos[2]
        closed = move_hand(0.1, pinch=1, fist=True)
        assert closed.action[6] == 0 and not closed.flags["clutch"]
        paused = move_hand(0.1, pinch=1, fist=True, keys="f")
        assert not runner.engage
        np.testing.assert_array_equal(paused.action, closed.action)
        move_hand(0.25)
        resumed = move_hand(0.25, pinch=1, fist=True, keys="f")
        assert runner.engage and resumed.action[6] == 0
        np.testing.assert_array_equal(resumed.action, paused.action)
        for step in range(65):
            tick = move_hand(0.25 + max(0, step - 20) * 0.006, pinch=1, fist=True)
            assert tick.action[6] == 0 and not tick.flags["clutch"]
        assert sim.data.body("apple").xpos[2] - start_height >= LIFT_M

        runner.engage = False
        paused = move_hand(0.9, pinch=1, fist=True)
        np.testing.assert_allclose(paused.action, tick.action, atol=1e-3)
    finally:
        runner.close()
        sim.close()


def test_manual_webcam_pick_after_recentering_without_approach():
    runner = TeleopRunner(Sim(render=False), keyboard_axes=True)
    try:
        runner.go_ready()
        referee = Referee(runner.sim)
        now = 0.0
        x = 0.5
        y, depth = 0.5, 0.5
        pinch = 0

        def move(keys="", steps=1):
            nonlocal now
            for _ in range(steps):
                now += 0.05
                hand = HandState(t=now, present=True, confidence=1.0,
                                 pos=np.array([x, y, depth]), pinch=pinch,
                                 open_palm=now >= 1.5 and pinch == 0)
                tick = runner.tick(hand, keys=keys, now=now)
                referee(tick)
            return tick

        move("f")
        move("j", 6)
        move("s", 11)
        move("f")
        x = 0.6
        move()
        move("f")
        held_axes = runner.teleop.target[1:].copy()
        for _ in range(20):
            x -= 0.20 / (20 * runner.teleop.cfg.scale[0])
            y -= 0.012  # slide diagonally down and toward the camera
            depth += 0.015
            move()
            np.testing.assert_allclose(runner.teleop.target[1:], held_axes)
        move(steps=12)
        assert runner.engage
        move("j", 10)
        move(steps=12)
        pinch = 1
        move(steps=15)
        move("k", 12)
        move(steps=30)
        assert referee.verdict.success, referee.verdict
    finally:
        runner.close()
        runner.sim.close()


def test_orange_depth_alignment_and_vertical_lift():
    runner = TeleopRunner(Sim(render=False), keyboard_axes=True)
    try:
        runner.go_ready()
        ref = Referee(runner.sim, body="orange")
        now, x, pinch = 0.0, 0.5, 0.0

        def move(keys="", steps=1):
            nonlocal now
            for _ in range(steps):
                now += 0.05
                hand = HandState(t=now, present=True, confidence=1.0,
                                 pos=np.array([x, 0.5, 0.5]), pinch=pinch)
                tick = runner.tick(hand, keys=keys, now=now)
                ref(tick)
            return tick

        move("f")
        x -= 0.05 / runner.teleop.cfg.scale[0]
        move("w", 23)  # screenshot depth -0.065 m
        move("j", 12)
        pinch = 1
        assert move(steps=10).flags["between_jaws"] is None
        pinch = 0
        move("k", 6)
        move("s", 39)  # stock orange at +0.130 m
        move("j", 10)
        move(steps=15)
        pinch = 1
        assert move(steps=15).flags["between_jaws"] == "orange"
        before = runner.teleop.target.copy()
        for _ in range(12):
            x += 0.008  # operator's wrist drifts sideways while tapping lift
            move("k")
            np.testing.assert_allclose(runner.teleop.target[:2], before[:2], atol=1e-6)
        move(steps=30)
        assert ref.verdict.success, ref.verdict
    finally:
        runner.close()
        runner.sim.close()
