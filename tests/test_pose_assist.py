from types import SimpleNamespace

import numpy as np
import pytest

from dreamhands.pose_assist import LevelHold, match_forearm
from dreamhands.runner import APPLE_HOVER_Z, TeleopRunner
from dreamhands.scene import scatter
from dreamhands.sim import Sim
from dreamhands.teleop import ENGAGED
from dreamhands.types import HandState


def test_pose_hold_uses_the_tracked_arm_and_fresh_frames():
    def landmark(x, y, visibility=0.96):
        return SimpleNamespace(x=x, y=y, visibility=visibility, presence=0.96)

    landmarks = [landmark(0, 0, visibility=0)] * 33
    landmarks[13], landmarks[15] = landmark(0.4, 0.4), landmark(0.2, 0.42)
    landmarks[14], landmarks[16] = landmark(0.6, 0.4), landmark(0.8, 0.6)
    result = SimpleNamespace(pose_landmarks=[landmarks])
    arm = match_forearm(result, (128, 202), 640, 480)
    assert arm.side == 0 and arm.level
    assert not match_forearm(result, (512, 288), 640, 480).level
    assert match_forearm(result, (320, 220), 640, 480) is None

    hold = LevelHold()
    assert hold.update(arm, 1.0, 1.01) == 0
    assert hold.update(arm, 1.0, 1.11) == 0  # repeated frame does not advance the hold
    assert hold.update(arm, 1.0, 1.19) == 0  # stale frame clears the hold
    for time in (1.2, 1.3, 1.4, 1.5, 1.6):
        assert hold.update(arm, time, time + 0.01) < 1
    assert hold.update(arm, 1.7, 1.71) == pytest.approx(1)


@pytest.mark.parametrize("seed", [None, 7])
def test_approach_reaches_apple_and_hands_back_control(seed):
    sim = Sim(render=False)
    runner = TeleopRunner(sim)
    try:
        if seed is not None:
            scatter(sim, rng=seed)
        runner.go_ready()
        runner.start_apple_approach()
        assert runner.approaching
        goal = np.r_[sim.data.body("apple").xpos[:2], APPLE_HOVER_Z]
        initial_action = runner.right_cmd()
        ticks = []
        for step in range(100):
            now = step * 0.05
            # The operator may still be holding the open palm used for the cue.
            hand = HandState(t=now, present=True, confidence=1.0, pos=np.full(3, 0.5),
                             open_palm=step >= 10)
            ticks.append(runner.tick(hand, now=now))
            if not runner.approaching:
                break
        assert runner.engage
        assert ticks[-1].tcp == pytest.approx(goal, abs=0.006)
        assert all(t.action[6] == 1 and not t.flags["ik_reject"] for t in ticks)
        actions = np.vstack([initial_action, *(t.action[:7] for t in ticks)])
        assert np.max(np.abs(np.diff(actions[:, :6], axis=0))) <= runner.ik.max_jump

        manual = runner.tick(hand, now=now + 0.05)
        assert manual.mode == ENGAGED
        assert manual.action[:7] == pytest.approx(ticks[-1].action[:7], abs=0.001)
    finally:
        runner.close()
        sim.close()


def test_lost_hand_aborts_motion_and_does_not_resume_it():
    sim = Sim(render=False)
    runner = TeleopRunner(sim)
    try:
        runner.go_ready()
        runner.start_apple_approach()
        hand = HandState(t=1.0, present=True, confidence=1.0)
        moving = runner.tick(hand, now=1.0)
        assert moving.flags["assisted_approach"]
        dropped = runner.tick(hand, now=1.15)
        assert not runner.approaching and not runner.engage
        assert dropped.action == pytest.approx(moving.action)
        hand.t = 1.2
        resumed = runner.tick(hand, now=1.2)
        assert "assisted_approach" not in resumed.flags
    finally:
        runner.close()
        sim.close()


def test_keyboard_approach_reaches_apple_without_webcam_tracking():
    sim = Sim(render=False)
    runner = TeleopRunner(sim)
    try:
        runner.go_ready()
        runner.start_apple_approach(require_hand=False)
        for step in range(100):
            tick = runner.tick(HandState(), now=step * 0.05)
            if not runner.approaching:
                break
        assert not runner.approaching
        assert not runner.engage
        goal = np.r_[sim.data.body("apple").xpos[:2], APPLE_HOVER_Z]
        assert tick.tcp == pytest.approx(goal, abs=0.006)
    finally:
        runner.close()
        sim.close()
