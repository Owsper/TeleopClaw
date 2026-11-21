import numpy as np

from dreamhands.runner import READY_TCP, TeleopRunner
from dreamhands.sim import Sim
from dreamhands.teleop import ENGAGED, HOLD, READY, Teleop
from dreamhands.types import HandState

DT = 0.05


def hand(t, pos, present=True, pinch=0.0, fist=False):
    return HandState(t=t, present=present, confidence=1.0 if present else 0.0,
                     pos=np.asarray(pos, float), pinch=pinch, fist=fist)


def engaged(start=(0.5, 0.5, 0.5)):
    tp = Teleop(READY_TCP)
    tp.arm()
    tp.update(hand(0.0, start), True, 0.0)
    assert tp.mode == ENGAGED
    tp.update(hand(DT, start), True, DT)  # anchor tick
    return tp


def run(tp, t0, n, pos_fn, **kw):
    t = t0
    for i in range(n):
        t += DT
        tp.update(hand(t, pos_fn(i), **kw), True, t)
    return t


def test_engage_does_not_jump():
    tp = Teleop(READY_TCP)
    tp.arm()
    tp.update(hand(0.0, (0.9, 0.1, 0.7)), True, 0.0)
    tgt, _ = tp.update(hand(DT, (0.9, 0.1, 0.7)), True, DT)
    assert np.allclose(tgt, READY_TCP)


def test_hand_right_and_up_move_tcp():
    tp = engaged()
    run(tp, DT, 40, lambda i: (0.6, 0.5, 0.5))  # +0.1 right
    assert tp.target[0] < READY_TCP[0] - 0.03  # right = -x world (default axes)
    tp2 = engaged()
    run(tp2, DT, 40, lambda i: (0.5, 0.4, 0.5))  # 0.1 down
    assert tp2.target[2] < READY_TCP[2] - 0.03


def test_rate_limit():
    tp = engaged()
    prev = tp.target.copy()
    t = DT
    for _ in range(5):
        t += DT
        tp.update(hand(t, (1.0, 0.0, 0.5)), True, t)
        assert np.linalg.norm(tp.target - prev) <= tp.cfg.max_speed * DT + 1e-9
        prev = tp.target.copy()


def test_jitter_below_5mm():
    rng = np.random.default_rng(0)
    tp = engaged()
    pts = []
    t = DT
    for _ in range(200):  # tracker noise ~0.4% of the image per axis
        t += DT
        tp.update(hand(t, 0.5 + rng.normal(0, 0.004, 3)), True, t)
        pts.append(tp.target.copy())
    pts = np.array(pts[20:])
    assert np.max(np.linalg.norm(pts - pts.mean(0), axis=1)) < 0.005


def test_lost_hand_holds_within_150ms_and_reanchors():
    tp = engaged()
    t = run(tp, DT, 10, lambda i: (0.55, 0.5, 0.5))
    last_seen = t
    held_at = None
    for _ in range(4):  # tracker stops publishing (stale frame)
        t += DT
        tp.update(hand(last_seen, (0.55, 0.5, 0.5)), True, t)
        if tp.mode == HOLD and held_at is None:
            held_at = t
    assert held_at is not None and held_at - last_seen <= 0.15
    frozen = tp.target.copy()
    for _ in range(2):  # hand comes back somewhere else: no jump
        t += DT
        tp.update(hand(t, (0.95, 0.1, 0.5)), True, t)
    assert tp.mode == ENGAGED and np.allclose(tp.target, frozen)


def test_clutch_release_goes_ready():
    tp = engaged()
    tp.update(hand(2 * DT, (0.5, 0.5, 0.5)), False, 2 * DT)
    assert tp.mode == READY


def test_workspace_clamp_and_edge_flag():
    tp = engaged()
    run(tp, DT, 80, lambda i: (0.5, 0.0, 0.5))  # push far down
    assert tp.target[2] >= tp.ws.lo[2] - 1e-9
    assert tp.flags["workspace_edge"]


def test_pinch_hysteresis():
    tp = engaged()
    t = DT
    for p, want in [(0.7, 0.0), (0.5, 0.0), (0.3, 1.0), (0.5, 1.0)]:
        t += DT
        _, g = tp.update(hand(t, (0.5, 0.5, 0.5), pinch=p), True, t)
        assert g == want


def test_runner_ready_and_follow():
    runner = TeleopRunner(Sim(render=False))
    assert runner.go_ready() < 0.01
    runner.engage = True
    ticks, t = [], 100.0
    runner.on_tick.append(ticks.append)
    for i in range(60):
        t += DT
        x = 0.5 + 0.15 * np.sin(i / 10)
        runner.tick(hand(t, (x, 0.5 - 0.1 * i / 60, 0.5)), keys="r" if i == 5 else "", now=t)
    assert len(ticks) == 60 and ticks[-1].state.shape == (14,) and ticks[-1].action.shape == (14,)
    assert not any(tk.flags["ik_reject"] for tk in ticks)
    assert ticks[5].events == ["record"]
    assert np.linalg.norm(ticks[-1].tcp - runner.teleop.target) < 0.02
    runner.take_over()
    assert np.allclose(runner.q, runner.sim.state()["right"][:6])
    runner.sim.close()
