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
    assert tp2.target[2] < READY_TCP[2] - 0.02  # 0.1 * y-scale 0.3 = 3 cm, minus EMA lag


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


def _top_pixel(sim, p):
    """Perspective image coords (right, up) of world point p in the top camera, unitless."""
    m, d = sim.model, sim.data
    c = m.camera("top").id
    R, o = d.cam_xmat[c].reshape(3, 3), d.cam_xpos[c]
    pc = R.T @ (p - o)  # camera frame: x right, y up, -z forward
    return np.array([pc[0], pc[1]]) / -pc[2]


def test_axes_match_top_camera():
    from dreamhands.runner import camera_axes
    from dreamhands.teleop import DEFAULT_AXES
    sim = Sim(render=False)
    assert np.allclose(camera_axes(sim), DEFAULT_AXES, atol=0.02)
    for delta, check in [((0.1, 0, 0), lambda a, b: b[0] > a[0] + 0.01),   # right -> image right
                         ((0, 0, 0.1), lambda a, b: b[1] > a[1] + 0.005)]:  # closer -> image up
        runner = TeleopRunner(sim)
        runner.go_ready()
        runner.engage = True
        t = 0.0
        for i in range(30):
            t += DT
            runner.tick(hand(t, 0.5 + (np.array(delta) if i > 1 else 0)), now=t)
        assert check(_top_pixel(sim, READY_TCP), _top_pixel(sim, runner.tcp()))
    assert runner.tcp()[2] > 0.8  # pushing forward keeps height
    sim.close()


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


def _travel(near=None, pinch=0.0, n=30):
    """TCP travel (m) for a 0.2 hand sweep to the right, given auto-precision inputs."""
    tp = engaged()
    start = tp.target.copy()
    t = DT
    for i in range(n):
        t += DT
        x = 0.5 + 0.2 * min(1.0, (i + 1) / 10)
        tp.update(hand(t, (x, 0.5, 0.5), pinch=pinch), True, t, near_dist=near)
    return float(np.linalg.norm(tp.target - start)), tp


def test_auto_precision_near_object():
    far, tp = _travel(near=0.20)
    near, tp2 = _travel(near=0.03)
    assert not tp.flags["precision"] and tp2.flags["precision"]
    assert near < 0.45 * far  # 0.3x scale (re-anchor on entry eats a tick)


def test_auto_precision_prepinch_but_not_when_holding():
    free, _ = _travel(pinch=0.0)
    pre, tp = _travel(pinch=0.3)          # fingers closing, gripper still open
    assert tp.flags["precision"] and pre < 0.45 * free
    held, tp = _travel(pinch=0.9, near=0.0)  # gripper closed: lift at full speed
    assert tp.grip == 0.0 and not tp.flags["precision"]
    assert held > 0.8 * free


def test_precision_hysteresis_and_no_jump():
    tp = engaged()
    t = DT
    seen, prev = [], tp.target.copy()
    for i, d in enumerate([0.08, 0.06, 0.049, 0.06, 0.069, 0.071, 0.06]):
        t += DT
        tgt, _ = tp.update(hand(t, (0.5 + 0.01 * i, 0.5, 0.5)), True, t, near_dist=d)
        seen.append(tp.flags["precision"])
        assert np.linalg.norm(tgt - prev) < 0.03  # toggling never jumps the TCP
        prev = tgt.copy()
    assert seen == [False, False, True, True, True, False, False]


def test_manual_precision_toggle_reanchors():
    tp = engaged()
    t = run(tp, DT, 10, lambda i: (0.5 + 0.02 * i, 0.5, 0.5))
    before = tp.target.copy()
    tp.set_precision(True)
    tgt, _ = tp.update(hand(t + DT, (0.7, 0.5, 0.5)), True, t + DT)
    assert np.allclose(tgt, before) and tp.flags["precision"]


def test_runner_reports_object_distance():
    r = TeleopRunner(Sim(render=False))
    assert "apple" in r.objects
    d = r.near_dist()
    assert d is not None and 0.0 < d < 1.0


def test_clutch_released_after_fast_reposition_has_no_drift():
    tp = engaged()
    parked = tp.target.copy()
    tp.update(hand(0.10, (0.8, 0.5, 0.5)), False, 0.10)
    run(tp, 0.10, 12, lambda i: (0.8, 0.5, 0.5))
    np.testing.assert_allclose(tp.target, parked)


def test_fast_sweep_can_reverse_at_reach_limit():
    r = TeleopRunner(Sim(render=False))
    try:
        r.go_ready()
        r.engage = True
        rejected = False
        for i, x in enumerate([0.5] + [0.1] * 25 + [0.14] * 12):
            t = i * DT
            before = r.q.copy()
            tick = r.tick(hand(t, (x, 0.5, 0.5)), now=t)
            assert np.max(np.abs(r.q - before)) <= r.ik.max_jump
            rejected |= tick.flags["ik_reject"]
            if i == 25:
                edge = tick.tcp.copy()
                assert rejected
                assert np.linalg.norm(edge - r.teleop.target) < 0.006
        assert tick.tcp[0] < edge[0] - 0.01
    finally:
        r.sim.close()


def test_reverse_after_hand_travels_past_workspace_floor():
    tp = engaged()
    t = run(tp, DT, 30, lambda i: (0.5, 0.0, 0.5))
    assert np.isclose(tp.target[2], tp.ws.lo[2], atol=tp.cfg.deadband)
    run(tp, t, 6, lambda i: (0.5, 0.05, 0.5))
    assert tp.target[2] > tp.ws.lo[2] + 0.01


def test_f_during_tracking_loss_keeps_follow_requested():
    r = TeleopRunner(Sim(render=False))
    try:
        r.go_ready()
        p = (0.5, 0.5, 0.5)
        first = r.tick(hand(0, p), keys="f", now=0)
        lost = r.tick(hand(DT, p, present=False), keys="f", now=DT)
        assert r.engage and lost.mode == HOLD
        resumed = r.tick(hand(2 * DT, (0.8, 0.2, 0.5)), now=2 * DT)
        assert resumed.mode == ENGAGED
        np.testing.assert_array_equal(first.action, resumed.action)
    finally:
        r.sim.close()
