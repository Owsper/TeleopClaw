import time

import numpy as np
import pytest

from dreamhands.runner import MAX_LAG, TeleopRunner, default_async
from dreamhands.sim import Sim
from dreamhands.types import HandState


def _hand():
    return HandState(t=time.monotonic(), present=True, confidence=1.0, pos=np.full(3, 0.5))


pytestmark = pytest.mark.skipif(not default_async(),
                                reason="background rendering is off on this platform")


def _runner(async_images):
    sim = Sim(render=True)
    if sim.images() is None:
        pytest.skip("no GL on this machine")
    r = TeleopRunner(sim, record_images=True, image_every=2, async_images=async_images)
    r.go_ready(seconds=0.5)
    return r


def test_async_images_in_order_and_match_sync():
    r = _runner(True)
    got = []
    r.on_tick.append(got.append)
    sync = {}
    for _ in range(12):
        r.tick(_hand())
        if r.step_i % 4 == 1:  # spot-check: sync render of the state the worker snapshotted
            sync[r.step_i - 1] = r.sim.images()
    assert len(got) >= 12 - MAX_LAG  # the loop never runs more than MAX_LAG ahead
    r.flush()
    assert [t.step for t in got] == list(range(12))
    for t in got:
        assert (t.images is not None) == (t.step % 2 == 0)
    for step, imgs in sync.items():
        a = next(t for t in got if t.step == step).images
        for cam in ("top", "left", "right"):
            assert a[cam].shape == (360, 640, 3)
            assert np.array_equal(a[cam], imgs[cam])
    r.close()


def test_events_flush_so_listeners_are_current():
    r = _runner(True)
    got = []
    r.on_tick.append(got.append)
    for _ in range(3):
        r.tick(_hand())
    r.tick(_hand(), keys="n")
    assert [t.step for t in got] == [0, 1, 2, 3] and got[-1].events == ["next"]
    r.close()


def test_async_keeps_the_20hz_budget():
    def tick_ms(r):
        r.warm_images()
        r.tick(_hand()); r.tick(_hand())
        ms = []
        for _ in range(30):  # paced like run_teleop: 50 ms per tick
            t0 = time.perf_counter()
            r.tick(_hand())
            ms.append((time.perf_counter() - t0) * 1000)
            time.sleep(max(0.0, 0.05 - ms[-1] / 1000))
        r.close()
        return float(np.mean(ms)), float(np.percentile(ms, 90))
    a_mean, a_max = tick_ms(_runner(True))
    s_mean, s_max = tick_ms(_runner(False))
    print(f"async {a_mean:.0f}/{a_max:.0f} ms, sync {s_mean:.0f}/{s_max:.0f} ms (mean/p90)")
    assert a_mean < 0.5 * s_mean and a_max < 50
