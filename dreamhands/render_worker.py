"""Render the three dataset cameras on a background thread.

Rendering all three cameras costs ~100 ms, two whole 50 ms ticks. The worker owns its own
MjData and Renderer (GL contexts are thread-bound), gets a snapshot of the sim state for a
step and renders that, so the teleop loop keeps 20 Hz and the images still match the state
of the step they belong to. One thread manages ~11 fps of 3-camera frames on the team
laptop (RTX 3050), too close to the dataset's 10 fps; two threads share one job queue and
reach ~16 fps (a third adds nothing).
"""
from __future__ import annotations

import queue
import threading
import warnings

import mujoco

from .sim import CAMERAS, _CROP


class RenderWorker:
    def __init__(self, model, cameras=CAMERAS, height: int = 480, width: int = 640,
                 threads: int = 2):
        self.model, self.cameras = model, tuple(cameras)
        self._in: queue.Queue = queue.Queue()
        self._out: queue.Queue = queue.Queue()
        self._started = 0
        self._lock = threading.Lock()
        self._ths = []
        for i in range(threads):
            ready = threading.Event()
            th = threading.Thread(target=self._run, args=(height, width, ready), daemon=True,
                                  name=f"dreamhands-render-{i}")
            th.start()
            ready.wait(10.0)
            self._ths.append(th)

    @property
    def ok(self) -> bool:
        return self._started > 0

    def _run(self, height: int, width: int, ready: threading.Event) -> None:
        try:
            d = mujoco.MjData(self.model)
            r = mujoco.Renderer(self.model, height=height, width=width)
        except Exception as exc:  # no GL here: the runner falls back to sync rendering
            warnings.warn(f"background rendering unavailable ({exc})", RuntimeWarning)
            ready.set()
            return
        with self._lock:
            self._started += 1
        ready.set()
        while True:
            job = self._in.get()
            if job is None:
                break
            step, qpos, mocap_pos, mocap_quat, t = job
            d.qpos[:], d.mocap_pos[:], d.mocap_quat[:], d.time = qpos, mocap_pos, mocap_quat, t
            mujoco.mj_forward(self.model, d)
            out = {}
            for name in self.cameras:
                r.update_scene(d, camera=name)
                out[name] = r.render()[_CROP].copy()
            self._out.put((step, out))
        r.close()

    def submit(self, step: int, data) -> None:
        """Queue a render of `data` as it is right now (copied, safe to keep stepping)."""
        self._in.put((step, data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy(),
                      float(data.time)))

    def poll(self) -> list[tuple[int, dict]]:
        done = []
        while True:
            try:
                done.append(self._out.get_nowait())
            except queue.Empty:
                return done

    def wait(self) -> tuple[int, dict]:
        """Block for the next finished render."""
        return self._out.get(timeout=5.0)

    def close(self) -> None:
        for _ in self._ths:
            self._in.put(None)
        for th in self._ths:
            th.join(timeout=2.0)
