"""Cross-platform wrapper around the official Dream YAM MuJoCo scene.

Works on Windows, macOS and Linux (desktop or headless):
- picks a MuJoCo GL backend before mujoco is imported (EGL on headless Linux,
  the platform default elsewhere; an explicit MUJOCO_GL always wins);
- finds `dream_examples/yam_mujoco/yam_env.py` inside the installed dreamlabs package
  (dream_examples is a namespace package, so no `__file__` tricks);
- rendering is optional: if no GL context can be created, physics keeps running and
  images come back as None instead of crashing the loop.

Run `python -m dreamhands.sim` for a smoke test.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import warnings
from pathlib import Path


def _pick_gl_backend() -> str | None:
    """Choose MUJOCO_GL for this machine; returns the chosen value (None = MuJoCo default)."""
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        return "egl"
    return None  # Windows: glfw/wgl, macOS: cgl, Linux desktop: glfw


GL_BACKEND = _pick_gl_backend()

import mujoco  # noqa: E402  (must come after MUJOCO_GL is set)
import numpy as np  # noqa: E402


def _load_yam_module():
    """Import yam_env from the installed dreamlabs package."""
    try:
        return importlib.import_module("dream_examples.yam_mujoco.yam_env")
    except ImportError:
        pass
    for base in map(Path, sys.path):
        cand = base / "dream_examples" / "yam_mujoco"
        if (cand / "yam_env.py").is_file():
            sys.path.insert(0, str(cand))
            return importlib.import_module("yam_env")
    raise ImportError(
        "dream_examples/yam_mujoco/yam_env.py not found. Install it with: pip install dreamlabs"
    )


_yam = _load_yam_module()
YamEnv = _yam.YamEnv
HOME = _yam.HOME
home_path = _yam.home_path
dc = getattr(_yam, "dream_control", None)

CAMERAS = ("top", "left", "right")
_CROP = slice(60, 420)  # same crop YamEnv.images() uses -> 360x640


class Sim:
    """Thin, OS-independent facade over YamEnv used by the teleop runner."""

    def __init__(self, render: bool = True):
        self.env = YamEnv()
        self.model = self.env.model
        self.data = self.env.data
        self._render_ok = render
        self._renderer = None

    def reset(self) -> dict:
        return self.env.reset()

    def state(self) -> dict:
        return self.env.state()

    def step(self, left, right) -> dict:
        return self.env.step(left, right)

    @property
    def can_render(self) -> bool:
        return self._render_ok

    def _get_renderer(self):
        if self._renderer is None and self._render_ok:
            try:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            except Exception as exc:  # no GL context (headless without EGL, etc.)
                self._render_ok = False
                warnings.warn(
                    f"MuJoCo rendering unavailable ({exc}); running physics only. "
                    "On headless Linux install libegl1 or set MUJOCO_GL=osmesa.",
                    RuntimeWarning,
                )
        return self._renderer

    def render(self, cameras=("top",)) -> dict | None:
        """Render only the requested cameras (top only is ~1/3 the cost of all three)."""
        r = self._get_renderer()
        if r is None:
            return None
        out = {}
        for name in cameras:
            r.update_scene(self.data, camera=name)
            out[name] = r.render()[_CROP].copy()
        return out

    def images(self) -> dict | None:
        """All three cameras, same format as YamEnv.images()."""
        return self.render(CAMERAS)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.env.close()


def _smoke() -> int:
    print(f"platform={sys.platform} python={sys.version.split()[0]} "
          f"mujoco={mujoco.__version__} MUJOCO_GL={GL_BACKEND or 'default'}")
    sim = Sim()
    ready = np.r_[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    t = time.perf_counter()
    for _ in range(20):
        sim.step(HOME, ready)
    step_ms = (time.perf_counter() - t) / 20 * 1000
    q = sim.state()["right"]
    moved = abs(q[1] - 1.0) < 0.05 and abs(q[2] - 1.0) < 0.05
    print(f"step {step_ms:.1f} ms, right arm reached ready: {moved}")
    sim.render()  # warm-up
    t = time.perf_counter()
    imgs = sim.render()
    if imgs is None:
        print("render: unavailable (physics-only mode)")
    else:
        print(f"render top {1000 * (time.perf_counter() - t):.0f} ms, shape {imgs['top'].shape}")
    sim.close()
    return 0 if moved else 1


if __name__ == "__main__":
    raise SystemExit(_smoke())
