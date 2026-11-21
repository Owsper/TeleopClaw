"""Drive the right YAM arm from the mouse (or A's tracker later) at 20 Hz.

    python run_teleop.py --mouse

Keys: space = arm/disarm (arming goes to the ready pose), f = engage on/off,
p = precision mode, t = take-over (rebase on measured joints), r/n/x = record/next/discard
events for C, q or Esc = quit.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from dreamhands.runner import TeleopRunner
from dreamhands.teleop import DISARMED

WIN = "DreamHands"
DT = 0.05


def draw_hud(img, tick, runner):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img is not None else np.zeros((360, 640, 3), np.uint8)
    f = tick.flags
    lines = [f"{tick.mode}  engage={'on' if runner.engage else 'off'}"
             f"{'  PRECISION' if runner.teleop.precision else ''}",
             "tcp " + " ".join(f"{v:+.3f}" for v in tick.tcp) + f"  grip {runner.grip:.0f}",
             f"step {f.get('step_ms', 0):.0f} ms  ik {f.get('ik_err_mm', '-')} mm"]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    warn = [k for k in ("ik_reject", "workspace_edge", "tracking_lost") if f.get(k)]
    if warn:
        cv2.putText(img, " ".join(warn).upper(), (8, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", action="store_true", help="use the mouse instead of the webcam")
    ap.add_argument("--hud-every", type=int, default=3, help="render the top camera every N steps")
    args = ap.parse_args()
    if not args.mouse:
        raise SystemExit("webcam tracker comes from A; use --mouse for now")

    from dreamhands.fake_tracker import MouseTracker
    runner = TeleopRunner()
    tracker = MouseTracker(WIN)
    frame = None
    nxt = time.perf_counter()
    while True:
        keys = ""
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            if runner.teleop.mode == DISARMED:
                runner.go_ready()
            else:
                runner.teleop.disarm()
                runner.engage = False
        elif k == ord("f"):
            runner.engage = not runner.engage
        elif k == ord("p"):
            runner.teleop.set_precision(not runner.teleop.precision)
        elif k == ord("t"):
            runner.take_over()
        elif k != 255:
            keys = chr(k)

        tick = runner.tick(tracker.read(), keys)
        if tick.events:
            print("events:", tick.events)
        if tick.step % args.hud_every == 0:
            r = runner.sim.render(("top",))
            frame = r["top"] if r else None
        cv2.imshow(WIN, draw_hud(frame, tick, runner))

        nxt += DT
        delay = nxt - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            nxt = time.perf_counter()  # fell behind: don't try to catch up
    runner.sim.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
