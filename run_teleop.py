"""Drive the right YAM arm from the mouse (or A's tracker later) at 20 Hz.

    python run_teleop.py            # webcam via A's tracker (fist = clutch, pinch = grip)
    python run_teleop.py --mouse    # mouse stand-in

Keys: space = arm/disarm (arming goes to the ready pose), f = engage on/off,
p = precision lock (auto: near an object or fingers starting to pinch), t = take-over (rebase on measured joints), r/n/x = record/next/discard
events for C, q or Esc = quit.

    python run_teleop.py --mouse --trials 10   # 17:00 go/no-go, results -> trials.json
    python run_teleop.py --record data/dreamhands   # r = start an episode, n = end, x = discard

Recording: the referee ends an episode by itself once the apple is lifted and held; the
quality gate prints KEEP / DROP and only KEEP episodes are written (LeRobot v2.1). The scene
resets after every episode with the apple at a new random spot (--fixed keeps the stock scene).
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


def draw_hud(img, tick, runner, rec=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img is not None else np.zeros((360, 640, 3), np.uint8)
    f = tick.flags
    lines = [f"{tick.mode}  engage={'on' if runner.engage else 'off'}"
             f"{'  PRECISION' if runner.teleop.precise else ''}",
             "tcp " + " ".join(f"{v:+.3f}" for v in tick.tcp) + f"  grip {runner.grip:.0f}",
             f"step {f.get('step_ms', 0):.0f} ms  ik {f.get('ik_err_mm', '-')} mm"]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if rec is not None:
        last = rec.on_done.last
        txt = f"REC {rec.ep.dur:.1f} s" if rec.ep else f"kept {rec.on_done.w.n}  {last or ''}"
        cv2.putText(img, txt[:70], (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if rec.ep else (0, 255, 0), 1)
    warn = [k for k in ("ik_reject", "workspace_edge", "tracking_lost") if f.get(k)]
    if warn:
        cv2.putText(img, " ".join(warn).upper(), (8, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", action="store_true", help="use the mouse instead of the webcam")
    ap.add_argument("--hud-every", type=int, default=3, help="render the top camera every N steps")
    ap.add_argument("--trials", type=int, default=0,
                    help="go/no-go: N apple pick trials judged by the referee (n = give up / next)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index for A's tracker")
    ap.add_argument("--record", metavar="DIR", help="save KEEP episodes as a LeRobot dataset in DIR")
    ap.add_argument("--task", default="pick up the apple", help="prompt stored with the episodes")
    ap.add_argument("--fixed", action="store_true", help="recording: keep the stock apple position")
    args = ap.parse_args()

    from dreamhands.trials import Referee
    runner = TeleopRunner()
    cv2.namedWindow(WIN)
    if args.mouse:
        from dreamhands.fake_tracker import MouseTracker
        tracker = MouseTracker(WIN)
    else:  # A's webcam + MediaPipe tracker (needs hand_landmarker.task, see README)
        from dreamhands.a_tracker import ATracker
        tracker = ATracker(args.camera)
    referee = Referee(runner.sim) if args.trials or args.record else None
    rec = None
    if args.record:
        from dreamhands.recorder import Recorder
        from dreamhands.save import Saver
        from dreamhands.scene import scatter
        if not args.fixed:
            scatter(runner.sim)
        rec = Recorder(args.task, on_done=Saver(args.record),
                       referee=lambda: referee.verdict.success)
        runner.on_tick.append(rec.on_tick)
        runner.warm_images()  # GL contexts up before the first r, no hitch mid-demo
    results: list[dict] = []

    def next_trial(verdict):
        results.append({"trial": len(results) + 1, "success": verdict.success,
                        "reason": verdict.reason or "skipped", "seconds": round(referee.t, 1)})
        ok = sum(r["success"] for r in results)
        print(f"trial {len(results)}/{args.trials}: {results[-1]['reason']}  -> {ok}/{len(results)}")
        runner.sim.reset()
        runner.engage = False
        runner.go_ready()
        referee.reset()
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
        if rec is not None:
            if "record" in tick.events:
                referee.reset()
            ended = "next" in tick.events or "discard" in tick.events
            if rec.ep is not None and referee(tick).done:  # lifted and held, timeout or dropped
                runner.flush()  # the recorder must have every tick up to now
                rec.stop()
                ended = True
            if ended:  # fresh scene for the next episode
                runner.sim.reset()
                if not args.fixed:
                    scatter(runner.sim)
                runner.engage = False
                runner.go_ready()
                referee.reset()
            runner.record_images = rec.ep is not None  # 3 cameras only while recording
        if args.trials and runner.teleop.mode != DISARMED:
            v = referee(tick)
            if v.done or "next" in tick.events:
                next_trial(v)
                if len(results) >= args.trials:
                    break
        if runner.record_images and runner.last_images is not None:
            frame = runner.last_images["top"]  # newest 3-camera render, no extra cost
        elif tick.step % args.hud_every == 0:
            r = runner.sim.render(("top",))
            frame = r["top"] if r else None
        cv2.imshow(WIN, draw_hud(frame, tick, runner, rec))

        nxt += DT
        delay = nxt - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            nxt = time.perf_counter()  # fell behind: don't try to catch up
    runner.close()  # delivers the last ticks, stops the render thread
    runner.sim.close()
    if hasattr(tracker, "close"):
        tracker.close()
    cv2.destroyAllWindows()
    if results:
        import json
        ok = sum(r["success"] for r in results)
        with open("trials.json", "w") as fh:
            json.dump({"success": ok, "trials": len(results), "results": results}, fh, indent=1)
        print(f"GO/NO-GO: {ok}/{len(results)} ({'GO' if ok >= 5 else 'NO-GO'}), saved trials.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
