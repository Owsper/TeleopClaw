"""Drive the YAM arms from the webcam or mouse at 20 Hz.

    python run_teleop.py            # open palm starts, move in 3D, pinch closes
    python run_teleop.py --pose-assist  # A primes a level-forearm cue to approach the apple
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
from pathlib import Path

import cv2
import numpy as np

from dreamhands.runner import TeleopRunner
from dreamhands.teleop import DISARMED
from dreamhands.pose_assist import LevelHold
from dreamhands import ui

WIN = "DreamHands"
WEBCAM_WIN = "DreamHands Webcam"
POSE_MODEL = "pose_landmarker_lite.task"
CUE_SECONDS = 8.0
DT = 0.05


def control_hint(tick, runner, pose_hint, mouse=False):
    if pose_hint:
        return pose_hint
    if runner.left_arm and tick.flags.get("left") and not runner.approaching:
        active = [arm for arm in (runner.left_arm, runner) if arm.engage]
        if not active:
            if any(arm.teleop.mode != DISARMED for arm in (runner, runner.left_arm)):
                return "Follow paused. Press F to resume the armed claws."
            return "Hold each open palm still for 1s to start its matching arm."
        for arm in active:
            side_flags = tick.flags[arm.side]
            if side_flags.get("between_jaws"):
                return f"{arm.side.title()} holding {side_flags['between_jaws']}. Pinch and raise to lift."
        if any(tick.flags[arm.side].get("readying") for arm in active):
            return "Keep your open palm visible while its arm moves into position."
        if any(tick.flags[arm.side].get("tracking_lost") for arm in active):
            return "Missing hand: its arm holds. Bring that hand back into view."
        for arm in active:
            side_flags = tick.flags[arm.side]
            if side_flags.get("ik_reject") or side_flags.get("blocked_axes"):
                return f"{arm.side.title()} arm at reach limit. Move back toward its last position."
            if side_flags.get("clutch"):
                return f"{arm.side.title()} arm paused. Relax that fist to move."
            if arm.grip < 0.5:
                return f"{arm.side.title()} claw closed without two-jaw contact. Open and adjust."
        return "Aim L or R at the fruit, lower that hand, then pinch."
    if tick.mode == DISARMED:
        if runner.teleop.cfg.gesture_planes or runner.teleop.cfg.direct_hand:
            return "Hold a still open palm for 1 second to start."
        return "Press F to ready the arm and start following."
    if not mouse and not runner.engage and tick.flags.get("tracking_lost"):
        return "Paused. Show your hand until green dots appear, then press F."
    if not runner.engage and runner.grip < 0.5 and not mouse:
        return "Paused, claw closed. Re-center, pinch, then press F."
    if runner.approaching or not runner.engage:
        return runner.approach_message or "Follow paused. Re-center your hand, then press F."
    if tick.flags.get("tracking_lost"):
        return "Follow is on. " + ("Move your mouse into the sim window." if mouse
                                    else "Move your hand back into view to resume.")
    if tick.flags.get("clutch"):
        return "Clutch paused. " + ("Release the right mouse button." if mouse
                                   else "Relax your fist to move.")
    if runner.teleop.cfg.fruit_feedback:
        if tick.flags.get("ik_reject"):
            return "Reach limit. Adjust height or depth, then move sideways."
        if tick.flags.get("blocked_axes"):
            return "At " + "/".join(tick.flags["blocked_axes"]) + " limit. Other axes still work."
    if runner.teleop.cfg.direct_hand:
        if runner.grip < 0.5:
            obj = tick.flags.get("between_jaws")
            return (f"Both jaws touch {obj}. Raise your hand while pinching." if obj else
                    "No two-jaw contact. Open, align the +, lower, then pinch.")
        return "Align + with fruit, lower your hand, then pinch."
    if runner.teleop.cfg.gesture_planes:
        if tick.hand.v_sign:
            mode = "HEIGHT" if runner.teleop.height_mode else "AIM"
            return f"{mode}   Hold V to switch once, then relax your fingers."
        if not runner.teleop.height_mode:
            return "AIM   Move the + onto a fruit. Hold a V sign for HEIGHT."
        obj = tick.flags.get("between_jaws")
        if obj:
            return f"Both jaws touch {obj}. Keep pinching and raise your hand."
        if runner.grip < 0.5:
            return "No two-jaw contact. Open, adjust height or aim, then pinch."
        return "HEIGHT   Lower your hand, pinch, then raise your hand."
    if runner.teleop.cfg.keyboard_axes:
        if runner.grip < 0.5:
            obj = tick.flags.get("between_jaws")
            if obj:
                return f"Both jaws touch {obj}. Keep pinching and tap K to lift."
            return "No two-jaw contact. Open, align the + with fruit, then lower."
        return "Align white + with fruit in top view. Tap J to lower, then pinch."
    if tick.flags.get("ik_reject"):
        return "Target is out of reach. Move back toward the last position."
    if runner.grip < 0.5:
        return "Claw closed. Keep pinching and raise your hand." if not mouse else "Claw closed."
    return "Following mouse. Left-click to close." if mouse else "Following your hand. Pinch to close."


def draw_aim_map(img, runner, tcp):
    """Table-plane inset sharing the operator's left/right and W/S directions."""
    ws, axes = runner.teleop.ws, runner.teleop.cfg.axes
    lo, hi = ws.lo, ws.hi
    if runner.left_arm:
        lo = np.minimum(lo, runner.left_arm.teleop.ws.lo)
        hi = np.maximum(hi, runner.left_arm.teleop.ws.hi)
    center = (lo + hi) / 2
    height, width = img.shape[:2]
    origin = np.array([width - 85, height - 52])
    scale = min(148 / (hi[0] - lo[0]), 74 / (hi[1] - lo[1]))

    def pixel(pos):
        delta = pos - center
        return tuple(np.rint(origin + scale * np.array([delta @ axes[:, 0],
                                                        -delta @ axes[:, 2]])).astype(int))

    cv2.rectangle(img, (width - 170, height - 111), (width - 2, height - 2), (28, 28, 28), -1)
    cv2.putText(img, "TOP   W up / S down" if runner.teleop.cfg.keyboard_axes else "TOP   + is claw", (width - 163, height - 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    for name, color in (("apple", (60, 60, 235)), ("orange", (0, 150, 255)),
                        ("banana", (45, 220, 235))):
        if name in runner.objects:
            cv2.circle(img, pixel(runner.sim.data.body(name).xpos), 5, color, -1, cv2.LINE_AA)
    if runner.left_arm:
        for arm in (runner.left_arm, runner):
            x, y = pixel(arm.tcp())
            cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
            cv2.putText(img, arm.side[0].upper(), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return
    cv2.drawMarker(img, pixel(tcp), (255, 255, 255), cv2.MARKER_CROSS, 17, 1, cv2.LINE_AA)


def draw_hud(img, tick, runner, rec=None, hint=""):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img is not None else np.zeros((360, 640, 3), np.uint8)
    f = tick.flags
    mode = "APPROACH" if runner.approaching else tick.mode
    lines = [f"{mode}  engage={'on' if runner.engage else 'off'}"
             f"{'  PRECISION' if runner.teleop.precise else ''}",
             "tcp " + " ".join(f"{v:+.3f}" for v in tick.tcp) + f"  grip {runner.grip:.0f}",
             f"step {f.get('step_ms', 0):.0f} ms  ik {f.get('ik_err_mm', '-')} mm"]
    if runner.left_arm:
        lines[:2] = [f"{arm.side.upper()}  {arm.teleop.mode}  claw {'open' if arm.grip else 'closed'}"
                     for arm in (runner.left_arm, runner)]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if rec is not None:
        last = rec.on_done.last
        txt = f"REC {rec.ep.dur:.1f} s" if rec.ep else f"kept {rec.on_done.w.n}  {last or ''}"
        cv2.putText(img, txt[:70], (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if rec.ep else (0, 255, 0), 1)
    cv2.putText(img, hint, (8, 102), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    controls = ("F follow/pause   W/S forward/back   J/K lower/lift   Space disarm"
                if runner.teleop.cfg.keyboard_axes else
                "F follow/pause   G above apple   Space ready/disarm")
    if runner.teleop.cfg.direct_hand:
        controls = "DIRECT   left/right, up/down, toward/away from lens"
    if runner.teleop.cfg.gesture_planes:
        controls = "V: index + middle fingers up, other fingers curled"
    cv2.putText(img, controls, (8, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    warn = [k for k in ("ik_reject", "workspace_edge", "tracking_lost") if f.get(k)]
    if runner.teleop.cfg.fruit_feedback and "workspace_edge" in warn:
        warn.remove("workspace_edge")  # the hint names the axis when motion is actually blocked
    if warn:
        cv2.putText(img, " ".join(warn).upper(), (8, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)
    if runner.teleop.cfg.fruit_feedback:
        draw_aim_map(img, runner, tick.tcp)
    return img


def draw_pose_cue(frame, arm, hint):
    if arm is not None:
        elbow = tuple(map(int, arm.elbow))
        wrist = tuple(map(int, arm.wrist))
        color = (80, 240, 100) if arm.level else (0, 190, 255)
        cv2.line(frame, elbow, wrist, color, 3, cv2.LINE_AA)
        cv2.circle(frame, elbow, 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, wrist, 6, color, -1, cv2.LINE_AA)
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 32), (width, height), (15, 15, 15), -1)
    cv2.putText(frame, hint, (12, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    input_mode = ap.add_mutually_exclusive_group()
    input_mode.add_argument("--mouse", action="store_true", help="use the mouse instead of the webcam")
    input_mode.add_argument("--keyboard-axes", action="store_true", help="use W/S for depth and J/K for height")
    input_mode.add_argument("--gesture-planes", action="store_true", help="use V to switch AIM/HEIGHT")
    input_mode.add_argument("--hand-3d", action="store_true",
                    help="use webcam hand height and palm depth instead of J/K and W/S")
    ap.add_argument("--hud-every", type=int, default=3, help="render the top camera every N steps")
    ap.add_argument("--trials", type=int, default=0,
                    help="go/no-go: N apple pick trials judged by the referee (n = give up / next)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index for A's tracker")
    ap.add_argument("--pose-assist", action="store_true",
                    help="press A and hold your forearm level to approach the apple")
    ap.add_argument("--record", metavar="DIR", help="save KEEP episodes as a LeRobot dataset in DIR")
    ap.add_argument("--split", action="store_true",
                    help="old layout: sim and webcam in two windows, no status bar")
    ap.add_argument("--task", default="pick up the apple", help="prompt stored with the episodes")
    ap.add_argument("--fixed", action="store_true", help="recording: keep the stock apple position")
    args = ap.parse_args()
    if args.pose_assist and args.mouse:
        ap.error("--pose-assist uses the webcam and cannot be combined with --mouse")
    if args.pose_assist and not Path(POSE_MODEL).is_file():
        ap.error(f"missing {POSE_MODEL}; see the pose assist setup in README.md")

    from dreamhands.trials import Referee
    direct = not (args.mouse or args.hand_3d or args.keyboard_axes or args.gesture_planes)
    runner = TeleopRunner(prefer_pinch=not args.mouse,
                          keyboard_axes=args.keyboard_axes,
                          gesture_planes=args.gesture_planes,
                          direct_hand=direct, bimanual=direct)
    cv2.namedWindow(WIN)
    if args.mouse:
        from dreamhands.fake_tracker import MouseTracker
        tracker = MouseTracker(WIN)
    else:  # A's webcam + MediaPipe tracker (needs hand_landmarker.task, see README)
        from dreamhands.a_tracker import ATracker, from_a
        from demo import draw as draw_tracking, draw_landmarks
        tracker = ATracker(args.camera, pose_model=POSE_MODEL if args.pose_assist else None, both_hands=direct)
        if args.split:
            cv2.namedWindow(WEBCAM_WIN)
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
        runner.reset_scene()
        runner.engage = False
        runner.go_ready()
        referee.reset()
    frame = None
    level_hold = LevelHold()
    cue_until = 0.0
    resume_follow = False
    nxt = time.perf_counter()
    while True:
        keys = ""
        k = cv2.waitKey(1) & 0xFF
        if ord("A") <= k <= ord("Z"):
            k += ord("a") - ord("A")
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            cue_until = 0.0
            if runner.teleop.mode == DISARMED and not (runner.left_arm and runner.left_arm.engage):
                runner.go_ready()
            else:
                runner.disarm()
                if runner.left_arm:
                    runner.left_arm.disarm()
        elif k == ord("f"):
            if (runner.teleop.mode == DISARMED and
                    (runner.left_arm is None or runner.left_arm.teleop.mode == DISARMED)):
                runner.arm_from_palm()
            keys = "f"
            cue_until = 0.0
        elif k == ord("g"):
            cue_until = 0.0
            if runner.approaching:
                runner.cancel_apple_approach()
            else:
                runner.start_apple_approach(require_hand=False)
        elif k == ord("a") and args.pose_assist:
            level_hold.reset()
            if cue_until:
                cue_until = 0.0
                runner.engage = resume_follow
                runner.approach_message = "Pose canceled. G moves above apple."
            elif runner.approaching:
                runner.cancel_apple_approach()
            elif runner.teleop.mode != DISARMED and runner.grip >= 0.5:
                resume_follow = runner.engage
                runner.engage = False
                cue_until = time.monotonic() + CUE_SECONDS
            else:
                runner.approach_message = "Space to ready the arm. Keep the claw open."
        elif k == ord("p"):
            runner.teleop.set_precision(not runner.teleop.precision)
        elif k == ord("t"):
            cue_until = 0.0
            runner.take_over()
        elif k != 255:
            keys = chr(k)

        camera_frame = arm = left_hand = None
        if args.mouse:
            hand = tracker.read()
        else:
            if tracker.tracker.error is not None:
                raise RuntimeError("Tracking worker failed") from tracker.tracker.error
            if cue_until:
                tracker.tracker.scan_pose.set()
            else:
                tracker.tracker.scan_pose.clear()
            camera_frame, states, fps, arm, pose_time = tracker.tracker.preview(both=True)
            camera_hand = states["right"]
            height, width = camera_frame.shape[:2] if camera_frame is not None else (480, 640)
            hand = from_a(camera_hand, width, height, wrist_origin=runner.teleop.cfg.direct_hand)
            if direct:
                left_hand = from_a(states["left"], width, height, wrist_origin=True, side="left")
        now = time.monotonic()
        pose_hint = ""
        if cue_until:
            if now >= cue_until or tracker.tracker.pose_error is not None or keys in ("n", "x"):
                cue_until = 0.0
                runner.engage = resume_follow
                runner.approach_message = "Pose not detected. G moves above apple."
            else:
                if pose_time < cue_until - CUE_SECONDS:
                    arm = None
                open_grip = not hand.fist and hand.pinch < runner.teleop.cfg.pinch_close
                progress = level_hold.update(arm if hand.present and open_grip else None, pose_time, now)
                if not hand.present:
                    pose_hint = "Hand not detected. Show your hand or press G."
                elif arm is None:
                    pose_hint = "Show your elbow and wrist, or press G."
                elif not open_grip:
                    pose_hint = "Separate thumb and index finger, or press G."
                else:
                    pose_hint = f"Forearm level   {progress:.0%}  |  G skips cue"
                if progress >= 1.0:
                    cue_until = 0.0
                    pose_hint = ""
                    runner.start_apple_approach()
        tick = runner.tick(hand, keys, now=now, left_hand=left_hand)
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
                cue_until = 0.0
                runner.reset_scene()
                if not args.fixed:
                    scatter(runner.sim)
                runner.engage = False
                runner.go_ready()
                referee.reset()
            runner.record_images = rec.ep is not None  # 3 cameras only while recording
        if args.trials and (runner.teleop.mode != DISARMED or (runner.left_arm and runner.left_arm.engage)):
            v = referee(tick)
            if v.done or "next" in tick.events:
                cue_until = 0.0
                next_trial(v)
                if len(results) >= args.trials:
                    break
        if runner.record_images and runner.last_images is not None:
            frame = runner.last_images["top"]  # newest 3-camera render, no extra cost
        elif tick.step % args.hud_every == 0:
            r = runner.sim.render(("top",))
            frame = r["top"] if r else None
        hint = control_hint(tick, runner, pose_hint, mouse=args.mouse)
        hud = draw_hud(frame, tick, runner, rec, hint)
        if camera_frame is not None:
            if direct:
                for side, state in states.items():
                    draw_landmarks(camera_frame, state, side.upper())
            else:
                draw_tracking(camera_frame, camera_hand, fps, keyboard_axes=args.keyboard_axes,
                              gesture_planes=args.gesture_planes)
            draw_pose_cue(camera_frame, arm, hint)
        if args.split:
            cv2.imshow(WIN, hud)
            if camera_frame is not None:
                cv2.imshow(WEBCAM_WIN, camera_frame)
        else:  # D's demo window: sim | webcam + status bar
            cv2.imshow(WIN, ui.compose(hud, camera_frame, ui.status(tick, runner, rec, now)))

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
