"""Take-over: Dream-1 drives the right arm, a human grabs control mid-episode and hands it back.

    python -m dreamhands.takeover --model <checkpoint id>            # webcam
    python -m dreamhands.takeover --mouse --dry-run                  # no network, scripted stand-in
    python -m dreamhands.takeover --model <id> --record data/takeover # save the human corrections

DREAM  : the policy loop of evalrun.py (frames every 4th step, 48-row chunks). While a request is
         in flight the sim is frozen, exactly like evalrun, but the window stays live.
HUMAN  : fist held >= 0.5 s (or key h). The runner rebases on the measured joints, so the arm
         never jumps; the fist is also the clutch, so it starts in READY and follows once the
         fist opens. With --record the segment is recorded from here.
back   : open palm held still (A's stop gesture) or key g. The segment is closed (the quality
         gate keeps it when >= 1.6 s), the policy gets reset_episode() and a refilled window.

Keys: h = human, g = give back to Dream-1, space / f / p as in run_teleop.py, q / Esc = quit.
"""
from __future__ import annotations

import argparse
import threading
import time
from collections import deque

import numpy as np

from .runner import TeleopRunner
from .teleop import DISARMED

DT = 0.05
FIST_S = 0.5
HUMAN, DREAM = "HUMAN", "DREAM"


class DryPolicy:
    """No-network stand-in: scripted apple pick through the same IK path, ~0.3 s per answer."""

    def __init__(self, sim, obj="apple", rtt=0.3):
        self.sim, self.obj, self.rtt = sim, obj, rtt
        self.rows, self.last = [], None

    def reset_episode(self):
        from .evalrun import pick
        self.rows = pick(self.sim, self.obj)
        self.last = self.sim.state()["right"]

    def set_observation(self, frames, obs, timestamp=None):
        pass

    def get_action(self):
        time.sleep(self.rtt)
        rows, self.rows = self.rows[:48], self.rows[48:]
        if rows:
            self.last = rows[-1]
        rows = rows + [self.last] * (48 - len(rows))
        left = self.sim.state()["left"].tolist()
        return {"q_r_chunk": [np.asarray(r).tolist() for r in rows], "q_l_chunk": [left] * 48}

    def close(self):
        pass


class Asker:
    """One get_action in a thread so the window and the fist detector stay live."""

    def __init__(self):
        self.th, self.rep, self.err = None, None, None

    @property
    def busy(self):
        return self.th is not None and self.th.is_alive()

    def start(self, pol):
        self.rep = self.err = None

        def run():
            try:
                self.rep = pol.get_action()
            except Exception as e:  # handled on the main thread
                self.err = e
        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()

    def take(self):
        if self.busy or (self.rep is None and self.err is None):
            return None, None
        rep, err, self.rep, self.err = self.rep, self.err, None, None
        return rep, err


class Takeover:
    """The mode machine, UI-free so tests can drive it with a fake tracker."""

    def __init__(self, runner: TeleopRunner, pol, task="pick up the apple", model="",
                 poses=False, actions=48, rec=None, start=DREAM):
        from .evalrun import dc, rate
        self.dc, self.r, self.pol = dc, runner, pol
        self.rate = rate(model)  # rows per s in a reply: 30 for our fine-tune, 20 for the stock models
        self.env = runner.sim.env
        self.task, self.model, self.poses, self.actions = task, model, poses, actions
        self.rec = rec
        if rec is not None and rec.on_tick not in runner.on_tick:
            runner.on_tick.append(rec.on_tick)
        self.mode = None
        self.queue: deque = deque()
        self.ask = Asker()
        self.fist_t = 0.0
        self.steps = 0          # policy steps (sim time the model has seen)
        self.fed = False        # whole window sent after reset_episode
        self.wait_until = 0.0   # retry backoff after a refused request
        self.status = ""
        self.frame = None
        self.switches = 0
        self.r.engage = False
        self.to_dream() if start == DREAM else self.to_human()

    # --- switches ---------------------------------------------------------------------
    def to_human(self):
        self.queue.clear()      # a reply still in flight is dropped when it lands
        r = self.r
        r.take_over()           # continue from measured joints: no jump
        if r.teleop.mode == DISARMED:
            r.teleop.arm(r.tcp())
            meas = r.sim.state()["right"]
            r.grip = r.teleop.grip = float(meas[6])  # keep a held object held
        r.engage = True
        self.mode, self.fist_t, self.status = HUMAN, 0.0, "human"
        self.switches += 1
        if self.rec is not None:
            r.record_images = True
            self.rec.start()

    def to_dream(self):
        r = self.r
        if self.rec is not None and self.rec.ep is not None:
            r.flush()           # the recorder must have every human tick
            self.rec.stop()
        r.record_images = False
        r.engage = False
        self.queue.clear()
        self.win = {n: deque(maxlen=self.dc.FRAMES) for n in self.dc.VIDEO_ROLES}
        self.fed = False
        try:
            self.pol.reset_episode()
        except Exception as e:
            self.status = f"reset_episode failed : {type(e).__name__} : {e}"
        st = r.sim.state()
        self.hold = (st["left"].copy(), st["right"].copy())
        self.mode = DREAM
        self.switches += 1

    # --- one 50 ms step -----------------------------------------------------------------
    def step(self, hand, keys=""):
        if "h" in keys and self.mode == DREAM:
            self.to_human()
        elif "g" in keys and self.mode == HUMAN:
            self.to_dream()
        self.fist_t = self.fist_t + DT if hand.fist else 0.0
        if self.mode == DREAM and self.fist_t >= FIST_S - 1e-9:
            self.to_human()
        elif self.mode == HUMAN and hand.open_palm:
            self.to_dream()
        if self.mode == HUMAN:
            return self.r.tick(hand, "".join(k for k in keys if k in "rnx"))
        self._dream()
        return None

    def _feed(self, every):
        self.dc.feed(self.pol, self.env, self.win, self.task, self.model, self.steps,
                     every=every, station=self.poses)

    def _sim(self, left, right):
        self.r.sim.step(left, right)
        if self.steps % self.dc.FRAME_EVERY_STEPS == 0:
            imgs = self.dc.capture(self.env, self.win)
            if imgs:
                self.frame = imgs.get("top")
            if self.fed and not self.ask.busy:
                self._feed(1)
        self.steps += 1

    def _dream(self):
        rep, err = self.ask.take()
        if err is not None:
            reason = getattr(err, "reason", None) or type(err).__name__
            after = getattr(err, "retry_after_s", None)
            self.status = f"{reason} : retrying"
            self.wait_until = time.monotonic() + (after or 2.0)
        elif rep is not None and self.mode == DREAM:
            from .evalrun import chunks, resample
            try:
                left, right = self.dc.validate_chunk(self.env, chunks(rep), self.actions, verbose=False)
                self.queue.extend(zip(resample(left, self.rate), resample(right, self.rate)))
                self.status = f"dream : chunk of {len(left)}"
            except (ValueError, KeyError, TypeError) as e:
                self.status = f"bad chunk : {type(e).__name__} : {e}"
        if self.ask.busy:
            return              # sim frozen while the model thinks, like evalrun
        if not self.fed:        # refill the frame window holding the arm still
            if len(self.win["top"]) < self.dc.FRAMES:
                self._sim(*self.hold)
                self.status = f"filling frames {len(self.win['top'])}/{self.dc.FRAMES}"
                return
            self._feed(0)
            self.fed = True
        if self.queue:
            left, right = self.queue.popleft()
            self.hold = (np.asarray(left, float), np.asarray(right, float))
            self._sim(*self.hold)
        elif time.monotonic() >= self.wait_until:
            self.ask.start(self.pol)
            self.status = "dream : thinking"


def draw(img, to: Takeover, tick):
    import cv2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img is not None else np.zeros((360, 640, 3), np.uint8)
    col = (0, 200, 255) if to.mode == DREAM else (0, 255, 0)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 4)
    cv2.putText(img, to.mode, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    sub = to.status if to.mode == DREAM else (tick.mode if tick else "")
    if to.rec is not None and to.rec.ep is not None:
        sub += f"  REC {to.rec.ep.dur:.1f} s"
    cv2.putText(img, sub[:70], (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    hint = "fist 0.5 s / h = take over" if to.mode == DREAM else "open palm / g = give back"
    if to.fist_t > 0 and to.mode == DREAM:
        hint += f"  [{'#' * int(to.fist_t / FIST_S * 10):10s}]"
    cv2.putText(img, hint, (8, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return img


def main(argv=None) -> int:
    import cv2

    from .evalrun import dc, model, ours
    ap = argparse.ArgumentParser(description="Dream-1 drives, the human takes over and gives back")
    ap.add_argument("--model", type=model, default=dc.BETA_MODEL,
                    help="beta, ft (our fine-tune, needs the key of the account that trained it) or a checkpoint id")
    ap.add_argument("--task", default="pick up the apple")
    ap.add_argument("--mouse", action="store_true", help="mouse instead of the webcam (right button = fist)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="no network: scripted stand-in policy")
    ap.add_argument("--record", metavar="DIR", help="save human segments as LeRobot episodes in DIR")
    ap.add_argument("--actions", type=int, default=48, help="rows executed per answer, 1-48")
    ap.add_argument("--start", choices=(DREAM, HUMAN), default=DREAM)
    a = ap.parse_args(argv)

    WIN = "DreamHands take-over"
    runner = TeleopRunner()
    from .scene import scatter
    scatter(runner.sim)
    if ours(a.model):  # our fine-tune learned from demos that start at the teleop ready pose
        runner.go_ready()
    cv2.namedWindow(WIN)
    if a.mouse:
        from .fake_tracker import MouseTracker
        tracker = MouseTracker(WIN)
    else:
        from .a_tracker import ATracker
        tracker = ATracker(a.camera)
    poses = a.model in dc.POSE_MODELS
    if a.dry_run:
        pol = DryPolicy(runner.sim)
    else:
        key = dc.settings().get("DREAM_API_KEY", "")
        if not key:
            ap.error("no DREAM_API_KEY (env or ~/.dreamlabs/credentials): set it or use --dry-run")
        pol = dc.open_policy(dc.policy_config(a.model, None, poses), key)
    rec = None
    if a.record:
        from .recorder import Recorder
        from .save import Saver
        rec = Recorder(a.task, on_done=Saver(a.record))
        runner.warm_images()
    to = Takeover(runner, pol, a.task, a.model, poses, a.actions, rec, a.start)
    print(f"{a.model} | {a.task} | start {a.start}" + (" | dry-run" if a.dry_run else ""))
    frame, nxt, last_mode = None, time.perf_counter(), to.mode
    try:
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            keys = ""
            if k == ord(" ") and to.mode == HUMAN:
                if runner.teleop.mode == DISARMED:
                    runner.go_ready()
                else:
                    runner.teleop.disarm()
                    runner.engage = False
            elif k == ord("f") and to.mode == HUMAN:
                runner.engage = not runner.engage
            elif k == ord("p"):
                runner.teleop.set_precision(not runner.teleop.precision)
            elif k != 255:
                keys = chr(k)
            tick = to.step(tracker.read(), keys)
            if to.mode != last_mode:
                print(f"-> {to.mode}")
                last_mode = to.mode
            if to.mode == HUMAN:
                if runner.record_images and runner.last_images is not None:
                    frame = runner.last_images["top"]
                elif tick is not None and tick.step % 3 == 0:
                    rr = runner.sim.render(("top",))
                    frame = rr["top"] if rr else frame
            else:
                frame = to.frame if to.frame is not None else frame
            cv2.imshow(WIN, draw(frame, to, tick))
            nxt += DT
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()
    finally:
        if rec is not None and rec.ep is not None:
            runner.flush()
            rec.stop()
        runner.close()
        try:
            pol.close()
        except Exception as e:
            print(f"session release failed, it will expire on the server : {e}")
        runner.sim.close()
        if hasattr(tracker, "close"):
            tracker.close()
        cv2.destroyAllWindows()
    print(f"switches : {to.switches}" + (f" | human segments kept : {rec.on_done.w.n}" if rec else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
