# TeleopClaw (DreamHands)

Webcam hand teleop for the official Dream Labs YAM MuJoCo sim: your hand drives the right
YAM arm, we record demonstrations, and the team fine-tunes Dream-1 on them. A take-over mode
lets a human grab control back from the policy mid-episode.

HackLab #2 Berlin, Dream Labs simulation track.

## Setup

Python 3.11+ (tested on 3.13), Windows / macOS / Linux.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m dreamhands.sim      # smoke test: steps the sim, renders the top camera
python -m pytest tests -q     # run after every change
```

Headless Linux picks EGL automatically (`apt install libegl1 libgl1`); without any GL the sim
still runs physics only. Dream API keys live in `~/.dreamlabs/credentials` per teammate and are
never printed, logged or committed. Nothing in the teleop loop calls the Dream API.

## Drive the arm

```bash
python run_teleop.py                       # direct 3D webcam control
python run_teleop.py --gesture-planes      # optional V sign to switch AIM/HEIGHT
python run_teleop.py --keyboard-axes       # optional W/S depth and J/K height
python run_teleop.py --hand-3d             # previous all-hand mapping, with open-palm stop
python run_teleop.py --mouse               # mouse stands in for the hand tracker
python run_teleop.py --mouse --trials 10   # apple go/no-go, results -> trials.json
```

| Input | Action |
| --- | --- |
| Still open palm for 1 second | start one-hand control |
| Move wrist left/right, up/down | move sideways, lower or lift |
| Move toward / away from the webcam | move forward / back on the table |
| Move hand across the webcam in AIM (with `--gesture-planes`) | move the white cross around the tabletop map |
| Hold a V sign briefly, then relax (with `--gesture-planes`) | switch between AIM and HEIGHT |
| Move hand down/up in HEIGHT (with `--gesture-planes`) | lower/lift with the table position locked |
| Touch thumb and index finger | close the claw; keep pinching to hold |
| `w` / `s` | with `--keyboard-axes`: move forward / back on the table, 5 mm per key event |
| `j` / `k` | with `--keyboard-axes`: lower / lift, 5 mm per key event |
| mouse move / wheel | hand x-y / depth |
| left button (pinch) | close gripper |
| right button (fist) | clutch: freeze, move the hand, let go, no jump |
| `space` | arm (go to the top-down ready pose) / disarm |
| `f` | ready the arm and start following, or toggle follow when already armed |
| `g` | move the claw above the apple directly, then press `f` for manual follow |
| `p` | precision mode (x0.3 motion), use it near the apple |
| `t` | take-over: continue from the measured joints |
| `r` / `n` / `x` | record / next / discard events for the recorder (in trials `n` = give up) |
| `q` / `Esc` | quit |

For a one-hand pick, center your hand and hold your open palm still for one second.
The wrist controls sideways motion and height. Move toward or away from the lens to
adjust depth. All three directions are active together. Use the top-view cross to
align with a fruit, lower your hand, then pinch. When the HUD reports contact on
both jaws, keep pinching and raise your hand. Spread thumb and index to release.
Depth is estimated from changes in apparent palm span relative to your starting pose.

With `--gesture-planes`, slide the white cross onto the fruit dot. Hold up your index and middle fingers
in a V with ring and little fingers curled and thumb apart. Once HEIGHT appears, relax
your fingers. Lower your hand, pinch, and raise your hand while keeping the pinch.
Spread thumb and index to release. Use the V sign again to return to AIM.

With `--keyboard-axes`, press `F` with your hand centered. Tap `J` six times to come down
3 cm from the starting height. Move sideways with your hand and line up the depth with
`W` or `S`. Tap `J` until the fingers straddle the fruit, pinch, and keep pinching
as you tap `K` to lift. The top-view inset shows the claw as a white cross and the
three fruits as colored dots. Align the cross and dot before lowering. The HUD names
the object when both fingertips touch it. A closed claw can also be empty.
A stationary open hand keeps follow on in this mode. Sideways hand movement carries
the fruit, and each `K` or `J` step holds the sideways target steady.

If your hand reaches the edge of the webcam view during a grasp, press `F` while following
to pause. The claw holds its position and grip while you move your hand back to the center.
Pinch again before pressing `F` to resume with the claw closed, then raise your hand (use HEIGHT with `--gesture-planes` or `K` with `--keyboard-axes`).
If the HUD already says `TRACKING_LOST`, keep pinching and bring your hand back into view.
Following resumes automatically when the camera sees it again.

The HUD shows the mode, TCP position, step time and warnings (`IK_REJECT`, `WORKSPACE_EDGE`,
`TRACKING_LOST`).

To try a forearm cue for moving above the apple, download the optional MediaPipe pose model
once and start the webcam app with pose assist enabled.

```bash
curl -fL -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
python run_teleop.py --pose-assist
```

Press `Space` to ready the arm, then `A` to request an approach. Keep your elbow, wrist, and
hand visible. Hold your forearm level for half a second with your fingers relaxed. The webcam
preview shows your forearm and hold progress. When the claw reaches the hover point, lower your hand, pinch to grab, then raise your hand (switch to HEIGHT with `--gesture-planes`, or use J/K with `--keyboard-axes`). Keep your hand visible during the move.
If the cue is hard to detect, press `G` to move above the apple directly and `F` to follow your
hand afterward. Press `F` or make a fist to cancel an approach.
Keep pinching while you lift. If you need to pause with the claw closed, press `F`.

## Hand tracker (A)

Single hand, mirrored 640x480 webcam, MediaPipe 21 landmarks, normalized pinch, debounced fist/clutch and stationary open-palm stop. **Robot control is deliberately not implemented here.**

### macOS quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
curl -L -o hand_landmarker.task 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
python demo.py
```

Allow camera access if macOS asks. Other OS: activate the venv through your platform-specific command.

For a different camera index: `python demo.py --camera 1`.

### Control integration

```python
from hand_tracker import HandTracker
tracker = HandTracker(camera_index=0).start()
try:
    while True:
        hand = tracker.latest()  # returns HandState, never waits for the webcam
        if not hand.valid:
            # HOLD last robot command: tracking is absent or stale (>120ms)
            pass
        else:
            # hand.px[0] = wrist, hand.px[9] = middle MCP
            # hand.palm = wrist-to-middle-MCP pixel distance
            # hand.pinch: 0 closed -> 1 open
            # hand.fist: debounced clutch
            # hand.stop: stationary open palm for >=1s
            pass
finally:
    tracker.close()
```

Use `FakeTracker().latest()` for a mock HandState without a webcam.

### What to test

- Hand open: 21 green dots, skeleton; `valid=True`.
- Thumb/index pinch: `pinch` approaches 0; separate them: approaches 1.
- Fist 5 processed frames: `fist=True`; open 3 processed frames: `fist=False`.
- Hold a **still** open palm ~1 second: `stop=True`; move or curl fingers: resets.
- Hide hand: `valid=False` (the controller MUST HOLD on invalid state).
- Compare `frame age` and `fps`; these are not hardware end-to-end latency measurements.

### Architecture and caveats

- Webcam capture and landmark inference have independent threads. The newest frame overwrites old ones, avoiding queue buildup.
- `VIDEO` mode processes captured frames with monotonic millisecond timestamps. Display runs in the main thread for macOS OpenCV compatibility.
- The frame is mirrored **before** landmark detection. MediaPipe handedness labels for mirrored inputs can be misleading, so this starter does not use them.
- Pinch values are estimates, not calibrated force/gripper width. Tune `0.25 / 0.95` in `gestures.py` for the operator.
- The all-fingers-curled fist rule sometimes fails when a fist is viewed edge-on; adjust gesture margin thresholds empirically.
- Webcam depth uses the apparent 3D span from wrist to knuckles. Relative landmark depth helps compensate when you tilt your palm during a sideways slide. The result is still a single-camera estimate.
- Camera and vision alone do NOT guarantee robot safety. The controller must HOLD on invalid/stale frames and enforce actuator limits.
- `HandState.t` is monotonic capture time. The `latest()` method invalidates if sample age >0.12 sec.

## Layout

| Path | What |
| --- | --- |
| `dreamhands/sim.py` | cross-platform wrapper around `dream_examples/yam_mujoco/yam_env.py` |
| `dreamhands/types.py` | shared `HandState` (tracker -> control) and `Tick` (control -> recorder). Draft: tell the team before changing fields |
| `dreamhands/ik.py` | damped-least-squares IK on `right_tcp`, gripper pointing down, free yaw |
| `dreamhands/teleop.py` | state machine DISARMED / READY / ENGAGED / HOLD, clutch-relative mapping, EMA, deadband, rate limit, workspace clamp |
| `dreamhands/runner.py` | `TeleopRunner`: 20 Hz tick -> IK -> sim -> `Tick` to `on_tick` listeners, `go_ready()`, `take_over()` |
| `dreamhands/trials.py` | apple referee + scripted pick baseline |
| `dreamhands/fake_tracker.py` | mouse tracker with the same `HandState` output |
| `dreamhands/a_tracker.py` | adapter from A's `common.types.HandState` (pixels, pinch 0 = closed, `stop`) to the control `HandState` |
| `run_teleop.py` | the operator app with HUD |

## Conventions

- Robot vectors are 14-D, right arm first: `right[0:6], right_grip[6], left[7:13], left_grip[13]`;
  gripper 1 = open, 0 = closed. Each sim step is 50 ms (20 Hz); the loop must stay within it.
- `HandState.pos`: x 0..1 left to right on the mirrored preview, y 0..1 bottom to top, z larger =
  closer to the camera. Hand motion maps to the view of the `top` camera: right = right, up = up,
  toward the webcam = forward.
- Workspace (m, world): x -0.35..0.15, y -0.10..0.15, z 0.78..0.88. Ready TCP (-0.05, 0.05, 0.88).
- `Tick.images` (all three cameras, 360x640) only when recording, on even steps: rendering all
  three takes ~120 ms, the top camera alone ~74 ms.

## Apple go/no-go

Referee: success = apple lifted >= 4 cm and held for 1 s with the gripper closed; fail = timeout
(60 s) or the apple drops off the table. A scripted pick through the same IK path succeeds 5/5 with
a perfect aim, about 4/5 at 5 mm aim error and 3/5 at 10-20 mm: aim with precision mode.

## Take-over (D)

Dream-1 drives the arm; a fist held 0.5 s (or `h`) hands control to the human with no jump
(the runner rebases on the measured joints), an open palm held still (or `g`) gives it back
(`reset_episode()` + a refilled frame window). With `--record` every human segment >= 1.6 s
is saved as its own episode: DAgger-style corrections exactly where the policy fails.

```bash
python -m dreamhands.takeover --mouse --dry-run                    # no network, scripted stand-in
python -m dreamhands.takeover --model <checkpoint id>              # webcam, needs a Dream key
python -m dreamhands.takeover --model <id> --record data/takeover  # keep the corrections
```

## Team

- A: hand tracker (webcam -> `HandState`)
- B: control + integration (sim, IK, teleop, runner)
- C: recording, dataset and Dream-1 fine-tune (owns all Dream API calls)

Work happens on `control` and is merged into `main` after each step.
