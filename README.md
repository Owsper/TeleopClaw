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
python run_teleop.py                       # webcam via A's tracker (download hand_landmarker.task first)
python run_teleop.py --mouse               # mouse stands in for the hand tracker
python run_teleop.py --mouse --trials 10   # apple go/no-go, results -> trials.json
```

| Input | Action |
| --- | --- |
| mouse move / wheel | hand x-y / depth |
| left button (pinch) | close gripper |
| right button (fist) | clutch: freeze, move the hand, let go, no jump |
| `space` | arm (go to the top-down ready pose) / disarm |
| `f` | engage: the arm follows the hand |
| `p` | precision mode (x0.3 motion), use it near the apple |
| `t` | take-over: continue from the measured joints |
| `r` / `n` / `x` | record / next / discard events for the recorder (in trials `n` = give up) |
| `q` / `Esc` | quit |

The HUD shows the mode, TCP position, step time and warnings (`IK_REJECT`, `WORKSPACE_EDGE`,
`TRACKING_LOST`).

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
- Palm size is a weak depth proxy: a tilted hand also appears smaller. The controller should use rate limits, workspace clamps and W/S fallback.
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
