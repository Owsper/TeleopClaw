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

## Team

- A: hand tracker (webcam -> `HandState`)
- B: control + integration (sim, IK, teleop, runner)
- C: recording, dataset and Dream-1 fine-tune (owns all Dream API calls)

Work happens on `control` and is merged into `main` after each step.
