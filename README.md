# TeleopClaw (DreamHands)

Webcam hand teleoperation for the two YAM arms in Dream Labs' MuJoCo simulator. Move your
hands in 3D, pinch to close the claws, record demonstrations, fine-tune Dream-1 on them, and
hand control back and forth between the policy and a human.

Built for the Dream Labs simulation track at HackLab #2 Berlin.

[Watch the demo](demo_video/dreamhands_demo.mp4) · [Pitch deck](docs/TELEOP.pptx)

## What it does

- **Two-hand teleop** — each hand drives its matching YAM arm; use either hand alone or both.
- **20 Hz control** — smoothed, rate-limited hand motion becomes joint targets through
  damped-least-squares IK, with a workspace clamp and no-jump protection.
- **Grasp feedback** — a top-view inset shows the claws and fruit, and the HUD reports when
  both fingertips touch an object.
- **Recording** — capture episodes, run a quality gate, and write a LeRobot v2.1 dataset from
  three camera views.
- **Fine-tuning** — a CLI drives the Dream Labs FaaS chain: validate, upload, process, fine-tune.
- **Evaluation and take-over** — score a checkpoint headlessly, or let a human grab control
  from Dream-1 mid-episode and hand it back.

## How it works

```text
webcam ──> MediaPipe hands ──> gestures ──> HandState
                                                │
                                                ▼
                          teleop mapping (clutch, smoothing, workspace)
                                                │
                                                ▼
                                IK ──> MuJoCo YAM sim (20 Hz)
                                                │
                                                ▼
                          Tick ──> recorder ──> quality gate
                                                │
                                                ▼
                                   LeRobot dataset ──> Dream-1 fine-tune
```

The webcam tracker and the sim run on independent threads. The tracker publishes the newest
hand state without blocking control, and the runner emits a `Tick` every 50 ms. Dataset
cameras are only rendered while recording.

## Setup

Python 3.11–3.13 on Windows, macOS or Linux. A webcam is needed only for hand control. A Dream
Labs API key is needed only for fine-tuning, evaluation and take-over.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` brings in `dreamlabs==1.0.42` (the YAM sim), OpenCV, NumPy, MediaPipe,
pytest and matplotlib. The LeRobot writer also imports `av` and `pyarrow`; install them if
your environment does not already provide them:

```bash
pip install av pyarrow
```

Smoke tests:

```bash
python -m dreamhands.sim      # step the sim, render the top camera
python -m pytest tests -q     # sim/render tests need dreamlabs + a GL context
```

Headless Linux picks EGL automatically (`sudo apt install libegl1 libgl1`). Without any GL,
the sim still runs physics only and images come back as `None`.

`hand_landmarker.task` is included in the repo. Pose assist is optional and needs one extra
model:

```bash
curl -fL -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

Dream keys are read from `DREAM_API_KEY` or `~/.dreamlabs/credentials` and are never printed,
logged or committed. The teleoperation loop itself makes no Dream API calls.

## Quick start

```bash
python demo.py          # check hand tracking on its own
python run_teleop.py    # drive both arms
```

By default both arms are active: your left hand controls the left arm, your right hand the
right arm. Hold an open palm still for about one second to start that arm and keep it visible
while the arm moves to its ready pose.

- Move your wrist left/right and up/down.
- Move toward or away from the camera for depth.
- Pinch thumb and index to close that claw; keep pinching to hold, spread to release.
- Make a fist to clutch: the arm pauses, you reposition your hand, and releasing resumes
  without a jump.
- If one hand leaves the view, only its arm holds; the other keeps following.

Depth comes from apparent palm size, so it is a single-camera estimate. Use the top-view cross
to line up a grasp and press `P` for precision mode near an object. The HUD shows
`IK_REJECT`, `WORKSPACE_EDGE` and `TRACKING_LOST` warnings.

Shared keys:

| Key | Action |
| --- | --- |
| `Space` | ready the arm(s) / disarm |
| `F` | start following, or pause/resume while armed |
| `G` | move the right claw above the apple |
| `P` | precision mode (0.3× motion) |
| `T` | take over from measured joints (no jump) |
| `R` / `N` / `X` | recorder: start / end and keep / discard |
| `Q` / `Esc` | quit |

Input modes:

| Command | What it does |
| --- | --- |
| `--mouse` | Mouse stand-in: move = x/y, wheel = depth, left button = pinch, right = fist |
| `--hand-3d` | Previous all-hand mapping with the open-palm stop gesture |
| `--gesture-planes` | V sign switches AIM (tabletop cross) ↔ HEIGHT (vertical motion) |
| `--keyboard-axes` | Hand for sideways; `W`/`S` depth, `J`/`K` height, 5 mm per key |
| `--pose-assist` | Press `A`, hold a level forearm 0.5 s, and approach the apple |
| `--trials 10` | Apple go/no-go; `n` gives up, results go to `trials.json` |
| `--record DIR` | Save KEEP episodes as a LeRobot dataset |
| `--split` | Sim and webcam in two windows |

`--mouse`, `--hand-3d`, `--gesture-planes` and `--keyboard-axes` drive the right arm.
`python run_teleop.py --mouse --trials 10` exercises the full control path without a webcam.

## Recording demonstrations

```bash
python run_teleop.py --record data/dreamhands
```

`R` starts an episode, `N` ends and keeps it, `X` discards it. The apple referee can also end
an episode automatically once the apple is lifted and held. A quality gate prints `KEEP` or
`DROP` and only `KEEP` episodes are written. The scene resets with a random apple position
unless `--fixed` is used.

Dataset details:

- LeRobot v2.1 at 10 fps, from a 20 Hz control loop
- Cameras: `top` → `observation.images.overhead`, `left` → `wrist_l`, `right` → `wrist_r`
  (360×640 RGB)
- 14-D state and action, right arm first:
  `right_j1..j6, right_gripper, left_j1..j6, left_gripper` (1 = open, 0 = closed)
- Quality gate: at least 1.6 s, tracking loss under 5 %, IK rejects under 3 %, no joint jump
  above 0.25 rad

`data/` is gitignored, so recordings stay local. Episodes can also be generated without a
person:

```bash
python -m dreamhands.scripted data/dreamhands -n 50   # scripted IK picks
python -m dreamhands.bot data/dreamhands -n 40        # simulated hand through the real teleop code
```

`bot.py` uses the actual teleop mapping, but its hand is simulated, so these are not human
demonstrations.

## Fine-tuning Dream-1

State lives in `data/faas.json`, so every step can be re-run:

```bash
python -m dreamhands.faas models                 # list models; also checks the key
python -m dreamhands.faas check data/dreamhands  # local dataset check, no key needed
python -m dreamhands.faas run data/dreamhands    # check → embodiment → upload → fine-tune
python -m dreamhands.faas status --watch         # follow the current job
python -m dreamhands.faas jobs                   # all fine-tune jobs
```

`run` defaults to 500 steps (500–10000 in hundreds), uploads under the `yam-sim` embodiment
and polls until a fine-tuned model id is returned. `--no-wait` queues the job and exits.

This repo's fine-tune is `dream-1-ft:0c2119f3-dreamhands`, exposed as `--model ft`. It only
runs with the API key of the Dream account that trained it. Fine-tunes answer at about
30 rows/s; the tools resample that to the 20 Hz sim. Stock models play one row per step.

## Evaluation

```bash
python -m dreamhands.evalrun --model ft   --tag ft   --episodes 10 --scatter --out data/eval.jsonl
python -m dreamhands.evalrun --model beta --tag base --episodes 10 --scatter --out data/eval.jsonl
python -m dreamhands.report data/eval.jsonl --out data/report
```

The default referee is the lift rule: success means the apple is raised at least 4 cm and held
for 1 s with the gripper closed. It fails on a 60 s timeout or if the apple leaves the table.
`--referee place` checks whether an object was released into the bowl or onto the plate.
`--dry-run` uses a scripted stand-in policy with no network. The report writes `report.md`
plus evaluation and quality charts; add `--quality data/dreamhands.quality.jsonl` to include
the recording gate log.

## Take-over

```bash
python -m dreamhands.takeover --mouse --dry-run               # no network
python -m dreamhands.takeover --model ft                      # webcam, needs a Dream key
python -m dreamhands.takeover --model ft --record data/takeover
```

Dream-1 drives the right arm. A fist held 0.5 s (or `H`) hands control to the human, starting
from the measured joints so the arm does not jump. An open palm held still (or `G`) gives
control back; the policy gets `reset_episode()` and a refilled frame window. With `--record`,
each human segment that passes the quality gate (minimum 1.6 s) is saved as its own episode,
giving DAgger-style corrections where the policy fails.

## Project layout

| Path | What it is |
| --- | --- |
| `run_teleop.py`, `demo.py` | Operator app and standalone tracker demo |
| `camera.py`, `hand_tracker.py`, `gestures.py`, `common/types.py` | Webcam, MediaPipe tracking, gestures, tracker contract |
| `dreamhands/sim.py`, `ik.py`, `teleop.py`, `runner.py` | Sim wrapper, IK, control mapping, 20 Hz runner |
| `dreamhands/a_tracker.py`, `fake_tracker.py` | Webcam and mouse adapters for the control `HandState` |
| `dreamhands/trials.py`, `referee.py`, `scene.py` | Lift and place referees, randomised starts |
| `dreamhands/episode.py`, `recorder.py`, `quality.py`, `save.py`, `lerobot.py` | Episode buffering, recording, QC, dataset writer |
| `dreamhands/scripted.py`, `bot.py` | Human-free demo generation |
| `dreamhands/faas.py`, `evalrun.py`, `takeover.py`, `report.py` | Fine-tuning, evaluation, take-over, reports |
| `dreamhands/pose_assist.py`, `ui.py`, `render_worker.py` | Forearm cue, demo window, background rendering |
| `tests/`, `.github/workflows/sim.yml` | Test suite and 3-OS CI (sim smoke + pytest) |

## Conventions and limits

- Each sim step is 50 ms (20 Hz), and the control loop must stay inside that budget.
- Workspace in metres: `x -0.35..0.15`, `y -0.10..0.15`, `z 0.78..0.88`. Ready TCP is
  `(-0.05, 0.05, 0.88)`.
- Hand coordinates: `x` 0..1 left to right on the mirrored preview, `y` 0..1 bottom to top,
  `z` larger = closer to the camera. Motion matches the top camera: right = right, up = up,
  toward the webcam = forward.
- Camera and vision do not guarantee robot safety. The controller holds on stale frames
  (older than 120 ms), clamps the workspace and rejects large joint jumps, but pinch is an
  estimate, not calibrated gripper force or width.
- Background rendering is enabled by default only on Windows; override with
  `DREAMHANDS_ASYNC_RENDER=1` or `0`.
- `--model ft` is tied to its training account; retrain with `python -m dreamhands.faas run`
  to get a model your own key can run.

## Team

HackLab #2 Berlin, Dream Labs simulation track, built by a three-person team covering hand
tracking, control/sim integration, and recording/fine-tuning.
