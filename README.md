# TeleopClaw

Single hand, mirrored 640x480 webcam, MediaPipe 21 landmarks, normalized pinch, debounced fist/clutch and stationary open-palm stop. **Robot control is deliberately not implemented here.**

## macOS quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
curl -L -o hand_landmarker.task 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
python demo.py
```

Allow camera access if macOS asks. Other OS: activate the venv through your platform-specific command.

For a different camera index: `python demo.py --camera 1`.

## Control integration

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

## What to test

- Hand open: 21 green dots, skeleton; `valid=True`.
- Thumb/index pinch: `pinch` approaches 0; separate them: approaches 1.
- Fist 5 processed frames: `fist=True`; open 3 processed frames: `fist=False`.
- Hold a **still** open palm ~1 second: `stop=True`; move or curl fingers: resets.
- Hide hand: `valid=False` (the controller MUST HOLD on invalid state).
- Compare `frame age` and `fps`; these are not hardware end-to-end latency measurements.

## Architecture and caveats

- Webcam capture and landmark inference have independent threads. The newest frame overwrites old ones, avoiding queue buildup.
- `VIDEO` mode processes captured frames with monotonic millisecond timestamps. Display runs in the main thread for macOS OpenCV compatibility.
- The frame is mirrored **before** landmark detection. MediaPipe handedness labels for mirrored inputs can be misleading, so this starter does not use them.
- Pinch values are estimates, not calibrated force/gripper width. Tune `0.25 / 0.95` in `gestures.py` for the operator.
- The all-fingers-curled fist rule sometimes fails when a fist is viewed edge-on; adjust gesture margin thresholds empirically.
- Palm size is a weak depth proxy: a tilted hand also appears smaller. The controller should use rate limits, workspace clamps and W/S fallback.
- Camera and vision alone do NOT guarantee robot safety. The controller must HOLD on invalid/stale frames and enforce actuator limits.
- `HandState.t` is monotonic capture time. The `latest()` method invalidates if sample age >0.12 sec.
