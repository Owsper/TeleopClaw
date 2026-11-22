"""Run MediaPipe VIDEO-mode hand tracking on the most recent webcam frame."""
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from camera import Camera
from common.types import HandState
from dreamhands.pose_assist import match_forearm
from gestures import Gestures


def hand_states(result, filters, width, height, t, both_hands):
    """Build independent gesture histories for the mirrored Tasks model output."""
    assigned = {}
    for i, landmarks in enumerate(result.hand_landmarks):
        side = "right"  # legacy single-hand mode routes either hand to the right arm
        if both_hands:
            labels = result.handedness[i] if i < len(result.handedness) else []
            if not labels or labels[0].score < 0.75:
                continue
            # Camera flips the image before inference. Tested with this model:
            # a mirrored anatomical right hand has the Tasks category "Left".
            side = {"Left": "right", "Right": "left"}.get(labels[0].category_name)
            if side is None:
                continue
        # Two detections claiming one side are ambiguous. Hold that arm.
        assigned[side] = None if side in assigned else landmarks
    states = {}
    for side, gestures in filters.items():
        landmarks = assigned.get(side)
        if not landmarks:
            gestures.reset()
            states[side] = HandState.invalid(t)
            continue
        px = np.array([(p.x * width, p.y * height) for p in landmarks], dtype=np.float32)
        depth_px = np.array([p.z * width for p in landmarks], dtype=np.float32)
        states[side] = gestures.update(px, t, depth_px)
    return states


class HandTracker:
    def __init__(self, camera_index: int = 0, model_path: str = "hand_landmarker.task",
                 pose_model: str | None = None, both_hands: bool = False):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Missing {model_path}; download hand_landmarker.task (see README)."
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2 if both_hands else 1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.pose_detector = None
        try:
            if pose_model is not None:
                self.pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(
                    mp.tasks.vision.PoseLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model),
                        running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    )
                )
            self.camera = Camera(index=camera_index)
        except Exception:
            self._close_detectors()
            raise
        self.both_hands = both_hands
        self.gestures = {side: Gestures() for side in (("right", "left") if both_hands else ("right",))}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.scan_pose = threading.Event()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._states = {side: HandState.invalid() for side in self.gestures}
        self._processed_frame = None
        self._forearm = None
        self._pose_time = 0.0
        self._process_fps = 0.0
        self.error = None
        self.pose_error = None

    def start(self):
        self.camera.start()
        self._worker.start()
        return self

    def _process_loop(self):
        last_seq, last_ms, last_finished = -1, -1, time.monotonic()
        last_pose = 0.0
        try:
            while not self._stop.is_set():
                seq, t, frame = self.camera.latest()
                if seq == last_seq or frame is None:
                    time.sleep(0.003)
                    continue
                last_seq = seq
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                # VIDEO mode requires strictly increasing timestamps.
                timestamp_ms = max(int(t * 1000), last_ms + 1)
                last_ms = timestamp_ms
                result = self.detector.detect_for_video(image, timestamp_ms)
                h, w = frame.shape[:2]
                states = hand_states(result, self.gestures, w, h, t, self.both_hands)
                state = states["right"]

                now = time.monotonic()
                fps = 1 / max(now - last_finished, 1e-4)
                last_finished = now
                with self._lock:
                    self._states = states
                    self._processed_frame = frame
                    self._process_fps = fps

                # Publish the hand immediately. Run the optional detector only
                # while waiting for a cue, at most about 12 times per second.
                if (self.pose_detector is not None and self.pose_error is None
                        and self.scan_pose.is_set() and state.valid and t - last_pose >= 0.08):
                    last_pose = t
                    try:
                        pose = self.pose_detector.detect_for_video(image, timestamp_ms)
                        forearm = match_forearm(pose, state.px[0], w, h)
                    except Exception as exc:
                        self.pose_error = exc
                    else:
                        with self._lock:
                            self._forearm, self._pose_time = forearm, t
        except Exception as exc:
            self.error = exc
        finally:
            self._close_detectors()

    def _close_detectors(self):
        self.detector.close()
        if self.pose_detector is not None:
            self.pose_detector.close()

    def latest(self) -> HandState:
        """Non-blocking hand state; mark stale frames invalid after 120ms."""
        with self._lock:
            state = self._states["right"]
        if time.monotonic() - state.t > 0.12:
            return HandState.invalid(time.monotonic())
        return state

    def latest_frame(self):
        with self._lock:
            frame = (self._processed_frame.copy()
                     if self._processed_frame is not None else None)
            fps = self._process_fps
        return frame, fps

    def preview(self, both=False):
        """Return a drawable frame, its hand, and the latest forearm sample."""
        with self._lock:
            frame = self._processed_frame.copy() if self._processed_frame is not None else None
            states, fps, forearm = self._states, self._process_fps, self._forearm
            pose_time = self._pose_time
        now = time.monotonic()
        states = {side: hand if now - hand.t <= 0.12 else HandState.invalid(hand.t)
                  for side, hand in states.items()}
        state = states["right"]
        if not state.valid or now - pose_time > 0.18:
            forearm = None
        return frame, states if both else state, fps, forearm, pose_time

    def close(self):
        self._stop.set()
        self._worker.join(timeout=2.0)
        self.camera.close()
