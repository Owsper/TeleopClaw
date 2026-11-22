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


class HandTracker:
    def __init__(self, camera_index: int = 0, model_path: str = "hand_landmarker.task",
                 pose_model: str | None = None):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Missing {model_path}; download hand_landmarker.task (see README)."
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
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
        self.gestures = Gestures()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.scan_pose = threading.Event()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._state = HandState.invalid()
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
                if result.hand_landmarks:
                    h, w = frame.shape[:2]
                    px = np.array(
                        [[point.x * w, point.y * h]
                         for point in result.hand_landmarks[0]],
                        dtype=np.float32,
                    )
                    depth_px = np.array([point.z * w for point in result.hand_landmarks[0]],
                                        dtype=np.float32)
                    state = self.gestures.update(px, t, depth_px)
                else:
                    self.gestures.reset()
                    state = HandState.invalid(t)

                now = time.monotonic()
                fps = 1 / max(now - last_finished, 1e-4)
                last_finished = now
                with self._lock:
                    self._state = state
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
            state = self._state
        if time.monotonic() - state.t > 0.12:
            return HandState.invalid(time.monotonic())
        return state

    def latest_frame(self):
        with self._lock:
            frame = (self._processed_frame.copy()
                     if self._processed_frame is not None else None)
            fps = self._process_fps
        return frame, fps

    def preview(self):
        """Return a drawable frame, its hand, and the latest forearm sample."""
        with self._lock:
            frame = self._processed_frame.copy() if self._processed_frame is not None else None
            state, fps, forearm = self._state, self._process_fps, self._forearm
            pose_time = self._pose_time
        now = time.monotonic()
        if now - state.t > 0.12:
            state = HandState.invalid(state.t)
        if not state.valid or now - pose_time > 0.18:
            forearm = None
        return frame, state, fps, forearm, pose_time

    def close(self):
        self._stop.set()
        self._worker.join(timeout=2.0)
        self.camera.close()
