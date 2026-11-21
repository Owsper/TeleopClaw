"""Read the webcam continuously; do not let slow hand inference queue frames."""
import sys
import threading
import time

import cv2


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {index}. Check permissions and try camera index 1."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._seq = 0
        self._timestamp = 0.0
        self._frame = None
        self._worker = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self._worker.start()
        return self

    def _read_loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)  # mirror: intuitive on-screen motion
            t = time.monotonic()
            with self._lock:
                self._frame = frame
                self._timestamp = t
                self._seq += 1

    def latest(self):
        """Return (seq, capture_time, newest BGR frame or None)."""
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return self._seq, self._timestamp, frame

    def close(self):
        self._stop.set()
        self._worker.join(timeout=2.0)
        self.cap.release()
