"""Run: python demo.py. Press Q to quit."""
import argparse
import time

import cv2

from hand_tracker import HandTracker

# MediaPipe 21-point skeleton (0 wrist, 4 thumb tip, 8 index tip, etc.)
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def draw(frame, state, fps):
    if state.valid:
        pts = [(int(x), int(y)) for x, y in state.px]
        for i, j in CONNECTIONS:
            cv2.line(frame, pts[i], pts[j], (0, 220, 255), 2)
        for point in pts:
            cv2.circle(frame, point, 4, (50, 255, 80), -1)
        cv2.circle(frame, pts[4], 7, (255, 90, 255), 2)
        cv2.circle(frame, pts[8], 7, (255, 90, 255), 2)

    info = [
        f"TRACKING: {'YES' if state.valid else 'NO'}   FPS: {fps:.1f}",
        f"PINCH (0 closed, 1 open): {state.pinch:.2f}",
        f"FIST/CLUTCH: {state.fist}   STOP: {state.stop}",
        f"PALM: {state.palm:.0f}px   FRAME AGE: {max(0, (time.monotonic()-state.t)*1000):.0f}ms",
        "Make a fist / pinch / hold open palm still 1s. Q = quit.",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 125), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    for i, line in enumerate(info):
        cv2.putText(frame, line, (12, 22 + i * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="hand_landmarker.task")
    args = parser.parse_args()

    tracker = HandTracker(camera_index=args.camera, model_path=args.model)
    try:
        tracker.start()
        while True:
            if tracker.error is not None:
                raise RuntimeError("Tracking worker failed") from tracker.error
            frame, fps = tracker.latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            state = tracker.latest()
            cv2.imshow("TeleopClaw", draw(frame, state, fps))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
