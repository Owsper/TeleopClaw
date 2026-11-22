"""D's demo window: one OpenCV window, two panes + a status bar.

    | sim top camera + HUD (640x360) | webcam with the hand skeleton (scaled to 360 high) |
    | TRACKING OK  ENGAGED OK  REC *  pinch 0.42  latency 58 ms  frames 487  kept 12/15     |

The sim pane stays at (0, 0) in its original size, so the mouse stand-in, which reads window
coordinates, keeps working. Without a webcam frame (mouse mode) the right pane is left out.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

BAR_H = 34
GREEN, RED, GREY, WHITE = (80, 220, 100), (60, 60, 230), (150, 150, 150), (235, 235, 235)


def status(tick, runner, rec=None, now: float | None = None) -> list[tuple[str, tuple]]:
    """Bar items as (text, BGR colour), in the order of the spec."""
    now = time.monotonic() if now is None else now
    hand = tick.hand
    if tick.left_hand is None:
        tracking = hand.present and not tick.flags.get("tracking_lost")
        engaged = tick.mode == "ENGAGED"
        items = [("TRACKING " + ("OK" if tracking else "--"), GREEN if tracking else RED),
                 ("ENGAGED " + ("OK" if engaged else "--"), GREEN if engaged else GREY)]
        pinch_text = f"pinch {hand.pinch:.2f}"
    else:
        items = []
        for side, sample, mode in (("left", tick.left_hand, runner.left_arm.teleop.mode),
                                   ("right", hand, tick.mode)):
            tracked = sample.present and not tick.flags[side].get("tracking_lost")
            text = f"{side.upper()} {mode}  pinch {sample.pinch:.2f}" if tracked else f"{side.upper()} --"
            color = GREEN if tracked and mode == "ENGAGED" else GREY
            items.append((text, color))
        # Both detections use the same capture timestamp.
        hand = hand if hand.present else tick.left_hand
        pinch_text = None
    if rec is not None:
        items.append(("REC" if rec.ep is not None else "rec", RED if rec.ep is not None else GREY))
    if pinch_text:
        items.append((pinch_text, WHITE))
    if hand.present and hand.t:
        items.append((f"latency {max(0.0, now - hand.t) * 1000:.0f} ms", WHITE))
    if rec is not None:
        if rec.ep is not None:
            items.append((f"frames {rec.ep.n}", WHITE))
        saver = rec.on_done
        kept = getattr(getattr(saver, "w", None), "n", None)
        if kept is not None:
            items.append((f"kept {kept}/{rec.done}", WHITE))
    return items


def _bar(width: int, items) -> np.ndarray:
    bar = np.full((BAR_H, width, 3), 20, np.uint8)
    x = 10
    for text, col in items:
        if text == "REC":  # the red dot of the spec
            cv2.circle(bar, (x + 6, BAR_H // 2), 6, RED, -1, cv2.LINE_AA)
            x += 16
        cv2.putText(bar, text, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 18
    return bar


def compose(sim_bgr: np.ndarray, cam_bgr: np.ndarray | None = None, items=()) -> np.ndarray:
    """Sim pane left at its own size, webcam right scaled to the same height, bar below."""
    h = sim_bgr.shape[0]
    panes = [sim_bgr]
    if cam_bgr is not None:
        w = round(cam_bgr.shape[1] * h / cam_bgr.shape[0])
        panes.append(cv2.resize(cam_bgr, (w, h), interpolation=cv2.INTER_AREA))
    top = np.hstack(panes)
    return np.vstack([top, _bar(top.shape[1], items)])
