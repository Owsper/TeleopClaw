"""Recognize a level forearm on the same mirrored image used for hand control."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite


@dataclass(frozen=True)
class Forearm:
    side: int
    elbow: tuple[float, float]
    wrist: tuple[float, float]

    @property
    def level(self) -> bool:
        dx = abs(self.wrist[0] - self.elbow[0])
        dy = abs(self.wrist[1] - self.elbow[1])
        return degrees(atan2(dy, dx)) <= 25


def match_forearm(result, hand_wrist, width: int, height: int) -> Forearm | None:
    """Match the tracked hand to the nearer visible pose wrist in image coordinates."""
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    closest = 0.15 * min(width, height)
    arm = None
    for side, (elbow_id, wrist_id) in enumerate(((13, 15), (14, 16))):
        elbow, wrist = landmarks[elbow_id], landmarks[wrist_id]
        if any((p.visibility or 0) < 0.5 or (p.presence is not None and p.presence < 0.5)
               or not (0 <= p.x <= 1 and 0 <= p.y <= 1) for p in (elbow, wrist)):
            continue
        elbow_px = (elbow.x * width, elbow.y * height)
        wrist_px = (wrist.x * width, wrist.y * height)
        if hypot(elbow_px[0] - wrist_px[0], elbow_px[1] - wrist_px[1]) < 0.06 * width:
            continue  # a forearm pointing toward the lens can look level
        distance = hypot(hand_wrist[0] - wrist_px[0], hand_wrist[1] - wrist_px[1])
        if isfinite(distance) and distance < closest:
            closest = distance
            arm = Forearm(side, elbow_px, wrist_px)
    return arm


class LevelHold:
    """Count 0.5 seconds of fresh level samples from the same arm."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._since = self._last = None
        self._side = None

    def update(self, arm: Forearm | None, captured: float, now: float) -> float:
        if arm is None:
            if self._last is not None and now - self._last > 0.22:
                self.reset()
            return 0.0
        if not arm.level or now - captured > 0.18:
            self.reset()
            return 0.0
        if (self._last is None or arm.side != self._side
                or not 0 <= captured - self._last <= 0.22):
            self._since = captured
        self._last, self._side = captured, arm.side
        return min(1.0, (captured - self._since) / 0.5)
