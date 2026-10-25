"""Pure-data stick container (no pygame import, so headless tools and tests can use it)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sticks:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0

    def magnitude(self) -> float:
        return abs(self.roll) + abs(self.pitch) + abs(self.yaw)
