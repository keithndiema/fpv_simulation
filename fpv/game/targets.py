"""Hittable targets: floating bullseye balloons you pop by flying through them."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from ..events import EventBus
from .terrain import Terrain

TARGET_RADIUS = 1.7      # metres - generous so a clean line through the centre is not required
DRONE_RADIUS = 0.15


@dataclass
class Target:
    pos: np.ndarray                  # base position (the target bobs a little around it when drawn)
    ground_z: float                  # terrain height below it (for the tether line)
    radius: float = TARGET_RADIUS
    points: int = 100
    alive: bool = True
    phase: float = 0.0               # bob animation offset

    def centre(self, t: float) -> np.ndarray:
        return self.pos + np.array([0.0, 0.0, 0.25 * math.sin(t * 1.6 + self.phase)])


def segment_hits_sphere(p0: np.ndarray, p1: np.ndarray, c: np.ndarray, r: float) -> bool:
    """True if the segment p0->p1 passes within r of c (a swept test, so fast drones can't tunnel through)."""
    d = p1 - p0
    dd = float(np.dot(d, d))
    u = 0.0 if dd < 1e-12 else max(0.0, min(1.0, float(np.dot(c - p0, d)) / dd))
    return float(np.linalg.norm(p0 + d * u - c)) <= r


class TargetField:
    """Owns the targets of a mission, detects hits and publishes `target_hit` events."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.targets: list[Target] = []
        self.hits = 0
        self.combo = 0
        self.score = 0
        self._last_hit_t = -1e9
        self._prev: np.ndarray | None = None

    # ---- building ----------------------------------------------------------------------------
    def clear(self) -> None:
        self.targets, self.hits, self.combo, self.score = [], 0, 0, 0
        self._last_hit_t, self._prev = -1e9, None

    def place(self, x: float, y: float, height: float, terrain: Terrain | None, points: int = 100) -> Target:
        g = float(terrain.ground(x, y)) if terrain is not None else 0.0
        t = Target(np.array([x, y, g + height], float), g, points=points, phase=random.random() * 6.28)
        self.targets.append(t)
        return t

    def scatter(self, terrain: Terrain, count: int, centre, radius: float, seed: int,
                min_sep: float = 22.0, h_range=(3.0, 12.0), avoid=(), blocked=None) -> None:
        """Random targets over dry land, spread out so each one is a small flight rather than a hover."""
        rng = random.Random(seed)
        tries = 0
        while len(self.targets) < count and tries < 4000:
            tries += 1
            a, r = rng.uniform(0, 2 * math.pi), radius * math.sqrt(rng.uniform(0.08, 1.0))
            x, y = centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)
            if terrain.height(x, y) < 1.0 or (blocked is not None and blocked(x, y)):
                continue
            if any(math.hypot(x - t.pos[0], y - t.pos[1]) < min_sep for t in self.targets):
                continue
            if any(math.hypot(x - ax, y - ay) < 14 for ax, ay in avoid):
                continue
            self.place(x, y, rng.uniform(*h_range), terrain)

    # ---- queries ---------------------------------------------------------------------------
    def alive(self) -> list[Target]:
        return [t for t in self.targets if t.alive]

    def remaining(self) -> int:
        return sum(t.alive for t in self.targets)

    def nearest(self, pos) -> Target | None:
        live = self.alive()
        return min(live, key=lambda t: float(np.linalg.norm(t.pos - pos))) if live else None

    # ---- per frame -----------------------------------------------------------------------------
    def update(self, pos: np.ndarray, t_now: float) -> list[Target]:
        """Swept hit test between last frame's and this frame's position. Returns the targets popped."""
        popped: list[Target] = []
        if self._prev is None:
            self._prev = pos.copy()
            return popped
        for tg in self.targets:
            if tg.alive and segment_hits_sphere(self._prev, pos, tg.centre(t_now), tg.radius + DRONE_RADIUS):
                tg.alive = False
                self.combo = self.combo + 1 if t_now - self._last_hit_t < 8.0 else 1   # chain pops within 8 s
                self._last_hit_t = t_now
                gained = tg.points * min(self.combo, 5)
                self.hits += 1
                self.score += gained
                popped.append(tg)
                self.bus.publish("target_hit", target=tg, points=gained, combo=self.combo,
                                 remaining=self.remaining())
        self._prev = pos.copy()
        return popped

    def reset_tracking(self) -> None:
        self._prev = None
