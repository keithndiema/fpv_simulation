"""Short-lived visual effects in world space: pop bursts, crash debris and floating score text."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Particle:
    pos: np.ndarray
    vel: np.ndarray
    ttl: float
    life: float
    color: tuple
    size: float = 3.0


@dataclass
class Floater:
    pos: np.ndarray
    text: str
    ttl: float
    life: float
    color: tuple


class Effects:
    GRAVITY = 6.0

    def __init__(self, rng: random.Random | None = None):
        self.particles: list[Particle] = []
        self.floaters: list[Floater] = []
        self.rng = rng or random.Random(1)

    def clear(self) -> None:
        self.particles.clear()
        self.floaters.clear()

    def burst(self, pos, color, n: int = 36, speed: float = 7.0, ttl: float = 0.9, size: float = 3.0) -> None:
        r = self.rng
        for _ in range(n):
            v = np.array([r.gauss(0, 1), r.gauss(0, 1), r.gauss(0, 1)])
            v = v / (np.linalg.norm(v) + 1e-9) * speed * r.uniform(0.3, 1.0)
            t = ttl * r.uniform(0.6, 1.0)
            self.particles.append(Particle(np.array(pos, float), v, t, t, color, size * r.uniform(0.6, 1.2)))

    def floater(self, pos, text: str, color, ttl: float = 1.4) -> None:
        self.floaters.append(Floater(np.array(pos, float) + np.array([0, 0, 1.0]), text, ttl, ttl, color))

    def update(self, dt: float) -> None:
        for p in self.particles:
            p.ttl -= dt
            p.vel[2] -= self.GRAVITY * dt
            p.vel *= max(0.0, 1.0 - 1.2 * dt)
            p.pos += p.vel * dt
        self.particles = [p for p in self.particles if p.ttl > 0]
        for f in self.floaters:
            f.ttl -= dt
            f.pos[2] += 1.6 * dt
        self.floaters = [f for f in self.floaters if f.ttl > 0]
