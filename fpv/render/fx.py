"""Analog-FPV video look: scanlines, subtle edge darkening, crash static and a red damage flash."""
from __future__ import annotations

import numpy as np
import pygame


class VideoFX:
    def __init__(self) -> None:
        self._scan: pygame.Surface | None = None
        self._noise: list[pygame.Surface] = []
        self._rng = np.random.default_rng(11)
        self.crash_t = 0.0                   # seconds since the crash (None-like when 0 and not crashed)
        self.active = False

    def _build_scanlines(self, size) -> pygame.Surface:
        w, h = size
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        for y in range(0, h, 3):
            pygame.draw.line(s, (0, 0, 0, 34), (0, y), (w, y))
        return s

    def _build_noise(self) -> None:
        for _ in range(4):
            a = (self._rng.random((90, 160)) * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(np.stack([a, a, a], axis=-1).transpose(1, 0, 2))
            self._noise.append(surf)

    def reset(self) -> None:
        self.crash_t, self.active = 0.0, False

    def update(self, dt: float, crashed: bool) -> None:
        if crashed:
            self.crash_t += dt
            self.active = True
        else:
            self.crash_t, self.active = 0.0, False

    def draw(self, surf: pygame.Surface, enabled: bool = True) -> None:
        w, h = surf.get_size()
        if enabled:
            if self._scan is None or self._scan.get_size() != (w, h):
                self._scan = self._build_scanlines((w, h))
            surf.blit(self._scan, (0, 0))
        if self.active:
            if not self._noise:
                self._build_noise()
            k = min(1.0, self.crash_t / 0.35)                    # static ramps up fast, then fades slowly
            fade = max(0.0, 1.0 - max(0.0, self.crash_t - 0.5) / 2.5)
            alpha = int(235 * k * fade)
            if alpha > 6:
                n = pygame.transform.scale(self._noise[int(self.crash_t * 24) % 4], (w, h))
                n.set_alpha(alpha)
                surf.blit(n, (0, 0))
                for _ in range(6):                               # horizontal tearing bars
                    y = int(self._rng.integers(0, h))
                    pygame.draw.rect(surf, (230, 230, 230), (0, y, w, int(self._rng.integers(2, 9))))
            red = pygame.Surface((w, h), pygame.SRCALPHA)
            red.fill((255, 30, 40, int(70 * max(0.0, 1.0 - self.crash_t / 1.2))))
            surf.blit(red, (0, 0))
