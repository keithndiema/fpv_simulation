"""Design tokens: palette + cached fonts so every widget shares one visual language."""
from __future__ import annotations

from functools import lru_cache

import pygame

# Synthwave-ish palette
SKY_TOP = (6, 8, 28)
SKY_MID = (38, 18, 80)
HORIZON = (214, 70, 140)
GROUND_FAR = (40, 18, 70)
GROUND_NEAR = (8, 10, 26)
GRID = (0, 200, 255)
GRID_MAJOR = (255, 60, 190)
FOG = (60, 24, 96)
OBSTACLE = (150, 110, 255)
GATE_NEXT = (0, 255, 220)
GATE_IDLE = (255, 150, 40)
GATE_DONE = (60, 200, 110)
OSD = (235, 255, 245)
OSD_DIM = (140, 170, 170)
ACCENT = (0, 230, 255)
WARN = (255, 190, 40)
DANGER = (255, 70, 90)
GOOD = (80, 255, 150)
PANEL = (8, 12, 28, 170)


def lerp(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("consolas,menlo,dejavusansmono,couriernew,monospace", size, bold=bold)


def text(surface, s: str, pos, size=18, color=OSD, bold=False, anchor="topleft", shadow=True):
    f = font(size, bold)
    img = f.render(s, True, color)
    rect = img.get_rect(**{anchor: pos})
    if shadow:
        sh = f.render(s, True, (0, 0, 0))
        surface.blit(sh, rect.move(1, 1))
    surface.blit(img, rect)
    return rect
