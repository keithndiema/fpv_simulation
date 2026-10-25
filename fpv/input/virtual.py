"""On-screen control sticks driven by mouse or touch.

Layout mirrors a mode-2 radio: the LEFT stick is throttle (vertical, latches where you leave it) + yaw
(horizontal, springs back); the RIGHT stick is pitch (vertical) + roll (horizontal), both spring back. Each
stick tracks the pointer that grabbed it, so on a touch screen both thumbs can fly at once.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

from ..render import theme
from .sticks import Sticks


@dataclass
class VStick:
    name: str
    top: str                         # label above
    spring_y: bool                   # vertical axis returns to centre when released (False = latches)
    x: float = 0.0                   # -1..1  (right positive)
    y: float = 0.0                   # -1..1  (up positive)
    pid: object = None               # pointer currently holding the stick
    centre: tuple = (0, 0)
    radius: int = 70

    def hit(self, pos) -> bool:
        return math.hypot(pos[0] - self.centre[0], pos[1] - self.centre[1]) <= self.radius * 1.25

    def drag_to(self, pos) -> None:
        dx = (pos[0] - self.centre[0]) / self.radius
        dy = -(pos[1] - self.centre[1]) / self.radius
        m = math.hypot(dx, dy)
        if m > 1.0:
            dx, dy = dx / m, dy / m
        self.x, self.y = dx, dy


class VirtualControls:
    SIZE = 150

    def __init__(self):
        self.left = VStick("left", "THR / YAW", spring_y=False)
        self.right = VStick("right", "PITCH / ROLL", spring_y=True)
        self.enabled = True
        self._hover = None

    # ---- layout -------------------------------------------------------------------------------------
    def layout(self, size: tuple[int, int], bottom_reserved: int) -> None:
        w, h = size
        r = max(48, min(self.SIZE // 2, int(min(w, h) * 0.11)))
        cy = h - bottom_reserved - r - 22
        self.left.radius = self.right.radius = r
        self.left.centre = (r + 28, cy)
        self.right.centre = (w - r - 28, cy)

    # ---- events -------------------------------------------------------------------------------------
    @staticmethod
    def _pointer(e: pygame.event.Event, size):
        """(kind, id, pos) for mouse + touch events, or None. Touch-synthesised mouse events are ignored."""
        if e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            if getattr(e, "touch", False):
                return None
            if e.type == pygame.MOUSEBUTTONDOWN and e.button != 1:
                return None
            kind = "down" if e.type == pygame.MOUSEBUTTONDOWN else "up" if e.type == pygame.MOUSEBUTTONUP else "move"
            if kind == "move" and not e.buttons[0]:
                return None
            return kind, "mouse", e.pos
        fing = {getattr(pygame, "FINGERDOWN", -1): "down", getattr(pygame, "FINGERUP", -2): "up",
                getattr(pygame, "FINGERMOTION", -3): "move"}
        if e.type in fing:
            return fing[e.type], ("finger", e.finger_id), (e.x * size[0], e.y * size[1])
        return None

    def handle_event(self, e: pygame.event.Event, size) -> bool:
        """Returns True when the event was consumed by a stick."""
        if not self.enabled:
            return False
        ptr = self._pointer(e, size)
        if ptr is None:
            return False
        kind, pid, pos = ptr
        sticks = (self.left, self.right)
        if kind == "down":
            for s in sticks:
                if s.pid is None and s.hit(pos):
                    s.pid = pid
                    s.drag_to(pos)
                    return True
            return False
        mine = [s for s in sticks if s.pid == pid]
        if not mine:
            return False
        s = mine[0]
        if kind == "move":
            s.drag_to(pos)
        else:                                       # released
            s.pid = None
            s.x = 0.0
            if s.spring_y:
                s.y = 0.0
        return True

    # ---- outputs ------------------------------------------------------------------------------------
    @property
    def holding(self) -> bool:
        return self.left.pid is not None or self.right.pid is not None

    def apply(self, base: Sticks) -> Sticks:
        """Overlay any held stick on `base` (keyboard / pad). Latching throttle is written to `base`."""
        out = Sticks(base.roll, base.pitch, base.yaw, base.throttle)
        if self.left.pid is not None:
            out.yaw = self.left.x
            out.throttle = max(0.0, min(1.0, (self.left.y + 1) / 2))
        if self.right.pid is not None:
            out.roll, out.pitch = self.right.x, self.right.y
        return out

    def sync(self, shown: Sticks) -> None:
        """Make idle knobs mirror whatever is actually flying the drone (keyboard, pad or autopilot)."""
        if self.left.pid is None:
            self.left.x, self.left.y = shown.yaw, shown.throttle * 2 - 1
        if self.right.pid is None:
            self.right.x, self.right.y = shown.roll, shown.pitch

    # ---- drawing ------------------------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, mouse=None) -> None:
        if not self.enabled:
            return
        mouse = mouse if mouse is not None else pygame.mouse.get_pos()
        for s in (self.left, self.right):
            cx, cy = s.centre
            r = s.radius
            held = s.pid is not None
            hover = s.hit(mouse)
            ring = pygame.Surface((r * 2 + 8, r * 2 + 8), pygame.SRCALPHA)
            c = (r + 4, r + 4)
            pygame.draw.circle(ring, (8, 14, 30, 150 if held else 105), c, r)
            pygame.draw.circle(ring, (*theme.ACCENT, 230 if (held or hover) else 130), c, r, 2)
            pygame.draw.circle(ring, (*theme.ACCENT, 60), c, int(r * 0.5), 1)
            pygame.draw.line(ring, (*theme.OSD_DIM, 110), (c[0] - r + 8, c[1]), (c[0] + r - 8, c[1]), 1)
            pygame.draw.line(ring, (*theme.OSD_DIM, 110), (c[0], c[1] - r + 8), (c[0], c[1] + r - 8), 1)
            surf.blit(ring, (cx - r - 4, cy - r - 4))
            kx, ky = cx + int(s.x * (r - 14)), cy - int(s.y * (r - 14))
            if not s.spring_y:                                  # throttle: show the latched level as a bar
                lvl = (s.y + 1) / 2
                pygame.draw.rect(surf, theme.lerp(theme.ACCENT, theme.WARN, lvl),
                                 (cx - r - 12, cy + r - int(2 * r * lvl), 5, int(2 * r * lvl)), border_radius=2)
            pygame.draw.circle(surf, (*theme.ACCENT,) if held else theme.OSD_DIM, (kx, ky), 16)
            pygame.draw.circle(surf, theme.OSD, (kx, ky), 16, 2)
            theme.text(surf, s.top, (cx, cy - r - 14), 12, theme.OSD_DIM, False, "midbottom", False)
