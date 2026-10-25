"""Reusable menu widget (keyboard/gamepad navigable) used by the title screen and pause overlay."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pygame

from . import theme
from .theme import text


@dataclass
class Item:
    label: str
    on_select: Callable[[], None] | None = None
    # choice / slider support
    get: Callable[[], object] | None = None
    set: Callable[[object], None] | None = None
    options: list | None = None            # discrete choices (displayed via str())
    rng: tuple[float, float, float] | None = None   # (min, max, step)
    fmt: str = "{:.2f}"

    def value_text(self) -> str:
        if self.get is None:
            return ""
        v = self.get()
        if self.options is not None:
            return str(self.options[int(v) % len(self.options)])
        return self.fmt.format(v)

    def adjust(self, direction: int) -> None:
        if self.get is None or self.set is None:
            return
        if self.options is not None:
            self.set((int(self.get()) + direction) % len(self.options))
        elif self.rng is not None:
            lo, hi, step = self.rng
            v = round(max(lo, min(hi, float(self.get()) + direction * step)) / step) * step
            self.set(max(lo, min(hi, v)))


@dataclass
class Menu:
    title: str
    items: list[Item]
    subtitle: str = ""
    index: int = 0
    footer: str = "UP/DOWN select    LEFT/RIGHT adjust    ENTER confirm"
    _rows: list = field(default_factory=list, repr=False)   # screen rects of the rows drawn last frame
    header_extra: int = 0              # extra pixels reserved under the subtitle for `decor`
    decor: Callable | None = None      # decor(surface, centre_x, top_y, panel_width) draws into that space

    def _row_at(self, pos) -> int | None:
        for i, r in enumerate(self._rows):
            if r.collidepoint(pos):
                return i
        return None

    def _handle_mouse(self, e: pygame.event.Event) -> bool:
        """Hover selects, click activates (left half of a slider row = decrease), wheel adjusts."""
        if e.type == pygame.MOUSEMOTION:
            i = self._row_at(e.pos)
            if i is not None:
                self.index = i
            return True
        if e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 4, 5):
            i = self._row_at(e.pos)
            if i is None:
                return True
            self.index = i
            it = self.items[i]
            if e.button in (4, 5):
                it.adjust(1 if e.button == 4 else -1)
            elif it.on_select:
                it.on_select()
            elif it.get is not None:
                it.adjust(-1 if e.pos[0] < self._rows[i].centerx else 1)
            return True
        return False

    def handle_event(self, e: pygame.event.Event) -> None:
        if self._handle_mouse(e):
            return
        key = None
        if e.type == pygame.KEYDOWN:
            key = e.key
        elif e.type == pygame.JOYHATMOTION:
            key = {(0, 1): pygame.K_UP, (0, -1): pygame.K_DOWN, (-1, 0): pygame.K_LEFT, (1, 0): pygame.K_RIGHT}.get(e.value)
        elif e.type == pygame.JOYBUTTONDOWN and e.button == 0:
            key = pygame.K_RETURN
        if key is None:
            return
        n = len(self.items)
        if key in (pygame.K_UP, pygame.K_w):
            self.index = (self.index - 1) % n
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.index = (self.index + 1) % n
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.items[self.index].adjust(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.items[self.index].adjust(1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            it = self.items[self.index]
            if it.on_select:
                it.on_select()
            elif it.options is not None:
                it.adjust(1)

    def draw(self, surf: pygame.Surface, t: float) -> None:
        w, h = surf.get_size()
        row = 38
        pw, ph = 620, 160 + self.header_extra + row * len(self.items)
        x, y = (w - pw) // 2, (h - ph) // 2
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(panel, (6, 8, 24, 205), panel.get_rect(), border_radius=16)
        pygame.draw.rect(panel, (*theme.ACCENT, 140), panel.get_rect(), 2, border_radius=16)
        surf.blit(panel, (x, y))
        text(surf, self.title, (w // 2, y + 22), 46, theme.ACCENT, True, "midtop")
        if self.subtitle:
            text(surf, self.subtitle, (w // 2, y + 76), 17, theme.OSD_DIM, False, "midtop")
        if self.decor:
            self.decor(surf, w // 2, y + 104, pw)
        self._rows = []
        for i, it in enumerate(self.items):
            ry = y + 112 + self.header_extra + i * row
            self._rows.append(pygame.Rect(x + 20, ry, pw - 40, row - 6))
            sel = i == self.index
            if sel:
                bar = pygame.Surface((pw - 40, row - 6), pygame.SRCALPHA)
                pulse = 0.5 + 0.5 * math.sin(t * 6)
                pygame.draw.rect(bar, (*theme.ACCENT, int(40 + 25 * pulse)), bar.get_rect(), border_radius=8)
                pygame.draw.rect(bar, (*theme.ACCENT, 200), bar.get_rect(), 1, border_radius=8)
                surf.blit(bar, (x + 20, ry))
            col = theme.OSD if sel else theme.OSD_DIM
            text(surf, it.label, (x + 40, ry + 6), 21, col, sel)
            if it.get is not None:
                v = it.value_text()
                text(surf, f"<  {v}  >" if sel else v, (x + pw - 40, ry + 6), 21, theme.WARN if sel else col,
                     sel, "topright")
        text(surf, self.footer, (w // 2, y + ph - 28), 14, theme.OSD_DIM, False, "midtop")



@dataclass
class Button:
    label: Callable[[], str]                      # evaluated every frame so labels can reflect state
    on_click: Callable[[], None]
    active: Callable[[], bool] = lambda: False    # highlighted while True (armed, autopilot on, ...)
    tip: str = ""                                 # key hint shown under the label


class ButtonBar:
    """Row of clickable buttons along the bottom edge. Single source of truth for layout, hit-testing
    and drawing, so what you see is always exactly what you can click."""

    HEIGHT = 44          # vertical space the bar reserves (other HUD elements stay above it)

    def __init__(self, buttons: list[Button], max_w: int = 112, h: int = 38, gap: int = 10):
        self.buttons, self.max_w, self.h, self.gap = buttons, max_w, h, gap

    def rects(self, size: tuple[int, int]) -> list[pygame.Rect]:
        w, hgt = size
        n = len(self.buttons)
        bw = max(40, min(self.max_w, (w - 24 - self.gap * (n - 1)) // n))   # shrink on narrow windows
        total = n * bw + (n - 1) * self.gap
        x0, y0 = (w - total) // 2, hgt - self.h - 10
        return [pygame.Rect(x0 + i * (bw + self.gap), y0, bw, self.h) for i in range(n)]

    def hit(self, pos, size) -> int | None:
        for i, r in enumerate(self.rects(size)):
            if r.collidepoint(pos):
                return i
        return None

    def handle_event(self, e: pygame.event.Event, size) -> bool:
        """Returns True if the click was consumed by a button."""
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            i = self.hit(e.pos, size)
            if i is not None:
                self.buttons[i].on_click()
                return True
        return False

    def draw(self, surf: pygame.Surface, mouse=None) -> None:
        mouse = mouse if mouse is not None else pygame.mouse.get_pos()
        f = theme.font(13, True)
        for btn, r in zip(self.buttons, self.rects(surf.get_size())):
            hover, on = r.collidepoint(mouse), btn.active()
            bg = pygame.Surface(r.size, pygame.SRCALPHA)
            fill = (*theme.ACCENT, 70) if on else (5, 16, 34, 235 if hover else 205)
            pygame.draw.rect(bg, fill, bg.get_rect(), border_radius=8)
            surf.blit(bg, r)
            pygame.draw.rect(surf, theme.ACCENT if (on or hover) else theme.OSD_DIM, r, 2 if hover else 1, border_radius=8)
            label = f.render(btn.label(), True, theme.OSD)
            surf.blit(label, label.get_rect(center=(r.centerx, r.centery - (5 if btn.tip else 0))))
            if btn.tip:
                tip = theme.font(10).render(btn.tip, True, theme.OSD_DIM)
                surf.blit(tip, tip.get_rect(center=(r.centerx, r.centery + 11)))


def draw_star(surf: pygame.Surface, centre, r: float, filled: bool, color=(255, 210, 60)) -> None:
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append((centre[0] + rr * math.cos(a), centre[1] + rr * math.sin(a)))
    if filled:
        pygame.draw.polygon(surf, color, pts)
        pygame.draw.polygon(surf, (255, 245, 190), pts, 2)
    else:
        pygame.draw.polygon(surf, (70, 80, 100), pts, 2)
