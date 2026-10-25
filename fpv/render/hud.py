"""OSD + instrument widgets drawn on top of the scene (Betaflight-style readouts, sticks, motors, graph)."""
from __future__ import annotations

import math

import numpy as np
import pygame

from . import theme
from .theme import text
from .widgets import ButtonBar

BOTTOM = ButtonBar.HEIGHT + 14   # vertical space reserved at the bottom for the button bar


def fmt_time(t: float | None) -> str:
    if t is None:
        return "--:--.---"
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"


class Toasts:
    def __init__(self) -> None:
        self.items: list[list] = []

    def add(self, msg: str, color=theme.ACCENT, ttl: float = 2.2) -> None:
        self.items.append([msg, color, ttl, ttl])
        del self.items[:-4]

    def update(self, dt: float) -> None:
        for it in self.items:
            it[2] -= dt
        self.items = [i for i in self.items if i[2] > 0]

    def draw(self, surf) -> None:
        w, h = surf.get_size()
        for k, (msg, color, ttl, total) in enumerate(reversed(self.items)):
            a = min(1.0, ttl / 0.4, (total - ttl) / 0.15 + 0.2)
            c = theme.lerp((0, 0, 0), color, a)
            text(surf, msg, (w // 2, h * 0.30 + k * 34), 28, c, True, "midtop")


class Hud:
    def __init__(self) -> None:
        self.toasts = Toasts()
        self._vignette: pygame.Surface | None = None
        self._panel_cache: dict[tuple, pygame.Surface] = {}

    # ---- helpers -------------------------------------------------------
    def _panel(self, size, radius=10) -> pygame.Surface:
        s = self._panel_cache.get(size)
        if s is None:
            s = pygame.Surface(size, pygame.SRCALPHA)
            pygame.draw.rect(s, theme.PANEL, s.get_rect(), border_radius=radius)
            pygame.draw.rect(s, (*theme.ACCENT, 90), s.get_rect(), 1, border_radius=radius)
            self._panel_cache[size] = s
        return s

    def _make_vignette(self, size) -> pygame.Surface:
        w, h = size
        small = pygame.Surface((64, 36), pygame.SRCALPHA)
        for y in range(36):
            for x in range(64):
                d = math.hypot((x - 31.5) / 32, (y - 17.5) / 18)
                small.set_at((x, y), (0, 0, 0, int(max(0, d - 0.55) * 210)))
        return pygame.transform.smoothscale(small, (w, h))

    # ---- main OSD --------------------------------------------------------
    def draw_osd(self, surf, st, runner, settings, cam_name: str, input_name: str, recording: bool) -> None:
        w, h = surf.get_size()
        if self._vignette is None or self._vignette.get_size() != (w, h):
            self._vignette = self._make_vignette((w, h))
        surf.blit(self._vignette, (0, 0))

        low = st.voltage / 6 < 3.45
        # top-left: battery
        text(surf, f"{st.voltage:5.1f}V", (24, 20), 30, theme.DANGER if low else theme.OSD, True)
        text(surf, f"{st.mah:4.0f}mAh  {st.current:5.1f}A", (24, 56), 18, theme.OSD_DIM)
        bar_w = 140
        pygame.draw.rect(surf, (0, 0, 0), (24, 82, bar_w + 2, 10), 1)
        frac = st.battery_pct / 100
        pygame.draw.rect(surf, theme.lerp(theme.DANGER, theme.GOOD, frac * 1.4), (25, 83, int(bar_w * frac), 8))

        # top-centre: mode + status, then the mission objective card
        mode = "ANGLE" if st.mode else "ACRO"
        status, scol = ("CRASHED", theme.DANGER) if st.crashed else \
            (("ARMED", theme.GOOD) if st.armed else ("DISARMED", theme.WARN))
        text(surf, f"{mode}  |  {status}", (w // 2, 14), 22, scol, True, "midtop")
        text(surf, f"{cam_name} cam   {input_name}", (w // 2, 40), 13, theme.OSD_DIM, False, "midtop")
        self._objective_card(surf, runner, settings, w)
        self._place_line(surf, st, runner.course)

        # top-right: timer
        m = runner.m
        if runner.phase.value in ("active", "complete", "failed") or m.kind == "free":
            t = runner.time_left if runner.time_left is not None else runner.elapsed
            danger = runner.time_left is not None and runner.time_left < 15 and int(st.time * 3) % 2 == 0
            text(surf, fmt_time(t if runner.phase.value != "ready" else 0.0), (w - 24, 18), 30,
                 theme.DANGER if danger else theme.OSD, True, "topright")
            label = "TIME LEFT" if runner.time_left is not None else "TIME"
            text(surf, label, (w - 24, 52), 13, theme.OSD_DIM, False, "topright")
        else:
            text(surf, "--:--.---", (w - 24, 18), 30, theme.OSD_DIM, True, "topright")
            text(surf, "STARTS AT TAKE-OFF", (w - 24, 52), 13, theme.WARN, False, "topright")
        rec = settings.record(m.id)
        if rec:
            val = f"{rec['value']:.0f} pts" if m.kind == "attack" else fmt_time(rec["value"])
            text(surf, f"BEST {val}  " + "*" * rec["stars"], (w - 24, 72), 14, theme.OSD_DIM, False, "topright")
        if recording:
            pygame.draw.circle(surf, theme.DANGER, (w - 34, 98), 6)
            text(surf, "REC", (w - 48, 90), 16, theme.DANGER, True, "topright")

        # crosshair
        cx, cy = w // 2, h // 2
        for dx, dy, ex, ey in ((-14, 0, -6, 0), (6, 0, 14, 0), (0, -14, 0, -6), (0, 6, 0, 14)):
            pygame.draw.line(surf, theme.OSD, (cx + dx, cy + dy), (cx + ex, cy + ey), 2)

        # bottom-centre: speed / altitude / throttle
        alt = max(0.0, st.pos[2] - 0.04)
        by = h - BOTTOM                                   # baseline just above the button bar
        text(surf, f"{st.speed * 3.6:5.0f}", (w // 2 - 120, by - 72), 44, theme.OSD, True, "topright")
        text(surf, "km/h", (w // 2 - 112, by - 50), 16, theme.OSD_DIM)
        r = text(surf, f"{alt:5.1f}", (w // 2 + 120, by - 72), 44, theme.OSD, True, "topleft")
        text(surf, "m", (r.right + 6, by - 50), 16, theme.OSD_DIM)
        thr_w = 220
        x0 = w // 2 - thr_w // 2
        pygame.draw.rect(surf, (0, 0, 0), (x0 - 1, by - 18, thr_w + 2, 12), 1)
        pygame.draw.rect(surf, theme.lerp(theme.ACCENT, theme.WARN, st.throttle), (x0, by - 17, int(thr_w * st.throttle), 10))
        text(surf, f"THR {st.throttle * 100:3.0f}%", (w // 2, by - 2), 15, theme.OSD_DIM, False, "midtop")

        # artificial-horizon tick: roll indicator at top of crosshair
        roll = math.degrees(st.euler[0])
        text(surf, f"R{roll:+4.0f}  P{math.degrees(st.euler[1]):+4.0f}", (w // 2, cy + 70), 15, theme.OSD_DIM, False, "midtop")

        if low and int(st.time * 2) % 2 == 0:
            text(surf, "LOW BATTERY", (w // 2, h * 0.18), 30, theme.DANGER, True, "midtop")
        if st.crashed:
            text(surf, "CRASHED  -  press R to reset", (w // 2, h * 0.42), 34, theme.DANGER, True, "midtop")
        elif not st.armed and st.on_ground:
            text(surf, "ARM (SPACE or button)   then raise throttle (W or left stick)", (w // 2, h * 0.40), 20, theme.WARN, True, "midtop")
        elif st.armed and st.on_ground and runner.phase.value == "ready" and runner.m.kind != "free":
            text(surf, "TAKE OFF TO START THE MISSION", (w // 2, h * 0.40), 22, theme.GOOD, True, "midtop")
        self.toasts.draw(surf)

    def _objective_card(self, surf, runner, settings, w: int) -> None:
        m = runner.m
        if m.kind == "free":
            return
        cw, ch = 330, 76
        x, y = w // 2 - cw // 2, 62
        surf.blit(self._panel((cw, ch)), (x, y))
        text(surf, m.title.upper(), (x + 14, y + 7), 12, theme.OSD_DIM, False, shadow=False)
        text(surf, runner.objective_text(), (x + 14, y + 22), 24, theme.GATE_NEXT, True)
        if m.kind in ("targets", "attack") and runner.field.combo > 1 and runner.phase.value == "active":
            text(surf, f"COMBO x{min(runner.field.combo, 5)}", (x + cw - 14, y + 10), 18, theme.WARN, True, "topright")
        if m.kind == "attack":
            text(surf, f"{runner.field.hits} popped", (x + cw - 14, y + 34), 14, theme.OSD_DIM, False, "topright")
        pygame.draw.rect(surf, (0, 0, 0), (x + 14, y + ch - 20, cw - 28, 8), 1)
        pygame.draw.rect(surf, theme.GOOD, (x + 15, y + ch - 19, int((cw - 30) * min(1.0, runner.progress())), 6))

    def draw_targets(self, surf, v, runner, st, t: float) -> None:
        """Target bullseye markers, lock brackets, distance read-outs and an edge arrow to the next objective."""
        import numpy as np
        w, h = surf.get_size()
        pos = np.array(st.pos)
        goal = None
        if runner.m.kind in ("targets", "attack"):
            goal = runner.field.nearest(pos)
            goal_pos = None if goal is None else goal.centre(t)
        elif runner.m.kind == "race" and runner.course.gates:
            goal_pos = runner.course.gates[runner.tracker.next_gate % len(runner.course.gates)].pos
        else:
            goal_pos = None
        for tg in runner.field.alive():
            c = tg.centre(t)
            cam = (c - v.pos) @ v.R
            if cam[0] < 0.5:
                continue
            x, y = v.cx - v.f * cam[1] / cam[0], v.cy - v.f * cam[2] / cam[0]
            if not (0 <= x < w and 0 <= y < h):
                continue
            dist = float(np.linalg.norm(c - pos))
            r = max(8.0, v.f * tg.radius / cam[0])
            nearest = goal is tg
            col = theme.WARN if nearest else (255, 255, 255)
            # diamond marker above, corner brackets for the nearest target
            my = int(y - r - 14)
            pygame.draw.polygon(surf, col, [(x, my - 6), (x + 6, my), (x, my + 6), (x - 6, my)], 0 if nearest else 1)
            if nearest:
                k = int(r * 1.35)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        cx_, cy_ = x + sx * k, y + sy * k
                        pygame.draw.line(surf, col, (cx_, cy_), (cx_ - sx * 10, cy_), 2)
                        pygame.draw.line(surf, col, (cx_, cy_), (cx_, cy_ - sy * 10), 2)
                text(surf, f"{dist:4.0f} m", (x, y + k + 6), 16, col, True, "midtop")
                if math.hypot(x - w / 2, y - h / 2) < r:           # crosshair on target
                    text(surf, "ON TARGET", (x, my - 26), 14, theme.GOOD, True, "midbottom")
        if goal_pos is not None:
            self._edge_arrow(surf, v, goal_pos, pos)

    def _edge_arrow(self, surf, v, goal_pos, pos) -> None:
        import numpy as np
        w, h = surf.get_size()
        cam = (goal_pos - v.pos) @ v.R
        dist = float(np.linalg.norm(goal_pos - pos))
        if cam[0] > 0.5:
            x, y = v.cx - v.f * cam[1] / cam[0], v.cy - v.f * cam[2] / cam[0]
            if 40 < x < w - 40 and 100 < y < h - 120:
                return                                              # already visible on screen
        # direction of the goal in screen space (behind-camera goals flip so the arrow points the short way round)
        dx, dy = -cam[1], -cam[2]
        if cam[0] < 0.5 and abs(dx) < 1e-3 and abs(dy) < 1e-3:
            dx = 1.0
        n = math.hypot(dx, dy) or 1.0
        dx, dy = dx / n, dy / n
        cx, cy = w / 2, h / 2
        k = min((w / 2 - 70) / abs(dx) if abs(dx) > 1e-6 else 1e9, (h / 2 - 150) / abs(dy) if abs(dy) > 1e-6 else 1e9)
        px, py = cx + dx * k, cy + dy * k
        ang = math.atan2(dy, dx)
        pts = [(px + math.cos(ang) * 22, py + math.sin(ang) * 22),
               (px + math.cos(ang + 2.5) * 16, py + math.sin(ang + 2.5) * 16),
               (px + math.cos(ang - 2.5) * 16, py + math.sin(ang - 2.5) * 16)]
        pygame.draw.polygon(surf, theme.WARN, pts)
        pygame.draw.polygon(surf, (0, 0, 0), pts, 2)
        text(surf, f"{dist:.0f} m", (px - math.cos(ang) * 34, py - math.sin(ang) * 34), 15, theme.WARN, True, "center")

    def draw_floaters(self, surf, v, effects) -> None:
        import numpy as np
        for f in effects.floaters:
            cam = (f.pos - v.pos) @ v.R
            if cam[0] < 0.5:
                continue
            x, y = v.cx - v.f * cam[1] / cam[0], v.cy - v.f * cam[2] / cam[0]
            a = max(0.0, min(1.0, f.ttl / f.life * 1.6))
            text(surf, f.text, (x, y), int(22 + 10 * (1 - f.ttl / f.life)), theme.lerp((0, 0, 0), f.color, a), True, "center")

    def draw_briefing(self, surf, mission, t: float) -> None:
        """Mission card shown before take-off."""
        w, h = surf.get_size()
        pw, ph = min(640, w - 40), 170
        x, y = (w - pw) // 2, int(h * 0.50)
        surf.blit(self._panel((pw, ph), 14), (x, y))
        text(surf, mission.title.upper(), (x + 20, y + 14), 24, theme.ACCENT, True)
        text(surf, mission.brief, (x + 20, y + 48), 18, theme.OSD, True)
        words, line, ly = mission.detail.split(), "", y + 80
        for wd in words:
            if len(line) + len(wd) > (pw - 40) // 9:
                text(surf, line, (x + 20, ly), 15, theme.OSD_DIM)
                line, ly = wd, ly + 20
            else:
                line = (line + " " + wd).strip()
        text(surf, line, (x + 20, ly), 15, theme.OSD_DIM)
        if int(t * 2) % 2 == 0:
            text(surf, "ARM  >  THROTTLE UP  >  GO", (x + pw - 20, y + 14), 14, theme.WARN, True, "topright")

    # ---- instruments -----------------------------------------------------
    def draw_sticks(self, surf, sticks, throttle_state: float) -> None:
        w, h = surf.get_size()
        size = 92
        for k, (ox, vx, vy) in enumerate(((24, sticks.yaw, throttle_state * 2 - 1), (w - 24 - size, sticks.roll, sticks.pitch))):
            oy = h - BOTTOM - size
            surf.blit(self._panel((size, size)), (ox, oy))
            pygame.draw.line(surf, (*theme.OSD_DIM,), (ox + size // 2, oy + 6), (ox + size // 2, oy + size - 6), 1)
            pygame.draw.line(surf, (*theme.OSD_DIM,), (ox + 6, oy + size // 2), (ox + size - 6, oy + size // 2), 1)
            px = ox + size // 2 + int(vx * (size / 2 - 10))
            py = oy + size // 2 - int(vy * (size / 2 - 10))
            pygame.draw.circle(surf, theme.ACCENT, (px, py), 7)
            pygame.draw.circle(surf, theme.OSD, (px, py), 7, 2)

    def draw_motors(self, surf, st) -> None:
        w, h = surf.get_size()
        size = 70
        ox, oy = 24 + 80, 112 + 150 + 10
        surf.blit(self._panel((size, size)), (ox, oy))
        # layout matches physical X: FR, RR, RL, FL
        pos = ((1, 0), (1, 1), (0, 1), (0, 0))
        for m, (cx, cy) in zip(st.motors, pos):
            x, y = ox + 8 + cx * 30, oy + 8 + cy * 30
            pygame.draw.rect(surf, (0, 0, 0), (x, y, 24, 24), 1)
            fh = int(22 * m)
            pygame.draw.rect(surf, theme.lerp(theme.ACCENT, theme.DANGER, m), (x + 1, y + 23 - fh, 22, fh))

    def _minimap_ground(self, surf, course, lo, hi, span, origin, size) -> None:
        """Real map underneath the gates: crop of the world's top-down image (north up, west left)."""
        world = getattr(course, "world", None)
        if world is None or world.preview is None:
            return
        key = id(world)
        if getattr(self, "_mm_key", None) != key:
            img = world.preview
            self._mm_key = key
            self._mm_surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))      # (w, h) layout for pygame
        n = self._mm_surf.get_width()
        full = 1200.0                                           # metres covered by the image (-600 .. 600)
        cx, cy = (lo + hi) / 2
        half = span / 2 * (size / (size - 14))
        col0 = (600.0 - (cy + half)) / full * n                  # west is on the left: col grows as y falls
        row0 = (600.0 - (cx + half)) / full * n
        side = span * (size / (size - 14)) / full * n
        r = pygame.Rect(int(col0), int(row0), max(2, int(side)), max(2, int(side))).clip(self._mm_surf.get_rect())
        if r.w < 2 or r.h < 2:
            return
        crop = pygame.transform.smoothscale(self._mm_surf.subsurface(r), (size - 4, size - 4))
        crop.set_alpha(215)
        surf.blit(crop, (origin[0] + 2, origin[1] + 2))

    def _place_line(self, surf, st, course) -> None:
        """Top-left under the minimap: city, coordinates, altitude above sea level and compass heading (a real FPV OSD has these)."""
        world = getattr(course, "world", None)
        if world is None:
            return
        city = world.city
        h = surf.get_height()
        heading = (-math.degrees(st.euler[2])) % 360.0          # world yaw is counter-clockwise, compass is clockwise
        asl = city.elevation + max(0.0, st.pos[2] - world.terrain.ground(*world.pad))
        card = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((heading + 22.5) // 45) % 8]
        y = 350                                                  # under the minimap and motor widget
        text(surf, city.name.upper(), (24, y), 20, theme.OSD, True)
        text(surf, city.coords, (24, y + 24), 13, theme.OSD_DIM, False)
        text(surf, f"HDG {heading:03.0f} {card}    ASL {asl:,.0f} m", (24, y + 41), 13, theme.OSD_DIM, False)

    def draw_minimap(self, surf, course, st, next_gate: int, passed: set, targets=()) -> None:
        """Top-down course map (north up = +x up) so you always know where the next gate is."""
        pts = [g.pos[:2] for g in course.gates] + [t.pos[:2] for t in targets] + [(-4.0, 0.0), (60.0, 60.0)]
        P = np.array(pts, float)
        lo, hi = P.min(axis=0) - 12, P.max(axis=0) + 12
        size = 150
        span = float(max(hi - lo))
        ox, oy = 24, 112
        surf.blit(self._panel((size, size)), (ox, oy))
        self._minimap_ground(surf, course, lo, hi, span, (ox, oy), size)

        def to_px(xy):
            u = (np.asarray(xy, float) - (lo + hi) / 2) / span
            return int(ox + size / 2 - u[1] * (size - 14)), int(oy + size / 2 - u[0] * (size - 14))

        if course.gates:
            ring = [to_px(g.pos[:2]) for g in course.gates]
            pygame.draw.lines(surf, theme.OSD_DIM, True, ring, 1)
            for i, p in enumerate(ring):
                col = theme.GATE_NEXT if i == next_gate else theme.GATE_DONE if i in passed else theme.GATE_IDLE
                pygame.draw.circle(surf, col, p, 5 if i == next_gate else 3)
        for b in course.boxes[::2]:   # thin the obstacle layer so the map stays readable
            surf.set_at(to_px(b[:2]), (140, 150, 170))
        for t in targets:
            if t.alive:
                pygame.draw.circle(surf, (240, 60, 60), to_px(t.pos[:2]), 3)
                pygame.draw.circle(surf, (255, 255, 255), to_px(t.pos[:2]), 1)
        px, py = to_px(st.pos[:2])
        yaw = st.euler[2]
        tip = (px - int(math.sin(yaw) * 9), py - int(math.cos(yaw) * 9))
        pygame.draw.line(surf, theme.GOOD, (px, py), tip, 2)
        pygame.draw.circle(surf, theme.GOOD, (px, py), 3)

    def draw_graph(self, surf, telemetry, st) -> None:
        w, h = surf.get_size()
        gw, gh = 340, 150
        ox, oy = w - 24 - gw, 120
        surf.blit(self._panel((gw, gh)), (ox, oy))
        n = len(telemetry.series("t"))
        if n < 4:
            return
        text(surf, "RATES deg/s  (pitch)", (ox + 10, oy + 6), 13, theme.OSD_DIM, shadow=False)
        for name, col, wd in (("sp_p", theme.WARN, 1), ("gyro_p", theme.ACCENT, 2)):
            ys = np.degrees(telemetry.series(name))
            xs = np.linspace(ox + 8, ox + gw - 8, len(ys))
            ypix = oy + gh / 2 + 6 - np.clip(ys, -900, 900) / 900 * (gh / 2 - 22)
            pygame.draw.lines(surf, col, False, np.stack([xs, ypix], axis=1).tolist(), wd)
        pygame.draw.line(surf, theme.OSD_DIM, (ox + 8, oy + gh // 2 + 6), (ox + gw - 8, oy + gh // 2 + 6), 1)
        text(surf, "setpoint", (ox + gw - 10, oy + gh - 18), 12, theme.WARN, False, "topright", False)
        text(surf, "gyro", (ox + gw - 84, oy + gh - 18), 12, theme.ACCENT, False, "topright", False)

    def draw_help(self, surf) -> None:
        w, h = surf.get_size()
        lines = [("FLIGHT", ""), ("W / S", "throttle up / down  (Shift = fast, X = cut)"), ("A / D", "yaw left / right"),
                 ("Arrows", "pitch / roll"), ("SPACE", "arm / disarm"), ("R", "reset to start"),
                 ("M", "toggle ACRO / ANGLE"), ("C", "toggle FPV / chase camera"), ("G", "toggle rate graph"),
                 ("T", "autopilot demo flight"), ("P / H", "pause / help"), ("Mouse", "drag the on-screen sticks, click buttons"), ("N", "mute / unmute sound"), ("F5", "start / stop CSV blackbox"), ("ESC", "pause menu"), ("GAMEPAD", ""),
                 ("Sticks", "mode-2 layout"), ("A / B / X / Y", "arm / reset / camera / mode")]
        pw, ph = 560, 40 + 26 * len(lines)
        x, y = (w - pw) // 2, (h - ph) // 2
        surf.blit(self._panel((pw, ph)), (x, y))
        for i, (k, v) in enumerate(lines):
            if not v:
                text(surf, k, (x + 20, y + 16 + i * 26), 18, theme.ACCENT, True)
            else:
                text(surf, k, (x + 30, y + 16 + i * 26), 18, theme.WARN, True)
                text(surf, v, (x + 190, y + 16 + i * 26), 18, theme.OSD)
