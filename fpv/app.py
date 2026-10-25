"""Application shell: State pattern (Menu / Missions / Settings / Flight / Pause / Result) over a shared
service context (simple dependency injection)."""
from __future__ import annotations

import math
import time

import numpy as np
import pygame

from .audio import AudioEngine
from .events import EventBus
from .game.autopilot import Autopilot
from .game.course import build_course, default_terrain
from .game.effects import Effects
from .game.missions import MISSIONS, Mission, MissionRunner, Phase, mission_by_id, scaled
from .game.terrain import Terrain
from .game.telemetry import Telemetry
from .input.devices import Action, InputManager
from .input.sticks import Sticks
from .render import atmosphere, theme
from .render.camera import ChaseCamera, FpvCamera, OrbitCamera
from .render.fx import VideoFX
from .render.hud import Hud, fmt_time
from .render.scene import WorldRenderer
from .render.widgets import Button, ButtonBar, Item, Menu, draw_star
from .settings import Settings
from .sim.native import Simulator
from .world import cities
from .world.loader import WorldLoader, preview_image

SIZE = (1280, 720)
BOTTOM = ButtonBar.HEIGHT + 14


class Context:
    """Service container shared by all states."""

    def __init__(self, settings: Settings, auto_build: bool = True):
        self.settings = settings
        self.bus = EventBus()
        self.sim = Simulator(auto_build)
        self.loader = WorldLoader()
        self.loader.live = bool(settings.live_data)
        self.city: cities.City | None = None
        self.terrain: Terrain = self._resolve_terrain(settings.city)
        self.sim.set_terrain(self.terrain.heights, self.terrain.cell, self.terrain.ox, self.terrain.oy)
        self.hud = Hud()
        self.input = InputManager(settings.kb_sensitivity)
        self.telemetry = Telemetry()
        self.autopilot = Autopilot()
        self.effects = Effects()
        self.fx = VideoFX()
        self.audio = AudioEngine(settings.volume, settings.muted)
        self.fpv_cam = FpvCamera(settings.cam_tilt, settings.fov, settings.cam_shake)
        self.chase_cam = ChaseCamera()
        self.scene = WorldRenderer(build_course(1, self.terrain))
        self.scene.set_quality(settings.quality)
        self._tod = None
        self.race_gold = 0.0                         # gold time of the current city's race (for the mission cards)
        self.apply_time_of_day()
        self.autopilot_on = False
        self.last_sticks = Sticks()
        self._was_crashed = False
        self.last_new_best = False
        self.runner: MissionRunner | None = None
        self._wire_events()
        self._update_race_gold()
        self.load_mission(mission_by_id("free"))
        self.push_tuning()

    # --- location --------------------------------------------------------
    def _resolve_terrain(self, city_id: str, city: "cities.City | None" = None) -> Terrain:
        """Terrain for a city id ('valley' = the original practice valley). Unknown ids fall back to Kampala."""
        if city_id == "valley":
            self.city = None
            return default_terrain()
        self.city = city or cities.find(city_id) or cities.find("kampala")
        self.settings.city = self.city.id if not self.city.custom else "custom"
        return self.loader.get(self.city).terrain

    def set_city(self, city_id: str, city: "cities.City | None" = None) -> None:
        """Switch the whole simulation to another town: physics heightfield, renderer, mission and course."""
        self.settings.city = city_id
        self.terrain = self._resolve_terrain(city_id, city)
        t = self.terrain
        self.sim.set_terrain(t.heights, t.cell, t.ox, t.oy)
        self._update_race_gold()
        self.load_mission(self.runner.m if self.runner else mission_by_id("free"))

    def _update_race_gold(self) -> None:
        self.race_gold = scaled(mission_by_id("race"), build_course(0, self.terrain)).gold if self.city is not None else 0.0

    def apply_time_of_day(self) -> None:
        idx = int(self.settings.time_of_day) % len(atmosphere.TIMES)
        if idx != self._tod:
            self._tod = idx
            self.scene.set_time_of_day(atmosphere.TIMES[idx])

    # --- configuration ---------------------------------------------------
    def push_tuning(self) -> None:
        s = self.settings
        cfg = self.sim.get_config()
        s.apply_to(cfg)
        self.sim.set_config(cfg)
        self.sim.set_mode(s.mode)
        self.fpv_cam.tilt, self.fpv_cam.fov, self.fpv_cam.shake = s.cam_tilt, s.fov, s.cam_shake
        self.audio.set_volume(s.volume)
        if self.audio.muted != s.muted:
            self.audio.muted = s.muted
            if s.muted:
                self.audio.silence()
        self.scene.set_quality(s.quality)
        self.apply_time_of_day()
        self.input.virtual.enabled = s.onscreen_sticks

    def load_mission(self, mission: Mission) -> None:
        self.runner = MissionRunner(mission, self.bus, build_course(mission.course, self.terrain))
        self.course = self.runner.course
        self.scene.set_course(self.course)
        self.apply_time_of_day()
        self.sim.set_obstacles(self.course.collision_boxes())
        self.reset_flight()

    def reset_flight(self) -> None:
        self.sim.reset(self.course.start_pose)
        self.sim.set_mode(self.settings.mode)
        self.runner.restart()
        self.input.keyboard.s.throttle = 0.0
        self.autopilot_on = False
        self._was_crashed = False
        self.effects.clear()
        self.fx.reset()
        self.chase_cam.pos = None
        self.audio.silence()

    def passed_gates(self) -> set[int]:
        tr = self.runner.tracker
        return set(range(tr.next_gate)) if tr.gates_passed else set()

    # --- events --------------------------------------------------------------
    def _wire_events(self) -> None:
        t, bus = self.hud.toasts, self.bus
        bus.subscribe("gate", lambda index, next: (t.add(f"GATE {index + 1} CLEARED", theme.GATE_NEXT, 0.8),
                                                   self.audio.play("gate")))
        bus.subscribe("lap", self._on_lap)
        bus.subscribe("gate_missed", lambda index: t.add("MISSED GATE - TURN BACK", theme.WARN, 1.4))
        bus.subscribe("crash", self._on_crash)
        bus.subscribe("arm", lambda armed: (t.add("ARMED" if armed else "DISARMED", theme.GOOD if armed else theme.WARN, 1.0),
                                            self.audio.play("arm" if armed else "disarm")))
        bus.subscribe("target_hit", self._on_target)
        bus.subscribe("mission_start", lambda mission: (t.add("GO!", theme.GOOD, 1.2), self.audio.play("go")))
        bus.subscribe("mission_complete", self._on_complete)
        bus.subscribe("mission_failed", self._on_failed)

    def _on_lap(self, lap: int, time: float, best: float) -> None:
        self.settings.record_lap(time)
        self.hud.toasts.add(f"LAP {lap}   {time:6.2f}s", theme.GOOD, 2.5)
        self.audio.play("star")

    def _on_target(self, target, points: int, combo: int, remaining: int) -> None:
        self.effects.burst(target.centre(0.0), (240, 60, 60), 28, 8.0, 0.9, 3.5)
        self.effects.burst(target.centre(0.0), (255, 255, 255), 18, 5.0, 0.7, 3.0)
        self.effects.floater(target.centre(0.0), f"+{points}", (255, 220, 70))
        self.audio.play("pop")
        if combo >= 2:
            self.hud.toasts.add(f"COMBO x{min(combo, 5)}", theme.WARN, 1.0)

    def _on_crash(self) -> None:
        st = self.sim.state()
        self.effects.burst(np.array(st.pos), (255, 150, 60), 40, 6.0, 1.1, 3.0)
        self.effects.burst(np.array(st.pos), (90, 90, 100), 24, 4.0, 1.4, 4.0)
        self.hud.toasts.add("CRASH", theme.DANGER, 2.0)
        self.audio.play("crash")

    def _on_complete(self, mission, result) -> None:
        self.last_new_best = self.settings.record_result(mission, result)
        self.audio.play("win")

    def _on_failed(self, mission, reason: str) -> None:
        if reason != "CRASHED":
            self.audio.play("fail")


class State:
    def __init__(self, app: "App"):
        self.app, self.ctx = app, app.ctx

    def handle_event(self, e: pygame.event.Event) -> None: ...
    def update(self, dt: float) -> None: ...
    def draw(self, surf: pygame.Surface, dt: float) -> None: ...


def _backdrop(ctx: Context, surf: pygame.Surface, orbit: OrbitCamera, dt: float, dim: int = 70) -> None:
    w, h = surf.get_size()
    v = orbit.view(ctx.sim.state(), w, h, dt)
    ctx.scene.draw(surf, v, ctx.sim.state(), 0, set(), dt, False, ctx.runner.field.targets, None)
    d = pygame.Surface((w, h), pygame.SRCALPHA)
    d.fill((4, 10, 30, dim))
    surf.blit(d, (0, 0))


def _orbit() -> OrbitCamera:
    return OrbitCamera((40, 45, 12), radius=110.0, height=38.0)


class MenuState(State):
    def __init__(self, app):
        super().__init__(app)
        self.t = 0.0
        self.orbit = _orbit()
        self.menu = Menu("FPV SIM", [
            Item("Fly  -  choose a city and mission", on_select=lambda: app.push(CitySelectState(app))),
            Item("Watch the autopilot", on_select=self._demo),
            Item("Settings", on_select=lambda: app.push(SettingsState(app))),
            Item("Controls", on_select=lambda: app.push(ControlsState(app))),
            Item("Quit", on_select=app.quit),
        ], subtitle="FPV drone simulator  -  fly over any town in Uganda",
            footer="UP/DOWN select    ENTER confirm    mouse works too    F11 fullscreen")

    def _demo(self):
        self.app.push(FlightState(self.app, mission_by_id("race"), demo=True))

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self.app.quit()
        self.menu.handle_event(e)

    def update(self, dt):
        self.t += dt

    def draw(self, surf, dt):
        _backdrop(self.ctx, surf, self.orbit, dt)
        self.menu.draw(surf, self.t)


class CitySelectState(State):
    """Pick where to fly: scrollable list of Ugandan towns, type-to-filter, live preview map, time of day.

    Typing "0.35, 32.58" (latitude, longitude) adds a "custom location" row at the top, so *any* point in Uganda
    can be flown, not just the named towns.  Worlds are generated on a worker thread (see world.loader).
    """
    ROWS = 12
    VALLEY = "valley"

    def __init__(self, app):
        super().__init__(app)
        self.t = 0.0
        self.orbit = _orbit()
        self.query = ""
        self.index = 0
        self.top = 0
        self.dwell = 0.0               # seconds the highlight has stayed on one entry (don't build while scrolling)
        self.go = False                # Enter pressed while the world was still building
        self.row_rects: list[tuple[pygame.Rect, int]] = []
        self._thumb: tuple | None = None
        self._refilter()
        cur = self.ctx.city.id if self.ctx.city else self.VALLEY
        for i, e in enumerate(self.entries):
            if getattr(e, "id", self.VALLEY) == cur:
                self.index = i
        self._scroll_to_index()

    # --- list model --------------------------------------------------------------------------------
    def _parse_coords(self):
        parts = self.query.replace(";", ",").replace(" ", ",").split(",")
        nums = []
        for p in parts:
            if p:
                try:
                    nums.append(float(p))
                except ValueError:
                    return None
        return (nums[0], nums[1]) if len(nums) == 2 else None

    def _refilter(self) -> None:
        self.entries: list = []
        self.note = ""
        ll = self._parse_coords()
        if ll is not None:
            if cities.in_uganda(*ll):
                self.entries.append(cities.custom_city(ll[0], ll[1]))
            else:
                self.note = "That point is outside Uganda"
        self.entries += cities.search(self.query) if ll is None else []
        if not self.query.strip() or "valley".startswith(self.query.strip().lower()):
            self.entries.append(self.VALLEY)
        self.index = min(self.index, max(0, len(self.entries) - 1))
        self.dwell = 0.0

    def _scroll_to_index(self) -> None:
        self.top = max(0, min(self.index - self.ROWS // 2, len(self.entries) - self.ROWS))

    def _current(self):
        return self.entries[self.index] if self.entries else None

    def _city_of(self, e):
        return None if e == self.VALLEY else e

    # --- actions -----------------------------------------------------------------------------------
    def _confirm(self) -> None:
        e = self._current()
        if e is None:
            return
        city = self._city_of(e)
        if city is None or self.ctx.loader.cached(city) is not None:
            self._enter(e)
        else:
            self.ctx.loader.request(city)
            self.go = True
        self.ctx.audio.play("click")

    def _enter(self, e) -> None:
        city = self._city_of(e)
        s = self.ctx.settings
        if city is None:
            self.ctx.set_city(self.VALLEY)
        else:
            if city.custom:
                s.custom_lat, s.custom_lon = city.lat, city.lon
            self.ctx.set_city(city.id, city)
        self.go = False
        self.app.push(MissionSelectState(self.app))

    def _toggle_live(self) -> None:
        s = self.ctx.settings
        s.live_data = not s.live_data
        self.ctx.loader.live = s.live_data
        self._thumb = None
        self.go = False

    def _cycle_time(self, d: int) -> None:
        s = self.ctx.settings
        s.time_of_day = (s.time_of_day + d) % len(atmosphere.TIMES)
        self.ctx.apply_time_of_day()

    def _move(self, d: int) -> None:
        if self.entries:
            self.index = max(0, min(len(self.entries) - 1, self.index + d))
            if self.index < self.top:
                self.top = self.index
            elif self.index >= self.top + self.ROWS:
                self.top = self.index - self.ROWS + 1
            self.dwell, self.go = 0.0, False

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                if self.query:
                    self.query = ""
                    self._refilter()
                else:
                    self.app.pop()
            elif e.key == pygame.K_DOWN:
                self._move(1)
            elif e.key == pygame.K_UP:
                self._move(-1)
            elif e.key == pygame.K_PAGEDOWN:
                self._move(self.ROWS)
            elif e.key == pygame.K_PAGEUP:
                self._move(-self.ROWS)
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._confirm()
            elif e.key == pygame.K_F2:
                self._toggle_live()
            elif e.key in (pygame.K_TAB, pygame.K_RIGHT):
                self._cycle_time(1)
            elif e.key == pygame.K_LEFT:
                self._cycle_time(-1)
            elif e.key == pygame.K_BACKSPACE:
                self.query = self.query[:-1]
                self._refilter()
            elif e.unicode and e.unicode.isprintable() and len(self.query) < 24:
                self.query += e.unicode
                self._refilter()
        elif e.type == pygame.MOUSEWHEEL:
            self.top = max(0, min(self.top - e.y * 2, max(0, len(self.entries) - self.ROWS)))
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if pygame.Rect(24, 20, 110, 38).collidepoint(e.pos):
                self.app.pop()
                return
            for r, i in self.row_rects:
                if r.collidepoint(e.pos):
                    if i == self.index:
                        self._confirm()
                    else:
                        self.index, self.dwell, self.go = i, 0.0, False
                    return
            for k, r in enumerate(self._chips):
                if r.collidepoint(e.pos):
                    self.ctx.settings.time_of_day = k
                    self.ctx.apply_time_of_day()
                    return

    _chips: list = []

    def update(self, dt):
        self.t += dt
        self.dwell += dt
        loader = self.ctx.loader
        loader.poll()
        e = self._current()
        city = self._city_of(e) if e is not None else None
        if city is not None and self.dwell > 0.35:           # build the highlighted town in the background
            loader.request(city)
        if self.go and city is not None and loader.cached(city) is not None:
            self._enter(e)
        elif self.go and city is not None and loader.error(city):
            self.go = False

    # --- drawing -----------------------------------------------------------------------------------
    def draw(self, surf, dt):
        _backdrop(self.ctx, surf, self.orbit, dt, 150)
        w, h = surf.get_size()
        theme.text(surf, "CHOOSE A CITY", (w // 2, 22), 38, theme.ACCENT, True, "midtop")
        back = pygame.Rect(24, 20, 110, 38)
        pygame.draw.rect(surf, (6, 12, 28), back, border_radius=8)
        pygame.draw.rect(surf, theme.OSD_DIM, back, 1, border_radius=8)
        theme.text(surf, "< BACK", back.center, 16, theme.OSD, True, "center", False)

        # --- left: search box + list
        lx, ly, lw = 60, 96, 400
        box = pygame.Rect(lx, ly, lw, 36)
        pygame.draw.rect(surf, (6, 12, 30), box, border_radius=8)
        pygame.draw.rect(surf, theme.ACCENT, box, 2, border_radius=8)
        cursor = "_" if int(self.t * 2) % 2 == 0 else " "
        shown = self.query + cursor if self.query else "town name, or  lat, lon"
        theme.text(surf, shown, (box.x + 12, box.y + 8), 18, theme.OSD if self.query else theme.OSD_DIM, False, shadow=False)
        self.row_rects = []
        y0 = ly + 46
        rh = 38
        panel = pygame.Surface((lw, self.ROWS * rh + 8), pygame.SRCALPHA)
        pygame.draw.rect(panel, (6, 12, 30, 205), panel.get_rect(), border_radius=10)
        surf.blit(panel, (lx, y0 - 4))
        cur = self.ctx.city.id if self.ctx.city else self.VALLEY
        for k in range(self.ROWS):
            i = self.top + k
            if i >= len(self.entries):
                break
            e = self.entries[i]
            r = pygame.Rect(lx + 6, y0 + k * rh, lw - 12, rh - 3)
            self.row_rects.append((r, i))
            sel = i == self.index
            if sel:
                pygame.draw.rect(surf, (20, 60, 100), r, border_radius=7)
                pygame.draw.rect(surf, theme.ACCENT, r, 2, border_radius=7)
            if e == self.VALLEY:
                name, sub = "Practice valley", "original training course"
            else:
                name = e.name
                sub = ("custom location" if e.custom else f"{e.region}  -  {e.size}")
            theme.text(surf, name, (r.x + 12, r.y + 4), 19, theme.OSD if sel else (200, 214, 232), sel, shadow=False)
            theme.text(surf, sub, (r.x + 12, r.y + 22), 11, theme.OSD_DIM, False, shadow=False)
            if (e == self.VALLEY and cur == self.VALLEY) or (e != self.VALLEY and e.id == cur):
                theme.text(surf, "current", (r.right - 10, r.y + 10), 12, theme.GOOD, False, "topright", False)
        if len(self.entries) > self.ROWS:                      # scroll bar
            th = (self.ROWS * rh) * self.ROWS / len(self.entries)
            ty = y0 + (self.ROWS * rh - th) * self.top / max(1, len(self.entries) - self.ROWS)
            pygame.draw.rect(surf, theme.OSD_DIM, (lx + lw - 6, ty, 3, th), border_radius=2)
        if not self.entries:
            theme.text(surf, self.note or "No match", (lx + 16, y0 + 12), 18, theme.WARN, False, shadow=False)
        elif self.note:
            theme.text(surf, self.note, (lx, y0 + self.ROWS * rh + 14), 14, theme.WARN, False)

        # --- right: preview + facts
        px, py, pw = 500, 96, 720
        e = self._current()
        card = pygame.Surface((pw, 520), pygame.SRCALPHA)
        pygame.draw.rect(card, (6, 12, 30, 205), card.get_rect(), border_radius=14)
        surf.blit(card, (px, py))
        if e is not None:
            self._draw_preview(surf, e, px + 18, py + 18, 360)
            self._draw_facts(surf, e, px + 396, py + 18, pw - 414)
        # time of day chips
        self._chips = []
        cx = px + 18
        theme.text(surf, "TIME OF DAY", (cx, py + 398), 12, theme.OSD_DIM, True, shadow=False)
        for k, name in enumerate(atmosphere.TIMES):
            r = pygame.Rect(cx + k * 168, py + 420, 160, 40)
            self._chips.append(r)
            on = k == self.ctx.settings.time_of_day
            pygame.draw.rect(surf, (20, 60, 100) if on else (10, 18, 40), r, border_radius=8)
            pygame.draw.rect(surf, theme.ACCENT if on else theme.OSD_DIM, r, 2 if on else 1, border_radius=8)
            theme.text(surf, name, r.center, 16, theme.OSD if on else theme.OSD_DIM, on, "center", False)
        live = self.ctx.settings.live_data
        theme.text(surf, "F2  LIVE MAP DATA: " + ("ON  (OpenStreetMap, needs internet - falls back if unavailable)" if live else "OFF  (generated towns)"),
                   (px + 18, py + 478), 13, theme.GOOD if live else theme.OSD_DIM, False, shadow=False)
        theme.text(surf, "ENTER fly here    TAB / arrows time of day    ESC clear / back    type to search    F2 live data",
                   (w // 2, h - 34), 15, theme.OSD_DIM, False, "midtop")

    def _draw_preview(self, surf, e, x, y, size) -> None:
        r = pygame.Rect(x, y, size, size)
        pygame.draw.rect(surf, (10, 18, 40), r, border_radius=8)
        city = self._city_of(e)
        world = self.ctx.loader.cached(city) if city is not None else None
        if city is None:
            theme.text(surf, "Open training valley", r.center, 18, theme.OSD_DIM, False, "center", False)
        elif world is not None:
            key = (id(world), size)
            if self._thumb is None or self._thumb[0] != key:
                img = preview_image(world)
                s = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
                self._thumb = (key, pygame.transform.smoothscale(s, (size, size)))
            surf.blit(self._thumb[1], r)
            theme.text(surf, "N", (r.right - 14, r.y + 6), 14, (255, 255, 255), True, "topright")
        else:
            err = self.ctx.loader.error(city)
            if err:
                theme.text(surf, "Could not build this town", r.center, 16, theme.DANGER, True, "center", False)
            else:
                dots = "." * (int(self.t * 3) % 4)
                msg = "Building" if self.ctx.loader.building(city) else "Select to build"
                theme.text(surf, f"{msg}{dots}", (r.centerx, r.centery - 10), 20, theme.OSD, True, "center", False)
                if self.ctx.loader.building(city):
                    theme.text(surf, self.ctx.loader.progress, (r.centerx, r.centery + 16), 13, theme.OSD_DIM, False, "center", False)
        pygame.draw.rect(surf, theme.OSD_DIM, r, 1, border_radius=8)

    def _draw_facts(self, surf, e, x, y, width) -> None:
        if e == self.VALLEY:
            theme.text(surf, "Practice valley", (x, y), 30, theme.OSD, True)
            lines = ["The original course: rolling hills, a lake, a few pylons.",
                     "A calm place to learn the sticks before you fly over a town."]
            facts = []
        else:
            theme.text(surf, e.name, (x, y), 30, theme.OSD, True)
            lines = [e.blurb]
            facts = [("REGION", e.region), ("POSITION", e.coords), ("ELEVATION", f"{e.elevation:,} m above sea level")]
            w = self.ctx.loader.cached(e)
            if w is not None:
                facts.append(("DATA", "OpenStreetMap streets + buildings" if w.source == "openstreetmap" else "generated layout"))
        yy = y + 46
        for k, v in facts:
            theme.text(surf, k, (x, yy), 11, theme.OSD_DIM, True, shadow=False)
            theme.text(surf, v, (x, yy + 14), 16, theme.OSD, False, shadow=False)
            yy += 40
        yy += 6
        for ln in lines:
            yy = self._wrap(surf, ln, x, yy, width, 14, (190, 205, 225))

    @staticmethod
    def _wrap(surf, s, x, y, width, size, color) -> int:
        line = ""
        per = max(10, int(width / (size * 0.56)))
        for wd in s.split():
            if len(line) + len(wd) + 1 > per:
                theme.text(surf, line, (x, y), size, color, False, shadow=False)
                line, y = wd, y + size + 5
            else:
                line = (line + " " + wd).strip()
        if line:
            theme.text(surf, line, (x, y), size, color, False, shadow=False)
            y += size + 5
        return y


class MissionSelectState(State):
    def __init__(self, app):
        super().__init__(app)
        self.t = 0.0
        self.index = 0
        self.orbit = _orbit()
        self.rects: list[pygame.Rect] = []

    def _start(self, i: int) -> None:
        self.ctx.audio.play("click")
        self.app.push(FlightState(self.app, MISSIONS[i]))

    def handle_event(self, e):
        n = len(MISSIONS)
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self.app.pop()
            elif e.key in (pygame.K_RIGHT, pygame.K_d):
                self.index = (self.index + 1) % n
            elif e.key in (pygame.K_LEFT, pygame.K_a):
                self.index = (self.index - 1) % n
            elif e.key in (pygame.K_DOWN, pygame.K_s):
                self.index = min(n - 1, self.index + 3)
            elif e.key in (pygame.K_UP, pygame.K_w):
                self.index = max(0, self.index - 3)
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._start(self.index)
        elif e.type == pygame.MOUSEMOTION:
            for i, r in enumerate(self.rects):
                if r.collidepoint(e.pos):
                    self.index = i
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if pygame.Rect(24, 20, 110, 38).collidepoint(e.pos):
                self.app.pop()
                return
            for i, r in enumerate(self.rects):
                if r.collidepoint(e.pos):
                    self._start(i)
                    return

    def update(self, dt):
        self.t += dt

    def draw(self, surf, dt):
        _backdrop(self.ctx, surf, self.orbit, dt, 110)
        w, h = surf.get_size()
        theme.text(surf, "SELECT MISSION", (w // 2, 26), 40, theme.ACCENT, True, "midtop")
        place = self.ctx.city.name if self.ctx.city else "Practice valley"
        theme.text(surf, place, (w - 30, 34), 22, theme.OSD, True, "topright")
        total = sum(self.ctx.settings.stars(m.id) for m in MISSIONS)
        theme.text(surf, f"{total} / {3 * (len(MISSIONS) - 1)} stars", (w // 2, 76), 16, theme.OSD_DIM, False, "midtop")
        back = pygame.Rect(24, 20, 110, 38)
        pygame.draw.rect(surf, (6, 12, 28), back, border_radius=8)
        pygame.draw.rect(surf, theme.OSD_DIM, back, 1, border_radius=8)
        theme.text(surf, "< BACK", back.center, 16, theme.OSD, True, "center", False)

        cols = 3
        gap = 22
        cw = min(380, (w - 2 * 60 - gap * (cols - 1)) // cols)
        ch = 210
        x0 = (w - (cols * cw + (cols - 1) * gap)) // 2
        y0 = 120
        self.rects = []
        for i, m in enumerate(MISSIONS):
            r = pygame.Rect(x0 + (i % cols) * (cw + gap), y0 + (i // cols) * (ch + gap), cw, ch)
            self.rects.append(r)
            sel = i == self.index
            card = pygame.Surface(r.size, pygame.SRCALPHA)
            pygame.draw.rect(card, (6, 12, 30, 225 if sel else 190), card.get_rect(), border_radius=14)
            pygame.draw.rect(card, (*theme.ACCENT, 255 if sel else 90), card.get_rect(), 3 if sel else 1, border_radius=14)
            surf.blit(card, r)
            lift = int(2 * math.sin(self.t * 5)) if sel else 0
            kind = {"targets": "TARGETS", "attack": "TIME ATTACK", "race": "RACE", "free": "FREE"}[m.kind]
            theme.text(surf, kind, (r.x + 18, r.y + 14 + lift), 12, theme.WARN, True, shadow=False)
            theme.text(surf, m.title, (r.x + 18, r.y + 34 + lift), 26, theme.OSD, True)
            self._wrap(surf, m.brief, r.x + 18, r.y + 72, cw - 36, 15, theme.OSD_DIM)
            if m.kind != "free":
                stars = self.ctx.settings.stars(m.id)
                for k in range(3):
                    draw_star(surf, (r.x + 34 + k * 36, r.bottom - 48), 15, k < stars)
                rec = self.ctx.settings.record(m.id)
                txt = "not flown yet" if not rec else (f"best {rec['value']:.0f} pts" if m.kind == "attack" else f"best {fmt_time(rec['value'])}")
                theme.text(surf, txt, (r.right - 18, r.bottom - 56), 14, theme.OSD_DIM, False, "topright", False)
                gold = self.ctx.race_gold if m.kind == "race" and self.ctx.race_gold else m.gold
                goal = f"gold: {m.gold:.0f} pts" if m.kind == "attack" else f"gold: under {int(gold // 60)}:{int(gold % 60):02d}"
                theme.text(surf, goal, (r.right - 18, r.bottom - 36), 13, (255, 210, 60), False, "topright", False)
            else:
                theme.text(surf, "no goals, no timer", (r.x + 18, r.bottom - 44), 14, theme.OSD_DIM, False, shadow=False)
        theme.text(surf, "ARROWS select    ENTER / click to fly    ESC back", (w // 2, h - 34), 15, theme.OSD_DIM, False, "midtop")

    @staticmethod
    def _wrap(surf, s, x, y, width, size, color):
        line, ly = "", y
        per = max(10, int(width / (size * 0.6)))
        for wd in s.split():
            if len(line) + len(wd) + 1 > per:
                theme.text(surf, line, (x, ly), size, color, False, shadow=False)
                line, ly = wd, ly + size + 5
            else:
                line = (line + " " + wd).strip()
        theme.text(surf, line, (x, ly), size, color, False, shadow=False)


class SettingsState(State):
    def __init__(self, app):
        super().__init__(app)
        self.t = 0.0
        self.orbit = _orbit()
        s = self.ctx.settings

        def setter(name, cast=None):
            def f(v):
                setattr(s, name, cast(v) if cast else v)
                self.ctx.push_tuning()
                self.ctx.audio.play("hover", 0.8)
            return f
        G = lambda name: (lambda: getattr(s, name))  # noqa: E731
        onoff = ["off", "on"]
        self.menu = Menu("SETTINGS", [
            Item("Flight mode", get=G("mode"), set=setter("mode"), options=["ACRO  (rate)", "ANGLE (self-level)"]),
            Item("Camera tilt", get=G("cam_tilt"), set=setter("cam_tilt"), rng=(0, 55, 1), fmt="{:.0f} deg"),
            Item("Field of view", get=G("fov"), set=setter("fov"), rng=(80, 150, 5), fmt="{:.0f} deg"),
            Item("Camera vibration", get=G("cam_shake"), set=setter("cam_shake"), rng=(0, 1.5, 0.25), fmt="{:.2f}"),
            Item("Analog video look", get=G("analog_fx"), set=setter("analog_fx", bool), options=onoff),
            Item("Time of day", get=G("time_of_day"), set=setter("time_of_day"), options=atmosphere.TIMES),
            Item("Graphics quality", get=G("quality"), set=setter("quality"), options=["Low", "Medium", "High"]),
            Item("On-screen sticks", get=G("onscreen_sticks"), set=setter("onscreen_sticks", bool), options=onoff),
            Item("Volume", get=G("volume"), set=setter("volume"), rng=(0, 1, 0.1), fmt="{:.0%}"),
            Item("Sound muted", get=G("muted"), set=setter("muted", bool), options=["no", "yes"]),
            Item("Drone tuning (rates, PID, wind)", on_select=lambda: app.push(TuningState(app))),
            Item("Back", on_select=app.pop),
        ], footer="UP/DOWN select    LEFT/RIGHT adjust    ESC back")

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self.app.pop()
            return
        self.menu.handle_event(e)

    def update(self, dt):
        self.t += dt

    def draw(self, surf, dt):
        _backdrop(self.ctx, surf, self.orbit, dt, 120)
        self.menu.draw(surf, self.t)


class TuningState(SettingsState):
    def __init__(self, app):
        State.__init__(self, app)
        self.t = 0.0
        self.orbit = _orbit()
        s = self.ctx.settings

        def setter(name):
            def f(v):
                setattr(s, name, v)
                self.ctx.push_tuning()
            return f
        G = lambda name: (lambda: getattr(s, name))  # noqa: E731
        self.menu = Menu("DRONE TUNING", [
            Item("Rates scale", get=G("rate_scale"), set=setter("rate_scale"), rng=(0.5, 1.6, 0.1), fmt="x{:.1f}"),
            Item("PID  P gain", get=G("pid_p"), set=setter("pid_p"), rng=(0.4, 1.8, 0.05), fmt="x{:.2f}"),
            Item("PID  I gain", get=G("pid_i"), set=setter("pid_i"), rng=(0.0, 2.5, 0.1), fmt="x{:.2f}"),
            Item("PID  D gain", get=G("pid_d"), set=setter("pid_d"), rng=(0.0, 3.0, 0.1), fmt="x{:.2f}"),
            Item("Wind", get=G("wind"), set=setter("wind"), rng=(0, 14, 1), fmt="{:.0f} m/s"),
            Item("Gusts", get=G("gust"), set=setter("gust"), rng=(0, 6, 0.5), fmt="{:.1f} m/s"),
            Item("Back", on_select=app.pop),
        ], subtitle="Changes apply immediately", footer="UP/DOWN select    LEFT/RIGHT adjust    ESC back")


class ControlsState(State):
    def __init__(self, app):
        super().__init__(app)
        self.t = 0.0
        self.orbit = _orbit()

    def handle_event(self, e):
        if e.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            self.app.pop()

    def update(self, dt):
        self.t += dt

    def draw(self, surf, dt):
        _backdrop(self.ctx, surf, self.orbit, dt, 130)
        self.ctx.hud.draw_help(surf)
        w, h = surf.get_size()
        theme.text(surf, "press any key or click to go back", (w // 2, h - 40), 15, theme.OSD_DIM, False, "midtop")


class FlightState(State):
    END_DELAY_WIN, END_DELAY_FAIL = 1.1, 1.6

    def __init__(self, app, mission: Mission | None = None, demo: bool = False):
        super().__init__(app)
        ctx = self.ctx
        self.mission = mission or mission_by_id("free")
        ctx.load_mission(self.mission)
        ctx.push_tuning()
        self.show_help = False
        self.t = 0.0
        self._end_timer: float | None = None
        ctx.autopilot_on = demo
        if demo:
            self._set_mode(1)
            ctx.sim.set_armed(True)
            ctx.bus.publish("arm", armed=True)
        self.bar = ButtonBar([
            Button(lambda: "DISARM" if ctx.sim.state().armed else "ARM", self._arm_toggle,
                   lambda: bool(ctx.sim.state().armed), "SPACE"),
            Button(lambda: "RESTART", self._restart, tip="R"),
            Button(lambda: "ANGLE" if ctx.settings.mode else "ACRO", self._toggle_mode, tip="M"),
            Button(lambda: "CHASE" if ctx.settings.chase_cam else "FPV", self._toggle_camera, tip="C"),
            Button(lambda: "AUTO", self._toggle_auto, lambda: ctx.autopilot_on, "T"),
            Button(lambda: "SOUND" if not ctx.audio.muted else "MUTED", self._toggle_mute,
                   lambda: not ctx.audio.muted, "N"),
            Button(lambda: "PAUSE", self._pause, tip="ESC"),
            Button(lambda: "HELP", self._toggle_help, lambda: self.show_help, "F1"),
        ])

    # --- actions ---------------------------------------------------------
    def _restart(self):
        self.ctx.reset_flight()
        self._end_timer = None

    def _pause(self):
        self.app.push(PauseState(self.app))

    def _arm_toggle(self):
        ctx, st = self.ctx, self.ctx.sim.state()
        if st.crashed:
            ctx.hud.toasts.add("CRASHED  -  PRESS RESTART", theme.DANGER, 1.2)
            return
        if not st.armed and ctx.last_sticks.throttle > 0.1 and not ctx.autopilot_on:
            ctx.hud.toasts.add("LOWER THROTTLE TO ARM", theme.WARN, 1.5)
            return
        ctx.sim.set_armed(not st.armed)
        ctx.bus.publish("arm", armed=not st.armed)

    def _set_mode(self, mode: int):
        self.ctx.settings.mode = mode
        self.ctx.push_tuning()

    def _toggle_mode(self):
        self._set_mode(self.ctx.settings.mode ^ 1)
        self.ctx.hud.toasts.add("ANGLE MODE" if self.ctx.settings.mode else "ACRO MODE", theme.ACCENT, 1.0)

    def _toggle_camera(self):
        self.ctx.settings.chase_cam = not self.ctx.settings.chase_cam

    def _toggle_help(self):
        self.show_help = not self.show_help

    def _toggle_mute(self):
        muted = self.ctx.audio.toggle_mute()
        self.ctx.settings.muted = muted
        self.ctx.hud.toasts.add("SOUND OFF" if muted else "SOUND ON", theme.ACCENT, 1.0)

    def _set_auto(self, on: bool, announce: bool = True):
        ctx = self.ctx
        if on == ctx.autopilot_on:
            return
        ctx.autopilot_on = on
        if on:
            self._set_mode(1)                        # the autopilot flies in self-level mode
            st = ctx.sim.state()
            if not st.armed and not st.crashed:
                ctx.sim.set_armed(True)
                ctx.bus.publish("arm", armed=True)
        else:
            ctx.input.keyboard.s.throttle = ctx.last_sticks.throttle   # hand over without a throttle drop
        if announce:
            ctx.hud.toasts.add("AUTOPILOT ON" if on else "AUTOPILOT OFF", theme.ACCENT, 1.2)

    def _toggle_auto(self):
        self._set_auto(not self.ctx.autopilot_on)

    def _toggle_record(self):
        ctx = self.ctx
        path = ctx.telemetry.toggle_recording()
        ctx.hud.toasts.add("RECORDING" if ctx.telemetry.recording else f"SAVED {path.name if path else ''}",
                           theme.DANGER, 1.8)

    def _layout(self):
        size = pygame.display.get_surface().get_size()
        self.ctx.input.virtual.layout(size, BOTTOM)
        return size

    def handle_event(self, e):
        ctx = self.ctx
        size = self._layout()
        if self.bar.handle_event(e, size):
            ctx.audio.play("click")
            return
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self._pause()
            return
        act = ctx.input.handle_event(e, size)
        handlers = {Action.ARM: self._arm_toggle, Action.RESET: self._restart, Action.MODE: self._toggle_mode,
                    Action.CAMERA: self._toggle_camera, Action.HELP: self._toggle_help, Action.PAUSE: self._pause,
                    Action.AUTOPILOT: self._toggle_auto, Action.RECORD: self._toggle_record,
                    Action.MUTE: self._toggle_mute,
                    Action.GRAPH: lambda: setattr(ctx.settings, "show_graph", not ctx.settings.show_graph)}
        if act in handlers:
            handlers[act]()

    def update(self, dt):
        ctx = self.ctx
        self.t += dt
        self._layout()
        st = ctx.sim.state()
        manual = ctx.input.poll(dt)
        if ctx.autopilot_on and (ctx.input.virtual.holding or manual.magnitude() > 0.25):
            self._set_auto(False, announce=False)
            ctx.hud.toasts.add("MANUAL CONTROL", theme.ACCENT, 1.0)
        if ctx.autopilot_on:
            sticks = ctx.autopilot.compute(st, ctx.course, ctx.runner.tracker.next_gate, ctx.runner.field)
        else:
            sticks = manual
        ctx.last_sticks = sticks
        ctx.input.virtual.sync(sticks)
        ctx.sim.set_sticks(sticks.roll, sticks.pitch, sticks.yaw, sticks.throttle)
        ctx.sim.advance(dt)
        st = ctx.sim.state()
        ctx.telemetry.push(st)
        ctx.runner.update(dt, st, st.time)
        if st.crashed and not ctx._was_crashed:
            ctx.bus.publish("crash")
            ctx.autopilot_on = False
        ctx._was_crashed = bool(st.crashed)
        ctx.fx.update(dt, bool(st.crashed))
        ctx.effects.update(dt)
        ctx.hud.toasts.update(dt)
        ctx.audio.update(st, True)

        if ctx.runner.phase in (Phase.COMPLETE, Phase.FAILED) and self._end_timer is None:
            self._end_timer = self.END_DELAY_WIN if ctx.runner.phase is Phase.COMPLETE else self.END_DELAY_FAIL
        if self._end_timer is not None:
            self._end_timer -= dt
            if self._end_timer <= 0:
                self._end_timer = None
                ctx.audio.silence()
                self.app.push(ResultState(self.app, self))

    def draw(self, surf, dt):
        ctx, s = self.ctx, self.ctx.settings
        w, h = surf.get_size()
        st = ctx.sim.state()
        cam = ctx.chase_cam if s.chase_cam else ctx.fpv_cam
        v = cam.view(st, w, h, dt)
        passed = ctx.passed_gates()
        run = ctx.runner
        ctx.scene.draw(surf, v, st, run.tracker.next_gate, passed, dt, s.chase_cam, run.field.targets, ctx.effects)
        ctx.hud.draw_targets(surf, v, run, st, ctx.scene.t)
        ctx.hud.draw_floaters(surf, v, ctx.effects)
        ctx.fx.draw(surf, s.analog_fx)
        ctx.hud.draw_osd(surf, st, run, s, cam.name, ctx.input.active.name + (" + AUTOPILOT" if ctx.autopilot_on else ""),
                         ctx.telemetry.recording)
        if s.show_sticks:
            ctx.hud.draw_motors(surf, st)
        if s.show_graph:
            ctx.hud.draw_graph(surf, ctx.telemetry, st)
        if ctx.course.gates or run.field.targets:
            ctx.hud.draw_minimap(surf, ctx.course, st, run.tracker.next_gate, passed, run.field.targets)
        if run.phase is Phase.READY and run.m.kind != "free" and not st.armed and self.t > 0.3:
            ctx.hud.draw_briefing(surf, run.m, self.t)
        if self.show_help:
            ctx.hud.draw_help(surf)
        ctx.input.virtual.draw(surf)
        self.bar.draw(surf)


class PauseState(State):
    def __init__(self, app):
        super().__init__(app)
        self.t = 0.0
        self.ctx.audio.silence()
        self.menu = Menu("PAUSED", [
            Item("Resume", on_select=app.pop),
            Item("Restart mission", on_select=self._restart),
            Item("Flight mode", get=lambda: self.ctx.settings.mode, set=self._set_mode, options=["ACRO", "ANGLE"]),
            Item("Chase camera", get=lambda: int(self.ctx.settings.chase_cam),
                 set=lambda v: setattr(self.ctx.settings, "chase_cam", bool(v)), options=["off", "on"]),
            Item("Sound", get=lambda: int(not self.ctx.settings.muted), set=self._set_sound, options=["off", "on"]),
            Item("Time of day", get=lambda: self.ctx.settings.time_of_day, set=self._set_time,
                 options=atmosphere.TIMES),
            Item("Change city", on_select=self._cities),
            Item("Mission select", on_select=self._missions),
            Item("Main menu", on_select=app.to_menu),
            Item("Quit", on_select=app.quit),
        ], footer="ESC resume    UP/DOWN select    ENTER confirm")

    def _restart(self):
        self.ctx.reset_flight()
        self.app.pop()

    def _set_mode(self, v):
        self.ctx.settings.mode = v
        self.ctx.push_tuning()

    def _set_sound(self, v):
        self.ctx.settings.muted = not bool(v)
        self.ctx.push_tuning()

    def _set_time(self, v):
        self.ctx.settings.time_of_day = int(v)
        self.ctx.apply_time_of_day()

    def _cities(self):
        self.app.to_menu()
        self.app.push(CitySelectState(self.app))

    def _missions(self):
        self.app.to_menu()
        self.app.push(MissionSelectState(self.app))

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self.app.pop()
            return
        self.menu.handle_event(e)

    def update(self, dt):
        self.t += dt

    def draw(self, surf, dt):
        self.app.stack[-2].draw(surf, 0.0)         # frozen flight view underneath
        dim = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 16, 140))
        surf.blit(dim, (0, 0))
        self.menu.draw(surf, self.t)


class ResultState(State):
    """Mission complete / failed overlay with star reveal and next-step buttons."""

    def __init__(self, app, flight: FlightState):
        super().__init__(app)
        self.flight, self.t = flight, 0.0
        run = self.ctx.runner
        self.run, self.m, self.res = run, run.m, run.result or {}
        self.won = run.phase is Phase.COMPLETE
        self.new_best = self.ctx.last_new_best if self.won else False
        self._stars_shown = 0
        idx = next((i for i, x in enumerate(MISSIONS) if x.id == self.m.id), 0)
        nxt = MISSIONS[idx + 1] if idx + 1 < len(MISSIONS) else None
        items = [Item("Retry", on_select=self._retry)]
        if self.won and nxt is not None:
            items.append(Item(f"Next: {nxt.title}", on_select=lambda: self._next(nxt)))
        items += [Item("Mission select", on_select=self._missions), Item("Main menu", on_select=app.to_menu)]
        self.menu = Menu("MISSION COMPLETE" if self.won else "MISSION FAILED", items,
                         subtitle=self.m.title, header_extra=118, decor=self._decor,
                         footer="R retry    ENTER confirm    ESC main menu")

    def _retry(self):
        self.app.pop()
        self.flight._restart()

    def _next(self, nxt):
        self.app.pop()
        self.app.stack[-1] = FlightState(self.app, nxt)

    def _missions(self):
        self.app.to_menu()
        self.app.push(MissionSelectState(self.app))

    def _decor(self, surf, cx, top, pw):
        res, m = self.res, self.m
        if self.won:
            n = int(self.t / 0.45)
            for k in range(3):
                draw_star(surf, (cx - 70 + k * 70, top + 28), 26, k < min(n, res.get("stars", 0)))
            if n > self._stars_shown and self._stars_shown < res.get("stars", 0):
                self._stars_shown = min(n, res.get("stars", 0))
                self.ctx.audio.play("star")
            main = f"{res.get('score', 0)} points" if m.kind == "attack" else fmt_time(res.get("time"))
            theme.text(surf, main, (cx, top + 62), 28, theme.OSD, True, "midtop")
            sub = f"score {res.get('score', 0)}" if m.kind != "attack" else f"{self.run.field.hits} targets popped"
            if self.new_best:
                sub += "     NEW BEST!"
            theme.text(surf, sub, (cx, top + 98), 15, theme.GOOD if self.new_best else theme.OSD_DIM, True, "midtop")
        else:
            reason = self.run.fail_reason or "FAILED"
            theme.text(surf, reason, (cx, top + 18), 32, theme.DANGER, True, "midtop")
            if m.kind in ("targets", "attack"):
                prog = f"{self.run.field.hits} targets popped  -  score {res.get('score', 0)}"
            elif m.kind == "race":
                prog = f"{self.run.tracker.gates_passed} gates cleared"
            else:
                prog = ""
            theme.text(surf, prog, (cx, top + 66), 16, theme.OSD_DIM, False, "midtop")
            theme.text(surf, "Tip: ANGLE mode (M) self-levels and is more forgiving", (cx, top + 92), 13,
                       theme.OSD_DIM, False, "midtop")

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            self._retry()
            return
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self.app.to_menu()
            return
        self.menu.handle_event(e)

    def update(self, dt):
        self.t += dt
        self.ctx.effects.update(dt)
        self.ctx.hud.toasts.update(dt)

    def draw(self, surf, dt):
        self.flight.draw(surf, 0.0)
        dim = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 16, 150))
        surf.blit(dim, (0, 0))
        self.menu.draw(surf, self.t)


class App:
    def __init__(self, settings: Settings | None = None, auto_build: bool = True, headless: bool = False):
        pygame.init()
        pygame.display.set_caption("FPV Sim")
        self.settings = settings or Settings.load()
        flags = pygame.RESIZABLE | (pygame.FULLSCREEN if self.settings.fullscreen else 0)
        try:
            self.screen = pygame.display.set_mode(SIZE, flags, vsync=1)
        except pygame.error:
            self.screen = pygame.display.set_mode(SIZE, flags)
        self.clock = pygame.time.Clock()
        self.ctx = Context(self.settings, auto_build)
        self.stack: list[State] = []
        self.running = True
        self._slow_frames = 0
        self.push(MenuState(self))

    # --- state stack -------------------------------------------------------
    def push(self, s: State) -> None:
        self.stack.append(s)

    def pop(self) -> None:
        if len(self.stack) > 1:
            self.stack.pop()

    def to_menu(self) -> None:
        del self.stack[1:]
        self.ctx.reset_flight()          # disarms, stops the autopilot, restarts the mission
        self.ctx.audio.silence()

    def quit(self) -> None:
        self.running = False

    # --- loop --------------------------------------------------------------------
    def _adapt_quality(self, frame_ms: float) -> None:
        """Lower graphics quality automatically if the machine can't hold ~30 fps."""
        self._slow_frames = self._slow_frames + 1 if frame_ms > 34.0 else max(0, self._slow_frames - 1)
        if self._slow_frames > 90 and self.settings.quality > 0:
            self.settings.quality -= 1
            self.ctx.scene.set_quality(self.settings.quality)
            self.ctx.hud.toasts.add("Graphics quality lowered for performance", theme.WARN, 2.5)
            self._slow_frames = 0

    def tick(self, dt: float) -> None:
        t0 = time.perf_counter()
        surf = pygame.display.get_surface()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()
                self.settings.fullscreen = not self.settings.fullscreen
            else:
                self.stack[-1].handle_event(e)
        top = self.stack[-1]
        top.update(dt)
        top.draw(surf, dt)
        pygame.display.flip()
        self._adapt_quality((time.perf_counter() - t0) * 1000.0)

    def run(self) -> None:
        try:
            while self.running:
                dt = min(self.clock.tick(self.settings.fps_cap) / 1000.0, 0.05)
                self.tick(dt)
        finally:
            self.settings.save()
            self.ctx.audio.silence()
            self.ctx.telemetry.toggle_recording() if self.ctx.telemetry.recording else None
            pygame.quit()
