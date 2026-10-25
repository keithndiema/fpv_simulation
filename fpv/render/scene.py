"""World renderer: sky + sun + soft clouds, horizon mountains, lit / fogged / shadowed terrain, water with sun glint,
forests of real species, the city layer (buildings, roads, traffic, landmarks), race gates, bullseye targets, the
launch pad, particles and the drone model.

Everything is a painter's-algorithm software renderer: geometry is projected / shaded / fogged with a few
vectorised numpy calls per frame, then filled polygons are drawn far-to-near. Two terrain levels of detail
(coarse far layer, fine near layer) keep the polygon count manageable.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from ..game.course import GATE_SIZE, Course, default_terrain
from ..game.terrain import (ACACIA, BANANA, BROADLEAF, CONIFER, EUCALYPTUS, FLOWERING, PALM, Terrain)
from ..world.noise import box_blur, fbm
from . import atmosphere, theme
from .atmosphere import Atmosphere
from .camera import NEAR, View, quat_to_matrix
from .city import K_BUILDING, CityLayer

# ---------------------------------------------------------------------------- palette (legacy valley + water)
WATER_DEEP = (22, 74, 124)
WATER_SHALLOW = (52, 118, 146)
SAND = (206, 190, 140)
GRASS_LOW = (58, 120, 46)
GRASS_HIGH = (104, 142, 62)
ROCK = (122, 114, 104)
SNOW = (240, 244, 248)

FAR_STRIDE = 3                      # coarse layer: 15 m cells
NEAR_STRIDE = 1                     # fine layer:    5 m cells
# quality -> (far radius in coarse cells, near radius in fine cells, tree draw distance, building distance,
#             max buildings per frame, road distance)
QUALITY = {0: (13, 11, 150.0, 150.0, 110, 150.0), 1: (18, 14, 200.0, 220.0, 190, 220.0),
           2: (21, 17, 230.0, 300.0, 300, 300.0)}

_BOX_CORNERS = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], dtype=float)
_FACES = [((0, 2, 6, 4), (-1, 0, 0)), ((1, 3, 7, 5), (1, 0, 0)), ((0, 1, 5, 4), (0, -1, 0)),
          ((2, 3, 7, 6), (0, 1, 0)), ((4, 5, 7, 6), (0, 0, 1))]

# ground-cover code -> base colour (see world/generator.py: GRASS .. PLANTATION)
_COVER_COL = np.array([
    (104, 140, 62),    # 0 grass
    (88, 150, 58),     # 1 green crops
    (178, 158, 92),    # 2 dry / harvested fields
    (172, 152, 124),   # 3 urban dust and concrete
    (168, 96, 64),     # 4 laterite (red earth)
    (74, 130, 98),     # 5 papyrus swamp
    (44, 120, 50),     # 6 tea
    (128, 120, 112),   # 7 rock
    (208, 192, 144),   # 8 sand
    (84, 148, 62),     # 9 lawn
    (50, 112, 46),     # 10 banana plantation
], float)
_GRASS_BY_STYLE = {"savanna": (168, 154, 84), "semiarid": (182, 148, 96), "cropland": (118, 146, 66),
                   "banana": (98, 140, 60), "tea": (96, 140, 64)}


def _mix(a, b, t):
    return a + (b - a) * t


def shadow_map(T: Terrain, atm: Atmosphere) -> np.ndarray:
    """Light factor per terrain cell (1 = sunlit, ~0.45 = in the shadow of a hill or building).

    Marches a ray from every cell towards the sun and checks it against terrain + rooftop heights.
    """
    n = T.n
    H = T.heights.astype(np.float64)
    world = T.world
    roof = np.zeros_like(H)
    if world is not None and world.occ_h is not None:
        oh = world.occ_h
        r = oh.shape[0] // n if oh.shape[0] >= n else 1
        # fold the 2.5 m rooftop raster down to the 5 m terrain grid (max over each 2x2)
        m = (oh.shape[0] // 2) * 2
        f = np.maximum.reduce([oh[0:m:2, 0:m:2], oh[1:m:2, 0:m:2], oh[0:m:2, 1:m:2], oh[1:m:2, 1:m:2]])
        roof[:f.shape[0], :f.shape[1]] = f[:n, :n]
    top = H + roof
    sun = atm.sun_dir
    el = max(math.asin(float(sun[2])), math.radians(1.5))
    hx, hy = float(sun[0]), float(sun[1])
    hn = math.hypot(hx, hy) + 1e-9
    hx, hy = hx / hn, hy / hn
    tan_el = math.tan(el)
    reach = min(260.0, 75.0 / tan_el)
    jj, ii = np.mgrid[0:n, 0:n]
    x = T.ox + ii * T.cell
    y = T.oy + jj * T.cell
    shadow = np.zeros((n, n), bool)
    step = 5.0
    t = step
    while t < reach:
        sx, sy = x + hx * t, y + hy * t
        si = np.clip(np.rint((sx - T.ox) / T.cell).astype(int), 0, n - 1)
        sj = np.clip(np.rint((sy - T.oy) / T.cell).astype(int), 0, n - 1)
        shadow |= top[sj, si] > (H + roof * 0.0 + t * tan_el + 0.4)
        t += step
    soft = box_blur(shadow.astype(float), 2)
    return 1.0 - 0.58 * soft


class _Shaded:
    """Pre-lit, pre-coloured quad grid for one terrain level of detail."""

    def __init__(self, T: Terrain, stride: int, atm: Atmosphere, light: np.ndarray, style: str):
        H = T.heights[::stride, ::stride].astype(np.float64)
        self.stride, self.cell = stride, T.cell * stride
        self.n = H.shape[0]
        self.H = np.maximum(H, T.water_level)                       # water is a flat plane at the water level
        h00, h10, h01, h11 = H[:-1, :-1], H[:-1, 1:], H[1:, :-1], H[1:, 1:]
        hq = (h00 + h10 + h01 + h11) / 4
        gx = ((h10 - h00) + (h11 - h01)) / (2 * self.cell)
        gy = ((h01 - h00) + (h11 - h10)) / (2 * self.cell)
        inv = 1 / np.sqrt(gx * gx + gy * gy + 1)
        sun = atm.sun_dir
        lam = np.clip((-gx * sun[0] - gy * sun[1] + sun[2]) * inv, 0, 1)
        slope = np.hypot(gx, gy)
        wet = hq < T.water_level + 0.05
        rng = np.random.default_rng(stride)
        # Ground colour variation at three scales (broad patches, medium mottling, fine grain). Pure white noise
        # per quad reads as a checkerboard; real fields vary smoothly with a little grain on top.
        def z(a):
            return (a - a.mean()) / (a.std() + 1e-9)
        white = rng.random(hq.shape)
        broad = box_blur(rng.random(hq.shape), max(2, int(30 / self.cell)))
        mid = box_blur(rng.random(hq.shape), max(1, int(9 / self.cell)))
        tone = np.clip(0.5 + 0.17 * (0.75 * z(broad) + 0.5 * z(mid) + 0.45 * z(white)), 0.0, 1.0)
        if T.landcover is not None:
            col = self._cover_colours(T, stride, hq, slope, tone, style)
        else:
            col = self._legacy_colours(hq, slope, tone)
        # light = ambient sky + direct sun (coloured, shadowed)
        lm = light[::stride, ::stride]
        sh = (lm[:-1, :-1] + lm[1:, 1:]) * 0.5
        lit = atm.amb_col[None, None, :] + atm.sun_col[None, None, :] * (lam * sh)[..., None]
        col = col * lit * self._contact_shading(T, stride, hq)[..., None]
        depth = np.clip(-(h00 + h10 + h01 + h11) / 4 / 3.0, 0, 1)
        deep = _mix(np.array(WATER_SHALLOW, float), np.array(WATER_DEEP, float), depth[..., None])
        water = deep * (0.85 + 0.20 * tone[..., None]) * (atm.amb_col + atm.sun_col * 0.35)[None, None, :]
        col = np.where(wet[..., None], water, col)
        self.rgb = np.clip(col, 0, 255)
        self.wet = wet
        self.tone = tone

    def _contact_shading(self, T: Terrain, stride: int, hq: np.ndarray) -> np.ndarray:
        """Ambient occlusion on the ground: soil next to a wall gets less sky light, so it is darker.

        Blur the building raster (1 = covered by a roof) and darken ground cells in proportion to how much of their
        neighbourhood is a building - but not the cells under the building itself (they are hidden anyway).
        """
        w = getattr(T, "world", None)
        out = np.ones(hq.shape)
        if w is None or getattr(w, "occ_h", None) is None:
            return out
        occ = (w.occ_h > 0.5).astype(np.float64)
        cell = w.occ_cell
        ao = box_blur(occ, max(1, int(round(4.5 / cell))))                  # ~4.5 m radius
        n = hq.shape[0]
        j, i = np.mgrid[0:n, 0:hq.shape[1]]
        x = T.ox + (i + 0.5) * self.cell
        y = T.oy + (j + 0.5) * self.cell
        ii = np.clip(((x - T.ox) / cell).astype(int), 0, ao.shape[1] - 1)
        jj = np.clip(((y - T.oy) / cell).astype(int), 0, ao.shape[0] - 1)
        under = occ[jj, ii] > 0.5
        out = 1.0 - 0.34 * np.clip(ao[jj, ii] * 1.7, 0, 1) * (~under)
        return out

    @staticmethod
    def _legacy_colours(hq, slope, tone):
        col = np.empty(hq.shape + (3,))
        lush = np.clip((hq - 1.5) / 30.0, 0, 1)[..., None]
        grass = _mix(np.array(GRASS_LOW, float), np.array(GRASS_HIGH, float), lush)
        col[:] = grass * (0.90 + 0.20 * tone[..., None])
        col = np.where((hq < 2.0)[..., None], np.array(SAND, float) * (0.93 + 0.1 * tone[..., None]), col)
        rocky = np.clip((slope - 0.42) / 0.25, 0, 1)[..., None]
        col = _mix(col, np.array(ROCK, float) * (0.85 + 0.3 * tone[..., None]), rocky)
        high = np.clip((hq - 70) / 50.0, 0, 1)[..., None]
        col = _mix(col, np.array(ROCK, float), np.clip(high * 1.4, 0, 1))
        snow = np.clip((hq - 118) / 25.0 + (0.3 - slope) * 0.3, 0, 1)[..., None] * (hq > 110)[..., None]
        return _mix(col, np.array(SNOW, float), snow)

    @staticmethod
    def _cover_colours(T, stride, hq, slope, tone, style):
        cov = T.landcover[::stride, ::stride]
        # quad colour comes from the cover at the quad's lower-left corner (good enough at 5 m / 15 m)
        c = cov[:-1, :-1]
        pal = _COVER_COL.copy()
        pal[0] = _GRASS_BY_STYLE.get(style, pal[0])
        pal[2] = np.array((176, 156, 92)) if style != "semiarid" else np.array((190, 160, 110))
        col = pal[c]
        # mottling: fields are never one flat colour; hills get a little lighter and drier with height
        col = col * (0.88 + 0.24 * tone[..., None])
        dry = np.clip((hq - 40) / 120.0, 0, 0.35)[..., None]
        col = _mix(col, np.array((150, 140, 100), float), dry * (c[..., None] == 0))
        rocky = np.clip((slope - 0.45) / 0.25, 0, 1)[..., None]
        return _mix(col, np.array(ROCK, float) * (0.85 + 0.3 * tone[..., None]), rocky)


class WorldRenderer:
    def __init__(self, course: Course):
        self.t = 0.0
        self._terrain: Terrain | None = None
        self._grids: dict[int, _Shaded] = {}
        self.city: CityLayer | None = None
        self.light: np.ndarray | None = None
        self.time_of_day = "Midday"
        self.atm = atmosphere.make("Midday", 0.2)
        self.set_quality(2)
        self._cloud_sprites: list[pygame.Surface] = []
        self._cloud_cache: dict = {}
        self._clouds = self._make_clouds()
        self._sun_glow = self._make_sun_glow()
        self._pad_cache = None
        self.set_course(course)

    # ------------------------------------------------------------------ setup
    def set_quality(self, q: int) -> None:
        self.quality = max(0, min(2, int(q)))
        (self.far_r, self.near_r, self.tree_far, self.bld_far, self.bld_max, self.road_far) = QUALITY[self.quality]
        self.tree_near = self.near_r * 5.0

    def set_time_of_day(self, name: str) -> None:
        self.time_of_day = name if name in atmosphere.TIMES else "Midday"
        haze = self._world().city.haze if self._world() is not None else 0.2
        self.atm = atmosphere.make(self.time_of_day, haze)
        self._rebuild_lighting()

    def _world(self):
        return getattr(self._terrain, "world", None) if self._terrain is not None else None

    def _style(self) -> str:
        w = self._world()
        return w.city.cover if w is not None else "banana"

    def _rebuild_lighting(self) -> None:
        T = self._terrain
        if T is None:
            return
        self.light = shadow_map(T, self.atm)
        style = self._style()
        self._grids = {FAR_STRIDE: _Shaded(T, FAR_STRIDE, self.atm, self.light, style),
                       NEAR_STRIDE: _Shaded(T, NEAR_STRIDE, self.atm, self.light, style)}
        self._cloud_sprites = []
        self._cloud_cache = {}
        self._sun_glow = self._make_sun_glow()
        if self.city is not None:
            self.city.set_atmosphere(self.atm)

    def set_course(self, course: Course) -> None:
        self.course = course
        T = course.terrain or default_terrain()
        new_terrain = T is not self._terrain
        if new_terrain:
            self._terrain = T
            w = self._world()
            self.atm = atmosphere.make(self.time_of_day, w.city.haze if w is not None else 0.2)
            self.city = CityLayer(w, self.atm) if w is not None else None
            self._rebuild_lighting()
            self._pad_cache = None
        self.terrain = T
        tr = T.trees
        self._tp, self._th, self._tr, self._tt, self._tk = tr.pos, tr.height, tr.radius, tr.tone, tr.kind
        # legacy boxes (pylons / blocks of the practice valley)
        self._boxes = []
        rng = np.random.default_rng(4)
        for cx, cy, cz, hx, hy, hz, yaw in course.boxes:
            c, s = math.cos(yaw), math.sin(yaw)
            local = _BOX_CORNERS * np.array([hx, hy, hz])
            world = np.stack([c * local[:, 0] - s * local[:, 1] + cx, s * local[:, 0] + c * local[:, 1] + cy,
                              local[:, 2] + cz], axis=1)
            normals = [np.array([c * nx - s * ny, s * nx + c * ny, nz]) for _, (nx, ny, nz) in _FACES]
            if hx < 0.9 and hy < 1.6 and hz > 1.4:
                tint = (206, 84, 70) if rng.random() < 0.5 else (205, 205, 210)
            else:
                tint = [(196, 190, 178), (172, 104, 92), (158, 170, 184), (188, 178, 140)][int(rng.integers(0, 4))]
            self._boxes.append((np.array([cx, cy, cz]), world, normals, tint, hx + hy))

    # ------------------------------------------------------------------ sky assets
    def _make_clouds(self):
        rng = np.random.default_rng(12)
        out = []
        for _ in range(18):
            az, el = rng.uniform(0, 2 * math.pi), rng.uniform(0.10, 0.62)
            out.append((az, el, rng.uniform(0.10, 0.22), int(rng.integers(0, 6))))
        return out

    def _build_cloud_sprites(self) -> None:
        """Soft cumulus sprites: fbm density with a flat shaded base and bright sunlit top."""
        tint = np.array(self.atm.cloud_tint, float)
        shade = np.array(self.atm.haze, float)
        self._cloud_sprites = []
        for k in range(6):
            w, h = 256, 112
            yy, xx = np.mgrid[0:h, 0:w]
            u, v = xx / w, yy / h
            dens = fbm(u * 5.0 + k * 7.3, v * 3.2 + k * 3.1, 40 + k, 4, gain=0.5)
            ell = 1.0 - np.clip(((u - 0.5) / 0.48) ** 2 + ((v - 0.60) / 0.44) ** 2, 0, 1)
            a = np.clip((dens * 1.25 + ell * 1.1 - 1.05) / 0.28, 0, 1)
            a *= np.clip((0.97 - v) / 0.18, 0, 1)                               # flat-ish bottom
            lit = np.clip(1.08 - v * 0.85 + (dens - 0.5) * 0.5, 0, 1.1)[..., None]
            rgb = shade * (1 - lit) * 0.55 + tint * lit
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            px = pygame.surfarray.pixels3d(surf)
            px[:] = np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 0, 2)
            del px
            al = pygame.surfarray.pixels_alpha(surf)
            al[:] = (a * 235).astype(np.uint8).T
            del al
            self._cloud_sprites.append(surf)

    def _make_sun_glow(self) -> pygame.Surface:
        n = 360
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
        a = np.clip(1 - r, 0, 1) ** 2.4
        core = (r < 0.055).astype(float)
        sc = np.array(self.atm.sun_col, float) if hasattr(self, "atm") else np.ones(3)
        rgb = np.stack([a * 120 * sc[0] + core * 135, a * 95 * sc[1] + core * 135 * min(1, sc[1]),
                        a * 52 * sc[2] + core * 135 * min(1, sc[2])], axis=-1).clip(0, 255)
        return pygame.surfarray.make_surface(rgb.astype(np.uint8).transpose(1, 0, 2))

    # ------------------------------------------------------------------ helpers
    def _fog_color(self, col, d):
        a = self.atm
        f = np.clip((d - a.fog_start) / (a.fog_end - a.fog_start), 0, 1) ** 1.15
        haze = np.array(a.haze, float)
        return col + (haze - col) * f[..., None] if np.ndim(f) else \
            tuple(int(c + (h - c) * f) for c, h in zip(col, haze))

    @staticmethod
    def _clip_near(cam: np.ndarray) -> np.ndarray:
        out = []
        n = len(cam)
        for i in range(n):
            a, b = cam[i], cam[(i + 1) % n]
            ain, bin_ = a[0] >= NEAR, b[0] >= NEAR
            if ain:
                out.append(a)
            if ain != bin_:
                t = (NEAR - a[0]) / (b[0] - a[0])
                out.append(a + (b - a) * t)
        return np.array(out)

    @staticmethod
    def _to_screen(v: View, cam: np.ndarray) -> np.ndarray:
        x = np.maximum(cam[..., 0], NEAR)
        return np.clip(np.stack([v.cx - v.f * cam[..., 1] / x, v.cy - v.f * cam[..., 2] / x], axis=-1), -6000, 6000)

    def _light_at(self, x, y):
        """Terrain light factor (shadow) at world points; used to shade road ribbons."""
        T = self._terrain
        i = np.clip(np.rint((np.asarray(x) - T.ox) / T.cell).astype(int), 0, T.n - 1)
        j = np.clip(np.rint((np.asarray(y) - T.oy) / T.cell).astype(int), 0, T.n - 1)
        return self.light[j, i]

    # ------------------------------------------------------------------ sky
    def _sky(self, surf: pygame.Surface, v: View) -> None:
        a = self.atm
        w, h = surf.get_size()
        u = v.R[2]                                  # world-up expressed in camera axes (forward, left, up)
        ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
        n = math.hypot(uy, uz)
        surf.fill(a.zenith)
        if n < 1e-5:
            surf.fill(a.mid if ux > 0 else a.haze)
            return
        nh = np.array([uy, uz]) / n                 # screen-space direction of "up"
        th = np.array([-nh[1], nh[0]])
        c = np.array([v.cx, v.cy])
        d0 = v.f * ux / n                           # signed screen distance of the true horizon from centre
        L = 4.0 * max(w, h)

        # NB: nh is the screen-space direction of world-up *as a pygame coordinate*, i.e. it points DOWN the screen
        # when the camera is level. The horizon line sits at  c + nh * d0;  the sky lies on the -nh side of it.
        def strip(off0, off1, color):
            """Band of sky between two lines parallel to the horizon, `off` px above it."""
            b0, b1 = c + nh * (d0 - off0), c + nh * (d0 - off1)
            pts = [b0 + th * L, b0 - th * L, b1 - th * L, b1 + th * L]
            pygame.draw.polygon(surf, color, [tuple(map(int, p)) for p in pts])

        # Why this curve?  Real clear skies are brightest and palest at the horizon (light has crossed the most air
        # there) and deepen towards the zenith.  Colour is therefore a function of the *elevation angle* above the
        # horizon, not of screen position: an exponential falloff from the haze colour to the mid-sky colour, then a
        # smooth blend to the zenith.  44 bands whose steps are equal in angle keep the gradient free of visible banding.
        bands = 44
        top = 1.35                                   # radians above the horizon covered by the gradient
        ang = [top * (i / bands) ** 1.7 for i in range(bands + 1)]
        offs = [v.f * math.tan(x) for x in ang[:-1]] + [L]
        for i in range(bands):
            th_mid = 0.5 * (ang[i] + ang[i + 1])
            s1 = 1.0 - math.exp(-th_mid / 0.17)                       # haze -> mid sky
            s2 = min(1.0, max(0.0, (th_mid - 0.12) / 1.1)) ** 1.4     # mid sky -> zenith
            col = theme.lerp(theme.lerp(a.haze, a.mid, s1), a.zenith, s2)
            strip(offs[i], offs[i + 1], col)
        b0 = c + nh * d0
        pts = [b0 + th * L, b0 - th * L, b0 - th * L + nh * L, b0 + th * L + nh * L]
        pygame.draw.polygon(surf, a.haze, [tuple(map(int, p)) for p in pts])      # ground haze below the horizon

    def _sun(self, surf: pygame.Surface, v: View) -> None:
        cam = self.atm.sun_dir @ v.R
        if cam[0] < 0.05:
            return
        x, y = v.cx - v.f * cam[1] / cam[0], v.cy - v.f * cam[2] / cam[0]
        g = self._sun_glow
        k = max(0.55, min(1.6, v.f / 520.0)) * (1.25 if self.atm.sun_el < 15 else 1.0)
        glow = pygame.transform.smoothscale(g, (int(g.get_width() * k), int(g.get_height() * k)))
        surf.blit(glow, glow.get_rect(center=(int(x), int(y))), special_flags=pygame.BLEND_RGB_ADD)

    def _clouds_draw(self, surf: pygame.Surface, v: View) -> None:
        if not self._cloud_sprites:
            self._build_cloud_sprites()
        for az0, el, size, sid in self._clouds:
            az = az0 + self.t * 0.004
            d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            cam = d @ v.R
            if cam[0] < 0.2:
                continue
            x, y = v.cx - v.f * cam[1] / cam[0], v.cy - v.f * cam[2] / cam[0]
            wpx = int(v.f * size * 3.4 / max(0.4, cam[0]))
            if wpx < 14 or not (-wpx < x < v.width + wpx and -wpx < y < v.height + wpx):
                continue
            hpx = max(6, int(wpx * 0.44 * (0.55 + el * 0.9)))
            key = (sid, wpx // 12, hpx // 6)
            spr = self._cloud_cache.get(key)
            if spr is None:
                spr = pygame.transform.smoothscale(self._cloud_sprites[sid], (key[1] * 12 or 12, key[2] * 6 or 6))
                if len(self._cloud_cache) > 160:
                    self._cloud_cache.clear()
                self._cloud_cache[key] = spr
            surf.blit(spr, spr.get_rect(center=(int(x), int(y))))

    def _mountains(self, surf: pygame.Surface, v: View) -> None:
        T = self.terrain
        a = self.atm
        layers = T.backdrop
        if layers is None:                                         # the classic valley's two silhouettes
            layers = [{"az": az, "el": el, "rgb": col, "snow": None}
                      for (az, el), col in zip(T.ridges, ((136, 160, 188), (160, 182, 206)))]
        tint = np.clip(a.amb_col * 1.55 + a.sun_col * 0.35, 0.42, 1.15)
        haze = np.array(a.haze, float)
        for li, lay in enumerate(layers):
            az, el = lay["az"], lay["el"]
            col = np.clip(np.array(lay["rgb"], float) * tint * 0.8 + haze * 0.26 + (0 if li else 0), 0, 255)
            col = tuple(int(c) for c in col)
            top = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
            bot = np.stack([np.full_like(az, math.cos(-0.12)) * np.cos(az), np.full_like(az, math.cos(-0.12)) * np.sin(az),
                            np.full_like(az, math.sin(-0.12))], axis=1)
            ct, cb = top @ v.R, bot @ v.R
            ok = (ct[:, 0] > 0.08) & (cb[:, 0] > 0.02)
            if not ok.any():
                continue
            st, sb = self._to_screen(v, ct), self._to_screen(v, cb)
            snow = lay.get("snow")
            snow_col = tuple(int(min(255, c * m)) for c, m in zip((240, 244, 250), np.clip(a.sun_col * 0.9 + 0.12, 0.5, 1.1)))
            # one polygon per contiguous run of visible azimuths (far cheaper than a quad per sample)
            idx = np.nonzero(ok)[0]
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                if len(run) < 2:
                    continue
                if st[run, 0].max() < 0 or st[run, 0].min() > v.width:
                    continue
                pts = [(int(x), int(y)) for x, y in st[run]] + [(int(x), int(y)) for x, y in sb[run][::-1]]
                pygame.draw.polygon(surf, col, pts)
                if snow is not None and (el[run] > snow[run]).any():
                    lo_el = np.minimum(el[run], snow[run])
                    lo = np.stack([np.cos(lo_el) * np.cos(az[run]), np.cos(lo_el) * np.sin(az[run]), np.sin(lo_el)], axis=1) @ v.R
                    if (lo[:, 0] > 0.08).all():
                        sl = self._to_screen(v, lo)
                        cap = np.nonzero(el[run] > snow[run])[0]
                        for cr in np.split(cap, np.nonzero(np.diff(cap) > 1)[0] + 1):
                            if len(cr) < 2:
                                continue
                            pygame.draw.polygon(surf, snow_col, [(int(x), int(y)) for x, y in st[run][cr]] +
                                                [(int(x), int(y)) for x, y in sl[cr][::-1]])

    # ------------------------------------------------------------------ terrain
    def _terrain_quads(self, v: View, grid: _Shaded, R: int, max_dist: float | None = None):
        """Project one terrain level; returns (keys, polygons, colours) for every visible quad."""
        T = self.terrain
        ci = int(round((v.pos[0] - T.ox) / grid.cell))
        cj = int(round((v.pos[1] - T.oy) / grid.cell))
        i0, i1 = max(0, ci - R), min(grid.n - 1, ci + R)
        j0, j1 = max(0, cj - R), min(grid.n - 1, cj + R)
        if i1 - i0 < 1 or j1 - j0 < 1:
            return np.empty(0), [], np.empty((0, 3))
        xs = T.ox + np.arange(i0, i1 + 1) * grid.cell
        ys = T.oy + np.arange(j0, j1 + 1) * grid.cell
        Hs = grid.H[j0:j1 + 1, i0:i1 + 1]
        P = np.stack([np.broadcast_to(xs[None, :], Hs.shape), np.broadcast_to(ys[:, None], Hs.shape), Hs], axis=-1)
        cam = (P - v.pos) @ v.R
        scr = self._to_screen(v, cam)
        behind = cam[..., 0] < NEAR
        c4 = np.stack([cam[:-1, :-1], cam[:-1, 1:], cam[1:, 1:], cam[1:, :-1]], axis=2)
        s4 = np.stack([scr[:-1, :-1], scr[:-1, 1:], scr[1:, 1:], scr[1:, :-1]], axis=2)
        nb = np.stack([behind[:-1, :-1], behind[:-1, 1:], behind[1:, 1:], behind[1:, :-1]], axis=2).sum(axis=2)
        w, h = v.width, v.height
        minx, maxx = s4[..., 0].min(axis=2), s4[..., 0].max(axis=2)
        miny, maxy = s4[..., 1].min(axis=2), s4[..., 1].max(axis=2)
        onscreen = (maxx > 0) & (minx < w) & (maxy > 0) & (miny < h)
        centre = c4.mean(axis=2)
        dist = np.linalg.norm(centre, axis=2)
        rgb = grid.rgb[j0:j1, i0:i1]
        vis = onscreen & (nb == 0)
        if max_dist is not None:
            # the fine layer is circular: anything beyond it is painted by the coarse layer (so far objects that
            # were drawn before the fine layer can never be overpainted by its square corners)
            vis &= dist <= max_dist
        keys = dist[vis]
        base = rgb[vis]
        wetq = grid.wet[j0:j1, i0:i1][vis]
        if wetq.any():
            base = self._water_shine(base, wetq, centre[vis], grid.tone[j0:j1, i0:i1][vis], v)
        # water keeps its colour at a distance (real lakes read as dark blue-grey against the pale horizon haze), so it
        # is fogged as if it were 40% nearer
        cols = self._fog_color(base, np.where(wetq, keys * 0.6, keys))
        polys = s4[vis].astype(int).tolist()
        strad = (nb > 0) & (nb < 4)
        if max_dist is not None:
            strad &= dist <= max_dist
        if strad.any():
            xmax = c4[..., 0].max(axis=2)
            lim = 2.0 * xmax + 2.0
            strad &= (np.abs(c4[..., 1]).min(axis=2) < lim) & (np.abs(c4[..., 2]).min(axis=2) < lim)
        if strad.any():
            ex_k, ex_c, ex_p = [], [], []
            for r_, c_ in zip(*np.nonzero(strad)):
                poly = self._clip_near(c4[r_, c_])
                if len(poly) >= 3:
                    sp = self._to_screen(v, poly)
                    ex_p.append(sp.astype(int).tolist())
                    d = float(np.linalg.norm(centre[r_, c_]))
                    ex_k.append(d)
                    ex_c.append(self._fog_color(rgb[r_, c_], np.array([d]))[0])
            if ex_k:
                keys = np.concatenate([keys, ex_k])
                cols = np.concatenate([cols, np.array(ex_c)])
                polys += ex_p
        return keys, polys, cols

    def _water_shine(self, base, wetq, centre, tone, v: View):
        """Sky reflection (stronger at grazing angles) + a sparkling sun glint on water quads."""
        a = self.atm
        base = base.copy()
        d = centre[wetq]
        dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        up = v.R[2]
        sun_c = a.sun_dir @ v.R
        cosz = np.abs(dn @ up)
        fres = (1.0 - cosz) ** 3
        sky = np.array(a.mid, float)
        out = base[wetq] * (1 - 0.5 * fres)[:, None] + sky * (0.5 * fres)[:, None]
        refl = dn - 2.0 * (dn @ up)[:, None] * up[None, :]
        g = np.clip(refl @ sun_c, 0, 1) ** 36
        twinkle = 0.55 + 0.45 * np.sin(self.t * 2.7 + tone[wetq] * 60.0)
        out = out + (np.array(a.sun_col, float) * 255.0 * 0.9)[None, :] * (g * twinkle)[:, None]
        base[wetq] = out
        return base

    # ------------------------------------------------------------------ scene objects: trees
    _TREE_BASE = {CONIFER: (34, 88, 40), BROADLEAF: (52, 112, 44), ACACIA: (100, 128, 52), EUCALYPTUS: (78, 116, 90),
                  PALM: (66, 122, 58), BANANA: (96, 154, 56), FLOWERING: (52, 112, 44)}

    def _collect_trees(self, v: View):
        d_xy = np.hypot(self._tp[:, 0] - v.pos[0], self._tp[:, 1] - v.pos[1])
        near = d_xy < self.tree_far
        if not near.any():
            return []
        idx = np.nonzero(near)[0]
        cam = (self._tp[idx] - v.pos) @ v.R
        x = cam[:, 0]
        ok = x > 0.6
        idx, cam, x = idx[ok], cam[ok], x[ok]
        if len(idx) == 0:
            return []
        f = v.f
        bx, by = v.cx - f * cam[:, 1] / x, v.cy - f * cam[:, 2] / x
        hp, wp = f * self._th[idx] / x, f * self._tr[idx] / x
        ok = (bx + wp > 0) & (bx - wp < v.width) & (by - hp < v.height) & (by > -20) & (hp > 2.0)
        idx, bx, by, hp, wp, x = idx[ok], bx[ok], by[ok], hp[ok], wp[ok], x[ok]
        dist = np.linalg.norm(cam[ok], axis=1)
        tone = self._tt[idx]
        kind = self._tk[idx]
        base = np.array([self._TREE_BASE[int(k)] for k in kind], float) if len(kind) else np.zeros((0, 3))
        flow = kind == FLOWERING
        if flow.any():                                  # jacaranda (purple) vs flame tree (red-orange)
            base[flow & (tone < 0.5)] = (150, 104, 190)
            base[flow & (tone >= 0.5)] = (214, 84, 44)
        a = self.atm
        lit = a.amb_col + a.sun_col * 0.55
        base = base * (0.78 + 0.5 * tone[:, None]) * lit
        fogc = self._fog_color(base, dist)
        out = []
        cut = 4.0 if True else 0
        for k in range(len(idx)):
            out.append((float(dist[k]) - (4.0 if dist[k] < self.tree_near else 10.0), 0,
                        (bx[k], by[k], hp[k], wp[k], tuple(int(c) for c in fogc[k]), int(kind[k]), float(dist[k]), float(tone[k]))))
        return out

    def _draw_tree(self, surf, d):
        bx, by, hp, wp, col, kind, dist, seed = d
        poly, line, ell = pygame.draw.polygon, pygame.draw.line, pygame.draw.ellipse
        dark = (int(col[0] * 0.70), int(col[1] * 0.72), int(col[2] * 0.74))
        light = (min(255, int(col[0] * 1.18)), min(255, int(col[1] * 1.18)), min(255, int(col[2] * 1.12)))
        hz = self.atm.haze
        trunk = theme.lerp((92, 66, 44), hz, min(1.0, max(0.0, (dist - 60) / 320)))
        # ground shadow: a flat dark ellipse thrown away from the sun (direction and length set per frame in draw())
        if wp >= 2.2 and self.atm.sun_el > 3.0:
            ux, uy, ln = self._shadow_vec
            fogf = min(1.0, max(0.0, (dist - self.atm.fog_start) / (self.atm.fog_end - self.atm.fog_start)))
            sc = theme.lerp(self._shadow_tint, hz, fogf)
            sw, sh = wp * (1.4 + 0.5 * ln), max(2.0, wp * 0.34)
            cxs, cys = bx + ux * wp * ln, by + uy * wp * ln * 0.5
            ell(surf, sc, (cxs - sw * 0.5 - wp * 0.1, cys - sh * 0.5, sw, sh))
        if wp < 2.2:                                              # far: one blob
            if kind == CONIFER:
                poly(surf, col, [(bx - wp, by), (bx + wp, by), (bx, by - hp)])
            else:
                ell(surf, col, (bx - wp, by - hp * 0.95, 2 * wp, hp * 0.8))
            return
        if kind == CONIFER:
            line(surf, trunk, (bx, by), (bx, by - hp * 0.30), max(1, int(wp * 0.22)))
            for base, top, half in ((0.16, 0.66, 1.0), (0.46, 1.0, 0.72)):
                y0, y1, hw = by - hp * base, by - hp * top, wp * half
                poly(surf, dark, [(bx - hw, y0), (bx + hw, y0), (bx, y1)])
                poly(surf, col, [(bx - hw, y0), (bx, y0 + hw * 0.12), (bx, y1)])
        elif kind == ACACIA:                                       # flat umbrella crown on a forked trunk
            line(surf, trunk, (bx, by), (bx, by - hp * 0.58), max(1, int(wp * 0.16)))
            line(surf, trunk, (bx, by - hp * 0.45), (bx - wp * 0.5, by - hp * 0.7), max(1, int(wp * 0.1)))
            ell(surf, dark, (bx - wp * 1.25, by - hp * 0.95, wp * 2.5, hp * 0.36))
            ell(surf, col, (bx - wp * 1.15, by - hp * 1.0, wp * 2.3, hp * 0.30))
            ell(surf, light, (bx - wp * 0.8, by - hp * 1.02, wp * 1.4, hp * 0.16))
        elif kind == EUCALYPTUS:                                   # tall, narrow, peeling trunk, drooping crown
            line(surf, theme.lerp((178, 170, 150), hz, min(1.0, dist / 400)), (bx, by), (bx, by - hp * 0.6), max(1, int(wp * 0.22)))
            ell(surf, dark, (bx - wp * 0.9, by - hp * 0.72, wp * 1.8, hp * 0.34))          # three drooping tufts
            ell(surf, col, (bx - wp * 1.0, by - hp * 0.88, wp * 1.9, hp * 0.34))
            ell(surf, light, (bx - wp * 0.7, by - hp * 1.0, wp * 1.4, hp * 0.3))
        elif kind == PALM:
            top = (bx + wp * 0.15, by - hp * 0.92)
            line(surf, theme.lerp((140, 112, 80), hz, min(1.0, dist / 400)), (bx, by), top, max(1, int(wp * 0.2)))
            for k in range(7):
                a = math.pi * (k / 6.0) * 0.95 + 0.05
                ex, ey = top[0] + math.cos(a) * wp * 1.4 * (1 if k % 2 else 1), top[1] - math.sin(a) * wp * 0.5 + wp * 0.6 * (0.3 + abs(math.cos(a)))
                line(surf, dark if k % 2 else col, top, (ex, ey), max(1, int(wp * 0.22)))
        elif kind == BANANA:                                       # fan of broad leaves on a short pseudo-stem
            top = (bx, by - hp * 0.55)
            line(surf, theme.lerp((120, 140, 70), hz, min(1.0, dist / 400)), (bx, by), top, max(1, int(wp * 0.4)))
            for k in range(6):
                a = math.radians(30 + k * 24)
                ex, ey = top[0] + math.cos(a) * wp * 1.5, top[1] - math.sin(a) * hp * 0.5 + (hp * 0.18 if k in (0, 5) else 0)
                line(surf, light if k % 2 else col, top, (ex, ey), max(2, int(wp * 0.5)))
        else:                                                      # round broadleaf / flowering crown
            line(surf, trunk, (bx, by), (bx, by - hp * 0.5), max(1, int(wp * 0.2)))
            if wp < 8.0:                                           # mid distance: the cheap three-ellipse crown
                ell(surf, dark, (bx - wp, by - hp * 1.0, wp * 2, hp * 0.62))
                ell(surf, col, (bx - wp * 0.9, by - hp * 1.0, wp * 1.7, hp * 0.52))
                ell(surf, light, (bx - wp * 0.65, by - hp * 1.0, wp * 1.1, hp * 0.3))
            else:
                # Near trees get a lumpy crown: a dark under-canopy, four overlapping leaf clusters whose positions
                # come from the tree's own seed (so each tree is different but never flickers), and a highlight
                # on the side facing the sun.
                sd = int(seed * 1000) % 97
                ell(surf, dark, (bx - wp * 1.0, by - hp * 0.98, wp * 2.0, hp * 0.62))
                for k in range(4):
                    ox = ((sd * (k + 3)) % 13 / 13.0 - 0.5) * wp * 1.05
                    oy = ((sd * (k + 5)) % 11 / 11.0 - 0.5) * hp * 0.16
                    r = wp * (0.62 + 0.14 * ((sd + k) % 3))
                    shade = 0.90 + 0.05 * ((sd + 2 * k) % 5)
                    c2 = (min(255, int(col[0] * shade)), min(255, int(col[1] * shade)), min(255, int(col[2] * shade)))
                    ell(surf, c2, (bx + ox - r, by - hp * 0.82 + oy - r * 0.62, r * 2.0, r * 1.25))
                sx = self._sun_side * wp * 0.38
                ell(surf, light, (bx + sx - wp * 0.5, by - hp * 0.98, wp * 1.0, hp * 0.26))
                ell(surf, dark, (bx - wp * 0.7, by - hp * 0.52, wp * 1.4, hp * 0.12))      # shade under the crown

    # ------------------------------------------------------------------ legacy boxes, gates, targets
    def _collect_boxes(self, v: View):
        out = []
        for centre, world, normals, tint, rad in self._boxes:
            to_cam = v.pos - centre
            dist = float(np.linalg.norm(to_cam))
            if dist > 330 or dist < 0.2:
                continue
            cam = (world - v.pos) @ v.R
            if cam[:, 0].max() < NEAR:
                continue
            out.append((dist - rad, 1, (cam, normals, tint, to_cam, dist)))
        return out

    def _draw_box(self, surf, v, d):
        cam, normals, tint, to_cam, dist = d
        a = self.atm
        for (idx, _), n in zip(_FACES, normals):
            if float(np.dot(n, to_cam)) <= 0:
                continue
            quad = cam[list(idx)]
            if (quad[:, 0] < NEAR).any():
                quad = self._clip_near(quad)
                if len(quad) < 3:
                    continue
            lam = a.amb_col + a.sun_col * max(0.0, float(np.dot(n, a.sun_dir)))
            base = np.array((88, 94, 106) if n[2] > 0.5 else tint, float)
            col = self._fog_color(tuple(base * lam), dist)
            sp = self._to_screen(v, quad)
            if sp[:, 0].max() < 0 or sp[:, 0].min() > v.width or sp[:, 1].max() < 0 or sp[:, 1].min() > v.height:
                continue
            pts = [(int(x), int(y)) for x, y in sp]
            col = tuple(int(min(255, max(0, c))) for c in col)
            pygame.draw.polygon(surf, col, pts)
            if n[2] <= 0.5:
                pygame.draw.polygon(surf, tuple(int(c * 0.78) for c in col), pts, 1)

    def _collect_gates(self, v: View, next_gate: int, passed: set[int]):
        out = []
        for i, g in enumerate(self.course.gates):
            dist = float(np.linalg.norm(g.pos - v.pos))
            out.append((dist - 1.0, 2, (i, g, dist, i == next_gate, i in passed)))
        return out

    def _draw_gate(self, surf, v: View, d):
        i, g, dist, is_next, done = d
        if is_next:
            color, wmul = theme.GATE_NEXT, 1.0
        elif done:
            color, wmul = theme.GATE_DONE, 0.7
        else:
            color, wmul = theme.GATE_IDLE, 0.85
        corners = g.corners(GATE_SIZE)
        A = list(corners)
        B = list(np.roll(corners, -1, axis=0))
        for c in (corners[0], corners[1]):
            A.append(c)
            B.append(np.array([c[0], c[1], float(self.terrain.ground(c[0], c[1]))]))
        A, B = np.array(A), np.array(B)
        ca, cb = (A - v.pos) @ v.R, (B - v.pos) @ v.R
        keep = (ca[:, 0] > NEAR) | (cb[:, 0] > NEAR)
        if not keep.any():
            return
        ca, cb = ca[keep], cb[keep]
        for k in range(len(ca)):
            a, b = ca[k].copy(), cb[k].copy()
            if a[0] < NEAR:
                a = a + (b - a) * ((NEAR - a[0]) / (b[0] - a[0]))
            if b[0] < NEAR:
                b = b + (a - b) * ((NEAR - b[0]) / (a[0] - b[0]))
            ca[k], cb[k] = a, b
        sa, sb = self._to_screen(v, ca), self._to_screen(v, cb)
        thick = int(max(3, min(18, v.f * 0.55 * wmul / max(dist, 1.0))))
        col = self._fog_color(color, dist)
        if is_next:
            pulse = 0.5 + 0.5 * math.sin(self.t * 5.0)
            glow = theme.lerp(self.atm.haze, color, 0.28 + 0.2 * pulse)
            for p, q in zip(sa.astype(int).tolist(), sb.astype(int).tolist()):
                pygame.draw.line(surf, glow, p, q, thick + 8)
        for p, q in zip(sa.astype(int).tolist(), sb.astype(int).tolist()):
            pygame.draw.line(surf, col, p, q, thick)

    def _collect_targets(self, v: View, targets):
        out = []
        for tg in targets or ():
            if not tg.alive:
                continue
            c = tg.centre(self.t)
            dist = float(np.linalg.norm(c - v.pos))
            out.append((dist - 1.0, 3, (tg, c, dist)))
        return out

    def _draw_target(self, surf, v: View, d):
        tg, c, dist = d
        cam = (c - v.pos) @ v.R
        if cam[0] < NEAR:
            return
        x, y = v.cx - v.f * cam[1] / cam[0], v.cy - v.f * cam[2] / cam[0]
        r = v.f * tg.radius / cam[0]
        if x + r < 0 or x - r > v.width or y + r < 0 or y - r > v.height or r < 0.8:
            return
        gcam = (np.array([tg.pos[0], tg.pos[1], tg.ground_z]) - v.pos) @ v.R
        if gcam[0] > NEAR:
            gx, gy = v.cx - v.f * gcam[1] / gcam[0], v.cy - v.f * gcam[2] / gcam[0]
            pygame.draw.line(surf, self._fog_color((235, 235, 235), dist), (int(x), int(y)), (int(gx), int(gy)), 1)
        rings = ((1.0, (232, 48, 52)), (0.78, (250, 250, 250)), (0.56, (232, 48, 52)), (0.34, (250, 250, 250)),
                 (0.16, (232, 48, 52)))
        pulse = 0.5 + 0.5 * math.sin(self.t * 4.0 + tg.phase)
        pygame.draw.circle(surf, theme.lerp(self.atm.haze, (255, 210, 60), 0.35 + 0.25 * pulse), (int(x), int(y)), int(r * 1.22) + 2, 2)
        for k, col in rings:
            pygame.draw.circle(surf, self._fog_color(col, dist), (int(x), int(y)), max(1, int(r * k)))

    # ------------------------------------------------------------------ launch pad (concrete disc, ring, "H")
    def _pad_polys(self):
        """Pad = yellow disc, concrete disc on top, white 'H' - four polygons, projected in one go."""
        if self._pad_cache is None:
            px, py = self.terrain.pad
            gz = float(self.terrain.ground(px, py)) + 0.10
            ang = np.linspace(0, 2 * math.pi, 28, endpoint=False)

            def disc(R, z):
                return np.stack([px + R * np.cos(ang), py + R * np.sin(ang), np.full_like(ang, gz + z)], axis=1)

            def bar(x0, x1, y0, y1):
                z = gz + 0.06
                return np.array([[px + x0, py + y0, z], [px + x1, py + y0, z], [px + x1, py + y1, z], [px + x0, py + y1, z]])
            self._pad_cache = [(disc(6.2, 0.0), (236, 196, 40)), (disc(5.5, 0.03), (150, 150, 152)),
                               (bar(-2.4, -1.5, -2.6, 2.6), (240, 240, 240)), (bar(1.5, 2.4, -2.6, 2.6), (240, 240, 240)),
                               (bar(-1.5, 1.5, -0.4, 0.4), (240, 240, 240))]
        return self._pad_cache

    def _collect_pad(self, v: View):
        px, py = self.terrain.pad
        if self._world() is None:
            return []
        d = float(np.hypot(px - v.pos[0], py - v.pos[1]))
        if d > 220:
            return []
        return [(math.hypot(d, v.pos[2] - self.terrain.ground(px, py)) - 14.0, 4, d)]

    def _draw_pad(self, surf, v: View, d):
        a = self.atm
        lam = a.amb_col + a.sun_col * a.sun_dir[2]
        for P, col in self._pad_polys():
            cam = (P - v.pos) @ v.R
            if (cam[:, 0] < NEAR).any():
                cam = self._clip_near(cam)
                if len(cam) < 3:
                    continue
            sc = self._to_screen(v, cam)
            c = self._fog_color(tuple(np.array(col, float) * lam), d)
            pygame.draw.polygon(surf, tuple(int(min(255, max(0, x))) for x in c), [(int(x), int(y)) for x, y in sc])

    # ------------------------------------------------------------------ drone + particles
    def _drone_model(self, surf, v: View, st) -> None:
        R = quat_to_matrix(st.quat)
        p = np.array(st.pos)
        m = 0.078
        motors = np.array([[m, -m, 0], [-m, -m, 0], [-m, m, 0], [m, m, 0]])
        A, B, colors = [], [], []
        for i in range(4):
            A.append(np.zeros(3)); B.append(motors[i]); colors.append(theme.OSD if motors[i][0] < 0 else theme.GATE_IDLE)
        ang = np.linspace(0, 2 * math.pi, 13)
        ring = np.stack([0.06 * np.cos(ang), 0.06 * np.sin(ang), np.full_like(ang, 0.012)], axis=1)
        spin = 1.0 + 6.0 * float(st.motors[0])
        for i in range(4):
            pts = ring + motors[i]
            for k in range(12):
                A.append(pts[k]); B.append(pts[k + 1])
                colors.append(theme.lerp((60, 70, 90), theme.ACCENT, 0.4 + 0.6 * min(1.0, spin / 3)))
        A.append(np.array([0.045, 0, 0.02])); B.append(np.array([0.09, 0, 0.02])); colors.append(theme.GOOD)
        body = np.array([[0.05, 0.03, 0.02], [-0.05, 0.03, 0.02], [-0.05, -0.03, 0.02], [0.05, -0.03, 0.02]])
        for k in range(4):
            A.append(body[k]); B.append(body[(k + 1) % 4]); colors.append(theme.OSD)
        k = 2.2
        A = (np.array(A) * k) @ R.T + p
        B = (np.array(B) * k) @ R.T + p
        s0, s1, depth, idx = v.segments(A, B)
        for a, b, i in zip(s0.astype(int).tolist(), s1.astype(int).tolist(), idx):
            pygame.draw.line(surf, colors[i], a, b, 3)

    def _shadow(self, surf, v: View, st) -> None:
        gx, gy = st.pos[0], st.pos[1]
        gz = float(self.terrain.ground(gx, gy)) + 0.05
        alt = max(0.0, st.pos[2] - gz)
        r = 0.22 + 0.04 * alt
        ang = np.linspace(0, 2 * math.pi, 20)
        P = np.stack([gx + r * np.cos(ang), gy + r * np.sin(ang), np.full_like(ang, gz)], axis=1)
        cam = (P - v.pos) @ v.R
        if (cam[:, 0] < NEAR).any():
            return
        pts = [(int(x), int(y)) for x, y in self._to_screen(v, cam)]
        pygame.draw.polygon(surf, (22, 40, 26), pts)

    def _draw_particles(self, surf, v: View, effects) -> None:
        if effects is None or not effects.particles:
            return
        P = np.array([p.pos for p in effects.particles])
        cam = (P - v.pos) @ v.R
        for p, c in zip(effects.particles, cam):
            if c[0] < NEAR:
                continue
            x, y = v.cx - v.f * c[1] / c[0], v.cy - v.f * c[2] / c[0]
            if not (0 <= x < v.width and 0 <= y < v.height):
                continue
            a = max(0.0, p.ttl / p.life)
            col = theme.lerp(self.atm.haze, p.color, 0.3 + 0.7 * a)
            pygame.draw.circle(surf, col, (int(x), int(y)), max(1, int(p.size * (0.4 + a) * min(2.5, 10.0 / max(c[0], 4.0)))))

    # ------------------------------------------------------------------ frame
    def draw(self, surf: pygame.Surface, v: View, st, next_gate: int, passed: set[int], dt: float,
             show_drone: bool, targets=None, effects=None) -> None:
        self.t += dt
        city = self.city
        if city is not None:
            city.update(dt)
        self._frame_vectors(v)
        self._sky(surf, v)
        self._sun(surf, v)
        self._clouds_draw(surf, v)
        self._mountains(surf, v)

        k_far, p_far, c_far = self._terrain_quads(v, self._grids[FAR_STRIDE], self.far_r)
        objs = self._collect_trees(v) + self._collect_boxes(v) + self._collect_gates(v, next_gate, passed) + \
            self._collect_targets(v, targets) + self._collect_pad(v)
        if city is not None:
            objs += city.building_objects(v, self.bld_far, self.bld_max) + city.vehicle_objects(v, 170.0) + \
                city.bird_objects(v) + city.mesh_objects(v, self.bld_far)
            (rk_n, rp_n, rc_n), (rk_f, rp_f, rc_f) = city.road_quads(v, self.road_far, self.tree_near, self._light_at)
            if len(rk_f):
                k_far = np.concatenate([k_far, rk_f])
                p_far = p_far + rp_f
                c_far = np.concatenate([c_far, rc_f]) if len(c_far) else rc_f
        far_objs = sorted([o for o in objs if o[0] > self.tree_near], key=lambda o: -o[0])
        near_objs = sorted([o for o in objs if o[0] <= self.tree_near], key=lambda o: -o[0])
        self._paint(surf, v, k_far, p_far, c_far, far_objs)

        k_near, p_near, c_near = self._terrain_quads(v, self._grids[NEAR_STRIDE], self.near_r, self.tree_near + 10.0)
        if city is not None and len(rk_n):
            k_near = np.concatenate([k_near, rk_n]) if len(k_near) else rk_n
            p_near = p_near + rp_n
            c_near = np.concatenate([c_near, rc_n]) if len(c_near) else rc_n
        self._paint(surf, v, k_near, p_near, c_near, near_objs)

        if show_drone:
            self._shadow(surf, v, st)
            self._drone_model(surf, v, st)
        self._draw_particles(surf, v, effects)
        if self.quality >= 1:
            self._post(surf, v)

    # ------------------------------------------------------------------ camera look
    def _frame_vectors(self, v: View) -> None:
        """Per-frame sun geometry used by trees: which way shadows fall on screen and which side is lit."""
        a = self.atm
        sun = a.sun_dir
        cam = sun @ v.R
        self._sun_side = -1.0 if cam[1] > 0 else 1.0               # sun on the left of the picture -> lit side is left
        if cam[0] < -0.2:
            self._sun_side = 0.0                                    # sun behind the camera: trees are lit face-on
        h = np.array([-sun[0], -sun[1], 0.0])
        hn = np.linalg.norm(h)
        h = h / hn if hn > 1e-6 else h
        c = h @ v.R                                                 # shadow direction in camera axes (forward, left, up)
        ln = min(2.4, max(0.45, 1.0 / math.tan(max(math.radians(a.sun_el), 0.12))))
        self._shadow_vec = (-float(c[1]), -float(c[0]) , ln)       # screen x grows to the right, y downwards
        self._shadow_tint = tuple(int(x) for x in (np.array((92, 88, 66), float) * (a.amb_col + a.sun_col * 0.05)).clip(0, 255))

    def _post(self, surf: pygame.Surface, v: View) -> None:
        """Camera-sensor look: film grain and a lens flare when the sun is in view.

        None of this changes the geometry - it hides the 'flat polygon' look the way a real lens and sensor do.
        (A bloom pass was tried and removed: it cost ~3 ms and washed out hazy lakes.)
        """
        w, h = surf.get_size()
        # film grain: one ADD blit of a pre-made noise frame (three variants cycled so it shimmers)
        if getattr(self, "_grain_size", None) != (w, h):
            rng = np.random.default_rng(3)
            self._grain = []
            for _ in range(3):
                n = (rng.random((w, h)) * 9).astype(np.uint8)
                self._grain.append(pygame.surfarray.make_surface(np.stack([n, n, n], axis=-1)))
            self._grain_size = (w, h)
        self._grain_i = (getattr(self, "_grain_i", 0) + 1) % 3
        surf.blit(self._grain[self._grain_i], (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        self._lens_flare(surf, v)

    def _lens_flare(self, surf: pygame.Surface, v: View) -> None:
        cam = self.atm.sun_dir @ v.R
        if cam[0] < 0.1 or self.atm.sun_visible < 0.5:
            return
        sx, sy = v.cx - v.f * cam[1] / cam[0], v.cy - v.f * cam[2] / cam[0]
        w, h = surf.get_size()
        if not (0 <= sx < w and 0 <= sy < h):
            return
        px = surf.get_at((int(sx), int(sy)))
        if px[0] + px[1] + px[2] < 560:                             # something (a building, a cloud) covers the sun
            return
        if getattr(self, "_flare", None) is None:
            self._flare = []
            for rad, k in ((16, 0.9), (30, 0.55), (56, 0.3)):
                n = rad * 2
                yy, xx = np.mgrid[0:n, 0:n]
                r = np.hypot(xx - rad + 0.5, yy - rad + 0.5) / rad
                ring = np.clip(1 - np.abs(r - 0.78) / 0.22, 0, 1) * 0.5 + np.clip(1 - r, 0, 1) * 0.5
                g = np.clip(ring * 60 * k, 0, 255).astype(np.uint8)
                self._flare.append(pygame.surfarray.make_surface(np.stack([g, (g * 0.9).astype(np.uint8), (g * 0.7).astype(np.uint8)], axis=-1)))
        d = math.hypot(sx - w / 2, sy - h / 2) / math.hypot(w / 2, h / 2)
        k = max(0.0, 1.0 - d * 0.9)                                 # strongest when the sun is near the middle
        if k <= 0.02:
            return
        for spr, t in zip(self._flare, (-0.55, -1.1, 0.5)):
            fx, fy = w / 2 + (sx - w / 2) * t, h / 2 + (sy - h / 2) * t
            s2 = spr if k > 0.85 else pygame.transform.smoothscale(spr, (max(2, int(spr.get_width() * (0.5 + 0.5 * k))),) * 2)
            surf.blit(s2, s2.get_rect(center=(int(fx), int(fy))), special_flags=pygame.BLEND_RGB_ADD)

    def _paint(self, surf, v: View, keys, polys, cols, objs) -> None:
        order = np.argsort(-keys) if len(keys) else []
        cols_l = np.clip(cols, 0, 255).astype(int).tolist() if len(keys) else []
        keys_l = keys.tolist() if len(keys) else []
        poly = pygame.draw.polygon
        oi, no = 0, len(objs)
        for q in order:
            kq = keys_l[q]
            while oi < no and objs[oi][0] >= kq:
                self._draw_obj(surf, v, objs[oi])
                oi += 1
            poly(surf, cols_l[q], polys[q])
        while oi < no:
            self._draw_obj(surf, v, objs[oi])
            oi += 1

    def _draw_obj(self, surf, v: View, o) -> None:
        kind, d = o[1], o[2]
        if kind == 0:
            self._draw_tree(surf, d)
        elif kind == 1:
            self._draw_box(surf, v, d)
        elif kind == 2:
            self._draw_gate(surf, v, d)
        elif kind == 3:
            self._draw_target(surf, v, d)
        elif kind == 4:
            self._draw_pad(surf, v, d)
        else:
            self.city.draw(surf, v, kind, d)


# Backwards-compatible name used by earlier code
SceneRenderer = WorldRenderer