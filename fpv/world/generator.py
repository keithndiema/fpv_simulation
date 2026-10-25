"""Procedural Ugandan town generator.

Pipeline (each step feeds the next):
  1. terrain        rolling hills + the profile's lake / river / rock / crater / ridge features, urban smoothing
  2. street lattice a warped, rotated grid (or radial arterials) -> road polylines, split by road class
  3. park + pad     one open block near the centre becomes a lawn, flattened, with the launch pad in it
  4. parcels        every block is subdivided (BSP) and filled by zone: CBD towers, shops, houses, informal, farms
  5. landmarks      mosque / cathedral / dam / runway ... built from the profile and stamped over the town
  6. trees, cover   street trees, gardens, banana groves, acacias; a ground-cover raster for colouring
  7. extras         traffic routes, horizon mountains, an occupancy raster (used for collisions, shadows, minimap)

World frame: +x north, +y west. Everything is deterministic in (city.seed), so a city always looks the same.
When `real` (live OSM + DEM data, see online.py) is given, steps 2 and 4 are replaced by the real streets and
buildings and step 1 uses the real elevation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..game.terrain import (ACACIA, BANANA, BROADLEAF, CELL, EUCALYPTUS, FLOWERING, N, ORIGIN, PALM, Terrain,
                            Trees)
from . import features
from .cities import City, Feature
from .model import (KIND_HOUSE, KIND_INDUSTRY, KIND_MIDRISE, KIND_SHOP, KIND_TOWER, KIND_HUT, ROAD_CLASSES,
                    ROOF_FLAT, ROOF_GABLE, ROOF_HUT, Buildings, RoadLine, Vehicle, World)
from .noise import (bearing_vec, bilinear, box_blur, dist_to_polyline, fbm, polygon_mask, resample_polyline, smooth,
                    value_noise, voronoi_ids)

# ground-cover codes (rendered in render/scene.py)
GRASS, FARM_GREEN, FARM_DRY, URBAN, LATERITE, WETLAND, TEA, ROCK_C, SAND, LAWN, PLANTATION = range(11)

OCC = 2.5                              # occupancy raster cell [m]
M = int((N - 1) * CELL / OCC)          # 512 cells per side

WALL_PAINT = [(236, 228, 210), (222, 206, 170), (205, 150, 120), (176, 196, 170), (160, 184, 204), (230, 214, 150),
              (240, 240, 236), (196, 168, 140), (214, 128, 110), (150, 180, 150)]
WALL_BRICK = [(170, 98, 74), (158, 86, 64), (182, 112, 82)]
WALL_CONCRETE = [(188, 186, 180), (170, 170, 166), (204, 200, 190), (150, 150, 148)]
SHOP_BRIGHT = [(238, 196, 60), (214, 70, 60), (70, 120, 190), (80, 160, 90), (236, 236, 232), (232, 130, 50)]
GLASS = [(168, 176, 186), (188, 192, 194), (148, 162, 174), (204, 198, 186), (176, 186, 178)]   # curtain-wall frames
ROOF_IRON = [(142, 62, 44), (118, 46, 40), (74, 112, 88), (62, 92, 142), (158, 158, 160), (176, 98, 58), (96, 66, 52),
             (132, 70, 50)]
ROOF_THATCH = [(168, 134, 78), (150, 118, 68), (178, 146, 90)]
MUD = [(150, 106, 78), (138, 96, 70), (164, 120, 86)]


@dataclass
class _Ctx:
    city: City
    rng: np.random.Generator
    X: np.ndarray
    Y: np.ndarray
    cx: float
    cy: float
    h: np.ndarray
    river_d: np.ndarray                # distance to the nearest river centreline [m] (large if none)
    river_w: float = 0.0


def _col(palette, rng, jitter=10):
    c = np.array(palette[int(rng.integers(0, len(palette)))], float)
    return np.clip(c + rng.normal(0, jitter, 3), 0, 255)


# ================================================================================================ 1. terrain
def _terrain_heights(city: City, cx: float, cy: float):
    X = ORIGIN[0] + np.arange(N) * CELL
    Y = ORIGIN[1] + np.arange(N) * CELL
    X, Y = np.meshgrid(X, Y)
    RX, RY = X - cx, Y - cy
    r = np.hypot(RX, RY)
    lam = 1400.0 / city.hill_freq
    # gain 0.4: each octave is smaller *and* gentler, so slopes stay walkable (equal-gain noise makes cliffs)
    n = fbm(X / lam, Y / lam, city.seed, 5, gain=0.4)
    z = np.clip((n - n.mean()) / (2.0 * n.std() + 1e-9), -1.0, 1.0)
    base = city.hills * 1.05 + 7.0
    h = base + city.hills * z
    h += (fbm(X / 55.0, Y / 55.0, city.seed + 5, 3) - 0.5) * (1.6 + 0.04 * city.hills)    # micro relief
    # the surrounding country rises into hills so the end of the map is always hidden in haze
    h += 55.0 * smooth((r - 470.0) / 170.0) * (0.5 + fbm(X / 300, Y / 300, city.seed + 9, 3))

    # built-up ground is graded: smooth it where the town is (more for denser towns)
    urban = 1.0 - smooth((r - 230.0) / 400.0)
    h = h * (1 - 0.82 * urban * min(1.0, city.density)) + box_blur(h, 5) * 0.82 * urban * min(1.0, city.density)

    river_d = np.full(h.shape, 1e9)
    river_w = 0.0
    lakes = []
    for f in city.features:
        if f.kind == "ridge":
            v = bearing_vec(f.bearing)
            proj = RX * v[0] + RY * v[1]
            h += f.a * smooth((proj - (f.dist - f.b)) / (2.0 * f.b)) * (0.85 + 0.3 * fbm(X / 90, Y / 90, city.seed + 3, 3))
        elif f.kind == "rock":
            v = bearing_vec(f.bearing)
            rx, ry = cx + v[0] * f.dist, cy + v[1] * f.dist
            d = np.hypot(X - rx, Y - ry) / max(f.b, 1.0)
            dome = 1.0 - smooth((d - 0.35) / 0.65)
            rough = 0.75 + 0.5 * fbm(X / 22, Y / 22, city.seed + 17, 3)
            h = np.maximum(h, h * (1 - dome) + (h.mean() * 0.4 + f.a) * dome * rough)
        elif f.kind == "river":
            ax = bearing_vec(f.c)
            nrm = bearing_vec(f.c + 90.0)
            t = np.arange(-1100.0, 1100.0, 8.0)
            meander = f.b * np.sin(t / 240.0 * math.pi + city.seed) + f.b * 0.4 * np.sin(t / 97.0 + city.seed * 2)
            pts = np.stack([cx + nrm[0] * f.dist + ax[0] * t + nrm[0] * meander,
                            cy + nrm[1] * f.dist + ax[1] * t + nrm[1] * meander], axis=1)
            d = dist_to_polyline(X, Y, pts)
            vw = np.maximum(70.0 + 2.2 * city.hills, 3.6 * np.maximum(h + 1.5, 0.0))     # banks no steeper than ~0.28
            tt = np.clip((d - f.a / 2.0) / vw, 0.0, 1.0)
            h = np.where(d < 1e8, -1.5 + (h + 1.5) * smooth(tt), h)
            river_d = np.minimum(river_d, d)
            river_w = max(river_w, f.a)
        elif f.kind == "lake":
            lakes.append(f)
    if city.terrace > 0:                                               # stepped farm terraces on slopes
        step = 5.5
        q = h / step
        stepped = (np.floor(q) + smooth((q - np.floor(q) - 0.55) / 0.45)) * step
        gy, gx = np.gradient(h, CELL)
        slope = np.hypot(gx, gy)
        w = city.terrace * smooth((slope - 0.05) / 0.15)
        h = h * (1 - w) + stepped * w
    for f in city.features:
        if f.kind == "crater":
            v = bearing_vec(f.bearing)
            rx, ry = cx + v[0] * f.dist, cy + v[1] * f.dist
            d = np.hypot(X - rx, Y - ry)
            bowl = smooth((f.a - d) / (0.35 * f.a))
            h = h * (1 - bowl) + (-2.5) * bowl
            river_d = np.minimum(river_d, np.maximum(d - f.a * 0.6, 0))
    for f in lakes:                                                    # Lake Victoria & co: water covers one side
        v = bearing_vec(f.bearing)
        proj = RX * v[0] + RY * v[1]
        wig = (fbm(X / 210.0, Y / 210.0, city.seed + 41, 3) - 0.5) * 2.0 * f.a
        shore = proj - (f.dist + wig)
        land = h * smooth(-shore / 170.0)
        lake = -np.minimum(0.06 * shore + 0.8, 9.0)
        h = np.where(shore <= 0, np.minimum(land, h), lake)
        river_d = np.minimum(river_d, np.maximum(shore * -1.0, 0.0))
    return X, Y, h, river_d, river_w


# ================================================================================================ 2. streets
def _axis(rng, half: float, spacing, offset: float) -> np.ndarray:
    pos = [offset]
    u = offset
    while u < half:
        u += spacing(u) * rng.uniform(0.86, 1.14)
        pos.append(u)
    u = offset
    while u > -half:
        u -= spacing(u) * rng.uniform(0.86, 1.14)
        pos.append(u)
    return np.array(sorted(pos))


class _Lattice:
    """Rotated, gently warped street grid. (u, v) are lattice metres from the town centre."""

    def __init__(self, city: City, cx: float, cy: float, rng: np.random.Generator):
        self.city, self.cx, self.cy = city, cx, cy
        self.theta = math.radians(rng.uniform(-24, 24)) if city.layout != "grid" else math.radians(rng.uniform(-12, 12))
        self.c, self.s = math.cos(self.theta), math.sin(self.theta)
        self.A = {"grid": 12.0, "radial": 28.0, "organic": 40.0}[city.layout]
        self.seed = city.seed
        dens = max(0.5, city.density)
        s0u = {"grid": 86.0, "radial": 92.0, "organic": 100.0}[city.layout] * (1.25 - 0.25 * min(dens, 1.4))
        s0v = s0u * 0.82
        rural = lambda s0: (lambda u: s0 * (1.0 + 1.5 * float(smooth((abs(u) - 260.0) / 380.0))))
        self.us = _axis(rng, 1000.0, rural(s0u), rng.uniform(-s0u / 2, s0u / 2))
        self.vs = _axis(rng, 1000.0, rural(s0v), rng.uniform(-s0v / 2, s0v / 2))

    def to_world(self, u, v):
        lw = 380.0
        wu = self.A * (fbm(np.asarray(u) / lw + 3.1, np.asarray(v) / lw, self.seed + 1, 3) - 0.5) * 2.0
        wv = self.A * (fbm(np.asarray(u) / lw, np.asarray(v) / lw + 7.7, self.seed + 2, 3) - 0.5) * 2.0
        uu, vv = u + wu, v + wv
        return self.cx + self.c * uu - self.s * vv, self.cy + self.s * uu + self.c * vv

    def yaw_at(self, u, v):
        x0, y0 = self.to_world(u, v)
        x1, y1 = self.to_world(u + 4.0, v)
        return np.arctan2(y1 - y0, x1 - x0)


def _road_kind(base: str, r: float, rural_r: float) -> str:
    """Degrade a street as it leaves town: tarmac -> murram -> footpath. Primary roads run on as highways."""
    if r < rural_r:
        return base
    if base == "primary":
        return "primary"
    return "murram" if base == "secondary" else "track"


def _build_roads(ctx: _Ctx, lat: _Lattice, rural_r: float) -> list[RoadLine]:
    city, rng = ctx.city, ctx.rng
    lines: list[RoadLine] = []
    lat.u_class, lat.v_class = [], []
    step = 12.0

    def add(pts: np.ndarray, base: str):
        r = np.hypot(pts[:, 0] - ctx.cx, pts[:, 1] - ctx.cy)
        kinds = [_road_kind(base, float(x), rural_r) for x in r]
        # inside the built-up area but on informal ground local streets are murram
        i = 0
        while i < len(pts):
            j = i
            while j + 1 < len(pts) and kinds[j + 1] == kinds[i]:
                j += 1
            if j > i:
                lines.append(RoadLine(pts[i:j + 1].copy(), kinds[i]))
            i = j + 1

    # --- lattice lines ----------------------------------------------------------------------------
    t = np.arange(-1000.0, 1000.0 + step, step)
    for i, u in enumerate(lat.us):
        x, y = lat.to_world(np.full_like(t, u), t)
        major = abs(u) < 900 and (i % 4 == (lat.us.searchsorted(0) % 4))
        kind = "primary" if major else ("secondary" if i % 2 == 0 else "local")
        lat.u_class.append(kind)
        add(np.stack([x, y], axis=1), kind)
    for j, v in enumerate(lat.vs):
        x, y = lat.to_world(t, np.full_like(t, v))
        major = abs(v) < 900 and (j % 5 == (lat.vs.searchsorted(0) % 5))
        kind = "primary" if major else ("secondary" if j % 3 == 0 else "local")
        lat.v_class.append(kind)
        add(np.stack([x, y], axis=1), kind)

    # --- radial arterials (Kampala-style: roads fanning out of the centre) ---------------------------------
    if city.layout == "radial":
        base_b = rng.uniform(0, 360)
        for k in range(7):
            b = base_b + k * 360.0 / 7 + rng.uniform(-14, 14)
            d = bearing_vec(b)
            tt = np.arange(0.0, 1000.0, step)
            bend = 30.0 * np.sin(tt / 330.0 * math.pi + k) * smooth(tt / 300.0)
            nx = np.array([-d[1], d[0]])
            pts = np.stack([ctx.cx + d[0] * tt + nx[0] * bend, ctx.cy + d[1] * tt + nx[1] * bend], axis=1)
            lines.append(RoadLine(pts, "primary"))
    return lines


def _clip_and_bridge(lines: list[RoadLine], h: np.ndarray, rural_r: float) -> list[RoadLine]:
    """Trim roads to the map; cut them off where they hit water or cliffs, but keep short wet gaps as bridges."""
    lo, hi = ORIGIN[0] + 12, ORIGIN[0] + (N - 1) * CELL - 12
    out: list[RoadLine] = []

    def hgt(p):
        return bilinear(h, p[:, 0], p[:, 1], ORIGIN[0], ORIGIN[1], CELL)

    gy, gx = np.gradient(h, CELL)
    slope_grid = np.hypot(gx, gy)
    for ln in lines:
        p = ln.pts
        inside = (p[:, 0] > lo) & (p[:, 0] < hi) & (p[:, 1] > lo) & (p[:, 1] < hi)
        hz = hgt(np.clip(p, lo, hi))
        wet = hz < 0.7
        slope = bilinear(slope_grid, np.clip(p[:, 0], lo, hi), np.clip(p[:, 1], lo, hi), ORIGIN[0], ORIGIN[1], CELL)
        steep = slope > (0.55 if ln.kind in ("primary", "secondary") else 0.4)
        ok = inside & ~steep
        # bridge: wet runs shorter than ~140 m that have dry road on both sides
        bridge = np.zeros(len(p), bool)
        i = 0
        while i < len(p):
            if wet[i] and inside[i]:
                j = i
                while j + 1 < len(p) and wet[j + 1]:
                    j += 1
                span = (j - i + 1) * 12.0
                if ln.kind in ("primary", "secondary") and span < 140.0 and i > 0 and j < len(p) - 1 \
                        and inside[i - 1] and inside[j + 1]:
                    bridge[i:j + 1] = True
                i = j + 1
            else:
                i += 1
        ok &= (~wet) | bridge
        k = 0
        while k < len(p):
            if not ok[k]:
                k += 1
                continue
            e = k
            while e + 1 < len(p) and ok[e + 1]:
                e += 1
            if e > k:
                seg = slice(k, e + 1)
                br = bridge[seg].copy()
                zseg = hz[seg].copy()
                idx = np.nonzero(br)[0]
                if len(idx):
                    # each contiguous bridge run gets a straight deck between its two dry abutments
                    runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
                    for run in runs:
                        a_, b_ = max(run[0] - 1, 0), min(run[-1] + 1, len(zseg) - 1)
                        zseg[a_:b_ + 1] = np.interp(np.arange(a_, b_ + 1), [a_, b_], [zseg[a_], zseg[b_]]) + 0.9
                out.append(RoadLine(p[seg].copy(), ln.kind, br, zseg))
            k = e + 1
    return out


# ================================================================================================ rasters
class _Occ:
    """2.5 m raster: what is built / paved / wet where. Used to place things without overlaps."""

    def __init__(self, h: np.ndarray):
        xs = ORIGIN[0] + (np.arange(M) + 0.5) * OCC
        XX, YY = np.meshgrid(xs, xs)
        self.xs = xs
        self.ground = bilinear(h, XX, YY, ORIGIN[0], ORIGIN[1], CELL)
        gy, gx = np.gradient(self.ground, OCC)
        self.slope = np.hypot(gx, gy)
        self.road = np.zeros((M, M), bool)
        self.height = np.zeros((M, M), np.float32)          # rooftop above ground
        self.tree = np.zeros((M, M), bool)

    def idx(self, x, y):
        i = np.clip(((np.asarray(x) - ORIGIN[0]) / OCC).astype(int), 0, M - 1)
        j = np.clip(((np.asarray(y) - ORIGIN[1]) / OCC).astype(int), 0, M - 1)
        return j, i

    def stamp_polyline(self, pts: np.ndarray, radius: float, target: np.ndarray) -> None:
        """Mark every cell within `radius` metres of the polyline (circular stamp, all samples at once)."""
        if len(pts) < 2:
            return
        seg = pts[1:] - pts[:-1]
        length = np.hypot(seg[:, 0], seg[:, 1])
        k = np.maximum(1, (length / 1.6).astype(int))
        t = np.concatenate([np.linspace(0, 1, n + 1)[:-1] for n in k])
        a = np.repeat(pts[:-1], k, axis=0)
        d = np.repeat(seg, k, axis=0)
        P = a + d * t[:, None]
        j, i = self.idx(P[:, 0], P[:, 1])
        r = int(math.ceil(radius / OCC))
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if (dj * dj + di * di) * OCC * OCC <= (radius + OCC * 0.5) ** 2:
                    target[np.clip(j + dj, 0, M - 1), np.clip(i + di, 0, M - 1)] = True

    def stamp_box(self, x, y, hx, hy, yaw, height) -> None:
        reach = math.hypot(hx, hy)
        j0, i0 = self.idx(x - reach, y - reach)
        j1, i1 = self.idx(x + reach, y + reach)
        sub_x, sub_y = np.meshgrid(self.xs[i0:i1 + 1], self.xs[j0:j1 + 1])
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = sub_x - x, sub_y - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        inside = (np.abs(lx) <= hx + 0.8) & (np.abs(ly) <= hy + 0.8)
        reg = self.height[j0:j1 + 1, i0:i1 + 1]
        np.maximum(reg, np.where(inside, height, 0.0).astype(np.float32), out=reg)

    def free(self, x, y, margin: float = 0.0) -> np.ndarray:
        """True where (x, y) is dry, gentle, off the road and out of any building."""
        j, i = self.idx(x, y)
        ok = (self.ground[j, i] > 1.0) & (self.slope[j, i] < 0.45) & ~self.road[j, i] & (self.height[j, i] <= 0)
        if margin > 0:
            r = int(math.ceil(margin / OCC))
            for dj in (-r, r):
                for di in (-r, r):
                    jj, ii = np.clip(j + dj, 0, M - 1), np.clip(i + di, 0, M - 1)
                    ok &= ~self.road[jj, ii] & (self.height[jj, ii] <= 0)
        return ok


# ================================================================================================ 4. parcels
def _bsp(rng, u0, v0, u1, v1, target_area, min_side, out, depth=0):
    w, d = u1 - u0, v1 - v0
    if w * d <= target_area * rng.uniform(0.8, 1.5) or depth > 7 or (w < 2 * min_side and d < 2 * min_side):
        out.append((u0, v0, u1, v1))
        return
    if (w >= d and w >= 2 * min_side) or d < 2 * min_side:
        s = u0 + w * rng.uniform(0.38, 0.62)
        _bsp(rng, u0, v0, s, v1, target_area, min_side, out, depth + 1)
        _bsp(rng, s, v0, u1, v1, target_area, min_side, out, depth + 1)
    else:
        s = v0 + d * rng.uniform(0.38, 0.62)
        _bsp(rng, u0, v0, u1, s, target_area, min_side, out, depth + 1)
        _bsp(rng, u0, s, u1, v1, target_area, min_side, out, depth + 1)


class _BuildingBuffer:
    def __init__(self):
        self.rows: list = []

    def add(self, u, v, hu, hv, wall, floors, kind, roof, rise, wall_rgb, roof_rgb, yaw_extra=0.0):
        self.rows.append((u, v, hu, hv, wall, floors, kind, roof, rise, wall_rgb, roof_rgb, yaw_extra))


def _fill_blocks(ctx: _Ctx, lat: _Lattice, occ: _Occ, park_block, rural_r: float, R0: float):
    """Subdivide each lattice block into parcels and place buildings by zone. Returns (buffer, zone_by_block)."""
    city, rng = ctx.city, ctx.rng
    buf = _BuildingBuffer()
    zones = {}
    us, vs = lat.us, lat.vs
    roadhalf = {"primary": 6.5, "secondary": 4.5, "local": 3.0}
    for bi in range(len(us) - 1):
        for bj in range(len(vs) - 1):
            ua, ub, va, vb = us[bi], us[bi + 1], vs[bj], vs[bj + 1]
            uc, vc = (ua + ub) / 2, (va + vb) / 2
            xc, yc = lat.to_world(np.array(uc), np.array(vc))
            xc, yc = float(xc), float(yc)
            if not (ORIGIN[0] + 10 < xc < ORIGIN[0] + (N - 1) * CELL - 10 and ORIGIN[1] + 10 < yc < ORIGIN[1] + (N - 1) * CELL - 10):
                continue
            r = math.hypot(xc - ctx.cx, yc - ctx.cy)
            jitter = (float(fbm(xc / 170.0, yc / 170.0, city.seed + 77, 3)) - 0.5) * 0.5
            s = r / R0 + jitter
            j, i = occ.idx(xc, yc)
            hg = float(occ.ground[j, i])
            low = hg < 6.0 or float(ctx.river_d[min(N - 1, int((yc - ORIGIN[1]) / CELL)), min(N - 1, int((xc - ORIGIN[0]) / CELL))]) < 120.0
            if s < 0.30 and city.density >= 0.6:
                zone = "cbd"
            elif s < 0.55:
                zone = "commercial"
            elif s < 1.0:
                zone = "informal" if (low and r > 110 and float(value_noise(xc / 90, yc / 90, city.seed + 5)) > 0.35 and city.density > 0.7) else "residential"
            else:
                zone = "rural"
            if city.density < 0.6 and zone == "cbd":
                zone = "commercial"
            zones[(bi, bj)] = zone
            if park_block == (bi, bj):
                continue
            # keep every parcel clear of the street on each side (the half-width depends on that street's class)
            verge = 4.0 if zone in ("cbd", "commercial") else 3.2
            u0, u1 = ua + roadhalf[lat.u_class[bi]] + verge, ub - roadhalf[lat.u_class[bi + 1]] - verge
            v0, v1 = va + roadhalf[lat.v_class[bj]] + verge, vb - roadhalf[lat.v_class[bj + 1]] - verge
            if u1 - u0 < 8 or v1 - v0 < 8:
                continue
            if zone == "rural":
                _rural_block(ctx, buf, rng, u0, v0, u1, v1, city)
                continue
            area = {"cbd": (1000, 2000), "commercial": (320, 620), "residential": (560, 1100), "informal": (95, 190)}[zone]
            target = rng.uniform(*area)
            hilltop = zone == "residential" and hg > 0 and float(fbm(xc / 400, yc / 400, city.seed + 21, 2)) > 0.62 and low is False
            if hilltop:
                target *= 2.0
            parcels: list = []
            _bsp(rng, u0, v0, u1, v1, target, 7.0 if zone != "informal" else 5.0, parcels)
            for (pu0, pv0, pu1, pv1) in parcels:
                _parcel(ctx, buf, rng, zone, pu0, pv0, pu1, pv1, hilltop, r / R0)
    return buf, zones


def _rural_block(ctx, buf, rng, u0, v0, u1, v1, city):
    n = int(rng.poisson(0.55 * max(0.4, city.density)))
    for _ in range(n):
        u = rng.uniform(u0 + 10, max(u0 + 11, u1 - 10))
        v = rng.uniform(v0 + 10, max(v0 + 11, v1 - 10))
        if rng.random() < 0.55:                                       # rectangular tin-roof farmhouse
            wall = _col(WALL_BRICK + MUD + WALL_PAINT, rng)
            buf.add(u, v, rng.uniform(3.0, 4.5), rng.uniform(2.4, 3.2), 2.7, 1, KIND_HOUSE, ROOF_GABLE,
                    rng.uniform(0.9, 1.4), wall, _col(ROOF_IRON, rng), rng.uniform(-0.3, 0.3))
        for _k in range(int(rng.integers(0, 4))):                      # round thatched huts of the compound
            hu, hv = u + rng.uniform(-14, 14), v + rng.uniform(-14, 14)
            rad = rng.uniform(1.5, 2.3)
            buf.add(hu, hv, rad, rad, 1.9, 1, KIND_HUT, ROOF_HUT, rng.uniform(1.4, 2.0), _col(MUD, rng, 6),
                    _col(ROOF_THATCH, rng, 8))


def _parcel(ctx, buf, rng, zone, pu0, pv0, pu1, pv1, hilltop, rel_r):
    city = ctx.city
    pw, pd = pu1 - pu0, pv1 - pv0
    if min(pw, pd) < 4.5:
        return
    uc, vc = (pu0 + pu1) / 2, (pv0 + pv1) / 2
    if zone == "cbd":
        floors = int(np.clip(rng.lognormal(1.55 + 0.35 * city.towers, 0.55), 2, 8 + 26 * city.towers))
        if rng.random() > 0.55 + 0.4 * city.towers:
            floors = int(rng.integers(2, 6))
        hu, hv = (pw - 3.0) / 2 * rng.uniform(0.82, 1.0), (pd - 3.0) / 2 * rng.uniform(0.82, 1.0)
        fh = 3.5
        if floors >= 9:
            wall = _col(GLASS, rng, 6)
            kind = KIND_TOWER
        elif rng.random() < 0.3:
            wall = _col(SHOP_BRIGHT, rng, 12)
            kind = KIND_MIDRISE
        else:
            wall = _col(WALL_CONCRETE + WALL_PAINT[:6], rng, 8)
            kind = KIND_MIDRISE
        roof = (112 + rng.normal(0, 16),) * 3                # weathered concrete / bitumen: lighter than fresh tar
        buf.add(uc, vc, hu, hv, floors * fh, floors, kind, ROOF_FLAT, 0.0, wall, np.array(roof))
    elif zone == "commercial":
        if rng.random() < 0.08:
            return                                                     # empty lot / parking
        floors = int(np.clip(rng.choice([1, 2, 2, 3, 3, 4, 5], p=[.2, .22, .18, .16, .12, .08, .04]), 1, 6))
        hu, hv = (pw - 2.5) / 2 * rng.uniform(0.8, 1.0), (pd - 2.5) / 2 * rng.uniform(0.8, 1.0)
        wall = _col(SHOP_BRIGHT if rng.random() < 0.28 else WALL_PAINT + WALL_CONCRETE, rng, 8)
        if floors >= 3 and rng.random() < 0.4:
            wall = _col(GLASS[2:], rng, 8)
        pitched = floors <= 2 and rng.random() < 0.45
        buf.add(uc, vc, hu, hv, floors * 3.3, floors, KIND_SHOP, ROOF_GABLE if pitched else ROOF_FLAT,
                rng.uniform(1.0, 1.8) if pitched else 0.0, wall,
                _col(ROOF_IRON, rng) if pitched else np.array((96, 98, 100.0)), 0.0)
    elif zone == "residential":
        if rng.random() < (0.10 if not hilltop else 0.02):
            return
        for _ in range(1 + (1 if (hilltop and rng.random() < 0.3) else 0)):
            w = rng.uniform(7.5, 13.5) * (1.3 if hilltop else 1.0)
            d = rng.uniform(6.0, 10.0) * (1.3 if hilltop else 1.0)
            w, d = min(w, pw - 5.0), min(d, pd - 5.0)
            if w < 4.0 or d < 4.0:
                return
            u = rng.uniform(pu0 + 2.5 + w / 2, pu1 - 2.5 - w / 2) if pw - 5.0 > w else uc
            v = rng.uniform(pv0 + 2.5 + d / 2, pv1 - 2.5 - d / 2) if pd - 5.0 > d else vc
            floors = int(rng.choice([1, 1, 1, 2, 2, 3], p=[.34, .22, .1, .2, .1, .04]))
            wall = _col(WALL_BRICK if rng.random() < 0.3 else WALL_PAINT, rng, 8)
            buf.add(u, v, w / 2, d / 2, floors * 3.0 + 0.4, floors, KIND_HOUSE, ROOF_GABLE, rng.uniform(1.1, 2.0),
                    wall, _col(ROOF_IRON, rng), rng.uniform(-0.05, 0.05))
    else:                                                              # informal settlement
        for _ in range(int(rng.integers(1, 3))):
            w, d = rng.uniform(3.4, 6.0), rng.uniform(3.0, 4.8)
            w, d = min(w, pw - 1.0), min(d, pd - 1.0)
            if w < 2.6 or d < 2.4:
                continue
            u = rng.uniform(pu0 + w / 2 + 0.3, max(pu0 + w / 2 + 0.31, pu1 - w / 2 - 0.3))
            v = rng.uniform(pv0 + d / 2 + 0.3, max(pv0 + d / 2 + 0.31, pv1 - d / 2 - 0.3))
            buf.add(u, v, w / 2, d / 2, 2.5, 1, KIND_HOUSE, ROOF_GABLE, rng.uniform(0.5, 0.9),
                    _col(WALL_BRICK + MUD + WALL_CONCRETE, rng, 10),
                    _col([(142, 62, 44), (158, 158, 160), (132, 70, 50), (120, 120, 124), (176, 98, 58)], rng, 10),
                    rng.uniform(-0.15, 0.15))


def _finish_buildings(buf: _BuildingBuffer, lat: _Lattice, occ: _Occ, rng) -> Buildings:
    """Map lattice-space parcels to world space (warp + rotation) and settle each on the ground."""
    if not buf.rows:
        return Buildings.empty()
    a = np.array([r[:9] for r in buf.rows], float)
    u, v, hu, hv = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x, y = lat.to_world(u, v)
    yaw = lat.yaw_at(u, v) + np.array([r[11] for r in buf.rows])
    wall_rgb = np.array([r[9] for r in buf.rows], float)
    roof_rgb = np.array([r[10] for r in buf.rows], float)
    # pitched roofs run their ridge along the building's own x axis, so make x the long side
    swap = (hv > hu) & (a[:, 7] == ROOF_GABLE)
    hx, hy = np.where(swap, hv, hu), np.where(swap, hu, hv)
    yaw = np.where(swap, yaw + math.pi / 2, yaw)
    # ground level = lowest of the centre and four corners, so no wall floats above a slope
    c, s = np.cos(yaw), np.sin(yaw)
    g = np.full(len(a), 1e9)
    for sx, sy in ((0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        j, i = occ.idx(x + c * hx * sx - s * hy * sy, y + s * hx * sx + c * hy * sy)
        g = np.minimum(g, occ.ground[j, i])
    return Buildings(np.stack([x, y, g], axis=1), np.stack([hx, hy], axis=1), a[:, 4], yaw,
                     a[:, 7].astype(np.int8), a[:, 8], a[:, 5].astype(np.int16), a[:, 6].astype(np.int8),
                     wall_rgb, roof_rgb)


def _stamp_buildings(occ: _Occ, b: Buildings) -> None:
    for i in range(len(b)):
        occ.stamp_box(b.pos[i, 0], b.pos[i, 1], b.half[i, 0], b.half[i, 1], b.yaw[i], float(b.wall[i] + b.rise[i]))


def _reject_overlaps(b: Buildings, occ: _Occ, extra_free: np.ndarray | None = None) -> Buildings:
    """Drop buildings that sit on a road, in water or on a cliff (checked at the centre and four corners)."""
    if not len(b):
        return b
    keep = np.ones(len(b), bool)
    c, s = np.cos(b.yaw), np.sin(b.yaw)
    for sx, sy in ((0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        px = b.pos[:, 0] + c * b.half[:, 0] * sx - s * b.half[:, 1] * sy
        py = b.pos[:, 1] + s * b.half[:, 0] * sx + c * b.half[:, 1] * sy
        j, i = occ.idx(px, py)
        keep &= (occ.ground[j, i] > 0.9) & ~occ.road[j, i] & (occ.slope[j, i] < 0.6)
    return b.select(keep)


# ================================================================================================ 6. trees & cover
def _plant_trees(ctx: _Ctx, occ: _Occ, b: Buildings, roads: list[RoadLine], zones_xy, rural_r: float, R0: float,
                 park, rng) -> Trees:
    city = ctx.city
    xs: list = []
    ys: list = []
    kinds: list = []
    hs: list = []
    rs: list = []
    tones: list = []

    def add(x, y, kind, h, r, tone):
        xs.append(x); ys.append(y); kinds.append(kind); hs.append(h); rs.append(r); tones.append(tone)

    flowering_share = 0.5 if city.id == "kampala" else 0.22
    avenue_kind = {"savanna": BROADLEAF, "semiarid": ACACIA}.get(city.cover, BROADLEAF)

    # street trees (avenues) along primary / secondary roads
    for ln in roads:
        if ln.kind not in ("primary", "secondary"):
            continue
        pts = resample_polyline(ln.pts, 17.0)
        if len(pts) < 2:
            continue
        d = np.gradient(pts, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        off = ln.width / 2 + 2.6
        for side in (-1, 1):
            for p, nv in zip(pts, nrm):
                rr = math.hypot(p[0] - ctx.cx, p[1] - ctx.cy)
                if rr > rural_r + 40 or rng.random() < 0.28 + (0.25 if rr < R0 * 0.3 else 0.0):
                    continue
                q = p + nv * side * (off + rng.uniform(-0.6, 0.6))
                if rng.random() < flowering_share:
                    add(q[0], q[1], FLOWERING, rng.uniform(8, 12.5), rng.uniform(3.6, 5.2), rng.random())
                else:
                    add(q[0], q[1], avenue_kind, rng.uniform(7, 13), rng.uniform(3.4, 5.6), rng.random())

    # candidates scattered over the whole map; density fields decide groves vs open country
    cand = 26000
    px = rng.uniform(ORIGIN[0] + 15, ORIGIN[0] + (N - 1) * CELL - 15, cand)
    py = rng.uniform(ORIGIN[1] + 15, ORIGIN[1] + (N - 1) * CELL - 15, cand)
    rr = np.hypot(px - ctx.cx, py - ctx.cy)
    dens = fbm(px / 140.0, py / 140.0, city.seed + 33, 3)
    urban_frac = 1.0 - smooth((rr - R0 * 0.5) / (R0 * 0.8))
    cover = city.cover
    if cover == "savanna":
        p_keep = 0.10 + 0.5 * smooth((dens - 0.6) / 0.25)
    elif cover == "semiarid":
        p_keep = 0.05 + 0.15 * smooth((dens - 0.55) / 0.3)
    elif cover == "tea":
        p_keep = 0.12 + 0.55 * smooth((dens - 0.55) / 0.3)
    else:
        p_keep = 0.25 + 0.7 * smooth((dens - 0.45) / 0.3)
    p_keep = p_keep * (0.25 + 0.55 * urban_frac) if cover != "semiarid" else p_keep
    keep = rng.random(cand) < p_keep
    keep &= occ.free(px, py, margin=2.0)
    # gardens in town: every parcel keeps a few shade trees, hilltops get many
    px, py, rr, dens = px[keep], py[keep], rr[keep], dens[keep]
    for x, y, r_, dn in zip(px, py, rr, dens):
        u = rng.random()
        hp = float(bilinear(ctx.h, x, y, ORIGIN[0], ORIGIN[1], CELL))
        if cover == "savanna" or cover == "semiarid":
            if u < 0.62:
                add(x, y, ACACIA, rng.uniform(4.5, 8.5), rng.uniform(3.6, 6.2), rng.random())
            else:
                add(x, y, BROADLEAF, rng.uniform(6, 11), rng.uniform(3.0, 5.2), rng.random())
        elif r_ > rural_r and cover in ("banana", "cropland") and u < 0.38:
            add(x, y, BANANA, rng.uniform(3.0, 4.4), rng.uniform(1.6, 2.3), rng.random())
        elif hp > 30 and u < 0.55:
            add(x, y, EUCALYPTUS, rng.uniform(14, 26), rng.uniform(1.6, 2.6), rng.random())
        elif u < 0.07 and city.id in ("entebbe", "kampala", "jinja"):
            add(x, y, PALM, rng.uniform(8, 15), rng.uniform(2.2, 3.0), rng.random())
        elif u < 0.12:
            add(x, y, FLOWERING, rng.uniform(8, 12), rng.uniform(3.4, 5.0), rng.random())
        else:
            add(x, y, BROADLEAF, rng.uniform(6, 13), rng.uniform(3.0, 5.6), rng.random())

    # banana plantations: tight grids of banana plants in patchwork farm cells
    if cover in ("banana",):
        for _ in range(26):
            x0 = rng.uniform(ORIGIN[0] + 60, ORIGIN[0] + (N - 1) * CELL - 90)
            y0 = rng.uniform(ORIGIN[1] + 60, ORIGIN[1] + (N - 1) * CELL - 90)
            if math.hypot(x0 - ctx.cx, y0 - ctx.cy) < rural_r * 0.9:
                continue
            for gx in range(0, 7):
                for gy in range(0, 6):
                    x, y = x0 + gx * 5.5 + rng.normal(0, 0.7), y0 + gy * 5.5 + rng.normal(0, 0.7)
                    if bool(occ.free(np.array([x]), np.array([y]))[0]):
                        add(x, y, BANANA, rng.uniform(3.2, 4.6), rng.uniform(1.6, 2.3), rng.random())

    # lawn trees around the park
    if park is not None:
        pcx, pcy, hx, hy, yaw = park
        c, s = math.cos(yaw), math.sin(yaw)
        for _ in range(14):
            lx, ly = rng.choice([-1, 1]) * rng.uniform(hx * 0.55, hx * 0.95), rng.uniform(-hy, hy)
            if rng.random() < 0.5:
                lx, ly = rng.uniform(-hx, hx), rng.choice([-1, 1]) * rng.uniform(hy * 0.6, hy * 0.95)
            add(pcx + c * lx - s * ly, pcy + s * lx + c * ly, BROADLEAF if rng.random() < 0.7 else FLOWERING,
                rng.uniform(8, 13), rng.uniform(4.0, 5.8), rng.random())

    # garden trees beside houses
    if len(b):
        n_garden = min(len(b), 900)
        for i in rng.choice(len(b), n_garden, replace=False):
            if b.kind[i] not in (KIND_HOUSE, KIND_HUT) or b.floors[i] > 3:
                continue
            a = rng.uniform(0, 2 * math.pi)
            dd = math.hypot(*b.half[i]) + rng.uniform(2.5, 7.0)
            x, y = b.pos[i, 0] + math.cos(a) * dd, b.pos[i, 1] + math.sin(a) * dd
            if bool(occ.free(np.array([x]), np.array([y]), margin=1.0)[0]) and rng.random() < 0.8:
                if rng.random() < 0.25 and cover in ("banana", "cropland"):
                    add(x, y, BANANA, rng.uniform(3.0, 4.2), rng.uniform(1.6, 2.2), rng.random())
                else:
                    add(x, y, rng.choice([BROADLEAF, BROADLEAF, ACACIA if cover != "banana" else FLOWERING]),
                        rng.uniform(5.5, 10.5), rng.uniform(2.8, 4.6), rng.random())

    if not xs:
        return Trees(np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0))
    x, y = np.array(xs), np.array(ys)
    # final filter: dry, gentle ground not under a building or road (street trees were placed at road margin)
    j, i = occ.idx(x, y)
    ok = (occ.ground[j, i] > 0.9) & (occ.slope[j, i] < 0.6) & ~occ.road[j, i] & (occ.height[j, i] <= 0)
    # trees must not stand on the launch pad
    ok &= np.hypot(x - ctx.pad[0], y - ctx.pad[1]) > 16.0 if hasattr(ctx, "pad") else True
    gz = bilinear(ctx.h, x, y, ORIGIN[0], ORIGIN[1], CELL)
    pos = np.stack([x, y, gz], axis=1)
    t = Trees(pos[ok], np.array(hs)[ok], np.array(rs)[ok], np.array(tones)[ok], np.array(kinds, np.int8)[ok])
    if len(t.height) > 16000:                                           # keep the per-frame cull cheap
        t = t.select(rng.choice(len(t.height), 16000, replace=False))
    return t


def _landcover(ctx: _Ctx, occ: _Occ, zones_xy, rural_r: float, R0: float, park) -> np.ndarray:
    city = ctx.city
    X, Y, h = ctx.X, ctx.Y, ctx.h
    cov = np.full(h.shape, GRASS, np.uint8)
    pid, pdist = voronoi_ids(X, Y, 85.0 if city.terrace == 0 else 48.0, city.seed + 3)
    gy, gx = np.gradient(h, CELL)
    slope = np.hypot(gx, gy)
    r = np.hypot(X - ctx.cx, Y - ctx.cy)
    style = city.cover
    if style == "banana":
        cov = np.where(pid < 0.40, FARM_GREEN, np.where(pid < 0.62, PLANTATION, np.where(pid < 0.78, FARM_DRY, GRASS))).astype(np.uint8)
    elif style == "cropland":
        cov = np.where(pid < 0.35, FARM_GREEN, np.where(pid < 0.6, FARM_DRY, np.where(pid < 0.75, LATERITE, GRASS))).astype(np.uint8)
    elif style == "tea":
        cov = np.where(pid < 0.6, TEA, np.where(pid < 0.8, FARM_GREEN, GRASS)).astype(np.uint8)
    elif style == "savanna":
        cov = np.where(pid < 0.18, FARM_DRY, np.where(pid < 0.28, FARM_GREEN, GRASS)).astype(np.uint8)
    elif style == "semiarid":
        cov = np.where(pid < 0.55, LATERITE, np.where(pid < 0.78, FARM_DRY, GRASS)).astype(np.uint8)
    urban = r < R0 * (0.95 + 0.12 * (fbm(X / 170.0, Y / 170.0, city.seed + 77, 3) - 0.5))
    # gardens/lawns are mottled in with bare urban ground
    mott = fbm(X / 24.0, Y / 24.0, city.seed + 12, 2)
    cov = np.where(urban & (mott > 0.42), URBAN, np.where(urban, LAWN, cov)).astype(np.uint8)
    cov = np.where(urban & (r < R0 * 0.55), URBAN, cov).astype(np.uint8)
    cov = np.where((slope > 0.55), ROCK_C, cov).astype(np.uint8)
    # wet valley floors: papyrus swamp; fringe of sand at lake shores
    wet = (h < 2.4) & (h > -0.3)
    cov = np.where(wet & (ctx.river_d < 160), WETLAND, cov).astype(np.uint8)
    cov = np.where((h < 0.9) & (h > -0.3) & (ctx.river_d > 160), SAND, cov).astype(np.uint8)
    for f in city.features:
        if f.kind == "rock":
            v = bearing_vec(f.bearing)
            d = np.hypot(X - (ctx.cx + v[0] * f.dist), Y - (ctx.cy + v[1] * f.dist))
            cov = np.where(d < f.b * 0.85, ROCK_C, cov).astype(np.uint8)
    # roads and buildings leave scars of dusty ground in the verges
    j = np.clip(((Y - ORIGIN[1]) / OCC).astype(int), 0, M - 1)
    i = np.clip(((X - ORIGIN[0]) / OCC).astype(int), 0, M - 1)
    built = occ.height[j, i] > 0
    cov = np.where(built, URBAN, cov).astype(np.uint8)
    if park is not None:
        pcx, pcy, hx, hy, yaw = park
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = X - pcx, Y - pcy
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        cov = np.where((np.abs(lx) < hx) & (np.abs(ly) < hy), LAWN, cov).astype(np.uint8)
    return cov


# ================================================================================================ 7. extras
def _make_backdrop(city: City, rng) -> list:
    """Three horizon layers (far -> near). Each: dict(az, el, rgb, snow_line) with el in radians.

    Azimuth `az` is the world-frame angle atan2(y, x) of the direction, i.e. -compass bearing.
    """
    az = np.linspace(0, 2 * math.pi, 361)
    layers = []
    haze = city.haze
    base_cols = [(150, 170, 196), (132, 156, 182), (112, 140, 160)]
    for k, (base, amp) in enumerate(((0.020, 0.030), (0.012, 0.022), (0.006, 0.014))):
        n = np.zeros_like(az)
        for q in range(1, 5):
            n += rng.uniform(0.4, 1.0) / q * np.sin(az * q * (3 + 2 * k) + rng.uniform(0, 6.28))
        prof = base + amp * (n - n.min()) / (n.max() - n.min())
        snow = np.full_like(az, 9.0)
        for pk in city.peaks:
            centre = -math.radians(pk.bearing)
            da = (az - centre + math.pi) % (2 * math.pi) - math.pi
            w = math.radians(pk.width) / 2.0
            ridge = 0.8 + 0.4 * np.abs(np.sin(da * 11 + pk.bearing)) ** 0.7
            bump = math.radians(pk.height) * np.exp(-(da / w) ** 2) * (1.0 if k == 0 else (0.74 if k == 1 else 0.0)) * ridge
            if pk.snow and k == 0:
                snow = np.minimum(snow, np.where(bump > math.radians(pk.height) * 0.72, math.radians(pk.height) * 0.72, 9.0))
            prof = np.maximum(prof, prof + bump) if k < 2 else prof
        prof[-1] = prof[0]
        snow[-1] = snow[0]
        mix = 0.35 + 0.35 * haze
        rgb = tuple(int(c + (205 - c) * mix * (1.0 - 0.3 * k)) for c in base_cols[k])
        layers.append({"az": az, "el": prof, "rgb": rgb, "snow": snow})
    return layers


def _routes(roads: list[RoadLine], occ: _Occ):
    routes = []
    for ln in roads:
        if ln.kind not in ("primary", "secondary", "local") or len(ln.pts) < 6:
            continue
        pts = resample_polyline(ln.pts, 8.0)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        routes.append((pts, cum, ln.kind))
    return routes


def _spawn_vehicles(routes, city: City, rng) -> list[Vehicle]:
    veh: list[Vehicle] = []
    if not routes:
        return veh
    weights = np.array([{"primary": 4.0, "secondary": 2.0, "local": 0.6}[r[2]] * r[1][-1] for r in routes])
    weights /= weights.sum()
    count = int(np.clip(70 * city.density, 22, 110))
    colors = [(210, 212, 216), (36, 40, 46), (150, 24, 30), (40, 70, 130), (226, 226, 222), (118, 124, 128), (60, 100, 70)]
    for _ in range(count):
        ri = int(rng.choice(len(routes), p=weights))
        L = routes[ri][1][-1]
        u = rng.random()
        kind = "boda" if u < 0.34 else "matatu" if u < 0.58 else "car" if u < 0.88 else "truck" if u < 0.95 else "bus"
        dirn = 1 if rng.random() < 0.5 else -1
        speed = {"boda": rng.uniform(6, 11), "matatu": rng.uniform(5, 10), "car": rng.uniform(5, 12),
                 "truck": rng.uniform(4, 8), "bus": rng.uniform(5, 9)}[kind] * dirn
        col = {"matatu": (240, 240, 238), "bus": (214, 150, 40), "truck": colors[int(rng.integers(0, len(colors)))],
               "boda": colors[int(rng.integers(0, len(colors)))],
               "car": colors[int(rng.integers(0, len(colors)))]}[kind]
        # Uganda drives on the left: lane offset is to the left of the direction of travel
        lane = (-1 if dirn > 0 else 1) * 1.7
        veh.append(Vehicle(ri, rng.uniform(0, L), speed, kind, col, lane * (0.7 if kind == "boda" else 1.0)))
    return veh


# ================================================================================================ main entry
def build_world(city: City, real=None, progress=None) -> World:
    """Generate the world for `city`. `real` optionally carries live map data (see online.RealData)."""
    say = progress or (lambda msg: None)
    rng = np.random.default_rng(city.seed * 7919 + 13)
    cx, cy = 40.0, 40.0
    say("Shaping the land")
    X, Y, h, river_d, river_w = _terrain_heights(city, cx, cy)
    if real is not None and real.heights is not None:
        h = real.heights.copy()
        river_d = real.water_distance if real.water_distance is not None else river_d
    ctx = _Ctx(city, rng, X, Y, cx, cy, h, river_d, river_w)
    R0 = 520.0 * (0.8 + 0.5 * min(1.6, city.density))
    rural_r = R0 * 1.0

    world = World(city=city, centre=(cx, cy))
    if real is not None and real.roads:
        say("Laying real streets")
        lines = real.roads
        lat = None
    else:
        say("Laying out streets")
        lat = _Lattice(city, cx, cy, rng)
        lines = _build_roads(ctx, lat, rural_r)
    lines = _clip_and_bridge(lines, ctx.h, rural_r)

    # ---- park + launch pad ------------------------------------------------------------------------------
    park_block, park = None, None
    if lat is not None:
        park_block, park = _choose_park(ctx, lat, lines)
    if park is not None:
        ctx.h = _flatten_rect(ctx, park)
    elif real is not None:
        park = _find_open_ground(ctx, real, lines)
        if park is not None:
            ctx.h = _flatten_rect(ctx, park)
    h = ctx.h
    occ = _Occ(h)
    # roads into the raster (width + 1.2 m verge)
    for ln in lines:
        if ln.bridge is not None and ln.bridge.all():
            continue
        occ.stamp_polyline(ln.pts, ln.width / 2 + 1.2, occ.road)
    world.roads = lines
    world.pad = (park[0], park[1]) if park is not None else (cx, cy)
    ctx.pad = world.pad                                   # type: ignore[attr-defined]
    world.park = park

    say("Building the town")
    if real is not None and real.buildings is not None:
        b = real.buildings
        b = _drop_near(b, world.pad, 24.0)
    else:
        buf, zones = _fill_blocks(ctx, lat, occ, park_block, rural_r, R0)
        b = _finish_buildings(buf, lat, occ, rng)
        b = _reject_overlaps(b, occ)
        b = _drop_near(b, world.pad, 24.0)
    # landmarks first: they claim space, and ordinary buildings under them are removed
    say("Raising landmarks")
    meshes, boxes, b = features.build_landmarks(city, ctx, b, occ, lines, world.pad)
    world.meshes = meshes
    _stamp_buildings(occ, b)
    for (bx, by, bz, hx, hy, hz, yaw) in boxes:           # landmark footprints block trees too
        occ.stamp_box(bx, by, hx, hy, yaw, hz * 2)
    world.buildings = b
    world.extra_colliders = boxes

    say("Planting trees")
    trees = _plant_trees(ctx, occ, b, lines, None, rural_r, R0, park, rng)
    tr = Terrain.from_arrays(h, trees, city.name)
    tr.landcover = _landcover(ctx, occ, None, rural_r, R0, park)
    tr.backdrop = _make_backdrop(city, np.random.default_rng(city.seed))
    tr.pad = world.pad
    tr.play_center = world.centre
    tr.interior = (0.0, 0.0)
    tr.world = world
    world.terrain = tr
    # tidy: keep a clear disc of air above the pad
    tr.clear_trees([world.pad], 16.0)

    # occupancy raster for runtime queries (shadows, clearance, target placement)
    world.occ_h = occ.height.copy()
    world.occ_cell, world.occ_origin = OCC, ORIGIN
    world.lattice = None if lat is None else {"lat": lat, "lines": lines, "park_block": park_block}
    say("Adding traffic")
    world.routes = _routes(lines, occ)
    world.vehicles = _spawn_vehicles(world.routes, city, rng)
    world.source = "openstreetmap" if (real is not None and real.roads) else "procedural"
    return world


def _drop_near(b: Buildings, p, radius: float) -> Buildings:
    if not len(b):
        return b
    return b.select(np.hypot(b.pos[:, 0] - p[0], b.pos[:, 1] - p[1]) > radius)


def _choose_park(ctx: _Ctx, lat: _Lattice, lines):
    """Pick the block near the centre that is flat, dry and big enough to hold a lawn + launch pad."""
    best, best_score = None, 1e9
    us, vs = lat.us, lat.vs
    for bi in range(len(us) - 1):
        for bj in range(len(vs) - 1):
            uc, vc = (us[bi] + us[bi + 1]) / 2, (vs[bj] + vs[bj + 1]) / 2
            x, y = lat.to_world(np.array(uc), np.array(vc))
            x, y = float(x), float(y)
            r = math.hypot(x - ctx.cx, y - ctx.cy)
            if r > 330 or r < 40:
                continue
            wu, wv = us[bi + 1] - us[bi] - 18.0, vs[bj + 1] - vs[bj] - 18.0
            if min(wu, wv) < 46:
                continue
            hh = bilinear(ctx.h, np.array([x, x + 20, x - 20, x, x]), np.array([y, y, y, y + 20, y - 20]),
                          ORIGIN[0], ORIGIN[1], CELL)
            if hh.min() < 3.0 or hh.max() - hh.min() > 4.0:
                continue
            score = abs(r - 150.0) + 6.0 * (hh.max() - hh.min())
            if score < best_score:
                best_score, best = score, (bi, bj, x, y, wu, wv, float(lat.yaw_at(np.array(uc), np.array(vc))))
    if best is None:
        return None, None
    bi, bj, x, y, wu, wv, yaw = best
    return (bi, bj), (x, y, wu / 2.0, wv / 2.0, yaw)


def _flatten_rect(ctx: _Ctx, park) -> np.ndarray:
    pcx, pcy, hx, hy, yaw = park
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = ctx.X - pcx, ctx.Y - pcy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    w = smooth((hx + 14 - np.abs(lx)) / 14.0) * smooth((hy + 14 - np.abs(ly)) / 14.0)
    inside = (np.abs(lx) < hx) & (np.abs(ly) < hy)
    mean = float(ctx.h[inside].mean()) if inside.any() else float(ctx.h[int((pcy - ORIGIN[1]) / CELL), int((pcx - ORIGIN[0]) / CELL)])
    return ctx.h * (1 - w) + mean * w


def _find_open_ground(ctx: _Ctx, real, lines):
    """Live-map mode: find an open square (no building / road) near the centre for the launch pad."""
    occ = _Occ(ctx.h)
    for ln in lines:
        occ.stamp_polyline(ln.pts, ln.width / 2 + 2.0, occ.road)
    if real.buildings is not None:
        _stamp_buildings(occ, real.buildings)
    free = (occ.ground > 3.0) & (occ.slope < 0.12) & ~occ.road & (occ.height <= 0)
    # a square is "open" if the 28 m box around it is entirely free: erode via cumulative sums
    k = int(28 / OCC)
    c = np.pad(np.cumsum(np.cumsum(free.astype(np.int32), 0), 1), ((1, 0), (1, 0)))
    box = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    ok = box == k * k
    jj, ii = np.nonzero(ok)
    if len(jj) == 0:
        return None
    px = ORIGIN[0] + (ii + k / 2.0) * OCC
    py = ORIGIN[1] + (jj + k / 2.0) * OCC
    d = np.hypot(px - ctx.cx, py - ctx.cy)
    n = int(np.argmin(d))
    return (float(px[n]), float(py[n]), 14.0, 14.0, 0.0)
