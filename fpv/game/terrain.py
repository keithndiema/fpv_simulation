"""Procedural world: heightfield, water, forests and distant mountains.

The same float32 heightfield is used by the renderer (this module) and the C++ physics core
(`Simulator.set_terrain`), so what you see is exactly what you can crash into.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CELL = 5.0                       # metres between height samples
N = 257                          # samples per side  -> 1280 m square world
ORIGIN = (-600.0, -600.0)        # world position of sample [0, 0]
WATER_LEVEL = 0.0
PAD_CENTER = (-4.0, 0.0)         # launch pad: flattened so take-off is always clean
PAD_RADIUS = 20.0
PAD_HEIGHT = 5.0
PLAY_CENTER = (40.0, 45.0)       # middle of the playable area; hills rise outside it


def _smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def _value_noise(rng: np.random.Generator, n: int, freq: int) -> np.ndarray:
    """Smooth 2-D value noise in [0, 1] sampled on an n x n grid with `freq` lattice cells."""
    lat = rng.random((freq + 2, freq + 2))
    u = np.linspace(0, freq, n, endpoint=False)
    i = u.astype(int)
    f = _smooth(u - i)
    a = lat[np.ix_(i, i)]
    b = lat[np.ix_(i, i + 1)]
    c = lat[np.ix_(i + 1, i)]
    d = lat[np.ix_(i + 1, i + 1)]
    fx, fy = f[None, :], f[:, None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# tree species codes (how the renderer draws them)
CONIFER, BROADLEAF, ACACIA, EUCALYPTUS, PALM, BANANA, FLOWERING = range(7)


@dataclass
class Trees:
    pos: np.ndarray        # (n, 3) base position on the ground
    height: np.ndarray     # (n,)
    radius: np.ndarray     # (n,) canopy radius
    tone: np.ndarray       # (n,) 0..1 colour variation
    kind: np.ndarray | None = None   # (n,) species code; None = all conifers (the original valley)

    def __post_init__(self):
        if self.kind is None:
            self.kind = np.zeros(len(self.height), np.int8)

    def select(self, keep) -> "Trees":
        return Trees(self.pos[keep], self.height[keep], self.radius[keep], self.tone[keep], self.kind[keep])


class Terrain:
    def __init__(self, seed: int = 5):
        rng = np.random.default_rng(seed)
        xs = ORIGIN[0] + np.arange(N) * CELL
        X, Y = np.meshgrid(xs, ORIGIN[1] + np.arange(N) * CELL)

        # --- fractal base (hills) -------------------------------------------------------------
        fbm = np.zeros((N, N))
        amp, tot = 1.0, 0.0
        for octave in range(6):
            fbm += amp * _value_noise(rng, N, 3 * 2 ** octave)
            tot += amp
            amp *= 0.5
        fbm /= tot
        fbm = np.clip(0.5 + (fbm - fbm.mean()) / (fbm.std() * 6.0), 0.0, 1.0)   # re-centred to mean 0.5, ~[0.2, 0.8]
        ridged = 1.0 - np.abs(2 * fbm - 1)                    # sharp crests for the mountains

        # --- shape: gentle playground in the middle, big mountains ringing it ------------------
        r = np.hypot(X - PLAY_CENTER[0], Y - PLAY_CENTER[1])
        rise = _smooth(np.clip((r - 230.0) / 330.0, 0, 1))    # 0 inside, 1 far out
        rolling = (fbm - 0.5) * 24.0 + 9.0                    # rolling hills, roughly 0-20 m in the play area
        mountains = ridged ** 1.6 * 150.0 + fbm * 40.0
        h = rolling * (1 - rise) + (rolling + mountains) * rise

        # --- a lake: smooth basin pushed below the water line --------------------------------
        lake_c, lake_r = (95.0, -35.0), 55.0
        d = np.hypot(X - lake_c[0], Y - lake_c[1])
        h -= 15.0 * _smooth(np.clip(1 - d / lake_r, 0, 1))

        # --- launch pad ----------------------------------------------------------------------
        dp = np.hypot(X - PAD_CENTER[0], Y - PAD_CENTER[1])
        k = _smooth(np.clip(1 - (dp - PAD_RADIUS * 0.55) / (PAD_RADIUS * 0.9), 0, 1))
        h = h * (1 - k) + PAD_HEIGHT * k

        self.heights = h.astype(np.float32)
        self.cell, self.ox, self.oy, self.n = CELL, ORIGIN[0], ORIGIN[1], N
        self.water_level = WATER_LEVEL
        self._trees_all = self._plant_trees(rng, fbm)
        self.trees = self._trees_all
        self.ridges = self._ridge_profiles(rng)
        # extras that city worlds fill in (the classic valley leaves them at their defaults)
        self.name = "Practice valley"
        self.landcover = None            # (N, N) uint8 ground-cover codes, see render/scene.py
        self.backdrop = None             # list of horizon layers (see world/generator.py)
        self.pad = PAD_CENTER
        self.play_center = PLAY_CENTER
        self.interior = (0.35, 12.0)     # mission area offset from the pad is  radius * a + b  (legacy layout)
        self.world = None                # world.model.World for cities

    @classmethod
    def from_arrays(cls, heights: np.ndarray, trees: Trees, name: str = "", water_level: float = 0.0) -> "Terrain":
        """Build a terrain from a ready-made heightfield (used by the city generator and the live-map loader)."""
        self = cls.__new__(cls)
        self.heights = np.ascontiguousarray(heights, dtype=np.float32)
        self.cell, self.ox, self.oy, self.n = CELL, ORIGIN[0], ORIGIN[1], self.heights.shape[0]
        self.water_level = water_level
        self._trees_all = self.trees = trees
        self.ridges = []
        self.name = name
        self.landcover = None
        self.backdrop = None
        self.pad = PAD_CENTER
        self.play_center = PLAY_CENTER
        self.interior = (0.35, 12.0)
        self.world = None
        return self

    def mission_centre(self, radius: float) -> tuple[float, float]:
        """Centre of the area where mission targets are scattered (relative to the launch pad)."""
        a, b = self.interior
        return (self.pad[0] + a * radius, self.pad[1] + b)

    # ------------------------------------------------------------------ queries
    def height(self, x, y):
        """Bilinear height at world (x, y); accepts scalars or arrays (matches the C++ implementation)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        u = np.clip((x - self.ox) / self.cell, 0, self.n - 1.000001)
        v = np.clip((y - self.oy) / self.cell, 0, self.n - 1.000001)
        i, j = u.astype(int), v.astype(int)
        fu, fv = u - i, v - j
        H = self.heights
        out = (H[j, i] * (1 - fu) + H[j, i + 1] * fu) * (1 - fv) + (H[j + 1, i] * (1 - fu) + H[j + 1, i + 1] * fu) * fv
        return float(out) if out.ndim == 0 else out

    def slope(self, x, y):
        e = self.cell * 0.5
        gx = (self.height(x + e, y) - self.height(x - e, y)) / (2 * e)
        gy = (self.height(x, y + e) - self.height(x, y - e)) / (2 * e)
        return np.hypot(gx, gy)

    def ground(self, x, y):
        """Height of whatever you would land on (water counts as its surface)."""
        return np.maximum(self.height(x, y), self.water_level)

    # ------------------------------------------------------------------ scenery
    def _plant_trees(self, rng: np.random.Generator, fbm: np.ndarray) -> Trees:
        count = 9000
        px = rng.uniform(ORIGIN[0] + 20, ORIGIN[0] + (N - 1) * CELL - 20, count)
        py = rng.uniform(ORIGIN[1] + 20, ORIGIN[1] + (N - 1) * CELL - 20, count)
        # forest density follows a second noise field so trees clump into groves with open meadows between
        dens = _value_noise(np.random.default_rng(99), N, 9)
        di = ((px - self.ox) / self.cell).astype(int)
        dj = ((py - self.oy) / self.cell).astype(int)
        h = self.heights[dj, di]
        keep = (dens[dj, di] > 0.52) & (h > self.water_level + 1.2) & (h < 90)
        keep &= np.hypot(px - PAD_CENTER[0], py - PAD_CENTER[1]) > PAD_RADIUS + 6
        gy, gx = np.gradient(self.heights, self.cell)             # no trees on cliffs
        keep &= np.hypot(gx[dj, di], gy[dj, di]) < 0.7
        px, py, h = px[keep], py[keep], h[keep]
        n = len(px)
        height = rng.uniform(5.0, 11.0, n) * (1 - np.clip((h - 40) / 80, 0, 0.6))
        radius = height * rng.uniform(0.17, 0.24, n)
        return Trees(np.stack([px, py, h], axis=1), height, radius, rng.random(n))

    def _ridge_profiles(self, rng: np.random.Generator):
        """Two layers of far mountain silhouettes: (elevation angle [rad] per azimuth sample)."""
        az = np.linspace(0, 2 * math.pi, 361)
        layers = []
        for base, amp, f in ((0.07, 0.10, 5), (0.03, 0.07, 9)):
            a = np.zeros_like(az)
            for k in range(1, 5):
                a += rng.uniform(0.4, 1.0) / k * np.sin(f * k * 0.37 * az + rng.uniform(0, 6.28))
            a = base + amp * (a - a.min()) / (a.max() - a.min())
            a[-1] = a[0]
            layers.append((az, a))
        return layers

    def clear_trees(self, points, radius: float) -> None:
        """Remove trees within `radius` of any (x, y) point (used to keep gates and targets accessible)."""
        t = self._trees_all                      # always filter from the full forest so switching courses is clean
        keep = np.ones(len(t.height), bool)
        for x, y in points:
            keep &= np.hypot(t.pos[:, 0] - x, t.pos[:, 1] - y) > radius
        self.trees = t.select(keep)

    def tree_colliders(self, centre, radius: float) -> list[tuple]:
        """Box colliders (cx, cy, cz, hx, hy, hz, yaw) for trees within `radius` of centre."""
        t = self.trees
        near = np.hypot(t.pos[:, 0] - centre[0], t.pos[:, 1] - centre[1]) < radius
        out = []
        for p, h, r in zip(t.pos[near], t.height[near], t.radius[near]):
            w = r * 0.55
            out.append((float(p[0]), float(p[1]), float(p[2] + h / 2), w, w, float(h / 2), 0.0))
        return out
