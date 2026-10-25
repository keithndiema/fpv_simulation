"""Small numpy toolbox for procedural worlds: hash noise, polygon/polyline rasterising, blur, smoothstep.

Everything here is pure numpy so world generation needs no extra dependencies.
"""
from __future__ import annotations

import math

import numpy as np

_M32 = 0xFFFFFFFF


def smooth(t):
    """Hermite smoothstep on [0, 1] (clamped)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bearing_vec(bearing_deg: float) -> np.ndarray:
    """Compass bearing -> unit vector in the world frame (+x north, +y west)."""
    b = math.radians(bearing_deg)
    return np.array([math.cos(b), -math.sin(b)])


def _hash01(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer lattice hash -> float in [0, 1). Deterministic and vectorised (works on any shape)."""
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + int(seed) * 2246822519) & _M32
    h = ((h ^ (h >> 13)) * 1274126177) & _M32
    h = (h ^ (h >> 16)) & _M32
    return h / float(_M32 + 1)


def value_noise(x, y, seed: int = 0):
    """Smooth 2-D value noise in [0, 1] evaluated at arbitrary points (x, y in lattice units)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = smooth(x - ix), smooth(y - iy)
    a, b = _hash01(ix, iy, seed), _hash01(ix + 1, iy, seed)
    c, d = _hash01(ix, iy + 1, seed), _hash01(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x, y, seed: int = 0, octaves: int = 5, gain: float = 0.5):
    """Fractal sum of value noise, normalised to roughly [0, 1]."""
    total, amp, norm, f = 0.0, 1.0, 0.0, 1.0
    for o in range(octaves):
        total = total + amp * value_noise(x * f, y * f, seed + 101 * o)
        norm += amp
        amp *= gain
        f *= 2.0
    return total / norm


def voronoi_ids(x, y, cell: float, seed: int = 0):
    """Patchwork cells (like farm plots): returns (id, nearest-seed distance in cells) for points x, y.

    Seeds sit on a jittered lattice; the nearest of the 3x3 neighbouring seeds wins, which is exact for
    jitter < 0.5 and plenty for scenery.
    """
    u, v = np.asarray(x, float) / cell, np.asarray(y, float) / cell
    iu, iv = np.floor(u), np.floor(v)
    best = np.full(u.shape, 1e9)
    bid = np.zeros(u.shape)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            cu, cv = iu + du, iv + dv
            su = cu + 0.15 + 0.7 * _hash01(cu, cv, seed)
            sv = cv + 0.15 + 0.7 * _hash01(cv, cu, seed + 7)
            d = (u - su) ** 2 + (v - sv) ** 2
            better = d < best
            best = np.where(better, d, best)
            bid = np.where(better, _hash01(cu, cv, seed + 13), bid)
    return bid, np.sqrt(best)


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur of radius r cells using cumulative sums (edge-clamped)."""
    if r <= 0:
        return a.copy()
    out = a.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out


def dist_to_polyline(X: np.ndarray, Y: np.ndarray, pts: np.ndarray, chunk: int = 24) -> np.ndarray:
    """Distance from every grid point to a polyline (n, 2), computed segment-wise in chunks (memory-safe)."""
    best = np.full(X.shape, 1e9)
    for s in range(0, len(pts) - 1, chunk):
        for k in range(s, min(s + chunk, len(pts) - 1)):
            a, b = pts[k], pts[k + 1]
            ab = b - a
            L2 = float(ab @ ab) + 1e-9
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0.0, 1.0)
            d = np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1]))
            np.minimum(best, d, out=best)
    return best


def polygon_mask(X: np.ndarray, Y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test for a whole grid (poly is (n, 2)); only scans the polygon's bounding box."""
    mask = np.zeros(X.shape, bool)
    x0, x1, y0, y1 = poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max()
    box = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    if not box.any():
        return mask
    px, py = X[box], Y[box]
    inside = np.zeros(px.shape, bool)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cond = (ay > py) != (by > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (bx - ax) * (py - ay) / (by - ay) + ax
        inside ^= cond & (px < xint)
    mask[box] = inside
    return mask


def resample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """Evenly re-space a polyline so consecutive points are ~step metres apart."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return pts[:1]
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def bilinear(H: np.ndarray, x, y, ox: float, oy: float, cell: float):
    """Bilinear sample of a square raster H[iy, ix] at world (x, y); clamps at the edges."""
    n = H.shape[0]
    u = np.clip((np.asarray(x, float) - ox) / cell, 0, n - 1.000001)
    v = np.clip((np.asarray(y, float) - oy) / cell, 0, n - 1.000001)
    i, j = u.astype(int), v.astype(int)
    fu, fv = u - i, v - j
    return (H[j, i] * (1 - fu) + H[j, i + 1] * fu) * (1 - fv) + (H[j + 1, i] * (1 - fu) + H[j + 1, i + 1] * fu) * fv
