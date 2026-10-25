"""Landmarks: mosques, cathedrals, towers, dams, an airport... built as small coloured meshes.

Every landmark is a handful of boxes / prisms / cylinders / domes made of flat-shaded polygons - the same
painter's-algorithm look as the rest of the scene, but with enough silhouette to be recognisable from the air.
Each builder returns (mesh, collision boxes). `build_landmarks` places them from the city profile, flattens the
ground under them, removes ordinary buildings that would overlap and cuts streets that run through them.
"""
from __future__ import annotations

import math

import numpy as np

from ..game.terrain import CELL, N, ORIGIN
from .model import Mesh, RoadLine
from .noise import bearing_vec, bilinear, dist_to_polyline, smooth

WHITE = (236, 236, 230)
CREAM = (226, 214, 182)
BRICK = (150, 78, 58)
CONCRETE = (176, 176, 172)
GLASS_BLUE = (66, 102, 130)
DOME_GOLD = (214, 176, 78)
ROOF_RED = (146, 56, 44)
GREEN = (60, 130, 80)


class _Builder:
    """Collects faces; orientation of each face's normal is fixed using a per-primitive interior reference point."""

    def __init__(self):
        self.verts: list = []
        self.faces: list = []
        self.cols: list = []
        self.emis: list = []

    def _v(self, p) -> int:
        self.verts.append(np.asarray(p, float))
        return len(self.verts) - 1

    def face(self, pts, col, ref, emissive=False):
        pts = [np.asarray(p, float) for p in pts]
        n = np.zeros(3)
        for i in range(len(pts)):                       # Newell's method: robust for any planar polygon
            a, b = pts[i], pts[(i + 1) % len(pts)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        if np.linalg.norm(n) < 1e-9:
            return
        centroid = np.mean(pts, axis=0)
        if float(n @ (centroid - np.asarray(ref, float))) < 0:
            pts = pts[::-1]
        self.faces.append(tuple(self._v(p) for p in pts))
        self.cols.append(col)
        self.emis.append(emissive)

    # --- primitives ----------------------------------------------------------------------------------
    def box(self, cx, cy, z0, hx, hy, h, yaw, col, top=None):
        c, s = math.cos(yaw), math.sin(yaw)
        P = lambda lx, ly, z: (cx + c * lx - s * ly, cy + s * lx + c * ly, z)       # noqa: E731
        ref = P(0, 0, z0 + h / 2)
        b = [P(-hx, -hy, z0), P(hx, -hy, z0), P(hx, hy, z0), P(-hx, hy, z0)]
        t = [P(-hx, -hy, z0 + h), P(hx, -hy, z0 + h), P(hx, hy, z0 + h), P(-hx, hy, z0 + h)]
        for i in range(4):
            self.face([b[i], b[(i + 1) % 4], t[(i + 1) % 4], t[i]], col, ref)
        self.face(t, top or col, ref)

    def gable(self, cx, cy, z0, hx, hy, rise, yaw, col, wall_col):
        """Roof prism sitting on top of a box whose top is at z0; ridge along local x."""
        c, s = math.cos(yaw), math.sin(yaw)
        P = lambda lx, ly, z: (cx + c * lx - s * ly, cy + s * lx + c * ly, z)       # noqa: E731
        ref = P(0, 0, z0)
        a, b, cc, d = P(-hx, -hy, z0), P(hx, -hy, z0), P(hx, hy, z0), P(-hx, hy, z0)
        r0, r1 = P(-hx, 0, z0 + rise), P(hx, 0, z0 + rise)
        self.face([a, b, r1, r0], col, ref)
        self.face([d, cc, r1, r0], col, ref)
        self.face([a, d, r0], wall_col, ref)
        self.face([b, cc, r1], wall_col, ref)

    def cylinder(self, cx, cy, z0, r0, r1, h, col, n=12, cap=None, bands=None):
        ref = (cx, cy, z0 + h / 2)
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            p = lambda a, rad, z: (cx + math.cos(a) * rad, cy + math.sin(a) * rad, z)   # noqa: E731
            if bands:
                for k in range(bands):
                    za, zb = z0 + h * k / bands, z0 + h * (k + 1) / bands
                    ra, rb = r0 + (r1 - r0) * k / bands, r0 + (r1 - r0) * (k + 1) / bands
                    self.face([p(a0, ra, za), p(a1, ra, za), p(a1, rb, zb), p(a0, rb, zb)],
                              (col if k % 2 == 0 else (200, 52, 48)), ref)
            else:
                self.face([p(a0, r0, z0), p(a1, r0, z0), p(a1, r1, z0 + h), p(a0, r1, z0 + h)], col, ref)
        if cap:
            self.face([(cx + math.cos(2 * math.pi * i / n) * r1, cy + math.sin(2 * math.pi * i / n) * r1, z0 + h)
                       for i in range(n)], cap, ref)

    def cone(self, cx, cy, z0, r, h, col, n=12):
        ref = (cx, cy, z0)
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            self.face([(cx + math.cos(a0) * r, cy + math.sin(a0) * r, z0), (cx + math.cos(a1) * r, cy + math.sin(a1) * r, z0),
                       (cx, cy, z0 + h)], col, ref)

    def dome(self, cx, cy, z0, r, col, rings=4, n=14, squash=0.9):
        ref = (cx, cy, z0)
        for k in range(rings):
            e0, e1 = (math.pi / 2) * k / rings, (math.pi / 2) * (k + 1) / rings
            r0, r1 = r * math.cos(e0), r * math.cos(e1)
            z_a, z_b = z0 + r * squash * math.sin(e0), z0 + r * squash * math.sin(e1)
            for i in range(n):
                a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
                q = [(cx + math.cos(a0) * r0, cy + math.sin(a0) * r0, z_a), (cx + math.cos(a1) * r0, cy + math.sin(a1) * r0, z_a),
                     (cx + math.cos(a1) * r1, cy + math.sin(a1) * r1, z_b), (cx + math.cos(a0) * r1, cy + math.sin(a0) * r1, z_b)]
                if k == rings - 1:
                    q = q[:2] + [q[2]]
                self.face(q, col, ref)

    def build(self, name: str, centre, radius) -> Mesh:
        V = np.array(self.verts)
        faces = self.faces
        normals = []
        for f in faces:
            p = V[list(f)]
            n = np.cross(p[1] - p[0], p[2] - p[0]) if len(f) == 3 else np.cross(p[2] - p[0], p[3 % len(f)] - p[1])
            if len(f) > 4:
                n = np.cross(p[len(f) // 3] - p[0], p[2 * len(f) // 3] - p[0])
            nn = np.linalg.norm(n)
            normals.append(n / nn if nn > 1e-9 else np.array([0, 0, 1.0]))
        return Mesh(V, faces, np.array(self.cols, float), np.array(normals), np.asarray(centre, float), float(radius), name,
                    np.array(self.emis, bool))


# ================================================================================================ landmark types
def _tower(b: _Builder, x, y, z, yaw, scale=1.0):
    h = 46 * scale
    b.box(x, y, z, 10 * scale, 10 * scale, h, yaw, GLASS_BLUE, top=CONCRETE)
    b.box(x, y, z + h, 7 * scale, 7 * scale, 16 * scale, yaw, (84, 120, 140), top=CONCRETE)
    b.box(x, y, z + h + 16 * scale, 3.0 * scale, 3.0 * scale, 9 * scale, yaw, CONCRETE)
    b.cylinder(x, y, z + h + 25 * scale, 0.5, 0.3, 12 * scale, (200, 200, 205), n=6)
    return [(x, y, z + (h + 16 * scale) / 2, 10 * scale, 10 * scale, (h + 16 * scale) / 2, yaw)]


def _mosque(b: _Builder, x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    P = lambda lx, ly: (x + c * lx - s * ly, y + s * lx + c * ly)       # noqa: E731
    b.box(x, y, z, 18, 18, 11, yaw, CREAM, top=(210, 205, 195))
    b.cylinder(x, y, z + 11, 11, 11, 4, WHITE, n=16, cap=None)
    b.dome(x, y, z + 15, 11, (226, 232, 226), rings=5, n=16)
    for lx, ly in ((-20, -20), (20, -20), (20, 20), (-20, 20)):
        px, py = P(lx, ly)
        b.cylinder(px, py, z, 1.9, 1.5, 30, WHITE, n=8)
        b.cylinder(px, py, z + 30, 2.4, 2.4, 1.4, (200, 190, 170), n=8, cap=(190, 180, 160))
        b.cone(px, py, z + 31.4, 1.7, 5.5, DOME_GOLD, n=8)
    return [(x, y, z + 7, 18, 18, 7, yaw), (x, y, z + 17, 11, 11, 4, yaw)]


def _cathedral(b: _Builder, x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    P = lambda lx, ly: (x + c * lx - s * ly, y + s * lx + c * ly)       # noqa: E731
    b.box(x, y, z, 22, 8.5, 13, yaw, BRICK, top=BRICK)
    b.gable(x, y, z + 13, 22, 8.5, 5.5, yaw, ROOF_RED, BRICK)
    b.box(x, y, z, 8.5, 20, 13, yaw, BRICK, top=BRICK)
    b.gable(x, y, z + 13, 20, 8.5, 5.5, yaw + math.pi / 2, ROOF_RED, BRICK)
    b.cylinder(x, y, z + 13, 8.5, 8.5, 5, (170, 96, 74), n=14)
    b.dome(x, y, z + 18, 8.5, (150, 70, 56), rings=4, n=14)
    b.cylinder(x, y, z + 25, 1.6, 1.6, 3, (200, 190, 170), n=8)
    b.cone(x, y, z + 28, 2.2, 6.5, (80, 80, 84), n=8)
    px, py = P(-24, 0)
    b.box(px, py, z, 4.5, 4.5, 24, yaw, BRICK, top=BRICK)
    b.cone(px, py, z + 24, 4.6, 9, ROOF_RED, n=4)
    return [(x, y, z + 8, 22, 8.5, 8, yaw), (x, y, z + 8, 8.5, 20, 8, yaw), (px, py, z + 14, 4.5, 4.5, 14, yaw)]


def _palace(b: _Builder, x, y, z, yaw):
    b.box(x, y, z, 24, 8, 8.5, yaw, CREAM)
    b.gable(x, y, z + 8.5, 24, 8, 3.6, yaw, ROOF_RED, CREAM)
    b.box(x, y, z + 8.5, 5, 5, 9, yaw, WHITE)
    b.cone(x, y, z + 17.5, 5.8, 7, ROOF_RED, n=4)
    return [(x, y, z + 6, 24, 8, 6, yaw)]


def _stadium(b: _Builder, x, y, z, yaw):
    n = 26
    a_, b_ = 70.0, 48.0
    c, s = math.cos(yaw), math.sin(yaw)
    P = lambda lx, ly, zz: (x + c * lx - s * ly, y + s * lx + c * ly, zz)       # noqa: E731
    for i in range(n):
        t0, t1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        o0 = P(a_ * math.cos(t0), b_ * math.sin(t0), z)
        o1 = P(a_ * math.cos(t1), b_ * math.sin(t1), z)
        ot0 = P(a_ * math.cos(t0), b_ * math.sin(t0), z + 17)
        ot1 = P(a_ * math.cos(t1), b_ * math.sin(t1), z + 17)
        i0 = P(0.62 * a_ * math.cos(t0), 0.62 * b_ * math.sin(t0), z + 3)
        i1 = P(0.62 * a_ * math.cos(t1), 0.62 * b_ * math.sin(t1), z + 3)
        b.face([o0, o1, ot1, ot0], (196, 196, 190), (x, y, z + 8))                  # outer wall
        b.face([ot0, ot1, i1, i0], (150, 150, 156) if i % 2 else (176, 60, 52), (x, y, z - 50))   # seating bowl (faces up/in)
    pitch = [P(0.62 * a_ * math.cos(2 * math.pi * i / n), 0.62 * b_ * math.sin(2 * math.pi * i / n), z + 2.9) for i in range(n)]
    b.face(pitch, (66, 140, 70), (x, y, z - 5))
    return [(x, y, z + 8, a_ * 0.95, b_ * 0.95, 0.0001, yaw)]          # solid walls handled by the ring boxes below


def _chimney(b: _Builder, x, y, z, yaw):
    b.cylinder(x, y, z, 4.2, 2.6, 62, (200, 200, 200), n=12, cap=(60, 60, 64), bands=6)
    b.box(x + 20, y, z, 16, 10, 16, yaw, CONCRETE)
    b.box(x + 20, y + 24, z, 12, 8, 11, yaw, (160, 150, 140))
    return [(x, y, z + 31, 4.2, 4.2, 31, 0.0), (x + 20, y, z + 8, 16, 10, 8, yaw), (x + 20, y + 24, z + 5.5, 12, 8, 5.5, yaw)]


def _mast(b: _Builder, x, y, z, yaw):
    b.cylinder(x, y, z, 1.1, 0.5, 70, (210, 210, 214), n=6, bands=7)
    return [(x, y, z + 35, 1.1, 1.1, 35, 0.0)]


def _dam(b: _Builder, x, y, z, yaw):
    b.box(x, y, z - 6, 82, 7, 22, yaw, CONCRETE, top=(150, 150, 150))
    c, s = math.cos(yaw), math.sin(yaw)
    for k in (-48, -16, 16, 48):
        px, py = x + c * k, y + s * k
        b.box(px, py, z + 16, 6, 5, 5, yaw, (120, 126, 130))
    return [(x, y, z + 5, 82, 7, 11, yaw)]


def _terminal(b: _Builder, x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    b.box(x, y, z, 52, 14, 12, yaw, (226, 228, 232), top=(70, 110, 160))
    b.box(x, y, z + 12, 46, 10, 2, yaw, (60, 100, 150))
    tx, ty = x + c * 60 - s * 4, y + s * 60 + c * 4
    b.cylinder(tx, ty, z, 3.2, 2.8, 30, WHITE, n=10)
    b.box(tx, ty, z + 30, 5.5, 5.5, 5, yaw, (80, 130, 150), top=CONCRETE)
    return [(x, y, z + 7, 52, 14, 7, yaw), (tx, ty, z + 18, 3.3, 3.3, 18, 0.0)]


def _clock(b: _Builder, x, y, z, yaw):
    b.box(x, y, z, 2.6, 2.6, 15, yaw, (236, 232, 220), top=(236, 232, 220))
    b.box(x, y, z + 15, 3.4, 3.4, 4, yaw, (236, 236, 236))
    b.cone(x, y, z + 19, 3.6, 5, ROOF_RED, n=4)
    return [(x, y, z + 11, 2.7, 2.7, 11, yaw)]


_BUILDERS = {"tower": _tower, "mosque": _mosque, "cathedral": _cathedral, "palace": _palace, "stadium": _stadium,
             "chimney": _chimney, "mast": _mast, "dam": _dam, "terminal": _terminal, "clock": _clock}
# radius (m) of ground cleared / flattened under each landmark
_CLEAR = {"tower": 24, "mosque": 36, "cathedral": 44, "palace": 36, "stadium": 78, "chimney": 40, "mast": 6,
          "dam": 90, "terminal": 70, "clock": 18}


def _on_land(h, x, y, lo=3.0):
    return float(bilinear(h, np.array([x]), np.array([y]), ORIGIN[0], ORIGIN[1], CELL)[0]) > lo


def _find_site(h, cx, cy, bearing, dist, need_land=True):
    """Start at the profile position and spiral outwards until the ground is dry and gentle."""
    v = bearing_vec(bearing)
    for extra in (0, 40, -40, 80, -80, 140, -140, 220):
        for dtheta in (0, 12, -12, 25, -25):
            vv = bearing_vec(bearing + dtheta)
            x, y = cx + vv[0] * (dist + extra), cy + vv[1] * (dist + extra)
            lim = ORIGIN[0] + 60, ORIGIN[0] + (N - 1) * CELL - 60
            if not (lim[0] < x < lim[1] and lim[0] < y < lim[1]):
                continue
            if not need_land or _on_land(h, x, y, 4.0):
                return x, y
    return None


def _flatten_disc(ctx, x, y, radius):
    d = np.hypot(ctx.X - x, ctx.Y - y)
    inner = smooth((radius + 16 - d) / 16.0)
    mask = d < radius
    mean = float(ctx.h[mask].mean()) if mask.any() else float(ctx.h[int((y - ORIGIN[1]) / CELL), int((x - ORIGIN[0]) / CELL)])
    ctx.h = ctx.h * (1 - inner) + mean * inner
    return mean


def _cut_roads(lines: list[RoadLine], x, y, radius) -> list[RoadLine]:
    out = []
    for ln in lines:
        d = np.hypot(ln.pts[:, 0] - x, ln.pts[:, 1] - y)
        ok = d > radius
        i = 0
        while i < len(ok):
            if not ok[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j > i:
                out.append(RoadLine(ln.pts[i:j + 1].copy(), ln.kind,
                                    None if ln.bridge is None else ln.bridge[i:j + 1].copy(),
                                    None if ln.z is None else ln.z[i:j + 1].copy()))
            i = j + 1
    return out


def build_landmarks(city, ctx, buildings, occ, lines: list, pad):
    """Place the profile's landmarks. Mutates `ctx.h` (flattening), `lines` (cut / runway / roundabouts) and `occ`.

    Returns (meshes, collision boxes, buildings-without-overlaps).
    """
    meshes: list[Mesh] = []
    boxes: list[tuple] = []
    keep = np.ones(len(buildings), bool)
    for lm in city.landmarks:
        builder = _BUILDERS.get(lm.kind)
        if builder is None:
            continue
        if lm.kind == "dam":
            # dams sit across the river: find the water nearest the profile position
            site = _find_site(ctx.h, ctx.cx, ctx.cy, lm.bearing, lm.dist, need_land=False)
        else:
            site = _find_site(ctx.h, ctx.cx, ctx.cy, lm.bearing, lm.dist)
        if site is None:
            continue
        x, y = site
        if math.hypot(x - pad[0], y - pad[1]) < _CLEAR[lm.kind] + 30:
            continue                                                  # never on top of the launch pad
        rng = np.random.default_rng(city.seed + int(lm.bearing * 7 + lm.dist))
        yaw = float(rng.uniform(0, math.pi))
        if lm.kind == "dam":
            yaw = math.radians(-city.features[-1].c) if city.features else yaw
            z = 0.0
            radius = _CLEAR[lm.kind]
        else:
            radius = _CLEAR[lm.kind]
            z = _flatten_disc(ctx, x, y, radius * 0.55)
        b = _Builder()
        bx = builder(b, x, y, z, yaw)
        mesh = b.build(lm.name or lm.kind, (x, y, z + 15), radius + 30)
        meshes.append(mesh)
        boxes.extend(bx)
        keep &= np.hypot(buildings.pos[:, 0] - x, buildings.pos[:, 1] - y) > radius * 0.7 + 6 if len(buildings) else keep
        lines[:] = _cut_roads(lines, x, y, radius * 0.5)
        if lm.kind == "clock":                                        # roundabout around the clock tower
            t = np.linspace(0, 2 * math.pi, 28)
            ring = np.stack([x + 15 * np.cos(t), y + 15 * np.sin(t)], axis=1)
            lines.append(RoadLine(ring, "secondary"))
        if lm.kind == "terminal":
            c, s = math.cos(yaw), math.sin(yaw)
            axis = np.array([c, s])
            lateral = np.array([-s, c])
            centre = np.array([x, y]) + lateral * 85.0
            t = np.linspace(-460, 460, 60)
            rw = np.stack([centre[0] + axis[0] * t, centre[1] + axis[1] * t], axis=1)
            # flatten a strip along the runway to its mean height, then clear streets and buildings off it
            mid = float(np.mean(bilinear(ctx.h, rw[:, 0], rw[:, 1], ORIGIN[0], ORIGIN[1], CELL)))
            dd = dist_to_polyline(ctx.X, ctx.Y, rw)
            w = smooth((90.0 - dd) / 30.0)
            ctx.h = ctx.h * (1 - w) + mid * w
            lines[:] = _cut_strip(lines, rw, 55.0)
            lines.append(RoadLine(rw, "runway"))
            occ.stamp_polyline(rw, 30.0, occ.road)
            if len(buildings):
                dist_b = np.min(np.hypot(buildings.pos[:, 0:1] - rw[None, :, 0], buildings.pos[:, 1:2] - rw[None, :, 1]), axis=1)
                keep &= dist_b > 70.0
    # refresh the occupancy raster's ground after flattening
    occ.ground = bilinear(ctx.h, *np.meshgrid(occ.xs, occ.xs), ORIGIN[0], ORIGIN[1], CELL)
    gy, gx = np.gradient(occ.ground, occ.xs[1] - occ.xs[0])
    occ.slope = np.hypot(gx, gy)
    return meshes, boxes, (buildings.select(keep) if len(buildings) else buildings)


def _cut_strip(lines, centreline: np.ndarray, half_w: float):
    out = []
    for ln in lines:
        d = np.array([dist_to_polyline(np.array([p[0]]), np.array([p[1]]), centreline)[0] for p in ln.pts])
        ok = d > half_w
        i = 0
        while i < len(ok):
            if not ok[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j > i:
                out.append(RoadLine(ln.pts[i:j + 1].copy(), ln.kind,
                                    None if ln.bridge is None else ln.bridge[i:j + 1].copy(),
                                    None if ln.z is None else ln.z[i:j + 1].copy()))
            i = j + 1
    return out
