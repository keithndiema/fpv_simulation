"""City layer of the software renderer: buildings (walls, windows, pitched roofs), road ribbons with markings,
traffic, birds and landmark meshes.

Design notes (the "why"):
  * Everything static is baked once per world / atmosphere into numpy arrays; each frame only does a handful of
    vectorised projections and then a Python loop of `pygame.draw.polygon` calls - drawing is the expensive part,
    so level-of-detail decides how many polygons each building gets.
  * Roads and their markings are returned as plain (key, polygon, colour) lists so the scene can sort them together
    with the terrain quads in ONE vectorised argsort - roads then lie correctly on hills without per-road logic.
  * Buildings, vehicles, birds and meshes are painter's-algorithm objects sorted by distance with the trees.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from ..world.model import (KIND_HOUSE, KIND_HUT, KIND_MIDRISE, KIND_SHOP, KIND_TOWER, ROAD_CLASSES, ROOF_FLAT, ROOF_GABLE,
                           ROOF_HUT, World)
from .camera import NEAR, View
from .atmosphere import Atmosphere

# object kinds understood by CityLayer.draw (scene.py uses 0..3 for tree / box / gate / target)
K_BUILDING, K_VEHICLE, K_BIRD, K_MESH = 10, 11, 12, 13

_KIND_COL = {"primary": (66, 68, 74), "secondary": (82, 83, 88), "local": (108, 104, 100), "murram": (172, 100, 70),
             "track": (166, 136, 104), "runway": (60, 62, 68)}
_KIND_ID = {k: i for i, k in enumerate(("primary", "secondary", "local", "murram", "track", "runway"))}
_ROAD_COLS = np.array([_KIND_COL[k] for k in _KIND_ID], float)
_ROAD_LIFT = 0.16
_GLASS = np.array([(70, 108, 138), (60, 114, 120), (92, 108, 138), (58, 84, 116), (84, 120, 130)], float)


def _lam(n: np.ndarray, atm: Atmosphere) -> np.ndarray:
    """Light multiplier (…, 3) for unit normals (…, 3): ambient sky + direct sun (no shadows for objects)."""
    d = np.clip(n @ atm.sun_dir, 0.0, 1.0)[..., None]
    return atm.amb_col + atm.sun_col * d


def _rgb(c) -> tuple:
    """Clamp a colour to a valid pygame (r, g, b): warm sunsets push lit colours past 255."""
    return (min(255, max(0, int(c[0]))), min(255, max(0, int(c[1]))), min(255, max(0, int(c[2]))))


def _fog(col, d, atm: Atmosphere):
    f = np.clip((np.asarray(d, float) - atm.fog_start) / (atm.fog_end - atm.fog_start), 0.0, 1.0) ** 1.15
    haze = np.array(atm.haze, float)
    col = np.asarray(col, float)
    return col + (haze - col) * (f[..., None] if np.ndim(f) else f)


def _clip_near(cam: np.ndarray) -> np.ndarray:
    out = []
    n = len(cam)
    for i in range(n):
        a, b = cam[i], cam[(i + 1) % n]
        ain, bin_ = a[0] >= NEAR, b[0] >= NEAR
        if ain:
            out.append(a)
        if ain != bin_:
            out.append(a + (b - a) * ((NEAR - a[0]) / (b[0] - a[0])))
    return np.array(out)


def _scr(v: View, cam: np.ndarray) -> np.ndarray:
    x = np.maximum(cam[..., 0], NEAR)
    return np.clip(np.stack([v.cx - v.f * cam[..., 1] / x, v.cy - v.f * cam[..., 2] / x], axis=-1), -6000, 6000)


class CityLayer:
    def __init__(self, world: World, atm: Atmosphere):
        self.world = world
        self.atm = atm
        self.t = 0.0
        self._bake_buildings()
        self._bake_roads()
        self._bake_vehicles()
        self._bake_birds()
        self.bake_light(atm)

    # ================================================================================== baking
    def _bake_buildings(self) -> None:
        b = self.world.buildings
        n = len(b)
        self.n = n
        if n == 0:
            self.W = np.zeros((0, 10, 3))
            return
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        hx, hy = b.half[:, 0], b.half[:, 1]
        z0 = b.pos[:, 2] - 1.0                                 # a metre of foundation hides slope gaps
        zt = b.pos[:, 2] + b.wall
        zr = zt + b.rise
        loc = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                        (-1, 0, 2), (1, 0, 2)], float)
        lx = loc[None, :, 0] * hx[:, None]
        ly = loc[None, :, 1] * hy[:, None]
        z = np.where(loc[None, :, 2] == 0, z0[:, None], np.where(loc[None, :, 2] == 1, zt[:, None], zr[:, None]))
        W = np.empty((n, 10, 3))
        W[..., 0] = b.pos[:, 0, None] + c[:, None] * lx - s[:, None] * ly
        W[..., 1] = b.pos[:, 1, None] + s[:, None] * lx + c[:, None] * ly
        W[..., 2] = z
        self.W = W
        self.centre = np.stack([b.pos[:, 0], b.pos[:, 1], (z0 + zr) / 2], axis=1)
        self.rad = np.hypot(hx, hy) + 0.5 * (zr - z0)
        # outward wall normals (world) and plane offsets from the centre
        nl = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], float)
        wn = np.empty((n, 4, 3))
        wn[..., 0] = c[:, None] * nl[None, :, 0] - s[:, None] * nl[None, :, 1]
        wn[..., 1] = s[:, None] * nl[None, :, 0] + c[:, None] * nl[None, :, 1]
        wn[..., 2] = 0
        self.wn = wn
        self.woff = np.stack([hy, hx, hy, hx], axis=1)
        # gable slope normals: tilt by the roof pitch (ridge runs along the local x axis)
        L = np.hypot(b.rise, hy) + 1e-9
        sa = np.stack([np.zeros(n), -b.rise / L, hy / L], axis=1)
        sb = np.stack([np.zeros(n), b.rise / L, hy / L], axis=1)

        def rot(v3):
            return np.stack([c * v3[:, 0] - s * v3[:, 1], s * v3[:, 0] + c * v3[:, 1], v3[:, 2]], axis=1)
        self.sn = np.stack([rot(sa), rot(sb)], axis=1)
        self.eave = np.stack([W[:, 4], W[:, 7]], axis=1)         # a point on each slope plane (an eave corner)
        self.top_z = zt
        self.window_rows = np.clip(b.floors, 1, 60)
        rng = np.random.default_rng(3)
        self.rand = rng.random(n)

    def _bake_roads(self) -> None:
        w = self.world
        T = w.terrain
        quads, kinds, centres, tone, bridge = [], [], [], [], []
        mk_quads, mk_cols, mk_par = [], [], []
        rng = np.random.default_rng(8)
        for ln in w.roads:
            P = ln.pts
            if len(P) < 2:
                continue
            # subdivide so each ribbon quad is at most ~10 m (it has to hug the ground)
            seg = np.hypot(*np.diff(P, axis=0).T)
            ex_p, ex_br, ex_z = [P[0]], [bool(ln.bridge[0]) if ln.bridge is not None else False], [ln.z[0] if ln.z is not None else 0.0]
            for i in range(len(P) - 1):
                k = max(1, int(math.ceil(seg[i] / 10.0)))
                for q in range(1, k + 1):
                    t = q / k
                    ex_p.append(P[i] + (P[i + 1] - P[i]) * t)
                    ex_br.append(bool(ln.bridge[i + 1]) if ln.bridge is not None else False)
                    ex_z.append((ln.z[i] + (ln.z[i + 1] - ln.z[i]) * t) if ln.z is not None else 0.0)
            P, br, zz = np.array(ex_p), np.array(ex_br), np.array(ex_z)
            d = np.gradient(P, axis=0)
            d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
            # mitred normals so the ribbon has no gaps / overlaps at bends
            hw = ln.width / 2.0
            L = P + nrm * hw
            R = P - nrm * hw
            ground = lambda X: T.height(X[:, 0], X[:, 1])                          # noqa: E731
            gL, gR = ground(L), ground(R)
            zL = np.where(br, zz, gL) + _ROAD_LIFT
            zR = np.where(br, zz, gR) + _ROAD_LIFT
            # a far-LOD version sits on the coarse terrain mesh (15 m cells) so it never sinks under a facet
            cL, cR = self._coarse(T, L), self._coarse(T, R)
            zLf = np.where(br, zz, np.maximum(gL, cL)) + _ROAD_LIFT + 0.25
            zRf = np.where(br, zz, np.maximum(gR, cR)) + _ROAD_LIFT + 0.25
            kid = _KIND_ID[ln.kind]
            for i in range(len(P) - 1):
                near = np.array([[*L[i], zL[i]], [*L[i + 1], zL[i + 1]], [*R[i + 1], zR[i + 1]], [*R[i], zR[i]]])
                far = near.copy()
                far[:, 2] = [zLf[i], zLf[i + 1], zRf[i + 1], zRf[i]]
                quads.append(np.stack([near, far]))
                kinds.append(kid)
                centres.append((near.mean(axis=0)))
                tone.append(rng.normal(0, 0.035))
                bridge.append(bool(br[i] and br[i + 1]))
                if ln.kind in ("primary", "secondary", "local"):
                    # pavement strip on each side (lighter concrete, kerb-height): roads stop looking like ribbons
                    for sgn in (-1, 1):
                        a0 = P[i] + nrm[i] * sgn * (hw + 0.05)
                        a1 = P[i + 1] + nrm[i + 1] * sgn * (hw + 0.05)
                        b0 = P[i] + nrm[i] * sgn * (hw + (1.5 if ln.kind != "local" else 1.0))
                        b1 = P[i + 1] + nrm[i + 1] * sgn * (hw + (1.5 if ln.kind != "local" else 1.0))
                        pz = lambda X, ok: (zz[ok] if br[ok] else float(T.height(*X))) + _ROAD_LIFT + 0.03   # noqa: E731
                        mk_quads.append(np.array([[*a0, pz(a0, i)], [*a1, pz(a1, i + 1)], [*b1, pz(b1, i + 1)], [*b0, pz(b0, i)]]))
                        mk_cols.append((168, 164, 156))
                        mk_par.append(len(quads) - 1)
                if ln.kind in ("primary", "secondary", "runway") and i % 2 == 0:
                    # centre-line dash
                    a = P[i] + (P[i + 1] - P[i]) * 0.15
                    bpt = P[i] + (P[i + 1] - P[i]) * 0.65
                    hwid = 0.45 if ln.kind == "runway" else 0.11
                    zc = (zz[i] if br[i] else float(T.height(*a))) + _ROAD_LIFT + 0.05
                    zc2 = (zz[i + 1] if br[i + 1] else float(T.height(*bpt))) + _ROAD_LIFT + 0.05
                    nn = nrm[i]
                    mk_quads.append(np.array([[*(a + nn * hwid), zc], [*(bpt + nn * hwid), zc2],
                                              [*(bpt - nn * hwid), zc2], [*(a - nn * hwid), zc]]))
                    mk_cols.append((236, 226, 150) if ln.kind != "runway" else (236, 236, 236))
                    mk_par.append(len(quads) - 1)
                if ln.kind in ("primary", "runway") and ln.width > 10:
                    for sgn in (-1, 1):                                 # solid edge lines
                        e0 = P[i] + nrm[i] * sgn * (hw - 0.7)
                        e1 = P[i + 1] + nrm[i + 1] * sgn * (hw - 0.7)
                        z0_ = (zz[i] if br[i] else float(T.height(*e0))) + _ROAD_LIFT + 0.05
                        z1_ = (zz[i + 1] if br[i + 1] else float(T.height(*e1))) + _ROAD_LIFT + 0.05
                        nn = nrm[i] * 0.09
                        mk_quads.append(np.array([[*(e0 + nn), z0_], [*(e1 + nn), z1_], [*(e1 - nn), z1_], [*(e0 - nn), z0_]]))
                        mk_cols.append((230, 230, 226))
                        mk_par.append(len(quads) - 1)
        self.rq = np.array(quads) if quads else np.zeros((0, 2, 4, 3))
        self.rkind = np.array(kinds, int)
        self.rcentre = np.array(centres) if centres else np.zeros((0, 3))
        self.rtone = np.array(tone)
        self.rbridge = np.array(bridge, bool)
        self.mq = np.array(mk_quads) if mk_quads else np.zeros((0, 4, 3))
        self.mcol = np.array(mk_cols, float) if mk_cols else np.zeros((0, 3))
        self.mcentre = self.mq.mean(axis=1) if len(self.mq) else np.zeros((0, 3))
        self._mp = np.array(mk_par, int)

    @staticmethod
    def _coarse(T, P: np.ndarray) -> np.ndarray:
        """Height of the 15 m (stride-3) terrain mesh at P: bilinear between its sample points."""
        st = 3
        H = np.maximum(T.heights[::st, ::st].astype(float), T.water_level)
        cell = T.cell * st
        n = H.shape[0]
        u = np.clip((P[:, 0] - T.ox) / cell, 0, n - 1.000001)
        v = np.clip((P[:, 1] - T.oy) / cell, 0, n - 1.000001)
        i, j = u.astype(int), v.astype(int)
        fu, fv = u - i, v - j
        return (H[j, i] * (1 - fu) + H[j, i + 1] * fu) * (1 - fv) + (H[j + 1, i] * (1 - fu) + H[j + 1, i + 1] * fu) * fv

    def _bake_vehicles(self) -> None:
        self.veh = self.world.vehicles
        # python lists: scalar bisect/interpolation is much cheaper than numpy on single values
        self.routes = [(pts[:, 0].tolist(), pts[:, 1].tolist(), cum.tolist()) for pts, cum, _ in self.world.routes]

    def _bake_birds(self) -> None:
        rng = np.random.default_rng(self.world.city.seed + 5)
        n = 14
        cx, cy = self.world.centre
        self.birds = np.stack([cx + rng.uniform(-220, 220, n), cy + rng.uniform(-220, 220, n), rng.uniform(38, 85, n),
                               rng.uniform(26, 60, n), rng.uniform(0, 6.28, n), rng.uniform(0.15, 0.32, n) * rng.choice([-1, 1], n)], axis=1)

    # ================================================================================== lighting bake
    def set_atmosphere(self, atm: Atmosphere) -> None:
        self.atm = atm
        self.bake_light(atm)

    def bake_light(self, atm: Atmosphere) -> None:
        b = self.world.buildings
        if self.n:
            lw = _lam(self.wn, atm)                               # (n, 4, 3)
            self.wall_cols = b.wall_rgb[:, None, :] * lw
            # a touch of vertical-gradient grime so walls don't read as flat paint
            lr = _lam(np.array([[0, 0, 1.0]]), atm)[0]
            ls = _lam(self.sn, atm)                               # (n, 2, 3)
            flat = b.roof[:, None] == ROOF_FLAT
            self.roof_flat = b.roof_rgb * lr
            self.roof_slope = b.roof_rgb[:, None, :] * ls
            self.gable_cols = b.wall_rgb[:, None, :] * _lam(np.array([[1, 0, 0.0], [-1, 0, 0.0]]), atm)[None]
            self._win_base = np.clip(b.wall_rgb * 0.30 + np.array([20, 34, 50]) * 0.9, 0, 255)
        self.road_cols = _ROAD_COLS * _lam(np.array([[0, 0, 1.0]]), atm)[0]
        self.mark_cols = self.mcol * _lam(np.array([[0, 0, 1.0]]), atm)[0] if len(self.mcol) else self.mcol

    # ================================================================================== roads
    def road_quads(self, v: View, max_d: float, split: float, shadow_fn=None):
        """Returns ((keys, polys, cols) for d <= split, same for d > split). Keys are biased towards the camera so
        a ribbon is painted after the terrain quads it lies on."""
        if not len(self.rq):
            e = (np.empty(0), [], np.empty((0, 3)))
            return e, e
        pos = v.pos
        d3 = np.linalg.norm(self.rcentre - pos, axis=1)
        cx = (self.rcentre - pos) @ v.R[:, 0]
        sel = (d3 < max_d) & (cx > -14.0)
        idx = np.nonzero(sel)[0]
        near_mask = d3[idx] <= split
        out = []
        for part, mask in (("near", near_mask), ("far", ~near_mask)):
            ii = idx[mask]
            if not len(ii):
                out.append((np.empty(0), [], np.empty((0, 3))))
                continue
            Q = self.rq[ii, 0 if part == "near" else 1]            # (m, 4, 3)
            keys, polys, cols = self._project_quads(v, Q, d3[ii] - 7.0, self.road_cols[self.rkind[ii]] * (1 + self.rtone[ii])[:, None],
                                                    d3[ii], shadow_fn, Q.mean(axis=1))
            # markings belonging to these quads
            if len(self.mq):
                pm = np.isin(self._mark_parent(), ii)
                if pm.any():
                    mi = np.nonzero(pm)[0]
                    dm = np.linalg.norm(self.mcentre[mi] - pos, axis=1)
                    mi = mi[dm < 110.0]
                    dm = dm[dm < 110.0]
                    if len(mi):
                        k2, p2, c2 = self._project_quads(v, self.mq[mi], dm - 7.5, self.mark_cols[mi], dm, shadow_fn,
                                                         self.mq[mi].mean(axis=1), raw=True)
                        keys, polys, cols = np.concatenate([keys, k2]), polys + p2, np.concatenate([cols, c2])
            out.append((keys, polys, cols))
        return out[0], out[1]

    def _mark_parent(self):
        return self._mp

    def _project_quads(self, v: View, Q: np.ndarray, keys: np.ndarray, cols: np.ndarray, dist: np.ndarray, shadow_fn,
                       centres: np.ndarray, raw: bool = False):
        cam = (Q - v.pos) @ v.R                                    # (m, 4, 3)
        behind = (cam[..., 0] < NEAR)
        nb = behind.sum(axis=1)
        scr = _scr(v, cam)
        w, h = v.width, v.height
        onscreen = (scr[..., 0].max(axis=1) > 0) & (scr[..., 0].min(axis=1) < w) & (scr[..., 1].max(axis=1) > 0) & \
                   (scr[..., 1].min(axis=1) < h)
        if shadow_fn is not None:
            cols = cols * shadow_fn(centres[:, 0], centres[:, 1])[:, None]
        cols = _fog(cols, dist, self.atm)
        ok = onscreen & (nb == 0)
        keys_o, polys_o, cols_o = keys[ok], scr[ok].astype(int).tolist(), np.clip(cols[ok], 0, 255)
        strad = np.nonzero(onscreen & (nb > 0) & (nb < 4))[0]
        if len(strad):
            ek, ep, ec = [], [], []
            for i in strad:
                poly = _clip_near(cam[i])
                if len(poly) >= 3:
                    ep.append(_scr(v, poly).astype(int).tolist())
                    ek.append(keys[i])
                    ec.append(np.clip(cols[i], 0, 255))
            if ek:
                keys_o = np.concatenate([keys_o, ek])
                cols_o = np.concatenate([cols_o, np.array(ec)])
                polys_o = polys_o + ep
        return keys_o, polys_o, cols_o

    # ================================================================================== buildings
    def building_objects(self, v: View, max_d: float, max_n: int):
        if not self.n:
            return []
        pos = v.pos
        rel = self.centre - pos
        d = np.linalg.norm(rel, axis=1)
        fwd = rel @ v.R[:, 0]
        sel = (d < max_d) & (fwd > -self.rad)
        # cheap lateral reject: building centre must be within the view cone (+ its own radius)
        tan_half = (v.width / 2) / v.f
        lat = np.abs(rel @ v.R[:, 1])
        up = np.abs(rel @ v.R[:, 2])
        sel &= (lat - self.rad < np.maximum(fwd, 1.0) * tan_half + 6.0) & (up - self.rad < np.maximum(fwd, 1.0) * (v.height / v.f / 2 + 0.4) + 12.0) | (d < self.rad + 4)
        idx = np.nonzero(sel)[0]
        if not len(idx):
            return []
        # skip things that would be a couple of pixels
        px = v.f * self.rad[idx] / np.maximum(d[idx], 1.0)
        idx = idx[px > 1.2]
        if len(idx) > max_n:
            idx = idx[np.argpartition(d[idx], max_n)[:max_n]]
        cam = (self.W[idx] - pos) @ v.R                         # (m, 10, 3)
        scr = _scr(v, cam)
        dist = d[idx]
        # visibility of each face
        to_cam = pos - self.centre[idx]
        wv = (np.einsum("mkc,mc->mk", self.wn[idx], to_cam) > self.woff[idx])           # (m, 4)
        sv = np.einsum("mkc,mkc->mk", self.sn[idx], pos - self.eave[idx]) > 0          # (m, 2)
        tv = pos[2] > self.top_z[idx]                                                    # flat roof visible
        out = []
        fogc_w = _fog(self.wall_cols[idx], dist[:, None], self.atm)                    # (m, 4, 3)
        fogc_rs = _fog(self.roof_slope[idx], dist[:, None], self.atm)
        fogc_rf = _fog(self.roof_flat[idx], dist, self.atm)
        fogc_g = _fog(self.gable_cols[idx], dist[:, None], self.atm)
        glow = self.atm.glow
        for m, i in enumerate(idx):
            data = (m, i, cam[m], scr[m], dist[m], wv[m], sv[m], bool(tv[m]), fogc_w[m], fogc_rs[m], fogc_rf[m], fogc_g[m], glow)
            out.append((float(dist[m] - self.rad[i] * 0.5), K_BUILDING, data))
        return out

    _WALL_IDX = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))

    def _draw_building(self, surf, v: View, d):
        self._plant_f = v.f
        m, i, cam, scr, dist, wv, sv, tv, fw, frs, frf, fg, glow = d
        b = self.world.buildings
        poly = pygame.draw.polygon
        kind = int(b.kind[i])
        if kind == KIND_HUT:
            self._draw_hut(surf, v, i, cam, dist, fw)
            return
        near_clip = bool((cam[:8, 0] < NEAR).any())
        sx = scr.astype(int)

        def quad(ids, col, pts_override=None):
            if near_clip:
                q = _clip_near(cam[list(ids)])
                if len(q) < 3:
                    return None
                pts = [(int(x), int(y)) for x, y in _scr(v, q)]
            else:
                pts = [(int(sx[k, 0]), int(sx[k, 1])) for k in ids]
            poly(surf, _rgb(col), pts)
            return pts

        # walls (at most two are front-facing); windows go on top of them
        for k in range(4):
            if not wv[k]:
                continue
            ids = self._WALL_IDX[k]
            pts = quad(ids, fw[k])
            if pts is not None and not near_clip and dist < 135:
                self._windows(surf, i, k, pts, dist, fw[k], glow)
            if pts is not None and not near_clip and dist < 150:
                self._wall_detail(surf, pts, fw[k], int(b.roof[i]) == ROOF_FLAT, dist)
        roof = int(b.roof[i])
        if roof == ROOF_FLAT:
            if tv:
                quad((4, 5, 6, 7), frf)
                if not near_clip and dist < 110 and kind != KIND_TOWER:
                    self._roof_plant(surf, sx[4:8], frf, i, dist)
        else:
            if wv[1] and not near_clip:                        # gable end triangles (x+ / x-)
                poly(surf, _rgb(fg[0]), [tuple(sx[k]) for k in (5, 6, 9)])
            if wv[3] and not near_clip:
                poly(surf, _rgb(fg[1]), [tuple(sx[k]) for k in (4, 7, 8)])
            if sv[0]:
                quad((4, 5, 9, 8), frs[0])
            if sv[1]:
                quad((7, 6, 9, 8), frs[1])

    @staticmethod
    def _wall_detail(surf, P, wall_col, flat_roof: bool, dist: float) -> None:
        """Two cheap cues that make a box read as a building: a darker band where the wall meets the ground (dirt,
        splash, less sky light) and a lighter parapet lip along a flat roof's edge.  P = [bottom-a, bottom-b, top-b, top-a]."""
        p0, p1, p2, p3 = (np.array(q, float) for q in P)
        hpx = float(np.hypot(*(p3 - p0)))
        if hpx < 14:
            return
        poly = pygame.draw.polygon
        t = 0.07
        a, b = p0 + (p3 - p0) * t, p1 + (p2 - p1) * t
        poly(surf, _rgb([c * 0.66 for c in wall_col]), [tuple(map(int, q)) for q in (p0, p1, b, a)])
        if flat_roof:
            t2 = 0.965
            a2, b2 = p0 + (p3 - p0) * t2, p1 + (p2 - p1) * t2
            poly(surf, _rgb([c * 1.14 + 6 for c in wall_col]), [tuple(map(int, q)) for q in (a2, b2, p2, p3)])

    def _roof_plant(self, surf, R, roof_col, i, dist) -> None:
        """A small stair/lift housing and a water tank on flat roofs: stops rooftops reading as bare grey slabs.
        R = screen corners of the roof (4..7). Placement comes from the building index, so it is stable."""
        poly = pygame.draw.polygon
        P = np.array(R, float)
        f = self._plant_f
        for n, (u0, v0, du, dv, h_m, k) in enumerate(((0.18, 0.2, 0.2, 0.26, 2.6, 0.82), (0.62, 0.55, 0.14, 0.18, 1.7, 1.12))):
            if (i + n) % 3 == 0:
                continue
            uu, vv = (u0 + 0.35 * ((i * 7 + n * 3) % 4) / 4) % 0.7, (v0 + 0.3 * ((i * 5 + n) % 4) / 4) % 0.6

            def at(u, v_):
                return P[0] * (1 - u) * (1 - v_) + P[1] * u * (1 - v_) + P[2] * u * v_ + P[3] * (1 - u) * v_
            base = [at(uu, vv), at(uu + du, vv), at(uu + du, vv + dv), at(uu, vv + dv)]
            hp = max(1.5, f * h_m / max(dist, 6.0))
            top = [(x, y - hp) for x, y in base]
            body = [c * k for c in roof_col]
            for a_, b_ in zip(range(4), (1, 2, 3, 0)):
                poly(surf, _rgb([c * 0.7 for c in body]), [tuple(map(int, q)) for q in (base[a_], base[b_], top[b_], top[a_])])
            poly(surf, _rgb(body), [tuple(map(int, q)) for q in top])

    def _windows(self, surf, i, k, P, dist, wall_col, glow):
        """Window rows on one wall face. P = screen quad [bottom-a, bottom-b, top-b, top-a]."""
        b = self.world.buildings
        kind = int(b.kind[i])
        F = int(self.window_rows[i])
        width_m = 2.0 * (b.half[i, 0] if k in (0, 2) else b.half[i, 1])
        P = np.array(P, float)
        # skip faces that are just slivers on screen
        wpx = np.hypot(*(P[1] - P[0]))
        hpx = np.hypot(*(P[3] - P[0]))
        if wpx < 5 or hpx < 6:
            return
        wh = float(b.wall[i]) + 1.0
        v0 = 1.0 / wh                                            # ground level (the foundation metre is below it)
        if kind == KIND_HOUSE or kind == KIND_HUT:
            if dist > 85 or F > 2:
                pass
            rows = F
            cols = 2 if width_m < 9 else 3
            u_edges = [(0.14 + 0.72 * (c / cols), 0.14 + 0.72 * (c / cols) + 0.72 / cols * 0.5) for c in range(cols)]
            v_rows = [(v0 + (1 - v0) * (r + 0.30) / rows, v0 + (1 - v0) * (r + 0.72) / rows) for r in range(rows)]
            if dist > 85:
                return
        else:
            rows = min(F, 14)
            if kind == KIND_TOWER:
                v_rows = [(v0 + (1 - v0) * (r + 0.12) / rows, v0 + (1 - v0) * (r + 0.78) / rows) for r in range(rows)]
            else:
                v_rows = [(v0 + (1 - v0) * (r + 0.22) / rows, v0 + (1 - v0) * (r + 0.74) / rows) for r in range(rows)]
            if dist < 50 and kind != KIND_TOWER:
                cols = int(np.clip(width_m / 3.2, 2, 8))
                gap = 0.34
                u_edges = [(0.06 + 0.88 * (c / cols), 0.06 + 0.88 * (c / cols) + 0.88 / cols * (1 - gap)) for c in range(cols)]
            else:
                u_edges = [(0.04, 0.96)]
        base = self._win_base[i]
        if kind == KIND_TOWER:
            # curtain-wall glass: a blue/teal tint (varies per tower) mixed with a reflection of the sky
            g = _GLASS[i % len(_GLASS)]
            amb = self.atm.amb_col * 0.9 + self.atm.sun_col * 0.25
            base = np.clip(_fog(g * amb + np.array(self.atm.mid, float) * 0.20, dist, self.atm), 0, 255)
        else:
            base = np.clip(_fog(np.array(base), dist, self.atm), 0, 255)
        lit = np.array((255, 206, 120.0))
        poly = pygame.draw.polygon
        p0, p1, p2, p3 = P[0], P[1], P[2], P[3]
        for r, (va, vb) in enumerate(v_rows):
            for c, (ua, ub) in enumerate(u_edges):
                def pt(u, vv):
                    return (int(p0[0] * (1 - u) * (1 - vv) + p1[0] * u * (1 - vv) + p2[0] * u * vv + p3[0] * (1 - u) * vv),
                            int(p0[1] * (1 - u) * (1 - vv) + p1[1] * u * (1 - vv) + p2[1] * u * vv + p3[1] * (1 - u) * vv))
                col = base
                if glow > 0.05 and ((i * 7 + r * 13 + c * 5 + k) % 5 < 3):
                    col = base * (1 - glow) + lit * glow
                poly(surf, _rgb(col), [pt(ua, va), pt(ub, va), pt(ub, vb), pt(ua, vb)])

    def _draw_hut(self, surf, v: View, i, cam, dist, fw):
        """Round thatched hut: a mud-wall drum plus a cone, drawn as a billboard (cheap and reads well)."""
        b = self.world.buildings
        base = np.array([b.pos[i, 0], b.pos[i, 1], b.pos[i, 2]])
        c = (base - v.pos) @ v.R
        if c[0] < NEAR:
            return
        f = v.f / c[0]
        bx, by = v.cx - f * c[1], v.cy - f * c[2]
        r = float(b.half[i, 0]) * f
        wh = float(b.wall[i]) * f
        rh = float(b.rise[i]) * f
        if r < 0.8:
            return
        wall = _rgb(fw[0])
        roof = _fog(self.world.buildings.roof_rgb[i] * 0.95, dist, self.atm)
        roof = _rgb(roof)
        pygame.draw.polygon(surf, wall, [(bx - r, by), (bx + r, by), (bx + r, by - wh), (bx - r, by - wh)])
        pygame.draw.polygon(surf, roof, [(bx - r * 1.28, by - wh), (bx + r * 1.28, by - wh), (bx, by - wh - rh)])
        if r > 4:
            dark = _rgb([x * 0.8 for x in roof])
            pygame.draw.polygon(surf, dark, [(bx + r * 0.1, by - wh), (bx + r * 1.28, by - wh), (bx, by - wh - rh)])

    # ================================================================================== vehicles
    _VEH = {  # kind -> (half_length, half_width, height, cabin_frac)
        "car": (2.1, 0.9, 1.45), "matatu": (2.6, 1.0, 2.25), "boda": (0.9, 0.28, 1.0), "truck": (3.6, 1.25, 3.0),
        "bus": (5.0, 1.3, 3.1)}

    def update(self, dt: float) -> None:
        self.t += dt
        for veh in self.veh:
            veh.s += veh.speed * dt
            L = self.routes[veh.route][2][-1]
            if veh.s > L:
                veh.s -= L
            elif veh.s < 0:
                veh.s += L

    def vehicle_objects(self, v: View, max_d: float):
        from bisect import bisect_right
        out = []
        T = self.world.terrain
        px, py = float(v.pos[0]), float(v.pos[1])
        for veh in self.veh:
            xs, ys, cum = self.routes[veh.route]
            k = min(max(bisect_right(cum, veh.s) - 1, 0), len(cum) - 2)
            seg = cum[k + 1] - cum[k]
            u = (veh.s - cum[k]) / seg if seg > 1e-6 else 0.0
            dx_, dy_ = xs[k + 1] - xs[k], ys[k + 1] - ys[k]
            x, y = xs[k] + dx_ * u, ys[k] + dy_ * u
            if abs(x - px) > max_d or abs(y - py) > max_d:
                continue
            ang = math.atan2(dy_, dx_) + (math.pi if veh.speed < 0 else 0.0)
            # Uganda drives on the left: offset to the left of the direction of travel
            off = abs(veh.lane)
            x -= math.sin(ang) * off
            y += math.cos(ang) * off
            d = math.hypot(x - px, y - py)
            if d > max_d:
                continue
            out.append((d - 2.0, K_VEHICLE, (veh, x, y, float(T.height(x, y)) + _ROAD_LIFT, ang, d)))
        return out

    def _draw_vehicle(self, surf, v: View, d):
        veh, x, y, z, ang, dist = d
        hl, hw, hh = self._VEH[veh.kind]
        c, s = math.cos(ang), math.sin(ang)
        col = np.array(veh.color, float)
        if veh.kind == "boda":
            parts = [(0.0, hl, hw, 0.35, 0.45, col), (-0.1, 0.22, 0.24, 0.8, 0.7, (200, 170, 60) if int(x * 3) % 3 else (60, 90, 170))]
        elif veh.kind == "car":
            parts = [(0.0, hl, hw, 0.25, 0.7, col), (-0.2, hl * 0.52, hw * 0.92, 0.95, 0.52, col * 0.55 + 20)]
        elif veh.kind == "matatu":
            parts = [(0.0, hl, hw, 0.3, 1.9, col), (0.0, hl * 1.003, hw * 1.003, 1.0, 0.45, (40, 70, 140))]
        elif veh.kind == "bus":
            parts = [(0.0, hl, hw, 0.35, 2.6, col), (0.0, hl * 1.003, hw * 1.003, 1.4, 0.7, (50, 70, 90))]
        else:
            parts = [(hl * 0.55, hl * 0.45, hw, 0.4, 1.6, col * 0.8), (-hl * 0.42, hl * 0.52, hw * 0.98, 0.6, 2.4, (170, 150, 90))]
        atm = self.atm
        pos = v.pos
        amb, sunc, sund = atm.amb_col, atm.sun_col, atm.sun_dir
        # local unit normals of the four sides (+x, +y, -x, -y) rotated into the world frame
        sides = ((c, s), (-s, c), (-c, -s), (s, -c))
        # corner order: (-,-), (+,-), (+,+), (-,+)  ->  side k joins corner k and k+1
        for (ox, phl, phw, z0, ph, pc) in parts:
            lx = np.array([-phl, phl, phl, -phl]) + ox
            ly = np.array([-phw, -phw, phw, phw])
            wx, wy = x + c * lx - s * ly, y + s * lx + c * ly
            C = np.empty((8, 3))
            C[:4, 0], C[:4, 1], C[:4, 2] = wx, wy, z + z0
            C[4:, 0], C[4:, 1], C[4:, 2] = wx, wy, z + z0 + ph
            cam = (C - pos) @ v.R
            if (cam[:, 0] < NEAR).any():
                continue
            sc = _scr(v, cam).astype(int).tolist()
            pc = np.array(pc, float)
            cz = z + z0 + ph / 2
            faces = []
            # which side normals face the camera: the side's normal is (sy, -sx) of its edge direction
            for k, (nx, ny) in enumerate(((-s, c), (c, s), (s, -c), (-c, -s))):
                # normals: side 0 is the -y... derive from the edge instead to stay correct
                a, b2 = k, (k + 1) % 4
                ex, ey = wx[b2] - wx[a], wy[b2] - wy[a]
                n_x, n_y = ey, -ex                          # outward for CCW corner order (-,-),(+,-),(+,+),(-,+)
                nl = math.hypot(n_x, n_y) + 1e-9
                n_x, n_y = n_x / nl, n_y / nl
                mx, my = (wx[a] + wx[b2]) / 2, (wy[a] + wy[b2]) / 2
                if n_x * (pos[0] - mx) + n_y * (pos[1] - my) > 0:
                    lam = amb + sunc * max(0.0, n_x * sund[0] + n_y * sund[1])
                    faces.append(([a, b2, b2 + 4, a + 4], pc * lam))
            if pos[2] > z + z0 + ph:
                faces.append(([4, 5, 6, 7], pc * (amb + sunc * sund[2]) * 1.05))
            for ids, col_ in faces:
                col_ = np.clip(_fog(col_, dist, atm), 0, 255)
                pygame.draw.polygon(surf, (int(col_[0]), int(col_[1]), int(col_[2])), [tuple(sc[q]) for q in ids])
        if atm.glow > 0.5 and veh.kind != "boda":                       # headlights at dusk
            sgn = 1 if veh.speed > 0 else -1
            cam = (np.array([x + c * hl * sgn, y + s * hl * sgn, z + 0.7]) - pos) @ v.R
            if cam[0] > NEAR:
                scr = _scr(v, cam)
                pygame.draw.circle(surf, (255, 240, 190), (int(scr[0]), int(scr[1])), max(1, int(v.f * 0.35 / cam[0])))

    # ================================================================================== birds
    def bird_objects(self, v: View):
        out = []
        for k, (cx, cy, z, rad, ph, w) in enumerate(self.birds):
            a = ph + self.t * w
            p = np.array([cx + math.cos(a) * rad, cy + math.sin(a) * rad, z + 3 * math.sin(self.t * 0.7 + k)])
            d = float(np.linalg.norm(p - v.pos))
            if d < 260:
                out.append((d, K_BIRD, (p, a + (math.pi / 2 if w > 0 else -math.pi / 2), d, k)))
        return out

    def _draw_bird(self, surf, v: View, dd):
        p, heading, dist, k = dd
        flap = math.sin(self.t * 7.5 + k * 1.7)
        span = 1.7 if k % 3 else 1.0              # marabou storks are big, kites smaller
        dirv = np.array([-math.sin(heading), math.cos(heading), 0.0])
        wing = flap * 0.55
        a = p + dirv * span + np.array([0, 0, wing])
        b = p - dirv * span + np.array([0, 0, wing])
        pts = np.array([a, p, b])
        cam = (pts - v.pos) @ v.R
        if (cam[:, 0] < NEAR).any():
            return
        sc = _scr(v, cam)
        col = _rgb(_fog(np.array((28, 28, 32.0)), dist, self.atm))
        pygame.draw.lines(surf, col, False, [tuple(map(int, q)) for q in sc], max(1, int(v.f * 0.18 / cam[1, 0])))

    # ================================================================================== meshes
    def mesh_objects(self, v: View, max_d: float):
        out = []
        for mi, m in enumerate(self.world.meshes):
            d = float(np.linalg.norm(m.centre - v.pos))
            if d - m.radius < max_d:
                out.append((d - m.radius * 0.35, K_MESH, (mi, d)))
        return out

    def _mesh_cache(self, mi, m):
        """Per-mesh static arrays: face centroids, padded vertex index table, lit colours (rebuilt per atmosphere)."""
        key = (mi, id(self.atm))
        ent = self._mc.get(mi)
        if ent is None:
            fc = np.array([m.verts[list(f)].mean(axis=0) for f in m.faces])
            ent = {"fc": fc, "key": None}
            self._mc[mi] = ent
        if ent["key"] != id(self.atm):
            lam = _lam(m.normals, self.atm)
            cols = m.colors * lam
            if m.emissive is not None:
                cols = np.where(m.emissive[:, None], m.colors, cols)
            ent["cols"] = cols
            ent["key"] = id(self.atm)
        return ent

    def _draw_mesh(self, surf, v: View, dd):
        mi, dist = dd
        m = self.world.meshes[mi]
        if not hasattr(self, "_mc"):
            self._mc = {}
        ent = self._mesh_cache(mi, m)
        fc = ent["fc"]
        vis = np.einsum("fc,fc->f", m.normals, v.pos - fc) > 0
        idx = np.nonzero(vis)[0]
        if not len(idx):
            return
        depth = np.linalg.norm(fc[idx] - v.pos, axis=1)
        order = np.argsort(-depth)
        idx, depth = idx[order], depth[order]
        cols = np.clip(_fog(ent["cols"][idx], depth, self.atm), 0, 255).astype(int).tolist()
        cam = (m.verts - v.pos) @ v.R
        scr = _scr(v, cam).astype(int)
        behind = cam[:, 0] < NEAR
        W, H = v.width, v.height
        poly = pygame.draw.polygon
        for n_, fi in enumerate(idx):
            f = m.faces[fi]
            if behind[list(f)].any():
                q = _clip_near(cam[list(f)])
                if len(q) < 3:
                    continue
                pts = [(int(x), int(y)) for x, y in _scr(v, q)]
            else:
                pts = [tuple(scr[k]) for k in f]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if max(xs) < 0 or min(xs) > W or max(ys) < 0 or min(ys) > H:
                continue
            poly(surf, cols[n_], pts)

    # ================================================================================== dispatch
    def draw(self, surf, v: View, kind: int, payload) -> None:
        if kind == K_BUILDING:
            self._draw_building(surf, v, payload)
        elif kind == K_VEHICLE:
            self._draw_vehicle(surf, v, payload)
        elif kind == K_BIRD:
            self._draw_bird(surf, v, payload)
        elif kind == K_MESH:
            self._draw_mesh(surf, v, payload)