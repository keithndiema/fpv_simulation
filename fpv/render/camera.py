"""Cameras (Strategy pattern) + vectorised pinhole projection with near-plane clipping."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

NEAR = 0.15


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class View:
    """R columns = camera forward / left / up expressed in world coordinates."""
    pos: np.ndarray
    R: np.ndarray
    width: int
    height: int
    fov_deg: float

    def __post_init__(self):
        self.f = (self.width / 2) / math.tan(math.radians(self.fov_deg) / 2)
        self.cx, self.cy = self.width / 2, self.height / 2

    def to_cam(self, P: np.ndarray) -> np.ndarray:
        return (P - self.pos) @ self.R

    def _screen(self, c: np.ndarray) -> np.ndarray:
        x = np.maximum(c[:, 0], NEAR)
        return np.stack([self.cx - self.f * c[:, 1] / x, self.cy - self.f * c[:, 2] / x], axis=1)

    def project_points(self, P: np.ndarray):
        c = self.to_cam(P)
        return self._screen(c), c[:, 0]

    def segments(self, A: np.ndarray, B: np.ndarray):
        """Clip world segments to the near plane and project. Returns (S0, S1, depth, keep_index)."""
        ca, cb = self.to_cam(A), self.to_cam(B)
        xa, xb = ca[:, 0], cb[:, 0]
        keep = (xa > NEAR) | (xb > NEAR)
        ca, cb, xa, xb = ca[keep], cb[keep], xa[keep], xb[keep]
        idx = np.nonzero(keep)[0]
        ta = np.where(xa < NEAR, (NEAR - xa) / (xb - xa + 1e-12), 0.0)[:, None]
        tb = np.where(xb < NEAR, (NEAR - xb) / (xa - xb + 1e-12), 0.0)[:, None]
        ca2, cb2 = ca + ta * (cb - ca), cb + tb * (ca - cb)
        s0, s1 = self._screen(ca2), self._screen(cb2)
        w, h = self.width, self.height
        off = (((s0[:, 0] < 0) & (s1[:, 0] < 0)) | ((s0[:, 0] > w) & (s1[:, 0] > w)) |
               ((s0[:, 1] < 0) & (s1[:, 1] < 0)) | ((s0[:, 1] > h) & (s1[:, 1] > h)))
        ok = ~off
        depth = 0.5 * (ca2[:, 0] + cb2[:, 0])
        return (np.clip(s0[ok], -20000, 20000), np.clip(s1[ok], -20000, 20000), depth[ok], idx[ok])


class Camera(ABC):
    name = "camera"

    @abstractmethod
    def view(self, state, width: int, height: int, dt: float) -> View: ...


class FpvCamera(Camera):
    """Body-mounted camera with adjustable uptilt - what a real FPV pilot sees."""
    name = "FPV"

    def __init__(self, tilt_deg: float, fov_deg: float, shake: float = 1.0):
        self.tilt, self.fov, self.shake = tilt_deg, fov_deg, shake
        self._rng = np.random.default_rng(5)

    def view(self, st, width, height, dt):
        t = math.radians(self.tilt)
        if self.shake > 0 and st.armed and not st.crashed:
            # motor vibration: tiny high-frequency camera wobble that grows with power (and a touch with speed)
            amp = math.radians(0.22) * self.shake * (0.3 + float(np.mean(st.motors))) * (1 + min(st.speed, 25) / 30)
            jr, jp = self._rng.normal(0, amp, 2)
        else:
            jr = jp = 0.0
        c, s = math.cos(t + jp), math.sin(t + jp)
        tilt = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])      # forward axis rotates UP by `tilt`
        cr, sr = math.cos(jr), math.sin(jr)
        R = quat_to_matrix(st.quat) @ tilt @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return View(np.array(st.pos) + quat_to_matrix(st.quat) @ np.array([0.06, 0, 0.03]), R, width, height,
                    self.fov)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f) + 1e-9
    l = np.cross([0, 0, 1.0], f)
    l /= np.linalg.norm(l) + 1e-9
    u = np.cross(f, l)
    return np.stack([f, l, u], axis=1)


class ChaseCamera(Camera):
    """Smoothed third-person camera following the drone's heading."""
    name = "Chase"

    def __init__(self, fov_deg: float = 80.0, distance: float = 3.2, height: float = 1.1):
        self.fov, self.dist, self.h = fov_deg, distance, height
        self.pos: np.ndarray | None = None

    def view(self, st, width, height, dt):
        p = np.array(st.pos)
        yaw = st.euler[2]
        vel = np.array(st.vel[:2])
        if np.linalg.norm(vel) > 2.0:                  # follow velocity when moving fast
            yaw = math.atan2(vel[1], vel[0])
        want = p - np.array([math.cos(yaw), math.sin(yaw), 0]) * self.dist + np.array([0, 0, self.h])
        want[2] = max(want[2], 0.4)
        self.pos = want if self.pos is None else self.pos + (want - self.pos) * min(1.0, 7.0 * dt)
        return View(self.pos.copy(), look_at(self.pos, p + np.array([0, 0, 0.25])), width, height, self.fov)


class OrbitCamera(Camera):
    """Slow cinematic orbit used behind the menu."""
    name = "Orbit"

    def __init__(self, centre, radius=60.0, height=22.0):
        self.centre, self.radius, self.height, self.t = np.array(centre, float), radius, height, 0.0

    def view(self, st, width, height, dt):
        self.t += dt * 0.08
        eye = self.centre + np.array([math.cos(self.t) * self.radius, math.sin(self.t) * self.radius, self.height])
        return View(eye, look_at(eye, self.centre + np.array([0, 0, 3.0])), width, height, 85.0)
