"""Velocity-guided pure-pursuit autopilot (ANGLE mode) for demo flights and as a regression pilot."""
from __future__ import annotations

import math

import numpy as np

from ..input.sticks import Sticks
from .course import Course

HOVER = 0.31


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class Autopilot:
    def __init__(self, cruise: float = 9.0):
        self.cruise = cruise  # target ground speed [m/s]
        self._align = 0.0     # lateral offset from the gate axis while approaching (slows us down when large)
        self._iz = 0.0        # altitude integral term: cancels the hover-throttle error caused by battery sag
        self._t = None

    def compute(self, st, course: Course, next_gate: int, field=None) -> Sticks:
        p = np.array(st.pos)
        tgt = field.nearest(p) if field is not None else None
        if tgt is not None and not course.gates:
            target = tgt.pos.copy()                   # mission targets: fly straight at the nearest bullseye
        elif course.gates:
            g = course.gates[next_gate]
            target = g.pos.copy()
            along = float(np.dot(p - g.pos, g.normal))
            lateral = float(np.dot(p - g.pos, g.lateral))
            if next_gate == 0 and -20 < along < 6 and abs(lateral) > 5.0:
                # Far off the gate axis after take-off (the pad sits inside the park block, beside the street): fly to a
                # staging point 20 m before the gate on its axis instead of cutting across at a steep angle -
                # city gates are lined up with the street, and a sideways entry hits the frame.
                target = g.pos - g.normal * 20.0
                self._align = abs(lateral)
            elif -35 < along < 1.5:                   # converge onto the gate axis with a carrot that slides ahead of us
                carrot = min(4.0, along + 8.0)
                target = g.pos + g.normal * carrot + np.array([0, 0, 0.0])
                self._align = abs(lateral)
            else:
                self._align = 0.0
        else:
            target = np.array([60.0, 0.0, 4.0 + float(course.ground(60.0, 0.0))])
        d = target - p
        # terrain following: never aim below a safe clearance over the ground a few seconds ahead
        h = np.hypot(d[0], d[1]) + 1e-6
        for ahead in (6.0, 14.0, 24.0):
            a = min(ahead, h)
            gx, gy = p[0] + d[0] / h * a, p[1] + d[1] / h * a
            target[2] = max(target[2], float(course.clearance(gx, gy)) + 3.0)
        d = target - p
        err = _wrap(math.atan2(d[1], d[0]) - st.euler[2])
        yaw = _clamp(-1.5 * err, -0.7, 0.7)

        c, s = math.cos(st.euler[2]), math.sin(st.euler[2])
        vf = st.vel[0] * c + st.vel[1] * s            # forward / left velocity in the yaw frame
        vl = -st.vel[0] * s + st.vel[1] * c
        vref = self.cruise * max(0.2, math.cos(min(abs(err), 1.4))) ** 2 * max(0.45, min(1.0, 1.25 - self._align / 6.0))
        pitch = _clamp(0.07 * (vref - vf), -0.3, 0.55)
        roll = _clamp(0.10 * vl, -0.5, 0.5)

        # Why an integral term?  The proportional term alone settles ~1 m off target once the battery sags, and a
        # city gate leaves only ~1 m of margin above the drone.  Integrating the error removes that offset.
        dt = 0.0 if self._t is None else max(0.0, min(0.05, st.time - self._t))
        if self._t is not None and st.time < self._t:
            self._iz = 0.0                                  # the simulation was reset
        self._t = st.time
        if st.on_ground:
            self._iz = 0.0
        self._iz = _clamp(self._iz + 0.025 * _clamp(target[2] - p[2], -1.5, 1.5) * dt, -0.06, 0.06)
        tilt = math.hypot(st.euler[0], st.euler[1])
        thr = HOVER / max(0.6, math.cos(min(tilt, 1.0))) + 0.06 * (target[2] - p[2]) - 0.09 * st.vel[2] + self._iz
        return Sticks(roll, pitch, yaw, _clamp(thr, 0.12, 0.7))
