"""Race course: gates, collision geometry, and lap/gate tracking (publishes events on the bus)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from ..events import EventBus
from .terrain import PAD_CENTER, Terrain

GATE_SIZE = 3.6      # opening [m] (wider than a real 2.8 m gate: building shadows and rooftop wind make city flying harder)
GATE_FRAME = 0.18


@dataclass
class Gate:
    pos: np.ndarray            # centre (x, y, z)
    yaw: float                 # travel direction [rad]

    @property
    def normal(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])

    @property
    def lateral(self) -> np.ndarray:
        return np.array([-math.sin(self.yaw), math.cos(self.yaw), 0.0])

    def corners(self, size: float = GATE_SIZE) -> np.ndarray:
        h = size / 2
        up = np.array([0, 0, 1.0])
        l = self.lateral
        return np.array([self.pos - l * h - up * h, self.pos + l * h - up * h,
                         self.pos + l * h + up * h, self.pos - l * h + up * h])


@dataclass
class Course:
    name: str
    start_pose: tuple
    gates: list[Gate] = field(default_factory=list)
    boxes: list[tuple] = field(default_factory=list)   # buildings / pylons: (cx, cy, cz, hx, hy, hz, yaw) - rendered AND collidable
    terrain: Terrain | None = None
    world: object = None                               # world.model.World when flying over a city

    def collision_boxes(self) -> list[tuple]:
        out = list(self.boxes)
        t = GATE_FRAME
        for g in self.gates:
            top = g.pos[2] + GATE_SIZE / 2 + t
            for sgn in (-1, 1):  # posts down to the terrain
                c = g.pos + g.lateral * sgn * (GATE_SIZE / 2 + t / 2)
                base = float(self.terrain.ground(c[0], c[1])) if self.terrain else 0.0
                out.append((c[0], c[1], (top + base) / 2, 0.06, t / 2, (top - base) / 2, g.yaw))
            out.append((g.pos[0], g.pos[1], g.pos[2] + GATE_SIZE / 2 + t / 2, 0.06,
                        GATE_SIZE / 2 + t, t / 2, g.yaw))
        if self.world is not None:
            # buildings + landmark footprints near the launch pad (the C++ core tests every box each step, so
            # far-away ones are left out; a 450 m radius covers the whole flyable town)
            pad = self.terrain.pad
            out += self.world.buildings.collision_boxes(pad, 450.0)
            out += list(self.world.extra_colliders)
            out += self.terrain.tree_colliders(pad, 300.0)
        elif self.terrain is not None:
            out += self.terrain.tree_colliders(self.terrain.play_center, 190.0)
        return out

    def ground(self, x, y):
        return self.terrain.ground(x, y) if self.terrain is not None else 0.0

    def clearance(self, x, y):
        """Lowest safe altitude over (x, y): terrain, or the roof of whatever building stands there.

        Trees are deliberately not included: the race corridor (take-off run + every leg) is cleared of trees in
        _build_city_course, and treating crowns as obstacles made the autopilot climb over the gates.
        """
        return self.world.clearance(x, y) if self.world is not None else self.ground(x, y)

    def is_blocked(self, x: float, y: float, margin: float = 5.0) -> bool:
        """True if (x, y) is over water or within `margin` metres of a building (no targets / spawns there)."""
        if self.terrain is not None and self.terrain.height(x, y) < 1.0:
            return True
        return self.world is not None and self.world.blocked(x, y, margin)


_RACE_WAYPOINTS = [(22, 0, 2.6), (46, 8, 3.0), (68, 26, 3.8), (76, 54, 4.6), (62, 80, 3.4), (36, 90, 5.0),
                   (10, 78, 2.8), (-6, 54, 2.4), (-4, 28, 3.4), (6, 8, 2.6)]   # (x, y, height above ground)

PLAY_CENTER = (40.0, 45.0)
_default_terrain: Terrain | None = None


def default_terrain() -> Terrain:
    """One shared world for every course (generation takes ~50 ms)."""
    global _default_terrain
    if _default_terrain is None:
        _default_terrain = Terrain()
    return _default_terrain


def _city_start(T: Terrain) -> tuple:
    """Launch pose: on the pad, nose pointing at the town centre."""
    px, py = T.pad
    cx, cy = T.world.centre
    yaw = math.atan2(cy - py, cx - px)
    return (px, py, float(T.height(px, py)) + 0.04, yaw)


def _city_circuit(T: Terrain) -> list[Gate]:
    """Ten gates on the streets that ring the park the pad sits in (falls back to a circle round the pad)."""
    w = T.world
    pts: list[tuple] = []
    lat = (w.lattice or {}).get("lat")
    pb = (w.lattice or {}).get("park_block")
    if lat is not None and pb is not None:
        bi, bj = pb
        us, vs = lat.us, lat.vs
        corners = [(us[bi], vs[bj]), (us[bi + 1], vs[bj]), (us[bi + 1], vs[bj + 1]), (us[bi], vs[bj + 1])]
        # Walk the block perimeter and drop gates along each side, never on a corner: a gate sitting on a 90 degree
        # corner has to be crossed at an awkward angle, which is unfair for a pilot and fatal for the autopilot.
        # Long sides get three gates, short sides two -> 10 gates like the original circuit.
        la = math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
        lb = math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
        long_first = la >= lb
        ring = []
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            lerp = lambda t, a=a, b=b: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)   # noqa: E731
            is_long = (k % 2 == 0) == long_first
            for t in ((0.2, 0.5, 0.8) if is_long else (0.33, 0.67)):
                ring.append(lerp(t))
        for u, v in ring:
            x, y = lat.to_world(np.array(u), np.array(v))
            pts.append((float(x), float(y)))
    else:
        px, py = T.pad
        for k in range(10):
            a = 2 * math.pi * k / 10
            pts.append((px + 58 * math.cos(a), py + 58 * math.sin(a)))
    rng = random.Random(5)
    P = np.array([(x, y, float(T.ground(x, y)) + rng.uniform(4.2, 5.8)) for x, y in pts], dtype=float)
    n = len(P)

    def directed(order):
        Q = P[order]
        return [Gate(Q[i].copy(), math.atan2(*(Q[(i + 1) % n] - Q[(i - 1) % n])[1::-1])) for i in range(n)]

    # Which way round, and which gate first?  The drone starts inside the park block, so the first gate must lie
    # ahead of it: the pad sits a little *before* the gate plane (negative "along") and close to the gate axis
    # (small lateral offset).  Otherwise the pilot has to loop back round to reach gate 1.
    pad = np.array(T.pad)
    best = None
    for order in (list(range(n)), list(range(n))[::-1]):
        cand = directed(order)
        for k, g in enumerate(cand):
            rel = pad - g.pos[:2]
            along = float(np.dot(rel, g.normal[:2]))
            lat = float(np.dot(rel, g.lateral[:2]))
            cost = abs(along + 22.0) * 0.6 + abs(lat) + (200.0 if along > -8.0 else 0.0)
            if best is None or cost < best[0]:
                best = (cost, cand, k)
    _, cand, k = best
    return cand[k:] + cand[:k]


def _build_city_course(kind: int, T: Terrain) -> Course:
    c = Course(T.name, _city_start(T), terrain=T, world=T.world)
    if kind == 0:
        c.gates = _city_circuit(T)
        corridor = []
        n = len(c.gates)
        for i in range(n):
            a_, b_ = c.gates[i].pos, c.gates[(i + 1) % n].pos
            for t_ in np.linspace(0, 1, max(2, int(np.linalg.norm(b_ - a_) / 5))):
                corridor.append(tuple((a_ + (b_ - a_) * t_)[:2]))
        # the take-off run too: pad -> staging point 20 m before gate 1 (on its axis) -> gate 1
        g0 = c.gates[0]
        way = [np.array(c.start_pose[:2]), g0.pos[:2] - g0.normal[:2] * 20.0, g0.pos[:2]]
        for a_, b_ in zip(way[:-1], way[1:]):
            for t_ in np.linspace(0, 1, max(2, int(np.linalg.norm(b_ - a_) / 5))):
                corridor.append(tuple(a_ + (b_ - a_) * t_))
        T.clear_trees(corridor, 10.0)
    else:
        T.clear_trees([], 0.0)                         # restore the full forest after a race course trimmed it
    return c


def build_course(kind: int, terrain: Terrain | None = None) -> Course:
    T = terrain or default_terrain()
    if getattr(T, "world", None) is not None:
        return _build_city_course(kind, T)
    start = (PAD_CENTER[0] - 4.0, PAD_CENTER[1], float(T.height(PAD_CENTER[0] - 4.0, PAD_CENTER[1])) + 0.04, 0.0)
    if kind == 1:
        rng = random.Random(7)
        c = Course("Open field", start, terrain=T)
        c.boxes = _scatter(rng, T, [], 22, area=110)
        T.clear_trees([(b[0], b[1]) for b in c.boxes], 6.0)
        return c
    pts = np.array([(x, y, float(T.ground(x, y)) + h) for x, y, h in _RACE_WAYPOINTS], dtype=float)
    n = len(pts)
    gates = []
    for i in range(n):
        d = pts[(i + 1) % n] - pts[(i - 1) % n]
        gates.append(Gate(pts[i].copy(), math.atan2(d[1], d[0])))
    c = Course("Race circuit", start, gates, terrain=T)
    c.boxes = _scatter(random.Random(11), T, [g.pos for g in gates], 24, area=130)
    # keep a clear flight corridor along every leg between gates (a tree in the racing line is just unfair)
    corridor = []
    for i in range(n):
        a_, b_ = pts[i], pts[(i + 1) % n]
        for t_ in np.linspace(0, 1, max(2, int(np.linalg.norm(b_ - a_) / 6))):
            corridor.append(tuple((a_ + (b_ - a_) * t_)[:2]))
    T.clear_trees(corridor + [(b[0], b[1]) for b in c.boxes], 8.0)
    return c


def _scatter(rng: random.Random, T: Terrain, keep_clear, count: int, area: float) -> list[tuple]:
    """Towers and buildings standing on the terrain (never in water, never on the launch pad)."""
    boxes: list[tuple] = []
    tries = 0
    while len(boxes) < count and tries < 5000:
        tries += 1
        x, y = rng.uniform(-area * 0.6, area), rng.uniform(-area * 0.5, area)
        if math.hypot(x - PAD_CENTER[0], y - PAD_CENTER[1]) < 22 or T.height(x, y) < 1.0 or \
                any(math.hypot(x - g[0], y - g[1]) < 9 for g in keep_clear):
            continue
        h = rng.choice([3.0, 4.5, 6.0, 9.0, 14.0])
        w = rng.choice([0.4, 1.2, 2.5, 4.0]) if h < 8 else rng.choice([3.0, 4.5, 6.0])
        hx = w / 2
        hy = w / 2 * rng.choice([1, 1, 1.8])
        # stand on the lowest ground under the footprint so nothing floats
        base = min(float(T.ground(x + dx * hx, y + dy * hy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        boxes.append((x, y, base + h / 2 - 0.3, hx, hy, h / 2 + 0.3, rng.uniform(0, math.pi)))
    return boxes


class RaceTracker:
    """Detects gate crossings (plane crossing + inside-opening test) and tracks lap timing."""

    def __init__(self, course: Course, bus: EventBus):
        self.course, self.bus = course, bus
        self.reset()

    def reset(self) -> None:
        self.next_gate = 0
        self.lap = 0
        self.lap_start: float | None = None
        self.best: float | None = getattr(self, "best", None)
        self.last_lap: float | None = None
        self.gates_passed = 0
        self._prev: np.ndarray | None = None

    def update(self, pos: np.ndarray, t: float) -> None:
        if self._prev is None or not self.course.gates:
            self._prev = pos.copy()
            return
        g = self.course.gates[self.next_gate]
        n = g.normal
        d0, d1 = float(np.dot(self._prev - g.pos, n)), float(np.dot(pos - g.pos, n))
        if d0 < 0 <= d1:
            f = d0 / (d0 - d1) if d0 != d1 else 0.0
            hit = self._prev + (pos - self._prev) * f - g.pos
            if abs(float(np.dot(hit, g.lateral))) < GATE_SIZE / 2 and abs(hit[2]) < GATE_SIZE / 2:
                self._passed(t)
            else:
                self.bus.publish("gate_missed", index=self.next_gate)
        self._prev = pos.copy()

    def _passed(self, t: float) -> None:
        idx = self.next_gate
        if idx == 0 and self.lap_start is None:
            self.lap_start = t
        elif idx == 0 and self.lap_start is not None and self.gates_passed > 0:
            lap = t - self.lap_start
            self.last_lap = lap
            self.best = lap if self.best is None else min(self.best, lap)
            self.lap += 1
            self.lap_start = t
            self.bus.publish("lap", lap=self.lap, time=lap, best=self.best)
        self.gates_passed += 1
        self.next_gate = (idx + 1) % len(self.course.gates)
        self.bus.publish("gate", index=idx, next=self.next_gate)
