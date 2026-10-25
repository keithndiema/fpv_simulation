"""Missions: objectives, scoring, star ratings and the state machine that runs them.

State flow:  READY (briefing) -> ACTIVE (timer starts at take-off) -> COMPLETE | FAILED
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace

import numpy as np

from ..events import EventBus
from .course import RaceTracker, build_course
from .targets import TargetField


class Phase(enum.Enum):
    READY = "ready"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class Mission:
    id: str
    title: str
    brief: str                      # one-line objective shown on the card
    detail: str                     # how-to shown on the briefing
    kind: str                       # "targets" | "race" | "attack" | "free"
    course: int                     # 0 = race circuit layout, 1 = open field
    targets: int = 0
    laps: int = 0
    time_limit: float | None = None
    # star thresholds: for time-based missions lower is better, for "attack" (score) higher is better
    gold: float = 0.0
    silver: float = 0.0
    bronze: float = 0.0
    seed: int = 1
    radius: float = 60.0


MISSIONS: list[Mission] = [
    Mission("training", "First Flight", "Pop 5 targets close to the pad.",
            "Take off, then fly through the red-and-white bullseyes. Follow the yellow arrow on screen to find the "
            "next one. No time limit - just learn the controls.", "targets", 1, targets=5,
            gold=70, silver=110, bronze=170, seed=21, radius=45.0),
    Mission("balloons", "Bullseye Run", "Pop 10 targets in 150 seconds.",
            "Targets are spread across the area around the pad. Chain pops within 8 seconds to build a x5 combo.", "targets", 1,
            targets=10, time_limit=150.0, gold=95, silver=125, bronze=150, seed=8, radius=95.0),
    Mission("race", "Circuit Race", "Fly 2 laps through all 10 gates.",
            "Pass through each glowing gate in order. Missing a gate means turning around to fly it again.",
            "race", 0, laps=2, gold=100, silver=130, bronze=170, seed=3),
    Mission("attack", "Time Attack", "Score as much as you can in 75 seconds.",
            "Targets respawn the moment you pop them. Keep the combo alive - each chained pop is worth more.",
            "attack", 1, targets=6, time_limit=75.0, gold=1800, silver=1100, bronze=600, seed=5, radius=85.0),
    Mission("free", "Free Flight", "Explore the town - no rules.",
            "Practice, tune your PIDs in the menu, or just sightsee.", "free", 1),
]


def scaled(m: Mission, course) -> Mission:
    """Race star times depend on the circuit: the valley circuit is ~350 m a lap, a town's block ring is longer
    or shorter.  Thresholds are derived from the lap length at a pace a good pilot can hold (~9.5 m/s average)."""
    if m.kind != "race" or course is None or getattr(course, "world", None) is None or len(course.gates) < 3:
        return m
    pts = [g.pos[:2] for g in course.gates]
    lap = sum(float(np.linalg.norm(pts[(i + 1) % len(pts)] - pts[i])) for i in range(len(pts)))
    gold = round(m.laps * lap / 9.5 / 5.0) * 5.0
    return replace(m, gold=gold, silver=round(gold * 1.3 / 5.0) * 5.0, bronze=round(gold * 1.8 / 5.0) * 5.0)


def mission_by_id(mid: str) -> Mission:
    return next(m for m in MISSIONS if m.id == mid)


def stars_for(m: Mission, value: float) -> int:
    """3/2/1/0 stars for a result value (seconds for timed missions, points for attack)."""
    if m.kind == "attack":
        return 3 if value >= m.gold else 2 if value >= m.silver else 1 if value >= m.bronze else 0
    return 3 if value <= m.gold else 2 if value <= m.silver else 1 if value <= m.bronze else 0


class MissionRunner:
    TAKEOFF_ALT = 1.2          # metres above the ground that count as "airborne" (starts the clock)

    def __init__(self, mission: Mission, bus: EventBus, terrain_course=None):
        self.bus = bus
        self.course = terrain_course or build_course(mission.course)
        self.m = scaled(mission, self.course)
        self.tracker = RaceTracker(self.course, bus)
        self.field = TargetField(bus)
        self.phase = Phase.READY
        self.elapsed = 0.0
        self.result: dict | None = None
        self._takeoff_t: float | None = None
        self.fail_reason = ""
        self._build()

    # ---- setup -----------------------------------------------------------------------------
    def _build(self) -> None:
        m, T = self.m, self.course.terrain
        self.field.clear()
        if m.kind in ("targets", "attack"):
            self.field.scatter(T, m.targets, T.mission_centre(m.radius), m.radius, m.seed,
                               min_sep=18.0, avoid=[T.pad], blocked=self.course.is_blocked)
        self.tracker.reset()
        self.phase, self.elapsed, self.result, self.fail_reason = Phase.READY, 0.0, None, ""
        self._takeoff_t = None

    def restart(self) -> None:
        self._build()

    # ---- derived values ----------------------------------------------------------------------
    @property
    def score(self) -> int:
        return self.field.score

    @property
    def time_left(self) -> float | None:
        return None if self.m.time_limit is None else max(0.0, self.m.time_limit - self.elapsed)

    def objective_text(self) -> str:
        m = self.m
        if m.kind == "targets":
            return f"TARGETS  {self.field.hits}/{m.targets}"
        if m.kind == "attack":
            return f"SCORE  {self.field.score}"
        if m.kind == "race":
            gates = len(self.course.gates)
            done = self.tracker.gates_passed
            return f"LAP {min(self.tracker.lap + 1, m.laps)}/{m.laps}   GATE {done % gates + 1 if done < gates * m.laps else gates}/{gates}"
        return "FREE FLIGHT"

    def progress(self) -> float:
        m = self.m
        if m.kind == "targets":
            return self.field.hits / max(1, m.targets)
        if m.kind == "race":
            return self.tracker.gates_passed / max(1, len(self.course.gates) * m.laps)
        if m.kind == "attack" and m.time_limit:
            return min(1.0, self.elapsed / m.time_limit)
        return 0.0

    # ---- per frame -------------------------------------------------------------------------------
    def update(self, dt: float, st, sim_time: float) -> None:
        """Advance mission logic from the latest sim state `st`."""
        pos = np.array(st.pos)
        if self.phase in (Phase.COMPLETE, Phase.FAILED):
            return
        ground = float(self.course.ground(pos[0], pos[1]))
        if self.phase is Phase.READY:
            if st.armed and pos[2] - ground > self.TAKEOFF_ALT:
                self.phase, self._takeoff_t = Phase.ACTIVE, sim_time
                self.field.reset_tracking()
                self.bus.publish("mission_start", mission=self.m)
            else:
                return
        self.elapsed = sim_time - (self._takeoff_t or sim_time)

        if st.crashed and self.m.kind != "free":
            return self._fail("CRASHED")

        if self.m.kind in ("targets", "attack"):
            popped = self.field.update(pos, sim_time)
            if self.m.kind == "attack":
                for tg in popped:                   # respawn each popped target somewhere new
                    self._respawn(tg)
        if self.m.kind == "race" and not st.crashed:
            self.tracker.update(pos, sim_time)

        if self.m.kind == "targets" and self.field.remaining() == 0:
            self._finish(self.elapsed)
        elif self.m.kind == "race" and self.tracker.lap >= self.m.laps:
            self._finish(self.elapsed)
        elif self.m.time_limit is not None and self.elapsed >= self.m.time_limit:
            if self.m.kind == "attack":
                self._finish(float(self.field.score))
            else:
                self._fail("TIME UP")

    def _respawn(self, old) -> None:
        import random
        rng = random.Random(self.m.seed * 977 + self.field.hits)
        T = self.course.terrain
        for _ in range(200):
            a, r = rng.uniform(0, 2 * math.pi), self.m.radius * math.sqrt(rng.uniform(0.15, 1.0))
            mc = T.mission_centre(self.m.radius)
            x, y = mc[0] + r * math.cos(a), mc[1] + r * math.sin(a)
            if self.course.is_blocked(x, y) or any(t.alive and math.hypot(x - t.pos[0], y - t.pos[1]) < 18 for t in self.field.targets):
                continue
            self.field.place(x, y, rng.uniform(3.0, 12.0), T)
            return

    def _finish(self, value: float) -> None:
        stars = stars_for(self.m, value)
        self.phase = Phase.COMPLETE
        self.result = {"value": value, "stars": stars, "score": self.field.score, "time": self.elapsed}
        self.bus.publish("mission_complete", mission=self.m, result=self.result)

    def _fail(self, reason: str) -> None:
        self.phase, self.fail_reason = Phase.FAILED, reason
        self.result = {"value": None, "stars": 0, "score": self.field.score, "time": self.elapsed}
        self.bus.publish("mission_failed", mission=self.m, reason=reason)
