"""Regression suite: python run_sim.py --selftest   (no display required)."""
from __future__ import annotations

import math
import sys
import time

from fpv.events import EventBus
from fpv.game.autopilot import Autopilot
from fpv.game.course import RaceTracker, build_course
from fpv.sim.native import Simulator

D2R = math.pi / 180


def _fly(sim: Simulator, seconds: float, sticks, dt: float = 0.002, hook=None):
    for i in range(int(seconds / dt)):
        sim.set_sticks(*(sticks(i * dt) if callable(sticks) else sticks))
        sim.advance(dt)
        if hook:
            hook(i * dt, sim.state())
    return sim.state()


def _fresh(mode: int, z: float = 80.0) -> Simulator:
    s = Simulator()
    s.reset((0, 0, z, 0))
    s.set_mode(mode)
    s.set_armed(True)
    return s


def test_hover():
    s = _fresh(Simulator.ANGLE)
    st = _fly(s, 4.0, (0, 0, 0, 0.31))                  # ~3% above true hover -> gentle climb, but level
    tilt = max(abs(st.euler[0]), abs(st.euler[1])) / D2R
    horiz = math.hypot(st.pos[0], st.pos[1])
    ok = abs(st.pos[2] - 80) < 4.0 and tilt < 1.0 and horiz < 0.5
    return ok, f"dz={st.pos[2] - 80:+.2f} m, horizontal drift={horiz:.3f} m, tilt={tilt:.2f} deg"


def test_angle_step():
    s = _fresh(Simulator.ANGLE)
    _fly(s, 1.0, (0, 0, 0, 0.31))
    peak, settle = 0.0, None

    def hook(t, st):
        nonlocal peak, settle
        a = st.euler[1] / D2R
        peak = max(peak, -a)
        if settle is None and abs(-a - 15) < 0.75 and t > 0.05:
            settle = t
    st = _fly(s, 1.5, (0, 0.2727, 0, 0.31), hook=hook)   # 0.2727 * 55 deg = 15 deg nose-down
    final = -st.euler[1] / D2R
    over = (peak - 15) / 15 * 100
    ok = abs(final - 15) < 0.6 and over < 12 and settle is not None and settle < 0.6
    return ok, f"final={final:.2f} deg overshoot={over:.1f}% reached +-5% at {settle:.2f}s"


def test_rate_tracking():
    s = _fresh(Simulator.ACRO)
    _fly(s, 0.5, (0, 0, 0, 0.31))
    st = _fly(s, 0.35, (0.5, 0, 0, 0.31))
    want = 200 * 1.2 * 0.5 / (1 - 0.5 * 0.72)
    got = st.gyro[0] / D2R
    return abs(got - want) / want < 0.08, f"roll rate {got:.0f} deg/s (setpoint {want:.0f})"


def test_yaw_direction():
    s = _fresh(Simulator.ACRO)
    st = _fly(s, 1.0, (0, 0, 0.5, 0.31))
    return st.gyro[2] < -1.0, f"yaw-right stick -> yaw rate {st.gyro[2] / D2R:.0f} deg/s (clockwise = negative)"


def test_forward_flight_direction():
    s = _fresh(Simulator.ANGLE)
    st = _fly(s, 2.5, (0, 0.3, 0, 0.33))
    return st.vel[0] > 3.0 and abs(st.vel[1]) < 0.5, f"pitch-forward stick -> vx={st.vel[0]:.1f} m/s"


def test_crash_and_reset():
    s = _fresh(Simulator.ACRO, 30.0)
    st = _fly(s, 5.0, (0, 0, 0, 0.0))
    crashed = st.crashed
    s.reset((0, 0, 0.04, 0))
    return bool(crashed) and not s.state().crashed, f"crashed={bool(crashed)} after free-fall, cleared by reset"


def test_obstacle_collision():
    s = Simulator()
    s.set_obstacles([(10, 0, 2, 0.5, 0.5, 2, 0)])
    s.reset((0, 0, 2.0, 0))
    s.set_mode(Simulator.ANGLE)
    s.set_armed(True)
    st = _fly(s, 4.0, (0, 0.35, 0, 0.33))
    return bool(st.crashed) and st.pos[0] < 10.5, f"hit pylon at x={st.pos[0]:.1f}"


def test_determinism():
    out = []
    for _ in range(2):
        s = _fresh(Simulator.ACRO)
        st = _fly(s, 2.0, lambda t: (0.3 * math.sin(3 * t), 0.2, 0.1, 0.4))
        out.append((st.pos[0], st.pos[2], st.gyro[0]))
    return out[0] == out[1], "bit-identical across runs"


def test_autonomous_lap():
    bus = EventBus()
    course = build_course(0)
    tracker = RaceTracker(course, bus)
    laps, crashes = [], []
    bus.subscribe("lap", lambda lap, time, best: laps.append(time))
    s = Simulator()
    T = course.terrain
    s.set_terrain(T.heights, T.cell, T.ox, T.oy)
    s.set_obstacles(course.collision_boxes())
    s.reset(course.start_pose)
    s.set_mode(Simulator.ANGLE)
    s.set_armed(True)
    ap = Autopilot()
    import numpy as np
    t_end = 0
    for i in range(int(70 / 0.01)):
        st = s.state()
        k = ap.compute(st, course, tracker.next_gate)
        s.set_sticks(k.roll, k.pitch, k.yaw, k.throttle)
        s.advance(0.01)
        st = s.state()
        tracker.update(np.array(st.pos), st.time)
        if st.crashed:
            crashes.append(st.time)
            break
        if laps:
            break
    return bool(laps) and not crashes, f"lap time {laps[0]:.1f}s" if laps else f"no lap (crash at {crashes})"


def test_event_bus():
    bus = EventBus()
    got = []
    off = bus.subscribe("x", lambda v: got.append(v))
    bus.publish("x", v=1)
    off()
    off()                                   # unsubscribing twice must be harmless
    bus.publish("x", v=2)
    return got == [1], f"received {got}"


def test_settings():
    from fpv.settings import Settings
    s = Settings(fov=999, mode="bogus", wind=-5).sanitize()
    first = s.record_lap(40.0)
    slower = s.record_lap(45.0)
    faster = s.record_lap(38.5)
    ok = s.fov == 150 and s.mode == 0 and s.wind == 0 and first and not slower and faster and s.best_lap() == 38.5
    return ok, f"clamped fov={s.fov}, mode={s.mode}; record lap {s.best_lap()}s"


def test_gate_tracking():
    import numpy as np
    bus = EventBus()
    course = build_course(0)
    tr = RaceTracker(course, bus)
    events = []
    bus.subscribe("gate", lambda index, next: events.append(("hit", index)))
    bus.subscribe("gate_missed", lambda index: events.append(("miss", index)))
    g = course.gates[0]
    off = g.pos + g.lateral * 6                      # crossing the plane far outside the opening
    tr.update(off - g.normal, 0.0)
    tr.update(off + g.normal, 0.1)
    tr.update(g.pos - g.normal * 2, 0.2)             # now straight through the middle
    tr.update(g.pos + g.normal * 2, 0.3)
    ok = events == [("miss", 0), ("hit", 0)] and tr.next_gate == 1
    return ok, f"events={events}, next gate={tr.next_gate}"


def test_button_bar_layout():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from fpv.render.widgets import Button, ButtonBar
    pygame.init()
    clicked = []
    bar = ButtonBar([Button(lambda i=i: str(i), lambda i=i: clicked.append(i)) for i in range(7)])
    ok = True
    for size in ((1280, 720), (640, 480), (400, 300)):          # must stay on-screen and non-overlapping
        rs = bar.rects(size)
        ok &= all(r.left >= 0 and r.right <= size[0] and r.bottom <= size[1] for r in rs)
        ok &= all(a.right <= b.left for a, b in zip(rs, rs[1:]))
    r3 = bar.rects((1280, 720))[3]
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=r3.center)
    ok &= bar.handle_event(ev, (1280, 720)) and clicked == [3]
    return bool(ok), "7 buttons fit at 1280/640/400 px wide; click hit-testing maps to the right button"


def test_terrain_matches_physics():
    import numpy as np
    from fpv.game.terrain import Terrain
    T = Terrain()
    s = Simulator()
    s.set_terrain(T.heights, T.cell, T.ox, T.oy)
    pts = [(10, 10), (-3, 4), (200, -40), (-700, 900), (55.5, 61.2)]
    err = max(abs(T.height(x, y) - s.ground_height(x, y)) for x, y in pts)
    return err < 1e-3, f"python vs C++ ground height max error {err:.1e} m"


def test_terrain_landing_and_crash():
    from fpv.game.course import default_terrain
    T = default_terrain()
    s = Simulator()
    s.set_terrain(T.heights, T.cell, T.ox, T.oy)
    x, y = 60.0, 60.0
    g = float(T.height(x, y))
    s.reset((x, y, g + 6.0, 0))
    s.set_mode(Simulator.ANGLE)
    s.set_armed(True)
    st = _fly(s, 3.0, (0, 0, 0, 0.0))                       # power cut: falls onto the hill
    on_ground = abs(st.pos[2] - (T.height(st.pos[0], st.pos[1]) + 0.04)) < 0.2
    return bool(st.crashed) and on_ground, f"free-fall onto terrain: crashed={bool(st.crashed)}, rests on surface={on_ground}"


def test_target_pop():
    import numpy as np
    from fpv.game.targets import TargetField, segment_hits_sphere
    bus = EventBus()
    got = []
    bus.subscribe("target_hit", lambda target, points, combo, remaining: got.append((points, combo, remaining)))
    f = TargetField(bus)
    t = f.place(10, 0, 3, None)
    f.update(np.array([0.0, 0.0, 3.0]), 0.0)
    f.update(np.array([20.0, 0.1, 3.0]), 0.5)                 # flew straight through in one frame (no tunnelling)
    miss = not segment_hits_sphere(np.array([0.0, 8.0, 3.0]), np.array([20.0, 8.0, 3.0]), t.pos, 1.9)
    return got == [(100, 1, 0)] and not t.alive and miss, f"events={got}, far pass misses={miss}"


def test_mission_flow():
    import numpy as np
    from fpv.game.missions import MissionRunner, Phase, mission_by_id
    bus = EventBus()
    r = MissionRunner(mission_by_id("training"), bus)
    ground = float(r.course.ground(*r.course.start_pose[:2]))

    class St:                                               # minimal stand-in for the native state
        armed, crashed = 1, 0
        pos = (0.0, 0.0, 0.0)
    st = St()
    st.pos = (r.course.start_pose[0], r.course.start_pose[1], ground + 0.1)
    r.update(0.01, st, 1.0)
    waiting = r.phase is Phase.READY
    st.pos = (r.course.start_pose[0], r.course.start_pose[1], ground + 3.0)
    r.update(0.01, st, 2.0)
    started = r.phase is Phase.ACTIVE
    for i, tg in enumerate(list(r.field.targets)):          # fly through every target
        st.pos = tuple(tg.pos - np.array([3, 0, 0]))
        r.update(0.01, st, 3.0 + i)
        st.pos = tuple(tg.pos + np.array([3, 0, 0]))
        r.update(0.01, st, 3.5 + i)
    done = r.phase is Phase.COMPLETE and r.result["stars"] == 3
    return waiting and started and done, f"ready->active at take-off, 5 pops -> complete, stars={r.result and r.result['stars']}"


def test_audio_synth():
    import numpy as np
    from fpv.audio import CHUNK, RATE, MotorSynth, make_sfx
    ms = MotorSynth()
    n = int(RATE * CHUNK)
    y = np.concatenate([ms.render([0.3] * 4, 0.0, True, n) for _ in range(20)])
    step = float(np.abs(np.diff(y)).max())                  # click-free: no big jumps at chunk boundaries

    def peak(m):
        s2 = MotorSynth()
        for _ in range(8):
            s2.render([m] * 4, 0.0, True, n)
        z = np.concatenate([s2.render([m] * 4, 0.0, True, n) for _ in range(16)])
        S = np.abs(np.fft.rfft(z * np.hanning(len(z))))
        f = np.fft.rfftfreq(len(z), 1 / RATE)
        return f[(f > 60)][S[f > 60].argmax()]
    lo, hi = peak(0.15), peak(0.85)
    sfx = make_sfx()
    silent = float(np.abs(MotorSynth().render([0.5] * 4, 0.0, False, n)).max())
    ok = step < 0.2 and hi > lo * 2 and len(sfx) >= 10 and all(np.abs(v).max() > 0.1 for v in sfx.values()) and silent < 1e-6
    return bool(ok), f"max step {step:.3f}, motor pitch {lo:.0f} Hz -> {hi:.0f} Hz with throttle, {len(sfx)} effects"


def test_virtual_sticks():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from fpv.input.sticks import Sticks
    from fpv.input.virtual import VirtualControls
    pygame.init()
    v = VirtualControls()
    v.layout((1280, 720), 58)
    size = (1280, 720)
    cx, cy = v.left.centre
    r = v.left.radius
    ev = lambda t, p, **k: pygame.event.Event(t, pos=p, button=1, buttons=(1, 0, 0), **k)  # noqa: E731
    v.handle_event(ev(pygame.MOUSEBUTTONDOWN, (cx, cy + r)), size)
    v.handle_event(ev(pygame.MOUSEMOTION, (cx + r // 2, cy - r), rel=(0, 0)), size)
    held = v.apply(Sticks())
    v.handle_event(ev(pygame.MOUSEBUTTONUP, (cx, cy)), size)
    after = v.apply(Sticks(throttle=held.throttle))
    rx, ry = v.right.centre
    v.handle_event(ev(pygame.MOUSEBUTTONDOWN, (rx, ry)), size)
    v.handle_event(ev(pygame.MOUSEMOTION, (rx + 10 * r, ry - 10 * r), rel=(0, 0)), size)   # dragged far: clamped to the rim
    rr = v.apply(Sticks())
    ok = (held.throttle > 0.9 and 0.2 < held.yaw < 0.7 and after.yaw == 0 and
          abs((rr.roll ** 2 + rr.pitch ** 2) ** 0.5 - 1.0) < 1e-6 and rr.pitch > 0.6)
    return bool(ok), f"throttle {held.throttle:.2f}, yaw {held.yaw:.2f} -> {after.yaw:.1f} on release; rim-clamped {rr.roll:.2f},{rr.pitch:.2f}"


def test_throughput():
    s = _fresh(Simulator.ACRO, 50.0)
    t0 = time.perf_counter()
    n = 0
    for i in range(60000):
        s.set_sticks(0.3 * math.sin(i * 0.002), 0.2, 0.1, 0.35)
        n += s.advance(0.002)
    wall = time.perf_counter() - t0
    sim_s = 60000 * 0.002
    return sim_s / wall > 50, f"{sim_s / wall:.0f}x real-time through ctypes ({n / wall / 1e6:.2f} M steps/s)"


# ----------------------------------------------------------------------------------------------- cities
_SAMPLE_CITIES = ["kampala", "mbarara", "kasese", "kabale", "moroto", "jinja"]
_world_cache: dict = {}


def _world(cid: str):
    from fpv.world.cities import find
    from fpv.world.generator import build_world
    if cid not in _world_cache:
        _world_cache[cid] = build_world(find(cid))
    return _world_cache[cid]


def test_city_catalogue():
    from fpv.world import cities
    ids = [c.id for c in cities.CITIES]
    named = {"kampala", "mbarara", "gulu", "jinja", "mbale", "arua", "kabale", "fortportal", "entebbe", "lira"}
    ok = (len(ids) == len(set(ids)) and named <= set(ids) and len(ids) >= 40 and
          all(cities.in_uganda(c.lat, c.lon) for c in cities.CITIES) and
          cities.search("mbar")[0].id == "mbarara" and not cities.in_uganda(51.5, -0.1))
    cc = cities.custom_city(0.35, 32.58)
    return ok and cc.custom, f"{len(ids)} cities, ids unique, all inside Uganda, search + custom location work"


def test_city_generation():
    import numpy as np
    bad, info = [], []
    for cid in _SAMPLE_CITIES:
        w = _world(cid)
        T = w.terrain
        n_b, n_t = len(w.buildings), len(T.trees.height)
        px, py = w.pad
        near = (np.hypot(w.buildings.pos[:, 0] - px, w.buildings.pos[:, 1] - py) < 20).sum() if n_b else 0
        flat = abs(T.height(px + 6, py) - T.height(px - 6, py)) < 1.5 and T.height(px, py) > 1.0
        finite = bool(np.isfinite(T.heights).all())
        if not (n_b > 150 and n_t > 100 and near == 0 and flat and finite and len(w.roads) > 10):
            bad.append((cid, n_b, n_t, int(near), flat, finite))
        info.append(f"{cid}:{n_b}b/{n_t}t")
    return not bad, (", ".join(info) if not bad else f"invalid worlds: {bad}")


def test_city_determinism():
    from fpv.world.cities import find
    from fpv.world.generator import build_world
    a, b = _world("mbarara"), build_world(find("mbarara"))
    same = (len(a.buildings) == len(b.buildings) and (a.terrain.heights == b.terrain.heights).all()
            and (a.buildings.pos == b.buildings.pos).all())
    other = _world("gulu")
    return same and len(other.buildings) != len(a.buildings), "same city -> identical world; different city -> different"


def test_city_autopilot_laps():
    import numpy as np
    from fpv.game.autopilot import Autopilot
    res = []
    for cid in ("kampala", "mbarara", "kabale"):
        T = _world(cid).terrain
        c = build_course(0, T)
        s = Simulator()
        s.set_terrain(T.heights, T.cell, T.ox, T.oy)
        s.set_obstacles(c.collision_boxes())
        s.reset(c.start_pose)
        s.set_mode(Simulator.ANGLE)
        s.set_armed(True)
        laps = []
        bus = EventBus()
        bus.subscribe("lap", lambda lap, time, best: laps.append(time))
        tr, ap = RaceTracker(c, bus), Autopilot()
        t = 0.0
        while t < 150 and not laps:
            st = s.state()
            k = ap.compute(st, c, tr.next_gate, None)
            s.set_sticks(k.roll, k.pitch, k.yaw, k.throttle)
            s.advance(0.01)
            t += 0.01
            st = s.state()
            tr.update(np.array(st.pos), st.time)
            if st.crashed:
                break
        res.append((cid, laps[0] if laps else None))
    return all(x[1] for x in res), "  ".join(f"{c}: {('%.0fs' % t) if t else 'FAILED'}" for c, t in res)


def test_world_queries():
    w = _world("kampala")
    b = w.buildings
    i = int(max(range(len(b)), key=lambda k: b.wall[k]))
    x, y = float(b.pos[i, 0]), float(b.pos[i, 1])
    roof = w.roof_height(x, y)
    far = w.roof_height(w.pad[0], w.pad[1])
    ok = roof > 8 and far == 0.0 and w.clearance(x, y) > w.terrain.ground(x, y) + 8
    return ok, f"tallest building {b.wall[i]:.0f} m reports roof {roof:.0f} m; pad reports {far:.0f} m"


def test_city_flow_and_records():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from fpv.app import App, CitySelectState, FlightState, MissionSelectState
    from fpv.game.missions import mission_by_id
    from fpv.settings import Settings
    s = Settings(city="mbarara")
    app = App(s, auto_build=False)
    try:
        picker = CitySelectState(app)
        app.push(picker)
        for ch in "kab":
            picker.handle_event(pygame.event.Event(pygame.KEYDOWN, key=ord(ch), unicode=ch))
        top = picker.entries[0].id
        picker.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"))
        for _ in range(400):                                   # wait for the background build, then it proceeds
            app.tick(0.02)
            if isinstance(app.stack[-1], MissionSelectState):
                break
            pygame.time.wait(20)
        switched = isinstance(app.stack[-1], MissionSelectState) and app.ctx.city.id == "kabale" == top
        app.push(FlightState(app, mission_by_id("race")))
        for _ in range(5):
            app.tick(0.02)
        gates = len(app.ctx.course.gates)
        r = {"value": 100.0, "stars": 2}
        s.record_result(mission_by_id("race"), r)
        s.city = "gulu"
        isolated = s.record("race") is None
        s.city = "kabale"
        kept = s.record("race") is not None
        return switched and gates == 10 and isolated and kept, f"typed 'kab' -> {top}, city switched, {gates} gates, records are per city"
    finally:
        pygame.quit()


def test_render_all_times():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import pygame
    pygame.init()
    surf = pygame.display.set_mode((640, 360))
    from fpv.render.atmosphere import TIMES
    from fpv.render.camera import View, look_at
    from fpv.render.scene import WorldRenderer
    T = _world("fortportal").terrain
    r = WorldRenderer(build_course(1, T))
    px, py = T.pad
    eye = np.array([px, py, T.ground(px, py) + 25.0])
    v = View(eye, look_at(eye, eye + np.array([1.0, 0.3, -0.15])), 640, 360, 100)
    means = []
    for name in TIMES:
        r.set_time_of_day(name)
        r.draw(surf, v, None, 0, set(), 0.016, False, [], None)
        means.append(float(np.asarray(pygame.surfarray.array3d(surf)).mean()))
    pygame.quit()
    ok = all(m > 5 for m in means) and means[1] > means[3]                # midday brighter than dusk
    return ok, "frames at " + ", ".join(f"{n} {m:.0f}" for n, m in zip(TIMES, means)) + " (mean brightness)"


def test_osm_parser():
    import numpy as np
    from fpv.world import online
    from fpv.world.cities import find
    from fpv.world.generator import build_world
    city = find("mbarara")
    s, w, n, e = online.bbox(city)
    dlat, dlon = (n - s) / 1200.0, (e - w) / 1200.0              # degrees per metre
    clat, clon = city.lat, city.lon

    def way(kind, tags, pts_m):                                    # pts in metres east/north of the centre
        return {"type": "way", "tags": {kind: tags[kind], **tags},
                "geometry": [{"lat": clat + y * dlat, "lon": clon + x * dlon} for x, y in pts_m]}
    els = []
    for k in range(-5, 6):                                         # street grid
        els.append(way("highway", {"highway": "residential"}, [(k * 80, -500), (k * 80, 500)]))
        els.append(way("highway", {"highway": "tertiary" if k else "primary"}, [(-500, k * 80), (500, k * 80)]))
    for i in range(-4, 4):                                         # a block of buildings, one tall
        for j in range(-4, 4):
            x0, y0 = i * 80 + 20, j * 80 + 20
            tags = {"building": "yes", "building:levels": "12" if (i, j) == (2, 2) else "2"}
            els.append(way("building", tags, [(x0, y0), (x0 + 30, y0), (x0 + 30, y0 + 18), (x0, y0 + 18), (x0, y0)]))
    els.append(way("natural", {"natural": "water"}, [(350, 350), (450, 350), (450, 450), (350, 450), (350, 350)]))
    elev = [1400.0 + 0.05 * (i % 33) + 0.02 * (i // 33) for i in range(33 * 33)]
    rd = online.build_real({"elements": els}, elev, city)
    tall = rd.buildings.floors.max() if rd.buildings is not None else 0
    wx, wy = online.to_world(clat + 400 * dlat, clon + 400 * dlon, city)[0]
    parsed = (rd.roads is not None and len(rd.roads) == 22 and len(rd.buildings) == 64 and tall == 12 and
              rd.heights is not None and rd.heights.shape == (257, 257) and
              abs(float(rd.heights[int((wx - online.ORIGIN[0]) / online.CELL), int((wy - online.ORIGIN[1]) / online.CELL)]) + 2.5) < 1e-3)
    world = build_world(city, real=rd)                           # the generator must accept it end to end
    ok = parsed and world.source == "openstreetmap" and len(world.buildings) > 40 and world.terrain.height(*world.pad) > 0
    return ok, f"{rd.note}; generator accepted it ({len(world.buildings)} buildings, source={world.source}); NOT tested against live servers"


TESTS = [test_hover, test_angle_step, test_rate_tracking, test_yaw_direction, test_forward_flight_direction,
         test_crash_and_reset, test_obstacle_collision, test_determinism, test_autonomous_lap, test_event_bus,
         test_settings, test_gate_tracking, test_button_bar_layout, test_terrain_matches_physics,
         test_terrain_landing_and_crash, test_target_pop, test_mission_flow, test_audio_synth, test_virtual_sticks,
         test_throughput, test_city_catalogue, test_city_generation, test_city_determinism, test_world_queries,
         test_city_autopilot_laps, test_render_all_times, test_osm_parser, test_city_flow_and_records]


def run() -> int:
    failed = 0
    for t in TESTS:
        try:
            ok, msg = t()
        except Exception as ex:  # noqa: BLE001
            ok, msg = False, f"exception: {ex!r}"
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {t.__name__[5:]:<24} {msg}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
