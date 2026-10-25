# FPV Sim v4 — fly a drone over any town in Uganda

A real-time first-person FPV quadcopter simulator for PC. The 6-DOF physics and the flight controller are C++
(2 kHz, RK4); Python does the world, rendering, sound, missions and UI.

**New in v4: choose a city.** Pick Kampala, Mbarara, Gulu, Jinja, Kabale, Kasese... (45 towns, plus any latitude /
longitude you type) and fly over streets, houses, tower blocks, landmarks, trees, rivers, lakes and the distant
mountains of that region, at morning, midday, golden hour or dusk.

## Run
```
./run.sh                           # one command: venv + deps + C++ build + launch
python run_sim.py                   # first run auto-builds the C++ core (needs CMake or g++/clang + C++17)
```
Other commands: `--city mbarara`, `--time Dusk`, `--list-cities`, `--live`, `--demo` (autopilot flight), `--selftest` (28 regression tests, no display needed), `--bench`, `--build`, `--fullscreen`.

## What you get
- **A real world**: procedural hills, a lake, beaches, forests (thousands of trees), buildings and pylons, mountains on
  the horizon, sun, clouds and distance fog. The *same* heightfield feeds the renderer and the physics, so hills,
  trees and buildings are solid and crash you.
- **True FPV**: tilted camera with adjustable FOV, motor vibration, optional analog-video look (scanlines, crash static).
- **Targets you pop**: bullseye targets float over the terrain. Fly through them (swept collision, no tunnelling).
  The nearest one gets lock brackets + distance, an edge arrow points to it, and chained pops build a x5 combo.
- **Missions with goals and stars**: First Flight, Bullseye Run (10 in 150 s), Circuit Race (2 laps of 10 gates; star times scale with the town's circuit),
  Time Attack (respawning targets, 75 s), Free Flight. Timer starts at take-off; crash / time-up fails; best
  results and stars are saved.
- **Sound**: live synthesised motor whine for each of the four motors (pitch follows the real motor commands), wind
  noise that follows airspeed, plus pops, gate chimes, arm/disarm, crash, countdown-style cues, win/fail jingles.
- **Real controls on screen**: two draggable sticks (mouse or multi-touch) — left = throttle (latches) + yaw,
  right = pitch + roll — plus a button bar: ARM, RESTART, ACRO/ANGLE, FPV/CHASE, AUTO, SOUND, PAUSE, HELP.
  Keyboard and gamepad work too; touching the sticks while the autopilot flies takes over instantly.


## Choosing a city
Main menu → **Fly** opens the city picker: scroll or **type to filter** (`mba` → Mbarara, Mbale), see a top-down map
and facts, press **TAB / arrows** for the time of day, **ENTER** to fly. Type coordinates such as `0.35, 32.58` to
fly over *any* point in Uganda. Towns are built on a background thread (1–3 s) and cached. "Practice valley" is the
original training course. Records and stars are kept **per city**. The pause menu can change time of day and city.

What each town has: terrain with the right character (Kampala's hills and Lake Victoria shore, Mbarara's Rwizi
valley, Kasese below the Rwenzori snow peaks, Moroto's crater and dry plain, Kabale's terraced hills...), a street
network with primary / secondary / local / murram roads and bridges, thousands of buildings by district (glass
towers and mid-rise in the centre, shophouses, brick houses with gable roofs, informal huts, farm plots), landmarks
(towers, mosques, cathedrals, stadium, dams, terminal, clock-tower roundabout), species-specific trees (jacaranda,
flame tree, eucalyptus, palms, bananas, acacia), moving traffic on the left, birds, shadows, sun glint on water.
The race course is 10 gates on the streets around the park the launch pad sits in. The OSD shows the city,
coordinates, altitude above sea level and compass heading; the minimap shows the real town map.

### How real is it?
Honest answer: the towns are **procedurally generated from a hand-tuned profile of each place** (real coordinates
and elevations, the right region, rivers, lakes, mountains and landmarks in roughly the right directions), so they
*look and feel* like Ugandan towns but are **not an exact map** of each street. For true streets and buildings,
press **F2** in the city picker (or run with `--live`): the simulator then tries to download OpenStreetMap
streets/buildings (Overpass API) and elevation (Open-Meteo) for that spot, caches them in `~/.fpv_sim/cache`, and
falls back to the generated town if the download fails. The converter for that data is unit-tested on synthetic
OSM data, but the downloads themselves have **not been tested against the live servers** in the environment this
was built in, so treat live mode as experimental.

## Controls
| Key | Action | | Key | Action |
|---|---|---|---|---|
| W / S | throttle up / down (Shift fast, X cut) | | SPACE | arm / disarm (throttle must be low) |
| A / D | yaw | | R | restart mission |
| Arrows | pitch / roll | | M | ACRO / ANGLE |
| C | FPV / chase camera | | T | autopilot |
| G | rate graph | | N | mute |
| F5 | CSV blackbox log | | F1 / H | help |
| ESC / P | pause menu | | F11 | fullscreen |

Gamepad (mode 2): sticks fly; A arm, B reset, X camera, Y mode. Axis map: `GamepadDevice.AXES`.

## How a mission goes
1. Pick a mission (arrows / mouse). Read the briefing.
2. **Arm** (SPACE or the ARM button), raise the throttle (W or drag the left stick up) and take off — the clock starts.
3. Hit the objective. Tips: ANGLE mode (M) self-levels; ACRO is for pros.
4. Result screen: stars, time/score, new-best banner, Retry / Next / Mission select.

## Architecture
```
core/ (C++17)                          fpv/ (Python)
  types.hpp       math, POD structs      sim/native.py      ctypes facade, auto-build, ABI v2 check
  controller.hpp  PIDF + angle + mixer   app.py             State pattern: Menu/Missions/Settings/Flight/Pause/Result
  simulator.cpp   RK4 6-DOF, battery,    game/terrain.py    heightfield, trees (shared with C++); practice valley
                  wind, heightfield      game/course.py     gates + buildings placed on the terrain
                  terrain + slope        game/targets.py    swept hit-testing, combo scoring
                  contact, obstacles     game/missions.py   objectives, state machine, star ratings
  c_api.cpp       flat C ABI             game/autopilot.py  terrain-following gate/target pursuit
  cli.cpp         step-response/bench    audio.py           numpy motor/wind synth + SFX, streamed to the mixer
                                         input/virtual.py   mouse/touch sticks;  input/devices.py keyboard/gamepad
                                         world/cities.py    45 town profiles + custom lat/lon
                                         world/generator.py terrain, streets, parcels, trees, landcover (seeded per city)
                                         world/features.py  landmark meshes;  world/online.py optional OSM + elevation
                                         world/loader.py    background build + cache;  world/preview.py map image
                                         render/scene.py    sky, mountains, lit+fogged terrain (2 LODs), trees, targets
                                         render/city.py     buildings, roads, vehicles, birds, landmark meshes
                                         render/atmosphere.py time-of-day presets
                                         render/hud.py      OSD, objective card, target markers, minimap
```
Rendering is a numpy-vectorised painter's algorithm in pure pygame (no GPU needed); a town frame is a few thousand polygons and takes roughly 20-30 ms on a modest CPU.
Graphics quality (Settings) trades draw distance for speed and lowers itself automatically if the frame rate drops.

Physics runs at 2 kHz with the controller at 1 kHz. Conventions: body frame FLU, stick right/forward = roll right/nose down.
Tuning: PID/rates/wind are live in Settings → Drone tuning; defaults live in `core/include/fpv/types.hpp`.

World frame: +x north, +y west, +z up (a compass bearing b is the vector (cos b, -sin b)); the town centre is at (40, 40).
Generation is deterministic per city (seeded), so everyone flying "Mbarara" sees the same town.
