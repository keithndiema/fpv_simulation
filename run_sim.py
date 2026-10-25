#!/usr/bin/env python3
"""FPV Sim launcher.

  python run_sim.py              launch the simulator (auto-builds the C++ core on first run)
  python run_sim.py --build      (re)build the C++ core only
  python run_sim.py --bench      headless physics throughput benchmark
  python run_sim.py --selftest   run the regression suite
  python run_sim.py --demo       start directly in autopilot demo flight
  python run_sim.py --city mbarara --time Dusk   start in a chosen Ugandan town (see --list-cities)
  python run_sim.py --live       try real OpenStreetMap streets/buildings (internet; falls back to generated towns)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import numpy, pygame  # noqa: F401
    except ImportError as ex:
        print(f"Missing dependency ({ex.name}). Easiest fix:  ./run.sh   (or: pip install -r requirements.txt)")
        return 1
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--city", help="town id or name, e.g. kampala, mbarara, gulu, valley (original practice valley)")
    ap.add_argument("--time", help="time of day: Morning, Midday, Golden hour or Dusk")
    ap.add_argument("--live", action="store_true", help="use live OpenStreetMap data when it can be downloaded")
    ap.add_argument("--list-cities", action="store_true")
    ap.add_argument("--no-build", action="store_true", help="skip the automatic rebuild check")
    a = ap.parse_args()

    from fpv.sim import native
    if a.list_cities:
        from fpv.world import cities
        for c in cities.CITIES:
            print(f"{c.id:<12} {c.name:<14} {c.region:<14} {c.coords}")
        return 0
    if a.build:
        print("built:", native.build())
        return 0
    if a.selftest:
        from tests.selftest import run
        return run()
    if a.bench:
        lib = native.load_library(not a.no_build)
        del lib
        cli = next(native.BUILD.rglob("fpv_cli*"), None)
        return subprocess.call([str(cli), "bench", "600"]) if cli else 1

    from fpv.app import App, FlightState
    from fpv.game.missions import mission_by_id
    from fpv.settings import Settings
    s = Settings.load()
    if a.fullscreen:
        s.fullscreen = True
    if a.city:
        from fpv.world import cities
        hit = cities.find(a.city.lower()) or next(iter(cities.search(a.city)), None)
        if a.city.lower() == "valley":
            s.city = "valley"
        elif hit is None:
            print(f"unknown city '{a.city}' - try --list-cities")
            return 1
        else:
            s.city = hit.id
    if a.time:
        from fpv.render.atmosphere import TIMES
        match = [i for i, t in enumerate(TIMES) if t.lower().startswith(a.time.lower())]
        if not match:
            print(f"unknown time '{a.time}' - choose from {', '.join(TIMES)}")
            return 1
        s.time_of_day = match[0]
    if a.live:
        s.live_data = True
    app = App(s, auto_build=not a.no_build)
    if a.demo:
        app.push(FlightState(app, mission_by_id("race"), demo=True))
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
