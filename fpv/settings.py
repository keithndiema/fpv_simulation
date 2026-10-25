"""Persistent user settings (JSON in ~/.fpv_sim/settings.json)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DIR = Path.home() / ".fpv_sim"
FILE = DIR / "settings.json"


@dataclass
class Settings:
    mode: int = 0                 # 0 = Acro, 1 = Angle
    course: int = 0               # 0 = Race circuit, 1 = Open field
    cam_tilt: float = 25.0        # FPV camera uptilt [deg]
    fov: float = 120.0            # horizontal FOV [deg]
    rate_scale: float = 1.0       # multiplies max rotation rates
    pid_p: float = 1.0            # PID multipliers (live tuning)
    pid_i: float = 1.0
    pid_d: float = 1.0
    wind: float = 0.0             # mean wind [m/s]
    gust: float = 0.0             # gust intensity [m/s]
    kb_sensitivity: float = 0.75  # keyboard stick deflection
    show_graph: bool = True
    show_sticks: bool = True
    chase_cam: bool = False
    fps_cap: int = 120
    fullscreen: bool = False
    best_laps: dict = field(default_factory=dict)   # course index (as str) -> best lap time [s]
    records: dict = field(default_factory=dict)     # mission id -> {"value": float, "stars": int}
    volume: float = 0.7           # master volume 0..1
    muted: bool = False
    cam_shake: float = 1.0        # FPV motor-vibration amount 0..1.5
    analog_fx: bool = True        # scanlines + crash static
    quality: int = 2              # 0 low .. 2 high (terrain / tree draw distance)
    onscreen_sticks: bool = True  # clickable / touchable virtual sticks
    intro_seen: bool = False
    city: str = "kampala"         # id from world.cities, or "valley" for the original practice valley
    time_of_day: int = 1          # index into render.atmosphere.TIMES
    live_data: bool = False       # try real OpenStreetMap streets/buildings first (needs internet)
    custom_lat: float = 0.3476    # last custom location typed in the city picker
    custom_lon: float = 32.5825

    # (min, max) for every numeric field, so a hand-edited / corrupt settings file can't break the sim
    _LIMITS = {"mode": (0, 1), "course": (0, 1), "cam_tilt": (0, 55), "fov": (80, 150), "rate_scale": (0.5, 1.6),
               "pid_p": (0.4, 1.8), "pid_i": (0.0, 2.5), "pid_d": (0.0, 3.0), "wind": (0, 14), "gust": (0, 6),
               "kb_sensitivity": (0.2, 1.0), "fps_cap": (30, 360), "volume": (0.0, 1.0),
               "cam_shake": (0.0, 1.5), "quality": (0, 2), "time_of_day": (0, 3),
               "custom_lat": (-1.6, 4.3), "custom_lon": (29.5, 35.1)}

    def sanitize(self) -> "Settings":
        defaults = Settings()
        for name, (lo, hi) in self._LIMITS.items():
            v, dflt = getattr(self, name), getattr(defaults, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                setattr(self, name, dflt)            # wrong type -> fall back to the default
            else:
                setattr(self, name, type(dflt)(max(lo, min(hi, v))))
        if not isinstance(self.city, str):
            self.city = "kampala"
        for name in ("best_laps", "records"):
            if not isinstance(getattr(self, name), dict):
                setattr(self, name, {})
        return self

    # Records are kept per city: the same mission is a different challenge in Kampala than in Kabale.
    # The original valley keeps the old plain keys so existing saves still work.
    def rkey(self, mission_id: str) -> str:
        return mission_id if self.city in ("valley", "") else f"{self.city}:{mission_id}"

    def record(self, mission_id: str):
        return self.records.get(self.rkey(mission_id))

    def record_lap(self, lap_time: float) -> bool:
        """Store lap_time as the course record if it beats the previous one. Returns True on a new record."""
        key = f"{self.city}:{self.course}" if self.city not in ("valley", "") else str(self.course)
        old = self.best_laps.get(key)
        if old is None or lap_time < old:
            self.best_laps[key] = round(lap_time, 3)
            return True
        return False

    def record_result(self, mission, result: dict) -> bool:
        """Persist a mission result if it beats the stored one (stars first, then the value). True = new best."""
        if result.get("value") is None:
            return False
        higher_is_better = mission.kind == "attack"
        old = self.records.get(self.rkey(mission.id))
        new = {"value": round(float(result["value"]), 2), "stars": int(result["stars"])}
        better = old is None or new["stars"] > old.get("stars", 0) or (
            new["stars"] == old.get("stars", 0) and (new["value"] > old["value"] if higher_is_better
                                                      else new["value"] < old["value"]))
        if better:
            self.records[self.rkey(mission.id)] = new
        return better

    def stars(self, mission_id: str) -> int:
        return int(self.records.get(self.rkey(mission_id), {}).get("stars", 0))

    def best_lap(self) -> float | None:
        return self.best_laps.get(f"{self.city}:{self.course}" if self.city not in ("valley", "") else str(self.course))

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(FILE.read_text())
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in raw.items() if k in known}).sanitize()
        except (OSError, ValueError, TypeError, AttributeError):
            return cls()

    def save(self) -> None:
        try:
            DIR.mkdir(parents=True, exist_ok=True)
            # write-then-rename so a crash mid-save can't leave a truncated settings file
            tmp = FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2))
            tmp.replace(FILE)
        except OSError:
            pass

    def apply_to(self, cfg) -> None:
        """Push tuning-related settings into a native Config struct."""
        base_rates = (1.2, 1.2, 1.1)
        for i in range(3):
            cfg.rc_rate[i] = base_rates[i] * self.rate_scale
        base = ((0.030, 0.35, 0.00055), (0.030, 0.35, 0.00055), (0.14, 0.9, 0.0))
        for i, (p, ii, d) in enumerate(base):
            cfg.kp[i], cfg.ki[i], cfg.kd[i] = p * self.pid_p, ii * self.pid_i, d * self.pid_d
        cfg.wind_speed, cfg.gust = self.wind, self.gust
