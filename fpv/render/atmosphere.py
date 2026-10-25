"""Time-of-day + air quality presets: sky colours, sun direction/colour, fog, window glow.

Sun position is given as a compass bearing + elevation (Uganda sits on the equator, so the midday sun is almost
straight overhead and sunrise / sunset are nearly due east / west - this is why the presets look the way they do).
World frame: +x north, +y west, so east is -y.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TIMES = ["Morning", "Midday", "Golden hour", "Dusk"]


@dataclass
class Atmosphere:
    name: str
    zenith: tuple
    mid: tuple
    haze: tuple                 # horizon colour == fog colour
    sun_dir: np.ndarray         # unit vector towards the sun
    sun_col: np.ndarray         # direct light colour multiplier (rgb 0..1.3)
    amb_col: np.ndarray         # ambient (sky) light multiplier
    fog_start: float
    fog_end: float
    glow: float                 # 0 = windows dark .. 1 = lit (dusk)
    cloud_tint: tuple
    sun_visible: float = 1.0
    local_hour: float = 12.0

    @property
    def sun_el(self) -> float:
        return math.degrees(math.asin(float(self.sun_dir[2])))


def _sun(bearing: float, elevation: float) -> np.ndarray:
    b, e = math.radians(bearing), math.radians(elevation)
    return np.array([math.cos(b) * math.cos(e), -math.sin(b) * math.cos(e), math.sin(e)])


_PRESETS = {
    "Morning": dict(zenith=(70, 128, 206), mid=(142, 186, 232), haze=(214, 214, 206), bearing=96, elev=24,
                    sun=(0.78, 0.66, 0.50), amb=(0.34, 0.38, 0.46), fog=(40, 360), glow=0.0, cloud=(255, 238, 220), hour=7.5),
    "Midday": dict(zenith=(46, 112, 212), mid=(116, 176, 238), haze=(196, 214, 228), bearing=176, elev=72,
                   sun=(0.72, 0.70, 0.64), amb=(0.36, 0.40, 0.50), fog=(50, 360), glow=0.0, cloud=(255, 255, 255), hour=12.5),
    "Golden hour": dict(zenith=(70, 98, 170), mid=(222, 168, 130), haze=(238, 190, 140), bearing=272, elev=11,
                        sun=(0.96, 0.62, 0.34), amb=(0.32, 0.30, 0.38), fog=(35, 330), glow=0.15, cloud=(255, 206, 160), hour=17.7),
    "Dusk": dict(zenith=(26, 34, 84), mid=(124, 98, 128), haze=(214, 118, 92), bearing=278, elev=2.5,
                 sun=(0.70, 0.34, 0.18), amb=(0.26, 0.26, 0.40), fog=(30, 300), glow=1.0, cloud=(226, 140, 124), hour=18.6),
}


def make(time_of_day: str, haze: float = 0.2) -> Atmosphere:
    """Preset for a time of day, dulled / warmed by `haze` (0 clear mountain air .. 1 dusty harmattan-like haze)."""
    p = _PRESETS.get(time_of_day, _PRESETS["Midday"])
    dust = np.array((206, 186, 150), float)
    mixw = 0.45 * haze

    def dull(c, k=1.0):
        return tuple(int(x + (d - x) * mixw * k) for x, d in zip(c, dust))

    return Atmosphere(
        name=time_of_day, zenith=dull(p["zenith"], 0.5), mid=dull(p["mid"], 0.8), haze=dull(p["haze"], 1.0),
        sun_dir=_sun(p["bearing"], p["elev"]), sun_col=np.array(p["sun"]) * (1.0 - 0.18 * haze),
        amb_col=np.array(p["amb"]), fog_start=p["fog"][0], fog_end=p["fog"][1] * (1.0 - 0.38 * haze),
        glow=p["glow"], cloud_tint=p["cloud"], local_hour=p["hour"])
