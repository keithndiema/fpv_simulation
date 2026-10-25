"""Ugandan places: one profile per town, describing what makes the place look like itself.

The sim shows a 1.3 km x 1.3 km window around the town centre. Profiles say where the lake / river /
rock / hills / mountains are (compass bearings from the centre), how the streets are laid out and what the
countryside looks like. With an internet connection the generator can replace the procedural streets and
buildings with real OpenStreetMap data and real elevation (see online.py); the profile still supplies the sky,
distant mountains and landmarks.

Coordinates are the town centre to about 3 decimals (~100 m); elevations are approximate metres above sea level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Feature:
    """A terrain feature placed relative to the centre. `bearing` is a compass bearing in degrees (0 = north)."""
    kind: str            # lake | river | rock | crater | ridge | wetland
    bearing: float
    dist: float          # metres from the centre
    a: float = 0.0       # lake: shore wiggle | river: width | rock: height | crater: radius | ridge: height
    b: float = 0.0       # river: meander amplitude | rock: radius | crater: depth | ridge: half-width
    c: float = 0.0       # river: axis bearing (flow direction) | ridge: unused


@dataclass(frozen=True)
class Landmark:
    kind: str            # tower | mosque | cathedral | palace | stadium | chimney | mast | dam | terminal | clock | temple
    bearing: float
    dist: float
    name: str = ""


@dataclass(frozen=True)
class Peak:
    """A distant mountain on the horizon: compass bearing, angular width and height (degrees)."""
    bearing: float
    width: float
    height: float
    snow: bool = False


@dataclass(frozen=True)
class City:
    id: str
    name: str
    region: str                       # Central | Eastern | Northern | Western | South-Western | Karamoja
    lat: float
    lon: float
    elevation: int                    # m above sea level (approximate)
    size: str = "town"                # capital | city | town
    blurb: str = ""
    layout: str = "grid"              # grid | radial | organic
    hills: float = 20.0               # rolling-hill amplitude [m]
    hill_freq: float = 2.4            # lower = broader hills
    terrace: float = 0.0              # 0..1: stepped farm terraces on slopes (Kabale)
    cover: str = "banana"             # banana | tea | savanna | cropland | semiarid
    density: float = 1.0              # building density multiplier
    towers: float = 0.0               # 0..1 how many tall buildings the centre gets
    haze: float = 0.2                 # 0 crystal clear .. 1 dusty
    features: tuple = ()
    landmarks: tuple = ()
    peaks: tuple = ()
    seed: int = 1
    generic: bool = False             # True for towns without a hand-tuned profile
    # custom locations typed by the user are flagged so the UI can say so
    custom: bool = False

    @property
    def coords(self) -> str:
        ns = "N" if self.lat >= 0 else "S"
        ew = "E" if self.lon >= 0 else "W"
        return f"{abs(self.lat):.3f}°{ns}  {abs(self.lon):.3f}°{ew}"


F, L, P = Feature, Landmark, Peak

# ------------------------------------------------------------------------------------------------ hand-tuned
_MAIN: list[City] = [
    City("kampala", "Kampala", "Central", 0.3136, 32.5811, 1190, "capital",
         "Uganda's capital, built on seven hills. Dense CBD towers, red-roofed hillsides, matatu taxis and "
         "boda-bodas, with the Nakivubo wetland running through the valley floor.",
         layout="radial", hills=48, hill_freq=2.0, cover="banana", density=1.6, towers=1.0, haze=0.45,
         features=(F("river", 0, 330, 34, 22, 330),),
         landmarks=(L("tower", 20, 120, "Crested Tower"), L("mosque", 300, 520, "Old Kampala hill mosque"),
                    L("cathedral", 255, 560, "Namirembe hill cathedral"), L("stadium", 150, 520, "Stadium")),
         peaks=(P(155, 70, 1.2), P(300, 50, 2.0), P(30, 60, 1.6)), seed=11),
    City("entebbe", "Entebbe", "Central", 0.0512, 32.4637, 1155, "town",
         "A leafy peninsula on Lake Victoria: shoreline, botanical garden forest, colonial low-rise and the "
         "international airport runway.",
         layout="grid", hills=10, hill_freq=2.4, cover="banana", density=0.7, towers=0.1, haze=0.15,
         features=(F("lake", 160, 330, 90), F("lake", 255, 420, 60)),
         landmarks=(L("terminal", 35, 430, "Airport terminal"), L("mast", 320, 380, "Radio mast")),
         peaks=(P(160, 120, 0.2),), seed=23),
    City("jinja", "Jinja", "Eastern", 0.4244, 33.2041, 1135, "city",
         "Where the White Nile leaves Lake Victoria. A wide river, the Nalubaale dam, Asian-quarter shops and "
         "tree-lined colonial streets.",
         layout="grid", hills=14, hill_freq=2.2, cover="banana", density=1.0, towers=0.2, haze=0.2,
         features=(F("lake", 135, 520, 70), F("river", 0, 250, 85, 18, 330)),
         landmarks=(L("dam", 160, 380, "Nalubaale dam"), L("chimney", 285, 470, "Mill chimney")),
         peaks=(P(135, 110, 0.3),), seed=31),
    City("mbarara", "Mbarara", "Western", -0.6072, 30.6545, 1420, "city",
         "Western Uganda's biggest town: rolling green hills, eucalyptus woodlots, banana farms and the "
         "Rwizi river winding through the valley.",
         layout="grid", hills=34, hill_freq=2.3, cover="banana", density=0.9, towers=0.15, haze=0.12,
         features=(F("river", 0, -280, 16, 40, 20),),
         landmarks=(L("clock", 40, 90, "Clock tower"), L("cathedral", 150, 380, "Cathedral")),
         peaks=(P(250, 90, 0.9), P(80, 60, 0.7)), seed=41),
    City("gulu", "Gulu", "Northern", 2.7747, 32.2990, 1100, "city",
         "Northern Uganda's hub: wide grid streets, mango-shaded avenues, savanna all around and granite "
         "inselbergs on the skyline.",
         layout="grid", hills=7, hill_freq=2.6, cover="savanna", density=0.8, towers=0.1, haze=0.3,
         features=(F("rock", 345, 420, 55, 85),),
         landmarks=(L("clock", 10, 80, "Roundabout clock"), L("mosque", 200, 300, "Mosque")),
         peaks=(P(60, 25, 0.9), P(300, 35, 0.7)), seed=53),
    City("lira", "Lira", "Northern", 2.2499, 32.8998, 1080, "city",
         "A flat, open savanna town with a clock-tower roundabout, broad roads, cycling traffic and swamp "
         "wetlands to the east.",
         layout="grid", hills=5, hill_freq=2.8, cover="savanna", density=0.8, towers=0.05, haze=0.3,
         features=(F("river", 0, 380, 40, 30, 20),),
         landmarks=(L("clock", 350, 70, "Clock tower"),), peaks=(P(90, 90, 0.35),), seed=61),
    City("mbale", "Mbale", "Eastern", 1.0821, 34.1750, 1155, "city",
         "Sits at the foot of Mount Elgon. The Wanale ridge rises right behind town and the Namatala river "
         "cuts through lush banana country.",
         layout="grid", hills=26, hill_freq=2.2, cover="banana", density=0.95, towers=0.1, haze=0.15,
         features=(F("ridge", 95, 620, 150, 220), F("river", 0, 300, 24, 25, 330)),
         landmarks=(L("mosque", 40, 260, "Mosque"), L("clock", 350, 70, "Clock tower")),
         peaks=(P(85, 55, 11.0), P(70, 30, 8.0)), seed=71),
    City("fortportal", "Fort Portal", "Western", 0.6710, 30.2750, 1537, "city",
         "Tea estates on rolling hills and crater lakes, with the snow-capped Rwenzori 'Mountains of the "
         "Moon' on the horizon.",
         layout="organic", hills=40, hill_freq=2.1, cover="tea", density=0.7, towers=0.05, haze=0.1,
         features=(F("crater", 20, 420, 95, 11), F("crater", 285, 560, 70, 9)),
         landmarks=(L("palace", 150, 330, "Palace on the hill"),),
         peaks=(P(215, 55, 6.5, True), P(240, 30, 4.5), P(180, 40, 2.2)), seed=83),
    City("kabale", "Kabale", "South-Western", -1.2490, 29.9899, 1869, "town",
         "'Switzerland of Africa': steep terraced hills stitched with farm plots, cool and misty, with the "
         "Virunga volcanoes beyond.",
         layout="organic", hills=85, hill_freq=2.3, terrace=0.8, cover="cropland", density=0.8, towers=0.05, haze=0.2,
         features=(F("crater", 225, 600, 70, 6),),
         landmarks=(L("cathedral", 20, 300, "Hill-top church"),),
         peaks=(P(205, 40, 5.0), P(190, 25, 4.0), P(240, 60, 1.8)), seed=97),
    City("masaka", "Masaka", "Central", -0.3136, 31.7413, 1150, "city",
         "A hilly trading town on the Kampala-Masaka highway: banana farms, red-brick streets and wetlands "
         "between the ridges.",
         layout="organic", hills=32, hill_freq=2.2, cover="banana", density=0.8, towers=0.05, haze=0.2,
         features=(F("river", 0, 300, 28, 30, 25),), landmarks=(L("cathedral", 330, 330, "Cathedral hill"),),
         peaks=(P(140, 70, 0.6),), seed=103),
    City("arua", "Arua", "Northern", 3.0200, 30.9110, 1200, "city",
         "West Nile's capital: low rolling savanna, mango avenues and a busy market square.",
         layout="grid", hills=12, hill_freq=2.5, cover="savanna", density=0.75, towers=0.05, haze=0.35,
         features=(F("river", 0, -360, 22, 25, 340),), landmarks=(L("mosque", 100, 280, "Mosque"),),
         peaks=(P(280, 80, 0.7),), seed=113),
    City("soroti", "Soroti", "Eastern", 1.7146, 33.6111, 1130, "city",
         "Flat plains punctuated by the enormous Soroti Rock, a granite inselberg rising straight out of the "
         "town, with papyrus swamps nearby.",
         layout="grid", hills=5, hill_freq=2.8, cover="savanna", density=0.8, towers=0.05, haze=0.3,
         features=(F("rock", 20, 300, 72, 105), F("river", 0, -380, 36, 30, 30)),
         landmarks=(L("clock", 190, 70, "Clock tower"),), peaks=(P(40, 80, 0.4),), seed=127),
    City("hoima", "Hoima", "Western", 1.4356, 31.3436, 1560, "town",
         "Plateau town above the Rift Valley, with eucalyptus ridges and the blue wall of the Albertine "
         "escarpment to the west.",
         layout="grid", hills=30, hill_freq=2.2, cover="banana", density=0.7, towers=0.05, haze=0.25,
         features=(F("river", 0, 330, 14, 30, 15),), landmarks=(L("cathedral", 160, 320, "Cathedral"),),
         peaks=(P(270, 100, 1.6), P(255, 60, 2.4)), seed=131),
    City("kasese", "Kasese", "Western", 0.1833, 30.0833, 960, "town",
         "A hot plain below the Rwenzori Mountains: acacia savanna, a stony river and glaciers glinting on "
         "the northern skyline.",
         layout="grid", hills=6, hill_freq=2.8, cover="savanna", density=0.7, towers=0.05, haze=0.22,
         features=(F("river", 0, 340, 30, 18, 20), F("lake", 110, 560, 60)),
         landmarks=(L("mast", 190, 400, "Mast"),),
         peaks=(P(335, 60, 9.0, True), P(5, 40, 6.0, True), P(300, 50, 3.5)), seed=139),
    City("tororo", "Tororo", "Eastern", 0.6928, 34.1810, 1170, "town",
         "Dominated by Tororo Rock, a volcanic plug 150 m high on the edge of town, next to the cement "
         "works chimney.",
         layout="grid", hills=12, hill_freq=2.5, cover="cropland", density=0.8, towers=0.05, haze=0.25,
         features=(F("rock", 330, 330, 120, 165),), landmarks=(L("chimney", 215, 360, "Cement works"),),
         peaks=(P(30, 50, 4.0), P(60, 40, 2.5)), seed=149),
    City("moroto", "Moroto", "Karamoja", 2.5345, 34.6666, 1400, "town",
         "Semi-arid Karamoja below Mount Moroto: red earth, acacia scrub, dry sandy riverbeds and "
         "round-hut homesteads.",
         layout="organic", hills=20, hill_freq=2.4, cover="semiarid", density=0.45, towers=0.0, haze=0.4,
         features=(F("rock", 80, 300, 55, 90), F("river", 0, 260, 38, 25, 40), F("ridge", 85, 640, 120, 160)),
         landmarks=(), peaks=(P(85, 60, 8.0), P(60, 30, 4.0)), seed=157),
]

# ------------------------------------------------------------------------------------------------ generic towns
# (id, name, region, lat, lon, elevation m, flavour) - flavour picks sensible landscape defaults.
_TOWNS = [
    ("mukono", "Mukono", "Central", 0.3533, 32.7553, 1190, "banana"),
    ("wakiso", "Wakiso", "Central", 0.4044, 32.4594, 1200, "banana"),
    ("luwero", "Luwero", "Central", 0.8500, 32.4667, 1100, "savanna"),
    ("mityana", "Mityana", "Central", 0.4167, 32.0500, 1250, "banana"),
    ("mubende", "Mubende", "Central", 0.5833, 31.3667, 1400, "banana"),
    ("rakai", "Rakai", "Central", -0.7167, 31.4000, 1250, "banana"),
    ("iganga", "Iganga", "Eastern", 0.6092, 33.4686, 1150, "cropland"),
    ("kamuli", "Kamuli", "Eastern", 0.9472, 33.1197, 1100, "cropland"),
    ("bugiri", "Bugiri", "Eastern", 0.5714, 33.7500, 1150, "cropland"),
    ("busia", "Busia", "Eastern", 0.4608, 34.0911, 1170, "cropland"),
    ("pallisa", "Pallisa", "Eastern", 1.1450, 33.7089, 1100, "cropland"),
    ("kumi", "Kumi", "Eastern", 1.4617, 33.9361, 1100, "savanna"),
    ("kapchorwa", "Kapchorwa", "Eastern", 1.4000, 34.4500, 1900, "cropland"),
    ("masindi", "Masindi", "Western", 1.6744, 31.7150, 1150, "savanna"),
    ("kyenjojo", "Kyenjojo", "Western", 0.6264, 30.6439, 1300, "tea"),
    ("ibanda", "Ibanda", "Western", -0.1319, 30.5000, 1400, "banana"),
    ("bushenyi", "Bushenyi", "Western", -0.5410, 30.1850, 1500, "banana"),
    ("ntungamo", "Ntungamo", "South-Western", -0.8794, 30.2654, 1450, "banana"),
    ("rukungiri", "Rukungiri", "South-Western", -0.8417, 29.9417, 1300, "banana"),
    ("kisoro", "Kisoro", "South-Western", -1.2853, 29.6853, 1860, "cropland"),
    ("kitgum", "Kitgum", "Northern", 3.2889, 32.8873, 940, "savanna"),
    ("apac", "Apac", "Northern", 1.9756, 32.5386, 1100, "savanna"),
    ("nebbi", "Nebbi", "Northern", 2.4783, 31.0889, 650, "savanna"),
    ("moyo", "Moyo", "Northern", 3.6526, 31.7240, 700, "savanna"),
    ("adjumani", "Adjumani", "Northern", 3.3769, 31.7889, 1000, "savanna"),
    ("koboko", "Koboko", "Northern", 3.4108, 30.9594, 1100, "savanna"),
    ("yumbe", "Yumbe", "Northern", 3.4647, 31.2467, 1000, "savanna"),
    ("kotido", "Kotido", "Karamoja", 2.9806, 34.1331, 1200, "semiarid"),
    ("kaabong", "Kaabong", "Karamoja", 3.5167, 34.1500, 1200, "semiarid"),
]
_REGION_STYLE = {   # region -> (hills, hill_freq, haze)
    "Central": (24, 2.2, 0.25), "Eastern": (14, 2.5, 0.22), "Northern": (9, 2.7, 0.32),
    "Western": (28, 2.3, 0.15), "South-Western": (60, 2.3, 0.15), "Karamoja": (18, 2.5, 0.4),
}


def _generic(i: int, row: tuple) -> City:
    cid, name, region, lat, lon, elev, cover = row
    hills, freq, haze = _REGION_STYLE[region]
    seed = 200 + i * 7
    # deterministic landscape variety so neighbouring towns don't look alike
    r = (seed * 2654435761 % 1000) / 1000.0
    feats = []
    if r < 0.55:
        feats.append(F("river", 0, (-1) ** i * (260 + 120 * r), 14 + 20 * r, 25, 10 + 70 * r * 4 % 90))
    if r > 0.7 and cover != "semiarid":
        feats.append(F("rock", (seed * 13) % 360, 330 + 150 * (r - 0.7), 35 + 60 * r, 60 + 60 * r))
    lm = (L("mosque", (seed * 7) % 360, 300, "Mosque"), L("cathedral", (seed * 11) % 360, 360, "Church"))
    return City(cid, name, region, lat, lon, elev, "town",
                f"{name}, {region} Uganda - a typical regional town. The street grid and landscape are "
                f"generated to match the region; switch on live map data for the real streets.",
                layout="grid" if region in ("Northern", "Eastern") else "organic",
                hills=hills, hill_freq=freq, cover=cover, density=0.65, towers=0.0, haze=haze,
                features=tuple(feats), landmarks=lm,
                peaks=(P((seed * 17) % 360, 60, 0.5 + 0.8 * r),), seed=seed, generic=True)


CITIES: list[City] = _MAIN + [_generic(i, row) for i, row in enumerate(_TOWNS)]
BY_ID = {c.id: c for c in CITIES}


def find(city_id: str) -> Optional[City]:
    return BY_ID.get(city_id)


def search(query: str) -> list[City]:
    """Case-insensitive substring search over name / region (used by the city picker's type-to-filter)."""
    q = query.strip().lower()
    if not q:
        return list(CITIES)
    return [c for c in CITIES if q in c.name.lower() or q in c.region.lower() or q in c.id]


def custom_city(lat: float, lon: float, name: str = "") -> City:
    """Any point in Uganda (roughly lat -1.5..4.3, lon 29.5..35.1): generic profile seeded from the coordinates."""
    seed = int(abs(lat * 1000) * 31 + abs(lon * 1000) * 17) % 100000
    region = ("Karamoja" if lat > 1.9 and lon > 34.0 else "Northern" if lat > 2.0 else
              "South-Western" if lat < -0.6 and lon < 30.4 else "Western" if lon < 31.2 else
              "Eastern" if lon > 33.2 else "Central")
    hills, freq, haze = _REGION_STYLE[region]
    elev = int(1100 + 400 * max(0.0, (30.5 - lon)) / 1.5 + 5) if lon < 30.5 else 1100
    cover = {"Central": "banana", "Eastern": "cropland", "Northern": "savanna", "Western": "banana",
             "South-Western": "cropland", "Karamoja": "semiarid"}[region]
    nm = name or f"{abs(lat):.3f}{'N' if lat >= 0 else 'S'} {abs(lon):.3f}E"
    return City("custom", nm, region, lat, lon, elev, "town",
                f"Custom location ({region} Uganda). Streets are generated unless live map data is available.",
                layout="grid", hills=hills, hill_freq=freq, cover=cover, density=0.7, haze=haze,
                features=(F("river", 0, 300, 20, 28, 15),), landmarks=(L("mosque", seed % 360, 300, "Mosque"),),
                peaks=(P(seed % 360, 70, 0.9),), seed=seed, generic=True, custom=True)


def in_uganda(lat: float, lon: float) -> bool:
    """Loose bounding box of Uganda - enough to catch typos, not a border test."""
    return -1.6 <= lat <= 4.3 and 29.5 <= lon <= 35.1


def metres_per_degree(lat: float) -> tuple[float, float]:
    """(metres per degree latitude, metres per degree longitude) at this latitude."""
    return 111_320.0, 111_320.0 * math.cos(math.radians(lat))
