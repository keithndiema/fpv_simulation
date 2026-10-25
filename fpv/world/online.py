"""Optional live map data: real streets + buildings (OpenStreetMap via the Overpass API) and real terrain
elevation (Open-Meteo), converted into the same structures the procedural generator produces.

HONESTY NOTE: the parsing / projection / rasterising code in this file is unit-tested against synthetic fixtures
(tests/selftest.py), but it was written in a sandbox with no route to the public servers, so the actual downloads
have NOT been exercised against the live services.  Everything fails soft: any network or format problem returns
None and the caller falls back to the procedural town.

Coordinates: the world is a 1200 m square (x north, y west) centred on world point (40, 40) = the city's lat/lon.
Downloads are cached as JSON in ~/.fpv_sim/cache so a town is only fetched once.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..game.terrain import CELL, N, ORIGIN
from .cities import City, metres_per_degree
from .model import (KIND_HOUSE, KIND_INDUSTRY, KIND_MIDRISE, KIND_SHOP, KIND_TOWER, ROOF_FLAT, ROOF_GABLE,
                    Buildings, RoadLine)
from .noise import polygon_mask

CACHE = Path.home() / ".fpv_sim" / "cache"
OVERPASS = "https://overpass-api.de/api/interpreter"
ELEVATION = "https://api.open-meteo.com/v1/elevation"
CENTRE = (40.0, 40.0)                      # world coordinates of the city's lat/lon
HALF = (N - 1) * CELL / 2.0                # 640 m: half the side of the world square
USER_AGENT = "fpv-sim/1.0 (educational flight simulator)"
GRID = 33                                  # elevation samples per side (33^2 = 1089 points = 11 requests)

HIGHWAY_KIND = {"motorway": "primary", "trunk": "primary", "primary": "primary", "secondary": "secondary",
                "tertiary": "secondary", "unclassified": "local", "residential": "local", "service": "local",
                "living_street": "local", "road": "local", "track": "track", "path": "track", "footway": "track",
                "cycleway": "track", "pedestrian": "local"}


@dataclass
class RealData:
    heights: np.ndarray | None = None          # (N, N) float32 terrain heights, water at <= 0
    roads: list | None = None                  # list[RoadLine]
    buildings: Buildings | None = None
    water_distance: np.ndarray | None = None
    note: str = ""


# ---------------------------------------------------------------------------------------- projection
def to_world(lat, lon, city: City) -> np.ndarray:
    """(lat, lon) -> world (x north, y west) metres.  Arrays in, (n, 2) array out."""
    mlat, mlon = metres_per_degree(city.lat)
    lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
    x = CENTRE[0] + (lat - city.lat) * mlat
    y = CENTRE[1] - (lon - city.lon) * mlon          # east is -y
    return np.stack([x, y], axis=-1)


def bbox(city: City) -> tuple[float, float, float, float]:
    """(south, west, north, east) in degrees covering the world square."""
    mlat, mlon = metres_per_degree(city.lat)
    return (city.lat - HALF / mlat, city.lon - HALF / mlon, city.lat + HALF / mlat, city.lon + HALF / mlon)


# ---------------------------------------------------------------------------------------- OSM parsing
def overpass_query(city: City) -> str:
    s, w, n, e = bbox(city)
    box = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    return (f"[out:json][timeout:40];(way[highway]({box});way[building]({box});way[natural=water]({box});"
            f"way[waterway=riverbank]({box}););out geom;")


def _oriented_rect(poly: np.ndarray):
    """Footprint polygon -> (cx, cy, half_u, half_v, yaw) of its principal-axis bounding rectangle."""
    c = poly.mean(axis=0)
    q = poly - c
    if len(q) < 3:
        return None
    w, v = np.linalg.eigh(q.T @ q)
    ax = v[:, 1]                                       # long axis
    yaw = math.atan2(ax[1], ax[0])
    cs, sn = math.cos(yaw), math.sin(yaw)
    u = q[:, 0] * cs + q[:, 1] * sn
    t = -q[:, 0] * sn + q[:, 1] * cs
    cu, ct = (u.max() + u.min()) / 2, (t.max() + t.min()) / 2
    cx = c[0] + cu * cs - ct * sn
    cy = c[1] + cu * sn + ct * cs
    return cx, cy, max(1.2, (u.max() - u.min()) / 2), max(1.2, (t.max() - t.min()) / 2), yaw


def _height_of(tags: dict) -> tuple[float, int]:
    """Wall height [m] and storeys from OSM tags (falls back to a one-storey house)."""
    try:
        if "height" in tags:
            h = float(str(tags["height"]).split()[0])
            return max(2.5, h), max(1, int(round(h / 3.2)))
        if "building:levels" in tags:
            lv = max(1, int(float(tags["building:levels"])))
            return 3.2 * lv + 0.8, lv
    except ValueError:
        pass
    return 4.0, 1


def parse_overpass(data: dict, city: City, rng: np.random.Generator | None = None):
    """Overpass JSON ('out geom') -> (roads, buildings, water_polygons)."""
    rng = rng or np.random.default_rng(city.seed)
    roads: list[RoadLine] = []
    rows = []
    water: list[np.ndarray] = []
    for el in data.get("elements", []):
        geom, tags = el.get("geometry"), el.get("tags", {})
        if el.get("type") != "way" or not geom or len(geom) < 2:
            continue
        pts = to_world([g["lat"] for g in geom], [g["lon"] for g in geom], city)
        if "highway" in tags:
            kind = HIGHWAY_KIND.get(tags["highway"])
            if kind is None:
                continue
            if kind == "local" and tags.get("surface") in ("unpaved", "ground", "dirt", "earth", "gravel", "compacted"):
                kind = "murram"
            roads.append(RoadLine(pts, kind))
        elif "building" in tags and len(pts) >= 4:
            rect = _oriented_rect(pts[:-1] if np.allclose(pts[0], pts[-1]) else pts)
            if rect is None:
                continue
            cx, cy, hu, hv, yaw = rect
            if not (abs(cx - CENTRE[0]) < HALF - 5 and abs(cy - CENTRE[1]) < HALF - 5):
                continue
            wall, floors = _height_of(tags)
            kind = (KIND_TOWER if floors >= 9 else KIND_MIDRISE if floors >= 4 else
                    KIND_INDUSTRY if tags.get("building") in ("industrial", "warehouse", "factory") else
                    KIND_SHOP if tags.get("building") in ("retail", "commercial", "shop", "supermarket") else KIND_HOUSE)
            gable = kind == KIND_HOUSE and hu >= hv
            wall_c = np.clip(np.array([196, 186, 168]) + rng.normal(0, 14, 3), 0, 255)
            roof_c = np.clip(np.array([150, 70, 52] if gable else [120, 120, 124]) + rng.normal(0, 12, 3), 0, 255)
            rows.append((cx, cy, hu, hv, yaw, wall, floors, kind, ROOF_GABLE if gable else ROOF_FLAT,
                         2.4 if gable else 0.0, wall_c, roof_c))
        elif tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
            if len(pts) >= 4:
                water.append(pts)
    if rows:
        b = Buildings(pos=np.array([[r[0], r[1], 0.0] for r in rows]), half=np.array([[r[2], r[3]] for r in rows]),
                      wall=np.array([r[5] for r in rows], float), yaw=np.array([r[4] for r in rows]),
                      roof=np.array([r[8] for r in rows], np.int8), rise=np.array([r[9] for r in rows]),
                      floors=np.array([r[6] for r in rows], np.int16), kind=np.array([r[7] for r in rows], np.int8),
                      wall_rgb=np.array([r[10] for r in rows]), roof_rgb=np.array([r[11] for r in rows]))
    else:
        b = Buildings.empty()
    return roads, b, water


# ---------------------------------------------------------------------------------------- elevation
def elevation_points(city: City) -> tuple[np.ndarray, np.ndarray]:
    """GRID x GRID lat/lon sample lattice over the world square (row = world x/north, col = world y/west)."""
    s, w, n, e = bbox(city)
    lats = np.linspace(s, n, GRID)
    lons = np.linspace(e, w, GRID)                       # col 0 = east edge ... because y grows towards the west
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return LA, LO


def heights_from_samples(elev: np.ndarray, water: list[np.ndarray]) -> np.ndarray:
    """(GRID, GRID) elevations [m above sea level] -> (N, N) heights.

    Shifted so the lowest ground sits at ~4 m (the world's water level is 0), smoothly upsampled, and the OSM
    water polygons are carved down to -2.5 m so lakes and wide rivers show up as water.
    """
    e = np.asarray(elev, float)
    h = _upsample(e - e.min() + 4.0, N)
    if water:
        xs = ORIGIN[0] + np.arange(N) * CELL
        X, Y = np.meshgrid(xs, xs, indexing="ij")           # h[i, j]: i -> world x, j -> world y
        wet = np.zeros(h.shape, bool)
        for poly in water:
            wet |= polygon_mask(X, Y, poly)
        h = np.where(wet, -2.5, h)
    return h.astype(np.float32)


def _upsample(a: np.ndarray, n: int) -> np.ndarray:
    """Bilinear upsample of a square array to (n, n) using numpy only."""
    g = a.shape[0]
    t = np.linspace(0, g - 1, n)
    i0 = np.clip(np.floor(t).astype(int), 0, g - 2)
    f = t - i0
    rows = a[i0] * (1 - f)[:, None] + a[i0 + 1] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]


# ---------------------------------------------------------------------------------------- network + cache
def _cache_path(city: City, what: str) -> Path:
    key = hashlib.sha1(f"{city.lat:.4f},{city.lon:.4f}".encode()).hexdigest()[:10]
    return CACHE / f"{city.id}_{key}_{what}.json"


def _http_json(url: str, data: bytes | None = None, timeout: float = 45.0):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _cached(city: City, what: str, fetch):
    path = _cache_path(city, what)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    data = fetch()
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except OSError:
        pass
    return data


def fetch_osm(city: City) -> dict:
    body = urllib.parse.urlencode({"data": overpass_query(city)}).encode()
    return _http_json(OVERPASS, body, timeout=60.0)


def fetch_elevation(city: City) -> list[float]:
    """GRID*GRID elevations, row-major (rows = south->north, cols = east->west), 100 points per request."""
    LA, LO = elevation_points(city)
    la, lo = LA.ravel(), LO.ravel()
    out: list[float] = []
    for i in range(0, len(la), 100):
        q = urllib.parse.urlencode({"latitude": ",".join(f"{v:.5f}" for v in la[i:i + 100]),
                                    "longitude": ",".join(f"{v:.5f}" for v in lo[i:i + 100])})
        out += _http_json(f"{ELEVATION}?{q}")["elevation"]
        time.sleep(0.15)                                  # be polite to the free service
    return out


def build_real(osm: dict, elev: list[float] | None, city: City) -> RealData:
    """Pure function (no network): downloaded JSON -> RealData.  This is what the unit tests exercise."""
    roads, bld, water = parse_overpass(osm, city)
    heights = None
    if elev is not None and len(elev) == GRID * GRID:
        heights = heights_from_samples(np.array(elev, float).reshape(GRID, GRID), water)
    if len(bld):
        # buildings sit on the ground at their lowest corner; without elevation data the ground is flat
        if heights is not None:
            from .noise import bilinear
            bld.pos[:, 2] = bilinear(heights, bld.pos[:, 0], bld.pos[:, 1], ORIGIN[0], ORIGIN[1], CELL)
        else:
            bld.pos[:, 2] = 8.0
    return RealData(heights=heights, roads=roads or None, buildings=bld if len(bld) else None,
                    note=f"{len(roads)} roads, {len(bld)} buildings, {len(water)} water bodies")


def fetch_real(city: City, progress=None) -> RealData | None:
    """Download (or load from cache) OSM + elevation for `city`.  Returns None on any failure."""
    say = progress or (lambda m: None)
    try:
        say("Downloading streets and buildings")
        osm = _cached(city, "osm", lambda: fetch_osm(city))
        try:
            say("Downloading elevation")
            elev = _cached(city, "elev", lambda: fetch_elevation(city))
        except Exception:
            elev = None                                    # streets alone are still worth having
        rd = build_real(osm, elev, city)
        if rd.roads is None:
            return None
        return rd
    except Exception:
        return None
