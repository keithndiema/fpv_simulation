"""Plain data containers for a generated (or downloaded) place. No rendering and no pygame in here."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# road classes -> (width m, paved?, label)
ROAD_CLASSES = {
    "primary": (13.0, True),
    "secondary": (9.0, True),
    "local": (6.0, True),
    "murram": (5.5, False),        # red laterite dirt road
    "track": (3.4, False),         # footpath / farm track
    "runway": (46.0, True),
}
# roof / building style codes (arrays below store these as small ints)
ROOF_FLAT, ROOF_GABLE, ROOF_HUT = 0, 1, 2
KIND_HOUSE, KIND_SHOP, KIND_MIDRISE, KIND_TOWER, KIND_HUT, KIND_INDUSTRY = 0, 1, 2, 3, 4, 5


@dataclass
class RoadLine:
    pts: np.ndarray                 # (n, 2) world x, y
    kind: str = "local"
    bridge: np.ndarray | None = None  # (n,) bool: point is on a bridge deck
    z: np.ndarray | None = None       # (n,) deck height for bridge points (ground height otherwise)

    @property
    def width(self) -> float:
        return ROAD_CLASSES[self.kind][0]


@dataclass
class Buildings:
    """Struct-of-arrays so the renderer can cull / project thousands of buildings with a few numpy calls."""
    pos: np.ndarray                 # (n, 3) footprint centre x, y and the ground level at the *lowest* corner
    half: np.ndarray                # (n, 2) half extents along the building's own axes (x, y)
    wall: np.ndarray                # (n,) wall height above `pos[:, 2]` (the eave line for pitched roofs)
    yaw: np.ndarray                 # (n,)
    roof: np.ndarray                # (n,) ROOF_* code
    rise: np.ndarray                # (n,) ridge height above the eaves (0 for flat roofs)
    floors: np.ndarray              # (n,) storeys (drives window rows)
    kind: np.ndarray                # (n,) KIND_* code
    wall_rgb: np.ndarray            # (n, 3) float 0..255
    roof_rgb: np.ndarray            # (n, 3)

    def __len__(self) -> int:
        return len(self.wall)

    @staticmethod
    def empty() -> "Buildings":
        z2, z3, z1 = np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0)
        return Buildings(z3.copy(), z2, z1.copy(), z1.copy(), z1.astype(np.int8), z1.copy(), z1.astype(np.int16),
                         z1.astype(np.int8), z3.copy(), z3.copy())

    def select(self, mask) -> "Buildings":
        return Buildings(*(getattr(self, f)[mask] for f in
                           ("pos", "half", "wall", "yaw", "roof", "rise", "floors", "kind", "wall_rgb", "roof_rgb")))

    @staticmethod
    def concat(parts: list["Buildings"]) -> "Buildings":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Buildings.empty()
        return Buildings(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                           ("pos", "half", "wall", "yaw", "roof", "rise", "floors", "kind", "wall_rgb", "roof_rgb")))

    def top(self) -> np.ndarray:
        """Highest point of each building (ridge for pitched roofs)."""
        return self.pos[:, 2] + self.wall + self.rise

    def collision_boxes(self, centre=None, radius: float = 1e9) -> list[tuple]:
        """(cx, cy, cz, hx, hy, hz, yaw) boxes for the C++ core. Pitched roofs are boxed to mid-rise (the drone
        clips the eaves rather than hovering inside the ridge)."""
        sel = np.ones(len(self), bool)
        if centre is not None:
            sel = np.hypot(self.pos[:, 0] - centre[0], self.pos[:, 1] - centre[1]) < radius
        out = []
        for i in np.nonzero(sel)[0]:
            h = float(self.wall[i] + 0.6 * self.rise[i]) + 1.0          # +1 m of foundation below the lowest corner
            out.append((float(self.pos[i, 0]), float(self.pos[i, 1]), float(self.pos[i, 2]) - 1.0 + h / 2,
                        float(self.half[i, 0]), float(self.half[i, 1]), h / 2, float(self.yaw[i])))
        return out


@dataclass
class Mesh:
    """A small convex-ish solid (landmarks): world-space vertices and coloured faces."""
    verts: np.ndarray               # (v, 3)
    faces: list                     # list of tuples of vertex indices
    colors: np.ndarray              # (f, 3) base colour per face
    normals: np.ndarray             # (f, 3) outward unit normals
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 10.0
    name: str = ""
    emissive: np.ndarray | None = None   # (f,) bool: face doesn't get sun shading (lit windows, signs)


@dataclass
class Vehicle:
    route: int
    s: float                         # distance along the route [m]
    speed: float                     # m/s (negative = driving the other way)
    kind: str                        # car | matatu | boda | truck | bus
    color: tuple
    lane: float                      # lateral offset from the road centreline [m]


@dataclass
class World:
    city: object                     # cities.City
    terrain: object = None           # game.terrain.Terrain (attached by the generator)
    preview: object = None           # cached top-down RGB image (world.loader.preview_image)
    roads: list = field(default_factory=list)
    buildings: Buildings = field(default_factory=Buildings.empty)
    meshes: list = field(default_factory=list)
    routes: list = field(default_factory=list)       # list[(pts (n,2), cum_len (n,))] drivable polylines
    vehicles: list = field(default_factory=list)
    pad: tuple = (0.0, 0.0)
    pad_radius: float = 9.0
    park: tuple | None = None        # (cx, cy, half_x, half_y, yaw) lawn the launch pad sits in
    centre: tuple = (40.0, 40.0)     # town centre in world metres
    occ_h: np.ndarray | None = None  # (m, m) rooftop height above the ground (0 = open air), 2.5 m cells
    occ_cell: float = 2.5
    occ_origin: tuple = (-600.0, -600.0)
    extra_colliders: list = field(default_factory=list)   # invisible boxes (landmark footprints, bridge decks)
    lattice: dict | None = None      # street lattice (for building race circuits)
    source: str = "procedural"       # procedural | openstreetmap
    notes: str = ""

    # ---- spatial queries --------------------------------------------------------------------------
    def _occ_index(self, x, y):
        n = self.occ_h.shape[0]
        i = np.clip(((np.asarray(x, float) - self.occ_origin[0]) / self.occ_cell).astype(int), 0, n - 1)
        j = np.clip(((np.asarray(y, float) - self.occ_origin[1]) / self.occ_cell).astype(int), 0, n - 1)
        return j, i

    def roof_height(self, x, y):
        """Height of the tallest building above the ground at (x, y); 0 in open air. Scalars or arrays."""
        if self.occ_h is None:
            return 0.0 if np.ndim(x) == 0 else np.zeros(np.shape(x))
        j, i = self._occ_index(x, y)
        out = self.occ_h[j, i]
        return float(out) if np.ndim(out) == 0 else out

    def clearance(self, x, y):
        """Lowest safe altitude (m, absolute) over (x, y): the higher of the terrain and any rooftop."""
        return self.terrain.ground(x, y) + self.roof_height(x, y)

    def blocked(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True if (x, y) is inside (or within `margin` of) a building footprint."""
        if self.occ_h is None:
            return False
        r = int(np.ceil(margin / self.occ_cell))
        j, i = self._occ_index(x, y)
        j, i = int(j), int(i)
        n = self.occ_h.shape[0]
        return bool((self.occ_h[max(0, j - r):min(n, j + r + 1), max(0, i - r):min(n, i + r + 1)] > 0).any())
