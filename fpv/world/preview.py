"""Top-down rendering of a World to an RGB array: used for the city picker preview, the minimap and debugging."""
from __future__ import annotations

import numpy as np

from ..game.terrain import CELL, N, ORIGIN
from .model import ROAD_CLASSES, World
from .noise import bilinear

_COVER = {  # ground-cover code -> colour
    0: (122, 150, 82), 1: (110, 156, 70), 2: (176, 160, 96), 3: (170, 152, 126), 4: (170, 98, 66), 5: (84, 138, 112),
    6: (70, 140, 60), 7: (130, 124, 116), 8: (206, 190, 140), 9: (96, 150, 72), 10: (62, 120, 52),
}
_ROAD = {"primary": (84, 86, 92), "secondary": (96, 98, 104), "local": (120, 118, 116), "murram": (176, 108, 78),
         "track": (170, 140, 110), "runway": (70, 72, 78)}


def topdown(world: World, size: int = 256, north_up: bool = True) -> np.ndarray:
    """RGB uint8 (size, size, 3). North is up and east is right when `north_up` (the usual map convention)."""
    T = world.terrain
    n = size
    xs = ORIGIN[0] + (np.arange(n) + 0.5) * (N - 1) * CELL / n
    XX, YY = np.meshgrid(xs, xs)                       # XX = north coordinate, YY = west coordinate
    H = bilinear(T.heights, XX, YY, ORIGIN[0], ORIGIN[1], CELL)
    gy, gx = np.gradient(H, (N - 1) * CELL / n)
    light = np.clip(0.75 + 1.6 * (-gx * 0.45 - gy * -0.35), 0.45, 1.25)       # sun from the north-west quadrant
    img = np.zeros((n, n, 3))
    if T.landcover is not None:
        lc = T.landcover[np.clip(((YY - ORIGIN[1]) / CELL).astype(int), 0, N - 1), np.clip(((XX - ORIGIN[0]) / CELL).astype(int), 0, N - 1)]
        for code, col in _COVER.items():
            img[lc == code] = col
    else:
        img[:] = (110, 150, 84)
    img = img * light[..., None]
    water = H < 0.0
    img[water] = np.array((56, 120, 170)) * (0.9 + 0.1 * light[water][:, None])
    scale = n / ((N - 1) * CELL)

    def px(p):                                         # world (x north, y west) -> raster (row, col)
        return np.clip(((p[:, 1] - ORIGIN[1]) * scale).astype(int), 0, n - 1), np.clip(((p[:, 0] - ORIGIN[0]) * scale).astype(int), 0, n - 1)

    for kind in ("track", "murram", "local", "secondary", "primary", "runway"):
        w = max(1, int(round(ROAD_CLASSES[kind][0] * scale * 1.2)))
        for ln in world.roads:
            if ln.kind != kind:
                continue
            P = ln.pts
            seg = np.concatenate([P[:-1][:, None, :] + (P[1:] - P[:-1])[:, None, :] * t for t in np.linspace(0, 1, 4)[None, :, None]], axis=1).reshape(-1, 2) if len(P) > 1 else P
            r, c = px(seg)
            for dr in range(-(w // 2), w - w // 2):
                for dc in range(-(w // 2), w - w // 2):
                    img[np.clip(r + dr, 0, n - 1), np.clip(c + dc, 0, n - 1)] = _ROAD[kind]
    b = world.buildings
    if len(b):
        r, c = px(b.pos[:, :2])
        tall = np.clip(b.wall / 40.0, 0, 1)
        col = b.roof_rgb * (1 - tall[:, None]) + np.array([200, 205, 215]) * tall[:, None]
        for dr in (0, 1):
            for dc in (0, 1):
                img[np.clip(r + dr * (b.wall > 6), 0, n - 1), np.clip(c + dc, 0, n - 1)] = col * 0.9
    T_ = T.trees
    if len(T_.height):
        r, c = px(T_.pos[:, :2])
        img[r, c] = img[r, c] * 0.55 + np.array([30, 80, 34]) * 0.45
    px_pad = px(np.array([world.pad]))
    img[max(0, px_pad[0][0] - 2):px_pad[0][0] + 3, max(0, px_pad[1][0] - 2):px_pad[1][0] + 3] = (255, 220, 40)
    out = np.clip(img, 0, 255).astype(np.uint8)
    # raster axes: row = world y (west), col = world x (north). For a north-up map: north = up, west = left.
    out = out[::-1, :][:, ::-1] if False else out
    out = np.rot90(out, 1)[:, ::-1] if False else out
    # we want  up = +x (north)  and  left = +y (west):  image[row, col] with row = -x, col = -y
    return np.ascontiguousarray(out.transpose(1, 0, 2)[::-1, ::-1])
