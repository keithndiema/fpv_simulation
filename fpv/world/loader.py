"""World cache + background generation.

Why a loader at all?  Generating a town takes 1-3 s of numpy work.  Doing that on the UI thread would freeze the
window, so the city picker asks for worlds through this class: `request()` starts a worker thread, `poll()` is
called every frame and returns the finished world (or None while it is still building).  A small LRU cache keeps
the last few cities so flipping back and forth in the picker is instant.

numpy releases the GIL for most of its heavy work, so the menu stays responsive while a town builds.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from .cities import City
from .generator import build_world
from . import online
from .model import World
from .preview import topdown

CACHE_SIZE = 4


def preview_image(world: World, size: int = 256) -> np.ndarray:
    """Top-down map of the world, computed once and stored on the world (the picker and minimap share it)."""
    img = getattr(world, "preview", None)
    if img is None or img.shape[0] != size:
        img = topdown(world, size)
        world.preview = img
    return img


class WorldLoader:
    def __init__(self, cache_size: int = CACHE_SIZE):
        self._cache: "OrderedDict[str, World]" = OrderedDict()
        self._size = cache_size
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._job: str | None = None          # id of the city currently being generated
        self._want: City | None = None        # latest request that arrived while a job was running
        self._error: dict[str, str] = {}
        self.progress = ""
        self.live = False                     # True: try real OpenStreetMap data first (see world.online)

    # -- cache --------------------------------------------------------------------------------------------
    def key(self, city: City) -> str:
        """Custom locations share the id 'custom', so their coordinates are part of the key; live and procedural
        versions of the same town are cached separately."""
        k = f"custom:{city.lat:.4f},{city.lon:.4f}" if city.custom else city.id
        return k + ":live" if self.live else k

    def cached(self, city: City) -> World | None:
        with self._lock:
            w = self._cache.get(self.key(city))
            if w is not None:
                self._cache.move_to_end(self.key(city))
            return w

    def _store(self, key: str, world: World) -> None:
        with self._lock:
            self._cache[key] = world
            self._cache.move_to_end(key)
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)

    def _make(self, city: City, progress=None) -> World:
        real = None
        if self.live:
            real = online.fetch_real(city, progress)
            if real is None and progress:
                progress("Live data unavailable - building a procedural town")
        return build_world(city, real=real, progress=progress)

    # -- synchronous (tests, command line) -----------------------------------------------------------------
    def get(self, city: City) -> World:
        w = self.cached(city)
        if w is None:
            w = self._make(city)
            preview_image(w)
            self._store(self.key(city), w)
        return w

    # -- asynchronous (UI) --------------------------------------------------------------------------------
    def request(self, city: City) -> None:
        """Start building `city` unless it is cached or already running. Only the newest request is kept."""
        if self.cached(city) is not None:
            return
        with self._lock:
            if self._job == self.key(city):
                return
            if self._thread is not None and self._thread.is_alive():
                self._want = city               # run it as soon as the current build ends
                return
        self._start(city)

    def _start(self, city: City) -> None:
        self._job = self.key(city)
        self.progress = "Starting"
        self._thread = threading.Thread(target=self._work, args=(city,), daemon=True)
        self._thread.start()

    def _work(self, city: City) -> None:
        key = self.key(city)
        try:
            w = self._make(city, lambda m: setattr(self, "progress", m))
            preview_image(w)
            self._store(key, w)
            self._error.pop(key, None)
        except Exception as exc:                # a broken town must never take the whole app down
            self._error[key] = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._job = None

    def poll(self) -> None:
        """Call once per frame: starts the queued request after a build finishes."""
        with self._lock:
            idle = self._thread is None or not self._thread.is_alive()
            nxt = self._want if idle else None
            if nxt is not None:
                self._want = None
        if nxt is not None and self.cached(nxt) is None:
            self._start(nxt)

    def building(self, city: City) -> bool:
        return self._job == self.key(city) or (self._want is not None and self.key(self._want) == self.key(city))

    def error(self, city: City) -> str | None:
        return self._error.get(self.key(city))

    def busy(self) -> bool:
        return self._job is not None or self._want is not None
