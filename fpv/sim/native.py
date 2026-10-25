"""ctypes bridge to the C++ core (Facade pattern): auto-locates, auto-builds and wraps libfpvcore.

The whole simulation hot path (RK4 physics + flight controller @ 2 kHz) lives in C++; Python makes one
`advance()` call and one `state()` read per rendered frame.
"""
from __future__ import annotations

import ctypes as C
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"

_D = C.c_double
ABI_VERSION = 2
_SCALARS_A = ["mass", "ixx", "iyy", "izz", "arm_x", "arm_y", "max_thrust", "max_omega", "tau_up", "tau_down",
              "c_yaw", "drag_lin", "drag_quad_xy", "drag_quad_z", "ang_drag"]
_ARRAYS = ["rc_rate", "super_rate", "expo", "kp", "ki", "kd", "kff"]
_SCALARS_B = ["angle_kp", "angle_max_deg", "throttle_expo", "gyro_noise_dps", "d_cut_hz", "i_limit", "idle",
              "wind_speed", "wind_dir_deg", "gust", "cells", "capacity_mah", "r_int", "physics_hz",
              "control_div", "crash_speed", "seed"]


class Config(C.Structure):
    _fields_ = ([(n, _D) for n in _SCALARS_A] + [(n, _D * 3) for n in _ARRAYS] + [(n, _D) for n in _SCALARS_B])


class State(C.Structure):
    _fields_ = [("time", _D), ("pos", _D * 3), ("vel", _D * 3), ("quat", _D * 4), ("euler", _D * 3),
                ("gyro", _D * 3), ("rate_sp", _D * 3), ("motors", _D * 4), ("throttle", _D), ("voltage", _D),
                ("current", _D), ("mah", _D), ("battery_pct", _D), ("speed", _D), ("accel_g", _D),
                ("vscale", _D), ("armed", C.c_int32), ("crashed", C.c_int32), ("on_ground", C.c_int32),
                ("mode", C.c_int32)]


class Obstacle(C.Structure):
    _fields_ = [(n, _D) for n in ("cx", "cy", "cz", "hx", "hy", "hz", "yaw")]


def _lib_name() -> str:
    return {"win32": "fpvcore.dll", "darwin": "libfpvcore.dylib"}.get(sys.platform, "libfpvcore.so")


def _find_lib() -> Path | None:
    for sub in ("bin", "bin/Release", "bin/Debug", "Release", "Debug", "."):
        p = BUILD / sub / _lib_name()
        if p.exists():
            return p
    return None


def _sources_newer(lib: Path) -> bool:
    t = lib.stat().st_mtime
    return any(p.stat().st_mtime > t for p in (ROOT / "core").rglob("*") if p.suffix in (".cpp", ".hpp"))


def build(verbose: bool = True) -> Path:
    """Configure + compile the C++ core with CMake (falls back to a direct compiler call)."""
    out = lambda *a: print("[build]", *a) if verbose else None  # noqa: E731
    if shutil.which("cmake"):
        out("cmake configure + build (Release)")
        for cmd in (["cmake", "-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(BUILD), "--config", "Release", "-j"]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"{' '.join(cmd[:2])} failed:\n{r.stdout}\n{r.stderr}")
    else:
        cxx = shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            raise RuntimeError("No CMake or C++ compiler found. Install CMake + a C++17 compiler "
                               "(MSVC Build Tools / MinGW / Xcode CLT / build-essential).")
        out("cmake not found - compiling directly with", cxx)
        (BUILD / "bin").mkdir(parents=True, exist_ok=True)
        cmd = [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-I", str(ROOT / "core/include"),
               str(ROOT / "core/src/simulator.cpp"), str(ROOT / "core/src/c_api.cpp"),
               "-o", str(BUILD / "bin" / _lib_name())]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"Compilation failed:\n{r.stderr}")
    lib = _find_lib()
    if lib is None:
        raise RuntimeError("Build finished but the core library was not found in build/.")
    return lib


def load_library(auto_build: bool = True) -> C.CDLL:
    lib = _find_lib()
    if auto_build and (lib is None or _sources_newer(lib)):
        lib = build()
    if lib is None:
        raise RuntimeError("C++ core not built. Run: python run_sim.py --build")
    dll = C.CDLL(str(lib))
    dll.fpv_create.restype = C.c_void_p
    for name, args in {
        "fpv_destroy": [C.c_void_p], "fpv_reset": [C.c_void_p],
        "fpv_set_pose": [C.c_void_p, _D, _D, _D, _D],
        "fpv_set_config": [C.c_void_p, C.POINTER(Config)], "fpv_get_config": [C.c_void_p, C.POINTER(Config)],
        "fpv_set_sticks": [C.c_void_p, _D, _D, _D, _D], "fpv_set_armed": [C.c_void_p, C.c_int],
        "fpv_set_mode": [C.c_void_p, C.c_int],
        "fpv_set_obstacles": [C.c_void_p, C.POINTER(Obstacle), C.c_int],
        "fpv_advance": [C.c_void_p, _D], "fpv_get_state": [C.c_void_p, C.POINTER(State)],
        "fpv_default_config": [C.POINTER(Config)], "fpv_create": [C.POINTER(Config)],
        "fpv_set_terrain": [C.c_void_p, C.POINTER(C.c_float), C.c_int, _D, _D, _D],
        "fpv_ground_height": [C.c_void_p, _D, _D],
    }.items():
        getattr(dll, name).argtypes = args
    dll.fpv_advance.restype = C.c_int
    dll.fpv_ground_height.restype = _D
    if dll.fpv_abi_version() != ABI_VERSION:
        raise RuntimeError("Stale C++ core (ABI mismatch): rebuild with  python run_sim.py --build")
    for fn, cls in (("fpv_config_size", Config), ("fpv_state_size", State), ("fpv_obstacle_size", Obstacle)):
        if getattr(dll, fn)() != C.sizeof(cls):
            raise RuntimeError(f"ABI mismatch for {cls.__name__}: rebuild the core (python run_sim.py --build)")
    return dll


class Simulator:
    """Pythonic facade over the C handle."""

    ACRO, ANGLE = 0, 1

    def __init__(self, auto_build: bool = True):
        self._dll = load_library(auto_build)
        cfg = Config()
        self._dll.fpv_default_config(C.byref(cfg))
        self.base_config = cfg
        self._h = self._dll.fpv_create(C.byref(cfg))
        self._state = State()
        self._obstacle_buf = None

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._dll.fpv_destroy(self._h)
            self._h = None

    __del__ = close

    # --- configuration -------------------------------------------------
    def default_config(self) -> Config:
        cfg = Config()
        self._dll.fpv_default_config(C.byref(cfg))
        return cfg

    def get_config(self) -> Config:
        cfg = Config()
        self._dll.fpv_get_config(self._h, C.byref(cfg))
        return cfg

    def set_config(self, cfg: Config) -> None:
        self._dll.fpv_set_config(self._h, C.byref(cfg))

    # --- control ---------------------------------------------------------
    def reset(self, pose: Sequence[float] | None = None) -> None:
        if pose is None:
            self._dll.fpv_reset(self._h)
        else:
            self._dll.fpv_set_pose(self._h, *pose)

    def set_sticks(self, roll: float, pitch: float, yaw: float, throttle: float) -> None:
        self._dll.fpv_set_sticks(self._h, roll, pitch, yaw, throttle)

    def set_armed(self, armed: bool) -> None:
        self._dll.fpv_set_armed(self._h, int(armed))

    def set_mode(self, mode: int) -> None:
        self._dll.fpv_set_mode(self._h, int(mode))

    def set_obstacles(self, boxes: Iterable[Sequence[float]]) -> None:
        boxes = list(boxes)
        arr = (Obstacle * max(1, len(boxes)))()
        for i, b in enumerate(boxes):
            arr[i] = Obstacle(*b)
        self._obstacle_buf = arr  # keep alive
        self._dll.fpv_set_obstacles(self._h, arr, len(boxes))

    def set_terrain(self, heights, cell: float, ox: float, oy: float) -> None:
        """Install a heightfield (n x n float array, indexed [iy, ix]); None restores the flat world."""
        if heights is None:
            self._terrain_buf = None
            self._dll.fpv_set_terrain(self._h, None, 0, 1.0, 0.0, 0.0)
            return
        import numpy as np
        buf = np.ascontiguousarray(heights, dtype=np.float32)
        if buf.ndim != 2 or buf.shape[0] != buf.shape[1]:
            raise ValueError("terrain must be a square 2-D array")
        self._terrain_buf = buf          # the core copies the data, this just keeps the pointer valid during the call
        self._dll.fpv_set_terrain(self._h, buf.ctypes.data_as(C.POINTER(C.c_float)), buf.shape[0], cell, ox, oy)

    def ground_height(self, x: float, y: float) -> float:
        return self._dll.fpv_ground_height(self._h, x, y)

    def advance(self, dt: float) -> int:
        return self._dll.fpv_advance(self._h, dt)

    def state(self) -> State:
        self._dll.fpv_get_state(self._h, C.byref(self._state))
        return self._state
