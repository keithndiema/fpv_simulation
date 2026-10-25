"""Telemetry: numpy ring buffer for live graphs + CSV blackbox logger."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

CHANNELS = ("t", "gyro_r", "gyro_p", "gyro_y", "sp_r", "sp_p", "sp_y", "throttle", "speed", "alt",
            "m0", "m1", "m2", "m3")
LOG_DIR = Path.home() / ".fpv_sim" / "logs"


class Telemetry:
    def __init__(self, seconds: float = 4.0, hz: float = 60.0):
        self.n = int(seconds * hz)
        self.buf = np.zeros((self.n, len(CHANNELS)), dtype=np.float32)
        self.count = 0
        self._log: list[list[float]] | None = None

    def push(self, st) -> None:
        row = (st.time, *st.gyro, *st.rate_sp, st.throttle, st.speed, st.pos[2], *st.motors)
        self.buf[self.count % self.n] = row
        self.count += 1
        if self._log is not None:
            self._log.append(list(row))

    def series(self, name: str) -> np.ndarray:
        """Chronologically ordered copy of one channel."""
        i = CHANNELS.index(name)
        if self.count < self.n:
            return self.buf[: self.count, i]
        return np.roll(self.buf[:, i], -(self.count % self.n))

    @property
    def recording(self) -> bool:
        return self._log is not None

    def toggle_recording(self) -> Path | None:
        if self._log is None:
            self._log = []
            return None
        path = self._flush()
        self._log = None
        return path

    def _flush(self) -> Path | None:
        if not self._log:
            return None
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / time.strftime("flight_%Y%m%d_%H%M%S.csv")
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CHANNELS)
            w.writerows(self._log)
        return path
