"""Audio engine: a live, throttle-driven motor + wind synthesiser plus a bank of synthesised effects.

All synthesis is plain numpy (see `MotorSynth`, `make_sfx`) so it can be unit-tested without an audio
device. `AudioEngine` streams the motor/wind mix through one mixer channel in short chunks, so the pitch of the
four motors follows the real simulated motor commands in (almost) real time. If no audio device exists
the engine silently disables itself.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import pygame
except ImportError:                      # pragma: no cover - pygame is a hard requirement of the app
    pygame = None

RATE = 22050
CHUNK = 0.045                            # seconds of audio generated per stream chunk
TAU = 2 * math.pi


def _env(n: int, attack: float = 0.004, decay: float = 6.0) -> np.ndarray:
    t = np.arange(n) / RATE
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-decay * t)


def _tone(freq, dur, amp=0.5, decay=6.0, harmonics=(1.0,)):
    n = int(RATE * dur)
    t = np.arange(n) / RATE
    f = np.broadcast_to(np.asarray(freq, float), (n,)) if np.ndim(freq) else freq
    phase = TAU * np.cumsum(np.broadcast_to(f, (n,))) / RATE
    y = sum(h * np.sin((k + 1) * phase) for k, h in enumerate(harmonics))
    return amp * y / sum(harmonics) * _env(n, decay=decay)


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _lowpass(x: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def make_sfx() -> dict[str, np.ndarray]:
    """Synthesise every one-shot effect as float arrays in [-1, 1]."""
    rng = np.random.default_rng(7)
    sfx: dict[str, np.ndarray] = {}
    # target pop: a bright blip + a short burst of noise and a sparkle
    n = int(RATE * 0.34)
    pop = _tone(np.linspace(900, 260, n), 0.34, 0.55, decay=11.0, harmonics=(1, 0.4))
    pop += 0.35 * _noise(n, rng) * _env(n, 0.001, 30.0)
    spark = _tone(np.linspace(1400, 2600, int(RATE * 0.22)), 0.22, 0.25, decay=14.0)
    pop[:len(spark)] += spark
    sfx["pop"] = pop
    # gate: two quick rising notes
    sfx["gate"] = np.concatenate([_tone(740, 0.08, 0.5, 14.0, (1, 0.3)), _tone(1110, 0.16, 0.5, 9.0, (1, 0.3))])
    # arm: low-high beep pair; disarm: high-low
    sfx["arm"] = np.concatenate([_tone(520, 0.07, 0.45, 12.0), _tone(780, 0.1, 0.45, 9.0)])
    sfx["disarm"] = np.concatenate([_tone(780, 0.07, 0.4, 12.0), _tone(480, 0.12, 0.4, 8.0)])
    sfx["click"] = _tone(660, 0.045, 0.35, 40.0)
    sfx["hover"] = _tone(980, 0.025, 0.18, 70.0)
    # crash: thump + crunchy low-passed noise
    n = int(RATE * 0.9)
    thump = _tone(np.linspace(140, 40, n), 0.9, 0.9, decay=5.0)
    crunch = _lowpass(_noise(n, rng), 6) * _env(n, 0.001, 4.5) * 1.6
    sfx["crash"] = np.clip(thump + crunch, -1, 1)
    # countdown beeps and GO
    sfx["tick"] = _tone(660, 0.12, 0.5, 8.0)
    sfx["go"] = _tone(1320, 0.4, 0.55, 5.0, (1, 0.35))
    # success jingle: major arpeggio; failure: falling minor
    notes = (523.25, 659.25, 783.99, 1046.5)
    sfx["win"] = np.concatenate([_tone(f, 0.15 if i < 3 else 0.55, 0.5, 5.0, (1, 0.3, 0.1)) for i, f in enumerate(notes)])
    sfx["fail"] = np.concatenate([_tone(f, 0.2, 0.45, 5.0, (1, 0.4)) for f in (392.0, 330.0, 262.0)])
    sfx["star"] = _tone(np.linspace(900, 1800, int(RATE * 0.2)), 0.2, 0.45, 8.0, (1, 0.3))
    sfx["lowbat"] = np.concatenate([_tone(880, 0.1, 0.4, 14.0), np.zeros(int(RATE * 0.08)), _tone(880, 0.1, 0.4, 14.0)])
    return sfx


class MotorSynth:
    """Streaming synthesiser. `render(motors, speed, n)` returns n int16-ready float samples in [-1, 1].

    Each motor is a harmonic stack whose fundamental follows its commanded power; the small per-motor detune
    gives the characteristic beating of a real quad. Wind is low-passed noise whose level follows airspeed.
    Phase and filter state persist across chunks, so the stream is click-free.
    """

    def __init__(self, seed: int = 3):
        self.rng = np.random.default_rng(seed)
        self.phase = np.zeros(4)
        self.freq = np.full(4, 90.0)
        self.gain = 0.0
        self.wind_gain = 0.0
        self._wind_tail = np.zeros(48)
        self._detune = np.array([1.000, 1.013, 0.991, 1.022])

    @staticmethod
    def motor_freq(m: float) -> float:
        return 95.0 + 520.0 * max(0.0, min(1.0, m)) ** 0.85

    def render(self, motors, speed: float, live: bool, n: int) -> np.ndarray:
        m = np.clip(np.asarray(motors, float), 0, 1)
        f_target = np.array([self.motor_freq(x) for x in m]) * self._detune
        g_target = (0.10 + 0.34 * float(m.mean())) if live else 0.0
        t = (np.arange(1, n + 1) / n)[None, :]
        f = self.freq[:, None] + (f_target - self.freq)[:, None] * t          # glide, no pitch steps
        phase = self.phase[:, None] + np.cumsum(TAU * f / RATE, axis=1)
        y = np.sin(phase) + 0.55 * np.sin(2 * phase + 0.6) + 0.30 * np.sin(3 * phase + 1.1) + 0.14 * np.sin(5 * phase)
        y = y.sum(axis=0) / 8.0
        # propeller "whoosh": noise lightly low-passed, level follows power
        whoosh = _lowpass(self.rng.standard_normal(n), 5) * (0.25 + 0.75 * float(m.mean()))
        gain = self.gain + (g_target - self.gain) * np.linspace(1 / n, 1, n)
        out = (y * 0.85 + whoosh * 0.35) * gain

        # wind: noise through a short moving average (stateful via the saved tail)
        w_target = min(0.28, (speed / 28.0) ** 1.3)
        raw = np.concatenate([self._wind_tail, self.rng.standard_normal(n)])
        k = 24
        lp = np.convolve(raw, np.ones(k) / k, mode="valid")[-n:]
        self._wind_tail = raw[-48:]
        wg = self.wind_gain + (w_target - self.wind_gain) * np.linspace(1 / n, 1, n)
        out = out + lp * wg * 4.0

        self.phase = phase[:, -1] % TAU
        self.freq = f_target
        self.gain, self.wind_gain = g_target, w_target
        return np.clip(out, -1, 1)


def to_int16(x: np.ndarray, volume: float = 1.0) -> bytes:
    return (np.clip(x * volume, -1, 1) * 32767).astype(np.int16).tobytes()


class AudioEngine:
    """Owns the mixer: one streaming channel for motors/wind and a pool of channels for effects."""

    def __init__(self, volume: float = 0.7, muted: bool = False):
        self.enabled = False
        self.volume, self.muted = volume, muted
        self.sounds: dict[str, "pygame.mixer.Sound"] = {}
        self.synth = MotorSynth()
        self._stream = None
        self._lowbat_t = 0.0
        if pygame is None:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=RATE, size=-16, channels=1, buffer=512)
            pygame.mixer.set_num_channels(12)
            self._stream = pygame.mixer.Channel(0)
            for name, arr in make_sfx().items():
                self.sounds[name] = pygame.mixer.Sound(buffer=to_int16(arr, 0.8))
            self.enabled = True
        except (pygame.error, OSError, NotImplementedError):
            self.enabled = False

    # ---- one-shots --------------------------------------------------------------------------------
    def play(self, name: str, volume: float = 1.0) -> None:
        if not self.enabled or self.muted or name not in self.sounds:
            return
        snd = self.sounds[name]
        snd.set_volume(max(0.0, min(1.0, volume * self.volume)))
        try:
            snd.play()
        except pygame.error:
            pass

    # ---- streaming ----------------------------------------------------------------------------------
    def update(self, st, active: bool) -> None:
        """Feed the motor/wind stream. Call once per frame while flying; `active` False silences it."""
        if not self.enabled or self._stream is None:
            return
        n = int(RATE * CHUNK)
        live = bool(st.armed and not st.crashed and active)
        # keep at most one chunk queued: low latency, no unbounded buffering
        try:
            for _ in range(2):
                if self._stream.get_queue() is not None:
                    break
                samples = self.synth.render(st.motors, st.speed, live, n)
                snd = pygame.mixer.Sound(buffer=to_int16(samples, 0.0 if self.muted else self.volume))
                if self._stream.get_busy():
                    self._stream.queue(snd)
                else:
                    self._stream.play(snd)
            if st.armed and st.battery_pct < 15 and not st.crashed:
                self._lowbat_t -= 1 / 60
                if self._lowbat_t <= 0:
                    self.play("lowbat", 0.6)
                    self._lowbat_t = 2.0
        except pygame.error:
            self.enabled = False

    def silence(self) -> None:
        if self.enabled and self._stream is not None:
            try:
                self._stream.stop()
            except pygame.error:
                pass

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        if self.muted:
            self.silence()
        return self.muted

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, v))
