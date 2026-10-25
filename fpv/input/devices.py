"""Input layer: Strategy per device (keyboard / gamepad) + Composite that picks the active one."""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod

import pygame

from .sticks import Sticks  # noqa: F401  (re-exported for backwards compatibility)
from .virtual import VirtualControls


class Action(enum.Enum):
    ARM = "arm"
    RESET = "reset"
    MODE = "mode"
    CAMERA = "camera"
    PAUSE = "pause"
    GRAPH = "graph"
    HELP = "help"
    RECORD = "record"
    AUTOPILOT = "autopilot"
    MUTE = "mute"


def _slew(cur: float, target: float, rate: float, dt: float) -> float:
    step = rate * dt
    return cur + max(-step, min(step, target - cur))


class InputDevice(ABC):
    name = "device"

    @abstractmethod
    def poll(self, dt: float) -> Sticks: ...

    @abstractmethod
    def action_for(self, event: pygame.event.Event) -> Action | None: ...


class KeyboardDevice(InputDevice):
    """W/S throttle (latching), A/D yaw, arrows roll/pitch - with slew-rate smoothing so taps aren't binary."""
    name = "Keyboard"
    KEYS = {pygame.K_SPACE: Action.ARM, pygame.K_r: Action.RESET, pygame.K_m: Action.MODE,
            pygame.K_c: Action.CAMERA, pygame.K_p: Action.PAUSE, pygame.K_g: Action.GRAPH,
            pygame.K_F1: Action.HELP, pygame.K_h: Action.HELP, pygame.K_F5: Action.RECORD,
            pygame.K_t: Action.AUTOPILOT, pygame.K_n: Action.MUTE}

    def __init__(self, sensitivity: float = 0.75):
        self.sensitivity = sensitivity
        self.s = Sticks()

    def action_for(self, event):
        return self.KEYS.get(event.key) if event.type == pygame.KEYDOWN else None

    def poll(self, dt: float) -> Sticks:
        k = pygame.key.get_pressed()
        axis = lambda neg, pos: (k[pos] - k[neg]) * self.sensitivity  # noqa: E731
        tr, tp, ty = axis(pygame.K_LEFT, pygame.K_RIGHT), axis(pygame.K_DOWN, pygame.K_UP), \
            axis(pygame.K_a, pygame.K_d)
        s = self.s
        for name, tgt in (("roll", tr), ("pitch", tp), ("yaw", ty)):
            cur = getattr(s, name)
            rate = 5.0 if tgt != 0 and (cur * tgt >= 0) else 9.0   # attack vs. spring-back
            setattr(s, name, _slew(cur, tgt, rate, dt))
        boost = 2.0 if (k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]) else 1.0
        s.throttle = max(0.0, min(1.0, s.throttle + (k[pygame.K_w] - k[pygame.K_s]) * 0.55 * boost * dt))
        if k[pygame.K_x]:
            s.throttle = 0.0
        return s


class GamepadDevice(InputDevice):
    """Mode-2 layout: left stick = throttle/yaw, right stick = pitch/roll. Remap via AXES if needed."""
    name = "Gamepad"
    AXES = {"yaw": (0, 1.0), "throttle": (1, -1.0), "roll": (2, 1.0), "pitch": (3, -1.0)}
    BUTTONS = {0: Action.ARM, 1: Action.RESET, 3: Action.MODE, 2: Action.CAMERA, 7: Action.PAUSE,
               6: Action.GRAPH}
    DEADZONE = 0.06

    def __init__(self, joystick: pygame.joystick.JoystickType):
        self.js = joystick
        self.js.init()
        self.name = f"Gamepad ({joystick.get_name()})"

    def _axis(self, key: str) -> float:
        idx, sign = self.AXES[key]
        if idx >= self.js.get_numaxes():
            return 0.0
        v = self.js.get_axis(idx) * sign
        if abs(v) < self.DEADZONE:
            return 0.0
        return (abs(v) - self.DEADZONE) / (1 - self.DEADZONE) * (1 if v > 0 else -1)

    def poll(self, dt: float) -> Sticks:
        return Sticks(self._axis("roll"), self._axis("pitch"), self._axis("yaw"),
                      max(0.0, min(1.0, (self._axis("throttle") + 1.0) / 2.0)))

    def action_for(self, event):
        return self.BUTTONS.get(event.button) if event.type == pygame.JOYBUTTONDOWN else None


class InputManager:
    """Composite: gamepad takes over as soon as it is moved, keyboard when keys are used; on-screen sticks
    (mouse / touch) override whichever axes they are holding."""

    def __init__(self, sensitivity: float):
        pygame.joystick.init()
        self.keyboard = KeyboardDevice(sensitivity)
        self.virtual = VirtualControls()
        self.pads: list[GamepadDevice] = []
        self.active: InputDevice = self.keyboard
        self.rescan()

    def rescan(self) -> None:
        self.pads = [GamepadDevice(pygame.joystick.Joystick(i)) for i in range(pygame.joystick.get_count())]
        if self.pads:
            self.active = self.pads[0]

    def handle_event(self, event, size=None) -> Action | None:
        if size is not None and self.virtual.handle_event(event, size):
            return None
        if event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
            self.rescan()
            return None
        if event.type == pygame.KEYDOWN:
            self.active = self.keyboard
            return self.keyboard.action_for(event)
        if event.type == pygame.JOYBUTTONDOWN and self.pads:
            self.active = self.pads[0]
            return self.pads[0].action_for(event)
        return None

    def poll(self, dt: float) -> Sticks:
        kb = self.keyboard.poll(dt)
        base = kb
        if self.pads:
            pad = self.pads[0].poll(dt)
            if pad.magnitude() > 0.05:
                self.active = self.pads[0]
            if self.active is self.pads[0]:
                base = pad
        out = self.virtual.apply(base)
        if self.virtual.left.pid is not None:
            self.keyboard.s.throttle = out.throttle          # keyboard throttle continues from where the stick was left
        if self.virtual.holding:
            self.active = self.keyboard if not self.pads else self.active
        return out
