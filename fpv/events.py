"""Tiny synchronous publish/subscribe bus (Observer pattern) decoupling game logic from UI/audio/logging."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[..., None]]] = defaultdict(list)

    def subscribe(self, topic: str, fn: Callable[..., None]) -> Callable[[], None]:
        self._subs[topic].append(fn)

        def unsubscribe() -> None:
            if fn in self._subs[topic]:   # idempotent: safe to call twice
                self._subs[topic].remove(fn)
        return unsubscribe

    def publish(self, topic: str, **data: Any) -> None:
        for fn in list(self._subs[topic]):
            fn(**data)
