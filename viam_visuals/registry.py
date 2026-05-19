"""In-process resource registry.

Lets a downstream resource hold a direct Python reference to an
upstream resource that lives in the same module binary, skipping
the gRPC round-trip the framework's ``Dependencies`` stubs use.

This matters when two resources ship from the same module and
exchange events at tick rate: a 30Hz tick × 50 events × cross-
process gRPC adds measurable latency and CPU. A direct method call
is essentially free.

Usage on the upstream side (e.g. a visualizer service that wants
to be addressable in-process):

    from viam_visuals import registry

    class PlaygroundVisualizer(...):
        @classmethod
        async def new(cls, config, deps):
            instance = await super().new(config, deps)
            registry.register(config.name, instance)
            return instance

        async def close(self):
            registry.unregister(self.name)
            await super().close()

Usage on the downstream side (e.g. a driver sensor):

    vis = registry.lookup("playground-visualizer")
    if vis is None:
        # Visualizer lives in another module — fall back to the
        # gRPC stub the framework injected via Dependencies.
        vis = deps[WorldStateStoreService.get_resource_name(name)]

The registry is module-binary local: each Python module process
gets its own dict. Modules that ship visualizer and driver as the
same binary share it; a separately-shipped driver module won't see
the upstream and naturally falls through to its gRPC stub.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

__all__ = ["register", "unregister", "lookup", "names"]


_lock = Lock()
_resources: dict[str, Any] = {}


def register(name: str, instance: Any) -> None:
    """Register ``instance`` under ``name``. Replaces any prior
    registration with the same name (mirrors the framework's behaviour
    on reconfigure — a fresh constructor produces a fresh instance)."""
    with _lock:
        _resources[name] = instance


def unregister(name: str) -> None:
    """Remove ``name`` from the registry. No-op if not registered."""
    with _lock:
        _resources.pop(name, None)


def lookup(name: str) -> Any | None:
    """Return the instance registered under ``name``, or ``None`` if
    it isn't registered (typically: lives in a different module
    process)."""
    with _lock:
        return _resources.get(name)


def names() -> list[str]:
    """Return the currently-registered names. Useful for debugging
    and the ``snapshot`` DoCommand verb."""
    with _lock:
        return sorted(_resources)
