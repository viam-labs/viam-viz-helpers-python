"""UUID strategy helpers for WorldStateStore services.

The Viam 3D viewer accepts two patterns for the on-wire ``uuid`` of
each Transform, and there's been historical ambiguity about which
one works at any given commit:

  * ``"stable"`` — every entity keeps its UUID for life. Animations
    push ``UPDATED`` events with field-mask paths. Matches the RDK
    fake at ``services/worldstatestore/fake/moving_geos_world.go``.
  * ``"versioned"`` — every tick allocates a fresh UUID (timestamp
    + monotonic counter suffix), emits ``REMOVED`` for the prior
    version and ``ADDED`` for the new one. Matches the
    apriltag-tracker pattern. Use this if the renderer drops
    UPDATED events for stable UUIDs.

The renderer also caches REMOVED UUIDs and silently drops
subsequent ADDED events for the same UUID — see
LESSONS.md::renderer-caches-removed-uuids-rotate-on-readd. Any
animation that mutates scene-graph membership (lifecycle, flicker)
needs to rotate the UUID even in stable strategy. The
``versioned_uuid`` helper here is the canonical UUID generator;
callers responsible for choosing when to use it.
"""

from __future__ import annotations

import threading
import time


__all__ = ["VALID_STRATEGIES", "initial_uuid", "versioned_uuid"]


VALID_STRATEGIES = ("stable", "versioned")


# Module-global monotonic counter shared across the process. Combined
# with epoch ms, guarantees uniqueness even when multiple UUIDs are
# allocated within the same millisecond (which the test suite does
# routinely). A simple int with a lock is enough — the contention
# domain is the animation tick loop, which doesn't need lock-free
# concurrency at this scale.
_counter_lock = threading.Lock()
_counter = 0


def initial_uuid(label: str, strategy: str) -> bytes:
    """Return the UUID for an entity at install time, given the
    service's strategy. ``"stable"`` → ``label`` bytes (so identity
    stays human-readable). ``"versioned"`` → a fresh timestamp-suffixed
    UUID via :func:`versioned_uuid`.
    """
    if strategy == "versioned":
        return versioned_uuid(label)
    return label.encode()


def versioned_uuid(label: str) -> bytes:
    """Allocate a fresh UUID for ``label`` of the form
    ``<label>_<epoch_ms>_<counter>``. Used by the ``versioned``
    strategy and by REMOVED→ADDED transitions in stable strategy
    when the entity's scene-graph membership changes (the renderer
    caches REMOVED UUIDs and drops subsequent ADDEDs for the same
    one)."""
    global _counter
    with _counter_lock:
        _counter += 1
        c = _counter
    return f"{label}_{int(time.time() * 1000)}_{c}".encode()
