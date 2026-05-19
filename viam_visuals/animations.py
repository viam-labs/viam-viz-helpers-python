"""Animation specifications.

Each :class:`Animation` subclass describes one per-tick behavior:
:class:`Spin`, :class:`Pulse`, :class:`Lifecycle`, and so on. An
:class:`Animation` knows how to serialize itself to the dict shape the
service-side tick loop reads — the rest of the wire format gymnastics
(field-mask paths in camelCase, UUID rotation on REMOVED→ADDED) lives
inside the service implementation, not here.

Typical use is to attach an Animation to a :class:`viam_visuals.shapes.Visual`
via the ``animation=`` keyword::

    Sphere("ball", radius_mm=80, animation=Spin(period_s=3))

Or directly via the deprecated dict shape (Visual accepts both)::

    Sphere("ball", radius_mm=80, animation={"mode": "spin", "period_s": 3})

The class form is the recommended path — its parameters are
type-checked at construction.

Available modes
---------------
* :class:`Static` — explicit "no animation"; equivalent to ``animation=None``.
* :class:`Spin` — continuous rotation around the entity's local Z.
* :class:`Swing` — bounded swing, like a pendulum.
* :class:`Oscillate` — linear back-and-forth along a world axis.
* :class:`Orbit` — circular translation around the entity's base pose.
* :class:`Pulse` — scale a primitive's size.
* :class:`Breathe` — opacity oscillation.
* :class:`Flicker` — entity blinks in and out of the scene.
* :class:`Lifecycle` — official appear → alive → disappear → gone cycle.
* :class:`ForceVector` — drives an :class:`Arrow`'s length, radius,
  orientation, and color simultaneously.
* :class:`Trajectory` — walk through a sequence of pose waypoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Mapping, MutableMapping, Optional, Union

from .pose import normalize_pose


__all__ = [
    "Animation",
    "AnimationLike",
    "Static",
    "Spin",
    "Swing",
    "Oscillate",
    "Orbit",
    "Pulse",
    "Breathe",
    "Flicker",
    "Lifecycle",
    "ForceVector",
    "Trajectory",
    "normalize_animation",
]


@dataclass
class Animation:
    """Base class for typed animations.

    Subclasses set ``_MODE`` to the wire-format mode string and
    override :meth:`_fields` to contribute their mode-specific keys.
    The default :meth:`to_dict` combines both.

    Subclasses also override :meth:`apply` to mutate a Visual based
    on its base (rest) state and elapsed time ``t``. The library's
    default :meth:`SceneServiceBase.scene_tick` iterates the scene,
    calls ``visual.animation.apply(visual, base, t)`` for each
    animated Visual, and emits the resulting ``scene.update``
    events.
    """

    _MODE: str = field(default="", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        return {}

    def to_dict(self) -> Mapping[str, Any]:
        if not self._MODE:
            raise ValueError(f"{type(self).__name__} forgot to set _MODE")
        out: MutableMapping[str, Any] = {"mode": self._MODE}
        out.update(self._fields())
        return out

    def apply(self, visual: Any, base: Any, t: float) -> None:
        """Mutate ``visual`` in place based on ``base`` (rest state)
        and elapsed time ``t`` (seconds since reconfigure).

        Default: no-op (Static / Animation-base behavior). Subclasses
        override.
        """
        return None


@dataclass
class Static(Animation):
    """No animation; emitted once on add. Equivalent to ``animation=None``."""

    _MODE: str = field(default="none", repr=False, init=False)


@dataclass
class Spin(Animation):
    """Continuous rotation around the entity's local Z axis.

    ``period_s`` is the time in seconds for one full revolution.
    """

    period_s: float = 6.0
    _MODE: str = field(default="spin", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        return {"period_s": float(self.period_s)}

    def apply(self, visual: Any, base: Any, t: float) -> None:
        from .anim_helpers import spin_pose
        visual.pose = spin_pose(base.pose, self.period_s, t)


@dataclass
class Swing(Animation):
    """Bounded swing around a fixed axis — pendulum motion.

    ``amplitude_deg`` is the half-amplitude in degrees;
    ``period_s`` is the cycle time; ``phase_offset_s`` shifts where
    in the cycle the entity starts.
    """

    amplitude_deg: float = 45.0
    period_s: float = 4.0
    phase_offset_s: float = 0.0
    _MODE: str = field(default="swing", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        out: MutableMapping[str, Any] = {
            "amplitude_deg": float(self.amplitude_deg),
            "period_s": float(self.period_s),
        }
        if self.phase_offset_s:
            out["phase_offset_s"] = float(self.phase_offset_s)
        return out

    def apply(self, visual: Any, base: Any, t: float) -> None:
        from .anim_helpers import swing_pose
        visual.pose = swing_pose(
            base.pose, self.period_s, self.amplitude_deg,
            t + self.phase_offset_s,
        )


@dataclass
class Oscillate(Animation):
    """Translate back and forth along a world axis.

    ``axis`` is one of ``"x"``, ``"y"``, ``"z"``. ``amplitude_mm``
    is the displacement amplitude (signed — negative inverts the
    cycle phase). ``period_s`` is the cycle time.
    """

    axis: str = "y"
    amplitude_mm: float = 100.0
    period_s: float = 3.0
    phase_offset_s: float = 0.0
    _MODE: str = field(default="oscillate", repr=False, init=False)

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"Oscillate.axis must be x|y|z; got {self.axis!r}")

    def _fields(self) -> Mapping[str, Any]:
        out: MutableMapping[str, Any] = {
            "axis": self.axis,
            "amplitude_mm": float(self.amplitude_mm),
            "period_s": float(self.period_s),
        }
        if self.phase_offset_s:
            out["phase_offset_s"] = float(self.phase_offset_s)
        return out

    def apply(self, visual: Any, base: Any, t: float) -> None:
        from .anim_helpers import oscillate_pose
        visual.pose = oscillate_pose(
            base.pose, self.period_s, self.amplitude_mm,
            t + self.phase_offset_s, axis=self.axis,
        )


@dataclass
class Orbit(Animation):
    """Circular translation in the XY plane around the entity's
    base pose. ``radius_mm`` is the orbit radius; ``period_s`` is
    the cycle time."""

    radius_mm: float = 100.0
    period_s: float = 4.0
    _MODE: str = field(default="orbit", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        return {
            "radius_mm": float(self.radius_mm),
            "period_s": float(self.period_s),
        }

    def apply(self, visual: Any, base: Any, t: float) -> None:
        from .anim_helpers import orbit_pose
        # Legacy compute_tick orbited only in the XY plane; preserve
        # that semantics here with axis="z".
        visual.pose = orbit_pose(
            base.pose, self.period_s, self.radius_mm, t, axis="z",
        )


@dataclass
class Pulse(Animation):
    """Scale a primitive's size over each period.

    For a :class:`Sphere` or :class:`Capsule`, ``amplitude_mm``
    modulates the radius. For a :class:`Box`, set ``axis`` to
    ``"x"``, ``"y"``, or ``"z"`` to modulate that dimension.
    """

    amplitude_mm: float = 50.0
    period_s: float = 2.0
    axis: Optional[str] = None
    _MODE: str = field(default="pulse", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        out: MutableMapping[str, Any] = {
            "amplitude_mm": float(self.amplitude_mm),
            "period_s": float(self.period_s),
        }
        if self.axis is not None:
            out["axis"] = self.axis
        return out

    def apply(self, visual: Any, base: Any, t: float) -> None:
        import math
        delta = self.amplitude_mm * math.sin(2 * math.pi * t / self.period_s)
        # Dispatch on Visual type. Local imports avoid the
        # animations ↔ shapes circular dep.
        from .shapes import Box, Capsule, Sphere
        if isinstance(visual, Box):
            base_dims = base.dims_mm
            if self.axis == "x":
                visual.dims_mm = (max(0.1, base_dims[0] + delta), base_dims[1], base_dims[2])
            elif self.axis == "y":
                visual.dims_mm = (base_dims[0], max(0.1, base_dims[1] + delta), base_dims[2])
            elif self.axis == "z":
                visual.dims_mm = (base_dims[0], base_dims[1], max(0.1, base_dims[2] + delta))
            else:
                visual.dims_mm = tuple(max(0.1, d + delta) for d in base_dims)
        elif isinstance(visual, Sphere):
            visual.radius_mm = max(0.1, base.radius_mm + delta)
        elif isinstance(visual, Capsule):
            visual.radius_mm = max(0.1, base.radius_mm + delta)
            visual.length_mm = max(0.1, base.length_mm + delta)
        # Other shapes: no-op (no scalable primary dim).


@dataclass
class Breathe(Animation):
    """Smooth opacity oscillation around the entity's base opacity.

    ``amplitude`` is the swing in [0, 1] space. With base opacity
    1.0 and amplitude 0.5, the entity breathes between 0.5 and 1.0.
    """

    amplitude: float = 0.5
    period_s: float = 3.0
    _MODE: str = field(default="breathe", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        return {
            "amplitude": float(self.amplitude),
            "period_s": float(self.period_s),
        }

    def apply(self, visual: Any, base: Any, t: float) -> None:
        import math
        base_opacity = base.opacity if base.opacity is not None else 1.0
        opacity = base_opacity + self.amplitude * math.sin(2 * math.pi * t / self.period_s)
        visual.opacity = max(0.0, min(1.0, opacity))


@dataclass
class Flicker(Animation):
    """Entity blinks in and out of the scene.

    ``duty_cycle`` in [0, 1] is the fraction of each period the entity
    is visible. ``rotate_uuid_on_readd`` defaults to ``True`` — leave
    it unless you're specifically demonstrating the renderer's
    REMOVED-UUID cache bug.
    """

    period_s: float = 1.0
    duty_cycle: float = 0.5
    phase_offset_s: float = 0.0
    rotate_uuid_on_readd: bool = True
    _MODE: str = field(default="flicker", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        out: MutableMapping[str, Any] = {
            "period_s": float(self.period_s),
            "duty_cycle": float(self.duty_cycle),
            # phase_offset_s is always emitted — callers and tests
            # read it unconditionally.
            "phase_offset_s": float(self.phase_offset_s),
        }
        if not self.rotate_uuid_on_readd:
            out["rotate_uuid_on_readd"] = False
        return out

    def apply(self, visual: Any, base: Any, t: float) -> None:
        # Compute the phase within the cycle. duty_cycle of 0.5 means
        # visible for the first half, invisible for the second.
        cycle = (t + self.phase_offset_s) % self.period_s
        visual.invisible = cycle / self.period_s >= self.duty_cycle


@dataclass
class Lifecycle(Animation):
    """Cycle through the worldstatestore lifecycle color convention.

    Phase order: appearing (blue, 50% opacity) → alive (orange,
    100%) → disappearing (red, 50%) → gone (REMOVED). Phase
    durations are in seconds. ``phase_offset_s`` shifts where in
    the cycle the entity starts.
    """

    appear_s: float = 1.0
    alive_s: float = 2.0
    disappear_s: float = 1.0
    gone_s: float = 2.0
    phase_offset_s: float = 0.0
    _MODE: str = field(default="lifecycle", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        out: MutableMapping[str, Any] = {
            "appear_s": float(self.appear_s),
            "alive_s": float(self.alive_s),
            "disappear_s": float(self.disappear_s),
            "gone_s": float(self.gone_s),
        }
        if self.phase_offset_s:
            out["phase_offset_s"] = float(self.phase_offset_s)
        return out

    # Official viam-visualization lifecycle colors / opacities.
    _COLOR_APPEAR: ClassVar[tuple] = (95, 150, 255)
    _COLOR_ALIVE: ClassVar[tuple] = (255, 150, 50)
    _COLOR_DISAPPEAR: ClassVar[tuple] = (255, 90, 70)
    _OPACITY_APPEAR: ClassVar[float] = 0.5
    _OPACITY_ALIVE: ClassVar[float] = 1.0
    _OPACITY_DISAPPEAR: ClassVar[float] = 0.5

    def apply(self, visual: Any, base: Any, t: float) -> None:
        cycle = self.appear_s + self.alive_s + self.disappear_s + self.gone_s
        local = (t + self.phase_offset_s) % cycle
        if local < self.appear_s:
            visual.color = self._COLOR_APPEAR
            visual.opacity = self._OPACITY_APPEAR
            visual.invisible = False
        elif local < self.appear_s + self.alive_s:
            visual.color = self._COLOR_ALIVE
            visual.opacity = self._OPACITY_ALIVE
            visual.invisible = False
        elif local < self.appear_s + self.alive_s + self.disappear_s:
            visual.color = self._COLOR_DISAPPEAR
            visual.opacity = self._OPACITY_DISAPPEAR
            visual.invisible = False
        else:
            visual.invisible = True


@dataclass
class ForceVector(Animation):
    """Drive a :class:`viam_visuals.shapes.Arrow`'s length, radius,
    orientation (precessing around world Z at a fixed tilt), and
    color simultaneously. Useful for force / wrench visualizations.
    """

    period_s: float = 5.0
    length_amplitude_mm: float = 80.0
    radius_amplitude_mm: float = 5.0
    tilt_deg: float = 45.0
    precession_speed: float = 1.0
    color_speed: float = 0.7
    _MODE: str = field(default="force_vector", repr=False, init=False)

    def _fields(self) -> Mapping[str, Any]:
        return {
            "period_s": float(self.period_s),
            "length_amplitude_mm": float(self.length_amplitude_mm),
            "radius_amplitude_mm": float(self.radius_amplitude_mm),
            "tilt_deg": float(self.tilt_deg),
            "precession_speed": float(self.precession_speed),
            "color_speed": float(self.color_speed),
        }

    def apply(self, visual: Any, base: Any, t: float) -> None:
        import math
        from .color import hsv_to_rgb
        from .pose import Pose
        from .shapes import Arrow
        if not isinstance(visual, Arrow):
            return  # ForceVector only meaningful for Arrows
        phase = 2 * math.pi * t / self.period_s
        # Length oscillates around base; radius offset by π/3 so they
        # don't tick in sync (reads as a real changing force).
        visual.length_mm = max(0.1, base.length_mm + self.length_amplitude_mm * math.sin(phase))
        visual.radius_mm = max(0.1, base.radius_mm + self.radius_amplitude_mm * math.sin(phase + math.pi / 3))
        # Orientation: precess around world +Z at fixed tilt.
        tilt_rad = math.radians(self.tilt_deg)
        precession = phase * self.precession_speed
        visual.pose = Pose(
            x=base.pose.x if isinstance(base.pose, Pose) else 0.0,
            y=base.pose.y if isinstance(base.pose, Pose) else 0.0,
            z=base.pose.z if isinstance(base.pose, Pose) else 0.0,
            ox=math.sin(tilt_rad) * math.cos(precession),
            oy=math.sin(tilt_rad) * math.sin(precession),
            oz=math.cos(tilt_rad),
            theta=0.0,
        )
        # Color: cycle hue.
        hue = (t * self.color_speed / self.period_s) % 1.0
        visual.color = hsv_to_rgb(hue, 1.0, 1.0)


@dataclass
class Trajectory(Animation):
    """Walk through a sequence of pose waypoints over ``duration_s``.

    Position and orientation are linearly interpolated between
    adjacent waypoints. With ``loop=True`` (default), the walk
    restarts at the first waypoint after reaching the last.

    ``waypoints`` is a list of :class:`viam_visuals.pose.Pose` instances
    or pose dicts (same shape as :meth:`Pose.to_dict`).
    """

    waypoints: List[Any] = field(default_factory=list)
    duration_s: float = 12.0
    loop: bool = True
    _MODE: str = field(default="trajectory", repr=False, init=False)

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError(
                f"Trajectory needs at least 2 waypoints; got {len(self.waypoints)}"
            )

    def _fields(self) -> Mapping[str, Any]:
        wps: List[Mapping[str, float]] = [dict(normalize_pose(wp))
                                          for wp in self.waypoints]
        return {
            "waypoints": wps,
            "duration_s": float(self.duration_s),
            "loop": bool(self.loop),
        }

    def apply(self, visual: Any, base: Any, t: float) -> None:
        from .anim_helpers import trajectory_pose
        visual.pose = trajectory_pose(
            self.waypoints, self.duration_s, t, loop=self.loop,
        )


# Type alias for what Visual.animation accepts.
AnimationLike = Union[None, Animation, Mapping[str, Any]]


def normalize_animation(a: AnimationLike) -> Optional[Mapping[str, Any]]:
    """Coerce an AnimationLike into the wire-format dict, or None."""
    if a is None:
        return None
    if isinstance(a, Animation):
        return a.to_dict()
    if isinstance(a, Mapping):
        return dict(a)
    raise TypeError(
        f"animation must be None | Animation | dict; got {type(a).__name__}"
    )
