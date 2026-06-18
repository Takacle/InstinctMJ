from dataclasses import MISSING, dataclass
from math import pi
from typing import Callable

from .points_generator import arc3d_points_generator, grid3d_points_generator


@dataclass(kw_only=True)
class PointsGeneratorCfg:
    """Specifying how the volume points are generated."""

    func: Callable = MISSING  # type: ignore


@dataclass(kw_only=True)
class Grid3dPointsGeneratorCfg(PointsGeneratorCfg):
    func: Callable = grid3d_points_generator

    x_min: float = -1.0
    """Minimum x coordinate of the grid."""
    x_max: float = 1.0
    """Maximum x coordinate of the grid."""
    x_num: int = 10
    """Number of points along the x axis."""
    y_min: float = -1.0
    """Minimum y coordinate of the grid."""
    y_max: float = 1.0
    """Maximum y coordinate of the grid."""
    y_num: int = 10
    """Number of points along the y axis."""
    z_min: float = -1.0
    """Minimum z coordinate of the grid."""
    z_max: float = 1.0
    """Maximum z coordinate of the grid."""
    z_num: int = 10
    """Number of points along the z axis."""


@dataclass(kw_only=True)
class Arc3dPointsGeneratorCfg(PointsGeneratorCfg):
    func: Callable = arc3d_points_generator

    centers: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    """Wheel centers ``(x_c, y_c)`` in the body frame. One lower-semicircle
    half-ring (single Y layer) is generated per center, matching one wheel of
    the foot sole. e.g. ``((0.0, 0.067), (0.0, -0.067), (0.20, 0.033), (0.20, -0.033))``
    for the staggered 4-wheel foot (union of L/R feet's wheel centers).
    """

    radii: tuple[float, ...] = (0.097, 0.105)
    """Radii of the arc shells in the body X-Z plane.
    0.097 = wheel envelope (R 0.085 + capsule r 0.012); 0.105 = +8 mm early-warning shell.
    """

    angle_min: float = pi
    """Start angle (rad). ``pi`` -> (-X, z=0) back side of the wheel."""
    angle_max: float = 2.0 * pi
    """End angle (rad). ``2*pi`` -> (+X, z=0) front side, sweeping through z=-r (bottom, ``3*pi/2``)."""
    angle_num: int = 17
    """Number of angular samples over ``[angle_min, angle_max]`` (17 -> every 11.25 deg)."""
