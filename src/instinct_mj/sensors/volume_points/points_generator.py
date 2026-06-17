from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .points_generator_cfg import Arc3dPointsGeneratorCfg, Grid3dPointsGeneratorCfg


def grid3d_points_generator(points_generator_cfg: Grid3dPointsGeneratorCfg) -> torch.Tensor:
    """Creates a grid of points in 3D space.

    Args:
        points_generator_cfg (Grid3dPointsGeneratorCfg): Configuration for the points generator.

    Returns:
        torch.Tensor shape (N, 3): A tensor containing the generated points in 3D space.
    """
    x = torch.linspace(
        points_generator_cfg.x_min,
        points_generator_cfg.x_max,
        points_generator_cfg.x_num,
    )
    y = torch.linspace(
        points_generator_cfg.y_min,
        points_generator_cfg.y_max,
        points_generator_cfg.y_num,
    )
    z = torch.linspace(
        points_generator_cfg.z_min,
        points_generator_cfg.z_max,
        points_generator_cfg.z_num,
    )

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)


def arc3d_points_generator(points_generator_cfg: Arc3dPointsGeneratorCfg) -> torch.Tensor:
    """Creates a curtain of points on circular arcs in the body X-Z plane.

    Points lie at ``(r*cos(theta), y, r*sin(theta))`` for each radius ``r``, each
    angle ``theta`` in ``[angle_min, angle_max]``, and each lateral offset ``y``
    in ``[y_min, y_max]``. This hugs a wheel-shaped foot sole (arc-shaped)
    instead of filling an axis-aligned box.

    Args:
        points_generator_cfg (Arc3dPointsGeneratorCfg): Configuration for the arc generator.

    Returns:
        torch.Tensor shape (N, 3): A tensor containing the generated arc-curtain points.
    """
    radii = torch.tensor(points_generator_cfg.radii, dtype=torch.float)
    angles = torch.linspace(
        points_generator_cfg.angle_min,
        points_generator_cfg.angle_max,
        points_generator_cfg.angle_num,
    )
    y = torch.linspace(
        points_generator_cfg.y_min,
        points_generator_cfg.y_max,
        points_generator_cfg.y_num,
    )

    grid_r, grid_th, grid_y = torch.meshgrid(radii, angles, y, indexing="ij")
    grid_x = grid_r * torch.cos(grid_th)
    grid_z = grid_r * torch.sin(grid_th)

    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
