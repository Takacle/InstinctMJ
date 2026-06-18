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
    """Creates half-ring arcs hugging wheel-shaped foot soles.

    For each wheel center ``(x_c, y_c)`` in ``centers``, each radius ``r`` in
    ``radii``, and each angle ``theta`` in ``[angle_min, angle_max]``, a point is
    placed at ``(x_c + r*cos(theta), y_c, r*sin(theta))``. Each center thus
    produces one lower-semicircle half-ring (single Y layer) -- matching one
    wheel of the foot sole.

    Args:
        points_generator_cfg (Arc3dPointsGeneratorCfg): Configuration for the arc generator.

    Returns:
        torch.Tensor shape (N, 3): A tensor containing the generated arc points.
    """
    centers = torch.tensor(points_generator_cfg.centers, dtype=torch.float)  # (C, 2)
    radii = torch.tensor(points_generator_cfg.radii, dtype=torch.float)       # (R,)
    angles = torch.linspace(
        points_generator_cfg.angle_min,
        points_generator_cfg.angle_max,
        points_generator_cfg.angle_num,
    )  # (A,)
    center_idx = torch.arange(centers.shape[0], dtype=torch.float)  # (C,)

    grid_ci, grid_r, grid_th = torch.meshgrid(center_idx, radii, angles, indexing="ij")
    grid_ci = grid_ci.long()
    grid_x = centers[grid_ci, 0] + grid_r * torch.cos(grid_th)
    grid_y = centers[grid_ci, 1]
    grid_z = grid_r * torch.sin(grid_th)

    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
