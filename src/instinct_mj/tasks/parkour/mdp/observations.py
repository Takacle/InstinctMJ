"""Observation terms for SSR Imagined Foothold Guidance.

Provides two observation groups used by the foothold imagination model:

- ``foothold_privilege_obs``: privileged per-foot state for the imagination
  model input and for support-deficiency computation inside auxiliary reward.
  Includes sole-patch terrain heights, foot position/velocity in base frame,
  contact state, terrain level, and one-hot(ish) terrain type index.

- ``foothold_target_obs``: supervision target (future contact xy in base
  frame) and a validity flag, sourced from the FootholdBuffer.

These terms are intended to be placed in observation groups that are only
visible to the critic (asymmetric actor-critic) and to the imagination
algorithm mixin, never to the policy directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from instinct_mj.tasks.parkour.mdp.foothold_buffer import (
    get_or_create_foothold_buffer,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def _get_contact_first_contact(
    contact_sensor: ContactSensor, dt: float
) -> torch.Tensor:
    """Thin wrapper around ContactSensor.compute_first_contact."""
    return contact_sensor.compute_first_contact(dt)


def foothold_privilege_obs(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    left_scanner_name: str,
    right_scanner_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body_name: str = "base_link",
    max_delay_steps: int = 50,
) -> torch.Tensor:
    """Concatenate privileged per-foot quantities for the imagination model.

    Output layout (last dim, in order):
      - left sole heights   (n_left_rays,)
      - right sole heights  (n_right_rays,)
      - left foot pos b xy  (2,)
      - right foot pos b xy (2,)
      - left foot vel b xy  (2,)
      - right foot vel b xy (2,)
      - left contact flag   (1,)
      - right contact flag  (1,)
      - terrain level       (1,)
      - terrain type index  (1,)

    The buffer for future-contact target tracking is created lazily here so
    that downstream ``foothold_target_obs`` can read from the same buffer.
    """
    asset: Entity = env.scene[asset_cfg.name]
    left_scanner: RayCastSensor = env.scene[left_scanner_name]
    right_scanner: RayCastSensor = env.scene[right_scanner_name]
    contact_sensor: ContactSensor = env.scene[contact_sensor_name]

    # Sole-patch heights: take hit z relative to root link z (signed).
    # Using world-frame z differences is consistent with feet_at_plane reward.
    root_pos_w = asset.data.root_link_pos_w  # (num_envs, 3)
    left_hit_z = left_scanner.data.hit_pos_w[..., 2]  # (num_envs, n_rays)
    right_hit_z = right_scanner.data.hit_pos_w[..., 2]
    # Mark missed rays (distance < 0) as "ground level" at root height.
    left_miss = left_scanner.data.distances < 0.0
    right_miss = right_scanner.data.distances < 0.0
    left_hit_z = torch.where(left_miss, root_pos_w[:, 2:1].expand_as(left_hit_z), left_hit_z)
    right_hit_z = torch.where(right_miss, root_pos_w[:, 2:1].expand_as(right_hit_z), right_hit_z)
    # Height relative to root link z (positive = sole is above root; terrain rises).
    left_heights = left_hit_z - root_pos_w[:, 2:1]
    right_heights = right_hit_z - root_pos_w[:, 2:1]

    # Foot pos/vel in base frame (xy only).
    body_ids = asset_cfg.body_ids
    assert len(body_ids) == 2, "foothold_privilege_obs expects exactly 2 feet"
    foot_pos_w = asset.data.body_link_pos_w[:, body_ids, :]  # (E, 2, 3)
    foot_vel_w = asset.data.body_link_lin_vel_w[:, body_ids, :]  # (E, 2, 3)
    root_pos = asset.data.root_link_pos_w  # (E, 3)
    root_quat = asset.data.root_link_quat_w  # (E, 4)
    foot_pos_b = quat_rotate_inverse(
        root_quat.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4),
        (foot_pos_w - root_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape(-1, 2, 3)[:, :, :2]  # (E, 2, 2)
    foot_vel_b = quat_rotate_inverse(
        root_quat.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4),
        foot_vel_w.reshape(-1, 3),
    ).reshape(-1, 2, 3)[:, :, :2]  # (E, 2, 2)

    # Contact flags.
    is_contact = (
        torch.max(
            torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1),
            dim=2,
        )[0]
        > 1.0
    )  # (E, 2)

    # Terrain level + type.
    terrain = env.scene.terrain
    terrain_level = terrain.terrain_levels.float().unsqueeze(-1)  # (E, 1)
    terrain_type = terrain.terrain_types.float().unsqueeze(-1)  # (E, 1)

    # Lazily create + update the foothold buffer (shared with foothold_target_obs).
    # NOTE: buffer is updated here so it sees foot_pos_b at the same step.
    is_first_contact = _get_contact_first_contact(contact_sensor, env.step_dt)
    buffer = get_or_create_foothold_buffer(
        env, "foothold_buffer", num_feet=2, max_delay_steps=max_delay_steps
    )
    buffer.update(foot_pos_b, is_first_contact)

    return torch.cat(
        [
            left_heights,
            right_heights,
            foot_pos_b[:, 0],
            foot_pos_b[:, 1],
            foot_vel_b[:, 0],
            foot_vel_b[:, 1],
            is_contact[:, 0:1].float(),
            is_contact[:, 1:2].float(),
            terrain_level,
            terrain_type,
        ],
        dim=-1,
    )


def foothold_target_obs(
    env: ManagerBasedRlEnv,
    max_delay_steps: int = 50,
) -> torch.Tensor:
    """Return the supervision target for the foothold imagination model.

    Output layout (last dim):
      - left  target xy (2,)
      - right target xy (2,)
      - left  valid flag (1,)
      - right valid flag (1,)

    The target is read from the FootholdBuffer that is updated in
    ``foothold_privilege_obs``. Callers must ensure both obs terms share
    the same ``buffer_name`` (currently hardcoded to "foothold_buffer").
    """
    buffer = get_or_create_foothold_buffer(
        env, "foothold_buffer", num_feet=2, max_delay_steps=max_delay_steps
    )
    target, valid = buffer.get_target()
    return torch.cat(
        [
            target[:, 0],
            target[:, 1],
            valid[:, 0:1].float(),
            valid[:, 1:2].float(),
        ],
        dim=-1,
    )
