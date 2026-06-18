from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from instinct_mj.envs.mdp.rewards.regularizations import (
    applied_torque_limits_by_ratio as _applied_torque_limits_by_ratio_general,
)
from instinct_mj.envs.mdp.rewards.regularizations import motors_power_square as _motors_power_square_general

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of reference linear velocity (x/y in body frame)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of reference yaw angular velocity (body frame)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def heading_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Compute heading command magnitude."""
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, 2])


def dont_wait(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_link_lin_vel_b[:, 0]

    return (lin_vel_cmd_x > 0.3) * (
        (lin_vel_x < 0.15).float() + (lin_vel_x < 0.0).float() + (lin_vel_x < -0.15).float()
    )


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset: Entity = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw_abs = torch.abs(cmd[:, 2])

    return (dof_error - offset) * (cmd_lin_norm < threshold) * (cmd_yaw_abs < threshold)


def feet_air_time(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_threshold: float,
    sensor_name: str,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold
    and also keeping one foot at a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step),
    then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

    # no reward for zero command
    cmd = env.command_manager.get_command(command_name)
    reward *= torch.logical_or(
        torch.norm(cmd[:, :2], dim=1) > vel_threshold,
        torch.abs(cmd[:, 2]) > vel_threshold,
    )
    return reward


def feet_slide(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot sliding speed while feet are in contact."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]

    in_contact = torch.max(torch.linalg.vector_norm(sensor.data.force_history, dim=-1), dim=2)[0] > threshold
    foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip_speed = torch.norm(foot_vel_xy, dim=-1)
    return torch.sum(slip_speed * in_contact.float(), dim=1)


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_deviation_square(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_error), dim=1)


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_error), dim=1)


def link_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    link_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.gravity_vec_w)
    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


def feet_orientation_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    body_link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = body_link_quat_w.shape[:2]

    gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_feet, -1)
    projected_gravity = quat_apply_inverse(body_link_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)).reshape(
        num_envs, num_feet, 3
    )
    orientation_error = torch.linalg.vector_norm(projected_gravity[:, :, :2], dim=-1)

    in_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    )

    return torch.sum(orientation_error * in_contact.float(), dim=1)


def feet_heading_align(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    trunk_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot forward direction misalignment with trunk heading when in contact.

    Computes the forward direction (+X axis in local frame) of the trunk and each foot,
    projects onto the world XY plane, and measures the cosine distance (1 - dot product).
    Returns 0 when perfectly aligned, up to 2 when pointing opposite directions.
    Only applies the penalty when the foot is in contact with the ground.
    """
    asset: Entity = env.scene[trunk_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    # Forward direction in local frame (+X)
    forward_local = torch.tensor([[1.0, 0.0, 0.0]], device=asset.data.body_link_quat_w.device)

    # Trunk forward direction projected to world XY
    trunk_quat = asset.data.body_link_quat_w[:, trunk_cfg.body_ids[0], :]  # (N, 4)
    trunk_forward = quat_apply(trunk_quat, forward_local.expand(trunk_quat.shape[0], -1))  # (N, 3)
    trunk_fwd_xy = trunk_forward[:, :2]  # (N, 2)
    trunk_fwd_xy = trunk_fwd_xy / (torch.linalg.vector_norm(trunk_fwd_xy, dim=-1, keepdim=True) + 1e-8)

    # Feet forward direction projected to world XY
    feet_quat = asset.data.body_link_quat_w[:, feet_cfg.body_ids, :]  # (N, num_feet, 4)
    num_envs, num_feet = feet_quat.shape[:2]
    feet_forward = quat_apply(
        feet_quat.reshape(-1, 4), forward_local.expand(num_envs * num_feet, -1)
    ).reshape(num_envs, num_feet, 3)
    feet_fwd_xy = feet_forward[:, :, :2]  # (N, num_feet, 2)
    feet_fwd_xy = feet_fwd_xy / (torch.linalg.vector_norm(feet_fwd_xy, dim=-1, keepdim=True) + 1e-8)

    # Cosine distance: 0 = aligned, 2 = opposite
    dot = torch.sum(trunk_fwd_xy.unsqueeze(1) * feet_fwd_xy, dim=-1)  # (N, num_feet)
    error = 1.0 - dot

    # Only penalize when foot is in ground contact
    in_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    )

    return torch.sum(error * in_contact.float(), dim=1)


def feet_at_plane(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    left_height_scanner_name: str,
    right_height_scanner_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.035,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    asset: Entity = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w

    contact_sensor: ContactSensor = env.scene[contact_sensor_name]
    is_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    )

    left_sensor: RayCastSensor = env.scene[left_height_scanner_name]
    right_sensor: RayCastSensor = env.scene[right_height_scanner_name]
    left_hit_z = left_sensor.data.hit_pos_w[..., 2]
    right_hit_z = right_sensor.data.hit_pos_w[..., 2]
    left_hit_z = torch.where(left_sensor.data.distances < 0.0, 0.0, left_hit_z)
    right_hit_z = torch.where(right_sensor.data.distances < 0.0, 0.0, right_hit_z)

    left_height = body_link_pos_w[:, asset_cfg.body_ids[0], 2].unsqueeze(-1)
    right_height = body_link_pos_w[:, asset_cfg.body_ids[1], 2].unsqueeze(-1)

    left_contact = is_contact[:, 0:1].float()
    right_contact = is_contact[:, 1:2].float()

    left_reward = torch.clamp(left_height - left_hit_z - height_offset, min=0.0, max=0.3) * left_contact
    right_reward = torch.clamp(right_height - right_hit_z - height_offset, min=0.0, max=0.3) * right_contact
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def feet_close_xy_gauss(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    asset: Entity = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]

    left_foot_xy = body_link_pos_w[:, 0, :2]
    right_foot_xy = body_link_pos_w[:, 1, :2]
    heading_w = asset.data.heading_w

    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    left_y = -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1]
    right_y = -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1]
    feet_distance_y = torch.abs(left_y - right_y)

    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def volume_points_penetration(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    tolerance: float = 0.0,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    penetration = sensor.data.penetration_offset
    points_vel = sensor.data.points_vel_w

    penetration_depth = torch.linalg.vector_norm(penetration.reshape(env.num_envs, -1, 3), dim=-1)
    in_obstacle = (penetration_depth > tolerance).float()
    points_vel_norm = torch.linalg.vector_norm(points_vel.reshape(env.num_envs, -1, 3), dim=-1)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth
    return torch.sum(velocity_times_penetration, dim=-1)


def motors_power_square(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = True,
    normalize_by_num_joints: bool = False,
) -> torch.Tensor:
    return _motors_power_square_general(
        env=env,
        asset_cfg=asset_cfg,
        normalize_by_stiffness=normalize_by_stiffness,
        normalize_by_num_joints=normalize_by_num_joints,
    )


def joint_vel_limits(
    env: ManagerBasedRlEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities if they cross soft limits.

    Per-joint velocity limits are read from actuator cfg metadata
    (``velocity_limit``). Excess is clipped to [0, 1] rad/s per joint.
    """
    asset: Entity = env.scene[asset_cfg.name]

    joint_vel_limits = torch.zeros_like(asset.data.joint_vel)
    for actuator in asset.actuators:
        base_actuator = actuator.base_actuator
        target_ids = base_actuator.target_ids
        joint_vel_limits[:, target_ids] = float(base_actuator.cfg.velocity_limit)

    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # Clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


def applied_torque_limits_by_ratio(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.8,
) -> torch.Tensor:
    """Penalize when the applied torque exceeds a ratio of torque limits."""
    return _applied_torque_limits_by_ratio_general(
        env=env,
        asset_cfg=asset_cfg,
        limit_ratio=limit_ratio,
    )


def undesired_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    is_contact = torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0] > threshold
    return torch.sum(is_contact.float(), dim=1)


# ---------------------------------------------------------------------------
# SSR Imagined Foothold Guidance — contact-time portion
#
# This reward implements r^f = exp(-(sum_i rho_i)^2 / sigma_f^2) where rho is
# the "support deficiency" of a sole patch centered at each foot. For stance
# feet, rho is evaluated at the current contact position. For swing feet, the
# current foot position is used as a contact-time approximation (the
# model-free portion of the SSR ablation "NoImgn"). The imagination-based
# expected rho under the imagined Gaussian distribution is computed in the
# algorithm side (instinct_rl ImaginatorAlgoMixin) and injected via
# compute_auxiliary_reward.
#
# Reference: Yu et al., "SSR: Scaling Surefooted and Symmetric Humanoid
# Traversal to the Open World", 2026, Section 3.2.
# ---------------------------------------------------------------------------


def _compute_support_deficiency(
    scanner: RayCastSensor,
    root_pos_w: torch.Tensor,
    support_threshold: float,
) -> torch.Tensor:
    """Compute per-foot support deficiency rho from a sole-patch RayCastSensor.

    Args:
        scanner: RayCastSensor sampling the sole patch under one foot.
          ``scanner.data.hit_pos_w[..., 2]`` is (E, N) world-z of hit points.
        root_pos_w: (E, 3) root link world position. Used to impute missed rays.
        support_threshold: A ray sample is considered unsupported if its
          height is more than this far below the sole height (max of samples).

    Returns:
        Tensor of shape (E,) — unsupported fraction in [0, 1].
    """
    hit_z = scanner.data.hit_pos_w[..., 2]  # (E, N)
    distances = scanner.data.distances  # (E, N)
    miss = distances < 0.0
    # Treat missed rays as ground level at the root (i.e. neither supported nor
    # a hard obstacle). Imputing root-z prevents spurious cliffs at scan edges.
    hit_z = torch.where(
        miss,
        root_pos_w[:, 2:1].expand_as(hit_z),
        hit_z,
    )
    # Sole height: the highest sample inside the patch (paper Eq. for h_f).
    sole_height = hit_z.max(dim=-1).values  # (E,)
    unsupported = (sole_height.unsqueeze(-1) - hit_z) > support_threshold  # (E, N)
    rho = unsupported.float().mean(dim=-1)  # (E,)
    # All-miss scans should not be penalized (avoid spurious cliffs).
    rho = torch.where(
        miss.all(dim=-1),
        torch.zeros_like(rho),
        rho,
    )
    return rho


def _is_slope_terrain(
    env: ManagerBasedRlEnv, slope_terrain_name: str
) -> torch.Tensor:
    """Return (num_envs,) bool: True for envs on the named slope sub-terrain.

    Resolves the sub-terrain name to an index using the terrain generator's
    sub_terrains dict (insertion order). Returns all-False if the name is
    not found or no terrain generator is attached.
    """
    terrain = env.scene.terrain
    if not hasattr(terrain, "terrain_generator") or terrain.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sub_terrains = terrain.terrain_generator.cfg.sub_terrains
    sub_names = list(sub_terrains.keys())
    if slope_terrain_name not in sub_names:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    slope_idx = sub_names.index(slope_terrain_name)
    return terrain.terrain_types == slope_idx


def foothold_support_deficiency(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    left_scanner_name: str,
    right_scanner_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.0625,
    contact_force_threshold: float = 1.0,
    support_threshold: float = 0.03,
    slope_terrain_name: str = "hf_pyramid_slope_inv",
    min_terrain_level: int = 3,
) -> torch.Tensor:
    """SSR foothold support deficiency reward (contact-time portion).

    Computes r^f = exp(-(rho_l + rho_r)^2 / sigma_f^2) where each rho is the
    unsupported fraction within a sole-patch ray scan centered at the foot.

    Behavior:
      - On slope terrains (e.g. ``hf_pyramid_slope_inv``), rho is set to 0 to
        avoid penalizing inclined contacts (paper footnote in Table 6).
      - Below ``min_terrain_level`` (curriculum gating), the reward is 1.0 to
        avoid noisy guidance when gaits are unstable early in training.
      - Both stance and swing feet use their current scanner output (swing uses
        current foot position as a contact-time approximation). The imagined
        Gaussian-sample-based swing guidance is added separately by the
        algorithm mixin.

    Args:
        env: The RL environment.
        contact_sensor_name: Name of the ContactSensor for feet.
        left_scanner_name: Name of the left sole-patch RayCastSensor.
        right_scanner_name: Name of the right sole-patch RayCastSensor.
        asset_cfg: Entity config resolving the robot entity (unused for
          body_ids since scanners are pre-attached, but kept for consistency).
        std: sigma_f in the Gaussian kernel.
        contact_force_threshold: Force threshold for contact detection.
        support_threshold: Height drop below sole-height to count as unsupported.
        slope_terrain_name: Sub-terrain name on which rho is forced to 0.
        min_terrain_level: Curriculum level below which reward = 1.0.
    """
    del contact_force_threshold  # scanner-based rho does not use contact threshold
    asset: Entity = env.scene[asset_cfg.name]
    left_scanner: RayCastSensor = env.scene[left_scanner_name]
    right_scanner: RayCastSensor = env.scene[right_scanner_name]
    root_pos_w = asset.data.root_link_pos_w  # (E, 3)

    # Per-foot support deficiency.
    rho_left = _compute_support_deficiency(
        left_scanner, root_pos_w, support_threshold
    )
    rho_right = _compute_support_deficiency(
        right_scanner, root_pos_w, support_threshold
    )

    rho_sum = rho_left + rho_right  # (E,)

    # Curriculum gating: reward = 1.0 below min_terrain_level (no constraint).
    terrain = env.scene.terrain
    terrain_levels = terrain.terrain_levels  # (E,)
    level_ok = terrain_levels >= min_terrain_level  # (E,) bool

    # Slope terrain exception: set rho_sum = 0 (=> reward = 1.0).
    is_slope = _is_slope_terrain(env, slope_terrain_name)  # (E,) bool

    # Effective rho_sum: zero where slope or below-curriculum.
    rho_effective = torch.where(
        level_ok & ~is_slope,
        rho_sum,
        torch.zeros_like(rho_sum),
    )

    reward = torch.exp(-(rho_effective**2) / (std**2))
    return reward
