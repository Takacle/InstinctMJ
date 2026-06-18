"""V11 SSR parkour task config.

Extends V11 parkour with the SSR (Surefooted and Symmetric tRaversal)
Imagined Foothold Guidance reward. This config is independent of the
baseline ``v11_parkour_target_amp_cfg``: it imports the base factory and
adds new sensors, observation groups, and rewards on top.

Reference: Yu et al., "SSR: Scaling Surefooted and Symmetric Humanoid
Traversal to the Open World", 2026.
"""

from __future__ import annotations

import math

import mjlab.envs.mdp as envs_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
)
from mjlab.sensor import (
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.utils.noise import UniformNoiseCfg

import instinct_mj.envs.mdp as instinct_envs_mdp
import instinct_mj.tasks.parkour.mdp as parkour_mdp
from instinct_mj.tasks.parkour.config.v11.v11_parkour_target_amp_cfg import (
    instinct_v11_parkour_amp_env_cfg,
)


def instinct_v11_ssr_env_cfg(
    *,
    play: bool = False,
    foothold_weight: float = 1.0,
    foothold_std: float = 0.0625,
    min_terrain_level: int = 3,
    support_threshold: float = 0.03,
) -> ManagerBasedRlEnvCfg:
    """Build the V11 SSR parkour environment configuration.

    Args:
      play: If True, apply play-mode overrides inherited from V11 parkour.
      foothold_weight: Reward weight for the contact-time foothold guidance
        reward. The imagined-distribution portion is added separately on the
        algorithm side.
      foothold_std: sigma_f in the Gaussian kernel of the foothold reward.
      min_terrain_level: Curriculum level below which the foothold reward
        returns 1.0 (no constraint).
      support_threshold: Height drop below sole-height to count as unsupported.

    Returns:
      A ``ManagerBasedRlEnvCfg`` instance with SSR foothold guidance enabled.
    """
    cfg = instinct_v11_parkour_amp_env_cfg(play=play)

    # ------------------------------------------------------------------
    # Add SSR-specific sensors: sole-patch RayCastScanners under each foot.
    # ------------------------------------------------------------------
    # The sole-patch size is chosen to match V11's ``leg_volume_points``
    # extent: x in [-0.06, 0.17] (≈0.23 m) and y in [-0.04, 0.04] (0.08 m).
    # Sampling at 0.025 m resolution yields ~10 x 4 rays per foot, mirroring
    # the SSR paper's 22.5 cm x 10 cm sole patch sampled at 2.5 cm.
    existing_sensors = list(cfg.scene.sensors)
    # Remove any existing scanners with the same names (defensive).
    existing_sensors = [
        s for s in existing_sensors if s.name not in
        ("left_foothold_scanner", "right_foothold_scanner")
    ]
    existing_sensors.extend([
        RayCastSensorCfg(
            name="left_foothold_scanner",
            frame=ObjRef(type="body", name="left_ankle_roll_link", entity="robot"),
            pattern=GridPatternCfg(resolution=0.025, size=(0.23, 0.08)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="right_foothold_scanner",
            frame=ObjRef(type="body", name="right_ankle_roll_link", entity="robot"),
            pattern=GridPatternCfg(resolution=0.025, size=(0.23, 0.08)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
    ])
    cfg.scene.sensors = tuple(existing_sensors)

    # ------------------------------------------------------------------
    # Add the contact-time foothold support deficiency reward.
    # ------------------------------------------------------------------
    cfg.rewards["foothold_support_deficiency"] = RewardTermCfg(
        func=parkour_mdp.foothold_support_deficiency,
        weight=foothold_weight,
        params={
            "contact_sensor_name": "contact_forces",
            "left_scanner_name": "left_foothold_scanner",
            "right_scanner_name": "right_foothold_scanner",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": foothold_std,
            "support_threshold": support_threshold,
            "slope_terrain_name": "hf_pyramid_slope_inv",
            "min_terrain_level": min_terrain_level,
        },
    )

    # ------------------------------------------------------------------
    # Add observation groups for the foothold imagination model.
    #
    # ``foothold_privilege``: input to the imagination model + used by the
    #   algorithm mixin to compute the imagined auxiliary reward. Contains
    #   sole heights, foot state, contact flags, terrain level/type. Added
    #   to BOTH the policy (for the imagination model) and the critic (so
    #   the model sees it during training).
    # ``foothold_target``: supervision target (future contact xy in base
    #   frame). Only the imagination algorithm reads this; it is added to
    #   the critic observation group as a side channel.
    # ------------------------------------------------------------------
    # Foothold privilege obs group.
    privilege_terms = {
        "foothold_privilege": ObservationTermCfg(
            func=parkour_mdp.foothold_privilege_obs,
            params={
                "contact_sensor_name": "contact_forces",
                "left_scanner_name": "left_foothold_scanner",
                "right_scanner_name": "right_foothold_scanner",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "root_body_name": "base_link",
                "max_delay_steps": 50,
            },
            noise=None,
        ),
    }
    cfg.observations["foothold_privilege"] = ObservationGroupCfg(
        terms=privilege_terms,
        concatenate_terms=True,
        enable_corruption=False,
    )

    # Foothold target obs group (supervision signal).
    target_terms = {
        "foothold_target": ObservationTermCfg(
            func=parkour_mdp.foothold_target_obs,
            params={"max_delay_steps": 50},
            noise=None,
        ),
    }
    cfg.observations["foothold_target"] = ObservationGroupCfg(
        terms=target_terms,
        concatenate_terms=True,
        enable_corruption=False,
    )

    return cfg


def instinct_v11_ssr_final_cfg(
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the final V11 SSR env config (factory wrapper).

    Mirrors the ``instinct_v11_parkour_amp_final_cfg`` pattern but does not
    re-apply viewer overrides since ``instinct_v11_parkour_amp_env_cfg`` is
    already called with ``play`` by ``instinct_v11_ssr_env_cfg``.
    """
    return instinct_v11_ssr_env_cfg(play=play)
