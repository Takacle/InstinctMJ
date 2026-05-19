"""V11 humanoid robot asset configuration for mjlab/MuJoCo.

V11 full-body variant uses v11_v4.xml with 29 drivable DOF:
- Legs (0-11):  hip_pitch/roll/yaw, knee, ankle_pitch/roll × 2
- Waist (12-14): waist_yaw, waist_roll, waist_pitch
- Left arm (15-21): shoulder_pitch/roll/yaw, elbow, wrist_roll/yaw/pitch
- Right arm (22-28): shoulder_pitch/roll/yaw, elbow, wrist_roll/yaw/pitch

Root link: base_link

MJCF joint order (v11_v4.xml tree traversal):
 0: left_hip_pitch_joint        12: waist_yaw_joint
 1: left_hip_roll_joint         13: waist_roll_joint
 2: left_hip_yaw_joint          14: waist_pitch_joint
 3: left_knee_joint             15: left_shoulder_pitch_joint
 4: left_ankle_pitch_joint      16: left_shoulder_roll_joint
 5: left_ankle_roll_joint       17: left_shoulder_yaw_joint
 6: right_hip_pitch_joint       18: left_elbow_joint
 7: right_hip_roll_joint        19: left_wrist_roll_joint
 8: right_hip_yaw_joint         20: left_wrist_yaw_joint
 9: right_knee_joint            21: left_wrist_pitch_joint
10: right_ankle_pitch_joint     22: right_shoulder_pitch_joint
11: right_ankle_roll_joint      23: right_shoulder_roll_joint
                                24: right_shoulder_yaw_joint
                                25: right_elbow_joint
                                26: right_wrist_roll_joint
                                27: right_wrist_yaw_joint
                                28: right_wrist_pitch_joint
"""

from __future__ import annotations

import copy
import os

import mujoco
from mjlab.actuator import ActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

from instinct_mj.actuators import DelayedInstinctActuatorCfg, InstinctActuatorCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

V11_XML_PATH: str = os.path.join(__file_dir__, "resources/v11/xml/v11.xml")
V11_MESHES_DIR: str = os.path.join(__file_dir__, "resources/v11/meshes")


def get_v11_assets(meshdir: str | None) -> dict[str, bytes]:
    """Load local V11 mesh assets keyed with MuJoCo meshdir prefix."""
    assets: dict[str, bytes] = {}
    normalized_meshdir = meshdir.rstrip("/") if meshdir else None
    update_assets(assets, V11_MESHES_DIR, normalized_meshdir)
    return assets


def get_v11_spec() -> mujoco.MjSpec:
    """Load the local v11_v4.xml as MjSpec."""
    spec = mujoco.MjSpec.from_file(V11_XML_PATH)
    spec.assets = get_v11_assets(spec.meshdir)
    return spec


# ============================================================================
# Motor constants (same motor types as G1)
# ============================================================================

ARMATURE_8112 = 0.0484866
ARMATURE_10020 = 0.069947
ARMATURE_4310 = 0.0247477
ARMATURE_6408 = 0.0393546

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_8112 = ARMATURE_8112 * NATURAL_FREQ**2
STIFFNESS_10020 = ARMATURE_10020 * NATURAL_FREQ**2
STIFFNESS_4310 = ARMATURE_4310 * NATURAL_FREQ**2
STIFFNESS_6408 = ARMATURE_6408 * NATURAL_FREQ**2

DAMPING_8112 = 2.0 * DAMPING_RATIO * ARMATURE_8112 * NATURAL_FREQ
DAMPING_10020 = 2.0 * DAMPING_RATIO * ARMATURE_10020 * NATURAL_FREQ
DAMPING_4310 = 2.0 * DAMPING_RATIO * ARMATURE_4310 * NATURAL_FREQ
DAMPING_6408 = 2.0 * DAMPING_RATIO * ARMATURE_6408 * NATURAL_FREQ


# ============================================================================
# BeyondMimic actuator configurations for V11 (12-DOF leg joints)
#
# Motor assignment and torque limits aligned with dros-motor SDK:
#   - hip_pitch/roll, knee: EC-A10020-P1-12  (SDK TAU150, peak 150 Nm)
#   - hip_yaw:              EC-A8112-P1-18   (SDK TAU90,  peak  90 Nm)
#   - ankle_pitch/roll:     EC-A4310-P2-36   (SDK TAU30,  peak  30 Nm)
#
# effort_limit = 0.85 × SDK peak (85% derating for training margin)
# ============================================================================

BEYONDMIMIC_V11_LEGS_HIP_PITCH_ROLL_KNEE = InstinctActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
    effort_limit=127.5,  # 0.85 × 150
    velocity_limit=14.66,
    stiffness=STIFFNESS_10020,
    damping=DAMPING_10020,
    armature=ARMATURE_10020,
)
BEYONDMIMIC_V11_LEGS_HIP_YAW = InstinctActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint",),
    effort_limit=76.5,  # 0.85 × 90
    velocity_limit=14.66,
    stiffness=STIFFNESS_8112,
    damping=5,
    armature=ARMATURE_8112,
)
BEYONDMIMIC_V11_FEET = InstinctActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    effort_limit=25.5,  # 0.85 × 30 (SDK TAU30, prev was 30.6 based on 36 Nm spec)
    velocity_limit=9.32,
    stiffness=STIFFNESS_4310,
    damping=5,
    armature=ARMATURE_4310,
)

beyondmimic_v11_locomotion_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    BEYONDMIMIC_V11_LEGS_HIP_PITCH_ROLL_KNEE,
    BEYONDMIMIC_V11_LEGS_HIP_YAW,
    BEYONDMIMIC_V11_FEET,
)

# Delayed variants for domain randomisation (communication latency simulation)
BEYONDMIMIC_V11_DELAYED_LEGS_HIP_PITCH_ROLL_KNEE = DelayedInstinctActuatorCfg(
    base_cfg=BEYONDMIMIC_V11_LEGS_HIP_PITCH_ROLL_KNEE,
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=2,
)
BEYONDMIMIC_V11_DELAYED_LEGS_HIP_YAW = DelayedInstinctActuatorCfg(
    base_cfg=BEYONDMIMIC_V11_LEGS_HIP_YAW,
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=2,
)
BEYONDMIMIC_V11_DELAYED_FEET = DelayedInstinctActuatorCfg(
    base_cfg=BEYONDMIMIC_V11_FEET,
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=2,
)

beyondmimic_v11_locomotion_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    BEYONDMIMIC_V11_DELAYED_LEGS_HIP_PITCH_ROLL_KNEE,
    BEYONDMIMIC_V11_DELAYED_LEGS_HIP_YAW,
    BEYONDMIMIC_V11_DELAYED_FEET,
)


# ============================================================================
# BeyondMimic actuator configurations for V11 upper body (17 DOF)
#
# Torque limits aligned with dros-motor SDK:
#   waist_yaw:              EC-A8112-P1-18   (SDK TAU90,  peak  90 Nm)
#   waist_roll/pitch:       EC-A6408-P2-25   (SDK TAU60,  peak  60 Nm)
#   shoulder_pitch/roll/yaw, elbow: TK motor  (SDK float2uint, peak 46.54 Nm)
#   wrist_roll:             TK motor          (SDK clamp ±200 int16, peak 19.5 Nm)
#   wrist_pitch/yaw:        TK motor          (SDK clamp ±300 int16, peak 22.0 Nm)
#
# effort_limit = 0.85 × SDK peak (85% derating for training margin)
#
# NOTE: 17-series and SHD11 armature values not yet available;
#       arms temporarily use 10020, wrists temporarily use 6408.
# ============================================================================

BEYONDMIMIC_V11_WAIST_YAW = InstinctActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    effort_limit=76.5,  # 0.85 × 90
    velocity_limit=14.66,
    stiffness=STIFFNESS_8112,
    damping=5,
    armature=ARMATURE_8112,
)
BEYONDMIMIC_V11_WAIST_ROLL_PITCH = InstinctActuatorCfg(
    target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
    effort_limit=51.0,  # 0.85 × 60
    velocity_limit=14.66,
    stiffness=STIFFNESS_6408,
    damping=5,
    armature=ARMATURE_6408,
)
BEYONDMIMIC_V11_ARMS = InstinctActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
    ),
    effort_limit=39.5,  # 0.85 × 46.54 (SDK TK shoulder/elbow effective max)
    velocity_limit=14.66,
    stiffness=STIFFNESS_10020,  # TODO: replace with 17-series armature when available
    damping=DAMPING_10020,
    armature=ARMATURE_10020,
)
BEYONDMIMIC_V11_WRISTS = InstinctActuatorCfg(
    target_names_expr=(".*_wrist_roll_joint", ".*_wrist_yaw_joint", ".*_wrist_pitch_joint"),
    effort_limit=16.6,  # 0.85 × min(19.5, 22.0) = 0.85 × 19.5 ≈ 16.6 (use min for uniform scale)
    velocity_limit=22.0,
    stiffness=STIFFNESS_6408,  # TODO: replace with SHD11 armature when available
    damping=DAMPING_6408,
    armature=ARMATURE_6408,
)

beyondmimic_v11_wholebody_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    BEYONDMIMIC_V11_LEGS_HIP_PITCH_ROLL_KNEE,
    BEYONDMIMIC_V11_LEGS_HIP_YAW,
    BEYONDMIMIC_V11_FEET,
    BEYONDMIMIC_V11_WAIST_YAW,
    BEYONDMIMIC_V11_WAIST_ROLL_PITCH,
    BEYONDMIMIC_V11_ARMS,
    BEYONDMIMIC_V11_WRISTS,
)

BEYONDMIMIC_V11_DELAYED_WAIST_YAW = DelayedInstinctActuatorCfg(
    base_cfg=BEYONDMIMIC_V11_WAIST_YAW, delay_target="position", delay_min_lag=0, delay_max_lag=2,
)
BEYONDMIMIC_V11_DELAYED_WAIST_ROLL_PITCH = DelayedInstinctActuatorCfg(
    base_cfg=BEYONDMIMIC_V11_WAIST_ROLL_PITCH, delay_target="position", delay_min_lag=0, delay_max_lag=2,
)
BEYONDMIMIC_V11_DELAYED_ARMS = DelayedInstinctActuatorCfg(
    base_cfg=BEYONDMIMIC_V11_ARMS, delay_target="position", delay_min_lag=0, delay_max_lag=2,
)
BEYONDMIMIC_V11_DELAYED_WRISTS = DelayedInstinctActuatorCfg(
    base_cfg=BEYONDMIMIC_V11_WRISTS, delay_target="position", delay_min_lag=0, delay_max_lag=2,
)

beyondmimic_v11_wholebody_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    BEYONDMIMIC_V11_DELAYED_LEGS_HIP_PITCH_ROLL_KNEE,
    BEYONDMIMIC_V11_DELAYED_LEGS_HIP_YAW,
    BEYONDMIMIC_V11_DELAYED_FEET,
    BEYONDMIMIC_V11_DELAYED_WAIST_YAW,
    BEYONDMIMIC_V11_DELAYED_WAIST_ROLL_PITCH,
    BEYONDMIMIC_V11_DELAYED_ARMS,
    BEYONDMIMIC_V11_DELAYED_WRISTS,
)


# ============================================================================
# Action scale: 0.25 * effort / stiffness  (BeyondMimic formula)
# ============================================================================

beyondmimic_action_scale: dict[str, float] = {}
for _act in beyondmimic_v11_wholebody_actuator_cfgs:
    _effort = _act.effort_limit
    _stiffness = _act.stiffness
    if _effort is None or _stiffness == 0.0:
        continue
    for _jname in _act.target_names_expr:
        beyondmimic_action_scale[_jname] = 0.25 * _effort / _stiffness


# ============================================================================
# Initial state (popsicle / crouched pose)
# ============================================================================

_POPSICLE_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.9),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
    },
    joint_vel={".*": 0.0},
)

_POPSICLE_INIT_STATE_29DOF = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.9),
    joint_pos={
        ".*_hip_pitch_joint": -0.3465,
        ".*_hip_roll_joint": -0.0306,
        ".*_hip_yaw_joint": -0.2425,
        ".*_knee_joint": 0.5,
        ".*_ankle_pitch_joint": -0.25,
        # Upper body default pose (natural standing)
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        ".*_elbow_joint": 0.6,
    },
    joint_vel={".*": 0.0},
)


# ============================================================================
# Articulation entity configs
# ============================================================================

V11_LOCOMOTION_POPSICLE_CFG = EntityCfg(
    init_state=copy.deepcopy(_POPSICLE_INIT_STATE),
    spec_fn=get_v11_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in beyondmimic_v11_locomotion_actuator_cfgs),
        soft_joint_pos_limit_factor=0.9,
    ),
)

V11_29DOF_POPSICLE_CFG = EntityCfg(
    init_state=copy.deepcopy(_POPSICLE_INIT_STATE_29DOF),
    spec_fn=get_v11_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in beyondmimic_v11_wholebody_actuator_cfgs),
        soft_joint_pos_limit_factor=0.9,
    ),
)


# ============================================================================
# Symmetric augmentation (MJCF native joint order, 12 DOF)
#
# MJCF order:
#  0: left_hip_pitch   ↔  6: right_hip_pitch
#  1: left_hip_roll    ↔  7: right_hip_roll
#  2: left_hip_yaw     ↔  8: right_hip_yaw
#  3: left_knee        ↔  9: right_knee
#  4: left_ankle_pitch ↔ 10: right_ankle_pitch
#  5: left_ankle_roll  ↔ 11: right_ankle_roll
# ============================================================================

V11_12Dof_symmetric_augmentation_joint_mapping = [
    6,   # 0: left_hip_pitch  → right_hip_pitch
    7,   # 1: left_hip_roll   → right_hip_roll
    8,   # 2: left_hip_yaw    → right_hip_yaw
    9,   # 3: left_knee       → right_knee
    10,  # 4: left_ankle_pitch → right_ankle_pitch
    11,  # 5: left_ankle_roll  → right_ankle_roll
    0,   # 6: right_hip_pitch → left_hip_pitch
    1,   # 7: right_hip_roll  → left_hip_roll
    2,   # 8: right_hip_yaw   → left_hip_yaw
    3,   # 9: right_knee      → left_knee
    4,   # 10: right_ankle_pitch → left_ankle_pitch
    5,   # 11: right_ankle_roll  → left_ankle_roll
]

V11_12Dof_symmetric_augmentation_joint_reverse_buf = [
    1,   # 0: left_hip_pitch   (pitch: same)
    -1,  # 1: left_hip_roll    (roll: reversed)
    -1,  # 2: left_hip_yaw     (yaw: reversed)
    1,   # 3: left_knee        (pitch: same)
    1,   # 4: left_ankle_pitch (pitch: same)
    -1,  # 5: left_ankle_roll  (roll: reversed)
    1,   # 6: right_hip_pitch  (pitch: same)
    -1,  # 7: right_hip_roll   (roll: reversed)
    -1,  # 8: right_hip_yaw    (yaw: reversed)
    1,   # 9: right_knee       (pitch: same)
    1,   # 10: right_ankle_pitch (pitch: same)
    -1,  # 11: right_ankle_roll  (roll: reversed)
]


# ============================================================================
# Symmetric augmentation (MJCF native joint order, 29 DOF)
#
# Joint groups:
#  legs (0-11):     left leg ↔ right leg
#  waist (12-14):   self-symmetric (midline joints)
#  left arm (15-21) ↔ right arm (22-28)
# ============================================================================

V11_29Dof_symmetric_augmentation_joint_mapping = [
    6, 7, 8, 9, 10, 11,          # 0-5:  left leg  → right leg
    0, 1, 2, 3, 4, 5,            # 6-11: right leg → left leg
    12, 13, 14,                   # 12-14: waist    → waist (self)
    22, 23, 24, 25, 26, 27, 28,  # 15-21: left arm → right arm
    15, 16, 17, 18, 19, 20, 21,  # 22-28: right arm → left arm
]

V11_29Dof_symmetric_augmentation_joint_reverse_buf = [
    1, -1, -1, 1, 1, -1,          # left leg:  pitch same, roll/yaw rev, knee same, ankle_pitch same, ankle_roll rev
    1, -1, -1, 1, 1, -1,          # right leg: same pattern
    -1, -1, 1,                     # waist: yaw rev, roll rev, pitch same
    1, -1, -1, 1, -1, -1, 1,      # left arm:  pitch same, roll/yaw rev, elbow same, wrist_roll/yaw rev, wrist_pitch same
    1, -1, -1, 1, -1, -1, 1,      # right arm: same pattern
]


__all__ = [
    "ARMATURE_6408",
    "ARMATURE_8112",
    "ARMATURE_10020",
    "ARMATURE_4310",
    "BEYONDMIMIC_V11_ARMS",
    "BEYONDMIMIC_V11_DELAYED_ARMS",
    "BEYONDMIMIC_V11_DELAYED_WAIST_YAW",
    "BEYONDMIMIC_V11_DELAYED_WAIST_ROLL_PITCH",
    "BEYONDMIMIC_V11_DELAYED_WRISTS",
    "BEYONDMIMIC_V11_WAIST_YAW",
    "BEYONDMIMIC_V11_WAIST_ROLL_PITCH",
    "BEYONDMIMIC_V11_WRISTS",
    "DAMPING_6408",
    "DAMPING_8112",
    "DAMPING_10020",
    "DAMPING_4310",
    "DAMPING_RATIO",
    "NATURAL_FREQ",
    "STIFFNESS_6408",
    "STIFFNESS_8112",
    "STIFFNESS_10020",
    "STIFFNESS_4310",
    "V11_12Dof_symmetric_augmentation_joint_mapping",
    "V11_12Dof_symmetric_augmentation_joint_reverse_buf",
    "V11_29Dof_symmetric_augmentation_joint_mapping",
    "V11_29Dof_symmetric_augmentation_joint_reverse_buf",
    "V11_LOCOMOTION_POPSICLE_CFG",
    "V11_29DOF_POPSICLE_CFG",
    "V11_MESHES_DIR",
    "V11_XML_PATH",
    "beyondmimic_action_scale",
    "beyondmimic_v11_locomotion_actuator_cfgs",
    "beyondmimic_v11_locomotion_delayed_actuator_cfgs",
    "beyondmimic_v11_wholebody_actuator_cfgs",
    "beyondmimic_v11_wholebody_delayed_actuator_cfgs",
    "get_v11_assets",
    "get_v11_spec",
]
