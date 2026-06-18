"""U20 humanoid robot asset configuration for mjlab/MuJoCo.

U20 full-body variant uses u20_popsicle.xml with 22 drivable DOF:
- Legs (0-7):   left_hip_pitch/roll/yaw, left_knee,
                 right_hip_pitch/roll/yaw, right_knee
- Waist (8-9):  waist_yaw, waist_roll
- Left arm (10-15):  shoulder_roll/pitch/yaw, elbow, wrist_pitch, wrist_yaw
- Right arm (16-21): shoulder_roll/pitch/yaw, elbow, wrist_pitch, wrist_yaw

Root link: base_link

MJCF joint order (u20_popsicle.xml tree traversal):
  0: left_hip_pitch_joint        12: left_shoulder_yaw_joint
  1: left_hip_roll_joint         13: left_elbow_joint
  2: left_hip_yaw_joint          14: left_wrist_pitch_joint
  3: left_knee_joint             15: left_wrist_yaw_joint
  4: right_hip_pitch_joint       16: right_shoulder_roll_joint
  5: right_hip_roll_joint        17: right_shoulder_pitch_joint
  6: right_hip_yaw_joint         18: right_shoulder_yaw_joint
  7: right_knee_joint            19: right_elbow_joint
  8: waist_yaw_joint             20: right_wrist_pitch_joint
  9: waist_roll_joint            21: right_wrist_yaw_joint
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

U20_XML_PATH: str = os.path.join(__file_dir__, "resources/u20/xml/u20_popsicle_ring.xml")
U20_MESHES_DIR: str = os.path.join(__file_dir__, "resources/u20/meshes")


def get_u20_assets(meshdir: str | None) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    normalized_meshdir = meshdir.rstrip("/") if meshdir else None
    update_assets(assets, U20_MESHES_DIR, normalized_meshdir)
    return assets


def get_u20_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(U20_XML_PATH)
    spec.assets = get_u20_assets(spec.meshdir)
    return spec


# ============================================================================
# Motor constants
# Designed for 10 Hz natural frequency, damping ratio 2.0 (same methodology
# as V11).  With sim timestep 0.005 this gives omega*dt = 0.314 (stable).
# Armature values chosen to approximate V11 motor tiers: legs > waist > arms.
# ============================================================================

ARMATURE_U20_LEG = 0.05
ARMATURE_U20_WAIST = 0.04
ARMATURE_U20_ARM = 0.025

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_U20_LEG = ARMATURE_U20_LEG * NATURAL_FREQ ** 2
STIFFNESS_U20_WAIST = ARMATURE_U20_WAIST * NATURAL_FREQ ** 2
STIFFNESS_U20_ARM = ARMATURE_U20_ARM * NATURAL_FREQ ** 2

DAMPING_U20_LEG = 2.0 * DAMPING_RATIO * ARMATURE_U20_LEG * NATURAL_FREQ
DAMPING_U20_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_U20_WAIST * NATURAL_FREQ
DAMPING_U20_ARM = 2.0 * DAMPING_RATIO * ARMATURE_U20_ARM * NATURAL_FREQ


# ============================================================================
# Actuator configurations for U20 (22 DOF)
#
# XML actuatorfrcrange is 200 Nm for all joints, but effort limits are graded
# by functional requirement to prevent over-powered arms from destabilising
# the robot during locomotion.
#   - legs (hip_yaw/pitch/roll + knee):  150 Nm (main load-bearing)
#   - waist (yaw + roll):                100 Nm (moderate)
#   - arms (shoulder/elbow/wrist):        40 Nm (low, matches V11 arm tier)
# ============================================================================

U20_LEGS = InstinctActuatorCfg(
    target_names_expr=(
        ".*_hip_yaw_joint",
        ".*_hip_pitch_joint",
        ".*_hip_roll_joint",
        ".*_knee_joint",
    ),
    effort_limit=150.0,
    velocity_limit=14.66,
    stiffness=STIFFNESS_U20_LEG,
    damping=DAMPING_U20_LEG,
    armature=ARMATURE_U20_LEG,
)

U20_WAIST = InstinctActuatorCfg(
    target_names_expr=("waist_yaw_joint", "waist_roll_joint"),
    effort_limit=100.0,
    velocity_limit=14.66,
    stiffness=STIFFNESS_U20_WAIST,
    damping=DAMPING_U20_WAIST,
    armature=ARMATURE_U20_WAIST,
)

U20_ARMS = InstinctActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",
    ),
    effort_limit=40.0,
    velocity_limit=22.0,
    stiffness=STIFFNESS_U20_ARM,
    damping=DAMPING_U20_ARM,
    armature=ARMATURE_U20_ARM,
)

u20_wholebody_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    U20_LEGS,
    U20_WAIST,
    U20_ARMS,
)

# Delayed variants for domain randomisation
U20_DELAYED_LEGS = DelayedInstinctActuatorCfg(
    base_cfg=U20_LEGS,
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=2,
)
U20_DELAYED_WAIST = DelayedInstinctActuatorCfg(
    base_cfg=U20_WAIST,
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=2,
)
U20_DELAYED_ARMS = DelayedInstinctActuatorCfg(
    base_cfg=U20_ARMS,
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=2,
)

u20_wholebody_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    U20_DELAYED_LEGS,
    U20_DELAYED_WAIST,
    U20_DELAYED_ARMS,
)


# ============================================================================
# Action scale: 0.25 * effort / stiffness  (BeyondMimic formula)
# ============================================================================

u20_action_scale: dict[str, float] = {}
for _act in u20_wholebody_actuator_cfgs:
    _effort = _act.effort_limit
    _stiffness = _act.stiffness
    if _effort is None or _stiffness == 0.0:
        continue
    for _jname in _act.target_names_expr:
        u20_action_scale[_jname] = 0.25 * _effort / _stiffness


# ============================================================================
# Initial state (mild crouch)
# hip_pitch=-0.15 + knee=0.30 gives a stable bent-knee stance.  The reset
# event adds +/-0.15 noise, so knee stays in [0.15, 0.45] (always bent).
# pos z=0.92 is estimated from leg geometry with this pose; verify in play.
# hip_roll / shoulder_roll are set to safe mid-range values to avoid
# boundary clamping during reset randomisation.
# ============================================================================

_U20_ZERO_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.92),
    joint_pos={
        "left_hip_pitch_joint": -0.15,
        "right_hip_pitch_joint": -0.15,
        "left_hip_roll_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.30,
        "right_knee_joint": 0.30,
        "left_shoulder_pitch_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": -1.57,
        "right_shoulder_yaw_joint": 1.57,
        "left_elbow_joint": 0.0,
        "right_elbow_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)


# ============================================================================
# Articulation entity config
# ============================================================================

U20_22DOF_CFG = EntityCfg(
    init_state=copy.deepcopy(_U20_ZERO_INIT_STATE),
    spec_fn=get_u20_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in u20_wholebody_actuator_cfgs),
        soft_joint_pos_limit_factor=0.9,
    ),
)


# ============================================================================
# Symmetric augmentation (MJCF native joint order, 22 DOF)
#
# Joint groups:
#   legs (0-7):      left leg (0-3) <-> right leg (4-7)
#   waist (8-9):     self-symmetric (midline joints)
#   left arm (10-15) <-> right arm (16-21)
#
# Reverse buf determined by MuJoCo forward kinematics empirical verification
# (non-zero qpos left-right mirror test). Y-axis joints (hip_pitch, knee,
# waist_yaw, shoulder_pitch) keep the same sign; X-axis and Z-axis joints
# flip sign, under the M=diag(1,-1,1) sagittal-plane reflection.
#
# Verification methodology:
#   1. Set all joints to random non-zero values, waist=0
#   2. Apply mapping + reverse_buf to produce mirrored qpos
#   3. Forward kinematics on both original and mirrored
#   4. Check left/right body pairs and midline bodies with M=diag(1,-1,1)
#   5. Results: max position error = 0 and max rotation error = 0
#
# NOTE: waist joints (8, 9) are self-mapped midline joints. waist_yaw keeps the
# same sign under this model's qpos convention, while waist_roll flips sign.
# ============================================================================

U20_22Dof_symmetric_augmentation_joint_mapping = [
    4, 5, 6, 7,                  # 0-3:   left leg  -> right leg
    0, 1, 2, 3,                  # 4-7:   right leg -> left leg
    8, 9,                        # 8-9:   waist -> waist (self)
    16, 17, 18, 19, 20, 21,      # 10-15: left arm  -> right arm
    10, 11, 12, 13, 14, 15,      # 16-21: right arm -> left arm
]

U20_22Dof_symmetric_augmentation_joint_reverse_buf = [
    1, -1, -1, 1,                  # left leg:  pitch same, roll rev, yaw rev, knee same
    1, -1, -1, 1,                  # right leg: same pattern
    1, -1,                         # waist: yaw same, roll rev
    -1, 1, -1, -1, -1, -1,         # left arm: roll rev, pitch same, yaw rev, elbow rev, wrist_pitch rev, wrist_yaw rev
    -1, 1, -1, -1, -1, -1,         # right arm: same pattern
]


__all__ = [
    "ARMATURE_U20_ARM",
    "ARMATURE_U20_LEG",
    "ARMATURE_U20_WAIST",
    "DAMPING_RATIO",
    "DAMPING_U20_ARM",
    "DAMPING_U20_LEG",
    "DAMPING_U20_WAIST",
    "NATURAL_FREQ",
    "STIFFNESS_U20_ARM",
    "STIFFNESS_U20_LEG",
    "STIFFNESS_U20_WAIST",
    "U20_22DOF_CFG",
    "U20_22Dof_symmetric_augmentation_joint_mapping",
    "U20_22Dof_symmetric_augmentation_joint_reverse_buf",
    "U20_ARMS",
    "U20_DELAYED_ARMS",
    "U20_DELAYED_LEGS",
    "U20_DELAYED_WAIST",
    "U20_LEGS",
    "U20_MESHES_DIR",
    "U20_WAIST",
    "U20_XML_PATH",
    "u20_action_scale",
    "u20_wholebody_actuator_cfgs",
    "u20_wholebody_delayed_actuator_cfgs",
    "get_u20_assets",
    "get_u20_spec",
]
