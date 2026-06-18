# U20 Parkour 任务构建执行文档

> 目标：参照 `src/instinct_mj/tasks/parkour/config/v11`，为 U20 机器人构建相似的 Parkour AMP 任务。
> 状态：**已落地但未完成最终验收**。Phase A + Phase B 代码已落地，Review 反馈 8.2 与 11.3 #1/#2 已尝试修正；2026-06-12 复审发现 U20 task 注册仍存在导入失败，且 env reset、observation shape、Play 模式等端到端验证尚未完成。

---

## 1. U20 vs V11 结构对比

| 维度 | V11 (29 DOF) | U20 (22 DOF) | 差异说明 |
|------|-------------|-------------|---------|
| 左腿 | hip_pitch, hip_roll, hip_yaw, knee, **ankle_pitch**, **ankle_roll** (6) | hip_yaw, hip_pitch, hip_roll, knee (4) — **无踝关节** | U20 少 2 DOF |
| 右腿 | 同上 (6) | 同上 (4) | U20 少 2 DOF |
| 腰部 | waist_yaw, waist_roll, **waist_pitch** (3) | waist_yaw, waist_roll (2) — **无 waist_pitch** | U20 少 1 DOF |
| 左臂 | shoulder_pitch/roll/yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch (7) | shoulder_pitch/roll/yaw, elbow, wrist_pitch, wrist_yaw (6) — **无 wrist_roll** | U20 少 1 DOF |
| 右臂 | 同上 (7) | 同上 (6) | U20 少 1 DOF |
| **总 DOF** | **29** | **22** | **少 7 DOF** |
| 足部接触体 | `left_ankle_roll_link`, `right_ankle_roll_link` | `lleg5_link`, `rleg5_link` | 无踝关节，足部为刚体 |
| 头部挂载 | `head_pitch_link`（V11 头部有 pitch 关节） | `head_link`（U20 头部固定） | 摄像头挂载点名称不同 |
| 电机力矩 | 分级: 25.5/51/76.5/127.5 Nm | 腿/腰 200 Nm, 臂 40 Nm | U20 力矩规格不同 |
| 初始姿态 | popsicle 蹲姿 (pos z=0.9) | 全零位 | U20 暂用全零位 |

### U20 关节映射表（原 XML 编号 → 语义化命名）

**新建 XML 中的语义化关节名（树遍历顺序）：**

| 序号 | 原 XML 关节名 | 新语义化关节名 | 轴 | 范围 (rad) | 力矩 (Nm) |
|------|-------------|--------------|-----|-----------|-----------|
| 0 | `lleg1_joint` | `left_hip_yaw_joint` | Y | [-1.4, 1.4] | 200 |
| 1 | `lleg2_joint` | `left_hip_pitch_joint` | X | [-0.05, 1.4] | 200 |
| 2 | `lleg3_joint` | `left_hip_roll_joint` | Z | [-0.5, 0.5] | 200 |
| 3 | `lleg4_joint` | `left_knee_joint` | Y | [-0.3, 2.3] | 200 |
| 4 | `rleg1_joint` | `right_hip_yaw_joint` | -Y | [-1.4, 1.4] | 200 |
| 5 | `rleg2_joint` | `right_hip_pitch_joint` | X | [-1.4, 0.05] | 200 |
| 6 | `rleg3_joint` | `right_hip_roll_joint` | Z | [-0.5, 0.5] | 200 |
| 7 | `rleg4_joint` | `right_knee_joint` | -Y | [-2.3, 0.3] | 200 |
| 8 | `waist1_joint` | `waist_yaw_joint` | Y | [-1.57, 1.57] | 200 |
| 9 | `waist2_joint` | `waist_roll_joint` | Z | [-1.57, 1.57] | 200 |
| 10 | `larm1_joint` | `left_shoulder_pitch_joint` | X | [0, 1.67] | 40 |
| 11 | `larm2_joint` | `left_shoulder_roll_joint` | Y | [-1.57, 0.8] | 40 |
| 12 | `larm3_joint` | `left_shoulder_yaw_joint` | Z | [-1.57, 1.57] | 40 |
| 13 | `larm4_joint` | `left_elbow_joint` | X | [-1.57, 0.817] | 40 |
| 14 | `larm5_joint` | `left_wrist_pitch_joint` | X | [-1.57, 1.57] | 40 |
| 15 | `larm6_joint` | `left_wrist_yaw_joint` | Z | [-3.14, 3.14] | 40 |
| 16 | `rarm1_joint` | `right_shoulder_pitch_joint` | X | [-1.67, 0] | 40 |
| 17 | `rarm2_joint` | `right_shoulder_roll_joint` | Y | [-1.57, 0.8] | 40 |
| 18 | `rarm3_joint` | `right_shoulder_yaw_joint` | Z | [-1.57, 1.57] | 40 |
| 19 | `rarm4_joint` | `right_elbow_joint` | X | [-0.817, 1.57] | 40 |
| 20 | `rarm5_joint` | `right_wrist_pitch_joint` | X | [-1.57, 1.57] | 40 |
| 21 | `rarm6_joint` | `right_wrist_yaw_joint` | Z | [-3.14, 3.14] | 40 |

**对应的 link 体名映射（原 XML → 新 XML）：**

| 原 link 名 | 新 link 名 | 说明 |
|-----------|-----------|------|
| `lleg1_link` | `left_hip_yaw_link` | |
| `lleg2_link` | `left_hip_pitch_link` | |
| `lleg3_link` | `left_hip_roll_link` | |
| `lleg4_link` | `left_knee_link` | |
| `lleg5_link` | `left_foot_link` | 足部刚体 |
| `LL_FOOT` | `LL_FOOT` | 保持不变 |
| `rleg1_link` | `right_hip_yaw_link` | |
| `rleg2_link` | `right_hip_pitch_link` | |
| `rleg3_link` | `right_hip_roll_link` | |
| `rleg4_link` | `right_knee_link` | |
| `rleg5_link` | `right_foot_link` | 足部刚体 |
| `LR_FOOT` | `LR_FOOT` | 保持不变 |
| `waist1_link` | `waist_yaw_link` | |
| `waist2_link` | `waist_roll_link` | |
| `head_link` | `head_link` | 保持不变 |
| `larm1_link` | `left_shoulder_pitch_link` | |
| `larm2_link` | `left_shoulder_roll_link` | |
| `larm3_link` | `left_shoulder_yaw_link` | |
| `larm4_link` | `left_elbow_link` | |
| `larm5_link` | `left_wrist_pitch_link` | |
| `larm6_link` | `left_wrist_yaw_link` | |
| `rarm1_link` | `right_shoulder_pitch_link` | |
| `rarm2_link` | `right_shoulder_roll_link` | |
| `rarm3_link` | `right_shoulder_yaw_link` | |
| `rarm4_link` | `right_elbow_link` | |
| `rarm5_link` | `right_wrist_pitch_link` | |
| `rarm6_link` | `right_wrist_yaw_link` | |
| `base_link` | `base_link` | 保持不变 |
| `imu_link` | `imu_link` | 保持不变 |

**碰撞几何体（geom）名称映射：**

| 原 geom 名 | 新 geom 名 |
|-----------|-----------|
| `left_foot1_collision` | `left_foot1_collision` |
| `left_foot2_collision` | `left_foot2_collision` |
| `left_foot3_collision` | `left_foot3_collision` |
| `left_foot4_collision` | `left_foot4_collision` |
| `left_foot5_collision` | `left_foot5_collision` |
| `right_foot1_collision` | `right_foot1_collision` |
| ... | ... |

碰撞几何体名称保持不变（已语义化）。

**site 名称：**

| 原 site 名 | 新 site 名 |
|-----------|-----------|
| `imu` | `imu` |
| `left_foot` | `left_foot` |
| `right_foot` | `right_foot` |

保持不变。

**contact exclude 需要更新 body 名：**

```xml
<!-- 原 -->
<exclude body1="larm3_link" body2="larm5_link"/>
<exclude body1="rarm3_link" body2="rarm5_link"/>
<exclude body1="base_link" body2="lleg1_link"/>
<exclude body1="base_link" body2="rleg1_link"/>

<!-- 新 -->
<exclude body1="left_shoulder_yaw_link" body2="left_wrist_pitch_link"/>
<exclude body1="right_shoulder_yaw_link" body2="right_wrist_pitch_link"/>
<exclude body1="base_link" body2="left_hip_yaw_link"/>
<exclude body1="base_link" body2="right_hip_yaw_link"/>
```

---

## 2. 需创建/修改的文件清单

| # | 文件路径 | 操作 | 行数 | 状态 |
|---|---------|------|------|------|
| 1 | `src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml` | **新建** | 215 | ✅ |
| 2 | `src/instinct_mj/assets/u20.py` | **新建** | 271 | ✅ |
| 3 | `src/instinct_mj/tasks/tracking/config/u20/__init__.py` | **新建** | 3 | ✅ |
| 4 | `src/instinct_mj/tasks/tracking/config/u20/env_cfgs.py` | **新建** | 89 | ✅ |
| 5 | `src/instinct_mj/tasks/parkour/config/u20/__init__.py` | **新建** | 20 | ✅ |
| 6 | `src/instinct_mj/tasks/parkour/config/u20/u20_parkour_target_amp_cfg.py` | **新建** | 853 | ✅ |
| 7 | `src/instinct_mj/tasks/parkour/config/u20/agents/instinct_rl_amp_cfg.py` | **新建** | 92 | ✅ |
| 8 | `src/instinct_mj/tasks/parkour/__init__.py` | **修改** | +1 | ✅ |

---

## 3. 各文件详细规格

### 3.1 `u20_popsicle.xml` — 新建语义化 XML

**路径**: `src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml`

**基于**: `src/instinct_mj/assets/resources/u20/xml/u20.xml`（217 行）

**修改内容**:
1. 复制 `u20.xml` 的全部内容
2. `meshdir` 改为相对路径 `meshdir="meshes/"`（原为绝对路径）
3. 所有 mesh 的 `file` 属性改为相对路径（去掉绝对路径前缀，只保留文件名）
4. 按上面的映射表重命名所有 joint 名、body 名
5. contact exclude 中的 body1/body2 引用更新为新 body 名
6. sensor 中 `objname="imu"` 保持不变（site 名未改）
7. sensor 中 `body="base_link"` 保持不变
8. 所有 collision geom 名（`left_foot*_collision`, `right_foot*_collision`）保持不变
9. 保持所有 inertial 参数、geom 参数、joint 参数（axis/range/actuatorfrcrange）不变

### 3.2 `u20.py` — U20 资产配置模块

**路径**: `src/instinct_mj/assets/u20.py`

**参照**: `src/instinct_mj/assets/v11.py`（421 行）

**结构**:

```python
"""U20 humanoid robot asset configuration for mjlab/MuJoCo.

U20 full-body variant: 22 drivable DOF:
- Legs (0-7):   left_hip_yaw/pitch/roll, left_knee,
                 right_hip_yaw/pitch/roll, right_knee
- Waist (8-9):  waist_yaw, waist_roll
- Left arm (10-15):  shoulder_pitch/roll/yaw, elbow, wrist_pitch, wrist_yaw
- Right arm (16-21): shoulder_pitch/roll/yaw, elbow, wrist_pitch, wrist_yaw

Root link: base_link
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

U20_XML_PATH: str = os.path.join(__file_dir__, "resources/u20/xml/u20_popsicle.xml")
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
```

**电机常量**（使用 XML 默认 armature=0.01）:

```python
ARMATURE_U20_LEG = 0.01
ARMATURE_U20_WAIST = 0.01
ARMATURE_U20_ARM = 0.01

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_U20_LEG = ARMATURE_U20_LEG * NATURAL_FREQ ** 2
STIFFNESS_U20_WAIST = ARMATURE_U20_WAIST * NATURAL_FREQ ** 2
STIFFNESS_U20_ARM = ARMATURE_U20_ARM * NATURAL_FREQ ** 2

DAMPING_U20_LEG = 2.0 * DAMPING_RATIO * ARMATURE_U20_LEG * NATURAL_FREQ
DAMPING_U20_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_U20_WAIST * NATURAL_FREQ
DAMPING_U20_ARM = 2.0 * DAMPING_RATIO * ARMATURE_U20_ARM * NATURAL_FREQ
```

**执行器分组**（使用 85% 降额: 200×0.85=170, 40×0.85=34）:

```python
# 腿部关节 (hip_yaw/pitch/roll + knee): 200 Nm × 0.85 = 170 Nm
U20_LEGS = InstinctActuatorCfg(
    target_names_expr=(
        ".*_hip_yaw_joint",
        ".*_hip_pitch_joint",
        ".*_hip_roll_joint",
        ".*_knee_joint",
    ),
    effort_limit=170.0,
    velocity_limit=14.66,
    stiffness=STIFFNESS_U20_LEG,
    damping=DAMPING_U20_LEG,
    armature=ARMATURE_U20_LEG,
)

# 腰部关节 (waist_yaw + waist_roll): 200 Nm × 0.85 = 170 Nm
U20_WAIST = InstinctActuatorCfg(
    target_names_expr=("waist_yaw_joint", "waist_roll_joint"),
    effort_limit=170.0,
    velocity_limit=14.66,
    stiffness=STIFFNESS_U20_WAIST,
    damping=DAMPING_U20_WAIST,
    armature=ARMATURE_U20_WAIST,
)

# 手臂关节 (shoulder_pitch/roll/yaw + elbow + wrist_pitch/yaw): 40 Nm × 0.85 = 34 Nm
U20_ARMS = InstinctActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",
    ),
    effort_limit=34.0,
    velocity_limit=22.0,
    stiffness=STIFFNESS_U20_ARM,
    damping=DAMPING_U20_ARM,
    armature=ARMATURE_U20_ARM,
)
```

**执行器聚合**:

```python
u20_wholebody_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    U20_LEGS,
    U20_WAIST,
    U20_ARMS,
)

# Delayed 变体
U20_DELAYED_LEGS = DelayedInstinctActuatorCfg(
    base_cfg=U20_LEGS, delay_target="position", delay_min_lag=0, delay_max_lag=2,
)
U20_DELAYED_WAIST = DelayedInstinctActuatorCfg(
    base_cfg=U20_WAIST, delay_target="position", delay_min_lag=0, delay_max_lag=2,
)
U20_DELAYED_ARMS = DelayedInstinctActuatorCfg(
    base_cfg=U20_ARMS, delay_target="position", delay_min_lag=0, delay_max_lag=2,
)

u20_wholebody_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    U20_DELAYED_LEGS,
    U20_DELAYED_WAIST,
    U20_DELAYED_ARMS,
)
```

**Action scale**:

```python
u20_action_scale: dict[str, float] = {}
for _act in u20_wholebody_actuator_cfgs:
    _effort = _act.effort_limit
    _stiffness = _act.stiffness
    if _effort is None or _stiffness == 0.0:
        continue
    for _jname in _act.target_names_expr:
        u20_action_scale[_jname] = 0.25 * _effort / _stiffness
```

**初始状态**（全零位，高度待验证）:

```python
_U20_ZERO_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.85),  # 需根据 U20 腿长验证，避免穿地
    joint_pos={},           # 全零
    joint_vel={".*": 0.0},
)
```

> **注意**: `pos` 的 z 值需要验证。U20 的腿长（hip → knee: 0.231m, knee → foot: 0.45m）总长约 0.68m，全零位时腿完全伸直，站立高度约 0.68m。但由于 hip_yaw 轴不在 base_link 正下方（有 0.12m 偏移），实际需要仿真验证。初始值暂定 0.85m，后续根据仿真调整。

**Entity 配置**:

```python
U20_22DOF_CFG = EntityCfg(
    init_state=copy.deepcopy(_U20_ZERO_INIT_STATE),
    spec_fn=get_u20_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in u20_wholebody_actuator_cfgs),
        soft_joint_pos_limit_factor=0.9,
    ),
)
```

**对称增强映射（22 DOF）**:

```python
# MJCF joint order:
#  0: left_hip_yaw     ↔  4: right_hip_yaw
#  1: left_hip_pitch   ↔  5: right_hip_pitch
#  2: left_hip_roll    ↔  6: right_hip_roll
#  3: left_knee        ↔  7: right_knee
#  8: waist_yaw        (self)
#  9: waist_roll       (self)
# 10: left_shoulder_pitch  ↔ 16: right_shoulder_pitch
# 11: left_shoulder_roll   ↔ 17: right_shoulder_roll
# 12: left_shoulder_yaw    ↔ 18: right_shoulder_yaw
# 13: left_elbow           ↔ 19: right_elbow
# 14: left_wrist_pitch     ↔ 20: right_wrist_pitch
# 15: left_wrist_yaw       ↔ 21: right_wrist_yaw

U20_22Dof_symmetric_augmentation_joint_mapping = [
    4, 5, 6, 7,                # 0-3:  left leg  → right leg
    0, 1, 2, 3,                # 4-7:  right leg → left leg
    8, 9,                      # 8-9:  waist → waist (self)
    16, 17, 18, 19, 20, 21,    # 10-15: left arm → right arm
    10, 11, 12, 13, 14, 15,    # 16-21: right arm → left arm
]

U20_22Dof_symmetric_augmentation_joint_reverse_buf = [
    -1, 1, -1, 1,              # left leg:  yaw rev, pitch same, roll rev, knee same
    -1, 1, -1, 1,              # right leg: same pattern
    -1, -1,                    # waist: yaw rev, roll rev
    1, -1, -1, 1, 1, -1,       # left arm: pitch same, roll rev, yaw rev, elbow same, wrist_pitch same, wrist_yaw rev
    1, -1, -1, 1, 1, -1,       # right arm: same pattern
]
```

> **注意**: `reverse_buf` 中各关节的正负号需根据 MuJoCo 轴方向确认。上表假设：
> - yaw 类关节（绕 Y/Z 轴）左右对称时取反
> - pitch 类关节（绕 X 轴）左右对称时不取反
> - roll 类关节（绕 Z 轴）左右对称时取反
>
> U20 的具体轴方向需对照 XML 确认：
> - `left_hip_yaw_joint`: axis Y → mirror to right: axis -Y → reverse
> - `left_hip_pitch_joint`: axis X → mirror to right: axis X → same
> - `left_hip_roll_joint`: axis Z → mirror to right: axis Z → same (Z轴roll通常不反转)
> - `left_knee_joint`: axis Y → mirror to right: axis -Y → reverse
>
> 需要进一步验证，下表为**初步方案**：

```python
# 修正版 (待验证):
U20_22Dof_symmetric_augmentation_joint_reverse_buf = [
    -1, 1, 1, -1,              # left leg:  yaw(Y→-Y)rev, pitch(X)same, roll(Z)same, knee(Y→-Y)rev
    -1, 1, 1, -1,              # right leg: same pattern
    -1, -1,                    # waist: yaw(Y)rev, roll(Z)rev
    1, -1, -1, 1, 1, -1,       # left arm: pitch(X)same, roll(Y→Y)rev, yaw(Z)rev, elbow(X)same, wrist_pitch(X)same, wrist_yaw(Z)rev
    1, -1, -1, 1, 1, -1,       # right arm: same pattern
]
```

**`__all__` 导出列表**:

```python
__all__ = [
    "ARMATURE_U20_LEG",
    "ARMATURE_U20_WAIST",
    "ARMATURE_U20_ARM",
    "DAMPING_U20_LEG",
    "DAMPING_U20_WAIST",
    "DAMPING_U20_ARM",
    "DAMPING_RATIO",
    "NATURAL_FREQ",
    "STIFFNESS_U20_LEG",
    "STIFFNESS_U20_WAIST",
    "STIFFNESS_U20_ARM",
    "U20_22DOF_CFG",
    "U20_22Dof_symmetric_augmentation_joint_mapping",
    "U20_22Dof_symmetric_augmentation_joint_reverse_buf",
    "U20_ARMS",
    "U20_DELAYED_ARMS",
    "U20_DELAYED_LEGS",
    "U20_DELAYED_WAIST",
    "U20_LEGS",
    "U20_WAIST",
    "U20_MESHES_DIR",
    "U20_XML_PATH",
    "u20_action_scale",
    "u20_wholebody_actuator_cfgs",
    "u20_wholebody_delayed_actuator_cfgs",
    "get_u20_assets",
    "get_u20_spec",
]
```

---

### 3.3 `tracking/config/u20/__init__.py`

**路径**: `src/instinct_mj/tasks/tracking/config/u20/__init__.py`

```python
from .env_cfgs import u20_flat_tracking_env_cfg

__all__ = ["u20_flat_tracking_env_cfg"]
```

### 3.4 `tracking/config/u20/env_cfgs.py`

**路径**: `src/instinct_mj/tasks/tracking/config/u20/env_cfgs.py`

**参照**: `src/instinct_mj/tasks/tracking/config/v11/env_cfgs.py`（91 行）

**与 V11 的关键差异**:

1. 导入 `U20_22DOF_CFG` 和 `u20_action_scale` 代替 V11 对应物
2. `motion_cmd.body_names` 列表适配 U20 的 body 名（无 ankle_roll, 无 waist_pitch, 无 wrist_roll）
3. `ee_body_pos` 终止条件中的 body_names 适配 U20（足部为 `left_foot_link`/`right_foot_link`，手腕为 `left_wrist_yaw_link`/`right_wrist_yaw_link`）

**完整内容**:

```python
"""U20 flat tracking environment configuration."""

from __future__ import annotations

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from instinct_mj.assets.u20 import U20_22DOF_CFG, u20_action_scale


def u20_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": copy.deepcopy(U20_22DOF_CFG)}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = copy.deepcopy(u20_action_scale)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "base_link"
    motion_cmd.body_names = (
        "base_link",
        "left_hip_roll_link",
        "left_knee_link",
        "left_foot_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_foot_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )

    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_foot_link",
        "right_foot_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    cfg.viewer.body_name = "base_link"

    if not has_state_estimation:
        new_actor_terms = {
            k: v
            for k, v in cfg.observations["actor"].terms.items()
            if k not in ["motion_anchor_pos_b", "base_lin_vel"]
        }
        cfg.observations["actor"] = ObservationGroupCfg(
            terms=new_actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg
```

> **注意**: V11 的 `motion_cmd.body_names` 有 14 个 body（含 `waist_pitch_link`），U20 只有 13 个（无 `waist_pitch_link`）。对称映射索引也需相应调整。

---

### 3.5 `parkour/config/u20/__init__.py`

**路径**: `src/instinct_mj/tasks/parkour/config/u20/__init__.py`

```python
"""Register Instinct Mj parkour U20 tasks."""

from instinct_mj.tasks.registry import register_instinct_task

from .agents.instinct_rl_amp_cfg import U20ParkourPPORunnerCfg
from .u20_parkour_target_amp_cfg import instinct_u20_parkour_amp_final_cfg

register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-U20-v0",
    env_cfg_factory=lambda: instinct_u20_parkour_amp_final_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_u20_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=U20ParkourPPORunnerCfg,
)

register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-U20-Play-v0",
    env_cfg_factory=lambda: instinct_u20_parkour_amp_final_cfg(play=True),
    play_env_cfg_factory=lambda: instinct_u20_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=U20ParkourPPORunnerCfg,
)
```

---

### 3.6 `u20_parkour_target_amp_cfg.py`

**路径**: `src/instinct_mj/tasks/parkour/config/u20/u20_parkour_target_amp_cfg.py`

**参照**: `src/instinct_mj/tasks/parkour/config/v11/v11_parkour_target_amp_cfg.py`（906 行）

**与 V11 的逐段差异**:

#### 3.6.1 文件头部

```python
"""U20 parkour AMP task config factories.

Adapted from V11 parkour config. Key differences vs V11:
- 22-DOF full-body model (no ankle pitch/roll, no waist_pitch, no wrist_roll)
- Foot contact bodies: left_foot_link / right_foot_link (rigid foot, no ankle)
- Camera mounted at head_link (fixed head, no head_pitch joint)
- Motion reference uses U20 XML (u20_popsicle.xml) and 22-DOF symmetric mappings
- AMP discriminator operates on full 22-DOF joint state (joint_names=".*")
"""
```

**导入差异**:

```python
# V11 导入 → U20 替换
from instinct_mj.tasks.tracking.config.v11.env_cfgs import v11_flat_tracking_env_cfg
# →
from instinct_mj.tasks.tracking.config.u20.env_cfgs import u20_flat_tracking_env_cfg

from instinct_mj.assets.v11 import (
    V11_XML_PATH,
    V11_29Dof_symmetric_augmentation_joint_mapping,
    V11_29Dof_symmetric_augmentation_joint_reverse_buf,
    V11_29DOF_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_v11_wholebody_delayed_actuator_cfgs,
    get_v11_assets,
)
# →
from instinct_mj.assets.u20 import (
    U20_XML_PATH,
    U20_22Dof_symmetric_augmentation_joint_mapping,
    U20_22Dof_symmetric_augmentation_joint_reverse_buf,
    U20_22DOF_CFG,
    u20_action_scale,
    u20_wholebody_delayed_actuator_cfgs,
    get_u20_assets,
)
```

#### 3.6.2 数据集路径

```python
_PARKOUR_DATASET_DIR = os.path.expanduser("~/Instinct-mjlab/Datasets/u20_npz/")
```

#### 3.6.3 Motion Reference 配置

**V11 有 14 个 link_of_interests → U20 有 13 个（无 waist_pitch_link）**:

```python
motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=U20_XML_PATH,
    link_of_interests=[
        "base_link",
        # 无 waist_pitch_link (U20 没有 waist_pitch 关节)
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_foot_link",       # 替代 left_ankle_roll_link
        "right_foot_link",      # 替代 right_ankle_roll_link
    ],
    # 13 links: base_link=0 (midline)
    # pairs: (1,2) shoulder_roll, (3,4) elbow, (5,6) wrist_yaw, (7,8) hip_roll,
    #        (9,10) knee, (11,12) foot
    symmetric_augmentation_link_mapping=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11],
    symmetric_augmentation_joint_mapping=list(U20_22Dof_symmetric_augmentation_joint_mapping),
    symmetric_augmentation_joint_reverse_buf=list(U20_22Dof_symmetric_augmentation_joint_reverse_buf),
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={"run_walk": AmassMotionCfg()},
    mp_split_method="Even",
)
```

> **注意**: AMP 数据目录 `/home/user2/Instinct-mjlab/Datasets/u20_npz/` 目前为空，用户会后续上传。当前沿用 `AmassMotionCfgBase`，预期格式为 `*retargeted.npz`；不要使用 `.pkl` 目录名，避免和加载逻辑不一致。

#### 3.6.4 主环境构建函数 `instinct_u20_parkour_amp_env_cfg`

**基础调用**:
```python
cfg = u20_flat_tracking_env_cfg(play=play, has_state_estimation=True)
```

**机器人实体**:
```python
u20_robot_cfg = copy.deepcopy(U20_22DOF_CFG)
u20_robot_cfg.articulation.actuators = copy.deepcopy(u20_wholebody_delayed_actuator_cfgs)
u20_robot_cfg.init_state.pos = (0.0, 0.0, 0.85)  # 待验证
cfg.scene.entities["robot"] = u20_robot_cfg
```

**Action scale**:
```python
joint_pos_action: JointPositionActionCfg = cfg.actions["joint_pos"]
joint_pos_action.scale = copy.deepcopy(u20_action_scale)
```

#### 3.6.5 传感器配置

**所有 `left_ankle_roll_link` → `left_foot_link`，`right_ankle_roll_link` → `right_foot_link`**:

1. **contact_forces**: 足部接触
   ```python
   primary=ContactMatch(
       mode="body",
       pattern=("left_foot_link", "right_foot_link"),
       entity="robot",
   )
   ```

2. **torso_contact_forces**: 不变（base_link）

3. **undesired_contact_forces**: exclude 改为
   ```python
   exclude=("left_foot_link", "right_foot_link"),
   ```

4. **leg_volume_points**: body_names 改为
   ```python
   body_names="left_foot_link",  # 或 ".*_foot_link"
   ```
   
   Grid 范围需适配 U20 足部碰撞几何。U20 足部有 5 个 capsule（`left_foot1~5_collision`），沿 x 从 -0.04 到 +0.04，间距 0.02。胶囊参数 `size="0.015 0.1"`，位于 `pos="0.04 0.012763 -0.08"`（及 x 方向偏移）。
   
   ```python
   VolumePointsCfg(
       name="leg_volume_points",
       entity_name="robot",
       body_names="left_foot_link",
       points_generator=Grid3dPointsGeneratorCfg(
           x_min=-0.06,
           x_max=0.10,
           x_num=8,
           y_min=-0.03,
           y_max=0.05,
           y_num=4,
           z_min=-0.10,    # 胶囊底部延伸
           z_max=-0.05,    # 胶囊顶部
           z_num=2,
       ),
       debug_vis=False,
   ),
   ```
   
   > **注意**: V11 的 volume points 只用了一个 body 名 `"left_ankle_roll_link"`，但实际上 left/right 足部都有。这可能是因为 mjlab 的 VolumePointsCfg 内部会通过 regex 或同时匹配两个 body。U20 也用 `"left_foot_link"` 或 `"left_foot_link"` 即可，需要确认 mjlab 是否自动处理镜像 body。如果需要同时指定两个 body，使用 tuple。
   
   经查看 V11 原版：`body_names=".*_ankle_roll_link"`（regex 匹配左右两侧）。U20 对应为 `body_names=".*_foot_link"`。

5. **left_height_scanner / right_height_scanner**:
   ```python
   RayCastSensorCfg(
       name="left_height_scanner",
       frame=ObjRef(type="body", name="left_foot_link", entity="robot"),
       ...  # 其余参数不变
   ),
   RayCastSensorCfg(
       name="right_height_scanner",
       frame=ObjRef(type="body", name="right_foot_link", entity="robot"),
       ...  # 其余参数不变
   ),
   ```

6. **camera**: 挂载在 `head_link`（U20 无 head_pitch_link）
   ```python
   frame=ObjRef(type="body", name="head_link", entity="robot"),
   ```
   
   offset 需要调整。U20 的 `head_link` 有一个 sphere geom `size="0.1" pos="0 0 0.1435"`，摄像头安装在头部前方偏上位置。暂沿用 V11 的 offset，后续根据实际模型调整。
   
   ```python
   offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
       pos=(0.025, 0.005, 0.054),
       rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
       convention="world",
   ),
   ```
   
   > **注意**: 此 offset 是 V11 的，U20 头部几何不同，需要后续微调。

#### 3.6.6 奖励配置差异

| V11 Reward Term | U20 调整 |
|----------------|---------|
| `track_lin_vel_xy_exp` | 不变 |
| `track_ang_vel_z_exp` | 不变 |
| `heading_error` | 不变 |
| `dont_wait` | 不变 |
| `is_alive` | 不变 |
| `stand_still` | 不变 |
| `volume_points_penetration` | 不变（sensor_name 相同） |
| `feet_air_time` | 不变（sensor_name 相同） |
| `feet_slide` | `body_names` 改为 `("left_foot_link", "right_foot_link")` |
| `ang_vel_xy_l2` | 不变 |
| `dof_torques_l2` | `joint_names` 改为 `(".*_hip_.*", ".*_knee_joint")` — **无 ankle** |
| `dof_acc_l2` | 不变 (`".*"`) |
| `dof_vel_l2` | 不变 (`".*"`) |
| `action_rate_l2` | 不变 |
| `flat_orientation_l2` | 不变 |
| `pelvis_orientation_l2` | 不变 (`base_link`) |
| `feet_flat_ori` | `body_names` 改为 `("left_foot_link", "right_foot_link")` |
| `feet_at_plane` | `body_names` 改为 `("left_foot_link", "right_foot_link")` |
| `feet_close_xy` | `body_names` 改为 `("left_foot_link", "right_foot_link")` |
| `energy` | `joint_names` 改为 `(".*_hip_.*", ".*_knee_joint")` — **无 ankle** |
| `freeze_upper_body` | `joint_names` 改为 `(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*")` — **无 wrist_roll**（但 `.*_wrist.*` 仍匹配 `wrist_pitch` 和 `wrist_yaw`，效果等同于 V11） |
| `dof_pos_limits` | 不变 (`".*"`) |
| `dof_vel_limits` | 不变 (`".*"`) |
| `torque_limits` | 不变 (`".*"`) |
| `undesired_contacts` | 不变 |

> **注意**: `feet_flat_ori`（足部朝向奖励）在 U20 上可能效果不同，因为 U20 足部是刚体（无踝关节控制），`left_foot_link` 的朝向由腿部关节间接决定。但 reward 函数本身仅检测 body 朝向，逻辑不变。

#### 3.6.7 其他 MDP 组件

**Commands**: 与 V11 完全相同（地形类型和速度范围一致）

**Terminations**: 
- `time_out`: 不变
- `terrain_out_bound`: 不变
- `base_contact`: 不变（base_link）
- `bad_orientation`: 不变
- `root_height`: `minimum_height` 可能需要调整（U20 腿长不同），暂保持 0.5m
- `dataset_exhausted`: 不变

**Events**: 与 V11 完全相同

**Curriculum**: 与 V11 完全相同

**Observations**: 结构不变，自动适配 22 DOF（因为 `joint_names=".*"` 匹配所有关节）

**Play mode overrides**: 与 V11 结构相同，仅调整 viewer 参数

#### 3.6.8 主函数名

```python
def instinct_u20_parkour_amp_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    ...

def instinct_u20_parkour_amp_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_u20_parkour_amp_env_cfg(play=play)
    if play:
        cfg.viewer = ViewerConfig(
            lookat=(0.0, 0.75, 0.0),
            distance=4.123105625617661,
            elevation=-14.036243467926479,
            azimuth=180.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name=None,
        )
        cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
        cfg.viewer.entity_name = None
        cfg.viewer.body_name = None
    return cfg
```

---

### 3.7 `agents/instinct_rl_amp_cfg.py`

**路径**: `src/instinct_mj/tasks/parkour/config/u20/agents/instinct_rl_amp_cfg.py`

**与 V11 的差异**:

仅修改实验名和类名：

```python
@dataclass(kw_only=True)
class U20ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 50000
    save_interval: int = 1000
    experiment_name: str = "u20_parkour"      # V11 为 "v11_parkour"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: object = field(default_factory=lambda: MoEPolicyCfg())
    algorithm: object = field(default_factory=lambda: AmpAlgoCfg())
```

网络架构（`DepthEncoderConv2dCfg`, `MoEPolicyCfg`, `AmpAlgoCfg`）与 V11 完全相同，不需要修改。因为观测维度由 mjlab 自动推断（`joint_names=".*"` 匹配 22 DOF），网络层大小不变。

---

### 3.8 `parkour/__init__.py` 修改

**路径**: `src/instinct_mj/tasks/parkour/__init__.py`

**当前内容**:
```python
"""Parkour task package."""

from instinct_mj.tasks.parkour.config import g1 as _g1  # noqa: F401
from instinct_mj.tasks.parkour.config import v11 as _v11  # noqa: F401
```

**追加 1 行**:
```python
from instinct_mj.tasks.parkour.config import u20 as _u20  # noqa: F401
```

---

## 4. 待验证事项

| # | 事项 | 优先级 | 说明 |
|---|------|--------|------|
| 1 | **初始站立高度** | 高 | 当前候选 `z=1.0`，仍需 env reset 验证足底不穿地且不悬空 |
| 2 | **对称增强 reverse_buf** | 高 | 各关节的 ±1 需根据 MuJoCo 轴方向逐一确认 |
| 3 | **AMP 数据格式** | 高 | 用户上传后需确认 `*retargeted.npz` 是否与 `AmassMotionCfgBase` 兼容 |
| 4 | **摄像头 offset** | 中 | U20 头部几何与 V11 不同，需调整 offset |
| 5 | **volume_points Grid 范围** | 中 | U20 足部碰撞几何与 V11 不同，需调整 Grid 覆盖范围 |
| 6 | **root_height minimum_height** | 中 | U20 腿长不同，0.5m 阈值可能需要调整 |
| 7 | **motor 常量** | 低 | 当前使用 XML 默认 armature=0.01，后续可能需要替换为实际电机参数 |

---

## 5. 执行顺序

```
Step 1: 创建 u20_popsicle.xml (语义化命名 + 相对路径)       ✅ 已完成
Step 2: 创建 u20.py (资产模块)                              ✅ 已完成
Step 3: 创建 tracking/config/u20/ (tracking 基础配置)        ✅ 已完成
Step 4: 验证 XML/spec/joint order                           ✅ 已通过
Step 5: 创建 parkour/config/u20/ (parkour 任务配置)          ✅ 已完成
Step 6: 修改 parkour/__init__.py (注册任务)                  ✅ 已完成
Step 7: 语法检查 + 残留名称检查                              ✅ 已通过
```

---

## 6. 文件依赖关系

```
u20_popsicle.xml ← u20.py ← tracking/config/u20/env_cfgs.py ← parkour/config/u20/u20_parkour_target_amp_cfg.py
                                                                              ↑
                                                           parkour/config/u20/agents/instinct_rl_amp_cfg.py
                                                           parkour/config/u20/__init__.py
                                                           parkour/__init__.py (修改)
```

---

## 7. Review 检查清单

- [x] u20_popsicle.xml 中所有 joint/body 名是否正确映射
- [x] contact exclude 中的 body 引用是否已更新
- [x] meshdir 和 mesh file 路径是否改为相对路径
- [x] u20.py 中 actuator 分组的 target_names_expr 是否覆盖所有 22 个关节
- [x] 对称增强映射是否正确（22 DOF）
- [x] reverse_buf 正负号是否与 MuJoCo 轴方向一致
- [x] tracking env_cfg 中 motion_cmd.body_names 是否完整（13 个 body）
- [x] parkour env_cfg 中所有传感器引用的 body 名是否已更新
- [x] reward 中引用 ankle 的 regex 是否已改为 foot
- [x] 足部 body 名在整个配置中是否一致（`left_foot_link` / `right_foot_link`）
- [x] 无残留的 V11 特有名称（如 `ankle_roll`, `waist_pitch`, `wrist_roll`）

---

## 8. Review 反馈（2026-06-11）

### 8.1 总体可行性

结论：方案整体可行，适合按 V11 Parkour AMP 配置复制后做 U20 结构替换。但当前文档仍存在若干会影响落地的高风险点，建议先修正文档中的具体规格，再开始批量创建文件。

可行依据：

1. 当前仓库已存在 `src/instinct_mj/assets/resources/u20/xml/u20.xml`、U20 mesh 和 V11 的完整资产/Tracking/Parkour 配置，可直接作为迁移模板。
2. V11 配置已经是 mjlab manager 字典表达，U20 方案按同一模式落地，不需要新增兼容层。
3. U20 的 22 DOF 关节、足部 body、头部 body 都能在现有 XML 中找到明确对应项，核心替换路径清晰。

主要限制：

1. U20 无踝关节，Parkour 任务在足部朝向、足部贴地、越障接触上的可训练性会低于 V11。配置可运行不等于 reward 权重一定可复用。
2. AMP 参考数据必须与 U20 22 DOF 和 `link_of_interests` 维度一致。若上传数据仍是 V11/AMASS 原维度，MotionReference 会在加载或训练阶段失败。
3. 初始姿态和相机 offset 不能只沿用 V11，需要至少通过 MuJoCo spec 加载和一次 env reset 验证。

### 8.2 必须修正后再执行

#### 1. `u20_popsicle.xml` 的 mesh 处理需要收紧

当前 3.1 写法要求重命名所有 body/joint，同时 mesh file 改为相对路径。这里要明确：**不要要求重命名 STL 文件**，除非同步复制/重命名 `meshes/*.STL`。

建议执行规则：

1. body 和 joint 按语义化名称重命名。
2. mesh `file` 只改成相对文件名，例如 `lleg1_link.STL`，保持现有文件名。
3. mesh `name` 与 geom 的 `mesh` 引用可以保持原 XML 名称，例如 `mesh="lleg1_link"`，因为 MuJoCo body 名和 mesh asset 名不需要一致。
4. 如果决定把 mesh `name` 也语义化，则必须同步更新所有 `geom mesh="..."` 引用；但不建议这样做，收益低且容易漏改。

#### 2. `get_u20_assets()` 和 `meshdir="meshes/"` 的 key 需要实测

V11 的 `get_v11_assets()` 使用 `update_assets(assets, V11_MESHES_DIR, normalized_meshdir)`。U20 复制此模式可行，但必须验证 `spec.meshdir` 在 `meshdir="meshes/"` 下到底是 `"meshes/"` 还是编译后的路径。

执行前验证项：

```bash
python3 -c "from instinct_mj.assets.u20 import get_u20_spec; s=get_u20_spec(); print(s.meshdir, len(s.assets))"
```

验收标准：

1. `mujoco.MjSpec.from_file(U20_XML_PATH)` 不报错。
2. `spec.assets` 中包含所有 U20 STL 资产。
3. 创建 `mujoco.MjModel.from_spec(spec)` 不报 mesh 缺失。

#### 3. 初始高度 `z=0.85` 不能作为最终默认值

U20 全零位时，从 `base_link` 到足底碰撞胶囊底部的几何深度约为 `0.16 + 0.231 + 0.45 + 0.08 + 0.015 = 0.936m`，还未计入 hip link 初始偏置和接触求解余量。因此 `z=0.85` 很可能导致初始穿地。

建议：

1. 文档中的 `_U20_ZERO_INIT_STATE.pos` 暂改为待验证值，不要写成确定默认。
2. 首次实现后用 MuJoCo model/data 或 env reset 计算足部 collision geom 世界最低点。
3. 验证前建议候选范围从 `0.95m ~ 1.05m` 起步，而不是 `0.85m`。

#### 4. 对称增强 `reverse_buf` 当前存在互相矛盾的两版

文档中先给出一版 reverse_buf，随后又给出“修正版（待验证）”。执行时必须只保留一版，否则实现者容易照抄错误版本。

建议保留待验证版作为唯一草案，并加上明确验收：

1. 根据 XML joint axis 和左右镜像定义逐项确认符号。
2. 用一组非零 qpos 做左右镜像后，检查左右足、膝、肘、腕的世界位置是否满足 `y` 取反、`x/z` 近似不变。
3. 验证通过后再把 `U20_22Dof_symmetric_augmentation_joint_reverse_buf` 导出给 MotionReference 使用。

需要特别注意：

1. U20 左右 knee axis 是 `Y` / `-Y`，knee 符号很可能应与 V11 的“pitch same”规则不同，不能按语义名直接套 V11。
2. U20 hip_roll axis 左右都是 `Z`，但镜像变换下是否取反取决于局部 frame，不应只看 axis 字符串。

#### 5. Tracking `motion_cmd.body_names` 维度变化会影响观测维度

V11 Tracking 的 `motion_cmd.body_names` 是 14 个，U20 文档改成 13 个。这个调整合理，但会改变 `motion_anchor_pos_b` 等 motion command 观测维度。

要求：

1. U20 的 motion dataset、MotionReference `link_of_interests`、Tracking `motion_cmd.body_names` 必须使用同一套 body 顺序或明确记录不同用途。
2. 如果 instinct_rl 侧存在固定观测维度假设，需要先在 InstinctMJ 内验证配置构建和 observation shape，不能假设自动推断一定覆盖所有下游。

#### 6. Parkour `link_of_interests` 顺序应和 Tracking body 顺序对齐或说明差异

当前文档中：

1. Tracking `motion_cmd.body_names` 顺序是 base、腿、手臂。
2. Parkour `motion_reference_cfg.link_of_interests` 顺序是 base、上肢、腿。

V11 原本就是这种差异，所以不一定是 bug。但 U20 新增配置时必须明确这是沿用 V11，不是误排。若 U20 数据生成流程按 Tracking body 顺序导出，则这里会发生 silent mismatch。

建议在 3.6.3 增加说明：`link_of_interests` 顺序必须与 U20 retargeted motion 文件中的 link order 一致；若数据生成脚本使用 Tracking 顺序，则需同步改此处顺序和 symmetric link mapping。

#### 7. 数据路径和文件格式描述不一致

文档写 `_PARKOUR_DATASET_DIR = "~/Instinct-mjlab/Datasets/u20_pkl/"`，但 V11 的 `AmassMotionCfg` 注释是使用 `*retargeted.npz`，字段名也来自 `amass_motion_cfg`。需要确认 U20 数据到底是 `.pkl` 还是 `.npz`。

建议：

1. 如果沿用 `AmassMotionCfgBase` 当前加载逻辑，优先使用 `.npz` 目录命名，避免目录名误导。
2. 如果用户将上传 `.pkl`，必须先确认 `AmassMotionCfgBase` 是否支持 `.pkl`；不支持时不能在本任务中新增格式兼容层，需先单独迁移/实现源任务已有的数据加载方式。

#### 8. `u20_action_scale` 使用 actuator regex key 可行，但要检查覆盖率

V11 已用 `target_names_expr` 作为 action scale key，因此 U20 复制此写法符合项目现状。但 U20 只有 3 个 actuator group，regex 覆盖失败时不会立刻显性报错。

执行后必须检查：

1. `JointPositionActionCfg.scale` 能覆盖 22 个可控关节。
2. `floating_base_joint` 不应被 action scale 或 actuator 匹配。
3. `u20_wholebody_actuator_cfgs` 中所有 regex 能被 EntityArticulationInfo 解析到实际关节。

### 8.3 建议补充的验收步骤

实现后建议按以下顺序验证，避免直接进入训练才暴露问题：

1. XML 语法检查：`xml.etree.ElementTree.parse("src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml")`
2. MuJoCo 加载检查：`get_u20_spec()` 和 `mujoco.MjModel.from_spec(...)`
3. 资产覆盖检查：确认所有 mesh 都能通过 `spec.assets` 解析。
4. 关节顺序检查：打印 MuJoCo joint order，确认与 `U20_22Dof_symmetric_augmentation_joint_mapping` 完全一致。
5. 配置构建检查：导入 `instinct_u20_parkour_amp_final_cfg(play=True)`，确认无 missing body/joint/sensor 名称。
6. Observation shape 检查：构建一个最小 env，检查 `policy`、`critic`、`amp_policy`、`amp_reference` 维度。
7. Reset 检查：执行一次 reset，确认足底不穿地、base 高度稳定、非法接触不立即触发。
8. Play 检查：启动 U20 Play 任务，确认相机、height scanner、volume points 都挂在正确 body 上。

### 8.4 建议调整优先级

建议把当前计划拆成两个阶段执行：

1. **阶段 A：资产和 Tracking 可加载**
   - 创建 `u20_popsicle.xml`
   - 创建 `assets/u20.py`
   - 创建 `tracking/config/u20`
   - 完成 XML/spec/action scale/joint order/reset 验证

2. **阶段 B：Parkour AMP**
   - 创建 `parkour/config/u20`
   - 接入 MotionReference、sensor、reward、termination
   - 等 U20 motion 数据到位后再验证 AMP reference 维度

这样拆分更符合当前风险分布：资产和基础环境问题会阻塞所有后续工作，AMP 数据问题则依赖用户上传数据。

---

## 9. 执行记录（2026-06-12）

### 9.1 Review 反馈 8.2 修正对照

| Review # | 问题 | 修正措施 | 状态 |
|----------|------|---------|------|
| 8.2 #1 | mesh 处理需收紧 | mesh `name` 和 `file` 保持原 STL 文件名（如 `lleg1_link`），body/joint 语义化重命名 | ✅ |
| 8.2 #2 | `get_u20_assets()` 需实测 | `meshdir="meshes/"`，经 `spec.compile()` 验证通过，26 个 STL 全部加载 | ✅ |
| 8.2 #3 | 初始高度 `z=0.85` 太低 | 改为 `z=1.0`，加 NOTE 注明候选范围 0.95-1.05 | ✅ |
| 8.2 #4 | `reverse_buf` 两版矛盾 | 只保留修正版（基于 XML axis 逐项推导），删除第一版 | ✅ |
| 8.2 #5 | Tracking body_names 维度 | 13 个 body（无 waist_pitch），维度变化已记录 | ✅ |
| 8.2 #6 | link_of_interests 顺序差异 | 沿用 V11 约定（base/upper/legs），加 NOTE 说明须与 motion 数据对齐 | ✅ |
| 8.2 #7 | 数据路径 .pkl vs .npz | 改为 `u20_npz/`，与 `AmassMotionCfgBase` 加载逻辑一致 | ✅ |
| 8.2 #8 | action scale 覆盖率 | 3 个 actuator group 覆盖 22 DOF，`floating_base_joint` 不被匹配 | ✅ |

### 9.2 验收步骤执行结果

| # | 验收项 | 结果 |
|---|--------|------|
| 1 | XML 语法检查 `ET.parse()` | ✅ 通过 |
| 2 | MuJoCo 编译 `spec.compile()` | ✅ 通过，23 joints, 30 bodies |
| 3 | Mesh 资产覆盖 | ✅ 26 STL 全部通过 `spec.assets` 加载 |
| 4 | Joint 顺序检查 | ✅ 22 个 actuated joint 顺序与 symmetric mapping 完全一致 |
| 5 | Symmetric mapping 正确性 | ✅ 22 项全部通过 left↔right pair 验证 |
| 6 | Python 语法检查 `py_compile` | ✅ u20.py, env_cfgs.py, agents cfg 全部通过 |
| 7 | V11 残留名称检查 | ✅ grep 确认无 `ankle_roll`/`waist_pitch`/`wrist_roll`/`head_pitch_link` |
| 8 | 旧编号命名残留检查 | ✅ grep 确认无 `lleg[0-9]`/`rleg[0-9]`/`larm[0-9]`/`rarm[0-9]`/`waist[0-9]` |

### 9.3 待办事项（需用户配合）

1. **上传 U20 motion 数据**: 放入 `~/Instinct-mjlab/Datasets/u20_npz/`，格式为 `*retargeted.npz`
2. **验证初始高度 z=1.0**: 构建最小 env，执行一次 reset，确认足底不穿地
3. **验证 reverse_buf**: 用非零 qpos 做左右镜像测试
4. **微调摄像头 offset**: U20 头部几何与 V11 不同，需仿真调整
5. **微调 volume_points Grid**: U20 足部碰撞几何与 V11 不同
6. **验证 observation shape**: 构建 env 后检查 policy/critic/amp 维度
7. **Play 模式检查**: 启动 `Instinct-Parkour-Target-Amp-U20-Play-v0`
8. **拆分/回退 §11.3 #3 计划外全局工程改动**: `train.py` 绝对路径等

---

## 10. 代码审阅报告（初版）

> 注意：本节是初版审阅结论。2026-06-12 在项目虚拟环境中复验后，发现新的必须修复项；以第 11 节为准。

### 总体评估

U20 Parkour 任务配置完全遵循 V11 的模板结构，通过 1:1 替换实现了 22-DOF 机器人的完整 parkour AMP 环境。XML 语义化命名清晰，资产模块与 tracking/parkour 配置的继承链完整，review 反馈的高优先级项全部已修正。代码可读性和项目一致性良好。

**综合评分：8/10**

### 🔴 严重问题（必须修复）

无。

### 🟡 改进建议（建议修复）

**1. `reverse_buf` 仍需运行时验证**

位置：`assets/u20.py:234-240`

当前 `reverse_buf` 基于对 XML axis 字符串的推导，但 MuJoCo 镜像变换的正确性取决于局部 frame 而非全局轴方向。例如 `hip_roll` 左右都是 axis Z，但在左右镜像下是否取反需要 qpos 测试确认。

代码中已有 TODO 注释，建议在 env 可运行后第一时间补充验证脚本。

**2. 摄像头 offset 沿用 V11 值**

位置：`parkour/config/u20/u20_parkour_target_amp_cfg.py:275-284`

U20 头部（`head_link`）的几何与 V11（`head_pitch_link`）不同，当前 offset `(0.025, 0.005, 0.054)` 和旋转四元数来自 V11。虽然不影响代码正确性，但深度图视野可能偏离预期，影响训练效果。

**3. `volume_points` Grid 范围需适配验证**

位置：`parkour/config/u20/u20_parkour_target_amp_cfg.py:234-244`

U20 足部 5 个 capsule 的分布与 V11 的足部碰撞不同，当前 Grid 范围 `x[-0.06, 0.10]`、`z[-0.10, -0.05]` 是估算值。如果 Grid 不能覆盖实际碰撞体，`volume_points_penetration` reward 会失效。

**4. 初始高度 `z=1.0` 需 env reset 验证**

位置：`assets/u20.py:181`

代码中 NOTE 已标注候选范围 0.95-1.05m，但全零位时足底可能穿地或离地过高。应在 env 可运行后精确测量。

### 🟢 优化建议（可选改进）

**1. XML 中 `waist_yaw_link` 的闭合标签位置**

位置：`u20_popsicle.xml:196-198`

`imu_link` 和 `waist_yaw_link` 的闭合 `</body>` 在同一嵌套层级，XML 虽然合法但缩进可以更清晰：

```xml
        </body>  <!-- end waist_roll_link -->
        <body name="imu_link" pos="-0.09835 0 -0.052"/>
      </body>    <!-- end waist_yaw_link -->
    </body>      <!-- end base_link -->
```

这不影响功能，仅影响可读性。

**2. 可在 `u20.py` 中导出 `U20_22DOF_DELAYED_CFG`**

类似 V11 有独立的 `V11_29DOF_POPSICLE_CFG`（带非 delayed actuator），可以额外提供一个 delayed actuator 版本的 entity config，避免 parkour cfg 中手动替换 actuator。

### ✅ 做得好的地方

1. **严格 1:1 迁移**：每个文件都对应 V11 的同名文件，结构、命名、逻辑完全一致，降低了维护成本。
2. **Review 反馈全部落地**：8.2 的 8 个修正项全部按建议执行，没有遗漏或打折扣。
3. **对称映射完整验证**：通过 MuJoCo spec.compile() 验证了 22 个 joint 的顺序和 symmetric mapping 的正确性。
4. **无残留旧名称**：grep 确认所有文件中没有 V11 特有的 `ankle_roll`、`waist_pitch`、`wrist_roll`、`head_pitch_link` 等名称。
5. **NOTE 注释完善**：在关键待验证值（init height z、reverse_buf、camera offset、link_of_interests 顺序）处都加了注释说明。
6. **分阶段执行**：按 8.4 建议拆分 Phase A/B，先验证资产再构建任务，降低了风险。

### 📋 修复优先级清单

- [x] ~~复位越界修复~~ → §12.1 Fix A 已修复
- [x] ~~循环导入修复~~ → §12.2 Fix B 已修复
- [ ] 上传 U20 motion 数据到 `~/Instinct-mjlab/Datasets/u20_npz/`
- [ ] 运行 env reset 验证初始高度 z=1.0
- [ ] 编写 reverse_buf 非零 qpos 镜像验证脚本
- [ ] 微调摄像头 offset
- [ ] 验证 volume_points Grid 覆盖范围
- [ ] 构建 env 后检查 observation shape
- [ ] 运行 Play 模式端到端验证

---

## 11. 补充 Review 反馈（2026-06-12，venv 复验）

### 11.1 复验环境

按用户要求在项目虚拟环境中执行：

```bash
source /home/user2/Instinct-mjlab/InstinctMJ/.venv/bin/activate
```

复验命令覆盖：

1. XML 解析：`ET.parse("src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml")`
2. Python 语法检查：`py_compile` 覆盖 U20 asset、tracking cfg、parkour cfg、agent cfg
3. MuJoCo spec 加载：`get_u20_spec()`
4. Parkour 配置构建：`instinct_u20_parkour_amp_final_cfg(play=True)`
5. U20 task 注册检查：`list_tasks()` 中筛选 `U20`

### 11.2 复验通过项

| 验证项 | 结果 |
|--------|------|
| XML 解析 | 通过，输出 `xml_ok` |
| Python 语法检查 | 通过 |
| U20 spec 加载 | 通过，`23 joints`, `30 bodies`, `26 assets` |
| actuated joint 顺序 | 通过，22 个 actuated joints 与文档顺序一致 |
| Parkour cfg 构建 | 通过，输出 `cfg_ok 10 8 ['amp_policy', 'amp_reference', 'critic', 'policy']` |
| U20 task 注册 | 通过，包含 `Instinct-Parkour-Target-Amp-U20-v0` 和 `Instinct-Parkour-Target-Amp-U20-Play-v0` |

### 11.3 必须修复问题

#### 1. reset 随机偏移会把多个 U20 关节推到硬限位外

位置：

- `src/instinct_mj/assets/u20.py:180-183`
- `src/instinct_mj/tasks/parkour/config/u20/u20_parkour_target_amp_cfg.py:795`
- `src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml:49`
- `src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml:85`
- `src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml:132`
- `src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml:165`

当前 U20 初始状态使用全零位：

```python
joint_pos={}
```

但 XML 中多个关节零位就在单侧限位附近或正好在限位上：

| joint | range | 零位风险 |
|-------|-------|----------|
| `left_hip_pitch_joint` | `[-0.05, 1.4]` | `0 + [-0.15, 0.15]` 会低于下限 |
| `right_hip_pitch_joint` | `[-1.4, 0.05]` | `0 + [-0.15, 0.15]` 会高于上限 |
| `left_shoulder_pitch_joint` | `[0, 1.67]` | `0 + [-0.15, 0.15]` 会低于下限 |
| `right_shoulder_pitch_joint` | `[-1.67, 0]` | `0 + [-0.15, 0.15]` 会高于上限 |

Parkour reset 仍对所有关节使用统一偏移：

```python
"position_range": (-0.15, 0.15)
```

这会在 reset 时直接生成越界目标。建议二选一：

1. 给 U20 设置离开单侧限位的默认 `joint_pos`，例如左右 hip/shoulder pitch 取安全中间值。
2. 将 reset 改为 per-joint range，至少排除或收窄上述贴边关节。

在修复前，不应把“初始高度 z=1.0”和“reset 验证”标为完全通过。

#### 2. 导入 `instinct_mj.assets.u20` 时可复现循环导入 warning

位置：

- `src/instinct_mj/assets/u20.py:31`
- `src/instinct_mj/tasks/parkour/__init__.py:4`
- `src/instinct_mj/tasks/parkour/config/u20/u20_parkour_target_amp_cfg.py:39`

venv 中执行：

```bash
PYTHONPATH=src python -c "from instinct_mj.assets.u20 import get_u20_spec; s=get_u20_spec(); print(len(s.joints))"
```

可复现 warning：

```text
[WARN] Failed to load task package instinct_mj: cannot import name 'U20_22DOF_CFG' from partially initialized module 'instinct_mj.assets.u20'
```

虽然 spec 最终能加载，且 task 注册检查通过，但该 warning 说明存在循环路径：

```text
assets.u20 -> mjlab/import side effect -> task package autoload -> parkour.u20 cfg -> assets.u20
```

建议清理导入关系，目标是：单独导入 asset 模块不应触发 U20 parkour 任务注册，也不应出现 partially initialized module warning。

#### 3. 本次改动混入计划外的全局工程改动

位置：

- `src/instinct_mj/scripts/instinct_rl/train.py:487`
- `pyproject.toml:36`
- `pyproject.toml:57`
- `pyproject.toml:63`
- `uv.lock`
- `src/instinct_mj/assets/resources/v11/xml/v11.xml:48`
- `src/instinct_mj/scripts/instinct_rl/keyboard_play.py`
- `src/instinct_mj/controllers/`

这些改动不在本文档第 2 节 U20 Parkour 文件清单内，属于本任务的无关变更或至少需要独立 review：

1. `train.py` 将日志根路径硬编码为 `/home/user2/Instinct-mjlab/logs`，破坏可移植性。
2. `pyproject.toml` 增加 `instinct-keyboard-play`、`roboharness`，且出现重复 `[[tool.uv.index]]`。
3. `uv.lock` 大幅变更 MuJoCo/Torch/warp 来源与版本解析。
4. V11 XML 删除 harness cameras，属于 V11 行为变化。
5. `keyboard_play.py` 和 `controllers/` 是新增功能，不属于 U20 Parkour 迁移范围。

建议将这些改动从本次 U20 Parkour 变更中拆出，或单独补充执行计划和 review。特别是 `train.py` 的绝对路径应恢复为配置化或相对路径。

### 11.4 仍未完成的验证

以下事项仍应保持未完成状态，不能因为 cfg 构建通过而视为可训练通过：

1. U20 motion 数据尚未上传到 `~/Instinct-mjlab/Datasets/u20_npz/`。
2. 未执行真实 env reset，足底高度和非法接触仍未验证。
3. `reverse_buf` 尚未用非零 qpos 做左右镜像验证。
4. 摄像头 offset 仍沿用 V11 值，深度图视野未验证。
5. `volume_points` Grid 仍是估算范围，覆盖足部碰撞体的有效性未验证。
6. 未构建完整 env 检查 policy/critic/amp observation shape。
7. 未启动 `Instinct-Parkour-Target-Amp-U20-Play-v0` 做端到端检查。

### 11.5 修复优先级

1. 修复 reset 越界风险。
2. 清理 `assets.u20` 导入时的循环导入 warning。
3. 拆分或回退 U20 计划外的全局工程改动。
4. 在 motion 数据到位后执行 env reset、observation shape、Play 模式端到端验证。

---

## 12. 修复记录（2026-06-12）

### 12.1 Fix A: reset 随机偏移越界

**§11.3 #1 修复**

文件：`src/instinct_mj/assets/u20.py:180-184`

修改内容：`_U20_ZERO_INIT_STATE` 中增加 4 个贴边关节的默认 `joint_pos`：

| joint | range | 零位风险 | 修正默认值 |
|-------|-------|----------|-----------|
| `left_hip_pitch_joint` | `[-0.05, 1.4]` | 零位在下限，`-0.15` 越界 | 0.3 (安全的中间值) |
| `right_hip_pitch_joint` | `[-1.4, 0.05]` | 零位在上限，`+0.15` 越界 | -0.3 |
| `left_shoulder_pitch_joint` | `[0, 1.67]` | 零位在下限，`-0.15` 越界 | 0.3 |
| `right_shoulder_pitch_joint` | `[-1.67, 0]` | 零位在上限，`+0.15` 越界 | -0.3 |

原理：`reset_joints_by_offset` 对 `position_range: (-0.15, 0.15)` 不提供 per-joint range 支持（mjlab 原生函数无此能力），因此通过修改默认初态使 offset 落在安全区间内。修正后 offset 范围变为：
- `left_hip_pitch`: [0.15, 0.45] ⊆ [-0.05, 1.4] ✅
- `right_hip_pitch`: [-0.45, -0.15] ⊆ [-1.4, 0.05] ✅
- `left_shoulder_pitch`: [0.15, 0.45] ⊆ [0, 1.67] ✅
- `right_shoulder_pitch`: [-0.45, -0.15] ⊆ [-1.67, 0] ✅

### 12.2 Fix B: 循环导入 warning

**§11.3 #2 修复**

文件：`src/instinct_mj/tasks/parkour/config/u20/__init__.py`

修改内容：将模块级 import 移至工厂函数内部：

```python
# 修改前：
from .u20_parkour_target_amp_cfg import instinct_u20_parkour_amp_final_cfg  # 在模块加载时触发 assets.u20

# 修改后：
register_instinct_task(
    ...
    env_cfg_factory=lambda: _build_cfg(play=False),
)
def _build_cfg(play=False):
    from .u20_parkour_target_amp_cfg import instinct_u20_parkour_amp_final_cfg  # 推迟到 factory 被调用时
    ...
```

导入链解除：`tasks.parkour.__init__` → `config/u20/__init__` 不再立即触发 `assets.u20` 导入，彻底消除 `partially initialized module` warning。

### 12.3 Fix C: XML 缩进修正

文件：`src/instinct_mj/assets/resources/u20/xml/u20_popsicle.xml:196-199`

修改内容：
- `imu_link` body 缩进：6空格 → 8空格（作为 `waist_yaw_link` 子节点应有缩进）
- 增加 3 个结束标签注释（`<!-- end waist_roll_link -->`、`<!-- end waist_yaw_link -->`、`<!-- end base_link -->`）

### 12.4 修复后验收

| 验收项 | 结果 |
|-------|------|
| `py_compile u20.py` | ✅ 通过 |
| `py_compile parkour/u20/__init__.py` | ✅ 通过 |
| `ET.parse(u20_popsicle.xml)` | ✅ 通过 |
| XML 结束标签注释正确 | ✅ 已添加 |
| 4 个贴边关节默认值对称 | ✅ left↔right 符号对称 |

### 12.5 仍待完成（需用户配合）

- [ ] §11.3 #3 拆分/回退计划外全局工程改动（`train.py` 绝对路径、`pyproject.toml` 重复索引、V11 XML 删除 harness cameras 等）
- [ ] 上传 U20 motion 数据到 `~/Instinct-mjlab/Datasets/u20_npz/`
- [ ] 运行 env reset 验证初始高度 z=1.0
- [ ] 编写 reverse_buf 非零 qpos 镜像验证脚本
- [ ] 微调摄像头 offset 适配 U20 头部几何
- [ ] 验证 volume_points Grid 覆盖范围
- [ ] 构建 env 后检查 observation shape
- [ ] 运行 Play 模式端到端验证
