# V11 Parkour 模块构造技术文档

> 任务: `Instinct-Parkour-Target-Amp-V11-v0`
> 日期: 2026-03-17

---

## 目录

1. [模块层级总览](#0-模块层级总览)
2. [Config 层 — 环境配置](#1-config-层--环境配置)
3. [Asset 层 — V11 机器人](#2-asset-层--v11-机器人)
4. [Actuator 层 — 执行器](#3-actuator-层--执行器)
5. [Sensor 层 — 传感器](#4-sensor-层--传感器)
6. [Motion Reference 层 — 运动参考](#5-motion-reference-层--运动参考)
7. [MDP 层 — 任务逻辑](#6-mdp-层--任务逻辑)
8. [RL Wrapper 层 — 环境适配器](#7-rl-wrapper-层--环境适配器)
9. [Policy Network 层 — 策略网络](#8-policy-network-层--策略网络)
10. [Algorithm 层 — 训练算法](#9-algorithm-层--训练算法)
11. [Runner 层 — 训练主循环](#10-runner-层--训练主循环)
12. [模块依赖图](#11-模块依赖图)

---

## 0. 模块层级总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Runner (训练主循环)                            │
│  instinct_rl.runners.OnPolicyRunner                                 │
├─────────────────────────────────────────────────────────────────────┤
│              Algorithm: WasabiPPO                                   │
│  instinct_rl.algorithms.{PPO ← WasabiAlgoMixin}                    │
├─────────────────────────────────────────────────────────────────────┤
│              Policy Network: EncoderMoEActorCritic                  │
│  instinct_rl.modules.{EncoderActorCriticMixin + MoEActorCritic}    │
├─────────────────────────────────────────────────────────────────────┤
│              RL Wrapper (VecEnv 适配器)                              │
│  instinct_mj.rl.InstinctRlVecEnvWrapper                            │
├─────────────────────────────────────────────────────────────────────┤
│              MDP 层 (奖励/终止/事件/课程/指令)                        │
│  instinct_mj.tasks.parkour.mdp.*                                   │
├───────────────────────┬─────────────────────────────────────────────┤
│  Sensor 层            │  Motion Reference 层                        │
│  - NoisyCamera        │  MotionReferenceManager                     │
│  - VolumePoints       │  AmassMotion                                │
│  - ContactSensor      │  MotionReferenceData                        │
│  - RayCastSensor      │                                             │
├───────────────────────┴─────────────────────────────────────────────┤
│              Asset 层: V11 (29-DOF) + Actuators                     │
│  instinct_mj.assets.v11, instinct_mj.actuators                     │
├─────────────────────────────────────────────────────────────────────┤
│              MuJoCo 物理引擎 (200 Hz)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Config 层 — 环境配置

### 1.1 `v11_parkour_target_amp_cfg.py`

**入口函数**

```python
def instinct_v11_parkour_amp_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg
def instinct_v11_parkour_amp_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg
```

**构建流程**

```
unitree_g1_flat_tracking_env_cfg(play, has_state_estimation=True)   # 基础模板
  └→ 替换 robot entity → V11_29DOF_POPSICLE_CFG
  └→ 替换 actuators → beyondmimic_v11_wholebody_delayed_actuator_cfgs
  └→ 替换 action_scale → beyondmimic_action_scale
  └→ 覆盖 scene (num_envs=2048, env_spacing=2.5)
  └→ 覆盖 sim (timestep=0.005, decimation=4, nconmax=128, njmax=700)
  └→ 挂载 sensors (camera, contact, volume_points, height_scanner, motion_reference)
  └→ 设置 observations (policy, critic, amp_policy, amp_reference)
  └→ 设置 rewards (21项)
  └→ 设置 curriculum, terminations, events
```

**`ManagerBasedRlEnvCfg` 关键字段**

| 字段 | 类型 | V11 Parkour 值 |
|------|------|---------------|
| `sim.mujoco.timestep` | float | `0.005`（200 Hz）|
| `decimation` | int | `4`（50 Hz 控制）|
| `episode_length_s` | float | `20.0`（train）/ `10.0`（play）|
| `scene.num_envs` | int | `2048`（train）/ `10`（play）|
| `scene.env_spacing` | float | `2.5` m |
| `sim.mujoco.iterations` | int | `10` |
| `sim.mujoco.ls_iterations` | int | `20` |
| `sim.mujoco.ccd_iterations` | int | `128` |
| `sim.nconmax` | int | `128` |
| `sim.njmax` | int | `700` |

---

### 1.2 `AmassMotionCfg`

```python
@dataclass(kw_only=True)
class AmassMotionCfg(AmassMotionCfgBase):
    path: str                             # ~/Datasets/hiking-in-the-wild_Data&Model/...
    filtered_motion_selection_filepath: str  # parkour_motion_without_run.yaml
    motion_start_from_middle_range: [0.0, 0.9]
    motion_start_height_offset: 0.0
    ensure_link_below_zero_ground: False
    buffer_device: "output_device"
    motion_interpolate_func: motion_interpolate_bilinear
    velocity_estimation_method: "frontward"
```

---

### 1.3 `motion_reference_cfg` (MotionReferenceManagerCfg)

```python
MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=V11_XML_PATH,
    link_of_interests=[...],        # 14个关键链接
    symmetric_augmentation_link_mapping=[0,1,3,2,...],
    symmetric_augmentation_joint_mapping=list(V11_29Dof...),
    symmetric_augmentation_joint_reverse_buf=list(...),
    frame_interval_s=0.02,          # 50 Hz
    update_period=0.02,
    num_frames=10,                  # 10帧历史窗口
    motion_buffers={"run_walk": AmassMotionCfg()},
    mp_split_method="Even",
)
```

---

### 1.4 `PoseVelocityCommandCfg`

```python
PoseVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(8.0, 12.0),
    velocity_control_stiffness=2.0,
    heading_control_stiffness=2.0,
    rel_standing_envs=0.05,
    ranges=Ranges(
        lin_vel_x=(0.0, 0.0),   # 平地静止, 地形相关覆盖
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
    ),
    random_velocity_terrain=["perlin_rough_stand"],
    velocity_ranges={
        "perlin_rough":     lin_vel_x=(0.45,1.0), ...
        "square_gaps":      lin_vel_x=(0.45,0.8), ...
        "pyramid_stairs*":  lin_vel_x=(0.45,0.8), ...
        ...
    },
    only_positive_lin_vel_x=True,
    target_dis_threshold=0.4,
)
```

---

## 2. Asset 层 — V11 机器人

**文件**: `InstinctMJ/src/instinct_mj/assets/v11.py`

### 2.1 DOF 布局（29维）

```
MJCF 关节顺序（v11_v4.xml 树遍历）:
  0-5:  left_hip_pitch/roll/yaw, left_knee, left_ankle_pitch/roll
  6-11: right_hip_pitch/roll/yaw, right_knee, right_ankle_pitch/roll
  12-14: waist_yaw/roll/pitch
  15-21: left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll/yaw/pitch
  22-28: right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/yaw/pitch
```

### 2.2 `EntityCfg` 构造

```python
V11_29DOF_POPSICLE_CFG = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint":       0.669,
            ".*_ankle_pitch_joint": -0.363,
            "left_shoulder_pitch_joint":  0.2,
            "right_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint":   0.2,
            "right_shoulder_roll_joint": -0.2,
            ".*_elbow_joint":            0.6,
        },
        joint_vel={".*": 0.0},
    ),
    spec_fn=get_v11_spec,   # 从 v11_v4.xml 加载 MjSpec
    articulation=EntityArticulationInfoCfg(
        actuators=(...),
        soft_joint_pos_limit_factor=0.9,
    ),
)
```

### 2.3 对称增强映射（29 DOF）

```python
V11_29Dof_symmetric_augmentation_joint_mapping = [
    6,7,8,9,10,11,   # 0-5:  left leg  → right leg
    0,1,2,3,4,5,     # 6-11: right leg → left leg
    12,13,14,        # 12-14: waist    → self (midline)
    22,23,24,25,26,27,28,  # 15-21: left arm  → right arm
    15,16,17,18,19,20,21,  # 22-28: right arm → left arm
]

V11_29Dof_symmetric_augmentation_joint_reverse_buf = [
    1,-1,-1, 1, 1,-1,  # left leg: pitch/knee/ankle_pitch 同向, roll/yaw/ankle_roll 反向
    1,-1,-1, 1, 1,-1,  # right leg: 同上
    -1,-1, 1,          # waist: yaw/roll 反向, pitch 同向
    1,-1,-1, 1,-1,-1,1,  # left arm
    1,-1,-1, 1,-1,-1,1,  # right arm
]
```

---

## 3. Actuator 层 — 执行器

**文件**: `InstinctMJ/src/instinct_mj/actuators/actuator_cfg.py`

### 3.1 类层次

```
BuiltinPositionActuatorCfg (mjlab)
  └─ InstinctActuatorCfg
        └─ (wrapped by) DelayedInstinctActuatorCfg
```

### 3.2 `InstinctActuatorCfg`

```python
@dataclass(kw_only=True)
class InstinctActuatorCfg(BuiltinPositionActuatorCfg):
    velocity_limit: float        # 额外的关节速度上限元数据
    # 继承字段:
    target_names_expr: tuple     # 正则匹配关节名
    effort_limit: float          # 力矩上限 [Nm]
    stiffness: float             # PD 刚度 [Nm/rad]
    damping: float               # PD 阻尼 [Nm·s/rad]
    armature: float              # 电机转子惯量 [kg·m²]
```

### 3.3 `DelayedInstinctActuatorCfg`

```python
@dataclass(kw_only=True)
class DelayedInstinctActuatorCfg(DelayedActuatorCfg):
    base_cfg: InstinctActuatorCfg
    delay_target: Literal["position"] = "position"

    @property
    def velocity_limit(self) -> float:
        return self.base_cfg.velocity_limit
```

**延迟范围**: `delay_min_lag=0, delay_max_lag=2`（0~2 控制步 = 0~40 ms）

### 3.4 V11 各关节组参数

| 执行器组 | 匹配关节 | 力矩限 (Nm) | 速度限 (rad/s) | 电机类型 |
|---------|---------|-----------|--------------|--------|
| LEGS_PITCH_YAW | `.*_hip_pitch/yaw` | 150 | 14.66 | 7520-14 |
| LEGS_ROLL_KNEE | `.*_hip_roll, .*_knee` | 150 | 14.66 | 7520-22 |
| FEET | `.*_ankle_pitch/roll` | 54 | 9.32 | 5020×2 |
| WAIST | `waist_yaw/roll/pitch` | 90 | 14.66 | 7520-14 |
| ARMS | `.*_shoulder_*, .*_elbow` | 80 | 14.66 | 7520-14 |
| WRISTS | `.*_wrist_*` | 11.5 | 22.0 | 4010 |

**Action Scale 公式**: `scale[joint] = 0.25 × effort_limit / stiffness`

---

## 4. Sensor 层 — 传感器

### 4.1 ContactSensor（三个实例）

```
contact_forces          — 踝部接触（left/right ankle_roll_link）
                           history_length=3, track_air_time=True
torso_contact_forces    — 躯干 base_link ↔ terrain
                           history_length=3
undesired_contact_forces — 所有非踝部身体
                           history_length=3
```

**输出字段**: `found` (bool), `force` (3-vector), `air_time`（仅 contact_forces）

---

### 4.2 `VolumePoints`（足部穿透检测）

**文件**: `InstinctMJ/src/instinct_mj/sensors/volume_points/volume_points.py`

```python
VolumePointsCfg(
    name="leg_volume_points",
    entity_name="robot",
    body_names=".*_ankle_roll_link",      # 正则匹配左右踝
    points_generator=Grid3dPointsGeneratorCfg(
        x_min=-0.075, x_max=0.155, x_num=14,   # 约 1.8cm 间距
        y_min=-0.04,  y_max=0.04,  y_num=5,
        z_min=-0.065, z_max=-0.04, z_num=2,
    ),
    debug_vis=True,  # play 模式开启
)
```

**每足点数**: 14×5×2 = **140 点**

**关键方法**

| 方法 | 输入 | 输出 shape |
|------|------|-----------|
| `initialize(mj_model, model, data, device)` | MuJoCo 模型 | — |
| `_refresh_volume_points()` | body xpos/xquat | `points_pos_w` (N,B,P,3) |
| `_refresh_penetration_offset()` | points_pos_w + 虚拟障碍 | `penetration_offset` (N,B,P,3) |
| `register_virtual_obstacles()` | obstacle buffers | — |

**`VolumePointsData` 字段**

```python
@dataclass
class VolumePointsData:
    pos_w:               (num_envs, num_bodies, 3)      # body 位置
    quat_w:              (num_envs, num_bodies, 4)      # body 姿态
    points_pos_w:        (num_envs, num_bodies, P, 3)   # 点云世界坐标
    points_vel_w:        (num_envs, num_bodies, P, 3)   # 点云速度
    penetration_offset:  (num_envs, num_bodies, P, 3)   # 穿透深度向量
```

---

### 4.3 `RayCastSensor`（高度扫描）

```python
RayCastSensorCfg(
    name="left_height_scanner",
    frame=ObjRef(type="body", name="left_ankle_roll_link", entity="robot"),
    pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),  # 1条射线
    ray_alignment="yaw",
    max_distance=10.0,
)
```

每侧 **1条竖直射线**，输出地面高度差，用于 `feet_at_plane` 奖励。

---

### 4.4 `NoisyGroupedRayCasterCamera`（深度相机）

**文件**: `InstinctMJ/src/instinct_mj/sensors/noisy_camera/`

```python
NoisyGroupedRayCasterCameraCfg(
    name="camera",
    frame=ObjRef(type="body", name="waist_pitch_link", entity="robot"),
    pattern=PinholeCameraPatternCfg(width=64, height=36, fovy=58.29),
    focal_length=1.0,
    horizontal_aperture=2*tan(rad(89.51)/2),
    vertical_aperture=2*tan(rad(58.29)/2),
    ray_alignment="yaw",
    offset=OffsetCfg(
        pos=(0.0488, 0.01, 0.4378),     # 腰部俯仰链接偏移
        rot=(0.9135, 0.0044, 0.4067, 0.0),
        convention="world",
    ),
    data_types=["distance_to_image_plane"],
    depth_clipping_behavior="max",
    noise_pipeline={
        "crop_and_resize": CropAndResizeCfg(crop_region=(18,0,16,16)),   # → 16×16
        "gaussian_blur":   GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
        "depth_normalization": DepthNormalizationCfg(
            depth_range=(0.0, 2.5), normalize=True, output_range=(0.0,1.0)
        ),
    },
    data_histories={"distance_to_image_plane_noised": 37},  # 740ms 历史
    min_distance=0.1,
    max_distance=2.5,
)
```

**噪声流水线（NoisyCameraMixin 执行）**

```
原始 64×36 px
    ↓ CropAndResize(crop_region=(18,0,16,16))
  16×16 px
    ↓ GaussianBlur(kernel=3, σ=1)
  16×16 px (模糊)
    ↓ DepthNormalization([0,2.5] → [0,1])
  16×16 px (归一化)
    ↓ 存入 AsyncCircularBuffer (37帧)
  "distance_to_image_plane_noised_history": (N, 37, 16, 16)
```

**关键方法**

| 方法 | 作用 |
|------|------|
| `build_noise_pipeline()` | 初始化各 ImageNoiseCfg 对象 |
| `apply_noise_pipeline(data, env_ids)` | 顺序执行噪声变换 |
| `build_history_buffers()` | 创建 AsyncCircularBuffer |
| `update_history_buffers(env_ids)` | 追加当前帧到历史 |

---

## 5. Motion Reference 层 — 运动参考

### 5.1 `MotionReferenceManager`（传感器式调度器）

**文件**: `InstinctMJ/src/instinct_mj/motion_reference/motion_reference_manager.py`

```python
class MotionReferenceManager:
    def __init__(cfg: MotionReferenceManagerCfg):
        # 从 V11 XML 构建前向运动学
        # 初始化各 MotionBuffer（AMASS 数据集）
        # 分配每个 env 的运动序列
```

**数据更新流程（每 20ms 控制步）**

```
self._sample_reference_motion()        # 从 motion_buffer 取当前帧
  → FK(joint_pos) → link poses
  → 存入 MotionReferenceData
self.fill_motion_data(env_ids, t, ...)  # 写入多帧滑动窗口
```

**输出的 `MotionReferenceData`**

```python
@dataclass
class MotionReferenceData:
    joint_pos:       (num_envs, num_frames, 29)   # 关节位置
    joint_vel:       (num_envs, num_frames, 29)   # 关节速度
    base_pos_w:      (num_envs, num_frames, 3)    # 根节点位置
    base_quat_w:     (num_envs, num_frames, 4)    # 根节点姿态 (wxyz)
    base_lin_vel_w:  (num_envs, num_frames, 3)    # 线速度
    base_ang_vel_w:  (num_envs, num_frames, 3)    # 角速度
    link_pos_w/b:    (num_envs, num_frames, 14, 3) # 链接位置
    link_quat_w/b:   (num_envs, num_frames, 14, 4) # 链接姿态
    link_lin_vel_w/b:(num_envs, num_frames, 14, 3) # 链接线速度
    link_ang_vel_w/b:(num_envs, num_frames, 14, 3) # 链接角速度
    time_to_target_frame: (num_envs, num_frames)   # 到各帧的时间差
```

---

### 5.2 `MotionBuffer`（抽象基类）

```python
class MotionBuffer(ABC):
    def __init__(
        self,
        cfg,
        articulation_view,
        link_of_interests: list[str],
        forward_kinematics_func,  # (joint_pos: [N,J]) → [N,L,7]
        device: str,
    ):
```

**抽象接口**

| 方法 | 签名 | 说明 |
|------|------|------|
| `reset` | `(env_ids, sym_mask)` | 重置 env 运动状态 |
| `fill_init_reference_state` | `(env_ids, origins, state_buf)` | 初始化 env 动作帧 |
| `fill_motion_data` | `(env_ids, timestamps, origins, data_buf)` | 填充多帧数据 |
| `get_current_motion_identifiers` | `(env_ids)` | 返回运动文件标识 |

---

### 5.3 `AmassMotion`（AMASS 数据集实现）

```python
class AmassMotion(MotionBuffer):
    def __init__(cfg: AmassMotionCfg, ...):
        self._refresh_motion_file_list()      # 扫描目录, 构建文件列表
        self._prepare_retargetting_func()     # 加载重定向函数
        # 加载运动到 _all_motion_sequences (ConcatBatchTensor)
```

**`MotionSequence` 数据结构（内存中存储的完整序列）**

```python
@dataclass
class MotionSequence:
    joint_pos:       (N, T, 29)   # N=序列数, T=帧数
    joint_vel:       (N, T, 29)
    base_pos_w:      (N, T, 3)
    base_quat_w:     (N, T, 4)   # wxyz
    base_lin_vel_w:  (N, T, 3)
    base_ang_vel_w:  (N, T, 3)
    link_pos_w/b:    (N, T, 14, 3)
    link_quat_w/b:   (N, T, 14, 4)
    link_lin_vel_w/b:(N, T, 14, 3)
    link_ang_vel_w/b:(N, T, 14, 3)
    framerate:       (N,)         # 每序列帧率
    buffer_length:   (N,)         # 序列长度
```

**采样机制**

```
_assigned_env_motion_selection  (num_envs,)  → 每 env 分配哪条轨迹
sample_start_times()            → Warp 核函数，从 CDF 加权采样起始帧
motion_start_from_middle_range  = [0.0, 0.9] → 起始帧在前 90% 内随机
```

**对称增强**

```
输入: motion frame (joint_pos, link_poses)
  ↓ joint_mapping[i] = j   →  左右对调
  ↓ joint_reverse_buf[i]   →  翻转轴方向
输出: 镜像运动帧（数据集翻倍）
```

---

## 6. MDP 层 — 任务逻辑

### 6.1 `PoseVelocityCommand`

**文件**: `InstinctMJ/src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py`

```python
class PoseVelocityCommand(CommandTerm):
    def __init__(cfg: PoseVelocityCommandCfg, env: ManagerBasedRlEnv):
```

**内部缓冲区**

| 缓冲区 | Shape | 说明 |
|--------|-------|------|
| `pos_command_w` | (N, 3) | 世界坐标目标位置 |
| `vel_command_b` | (N, 3) | 基座坐标系速度指令 |
| `heading_command_w` | (N,) | 目标朝向 (rad) |
| `max_command_b` | (N, 3) | 当前地形允许的最大速度 |
| `valid_targets` | (L, T, P, 3) | 地形平坦区域采样点 |

**`command` 属性**: 返回 `vel_command_b` (N, 3) → `[lin_vel_x, lin_vel_y, ang_vel_z]`

**指令更新流程**

```python
def _resample_command(env_ids):
    # 1. 从 terrain.flat_patches 采样目标位置
    # 2. 根据地形类型查表 velocity_ranges
    # 3. 随机采样速度幅值
    # 4. 更新 max_command_b

def _update_command():
    # 1. 计算 robot → target 向量
    # 2. 刚度控制: vel_cmd ∝ (target - robot_pos) * stiffness
    # 3. 裁剪到 max_command_b
    # 4. 更新 heading_command_w
```

---

### 6.2 奖励函数（`rewards.py`）

所有函数签名: `(env: ManagerBasedRlEnv, ...) -> Tensor[N]`

| 函数 | 权重 | 公式 / 机制 |
|------|------|-----------|
| `track_lin_vel_xy_exp` | +2.0 | `exp(-(‖v_cmd - v_act‖²)/std²)` |
| `track_ang_vel_z_exp` | +2.0 | `exp(-(ω_cmd - ω_act)²/std²)` |
| `heading_error` | −1.0 | 机器人朝向误差（rad） |
| `dont_wait` | −0.5 | 有指令但静止的惩罚 |
| `is_alive` | +3.0 | 存活每步 +1（mjlab 内置） |
| `stand_still` | −0.3 | 无指令时仍在运动的惩罚 |
| `volume_points_penetration` | −4.0 | 足部点云穿透深度累加 |
| `feet_air_time` | +0.5 | 腾空时长奖励（促进跨步） |
| `feet_slide` | −0.4 | 接触时足部水平速度惩罚 |
| `joint_deviation_hip` | −0.5 | 髋关节偏离零位的平方 |
| `ang_vel_xy_l2` | −0.05 | 躯干滚转/俯仰角速度 L2 |
| `dof_torques_l2` | −1.5e-7 | 腿部力矩 L2（能量） |
| `dof_acc_l2` | −1.25e-7 | 全关节加速度 L2 |
| `dof_vel_l2` | −1e-4 | 全关节速度 L2 |
| `action_rate_l2` | −0.005 | 动作变化率 L2（抖动） |
| `flat_orientation_l2` | −3.0 | 躯干姿态水平度 L2 |
| `pelvis_orientation_l2` | −3.0 | base_link 朝向 L2 |
| `feet_flat_ori` | −0.4 | 接触时脚底倾斜惩罚 |
| `feet_at_plane` | −0.1 | 脚底相对地面高度误差 |
| `feet_close_xy` | +0.4 | 双脚 XY 间距高斯奖励 |
| `energy` | −5e-5 | 电机功率平方（归一化刚度）|
| `dof_pos_limits` | −1.0 | 关节位置超限惩罚 |
| `dof_vel_limits` | −1.0 | 关节速度超限（soft 90%）|
| `torque_limits` | −0.01 | 力矩超限（80% 阈值）|
| `undesired_contacts` | −1.0 | 非踝部接触惩罚 |

---

### 6.3 终止条件（`terminations.py`）

```python
"time_out"            → time_out=True,  episode_length_s=20s
"terrain_out_bound"   → time_out=True,  distance_buffer=2.0m
"base_contact"        → base_link 与地形接触力 > 1N
"bad_orientation"     → 倾斜角 > 1.0 rad
"root_height"         → 质心高度 < 0.5m (play 时禁用)
"dataset_exhausted"   → time_out=True, 运动序列耗尽
```

---

### 6.4 课程学习（`curriculums.py`）

```python
"terrain_levels": CurriculumTermCfg(
    func=parkour_mdp.tracking_exp_vel,
    params={
        "lin_vel_threshold": (0.3, 0.6),   # 升级条件
        "ang_vel_threshold": (0.0, 0.0),
    }
)
```

**机制**: 按 env 的速度跟踪指数均值决定地形等级升降。

---

### 6.5 随机化事件（`events.py`）

| 事件 | 模式 | 参数 |
|------|------|------|
| `physics_material` | startup | 摩擦系数 [0.3, 1.6] 随机 |
| `reset_base` | reset | 位置 ±0.1m，速度 ±0.2 m/s |
| `register_virtual_obstacles` | startup | 注册虚拟障碍到 VolumePoints |
| `reset_robot_joints` | reset | 关节位置 ±0.15 rad |

---

## 7. RL Wrapper 层 — 环境适配器

**文件**: `InstinctMJ/src/instinct_mj/rl/vecenv_wrapper.py`

```python
class InstinctRlVecEnvWrapper(VecEnv):
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        policy_group: str = "policy",
        critic_group: str | None = "critic",
    ):
```

**初始化时完成**

```python
self.num_envs    = env.num_envs
self.num_actions = env.action_manager.total_action_dim  # = 29
self.num_rewards = env.num_rewards                       # = 1
self.num_obs     = _group_flat_dim("policy")             # = 896
self.num_critic_obs = _group_flat_dim("critic")          # = 920
self.env.reset()  # 初始化一次（runner 不会自动 reset）
```

**关键接口**

| 方法 | 签名 | 返回 |
|------|------|------|
| `get_obs_format()` | `→ dict[group, dict[term, shape]]` | 观测格式描述 |
| `get_obs_segments(group)` | `→ dict[term, shape]` | 单组观测格式 |
| `step(actions)` | `actions: (N,29)` | `(obs, rewards, dones, extras)` |
| `get_observations()` | — | `(policy_obs, {observations: ...})` |
| `reset()` | — | `(obs, extras)` |

**`_pack_observations` 流程**

```python
def _pack_observations(obs_dict: dict) -> dict[str, Tensor]:
    for group_name in _group_map:
        packed[group_name] = _flatten_group(group_name, obs_dict[group_name])
    # 额外透传 amp_policy, amp_reference 等组
    return packed

def _flatten_group(group_name, group_obs) -> Tensor:
    # 按 active_terms 顺序拼接，每个 term flatten(start_dim=1)
    return torch.cat([term.flatten(1) for term in active_terms], dim=1)
```

**多奖励处理**

```python
if isinstance(rewards, dict):
    rewards = torch.stack(list(rewards.values()), dim=-1)  # (N, num_rewards)
if rewards.ndim == 1:
    rewards = rewards.unsqueeze(1)                          # (N, 1)
```

---

## 8. Policy Network 层 — 策略网络

### 8.1 `ActorCritic`（基类）

**文件**: `instinct_rl/instinct_rl/modules/actor_critic.py`

```python
class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_format: dict[str, dict[str, tuple]],
        num_actions: int,
        actor_hidden_dims: list = [256, 256, 256],
        critic_hidden_dims: list = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        num_rewards: int = 1,
        mu_activation: str | None = None,
    ):
```

**网络结构**

```
Actor MLP:
  input_dim → [256, 128, 64] → num_actions
  每层: Linear + ELU（最后层无激活）

Critic MLP:
  input_dim → [256, 128, 64] → 1 (或 num_rewards)
  多奖励: ModuleList 多个 critic 头

Action Std:
  nn.Parameter(ones(num_actions) * init_noise_std)
  训练可学习
```

**关键方法**

| 方法 | 输入 | 输出 |
|------|------|------|
| `act(obs)` | (N, obs_dim) | action_mean (N, 29) |
| `evaluate(critic_obs)` | (N, critic_dim) | value(s) (N, 1) |
| `get_actions_log_prob(a)` | actions (N, 29) | log_prob (N,) |
| `act_inference(obs)` | (N, obs_dim) | action_mean（确定性，无采样）|

---

### 8.2 `MoEActorCritic`（混合专家）

**文件**: `instinct_rl/instinct_rl/modules/moe_actor_critic.py`

```python
class MoEActorCritic(ActorCritic):
    def __init__(self, ..., num_moe_experts=4, moe_gate_hidden_dims=[], ...):
        self.num_moe_experts = num_moe_experts
        super().__init__(...)

    def _build_actor(self, num_actions):
        return MoeLayer(
            self.mlp_input_dim_a,
            self.num_moe_experts,       # = 4
            output_dim=num_actions,     # = 29
            activation=self.activation, # = "elu"
            expert_hidden_dims=self.actor_hidden_dims,  # = [256, 128, 64]
            gate_hidden_dims=[],
        )

    def _build_critic(self, num_values=1):
        return MoeLayer(
            self.mlp_input_dim_c, self.num_moe_experts,
            output_dim=num_values,
            expert_hidden_dims=self.critic_hidden_dims, # = [256, 128, 64]
        )
```

---

### 8.3 `MoeLayer`（专家混合层）

**文件**: `instinct_rl/instinct_rl/modules/moe.py`

```python
class MoeLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,   # = 4
        output_dim: int,    # = 29 (actor) 或 1 (critic)
        activation: str = "elu",
        expert_hidden_dims: list = [],    # = [256, 128, 64]
        gate_hidden_dims: list = [],      # = [] (直接线性 gate)
    ):
        # Gate: MLP(input_dim → num_experts) + Softmax
        # Experts: ModuleList of 4 × MLP([input_dim,256,128,64,output_dim])
```

**前向计算**

```python
def forward(self, x):
    weights = softmax(gate(x))          # (N, 4)
    expert_outputs = stack([e(x) for e in experts], dim=1)  # (N, 4, 29)
    output = einsum("be,beo->bo", weights, expert_outputs)  # (N, 29)
    return output
```

---

### 8.4 `EncoderActorCriticMixin` + `EncoderMoEActorCritic`

**文件**: `instinct_rl/instinct_rl/modules/encoder_actor_critic.py`

```python
class EncoderMoEActorCritic(EncoderActorCriticMixin, MoEActorCritic):
    # Encoder 对指定 obs 组件预处理，输出 embedding 拼接到本体感知特征
```

**Encoder 配置（V11 Parkour）**

```python
encoder_configs = EncoderConfigs(
    depth_encoder=DepthEncoderConv2dCfg(
        class_name="Conv2dHeadModel",
        output_size=128,
        channels=[4],              # 1层, 输出4通道
        kernel_sizes=[3],
        strides=[1],
        paddings=[1],
        hidden_sizes=[256, 256],   # 卷积后 MLP
        nonlinearity="ReLU",
        use_maxpool=True,
        component_names=["depth_image"],  # 处理 depth_image 组件
    )
)
```

**`Conv2dHeadModel` 前向流程**

```
输入: (N, 8, 16, 16)  —— 8帧作为8通道
  Conv2d(8→4, k=3, p=1)     → (N, 4, 16, 16)
  ReLU
  MaxPool2d(2)               → (N, 4, 8, 8)
  Flatten                    → (N, 256)
  Linear(256→256) + ReLU
  Linear(256→256) + ReLU
  Linear(256→128)
输出: (N, 128)  —— 深度特征向量
```

**完整 Actor 前向流程**

```
policy_obs = {
    "base_ang_vel":       (N, 24),
    "projected_gravity":  (N, 24),
    "velocity_commands":  (N, 24),
    "joint_pos":          (N, 232),
    "joint_vel":          (N, 232),
    "actions":            (N, 232),
    "depth_image":        (N, 8, 16, 16),   ← encoder 处理
}
    ↓ EncoderActorCriticMixin.act()
depth_feat = depth_encoder(depth_image)      # (N, 128)
proprio_feat = concat(scalar_obs)            # (N, 768)
combined = concat([proprio_feat, depth_feat]) # (N, 896)
    ↓ MoEActorCritic._build_actor()
actions = MoeLayer(896 → [256,128,64] → 29)  # (N, 29)
```

---

## 9. Algorithm 层 — 训练算法

### 9.1 `PPO`（基础算法）

**文件**: `instinct_rl/instinct_rl/algorithms/ppo.py`

```python
class PPO:
    def __init__(
        self,
        actor_critic: ActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.006,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        clip_min_std: float = 1e-12,
        optimizer_class_name: str = "AdamW",
        schedule: str = "adaptive",   # KL 自适应 lr 调整
        desired_kl: float = 0.01,
        device: str = "cpu",
    ):
```

**`init_storage` 创建的 `RolloutStorage` 缓冲区**

```python
def init_storage(num_envs, num_steps, obs_format, num_actions, num_rewards):
    # 每步轨迹:
    observations:  dict[group → (num_envs, num_steps, dim)]
    actions:       (num_envs, num_steps, 29)
    rewards:       (num_envs, num_steps, num_rewards)
    dones:         (num_envs, num_steps)
    values:        (num_envs, num_steps, num_rewards)
    returns:       (num_envs, num_steps, num_rewards)
    advantages:    (num_envs, num_steps, num_rewards)
    log_probs:     (num_envs, num_steps)
```

**`compute_returns` (GAE)**

```python
def compute_returns(last_critic_obs):
    # 从 last_critic_obs 计算 last_values
    # 逆向迭代: δ_t = r_t + γ*V_{t+1} - V_t
    # A_t = δ_t + γ*λ*A_{t+1}
    # Returns_t = A_t + V_t
```

**`update` 梯度步**

```python
def update(iter):
    for epoch in range(num_learning_epochs):      # = 5
        for mini_batch in storage.mini_batches(num_mini_batches):  # = 4
            ratio = exp(new_log_prob - old_log_prob)
            surrogate1 = ratio * advantages
            surrogate2 = clamp(ratio, 1-ε, 1+ε) * advantages
            policy_loss = -min(surrogate1, surrogate2).mean()
            value_loss  = MSE(values, returns)
            entropy     = -entropy_coef * distribution.entropy().mean()
            total_loss  = policy_loss + value_loss_coef * value_loss + entropy
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(params, max_grad_norm=1.0)
            optimizer.step()
    if schedule == "adaptive":
        adjust_lr_by_kl()  # KL > desired_kl → lr down; KL < 0.5*desired_kl → lr up
```

---

### 9.2 `WasabiPPO`（PPO + AMP 判别器）

**文件**: `instinct_rl/instinct_rl/algorithms/wasabi.py`

```python
class WasabiAlgoMixin:
    def __init__(
        self,
        ...,
        actor_state_key: str = "amp_policy",
        reference_state_key: str = "amp_reference",
        discriminator_class_name: str = "Discriminator",
        discriminator_kwargs: dict = {
            "hidden_sizes": [1024, 512],
            "nonlinearity": "ReLU",
        },
        discriminator_reward_coef: float = 0.25,
        discriminator_reward_type: str = "quad",  # log/quad/wasserstein
        discriminator_loss_func: str = "MSELoss",
        discriminator_gradient_penalty_coef: float = 5.0,
        discriminator_optimizer_class_name: str = "AdamW",
        discriminator_weight_decay_coef: float = 3e-4,
        discriminator_logit_weight_decay_coef: float = 0.04,
        discriminator_optimizer_kwargs: dict = {"lr": 1e-4, "betas": [0.9, 0.999]},
    ):
```

**判别器网络**

```
输入: amp_obs (N, 670)   ← amp_policy 或 amp_reference
  Linear(670 → 1024) + ReLU
  Linear(1024 → 512)  + ReLU
  Linear(512  → 1)
输出: 判别分数 (N, 1)
```

**风格奖励计算**

```python
def compute_auxiliary_reward(actor_state, reference_state):
    d_actor = discriminator(actor_state)    # (N, 1)
    # quad 模式:
    r_style = discriminator_reward_coef * max(0, 1 - 0.25*(D-1)²)
    # log 模式:
    r_style = discriminator_reward_coef * (-log(1 - sigmoid(D) + ε))
    return r_style  # (N,) 加到 PPO 奖励中
```

**判别器损失**

```python
# 真实数据标签=1, 策略数据标签=0
loss = MSELoss(D(ref), 1) + MSELoss(D(policy), 0)
     + gradient_penalty_coef * GP(D, ref, policy)
     + weight_decay * Σ‖w‖² + logit_decay * Σ‖last_layer‖²
```

**`update` 双优化器交替**

```
1. actor_critic optimizer: PPO 损失 (policy + value + entropy)
2. discriminator optimizer: 判别器损失 (MSE + GP + 正则)
```

---

## 10. Runner 层 — 训练主循环

**文件**: `instinct_rl/instinct_rl/runners/on_policy_runner.py`

```python
class OnPolicyRunner:
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,    # V11ParkourPPORunnerCfg.to_dict()
        log_dir: str | None,
        device: str = "cuda:0",
    ):
        obs_format = env.get_obs_format()
        actor_critic = modules.build_actor_critic(
            "EncoderMoEActorCritic", policy_cfg, obs_format,
            num_actions=29, num_rewards=1,
        )
        self.alg = WasabiPPO(actor_critic, **alg_cfg)
        self.alg.init_storage(
            num_envs=2048,
            num_steps_per_env=24,
            obs_format=obs_format,
            num_actions=29,
        )
```

**`learn` 主训练循环**

```python
def learn(num_learning_iterations: int = 30000):
    for iteration in range(num_learning_iterations):

        # ── Rollout 阶段（inference_mode）──
        for step in range(num_steps_per_env=24):
            with torch.inference_mode():
                actions = alg.act(obs, critic_obs)    # (2048, 29)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(rewards, dones, extras)

        # ── 计算 GAE ──
        alg.compute_returns(last_critic_obs)

        # ── 学习阶段（5 epoch × 4 minibatch）──
        mean_loss = alg.update(iteration)

        # ── 日志/保存 ──
        if iteration % save_interval == 0:
            save_checkpoint(iteration)
        log_to_tensorboard(iteration, losses, metrics)
```

**每步数据流（rollout_step）**

```
obs (2048, 896)  →  Actor.act()  →  actions (2048, 29)
                                          ↓
                                    env.step(actions)
                                          ↓
next_obs, rewards (2048,1), dones (2048,), extras
  extras["observations"]["amp_policy"]    (2048, 670)
  extras["observations"]["amp_reference"] (2048, 670)
  extras["log"]                           {reward terms}
                                          ↓
                               alg.process_env_step()
  → RolloutStorage: obs/actions/rewards/dones/values/log_probs
  → AmpStorage: actor_states/reference_states (AMP 用)
```

**检查点管理**

```
experiment_name: "v11_parkour"
save_interval:   1000 次迭代
load_run:        正则 "^(?!_play$).*"（排除 play 目录）
```

**`V11ParkourPPORunnerCfg` 完整参数**

```python
@dataclass
class V11ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env:          int = 24
    policy_observation_group:   str = "policy"
    critic_observation_group:   str = "critic"
    max_iterations:             int = 30000
    save_interval:              int = 1000
    experiment_name:            str = "v11_parkour"
    resume:                     bool = False
    load_run:                   str = "^(?!_play$).*"
    empirical_normalization:    bool = False
    policy:   MoEPolicyCfg       # EncoderMoEActorCritic
    algorithm: AmpAlgoCfg        # WasabiPPO
```

---

## 11. 模块依赖图

```
train.py
  │
  ├─ load_env_cfg("Instinct-Parkour-Target-Amp-V11-v0")
  │    └─ instinct_v11_parkour_amp_final_cfg(play=False)
  │         ├─ unitree_g1_flat_tracking_env_cfg()      [基础模板]
  │         ├─ V11_29DOF_POPSICLE_CFG                  [机器人实体]
  │         │    └─ get_v11_spec() → v11_v4.xml
  │         ├─ beyondmimic_v11_wholebody_delayed_actuator_cfgs
  │         │    └─ DelayedInstinctActuatorCfg × 6组
  │         ├─ Sensors: camera, contact×3, volume_points, scanner×2
  │         │    ├─ NoisyGroupedRayCasterCamera
  │         │    ├─ VolumePoints → Grid3dPointsGenerator
  │         │    └─ ContactSensor × 3
  │         ├─ motion_reference_cfg → MotionReferenceManager
  │         │    └─ AmassMotion → AMASS 数据集
  │         ├─ PoseVelocityCommandCfg
  │         ├─ Observations: policy(896), critic(920), amp(670×2)
  │         ├─ Rewards: 21项
  │         ├─ Terminations: 6项
  │         ├─ Curriculum: tracking_exp_vel
  │         └─ Events: 4项
  │
  ├─ ManagerBasedRlEnv(cfg)          [mjlab 环境]
  │
  ├─ InstinctRlVecEnvWrapper(env)    [VecEnv 适配]
  │    └─ get_obs_format()
  │         └─ {policy: 896, critic: 920, amp_policy: 670, ...}
  │
  └─ OnPolicyRunner(env, cfg)
       ├─ build_actor_critic("EncoderMoEActorCritic", ...)
       │    ├─ EncoderActorCriticMixin
       │    │    └─ Conv2dHeadModel(8ch,16×16 → 128)
       │    └─ MoEActorCritic(4专家, [256,128,64])
       │         └─ MoeLayer × 2 (actor + critic)
       ├─ WasabiPPO(actor_critic, ...)
       │    ├─ Discriminator([1024,512])
       │    ├─ RolloutStorage(2048, 24, ...)
       │    └─ AmpStorage(actor_states, reference_states)
       └─ learn(30000 iterations)
            ├─ rollout × 24 steps
            ├─ compute_returns (GAE)
            └─ update (5 epochs × 4 minibatches)
```

---

*文档生成时间: 2026-03-17*
*覆盖模块: config / asset / actuator / sensor / motion_reference / mdp / rl_wrapper / policy_network / algorithm / runner*
