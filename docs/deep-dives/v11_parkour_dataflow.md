# V11 Parkour 模块数据流技术文档

> 任务 ID: `Instinct-Parkour-Target-Amp-V11-v0`
> 配置文件: `InstinctMJ/src/instinct_mj/tasks/parkour/config/v11/v11_parkour_target_amp_cfg.py`
> Agent 配置: `InstinctMJ/src/instinct_mj/tasks/parkour/config/v11/agents/instinct_rl_amp_cfg.py`

---

## 1. 脑图（ASCII Mind Map）

```
V11-Parkour-AMP
│
├── 机器人 (V11, 29-DOF 全身)
│   ├── 腿部 12 DOF (left/right × hip_pitch/roll/yaw + knee + ankle_pitch/roll)
│   ├── 腰部 3 DOF (waist_yaw/roll/pitch)
│   └── 手臂 14 DOF (left/right × shoulder×3 + elbow + wrist×3)
│
├── 仿真频率
│   ├── 物理步长 5ms → 200 Hz
│   ├── 控制步长 20ms (decimation=4) → 50 Hz
│   ├── 运动参考更新 20ms → 50 Hz
│   └── 相机深度历史 37 帧 × 20ms = 0.74s 时间窗口
│
├── 观测空间
│   ├── Policy Obs (Actor)
│   │   ├── base_ang_vel    3×8=24  (加噪声)
│   │   ├── projected_gravity 3×8=24 (加噪声)
│   │   ├── velocity_commands 3×8=24
│   │   ├── joint_pos       29×8=232 (加噪声)
│   │   ├── joint_vel       29×8=232 (加噪声, ×0.05)
│   │   ├── actions         29×8=232
│   │   └── depth_image     8帧×16×16 → Conv2d编码→128
│   │       本体感知合计: 768  深度编码: 128  总: 896
│   │
│   ├── Critic Obs (额外含真实线速度)
│   │   ├── base_lin_vel    3×8=24  (真值, 无噪声)
│   │   ├── base_ang_vel    3×8=24
│   │   ├── projected_gravity 3×8=24
│   │   ├── velocity_commands 3×8=24
│   │   ├── joint_pos       29×8=232
│   │   ├── joint_vel       29×8=232 (×0.05)
│   │   ├── actions         29×8=232
│   │   └── depth_image     8帧×16×16 → Conv2d编码→128
│   │       本体感知合计: 792  深度编码: 128  总: 920
│   │
│   ├── AMP Policy Obs (判别器用, history=10)
│   │   ├── projected_gravity 3×10=30
│   │   ├── joint_pos_rel   29×10=290
│   │   ├── joint_vel       29×10=290 (×0.05)
│   │   ├── base_lin_vel    3×10=30
│   │   └── base_ang_vel    3×10=30  合计: 670
│   │
│   └── AMP Reference Obs (来自运动参考, 结构同上) 合计: 670
│
├── 动作输出
│   └── joint_pos_target  29 × float32  (BeyondMimic action scale)
│
├── 策略网络 (EncoderMoEActorCritic)
│   ├── 深度编码器 (Conv2dHeadModel)
│   │   ├── 输入: depth_image 8ch×H×W (裁剪后16×16)
│   │   ├── Conv2d: 8→4ch, kernel=3, stride=1, pad=1
│   │   ├── MaxPool
│   │   ├── MLP: →256→256→128
│   │   └── 输出: 128维特征
│   │
│   ├── Actor (MoE, 4专家)
│   │   ├── 输入: proprio(768) + depth_feat(128) = 896
│   │   ├── 隐藏: [256, 128, 64]
│   │   └── 输出: 29维关节位置目标
│   │
│   └── Critic (MoE, 4专家)
│       ├── 输入: critic_proprio(792) + depth_feat(128) = 920
│       ├── 隐藏: [256, 128, 64]
│       └── 输出: 1维价值估计
│
├── AMP 判别器 (WasabiPPO)
│   ├── 输入: AMP Policy/Reference Obs (670维)
│   ├── 隐藏: [1024, 512]
│   ├── 判别器奖励系数: 0.25
│   └── 损失函数: MSELoss + 梯度惩罚(系数=5.0)
│
├── 传感器
│   ├── contact_forces (ankle, history=3)
│   ├── torso_contact_forces (base_link↔terrain, history=3)
│   ├── undesired_contact_forces (非踝部, history=3)
│   ├── leg_volume_points (每脚14×5×2=140点)
│   ├── left/right_height_scanner (脚踝射线, 1条)
│   ├── camera (64×36深度, waist_pitch_link挂载)
│   └── motion_reference (AMASS, 29DOF, 10帧@50Hz)
│
├── 奖励函数 (21项)
│   ├── 任务类 (正): track_lin_vel_xy_exp, track_ang_vel_z_exp, is_alive
│   ├── 任务类 (负): heading_error, dont_wait, stand_still
│   ├── 正则化: feet_air_time, feet_slide, joint_deviation_hip
│   │           ang_vel_xy_l2, dof_torques_l2, dof_acc_l2, dof_vel_l2
│   │           action_rate_l2, flat_orientation_l2, pelvis_orientation_l2
│   │           feet_flat_ori, feet_at_plane, feet_close_xy, energy
│   │           volume_points_penetration
│   └── 安全类 (负): dof_pos_limits, dof_vel_limits, torque_limits, undesired_contacts
│
├── 地形课程 (10类, 10×20网格)
│   ├── perlin_rough (5%) / perlin_rough_stand (5%)
│   ├── square_gaps (10%) — 缝隙0.1~0.7m
│   ├── pyramid_stairs (15%) + pyramid_stairs_high (10%)
│   ├── pyramid_stairs_inv (15%) + pyramid_stairs_inv_high (10%)
│   ├── boxes (10%) + mesh_boxes (10%)
│   └── hf_pyramid_slope_inv (10%)
│
└── 训练超参数 (WasabiPPO)
    ├── 并行环境: 2048
    ├── 每环境步数: 24 (rollout = 0.48s)
    ├── Minibatch: 4, 学习轮数: 5
    ├── 学习率: 1e-3 (adaptive)
    ├── γ=0.99, λ=0.95, KL target=0.01
    └── 最大迭代: 30000
```

---

## 2. 频率体系

| 层级 | 周期 | 频率 | 说明 |
|------|------|------|------|
| 物理仿真 | 5 ms | 200 Hz | MuJoCo `timestep=0.005` |
| 控制步 | 20 ms | 50 Hz | `decimation=4` |
| 运动参考 | 20 ms | 50 Hz | `frame_interval_s=update_period=0.02` |
| 深度相机帧 | 20 ms | 50 Hz | 随控制步更新 |
| 深度历史窗口 | 740 ms | — | 37帧 × 20ms |
| 指令重采样 | 8~12 s | — | `resampling_time_range=(8,12)` |
| Episode 时长 | 20 s (训练) / 10 s (Play) | — | `episode_length_s` |

---

## 3. 观测维度详表

### 3.1 Policy 观测组（Actor，有噪声）

| 字段 | 原始维度 | 历史长度 | 展平后维度 | 缩放 | 噪声 |
|------|---------|---------|-----------|------|------|
| `base_ang_vel` | 3 | 8 | **24** | ×0.25 | U(−0.2, 0.2) |
| `projected_gravity` | 3 | 8 | **24** | — | U(−0.05, 0.05) |
| `velocity_commands` | 3 | 8 | **24** | — | — |
| `joint_pos` | 29 | 8 | **232** | — | U(−0.01, 0.01) |
| `joint_vel` | 29 | 8 | **232** | ×0.05 | U(−0.5, 0.5) |
| `actions` | 29 | 8 | **232** | — | — |
| **本体感知合计** | | | **768** | | |
| `depth_image` | 16×16 | 8 帧* | 2048(原始) | — | 见下 |
| → Conv2d 编码后 | | | **128** | | |
| **Policy 总维度** | | | **896** | | |

*深度历史：camera 存 37 帧，`history_skip_frames=5, num_output_frames=8`，等效覆盖 37 帧×20ms=0.74s

### 3.2 Critic 观测组（无噪声，含真实线速度）

| 字段 | 展平维度 | 备注 |
|------|---------|------|
| `base_lin_vel` | 24 | 真值，训练专用 |
| `base_ang_vel` | 24 | |
| `projected_gravity` | 24 | |
| `velocity_commands` | 24 | |
| `joint_pos` | 232 | |
| `joint_vel` | 232 | ×0.05 |
| `actions` | 232 | |
| **本体感知合计** | **792** | |
| `depth_image`（编码后）| **128** | |
| **Critic 总维度** | **920** | |

### 3.3 AMP 观测组（判别器用，history=10）

| 字段 | 维度 | 对应 motion_reference 字段 |
|------|------|--------------------------|
| `projected_gravity` | 3×10 = 30 | `parkour_amp_reference_projected_gravity` |
| `joint_pos_rel` | 29×10 = 290 | `parkour_amp_reference_joint_pos_rel` |
| `joint_vel` | 29×10 = 290 | `parkour_amp_reference_joint_vel_rel` |
| `base_lin_vel` | 3×10 = 30 | `parkour_amp_reference_base_lin_vel` |
| `base_ang_vel` | 3×10 = 30 | `parkour_amp_reference_base_ang_vel` |
| **AMP 合计** | **670** | Policy 侧 = Reference 侧 = 670 |

---

## 4. 动作空间

| 项目 | 值 |
|------|---|
| 输出维度 | **29**（全关节位置目标） |
| 动作类型 | `JointPositionAction` |
| Action Scale | BeyondMimic 公式：`0.25 × effort_limit / stiffness`（各关节独立） |
| 典型关节范围 | 腿部: ~0.12~0.19 rad/cmd；踝部: ~0.08 rad/cmd；腰臂: 更小 |
| 延迟随机化 | 0~2 控制步（20~40ms）通信延迟 |

---

## 5. 深度感知通道详解

```
物理相机参数
  挂载点: waist_pitch_link (腰部俯仰关节)
  分辨率: 64×36 px
  FOV: 水平 89.51°，垂直 58.29°
  深度范围: 0.1 ~ 2.5 m

噪声流水线
  1. CropAndResize (crop_region=(18,0,16,16)) → 16×16 px
  2. GaussianBlur (kernel=3, σ=1)
  3. DepthNormalization (range→[0,1])

历史缓存
  存储 37 帧 → "distance_to_image_plane_noised_history"
  时间窗口: 37 × 20ms = 740ms

策略取帧 (delayed_visualizable_image)
  num_output_frames=8, history_skip_frames=5
  采样位置: [0, 6, 12, 18, 24, 30, 36] (0-based，近7帧均匀)
  delayed_frame_ranges=(0,1): 允许 0~1 帧随机延迟

Conv2d 编码器输入
  shape: (batch, 8, 16, 16)  — 8通道=8帧
  Conv: 8→4 ch, 3×3, pad=1 → (4,16,16)
  MaxPool → (4,8,8)
  Flatten → 256
  MLP: 256→256→128
  输出: 128维深度特征
```

---

## 6. 运动参考系统（Motion Reference / AMP）

```
来源: AMASS 数据集 (parkour_motion_without_run)
模型: V11 XML (v11_v4.xml)
DOF: 29 (全身)
帧间隔: 20ms (50Hz)
缓存帧数: 10 帧 = 200ms 时间窗口

关注链接 (14个):
  base_link, waist_pitch_link
  left/right: shoulder_roll_link, elbow_link, wrist_yaw_link
             hip_roll_link, knee_link, ankle_roll_link

左右对称增强:
  joint_mapping: 29维对称映射
  joint_reverse_buf: 各轴反向标志
  link_mapping: 14个链接对称映射

速度估计方法: "frontward" (前向差分)
初始帧范围: [0%, 90%] 随机起始

AMP 判别器
  输入: amp_policy_obs (670) + amp_reference_obs (670) 各自独立
  网络: FC [1024, 512] + ReLU
  损失: MSELoss + gradient_penalty (coef=5.0)
  奖励: r_style = 0.25 × D(s,s') (quad 变换)
```

---

## 7. 终止条件

| 条件 | 类型 | 参数 |
|------|------|------|
| `time_out` | timeout | episode_length_s=20s |
| `terrain_out_bound` | timeout | distance_buffer=2.0m |
| `base_contact` | hard | base_link 与地形接触力 > 1N |
| `bad_orientation` | hard | 倾斜 > 1.0 rad |
| `root_height` | hard | 质心高度 < 0.5m (play 时禁用) |
| `dataset_exhausted` | timeout | 运动参考播放完毕 |

---

## 8. 数据流全局图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MuJoCo 物理引擎 (200 Hz)                      │
│  每 4 步 (20ms) 输出一次观测                                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         │                     │                      │
         ▼                     ▼                      ▼
  ┌────────────┐     ┌──────────────────┐    ┌───────────────┐
  │ 本体感知   │     │   深度相机       │    │  运动参考     │
  │ (29DOF)    │     │ 64×36@50Hz       │    │ AMASS@50Hz    │
  │ IMU/关节   │     │ 历史37帧=740ms   │    │ 10帧=200ms    │
  └─────┬──────┘     └────────┬─────────┘    └───────┬───────┘
        │                     │                      │
        │           CropResize│                      │
        │           Blur      │                      │
        │           Normalize ▼                      │
        │            16×16×8帧                       │
        │                     │                      │
        ▼                     ▼                      │
  history×8 展平       Conv2d编码器                  │
  ┌──────────────┐    ┌──────────────┐               │
  │  768 维      │    │  128 维      │               │
  │  本体特征    │    │  深度特征    │               │
  └──────┬───────┘    └──────┬───────┘               │
         └─────────┬─────────┘                       │
                   ▼                                 │
           ┌──────────────┐                         ▼
           │ Policy Actor │             ┌───────────────────┐
           │ MoE (4专家)  │             │  AMP 判别器       │
           │ 896→[256,128,│             │  670+670→[1024,   │
           │ 64]→29       │             │  512]→判别分数    │
           └──────┬───────┘             └────────┬──────────┘
                  │                              │
                  ▼                              ▼
           29维关节目标              AMP 风格奖励 (×0.25)
                  │
          延迟执行器 (0~2步)
                  │
                  ▼
           ┌──────────────┐
           │  MuJoCo 仿真 │ ←─ 奖励 = 任务奖励 + AMP风格奖励
           │  (下一帧)    │
           └──────────────┘
```

---

## 9. 关键超参数汇总

| 类别 | 参数 | 值 |
|------|------|---|
| **仿真** | physics_dt | 5 ms |
| **仿真** | decimation | 4 (50 Hz 控制) |
| **仿真** | episode_length | 20 s (训练) / 10 s (Play) |
| **机器人** | DOF | 29 (全身) |
| **机器人** | 根节点 | base_link |
| **机器人** | 初始高度 | 0.9 m |
| **训练** | num_envs | 2048 |
| **训练** | num_steps_per_env | 24 (rollout=0.48s) |
| **训练** | mini_batches | 4 |
| **训练** | learning_epochs | 5 |
| **训练** | 算法 | WasabiPPO |
| **策略网络** | 类型 | EncoderMoEActorCritic |
| **策略网络** | MoE 专家数 | 4 |
| **策略网络** | Actor 隐藏层 | [256, 128, 64] |
| **策略网络** | 深度编码输出 | 128 |
| **观测** | Policy 输入 | 896 (768 本体 + 128 深度) |
| **观测** | Critic 输入 | 920 (792 本体 + 128 深度) |
| **观测** | AMP 输入 | 670 (Policy) + 670 (Ref) |
| **动作** | 维度 | 29 |
| **AMP** | 判别器奖励系数 | 0.25 |
| **AMP** | 判别器隐藏层 | [1024, 512] |
| **地形** | 类型数 | 10 类 |
| **地形** | 格网 | 10 行 × 20 列 |
| **运动参考** | 数据集 | AMASS (parkour, 无奔跑) |
| **运动参考** | 帧率 | 50 Hz |

---

*文档生成时间: 2026-03-17*
*源码路径: `InstinctMJ/src/instinct_mj/tasks/parkour/config/v11/`*
