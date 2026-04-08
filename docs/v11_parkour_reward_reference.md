# V11 Parkour AMP 奖励函数值域参考表

> 配置文件：`tasks/parkour/config/v11/v11_parkour_target_amp_cfg.py`
>
> 奖励计算：`value = raw_func() * weight`，日志中 `/sum` = Σ(value × dt) 全 episode 累积，`/timestep` = sum / episode_length
>
> 反算 raw：`raw_per_step = (timestep值) / (weight × dt)`

## 奖励分类总览

| 类别 | 正向奖励 | 惩罚项 |
|------|---------|--------|
| Task（任务目标） | track_lin_vel_xy_exp, track_ang_vel_z_exp, is_alive | heading_error, dont_wait, stand_still |
| Regularization（运动质量） | feet_air_time, feet_close_xy | volume_points_penetration, feet_slide, ang_vel_xy_l2, dof_torques_l2, dof_acc_l2, dof_vel_l2, action_rate_l2, flat_orientation_l2, pelvis_orientation_l2, feet_flat_ori, feet_at_plane, energy, freeze_upper_body |
| Safety（安全约束） | — | dof_pos_limits, dof_vel_limits, torque_limits, undesired_contacts |

## 详细值域表

### Task 奖励

| 名称 | 权重 | 函数 | raw 值域 | raw 理想值 | 说明 |
|------|------|------|----------|-----------|------|
| `track_lin_vel_xy_exp` | **+2.0** | `exp(-‖v_cmd - v_actual‖² / 0.25)` | [0, 1] | → 1.0 | 线速度跟踪，std=0.5。完美跟踪=1，误差越大越接近0 |
| `track_ang_vel_z_exp` | **+2.0** | `exp(-(ω_cmd - ω_actual)² / 0.25)` | [0, 1] | → 1.0 | 角速度跟踪，std=0.5 |
| `heading_error` | **-1.0** | `|yaw_cmd|` | [0, +∞) | → 0 | 航向偏差绝对值，取决于指令大小。典型 [0, π] |
| `dont_wait` | **-0.5** | 三阶阶梯布尔求和 | {0, 1, 2, 3} | → 0 | 有前进指令(>0.3)但速度不足时惩罚。v<0.15=1, v<0=2, v<-0.15=3 |
| `is_alive` | **+3.0** | `~terminated` | {0, 1} | → 1.0 | 存活=1，终止=0 |
| `stand_still` | **-0.3** | `(Σ|q-q_default| - 4.0) × 零指令标志` | (-∞, +∞) | → 0 | 零指令时惩罚关节偏移。offset=4.0 使小偏移为负（即奖励）。典型 [-4, 10] |

### Regularization 奖励

| 名称 | 权重 | 函数 | raw 值域 | raw 理想值 | 说明 |
|------|------|------|----------|-----------|------|
| `volume_points_penetration` | **-4.0** | Σ(穿透深度 × 速度) | [0, +∞) | → 0 | 腿部体积点地形穿透。无穿透=0。典型 [0, 5] |
| `feet_air_time` | **+0.5** | min(单脚摆动/支撑时间) | [0, +∞) | 适度 | 鼓励交替步态。单腿支撑时取 min(接触/腾空时间)。典型 [0, 0.5]s |
| `feet_slide` | **-0.4** | Σ(脚底滑动速度 × 接触标志) | [0, +∞) | → 0 | 2只脚。接触时脚底XY速度之和。典型 [0, 3] m/s |
| `ang_vel_xy_l2` | **-0.05** | `ω_x² + ω_y²` | [0, +∞) | → 0 | 躯干横滚/俯仰角速度平方和。典型 [0, 10] rad²/s² |
| `dof_torques_l2` | **-1.5e-7** | Σ(τ²)，12个腿部关节 | [0, +∞) | → 0 | 腿部关节力矩平方和。量级很大（~1e6），配极小权重 |
| `dof_acc_l2` | **-1.25e-7** | Σ(q̈²)，全29关节 | [0, +∞) | → 0 | 关节加速度平方和。量级大，配极小权重 |
| `dof_vel_l2` | **-1e-4** | Σ(q̇²)，全29关节 | [0, +∞) | → 0 | 关节速度平方和。典型 [0, 500] |
| `action_rate_l2` | **-0.005** | Σ(a_t - a_{t-1})² | [0, +∞) | → 0 | 动作变化率平方和。全29维动作。典型 [0, 50] |
| `flat_orientation_l2` | **-3.0** | `grav_proj_x² + grav_proj_y²` | [0, 1] | → 0 | 根链接（root_link）投影重力XY分量平方和。完全水平=0，完全侧翻=1 |
| `pelvis_orientation_l2` | **-3.0** | 同上，对 `base_link` | [0, 1] | → 0 | V11 的 base_link 朝向惩罚。与 flat_orientation 类似但作用于骨盆 |
| `feet_flat_ori` | **-0.4** | Σ(脚部投影重力XY范数 × 接触) | [0, 2] | → 0 | 2只脚，接触时脚底朝向偏差。每脚 [0, 1]，求和 |
| `feet_at_plane` | **-0.1** | Σ clamp(脚高-地面高-0.05, 0, 0.3) × 接触 | [0, ~0.6] | → 0 | 接触脚应紧贴地面。每脚最大0.3，两脚求和 |
| `feet_close_xy` | **+0.4** | `exp(-clamp(0.12-dist_y, 0)/0.05) - 1` | [-1, 0] | → 0 | 双脚Y距离不应过近。距离>阈值=0，过近→-1。注意这是正权重×负raw |
| `energy` | **-5e-5** | Σ((τ×q̇/stiffness)²)，12腿关节 | [0, +∞) | → 0 | 归一化后的电机功率平方。量级大，配极小权重 |
| `freeze_upper_body` | **-0.004** | Σ|q-q_default|，上肢+腰 | [0, +∞) | → 0 | 17个关节（肩/肘/腕/腰）L1偏差和。典型 [0, 10] rad |

### Safety 奖励

| 名称 | 权重 | 函数 | raw 值域 | raw 理想值 | 说明 |
|------|------|------|----------|-----------|------|
| `dof_pos_limits` | **-1.0** | Σ clamp(超出软限位, 0, +∞) | [0, +∞) | → 0 | 全29关节。超出软限位部分求和。典型 [0, 5] rad |
| `dof_vel_limits` | **-1.0** | Σ clamp(|q̇|-limit×0.9, 0, 1) | [0, 29] | → 0 | 全29关节。每关节上限clip到1。soft_ratio=0.9 |
| `torque_limits` | **-0.01** | Σ(超出力矩限×0.8)² | [0, +∞) | → 0 | 力矩超出80%限的部分平方求和 |
| `undesired_contacts` | **-1.0** | Σ(接触标志) | [0, N_bodies] | → 0 | 非期望接触的body数量。取决于 sensor 配置的 body 数 |

## 已注释掉的奖励（inactive）

| 名称 | 权重 | 说明 |
|------|------|------|
| ~~`joint_deviation_hip`~~ | -0.5 | hip yaw/roll 4关节平方偏差。raw=[0, +∞)，典型 ~2.0 |
| ~~`feet_heading_align`~~ | -0.3 | 脚朝向与躯干对齐。raw=[0, 4]（2脚×余弦距离[0,2]） |

## 奖励量级速查

按 **单步 |weight × typical_raw|** 排序（仅比较量级，不精确）：

| 量级 | 奖励项 |
|------|--------|
| **~2-3** | is_alive(3), track_lin_vel(~2), track_ang_vel(~2), flat_orientation(~0.3→×3), pelvis_orientation(~0.3→×3) |
| **~0.5-1** | heading_error, dont_wait, stand_still, volume_points_penetration, dof_pos_limits, dof_vel_limits |
| **~0.1-0.5** | feet_air_time, feet_slide, feet_flat_ori, feet_close_xy, ang_vel_xy_l2 |
| **~0.01-0.1** | action_rate_l2, energy, dof_torques_l2, dof_acc_l2, dof_vel_l2, freeze_upper_body, torque_limits |

## 注意事项

1. **日志中的 `/sum` 不是单步值**，是 Σ(raw × weight × dt) 整个 episode 累积
2. **`/timestep`** 是 sum / episode_length，更接近"每步平均"但仍含 weight 和 dt
3. 反算无权重的 raw：`raw = timestep_value / (weight × dt)`，其中 dt 通常是仿真控制步长
4. 上表中"典型值"基于正常行走状态估计，极端情况（摔倒、卡住）会远超典型范围
