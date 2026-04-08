# V11 Parkour 问题深度分析报告

**生成时间**: 2026-04-01 10:36  
**分析范围**: 地形等级上限问题 + 下楼梯步态问题

---

## 一、问题1：地形等级只能到6

### 1.1 课程升级的完整数值链路

```
[每帧] → _update_metrics() → tracking_exp_vel_xy 累计
                              tracking_exp_vel_yaw 累计
         ↓
[Episode结束/重采样] → tracking_exp_vel() → 升降级判定
         ↓
[升降级] → update_env_origins() → terrain_levels ± 1
```

#### Step 1: tracking指标如何累计

```python
# pose_velocity_command.py → _update_metrics()
lin_vel_error = sum(square(cmd_vel_xy - actual_vel_xy), dim=1)
tracking_exp_vel_xy += exp(-lin_vel_error / std²) / max_episode_length

angular_vel_error = square(cmd_vel_yaw - actual_vel_yaw)
tracking_exp_vel_yaw += exp(-angular_vel_error / std²) / max_episode_length
```

**关键参数**：
- `lin_vel_metrics_std`（来自PoseVelocityCommandCfg默认值）：未显式设置，使用默认值
- `max_episode_length` = 20.0 / step_dt ≈ 1000步
- 每帧贡献 ≈ `exp(-error/std²) / 1000`
- 完美追踪时：每帧贡献 ≈ 1/1000 = 0.001
- 完美追踪整个episode：tracking_exp ≈ 1.0

#### Step 2: 升降级阈值判定

```python
# curriculums.py
move_up = (tracking_exp_vel_xy > 0.7) * (tracking_exp_vel_yaw > 0.5)
move_down = (tracking_exp_vel_xy < 0.35) * ~move_up
```

| 条件 | V11 | G1 |
|------|-----|-----|
| 升级-线速度 | `> 0.7` | `> 0.6` |
| 升级-角速度 | `> 0.5` | `> 0.0` (永远为True) |
| 降级-线速度 | `< 0.35` | `< 0.3` |
| **有效升级条件** | **线速度AND角速度同时达标** | **仅线速度达标** |

**这是最关键的差异**：G1实际上只要求线速度追踪分数超过0.6即可升级，而V11要求线速度>0.7且角速度>0.5。

#### Step 3: 等级更新

```python
# terrain_entity.py → update_env_origins()
self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
# 上限处理：如果达到max_terrain_level(=10)，随机降到[0,10)
self.terrain_levels[env_ids] = torch.where(
    self.terrain_levels[env_ids] >= self.max_terrain_level,
    torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
    torch.clip(self.terrain_levels[env_ids], 0),
)
```

### 1.2 为什么卡在6？定量分析

#### 因素A：课程升级门槛过高（主因）

V11的升级条件是 **双AND条件**：
```
move_up = (tracking_exp_vel_xy > 0.7) * (tracking_exp_vel_yaw > 0.5)
```

假设在某一等级地形上：
- 线速度追踪良好，`tracking_exp_vel_xy = 0.75`（满足>0.7）
- 角速度追踪一般，`tracking_exp_vel_yaw = 0.45`（不满足>0.5）
- **结果：无法升级**

而G1在同样条件下：
```
move_up = (0.75 > 0.6) * (0.45 > 0.0) = True * True = True  → 升级成功
```

**概率估算**：如果 `tracking_exp_vel_yaw` 的分布均值为0.5，标准差为0.15，则约50%的episode无法满足角速度条件。这意味着即使线速度完美，也只有一半的episode能升级。

#### 因素B：高等级地形提前终止

等级6→7的难度跳跃：

| 地形类型 | Level 6 (difficulty=0.667) | Level 7 (difficulty=0.778) | 变化 |
|---------|--------------------------|--------------------------|------|
| `pyramid_stairs_inv` 台阶高度 | 0.170m | 0.190m | +12% |
| `pyramid_stairs_inv_high` 台阶高度 | 0.317m | 0.361m | +14% |
| `pyramid_stairs` 台阶高度 | 0.170m | 0.190m | +12% |
| `boxes` 障碍高度 | 0.350m | 0.389m | +11% |
| `hf_pyramid_slope_inv` 坡度 | 0.467 rad | 0.545 rad | +17% |

更高的台阶 → 更多terminations：
- `bad_orientation`（limit_angle=1.0rad≈57°）：机器人下大台阶时更容易倾斜过大
- `root_height`（minimum_height=0.5m）：下大台阶时重心下降更多
- `base_contact`：下台阶不稳时base_link更容易撞地

提前终止 → episode变短 → tracking指标累计不足 → 无法升级 → **恶性循环**

#### 因素C：初始等级分布偏窄

- `max_init_terrain_level = 5`
- 初始等级均匀分布在 [0, 5]
- 没有任何env从等级6-9开始
- 需要逐级从5升到6、7、8、9，每级都需要在多次episode中持续表现良好

#### 因素D：V11与G1的关键配置差异汇总

| 参数 | V11 | G1 | 影响 |
|------|-----|-----|------|
| 课程线速度上限 | 0.7 | 0.6 | V11更难升级 |
| 课程角速度阈值 | (0.3, 0.5) | (0.0, 0.0) | V11多一个约束 |
| 足部穿透惩罚权重 | -2.0 | -4.0 | V11脚更容易穿地 |
| nconmax | 256 | 128 | V11有更多碰撞点 |
| 足部模型 | 7胶囊赤脚 | 鞋模型 | 赤脚接触面积小 |

### 1.3 问题1的根因排序

| 排名 | 原因 | 确信度 | 影响 |
|------|------|--------|------|
| 1 | 课程升级要求角速度>0.5（G1为0） | 高 | 直接阻止升级 |
| 2 | 课程升级要求线速度>0.7（G1为0.6） | 高 | 进一步提高门槛 |
| 3 | 高等级地形提前终止导致指标累计不足 | 中 | 恶性循环 |
| 4 | V11赤脚模型足部穿透惩罚弱(-2.0) | 中 | 影响稳定性 |

---

## 二、问题2：下楼梯时一步跨多个台阶

### 2.1 下楼梯地形的几何结构

V11 parkour有两种下楼梯地形：

**普通下楼梯** (`pyramid_stairs_inv`)：
- `step_height_range = (0.05, 0.23)` → 随difficulty变化
- `step_width = 0.3m` → 每级台阶宽度仅30cm
- `platform_width = 2.5m` → 中心平台宽度

**高级下楼梯** (`pyramid_stairs_inv_high`)：
- `step_height_range = (0.05, 0.45)` → 台阶更高
- `step_width = 1.5m` → 台阶更宽
- `platform_width = 4.0m`

**地形生成算法**（`perlin_pyramid_stairs_terrain`）：
```python
# 同心环结构：从外圈(高处)到内圈(低处)
while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
    start_x += step_width; stop_x -= step_width
    start_y += step_width; stop_y -= step_y
    current_step_height += step_height  # 每环下降一步
    hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height
```

**在level 6 (difficulty=0.667) 时**：
- 普通台阶高度：0.05 + 0.667 × 0.18 = **0.170m**
- 高级台阶高度：0.05 + 0.667 × 0.40 = **0.317m**

### 2.2 速度命令与台阶跨度的关系

下楼梯地形的速度命令：
```python
"pyramid_stairs_inv": {"lin_vel_x": (0.45, 0.8), ...}
"pyramid_stairs_inv_high": {"lin_vel_x": (0.45, 0.8), ...}
```

**关键数值分析**：
- 最低速度 0.45 m/s，最高速度 0.8 m/s
- `pyramid_stairs_inv` 台阶宽度 0.3m
- 机器人以 0.5 m/s 前进时，跨越一级0.3m的台阶需要 0.6秒
- 以 0.8 m/s 前进时，仅需 0.375秒

**V11机器人腿长参数**（从XML提取）：
- hip_pitch → hip_roll: 0.03m
- hip_roll → hip_yaw: 0.13m
- hip_yaw → knee: 0.151m
- knee → ankle_pitch: 0.339m
- ankle_pitch → ankle_roll: 0.015m
- **总腿长约 0.665m**

正常行走步幅 ≈ 0.4-0.6m（与腿长相关），而台阶宽度只有0.3m，这意味着**机器人一步的自然步幅就可能跨越1-2级台阶**。

### 2.3 奖励函数对下楼梯行为的影响

#### `feet_air_time`（权重=0.5）

```python
def feet_air_time(env, command_name, vel_threshold, sensor_name):
    in_contact = contact_time > 0.0
    single_stance = sum(in_contact.int(), dim=1) == 1
    reward = min(where(single_stance, in_mode_time, 0.0), dim=1)[0]
    reward *= norm(cmd[:,:2]) > vel_threshold or abs(cmd[:,2]) > vel_threshold
```

**问题**：
- 该奖励鼓励"单脚支撑时间"和"空中时间"
- 在下楼梯时，机器人可以跳过多个台阶来获得更多air_time奖励
- **跳跃策略比逐步下降策略获得更多feet_air_time奖励**
- V11和G1的权重相同(0.5)，但V11赤脚模型的接触模式更不稳定，可能更容易触发"空中"状态

#### 缺失的关键奖励

V11没有任何以下奖励：
1. **无"逐步下降"引导奖励**：没有惩罚脚一次性下降过大高度
2. **无"台阶停留"奖励**：没有鼓励脚在每一级台阶上短暂停留
3. **无"下楼梯速度限制"**：所有台阶地形的速度命令范围相同(0.45-0.8)

#### `volume_points_penetration`（权重差异）

```python
# V11
"volume_points_penetration": RewardTermCfg(
    weight=-2.0,  # ← 惩罚较弱
    params={"sensor_name": "leg_volume_points"},
)

# G1  
"volume_points_penetration": RewardTermCfg(
    weight=-4.0,  # ← 惩罚更强
    params={"sensor_name": "leg_volume_points"},
)
```

V11的足部穿透惩罚只有G1的**一半**。这意味着：
- V11的脚更容易穿入台阶边缘的地形网格
- 穿透后脚可能直接滑到下一级台阶，而不是被"挡住"
- 减少了逐步下降的物理约束

#### `feet_at_plane` 的 height_offset 差异

```python
# V11: height_offset = 0.05
# G1 (with shoe): height_offset = 0.058
```

V11的height_offset比G1小，对脚位置的精度要求不同，但差异不大。

### 2.4 感知系统对下楼梯的影响

#### 深度相机配置

```python
NoisyGroupedRayCasterCameraCfg(
    name="camera",
    frame=ObjRef(type="body", name="head_pitch_link", entity="robot"),
    pattern=PinholeCameraPatternCfg(width=64, height=36, fovy=58.29),
    offset=OffsetCfg(pos=(0.025, 0.005, 0.054), rot=(...45° pitch...)),
    noise_pipeline={
        "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),  # 64x36 → 16x16
        "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
        "depth_normalization": DepthNormalizationCfg(depth_range=(0.0, 2.5)),
    },
    min_distance=0.1, max_distance=2.5,
)
```

**关键问题**：
- 原始分辨率 64×36，裁剪后仅 **16×16 = 256像素**
- 相机朝向前方偏下（pitch约45°），FOV=58.29°
- 最大感知距离2.5m，但对于脚下台阶的感知能力有限
- 16×16的分辨率**无法分辨0.3m宽的单个台阶**
- 深度图经过高斯模糊(sigma=1)，进一步模糊了台阶边缘

**估算**：
- 相机在2.5m处，水平视野约89°，16个像素覆盖约2.5m
- 每像素约 2.5/16 = 0.156m
- 0.3m的台阶宽度 ≈ **不到2个像素**
- 经过高斯模糊后，台阶边缘几乎不可见

#### 高度扫描器

```python
RayCastSensorCfg(
    name="left_height_scanner",
    frame=ObjRef(type="body", name="left_ankle_roll_link", entity="robot"),
    pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
)
```

- 仅扫描脚踝下方一个点的距离
- resolution=0.12m，即每0.12m一个采样点
- **无法提前感知前方的台阶高度变化**
- 这只是一个局部高度测量，不是前瞻性感知

### 2.5 V11足部碰撞模型的特殊性

V11赤脚使用7个胶囊体（capsule），参数：
```xml
<!-- 全部沿y轴排列（quat约45°旋转） -->
<geom name="left_foot1_collision" size="0.012 0.08" pos="0.04 -0.03 -0.055" type="capsule"/>
<geom name="left_foot2_collision" size="0.012 0.1"  pos="0.05 -0.02 -0.055" type="capsule"/>
<geom name="left_foot3_collision" size="0.012 0.11" pos="0.054 -0.01 -0.055" type="capsule"/>
<geom name="left_foot4_collision" size="0.012 0.11" pos="0.054 0 -0.055" type="capsule"/>
<geom name="left_foot5_collision" size="0.012 0.11" pos="0.054 0.01 -0.055" type="capsule"/>
<geom name="left_foot6_collision" size="0.012 0.1"  pos="0.05 0.02 -0.055" type="capsule"/>
<geom name="left_foot7_collision" size="0.012 0.08" pos="0.04 0.03 -0.055" type="capsule"/>
```

**问题分析**：
- 胶囊半径仅 **0.012m**（12mm），非常纤细
- 7个胶囊**沿y轴排列**，形成一条"横条"状的接触面
- 胶囊在x方向（前后）的覆盖仅24mm（直径）
- **前后方向几乎没有接触缓冲**：当脚踩到台阶边缘时，12mm的半径很容易滑过台阶棱角
- 相比之下，G1的鞋模型有更大的接触面积和更圆滑的接触面

这种"薄片状"的足部碰撞模型在下楼梯时的问题：
1. 踩到台阶边缘时，容易滑到下一级（因为前后方向缓冲小）
2. 落地时接触点少，容易不稳定
3. 配合弱的穿透惩罚(-2.0)，脚可能直接穿透台阶边缘

### 2.6 问题2的根因排序

| 排名 | 原因 | 确信度 | 影响 |
|------|------|--------|------|
| 1 | 台阶宽度(0.3m) < 机器人自然步幅(~0.5m) | 高 | 几何上就容易跳级 |
| 2 | `feet_air_time`奖励鼓励跳跃行为 | 高 | 强化学习倾向学跳跃 |
| 3 | 缺少逐步下降的引导奖励 | 高 | 没有反向约束力 |
| 4 | 深度相机分辨率过低(16×16) | 高 | 无法感知台阶细节 |
| 5 | V11赤脚模型前后缓冲仅12mm | 中 | 容易滑过台阶边缘 |
| 6 | 足部穿透惩罚弱(-2.0) | 中 | 脚可穿透台阶 |
| 7 | 下楼梯速度命令过高(0.45-0.8m/s) | 中 | 加剧跳跃倾向 |

---

## 三、两个问题的关联性

这两个问题存在**相互加剧**的关系：

```
地形等级卡在6 → 机器人无法接触到更高难度训练
                → 在level 0-6上过度训练
                → 学到的策略偏向"快速通过"而非"精确控制"
                → 下楼梯时选择跳跃而非逐步下降

下楼梯跳跃行为 → 在台阶地形上不稳定
               → 频繁触发终止条件
               → tracking指标累计不足
               → 无法升级到更高等级
               → 地形等级卡在6
```

---

## 四、建议修复方向

### 问题1（地形等级上限）修复优先级

1. **放宽课程升级条件**（效果最大）：
   ```python
   # 方案A：对齐G1
   "lin_vel_threshold": (0.3, 0.6),
   "ang_vel_threshold": (0.0, 0.0),
   
   # 方案B：保留双条件但降低阈值
   "lin_vel_threshold": (0.3, 0.6),
   "ang_vel_threshold": (0.2, 0.3),
   ```

2. **增加足部穿透惩罚**：`-2.0` → `-4.0`（对齐G1）

3. **提高初始等级范围**：`max_init_terrain_level` 从5提到7

4. **增加num_rows**：从10提到15，使相邻等级间难度变化更平滑

### 问题2（下楼梯跳跃）修复优先级

1. **增加台阶宽度**：`step_width` 从0.3m增大到0.5-0.6m（需要验证地形生成器的有效性）

2. **添加下楼梯专用速度限制**：
   ```python
   "pyramid_stairs_inv": {"lin_vel_x": (0.2, 0.5), ...}  # 降低速度
   ```

3. **添加"逐步下降"奖励**：惩罚脚在短时间内下降过大高度

4. **增加足部穿透惩罚**：`-2.0` → `-4.0`

5. **降低stairs地形上的`feet_air_time`权重**：或按地形类型条件化该奖励

6. **提高深度相机分辨率**：增加policy输入的台阶感知能力

---

## 五、涉及的关键文件

| 文件 | 相关性 |
|------|--------|
| `InstinctMJ/src/instinct_mj/tasks/parkour/config/v11/v11_parkour_target_amp_cfg.py` | V1主配置（课程阈值、奖励权重、速度命令） |
| `InstinctMJ/src/instinct_mj/tasks/parkour/config/parkour_env_cfg.py` | 地形生成器配置（台阶参数） |
| `InstinctMJ/src/instinct_mj/tasks/parkour/mdp/curriculums.py` | 课程升降级逻辑 |
| `InstinctMJ/src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py` | 速度命令生成与指标累计 |
| `InstinctMJ/src/instinct_mj/tasks/parkour/mdp/rewards.py` | 奖励函数定义 |
| `mjlab/src/mjlab/terrains/terrain_entity.py` | 地形等级更新逻辑 |
| `InstinctMJ/src/instinct_mj/terrains/height_field/hf_terrains.py` | 台阶地形生成算法 |
| `InstinctMJ/src/instinct_mj/assets/resources/v11/xml/v11.xml` | V11机器人模型（足部碰撞） |
| `InstinctMJ/src/instinct_mj/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py` | G1配置（对比参考） |