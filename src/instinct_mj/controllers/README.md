# 键盘控制器模块

## 任务目标

提供一个**独立于 `play.py` 命令区域机制**的键盘控制器，允许用户通过键盘直接操控机器人前后左右移动。

### 背景

`play.py` 依赖环境配置中的命令管理器（`CommandManager`）在固定时间间隔内**随机重采样**速度命令（如 `UniformVelocityCommandCfg`）。用户无法手动干预机器人的运动方向，只能观看策略按随机命令执行。这种"命令区域"（command-area）方案适用于训练和评估，但不满足交互式操控需求。

### 需求

- 通过键盘 ↑/↓/←/→ 或 I/K/J/L 控制机器人前进/后退/左转/右转
- 通过 U/O 控制左移/右移
- 不依赖命令管理器的随机重采样逻辑
- 控制器为独立文件，可被任意脚本导入使用
- 支持 locomotion 和 parkour 任务

### 适用范围

此模块支持所有暴露 ``vel_command_b`` 张量的命令术语，包括：

- **`UniformVelocityCommand`**（locomotion 平地）：`_update_command` 仅在 `heading_command=True` 时修改角速度分量，或在 `is_standing_env` 时清零命令。前后各覆盖一次即可可靠覆盖。

- **`PoseVelocityCommand`**（parkour）：`_update_command` 每步从 `pos_command_w`、heading、target distance、standing/random masks 完全重算 `vel_command_b`。通过在 `env.step()` 前后各覆盖一次 `vel_command_b`，确保策略推理和观测都能看到键盘命令。同时填充 `velocity_commands` 观测项的历史缓冲区（`history_length=8`），使策略在带历史观测的任务中也能正确响应键盘输入。

## 设计方案

### 架构

```
┌──────────────────────┐     键盘事件       ┌──────────────────────────┐
│  KeyboardPlayViewer  │ ────────────────▶  │ KeyboardVelocityController│
│  (NativeMujocoViewer  │                    │  (状态机: lin_vel_x/y,    │
│   子类)               │ ◀────────────────  │   ang_vel_z)             │
│                      │   get_command()     │                          │
└──────┬───────────────┘                    └──────────────────────────┘
       │
       │ vel_command_b.copy_(cmd)
       ▼
┌──────────────────────────┐
│  CommandTerm              │
│  (vel_command_b 张量)      │
│  UniformVelocityCommand   │
│  PoseVelocityCommand      │
└──────────────────────────┘
```

### 核心设计决策

1. **纯状态机分离**：`KeyboardVelocityController` 不依赖任何环境/视图对象，仅维护速度状态和键盘映射逻辑。控制器不产生任何输出（无 `print`/`log`），通过 `_changed`/`_cleared` 标志让 viewer 决定是否记录状态变更。

2. **Viewer 子类拦截**：`KeyboardPlayViewer` 继承 `NativeMujocoViewer`，重写 `_safe_key_callback`（拦截按键）和 `_execute_step`（注入命令）。在策略推理前后各执行一次命令覆盖，确保即使命令术语在 `_update_command()` 中重算 `vel_command_b`，观测和策略仍能看到键盘值。

3. **历史缓冲区覆盖**：对于使用 `history_length > 0` 的 `velocity_commands` 观测项（如 parkour 的 `history_length=8`），`_override_velocity_command()` 不仅覆盖 `vel_command_b`，还通过 `_fill_velocity_obs_history()` 将所有历史缓冲区 slot 填充为当前键盘值。这确保策略观测的完整历史窗口都反映键盘输入。

4. **通用命令术语查找**：`_get_command_term()` 通过 `hasattr(term, 'vel_command_b')` 检查命令术语兼容性，不限定具体类型。支持 `UniformVelocityCommand`（locomotion）和 `PoseVelocityCommand`（parkour）。

5. **键位对齐 mjlab 原生**：运动控制使用 I/K/J/L + 箭头键，不与 mjlab native viewer 快捷键冲突。原生快捷键（R → debug 可视化、A → 显示全部 env、Enter → 重置、P → reward 图）均完整保留。

6. **运行时地形切换**（仅 parkour）：数字键 1-9/0 切换地形类型，`[`/`]` 调整难度等级。切换通过修改 `terrain_types` 和 `terrain_levels` 后重置环境实现，地形高度场在初始化时已全部生成，无需重新计算。

7. **统一日志通道**：所有运行时输出通过 viewer 的 `self.log()` 方法，遵循 mjlab viewer 的 verbosity 控制。

### 文件结构

```
src/instinct_mj/controllers/
├── __init__.py              # 导出 KeyboardVelocityController, KeyboardPlayViewer, build_keyboard_play_viewer
└── keyboard_controller.py   # 核心实现
```

### 主要接口

```python
class KeyboardVelocityController:
  def __init__(self, device: str, vel_delta=0.1, max_lin_vel=1.5, max_ang_vel=3.0, ang_vel_factor=2.0)
  def handle_key(self, key: int) -> None       # 处理按键事件，设置 _changed/_cleared 标志
  def get_command(self, num_envs: int) -> Tensor | None  # 获取速度命令
  def reset(self) -> None                       # 重置状态

class KeyboardPlayViewer(NativeMujocoViewer):
  def __init__(self, env, policy, vel_controller, command_name="base_velocity", **kwargs)

def build_keyboard_play_viewer(env, policy, device, ...) -> KeyboardPlayViewer  # 工厂函数
```

### 使用方式

```bash
# Locomotion (G1)
instinct-keyboard-play Instinct-Locomotion-Flat-G1-Play-v0 --agent zero

# Locomotion (V11)
instinct-keyboard-play Instinct-Locomotion-Flat-V11-Play-v0 --agent zero

# Parkour (V11)
instinct-keyboard-play Instinct-Parkour-Target-Amp-V11-Play-v0 --agent zero

# Parkour (G1)
instinct-keyboard-play Instinct-Parkour-Target-Amp-G1-Play-v0 --agent zero

# 加载训练模型
instinct-keyboard-play Instinct-Locomotion-Flat-G1-Play-v0 \
    --checkpoint-file /path/to/model_8000.pt
```
