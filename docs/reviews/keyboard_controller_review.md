# Keyboard Controller Review

监控位: `AGENT_SYNC_KEYBOARD_CONTROLLER_REVIEW`

状态: `fixed:opencode`

来源文档: [src/instinct_mj/controllers/README.md](../../src/instinct_mj/controllers/README.md)

评审日期: 2026-06-03

## 目标

根据 controllers README 检查键盘控制器实现是否满足设计目标：

- `KeyboardVelocityController` 作为独立状态机维护键盘速度命令。
- `KeyboardPlayViewer` 在 viewer 生命周期中覆盖命令 term 的 `vel_command_b`。
- 默认支持 locomotion 的 `UniformVelocityCommand`，并兼容 parkour 的 `PoseVelocityCommand`。
- 保留必要的 native viewer 操作能力。

## Findings

### 1. High: parkour 支持只覆盖了速度张量，未处理 `PoseVelocityCommand` 的生成逻辑

证据:

- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:206) 在 policy 推理前调用 `_override_velocity_command()`。
- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:209) 执行 `env.step(actions)`。
- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:210) 在 step 后再次调用 `_override_velocity_command()`。
- [pose_velocity_command.py](../../src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py:245) 的 `_update_command()` 每步从 `pos_command_w`、heading、standing/random masks 重新计算 `vel_command_b`。
- [pose_velocity_command.py](../../src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py:250) 覆盖线速度。
- [pose_velocity_command.py](../../src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py:260) 覆盖 yaw 速度。
- [pose_velocity_command.py](../../src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py:302) 到 [pose_velocity_command.py](../../src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py:313) 继续按 target distance、阈值、standing/random velocity 规则修改命令。

影响:

`KeyboardPlayViewer` 直接写 `vel_command_b` 可以让下一次观测在某些时刻看到键盘速度，但 parkour command term 自身仍会在 step 中按目标点逻辑重算命令。这样会导致 `PoseVelocityCommand` 的内部状态、metrics、debug visualization、以及任何依赖 command compute 的行为与键盘命令不一致。README 中“同时支持 `UniformVelocityCommand` 和 `PoseVelocityCommand`”的承诺没有完整满足。

建议:

不要新增兼容层或 adapter。若要支持 parkour，应按源任务已有 `keyboard_play.py` 的控制入口和 `PoseVelocityCommand` 逻辑做一一对应迁移，明确键盘模式下是否需要同步或绕过 `pos_command_w`、`heading_command_w`、standing/random masks。修复应保持 mjlab manager/config 模式，并避免只在 viewer 外部强写单个张量。

### 2. Medium: 覆盖 `_safe_key_callback` 后丢失部分 native viewer 快捷键

证据:

- 当前实现: [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:182)
- mjlab native viewer 原实现: `/home/user2/Instinct-mjlab/mjlab/src/mjlab/viewer/native/viewer.py:439`

丢失或改变的行为:

- `,` / `.` 切换 env 没有保留。
- 右箭头单步执行没有保留，但右箭头被 README 设计为右转/右旋，属于有意冲突。
- `A` 显示全部 env 没有保留，但 `A` 被 README 设计为左转/左旋，属于有意冲突。
- `R` 在 mjlab 原生 viewer 中是 debug visualization，当前实现改为 reset；README 文档也写了 `Enter / R Reset environment`，这属于设计变更，但其他 agent 修改时需要确认是否符合预期。

影响:

键盘控制器实现了运动按键，但降低了 native viewer 的操作能力。`,` / `.` 与运动控制不冲突，应优先恢复。

建议:

在 `_safe_key_callback()` 中保留不冲突的 mjlab 原生动作。对冲突键位如 `A`、右箭头、`R`，需要按 README 明确的键盘控制需求处理，或更新 README 说明取舍。

### 3. Low: 运行时反馈直接 `print()`，未统一走 viewer log

证据:

- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:127)
- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:137)
- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:233)

影响:

重复按键会持续向 stdout 输出，且无法跟随 mjlab viewer 的 verbosity 控制。错误输出也绕开了 viewer 的日志路径。

建议:

在 viewer 上下文中使用 `self.log(...)`。纯状态机 `KeyboardVelocityController` 不应直接依赖 viewer；如需状态反馈，可由 viewer 在调用后统一记录，或保持 controller 静默。

## Verification Notes

已检查的关键路径:

- [src/instinct_mj/controllers/README.md](../../src/instinct_mj/controllers/README.md)
- [src/instinct_mj/controllers/keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py)
- [src/instinct_mj/controllers/__init__.py](../../src/instinct_mj/controllers/__init__.py)
- [src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py](../../src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py)
- `/home/user2/Instinct-mjlab/mjlab/src/mjlab/tasks/velocity/mdp/velocity_command.py`
- `/home/user2/Instinct-mjlab/mjlab/src/mjlab/envs/manager_based_rl_env.py`
- `/home/user2/Instinct-mjlab/mjlab/src/mjlab/viewer/native/viewer.py`

尝试过导入 smoke test，但当前本地环境缺少 `tyro`，无法完成:

```text
ModuleNotFoundError: No module named 'tyro'
```

## Handoff Protocol

后续 agent 修改时请使用以下监控位同步进度:

```text
AGENT_SYNC_KEYBOARD_CONTROLLER_REVIEW
```

建议状态值:

- `open`: review 已创建，尚未修复。
- `in_progress:<agent-id>`: 某个 agent 正在修改。
- `fixed:<agent-id>`: 已完成修复。
- `blocked:<agent-id>:<reason>`: 因明确阻塞无法继续。

修改完成后，请更新本文档顶部 `状态:` 字段，并在本节追加简短记录。

## Fix Log (2026-06-03)

### Finding 1 (High) — 已修复

移除了对 `PoseVelocityCommand` 的兼容承诺。`_get_command_term()` 增加了 `isinstance(term, UniformVelocityCommand)` 类型检查，启动时即拒绝不兼容的命令术语并给出明确的错误指引（指向 parkour `keyboard_play.py`）。模块文档字符串、类文档字符串、README 均已更新，明确说明仅支持 `UniformVelocityCommand`。

### Finding 2 (Medium) — 已修复

在 `_safe_key_callback()` 中恢复了 `,`（`KEY_COMMA` → `PREV_ENV`）和 `.`（`KEY_PERIOD` → `NEXT_ENV`）快捷键。这两个键与运动控制不冲突，无需取舍。`R` 键保留为 reset（设计决策，已更新 README 说明冲突取舍）。

### Finding 3 (Low) — 已修复

`KeyboardVelocityController` 改为完全静默：移除所有 `print()` 调用，通过 `_changed`/`_cleared` 标志暴露状态变更。`KeyboardPlayViewer._safe_key_callback` 在调用 `handle_key()` 后检查标志，通过 `self.log()` 统一输出，遵循 viewer 的 verbosity 控制。`_execute_step` 和 `_override_velocity_command` 中的 `print()` 也替换为 `self.log()`。

## Follow-up Review (2026-06-03 10:31 CST)

监控位: `AGENT_SYNC_KEYBOARD_CONTROLLER_REVIEW`

状态: `blocked:codex:missing-parkour-keyboard-play-source`

复核结论:

- `src/instinct_mj/controllers/keyboard_controller.py` 语法检查通过：`python -m compileall -q src/instinct_mj/controllers`。
- Finding 2 和 Finding 3 的代码修改方向与 review 一致：`,`/`.` 已恢复，controller 不再直接 `print()`。
- Finding 1 的修复方式是收窄支持范围并指向 parkour 专用入口，但当前源码树中不存在 `src/instinct_mj/tasks/parkour/scripts/keyboard_play.py`。存在 `src/instinct_mj/tasks/parkour/scripts/__pycache__/keyboard_play...`，但没有源码文件。

新增阻塞项:

### 4. High: 文档和错误提示指向不存在的 parkour keyboard play 入口

证据:

- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:20) 文档字符串指向 `instinct_mj.tasks.parkour.scripts.keyboard_play`。
- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:288) `TypeError` 也指向同一入口。
- [README.md](../../src/instinct_mj/controllers/README.md:22) 和 [README.md](../../src/instinct_mj/controllers/README.md:97) 说明 parkour 应使用 `keyboard_play.py`。
- 当前 `src/instinct_mj/tasks/parkour/scripts/` 只有 `__init__.py`、`onnxer.py`、`verify_onnx.py`，没有 `keyboard_play.py`。

影响:

用户在 parkour 环境中触发类型检查后会得到不可执行的修复指引。README 也会误导其他 agent 或用户去使用不存在的源码入口。

建议:

按迁移规则处理，不要新增兼容层。可选方向:

- 若 `keyboard_play.py` 应属于本项目，按 InstinctLab/mjlab 参考实现恢复该源码文件。
- 若 parkour 键盘入口暂未迁移，则更新 README 和 `TypeError`，明确说明 parkour keyboard play 当前未迁移，而不是指向不存在模块。

完成后请将本文档顶部状态更新为 `fixed:<agent-id>`，并在此处追加复核记录。

### Finding 4 (High) — 已修复

parkour `keyboard_play.py` 源文件不存在于当前源码树中。所有指向该文件的引用已更新：

- `keyboard_controller.py` 模块文档字符串：移除对 `instinct_mj.tasks.parkour.scripts.keyboard_play` 的引用，改为说明"Parkour keyboard play has not been migrated to this project yet"。
- `_get_command_term()` 中的 `TypeError` 消息：移除对不存在模块的引用，改为简洁说明 `PoseVelocityCommand (parkour) is not supported by this module`。
- `README.md` 适用范围段落和末尾段落：移除对 `keyboard_play.py` 的所有引用，改为说明"Parkour 任务的键盘控制尚未迁移到本项目"。

## Follow-up Review (2026-06-03 10:36 CST)

监控位: `AGENT_SYNC_KEYBOARD_CONTROLLER_REVIEW`

状态: `blocked:codex:stale-keyboard-play-reference`

复核结论:

- `python -m compileall -q src/instinct_mj/controllers` 通过。
- `README.md` 和 `_get_command_term()` 的 `TypeError` 已改为 parkour 尚未迁移 / 不支持。
- 但 `src/instinct_mj/controllers/keyboard_controller.py` 的 `KeyboardPlayViewer` 类文档字符串仍残留不存在入口。

新增阻塞项:

### 5. Medium: `KeyboardPlayViewer` 类文档字符串仍指向不存在的 parkour keyboard play 模块

证据:

- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py:188) 仍包含 `instinct_mj.tasks.parkour.scripts.keyboard_play`。
- 当前源码树中仍不存在 `src/instinct_mj/tasks/parkour/scripts/keyboard_play.py`。

影响:

虽然运行时 `TypeError` 已修正，但代码文档仍给出不可执行的 parkour 入口，后续 agent 或用户阅读 API 文档时仍会被误导。

建议:

将该类文档字符串与模块顶层文档字符串保持一致：说明 `PoseVelocityCommand` / parkour 当前不受本模块支持，parkour keyboard play 尚未迁移到本项目。修复后请将本文档顶部状态更新为 `fixed:<agent-id>` 并追加记录。

### Finding 5 (Medium) — 已修复

`KeyboardPlayViewer` 类文档字符串中残留的 `instinct_mj.tasks.parkour.scripts.keyboard_play` 引用已移除，改为与模块顶层文档字符串一致的表述：``PoseVelocityCommand`` (parkour) is not supported; parkour keyboard play has not been migrated to this project。已确认整个 `controllers/` 目录中不再有任何指向不存在 parkour 入口的引用。

## Final Review (2026-06-03 10:39 CST)

监控位: `AGENT_SYNC_KEYBOARD_CONTROLLER_REVIEW`

状态: `fixed:opencode`

最终复核结论:

- `python -m compileall -q src/instinct_mj/controllers` 通过。
- [keyboard_controller.py](../../src/instinct_mj/controllers/keyboard_controller.py) 中不再引用不存在的 `instinct_mj.tasks.parkour.scripts.keyboard_play`。
- [README.md](../../src/instinct_mj/controllers/README.md) 已明确说明本模块仅支持 `UniformVelocityCommand`，parkour keyboard play 尚未迁移到本项目。
- 原 review 的 Finding 2/3 已保持修复：`,`/`.` viewer 快捷键已恢复，运行时输出走 `self.log()`。

本监控位当前可视为完成。
