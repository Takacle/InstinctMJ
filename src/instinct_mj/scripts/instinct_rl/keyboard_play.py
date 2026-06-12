"""Keyboard-controlled play for locomotion tasks.

Launches a native viewer where the user controls velocity commands via
keyboard while a trained (or dummy) policy drives the robot.

Usage
-----
    # Random agent (zero actions)
    python -m instinct_mj.scripts.instinct_rl.keyboard_play \\
        Instinct-Locomotion-Flat-G1-Play-v0 --agent zero

    # Trained agent with checkpoint (file path or iteration number)
    python -m instinct_mj.scripts.instinct_rl.keyboard_play \\
        Instinct-Locomotion-Flat-G1-Play-v0 \\
        --checkpoint /path/to/model_8000.pt

    # Trained agent with iteration number
    python -m instinct_mj.scripts.instinct_rl.keyboard_play \\
        Instinct-Locomotion-Flat-G1-Play-v0 \\
        --checkpoint 8000

Keyboard Controls
-----------------
    W / Up       Forward    S / Down    Backward
    A / Left     Turn left  D / Right   Turn right
    Q            Strafe L   E           Strafe R
    X            Stop (return to auto)
    R / Enter    Reset      Space       Pause / resume
    +/-          Speed      ,/.         Switch env
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from instinct_rl.runners import OnPolicyRunner
from mjlab.utils.torch import configure_torch_backends

import instinct_mj.tasks  # noqa: F401
from instinct_mj.controllers.keyboard_controller import (
  KeyboardVelocityController,
  KeyboardPlayViewer,
)
from instinct_mj.envs import InstinctRlEnv
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_instinct_rl_cfg,
  load_runner_cls,
)


@dataclass(frozen=True)
class KeyboardPlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint: str | None = None
  load_run: str | None = None
  checkpoint_pattern: str | None = None
  num_envs: int = 1
  device: str | None = None
  no_terminations: bool = False
  vel_delta: float = 0.1
  max_lin_vel: float = 1.5
  max_ang_vel: float = 3.0


def _resolve_device(device: str | None) -> str:
  if device is not None:
    return device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _build_dummy_policy(agent_mode: str, action_shape: tuple[int, ...], device: str):
  if agent_mode == "zero":

    def zero_policy(_obs: torch.Tensor) -> torch.Tensor:
      return torch.zeros(action_shape, device=device)

    return zero_policy

  def random_policy(_obs: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.rand(action_shape, device=device) - 1.0

  return random_policy


def _resolve_checkpoint(
  task_id: str,
  cfg: KeyboardPlayConfig,
  agent_cfg,
) -> Path:
  if cfg.checkpoint is not None:
    ckpt_path = Path(cfg.checkpoint).expanduser()
    if ckpt_path.is_file() or "/" in cfg.checkpoint or cfg.checkpoint.endswith(".pt"):
      ckpt = ckpt_path.resolve()
      if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
      return ckpt
    try:
      iter_num = int(cfg.checkpoint)
    except ValueError:
      raise ValueError(
        f"--checkpoint accepts a file path or iteration number, got: {cfg.checkpoint!r}"
      )
    target = f"model_{iter_num}.pt"
    log_root = Path("logs") / "instinct_rl" / agent_cfg.experiment_name
    if not log_root.exists():
      raise ValueError(f"Log path does not exist: {log_root}")
    run_regex = cfg.load_run if cfg.load_run not in (None, "") else getattr(agent_cfg, "load_run", None) or ".*"
    candidates: list[Path] = []
    for run in log_root.iterdir():
      if not run.is_dir() or run.name == "wandb_checkpoints":
        continue
      if run_regex == ".*" and run.name == "_play":
        continue
      if re.match(run_regex, run.name):
        candidates.append(run)
    if not candidates:
      raise ValueError(f"No run dirs found in {log_root} matching '{run_regex}'")
    candidates.sort()
    for run_path in reversed(candidates):
      ckpt = run_path / target
      if ckpt.exists():
        return ckpt
    raise ValueError(f"Checkpoint '{target}' not found in matching runs under {log_root}")

  log_root = Path("logs") / "instinct_rl" / agent_cfg.experiment_name
  run_regex = cfg.load_run if cfg.load_run not in (None, "") else getattr(agent_cfg, "load_run", None) or ".*"
  if run_regex in (None, ""):
    run_regex = ".*"

  ckpt_regex = cfg.checkpoint_pattern
  if ckpt_regex in (None, ""):
    ckpt_regex = getattr(agent_cfg, "load_checkpoint", None) or "model_.*.pt"
  if not log_root.exists():
    raise ValueError(f"Log path does not exist: {log_root}")
  candidates = []
  for run in log_root.iterdir():
    if not run.is_dir() or run.name == "wandb_checkpoints":
      continue
    if run_regex == ".*" and run.name == "_play":
      continue
    if re.match(run_regex, run.name):
      candidates.append(run)
  if not candidates:
    raise ValueError(f"No run dirs found in {log_root} matching '{run_regex}'")
  candidates.sort()
  for run_path in reversed(candidates):
    files = [
      f.name for f in run_path.iterdir() if f.is_file() and re.match(ckpt_regex, f.name)
    ]
    if not files:
      continue
    files.sort(key=lambda n: f"{n:0>15}")
    return run_path / files[-1]
  raise ValueError(
    f"No checkpoint found in {log_root} (run_regex={run_regex}, ckpt_regex={ckpt_regex})"
  )


def _disable_headless_debug_visualization(env_cfg) -> None:
  scene = getattr(env_cfg, "scene", None)
  if scene is not None:
    for sensor_cfg in getattr(scene, "sensors", ()):
      if getattr(sensor_cfg, "debug_vis", None) is not None:
        sensor_cfg.debug_vis = False
  commands = getattr(env_cfg, "commands", None)
  if isinstance(commands, dict):
    for cmd_cfg in commands.values():
      if getattr(cmd_cfg, "debug_vis", None) is not None:
        cmd_cfg.debug_vis = False


class _ViewerEnvAdapter:
  def __init__(self, vec_env: InstinctRlVecEnvWrapper):
    self._ve = vec_env
    self.num_envs = vec_env.num_envs

  @property
  def device(self):
    return self._ve.device

  @property
  def cfg(self):
    return self._ve.cfg

  @property
  def unwrapped(self):
    return self._ve.unwrapped

  def get_observations(self):
    obs, _ = self._ve.get_observations()
    return obs

  def step(self, actions):
    return self._ve.step(actions)

  def reset(self):
    return self._ve.reset()

  def close(self):
    return self._ve.close()


def run_keyboard_play(task_id: str, cfg: KeyboardPlayConfig) -> None:
  configure_torch_backends()
  os.environ["MUJOCO_GL"] = "glfw"

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_instinct_rl_cfg(task_id)
  device = _resolve_device(cfg.device)

  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  if not has_display:
    _disable_headless_debug_visualization(env_cfg)

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.no_terminations:
    env_cfg.terminations = {}

  env = InstinctRlEnv(cfg=env_cfg, device=device, render_mode=None)
  vec_env = InstinctRlVecEnvWrapper(
    env,
    policy_group=agent_cfg.policy_observation_group,
    critic_group=agent_cfg.critic_observation_group,
  )
  viewer_env = _ViewerEnvAdapter(vec_env)

  if cfg.agent in {"zero", "random"}:
    policy = _build_dummy_policy(cfg.agent, (vec_env.num_envs, vec_env.num_actions), device)
  else:
    checkpoint = _resolve_checkpoint(task_id, cfg, agent_cfg)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(checkpoint))
    print(f"[INFO] Loaded checkpoint: {checkpoint}")
    policy = runner.get_inference_policy(device=device)

  kb = KeyboardVelocityController(
    device=device,
    vel_delta=cfg.vel_delta,
    max_lin_vel=cfg.max_lin_vel,
    max_ang_vel=cfg.max_ang_vel,
  )
  viewer = KeyboardPlayViewer(viewer_env, policy, kb)

  print()
  print("=" * 60)
  print(f"  Keyboard Play — {task_id}")
  print("=" * 60)
  print(f"  Envs: {cfg.num_envs}   Device: {device}")
  print()
  print("  Movement:")
  print("    ↑/I  Forward    ↓/K  Backward")
  print("    ←/J  Turn L     →/L  Turn R")
  print("    U    Strafe L   O    Strafe R")
  print("    X    Stop       Enter  Reset")
  print()
  print("  Terrain (parkour):")
  print("    1-9,0  Switch terrain type")
  print("    [  Easier    ]  Harder")
  print()
  print("  Viewer:")
  print("    Space  Pause    +/-  Speed")
  print("    ,/.  Switch env  P  Plots")
  print("    R  Debug vis    A  Show all envs")
  print("=" * 60)
  print()

  try:
    viewer.run()
  finally:
    viewer_env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    KeyboardPlayConfig,
    args=remaining_args,
    default=KeyboardPlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_keyboard_play(chosen_task, args)


if __name__ == "__main__":
  main()
