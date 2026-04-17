"""Evaluate Instinct-RL policies with roboharness visual checkpoints.

Usage::

    # With trained policy
    instinct-play-harness Instinct-Locomotion-Flat-G1-v0 --checkpoint 5000

    # With zero-action policy (test harness pipeline)
    instinct-play-harness Instinct-Locomotion-Flat-G1-v0 --agent zero

    # Custom cameras and phases
    instinct-play-harness Instinct-Locomotion-Flat-G1-v0 --agent trained \\
        --cameras front side top --phase-steady 400 --num-episodes 5
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from instinct_rl.runners import OnPolicyRunner
from mjlab.utils.torch import configure_torch_backends
from roboharness.backends.mjlab_backend import MjlabBackend
from roboharness.core.checkpoint import Checkpoint
from roboharness.core.harness import Harness

import instinct_mj.tasks  # noqa: F401 — trigger entry-point task registration
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_instinct_rl_cfg,
    load_runner_cls,
)


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for roboharness evaluation."""

    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    checkpoint: int | None = None
    load_run: str | None = None
    device: str | None = None
    num_episodes: int = 3
    episode_steps: int = 1000
    output_dir: str = "./harness_output"
    cameras: tuple[str, ...] = ("robot/front", "robot/side", "robot/top")
    render_width: int = 640
    render_height: int = 480
    phase_initial: int = 50
    phase_steady: int = 500
    phase_terminal: int = 900


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_policy(
    task_id: str,
    cfg: HarnessConfig,
    vec_env: InstinctRlVecEnvWrapper,
    device: str,
):
    """Load a policy callable from config."""
    action_shape = (vec_env.num_envs, vec_env.num_actions)

    if cfg.agent == "zero":

        def zero_policy(_obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(action_shape, device=device)

        return zero_policy

    if cfg.agent == "random":

        def random_policy(_obs: torch.Tensor) -> torch.Tensor:
            return 2.0 * torch.rand(action_shape, device=device) - 1.0

        return random_policy

    # --- trained agent: resolve checkpoint and load ---
    from instinct_mj.scripts.instinct_rl.play import PlayConfig, _resolve_checkpoint

    play_cfg = PlayConfig(
        checkpoint_file=cfg.checkpoint_file,
        checkpoint=cfg.checkpoint,
        load_run=cfg.load_run,
    )
    agent_cfg = load_instinct_rl_cfg(task_id)
    checkpoint_path = _resolve_checkpoint(task_id, play_cfg, agent_cfg)

    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(
        vec_env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=device,
    )
    runner.load(str(checkpoint_path))
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    return runner.get_inference_policy(device=device)


def run_harness(task_id: str, cfg: HarnessConfig) -> None:
    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")

    device = _resolve_device(cfg.device)
    agent_cfg = load_instinct_rl_cfg(task_id)
    camera_list = list(cfg.cameras)

    # --- Build env config ourselves to avoid registry path issues ---
    env_cfg = load_env_cfg(task_id, play=True)

    # --- Create backend (wraps ManagerBasedRlEnv) ---
    backend = MjlabBackend(
        env_cfg=env_cfg,
        cameras=camera_list,
        render_width=cfg.render_width,
        render_height=cfg.render_height,
        device=device,
    )
    env = backend._env

    # --- Create harness for checkpoint capture ---
    task_slug = task_id.replace("/", "_")
    harness = Harness(backend, output_dir=cfg.output_dir, task_name=task_slug)

    # --- Wrap env for policy loading (shares same env instance) ---
    vec_env = InstinctRlVecEnvWrapper(
        env,
        policy_group=agent_cfg.policy_observation_group,
        critic_group=agent_cfg.critic_observation_group,
    )

    # --- Load policy ---
    policy = _load_policy(task_id, cfg, vec_env, device)

    # --- Phase boundaries ---
    phases = {
        "initial": cfg.phase_initial,
        "steady": cfg.phase_steady,
        "terminal": cfg.phase_terminal,
    }

    # --- Evaluation loop ---
    for ep in range(cfg.num_episodes):
        print(f"\n=== Episode {ep + 1}/{cfg.num_episodes} ===")

        # Reset via harness (resets backend → env)
        harness.reset()
        # Sync vec_env observations after harness reset
        obs, _ = vec_env.get_observations()

        captures = {}

        for step in range(cfg.episode_steps):
            with torch.no_grad():
                actions = policy(obs)

            # Step via vec_env (correct obs packing, no double-compute)
            obs, _rew, _dones, _extras = vec_env.step(actions)

            # Capture at phase boundaries (reads mj_data from same env)
            for phase_name, phase_step in phases.items():
                if step == phase_step:
                    cp = Checkpoint(name=phase_name, cameras=camera_list)
                    result = harness.capture(cp)
                    captures[phase_name] = result
                    print(
                        f"  [{phase_name}] step={step}  "
                        f"sim_time={result.sim_time:.2f}s  "
                        f"views={[v.name for v in result.views]}"
                    )

        print(f"  Done: {cfg.episode_steps} steps, {len(captures)} checkpoints")

    output_path = Path(cfg.output_dir) / task_slug
    print(f"\nOutput: {output_path}")


def main() -> None:
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        HarnessConfig,
        args=remaining_args,
        default=HarnessConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_harness(chosen_task, args)


if __name__ == "__main__":
    main()
