"""Register Instinct Mj locomotion V11 tasks."""

from instinct_mj.tasks.registry import register_instinct_task

from .flat_env_cfg import instinct_v11_locomotion_flat_env_cfg
from .rl_cfgs import v11_locomotion_instinct_rl_cfg

register_instinct_task(
    task_id="Instinct-Locomotion-Flat-V11-v0",
    env_cfg_factory=lambda: instinct_v11_locomotion_flat_env_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_v11_locomotion_flat_env_cfg(play=True),
    instinct_rl_cfg_factory=v11_locomotion_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Flat-V11-Play-v0",
    env_cfg_factory=lambda: instinct_v11_locomotion_flat_env_cfg(play=True),
    play_env_cfg_factory=lambda: instinct_v11_locomotion_flat_env_cfg(play=True),
    instinct_rl_cfg_factory=v11_locomotion_instinct_rl_cfg,
)
