"""Register Instinct Mj parkour U20 tasks.

NOTE: Uses lazy imports in factory functions to avoid circular import
when mjlab's task autoloader triggers eager loading of instinct_mj.tasks
before instinct_mj.assets.u20 has finished initializing.

Import chain that would cause the warning:
  assets.u20 -> mjlab import side effect -> task autoload -> parkour __init__
  -> u20 __init__ -> u20_parkour_target_amp_cfg -> assets.u20 (partially init)
"""

from instinct_mj.tasks.registry import register_instinct_task


def _build_cfg(play: bool = False):
    from .u20_parkour_target_amp_cfg import instinct_u20_parkour_amp_final_cfg
    return instinct_u20_parkour_amp_final_cfg(play=play)


def _build_runner_cfg():
    from .agents.instinct_rl_amp_cfg import U20ParkourPPORunnerCfg
    return U20ParkourPPORunnerCfg


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-U20-v0",
    env_cfg_factory=lambda: _build_cfg(play=False),
    play_env_cfg_factory=lambda: _build_cfg(play=True),
    instinct_rl_cfg_factory=_build_runner_cfg,
)

register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-U20-Play-v0",
    env_cfg_factory=lambda: _build_cfg(play=True),
    play_env_cfg_factory=lambda: _build_cfg(play=True),
    instinct_rl_cfg_factory=_build_runner_cfg,
)
