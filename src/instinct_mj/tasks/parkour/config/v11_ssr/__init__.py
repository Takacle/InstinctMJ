"""Register Instinct Mj parkour V11 SSR tasks.

SSR (Surefooted and Symmetric tRaversal) adds the Imagined Foothold
Guidance reward on top of the V11 parkour baseline. See:

  Yu et al., "SSR: Scaling Surefooted and Symmetric Humanoid Traversal
  to the Open World", 2026.
"""

from instinct_mj.tasks.registry import register_instinct_task

from .agents.instinct_rl_imaginator_cfg import V11SsrPPORunnerCfg
from .v11_ssr_env_cfg import instinct_v11_ssr_final_cfg

register_instinct_task(
    task_id="Instinct-Parkour-SSR-V11-v0",
    env_cfg_factory=lambda: instinct_v11_ssr_final_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_v11_ssr_final_cfg(play=True),
    instinct_rl_cfg_factory=V11SsrPPORunnerCfg,
)

register_instinct_task(
    task_id="Instinct-Parkour-SSR-V11-Play-v0",
    env_cfg_factory=lambda: instinct_v11_ssr_final_cfg(play=True),
    play_env_cfg_factory=lambda: instinct_v11_ssr_final_cfg(play=True),
    instinct_rl_cfg_factory=V11SsrPPORunnerCfg,
)
