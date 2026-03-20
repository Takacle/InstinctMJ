"""Register Instinct Mj parkour V11 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task

from .agents.instinct_rl_amp_cfg import V11ParkourBaselinePPORunnerCfg, V11ParkourPPORunnerCfg
from .v11_parkour_target_amp_cfg import instinct_v11_parkour_amp_baseline_cfg, instinct_v11_parkour_amp_final_cfg

# Plan-B: upper-body suppressed, discriminator_reward_coef=0.0
register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-V11-v0",
    env_cfg_factory=lambda: instinct_v11_parkour_amp_final_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_v11_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=V11ParkourPPORunnerCfg,
)

# Baseline (对照组): full action scale, discriminator_reward_coef=0.25
register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-V11-Baseline-v0",
    env_cfg_factory=lambda: instinct_v11_parkour_amp_baseline_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_v11_parkour_amp_baseline_cfg(play=True),
    instinct_rl_cfg_factory=V11ParkourBaselinePPORunnerCfg,
)

register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-V11-Play-v0",
    env_cfg_factory=lambda: instinct_v11_parkour_amp_final_cfg(play=True),
    play_env_cfg_factory=lambda: instinct_v11_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=V11ParkourPPORunnerCfg,
)
