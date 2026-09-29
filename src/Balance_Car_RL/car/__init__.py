# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Car: Pololu Balboa 32U4 balancing tasks (``BalanceCar-*``).

Upright (baseline), Pitch (cascade stage 1) and Velocity (cascade stage 2).

The policy action is one normalized torque per wheel (``rl_control/upright_env_cfg.py``). Physical constants live in
``car_cfg.py``; the URDF is ``assets/data/balboa/balboa.urdf``. Each controller is its own package: ``rl_control/``
(PPO), ``pid_control/`` (cascaded PID) and ``lqr_control/`` (LQR).
"""

import gymnasium as gym

from .car_cfg import *  # noqa: F401, F403

##
# Register Gym environments.
##

gym.register(
    id="BalanceCar-Upright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_control.upright_env_cfg:BalanceCarEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{__name__}.rl_control.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="BalanceCar-Pitch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_control.pitch_env_cfg:BalanceCarPitchEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{__name__}.rl_control.agents.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)

gym.register(
    id="BalanceCar-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_control.velocity_env_cfg:BalanceCarVelocityEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{__name__}.rl_control.agents.rsl_rl_ppo_cfg:VelocityPPORunnerCfg",
    },
)
