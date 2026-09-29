# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Reward in (0, 1] for keeping the body upright: exp(-(tilt / std)^2). See ``docs/04-training.md``.

    Uses the horizontal components of gravity in the body frame, i.e. sin(tilt), so roll and pitch
    both count.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)
    return torch.exp(-tilt_sq / std**2)


def wheel_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared wheel speed [rad^2/s^2], discourages the car from driving away while balancing."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1)


def pitch_track_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward in (0, 1] for holding the commanded pitch: exp(-((pitch - target) / std)^2)."""
    asset: Articulation = env.scene["robot"]
    g_b = asset.data.projected_gravity_b.torch
    pitch = torch.atan2(g_b[:, 0], -g_b[:, 2])
    error = pitch - env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square(error) / std**2)


def speed_track_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward in (0, 1] for tracking the commanded forward speed: exp(-((v - target) / std)^2), v in the body frame."""
    asset: Articulation = env.scene["robot"]
    error = asset.data.root_lin_vel_b.torch[:, 0] - env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square(error) / std**2)


def speed_error_l1(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Absolute forward-speed error [m/s]: dense, the kernel of ``speed_track_exp`` is flat far from the target."""
    asset: Articulation = env.scene["robot"]
    return torch.abs(asset.data.root_lin_vel_b.torch[:, 0] - env.command_manager.get_command(command_name)[:, 0])
