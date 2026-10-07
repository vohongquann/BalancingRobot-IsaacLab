"""Rewards of the car tasks that Isaac Lab does not have: upright, turn speed, and the error to a scalar command.

Written like Drone_RL's rewards and Isaac Lab's ``track_lin_vel_xy_exp``: the robot is read through Isaac Lab
(``projected_gravity``, ``base_lin_vel``, ``joint_vel``). The ``_exp`` terms are ``exp(-error^2 / std^2)``: 1 on the
command, 0 far from it; ``speed_error_l1`` keeps a slope where the kernel is flat. The command comes from
``commands.ScalarCommand``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg

from Balance_Car_RL.car.estimation import pitch_from_gravity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Body upright: ``exp(-sin^2(tilt) / std^2)``. The horizontal components of gravity in the body frame are
    sin(tilt), so roll and pitch both count. See ``guide/04_training.md``."""
    return torch.exp(-isaac_mdp.projected_gravity(env, asset_cfg)[:, :2].square().sum(-1) / std**2)


def yaw_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared turn rate about the body z axis [rad^2/s^2]: the car has no reason to spin, but two independent wheel
    torques can make it (``ang_vel_xy_l2`` of Isaac Lab only counts roll and pitch)."""
    return isaac_mdp.base_ang_vel(env)[:, 2].square()


# ── Cascade stages ──────────────────────────────────────────────────────────────────────
# ``command_name``: the command term of the stage in training (``target``).


def body_pitch(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pitch [rad] of the body, positive leaning forward, from the gravity direction in the body frame."""
    return pitch_from_gravity(isaac_mdp.projected_gravity(env))


def pitch_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "target") -> torch.Tensor:
    """Pitch against the commanded one (pitch stage)."""
    command = env.command_manager.get_command(command_name)
    return torch.exp(-(body_pitch(env) - command[:, 0]).square() / std**2)


def speed_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "target") -> torch.Tensor:
    """Forward speed (body frame) against the commanded one (velocity stage)."""
    command = env.command_manager.get_command(command_name)
    return torch.exp(-(isaac_mdp.base_lin_vel(env)[:, 0] - command[:, 0]).square() / std**2)


def speed_error_l1(env: ManagerBasedRLEnv, command_name: str = "target") -> torch.Tensor:
    """Absolute forward-speed error [m/s] (velocity stage): dense, the kernel of ``speed_error_exp`` is flat far from
    the command."""
    command = env.command_manager.get_command(command_name)
    return (isaac_mdp.base_lin_vel(env)[:, 0] - command[:, 0]).abs()
