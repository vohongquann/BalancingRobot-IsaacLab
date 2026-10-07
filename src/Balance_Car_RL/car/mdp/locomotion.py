"""Terms of the velocity task (``BalanceCar-Velocity-v0``), written after Isaac Lab's locomotion velocity task.

Isaac Lab's legged robots follow a velocity command ``(v_x, v_y, w_z)`` with one policy that writes the joint actions
directly (``isaaclab_tasks/core/velocity``). A two-wheeled car cannot move sideways, so its command is ``(v_x, w_z)``:
forward speed and turn rate. The command term is Isaac Lab's ``UniformVelocityCommand`` with ``lin_vel_y = 0``, the
tracking rewards are Isaac Lab's ``track_lin_vel_xy_exp`` / ``track_ang_vel_z_exp``; what is here is what the car needs
on top: the command without its always-zero ``v_y`` and an L1 term of the turn-rate error. The speed and turn rate the
robot measures are ``observations.wheel_speed_estimate`` (encoders) and ``observations.gyro_yaw_rate``.

Kept out of ``observations.py`` and ``rewards.py`` on purpose: those files are in the contract of the frozen pitch gain
stage (``actions/frozen_policy.py``), and editing them would make it look stale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

WHEELS = SceneEntityCfg("robot", joint_names=["left_wheel_joint", "right_wheel_joint"], preserve_order=True)
"""Left wheel first, then right: the order of every per-wheel vector of the velocity task."""


def velocity_command(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """``[v_x, w_z]`` of a ``UniformVelocityCommand`` (its ``v_y`` is always zero for the car), shape (num_envs, 2)."""
    return env.command_manager.get_command(command_name)[:, [0, 2]]


def yaw_rate_error_l1(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Absolute turn-rate error [rad/s]: keeps a slope where the kernel of ``track_ang_vel_z_exp`` is flat."""
    command = env.command_manager.get_command(command_name)
    return (isaac_mdp.base_ang_vel(env)[:, 2] - command[:, 2]).abs()
