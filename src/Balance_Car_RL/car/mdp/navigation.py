"""Terms of the position task (``BalanceCar-Position-v0``), written after Isaac Lab's navigation task.

Isaac Lab's navigation task (``isaaclab_tasks/contrib/navigation``) puts a policy over a frozen locomotion policy: the
command is a 2D pose (goal x, y and heading, in the robot's frame), the action is the velocity command of the frozen
policy, and the reward is ``1 - tanh(distance / std)`` twice (a coarse and a fine std) and the heading error. The task
uses those reward functions of Isaac Lab (``nav_mdp``), with three changes for a two-wheeled car:

- the goal is drawn around the car where it is (Isaac Lab draws it around the environment origin), so a goal that comes
  while the car drives is as near as one at the start;
- the heading only counts near the goal (:func:`heading_error_near_goal`): the car cannot drive sideways, so on the way
  it has to face the goal, not the final heading;
- the observed goal is clipped to ``max_distance`` (a goal far away is "that way").

The pose is the simulator's, as in Isaac Lab; on the robot it is the wheel odometry (encoders and gyro).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformPose2dCommand, UniformPose2dCommandCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.navigation.mdp as nav_mdp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv


class CarPose2dCommand(UniformPose2dCommand):
    """Isaac Lab's ``UniformPose2dCommand`` with the goal drawn around the car's current position.

    ``command`` is ``[x, y, z, heading]`` of the goal in the car's yaw frame (Isaac Lab's layout). With probability
    ``hold_probability`` the goal is where the car is, facing where it faces: stand still.
    """

    cfg: CarPose2dCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        here = self.robot.data.root_pos_w.torch[env_ids, :2]
        hold = torch.rand(len(env_ids), device=self.device) < self.cfg.hold_probability
        offset = self.pos_command_w[env_ids, :2] - self._env.scene.env_origins[env_ids, :2]
        self.pos_command_w[env_ids, :2] = here + torch.where(hold.unsqueeze(-1), torch.zeros_like(offset), offset)
        heading = self.robot.data.heading_w.torch[env_ids]
        self.heading_command_w[env_ids] = torch.where(hold, heading, self.heading_command_w[env_ids])
        self._update_command()  # the observation of this step already sees the new goal


@configclass
class CarPose2dCommandCfg(UniformPose2dCommandCfg):
    class_type: type = CarPose2dCommand
    hold_probability: float = 0.1
    """Share of the goals at the car itself (stand still where it is)."""


def pose_command_2d(env: ManagerBasedRLEnv, command_name: str = "pose_command", max_distance: float = 2.0):
    """Goal ``[x, y, heading]`` in the car's frame, shape (num_envs, 3); ``(x, y)`` is shortened to ``max_distance``
    [m] if the goal is farther (same direction)."""
    command = env.command_manager.get_command(command_name)
    xy = command[:, :2]
    scale = (max_distance / xy.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
    return torch.cat([xy * scale, command[:, 3:4]], dim=1)


def heading_error_near_goal(env: ManagerBasedRLEnv, std: float, command_name: str = "pose_command"):
    """Heading error [rad] weighted by how near the goal the car is: Isaac Lab's ``heading_command_error_abs`` times its
    ``position_command_error_tanh`` (0 far from the goal, 1 on it)."""
    near = nav_mdp.position_command_error_tanh(env, std=std, command_name=command_name)
    return nav_mdp.heading_command_error_abs(env, command_name=command_name) * near
