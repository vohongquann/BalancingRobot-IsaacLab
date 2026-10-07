"""Action of the velocity task: a common torque and a turning torque, mixed into the two wheel torques.

``[common, turn]`` in [-1, 1] -> left = common - share x turn, right = common + share x turn (clipped to [-1, 1]), times
the stall torque. The yaw of the Balboa is very light: 0.06 N m of difference between the wheels spins it at 16 rad/s
within a few steps. With one independent torque per wheel, the exploration noise of PPO (std 0.3 of the stall torque
on each wheel) alone turned the cars at about 1.5 rad/s, and the first run never learned to follow a turn-rate
command. The turn gets ``share`` of the range, so its noise is that much smaller, and the action is symmetric by
construction (a front-back mirror flips both, a left-right mirror only the turn).
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import JointEffortAction, JointEffortActionCfg
from isaaclab.utils import configclass


def mix_wheels(action: torch.Tensor, turn_share: float) -> torch.Tensor:
    """``[common, turn]`` (N, 2), clipped to [-1, 1], to ``[left, right]`` wheel actions in [-1, 1]."""
    action = action.clamp(-1.0, 1.0)
    common, turn = action[:, 0], turn_share * action[:, 1]
    return torch.stack([common - turn, common + turn], dim=1).clamp(-1.0, 1.0)


class WheelMixAction(JointEffortAction):
    """Isaac Lab's ``JointEffortAction`` on the two wheels (left, right), with the common and turning torque as its
    action: the processed actions are the wheel torques [N m]."""

    cfg: WheelMixActionCfg

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = mix_wheels(actions, self.cfg.turn_share) * self._scale


@configclass
class WheelMixActionCfg(JointEffortActionCfg):
    class_type: type = WheelMixAction
    joint_names: list[str] = ["left_wheel_joint", "right_wheel_joint"]
    preserve_order: bool = True
    scale: float = MISSING
    """Torque [N m] of a wheel action of 1: the stall torque (``car_cfg.WHEEL_STALL_TORQUE_NM``)."""
    turn_share: float = MISSING
    """Share of the wheel range the turning action spans."""
