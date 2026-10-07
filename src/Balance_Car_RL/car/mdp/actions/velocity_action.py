"""Action of the position task: a velocity command, which the frozen velocity policy turns into wheel torques.

policy action (position task, 50 Hz) -> (v_x, w_z) command -> frozen velocity policy -> wheel torques -> PhysX

Isaac Lab's ``PreTrainedPolicyAction`` (navigation task) does the same for a legged robot; this one rebuilds the
observation of the frozen policy itself (:func:`velocity_observation`) instead of a second observation manager, because
the IMU filter of the car (``ImuPitchAndRate``) must run once per step and is shared with the outer observation.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car.mdp.actions.frozen_policy import load_frozen
from Balance_Car_RL.car.mdp.actions.wheel_action import mix_wheels
from Balance_Car_RL.car.mdp.observations import axle_speed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def velocity_observation(
    imu: torch.Tensor, speed: torch.Tensor, yaw_rate: torch.Tensor, command: torch.Tensor, last_action: torch.Tensor
) -> torch.Tensor:
    """Exactly the observation vector the velocity task (``rl_control/velocity_env_cfg.py``) was trained on: ``[imu (2),
    speed, yaw rate, command (v_x, w_z), last action (2)]``. A free function so ``tests/test_cascade.py`` can check the
    order without a simulator."""
    return torch.cat([imu, speed, yaw_rate, command, last_action], dim=1)


class FrozenVelocityAction(ActionTerm):
    """Velocity command in, wheel torques out, through the frozen velocity policy.

    The action ``a`` (shape (num_envs, 2), in [-1, 1]) is scaled to ``(v_x, w_z) = a * (speed_scale, yaw_rate_scale)``.
    The frozen policy of stage ``velocity`` sees what it saw in training and returns its common and turning torque,
    which are mixed into the wheel torques like in that task (``wheel_action.mix_wheels``). Its weights are never
    updated; only the outer policy learns. The IMU estimate is the one the outer observation already computed this step
    (``env.car_imu``).
    """

    cfg: FrozenVelocityActionCfg

    def __init__(self, cfg: FrozenVelocityActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.policy = load_frozen(cfg.stage, self.device)
        joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True, as_proxy=True)
        self._joint_ids = joint_ids.torch
        self._scale = torch.tensor([cfg.speed_scale, cfg.yaw_rate_scale], device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._policy_last_action = torch.zeros(self.num_envs, 2, device=self.device)
        self._torque = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The velocity command ``(v_x [m/s], w_z [rad/s])`` given to the frozen policy."""
        return self._command

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self._command[env_ids] = 0.0
        self._policy_last_action[env_ids] = 0.0
        self._torque[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._command[:] = actions.clamp(-1.0, 1.0) * self._scale
        inner_obs = velocity_observation(
            self._env.car_imu.cache,  # [pitch, pitch rate] of this step's observation, the one the outer policy saw
            axle_speed(self._env, self._joint_ids).unsqueeze(-1),
            self._env.car_imu.yaw_rate,
            self._command,
            self._policy_last_action,
        )
        with torch.inference_mode():
            inner_action = self.policy(inner_obs).clamp(-1.0, 1.0)  # as in training (clip_actions = 1.0)
        self._policy_last_action[:] = inner_action
        self._torque[:] = mix_wheels(inner_action, self.cfg.turn_share) * self.cfg.torque_scale

    def apply_actions(self):
        self._asset.set_joint_effort_target_index(target=self._torque, joint_ids=self._joint_ids)


@configclass
class FrozenVelocityActionCfg(ActionTermCfg):
    class_type: type = FrozenVelocityAction
    joint_names: list[str] = ["left_wheel_joint", "right_wheel_joint"]
    """Left, then right: the order of the velocity task (``mdp/locomotion.py``, ``WHEELS``)."""
    stage: str = "velocity"
    """Name of the frozen stage whose policy is used (``rl_control/frozen/<stage>/policy.pt``)."""
    speed_scale: float = MISSING
    """Forward speed [m/s] for an action of 1: the velocity task's ``SPEED_MAX_M_S``."""
    yaw_rate_scale: float = MISSING
    """Turn rate [rad/s] for an action of 1: the velocity task's ``YAW_RATE_MAX_RAD_S``."""
    torque_scale: float = MISSING
    """Torque [N m] of a wheel action of 1: the wheel stall torque (``car_cfg.WHEEL_STALL_TORQUE_NM``)."""
    turn_share: float = MISSING
    """Share of the wheel range of the turning action: the velocity task's ``TURN_SHARE``."""
