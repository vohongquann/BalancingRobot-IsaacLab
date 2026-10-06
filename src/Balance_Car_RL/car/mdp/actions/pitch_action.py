"""Action of the velocity stage: the action is a pitch target, the frozen pitch policy turns it into wheel torques.

policy action (velocity stage, 50 Hz) -> pitch target -> frozen pitch policy -> wheel torques -> PhysX
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car.mdp.actions.frozen_policy import load_frozen
from Balance_Car_RL.car.mdp.observations import axle_speed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _stage1_observation(
    imu: torch.Tensor, wheel_vel: torch.Tensor, last_action: torch.Tensor, pitch_target: torch.Tensor
) -> torch.Tensor:
    """Exactly the observation vector stage 1 (``rl_control/pitch_env_cfg.py``) was trained on: ``[imu, wheel_vel,
    last_action, pitch_target]``. A free function so ``tests/test_cascade.py`` can check the order without a
    simulator: :class:`FrozenPitchAction` cannot be constructed outside one (it needs a live ``env.scene``)."""
    return torch.cat([imu, wheel_vel, last_action, pitch_target], dim=1)


class FrozenPitchAction(ActionTerm):
    """Pitch target in, wheel torque out, through a frozen stage-1 policy.

    The action ``a`` (shape (num_envs, 1), in [-1, 1]) is scaled to a pitch target ``a * pitch_scale`` [rad]. The frozen
    policy of stage ``pitch`` sees exactly what it saw in training, ``[pitch, pitch rate, left wheel speed, right wheel
    speed, its previous action x2, pitch target]``, and returns the wheel actions, which are scaled by the motor stall
    torque like in stage 1. Its weights are never updated; only the outer policy learns.

    The IMU estimate is the one the environment's observation already computed this step (``env.car_imu``), so both
    stages read the same sensor.
    """

    cfg: FrozenPitchActionCfg

    def __init__(self, cfg: FrozenPitchActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.policy = load_frozen(cfg.stage, self.device)
        joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True, as_proxy=True)
        self._joint_ids = joint_ids.torch
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._output = torch.zeros(self.num_envs, 1, device=self.device)
        self._policy_last_action = torch.zeros(self.num_envs, 2, device=self.device)
        self._torque = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._output

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self._output[env_ids] = 0.0
        self._policy_last_action[env_ids] = 0.0
        self._torque[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target = actions.clamp(-1.0, 1.0) * self.cfg.pitch_scale
        if self.cfg.speed_guard is not None:
            # stage 1 never trained on a same-direction target above this speed (mdp/commands.py, speed_guard):
            # zero it instead, rather than feed the frozen policy an input it has never seen. The speed is the
            # estimate of the robot (encoders + IMU), not the simulator's, so this is deployable
            speed = axle_speed(self._env, self._joint_ids).unsqueeze(-1)
            out_of_distribution = (speed.abs() > self.cfg.speed_guard) & (torch.sign(target) == torch.sign(speed))
            target = torch.where(out_of_distribution, torch.zeros_like(target), target)
        self._output[:] = target
        imu = self._env.car_imu.cache  # [pitch, pitch rate] of the last observation, the one stage 1 saw when it chose
        wheel_vel = self._asset.data.joint_vel.torch[:, self._joint_ids]
        inner_obs = _stage1_observation(imu, wheel_vel, self._policy_last_action, self._output)
        with torch.inference_mode():
            inner_action = self.policy(inner_obs).clamp(-1.0, 1.0)  # as in training (clip_actions = 1.0)
        self._policy_last_action[:] = inner_action
        self._torque[:] = inner_action * self.cfg.torque_scale

    def apply_actions(self):
        self._asset.set_joint_effort_target_index(target=self._torque, joint_ids=self._joint_ids)


@configclass
class FrozenPitchActionCfg(ActionTermCfg):
    class_type: type = FrozenPitchAction
    joint_names: list[str] = [".*_wheel_joint"]
    stage: str = "pitch"
    """Name of the frozen stage whose policy is used (``rl_control/frozen/<stage>/policy.pt``)."""
    pitch_scale: float = MISSING
    """Pitch target [rad] for an action of 1. Must equal the frozen stage's own ``PITCH_TARGET_MAX_RAD``; the check is
    ``tests/test_cascade.py``."""
    torque_scale: float = MISSING
    """Torque [N m] for an inner action of 1: the wheel stall torque (``car_cfg.WHEEL_STALL_TORQUE_NM``)."""
    speed_guard: float | None = None
    """Estimated forward speed [m/s] above which a target that would speed the car up further is zeroed instead of
    applied: the same guard the frozen stage's own command was trained with (``mdp/commands.py::ScalarCommand``), so
    the outer stage cannot ask it for a lean it has never seen. ``None``: no guard."""
