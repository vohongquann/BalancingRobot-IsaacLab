"""Actions of the gain tasks: a network writes the PID gains of its layer, and the PID of ``pid_control/`` computes the
command with them (``mdp/gains.py``). The same two layers as the RL cascade, with a PID between the network and the
layer below:

    PitchGainAction   gains (3) -> pitch PID -> wheel torques                                    (50 Hz)
    SpeedGainAction   gains (3) -> speed PID -> pitch target -> pitch gain layer below -> wheel torques

The pitch gain layer below the speed layer is the frozen network of ``BalanceCar-Pitch-Gains-v0``
(``frozen/pitch_gains/policy.pt``).
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.mdp.actions.frozen_policy import load_frozen
from Balance_Car_RL.car.mdp.actions.pitch_action import apply_speed_guard
from Balance_Car_RL.car.mdp.gains import GAIN_LAYERS
from Balance_Car_RL.car.mdp.observations import axle_speed
from Balance_Car_RL.car.pid_control import PID, wheel_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _pitch_gains_observation(
    imu: torch.Tensor, wheel_vel: torch.Tensor, last_gains: torch.Tensor, pitch_target: torch.Tensor
) -> torch.Tensor:
    """Exactly the observation vector of ``BalanceCar-Pitch-Gains-v0`` (``rl_control/pitch_env_cfg.py``): ``[imu,
    wheel_vel, last gains, pitch_target]``, 8 values. A free function so ``tests/test_gains.py`` can check the order
    without a simulator."""
    return torch.cat([imu, wheel_vel, last_gains, pitch_target], dim=1)


class PitchGainLayer:
    """The pitch loop of ``CascadePID`` with the gains of a network: ``[pitch, pitch rate]``, pitch target and gain
    action -> normalized wheel torques (N, 2). Pure torch."""

    def __init__(self, dt: float):
        self.dt = dt
        self.layer = GAIN_LAYERS["pitch"]
        self.pid = PID(kp=0.0, ki=0.0, int_limit=self.layer.int_limit)
        self.gains = torch.zeros(0, 3)  # kp, ki, kd of the last step

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def step(self, gain_action: torch.Tensor, imu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The gains are those of ``gain_action`` (N, 3); a zero action is the tuned PID."""
        kp, ki, kd = self.layer.gains(gain_action)
        self.pid.kp, self.pid.ki = kp, ki
        self.gains = torch.stack([kp, ki, kd], dim=-1)
        return wheel_action(self.pid, kd, imu[:, 0], imu[:, 1], target[:, 0], self.dt)


class SpeedGainLayer:
    """The speed loop of ``CascadePID`` with the gains of a network: forward speed and its target [m/s] and gain action
    -> pitch target (N, 1) [rad], limited to ``pitch_scale``. Pure torch."""

    def __init__(self, dt: float, pitch_scale: float):
        self.dt = dt
        self.layer = GAIN_LAYERS["speed"]
        self.pid = PID(kp=0.0, ki=0.0, kd=0.0, out_limit=pitch_scale, int_limit=self.layer.int_limit)

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def step(self, gain_action: torch.Tensor, speed: torch.Tensor, speed_target: torch.Tensor) -> torch.Tensor:
        self.pid.kp, self.pid.ki, self.pid.kd = self.layer.gains(gain_action)
        error = (speed_target[:, 0] - speed[:, 0]) / car_cfg.WHEEL_RADIUS_M  # the loop works on wheel rad/s
        return self.pid.update(error, self.dt).unsqueeze(-1)


class _GainAction(ActionTerm):
    """Common part: three gains in [-1, 1] as the action, the wheel torques as the output."""

    cfg: _GainActionCfg

    def __init__(self, cfg: _GainActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True, as_proxy=True)
        self._joint_ids = joint_ids.torch
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._gains = torch.zeros(self.num_envs, 3, device=self.device)
        self._torque = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The gains of the trained layer now (N, 3): kp, ki, kd."""
        return self._gains

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self._gains[env_ids] = 0.0
        self._torque[env_ids] = 0.0

    def apply_actions(self):
        self._asset.set_joint_effort_target_index(target=self._torque, joint_ids=self._joint_ids)


class PitchGainAction(_GainAction):
    """Gains of the pitch loop in, wheel torque out: the PID is run with the gains the network writes, on the pitch
    target of the command (``cfg.command_name``) and the IMU estimate of the observation (``env.car_imu``)."""

    cfg: PitchGainActionCfg

    def __init__(self, cfg: PitchGainActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.layer = PitchGainLayer(env.step_dt)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.layer.reset(env_ids)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        target = self._env.command_manager.get_command(self.cfg.command_name)
        action = self.layer.step(self._raw_actions, self._env.car_imu.cache, target)
        self._gains[:] = self.layer.gains
        self._torque[:] = action * self.cfg.torque_scale


class SpeedGainAction(_GainAction):
    """Gains of the speed loop in, wheel torque out: the speed PID turns the speed error into a pitch target, which the
    pitch gain layer below turns into wheel torques."""

    cfg: SpeedGainActionCfg

    def __init__(self, cfg: SpeedGainActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.layer = SpeedGainLayer(env.step_dt, cfg.pitch_scale)
        self.below = PitchGainLayer(env.step_dt)
        self.policy = load_frozen(cfg.below_stage, self.device)
        self._below_action = torch.zeros(self.num_envs, 3, device=self.device)  # gains of the layer below; 0: tuned
        self._pitch_target = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self.layer.reset(env_ids)
        self.below.reset(env_ids)
        self._below_action[env_ids] = 0.0
        self._pitch_target[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        speed = axle_speed(self._env, self._joint_ids).unsqueeze(-1)  # the robot's own estimate, not the simulator's
        speed_target = self._env.command_manager.get_command(self.cfg.command_name)
        pid = self.layer.pid
        target = self.layer.step(self._raw_actions, speed, speed_target)
        self._gains[:] = torch.stack([pid.kp, pid.ki, pid.kd], dim=-1)
        if self.cfg.speed_guard is not None:
            target = apply_speed_guard(target, speed, self.cfg.speed_guard)
        self._pitch_target[:] = target
        imu = self._env.car_imu.cache
        wheel_vel = self._asset.data.joint_vel.torch[:, self._joint_ids]
        observation = _pitch_gains_observation(imu, wheel_vel, self._below_action, self._pitch_target)
        with torch.inference_mode():
            self._below_action[:] = self.policy(observation).clamp(-1.0, 1.0)  # as in training (clip_actions = 1)
        self._torque[:] = self.below.step(self._below_action, imu, self._pitch_target) * self.cfg.torque_scale


@configclass
class _GainActionCfg(ActionTermCfg):
    joint_names: list[str] = [".*_wheel_joint"]
    command_name: str = "target"
    """Command term the PID tracks: the pitch target [rad] or the forward speed [m/s]."""
    torque_scale: float = MISSING
    """Torque [N m] for a normalized wheel action of 1: the wheel stall torque (``car_cfg.WHEEL_STALL_TORQUE_NM``)."""


@configclass
class PitchGainActionCfg(_GainActionCfg):
    class_type: type = PitchGainAction


@configclass
class SpeedGainActionCfg(_GainActionCfg):
    class_type: type = SpeedGainAction
    pitch_scale: float = MISSING
    """Largest pitch target [rad] the speed PID writes: the range the pitch layer below was trained on
    (``pitch_env_cfg.PITCH_TARGET_MAX_RAD``)."""
    speed_guard: float | None = None
    """Estimated forward speed [m/s] above which a pitch target that would speed the car up further is zeroed: the
    guard of the command the pitch layer below was trained with. ``None``: no guard."""
    below_stage: str = "pitch_gains"
    """Frozen stage of the pitch gain layer below (``rl_control/frozen/<stage>/policy.pt``)."""
