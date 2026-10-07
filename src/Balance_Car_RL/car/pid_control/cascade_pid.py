"""Cascade of two loops, each a plain ``PID`` (``pid.py``, the same class as Drone_RL), pure torch, no Isaac.

    wheel speed --> speed loop (P) --> pitch target --> pitch loop (PI, measured pitch rate as D) --> wheel torque

Structure and closed-loop poles: ``guide/05_controllers.md``.
"""

from __future__ import annotations

import torch

from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM
from Balance_Car_RL.car.pid_control.pid import PID

PITCH_KP = 1.6
"""Pitch loop P gain [N m/rad]."""
PITCH_KI = 0.05
"""Pitch loop I gain [N m/(rad s)]."""
PITCH_KD = 0.05
"""Pitch loop D gain on the measured pitch rate [N m s/rad]."""
PITCH_INT_LIMIT = 0.2
"""Limit of the integral of the pitch error [rad s]."""
SPEED_KV = 0.015
"""Speed loop P gain [rad/(rad/s)]."""
DT_S = 0.02
"""Control period [s]: the policy rate of the environments, 50 Hz."""


def upright_state(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(pitch, pitch rate, absolute wheel speed)`` from the observation of the upright task ``[pitch, pitch_rate,
    wheel_vel_L, wheel_vel_R, ...]``: the encoders are relative to the body, so the pitch rate is added."""
    pitch, pitch_rate = obs[:, 0], obs[:, 1]
    return pitch, pitch_rate, 0.5 * (obs[:, 2] + obs[:, 3]) + pitch_rate


def both_wheels(total_torque: torch.Tensor) -> torch.Tensor:
    """Total wheel torque [N m] (N,) -> the same normalized action on both wheels (N, 2)."""
    action = torch.clamp(total_torque / (2.0 * WHEEL_STALL_TORQUE_NM), -1.0, 1.0)
    return action.unsqueeze(1).repeat(1, 2)


def wheel_action(
    pitch_loop: PID, kd, pitch: torch.Tensor, pitch_rate: torch.Tensor, target: torch.Tensor, dt: float
) -> torch.Tensor:
    """The pitch loop of the cascade: ``pitch_loop`` (PI) on ``pitch - target``, plus ``kd`` times the measured pitch
    rate, as a normalized torque on both wheels, shape (N, 2). Gains of ``pitch_loop`` and ``kd`` are floats or
    tensors of shape (N,); :class:`CascadePID` and the gain tasks (``mdp/actions/gain_action.py``) share it."""
    return both_wheels(pitch_loop.update(pitch - target, dt) + kd * pitch_rate)


class CascadePID:
    """Outer: P on wheel speed. Inner: PI on pitch, plus the measured pitch rate as the D term.

    A robot that is leaning forward must drive its wheels forward, so all gains are positive and the pitch error is
    ``pitch - target``. The outer loop leans the robot against its own drift: a positive speed asks for a backward
    pitch target.

    The default gains (N m/rad, N m/(rad s), N m/rad, rad/(rad/s)) were found by a search in the simulator with the IMU
    observation; a first guess from the linearized model (:mod:`..lqr_control.model`) was too stiff and amplified the
    pitch-estimate noise into torque chatter. See ``guide/05_controllers.md``.
    """

    def __init__(
        self,
        kp: float = PITCH_KP,
        ki: float = PITCH_KI,
        kd: float = PITCH_KD,
        kv: float = SPEED_KV,
        dt: float = DT_S,
        i_limit: float = PITCH_INT_LIMIT,
    ):
        self.kd, self.dt = kd, dt
        self.speed = PID(kp=kv)  # wheel speed error -> pitch target [rad]
        self.pitch = PID(kp=kp, ki=ki, int_limit=i_limit)  # pitch error -> wheel torque [N m]

    def reset(self, env_ids=None) -> None:
        for loop in (self.speed, self.pitch):
            loop.reset(env_ids)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` is ``[pitch, pitch_rate, wheel_vel_L, wheel_vel_R, ...]``; returns actions (N, 2)."""
        return self.step(*upright_state(obs))

    def step(self, pitch, pitch_rate, psi_dot, psi_dot_target=0.0) -> torch.Tensor:
        """Balance at the absolute wheel speed ``psi_dot_target`` [rad/s] (0: stand still); actions (N, 2)."""
        pitch_target = self.speed.update(psi_dot_target - psi_dot, self.dt)
        return wheel_action(self.pitch, self.kd, pitch, pitch_rate, pitch_target, self.dt)
