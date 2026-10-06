"""Cascade of two loops, each a plain ``PID`` (``pid.py``, the same class as Drone_RL), pure torch, no Isaac.

    wheel speed --> speed loop (P) --> pitch target --> pitch loop (PI, measured pitch rate as D) --> wheel torque

Structure and closed-loop poles: ``guide/05_controllers.md``.
"""

from __future__ import annotations

import torch

from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM
from Balance_Car_RL.car.pid_control.pid import PID


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
        kp: float = 1.6,
        ki: float = 0.05,
        kd: float = 0.05,
        kv: float = 0.015,
        dt: float = 0.02,
        i_limit: float = 0.2,
    ):
        self.kd, self.dt = kd, dt
        self.speed = PID(kp=kv)  # wheel speed error -> pitch target [rad]
        self.pitch = PID(kp=kp, ki=ki, int_limit=i_limit)  # pitch error -> wheel torque [N m]

    def reset(self, env_ids=None) -> None:
        for loop in (self.speed, self.pitch):
            loop.reset(env_ids)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` is ``[pitch, pitch_rate, wheel_vel_L, wheel_vel_R, ...]``; returns actions (N, 2)."""
        pitch, pitch_rate = obs[:, 0], obs[:, 1]
        psi_dot = 0.5 * (obs[:, 2] + obs[:, 3]) + pitch_rate  # joint speed is relative to the body
        pitch_target = self.speed.update(0.0 - psi_dot, self.dt)
        total_torque = self.pitch.update(pitch - pitch_target, self.dt) + self.kd * pitch_rate
        action = torch.clamp(total_torque / (2.0 * WHEEL_STALL_TORQUE_NM), -1.0, 1.0)
        return action.unsqueeze(1).repeat(1, 2)
