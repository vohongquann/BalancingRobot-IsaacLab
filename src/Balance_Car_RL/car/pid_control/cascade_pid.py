# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cascaded PID balance controller: an outer wheel-speed loop sets the pitch target of an inner pitch PID.

Structure and closed-loop poles: ``docs/05-controllers.md``.
"""

from __future__ import annotations

import torch

from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM


class CascadePID:
    """Vectorized cascade. Inner: PID on pitch with the measured pitch rate as the D term. Outer: P on wheel speed.

    A robot that is leaning forward must drive its wheels forward, so all gains are positive. The outer loop
    leans the robot against its own drift: a positive speed asks for a backward pitch target.

    The default gains (N m/rad, N m/(rad s), N m/rad, rad/(rad/s)) were found by a search in the simulator with the IMU
    observation; a first guess from the linearized model (:mod:`..lqr_control.model`) was too stiff and amplified the
    pitch-estimate noise into torque chatter. See ``docs/05-controllers.md``.
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
        self.kp, self.ki, self.kd, self.kv, self.dt, self.i_limit = kp, ki, kd, kv, dt, i_limit
        self._integral: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._integral is not None:
            if env_ids is None:
                self._integral.zero_()
            else:
                self._integral[env_ids] = 0.0

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` is ``[pitch, pitch_rate, wheel_vel_L, wheel_vel_R, ...]``; returns actions (N, 2)."""
        pitch, pitch_rate = obs[:, 0], obs[:, 1]
        psi_dot = 0.5 * (obs[:, 2] + obs[:, 3]) + pitch_rate
        if self._integral is None:
            self._integral = torch.zeros_like(pitch)
        pitch_target = -self.kv * psi_dot
        error = pitch - pitch_target
        self._integral = torch.clamp(self._integral + error * self.dt, -self.i_limit, self.i_limit)
        total_torque = self.kp * error + self.ki * self._integral + self.kd * pitch_rate
        action = torch.clamp(total_torque / (2.0 * WHEEL_STALL_TORQUE_NM), -1.0, 1.0)
        return action.unsqueeze(1).repeat(1, 2)
