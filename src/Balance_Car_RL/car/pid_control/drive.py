"""Classical drive controllers, the baselines of the RL velocity and position tasks: pure torch, no Isaac.

    (v_x, w_z) --> balance controller at the wheel speed v_x / r (CascadePID or LQR) --> common torque
               --> PI on the turn rate (gyro)                                          --> turning torque

The output is the ``[common, turn]`` action of ``BalanceCar-Velocity-v0`` (``mdp/actions/wheel_action.py``), so a
classical controller drives the same robot through the same action as the velocity network. :class:`DrivePosition` puts
the go-to-goal PID (``go_to_goal.py``) on top: the classical counterpart of ``BalanceCar-Position-v0``.
"""

from __future__ import annotations

import torch

from Balance_Car_RL.car.car_cfg import WHEEL_RADIUS_M
from Balance_Car_RL.car.pid_control.go_to_goal import GoToGoalPID
from Balance_Car_RL.car.pid_control.pid import PID

TURN_KP = 0.3
"""Turn loop P gain [turn action/(rad/s)]. Tuned in the simulator on steps of 1 and 2 rad/s (``tools/compare_layers.py
--layer turn``): 0.1 rose in 0.2 s and overshot 15 % on the integral, 0.3 rises in 0.04 s with 2-6 % overshoot."""
TURN_KI = 2.0
"""Turn loop I gain [turn action/rad]: removes the steady error the wheel scrub leaves."""


class DriveVelocity:
    """``step(pitch, pitch_rate, speed, turn_rate, command) -> [common, turn]`` (N, 2), command ``(v_x, w_z)`` (N, 2).

    ``balance`` is a :class:`~Balance_Car_RL.car.pid_control.CascadePID` or an
    :class:`~Balance_Car_RL.car.lqr_control.LQRController`; ``speed`` is the encoder speed estimate [m/s]."""

    def __init__(self, balance, dt: float):
        self.balance, self.dt = balance, dt
        self.turn = PID(kp=TURN_KP, ki=TURN_KI, out_limit=1.0, int_limit=1.0 / TURN_KI)

    def reset(self, env_ids=None) -> None:
        if hasattr(self.balance, "reset"):
            self.balance.reset(env_ids)
        self.turn.reset(env_ids)

    def step(self, pitch, pitch_rate, speed, turn_rate, command) -> torch.Tensor:
        common = self.balance.step(pitch, pitch_rate, speed / WHEEL_RADIUS_M, command[:, 0] / WHEEL_RADIUS_M)[:, 0]
        turn = self.turn.update(command[:, 1] - turn_rate, self.dt)
        return torch.stack([common, turn], dim=1)


class DrivePosition:
    """``step(pitch, pitch_rate, speed, turn_rate, goal) -> [common, turn]``: the go-to-goal PID writes the command of
    :class:`DriveVelocity`; goal ``[x, y, heading]`` (N, 3) in the car's frame."""

    def __init__(self, balance, dt: float, speed_max: float, turn_max: float):
        self.goal = GoToGoalPID(dt, speed_max, turn_max)
        self.drive = DriveVelocity(balance, dt)
        self.command = None  # the last (v_x, w_z) command, for plots

    def reset(self, env_ids=None) -> None:
        self.goal.reset(env_ids)
        self.drive.reset(env_ids)

    def step(self, pitch, pitch_rate, speed, turn_rate, goal) -> torch.Tensor:
        self.command = torch.stack(self.goal.step(goal), dim=1)
        return self.drive.step(pitch, pitch_rate, speed, turn_rate, self.command)
