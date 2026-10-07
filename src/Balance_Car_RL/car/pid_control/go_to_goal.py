"""Classical go-to-goal controller of the position layer: two plain ``PID`` (``pid.py``), pure torch, no Isaac.

    goal (x, y, heading) in the car's frame --> speed PID on x       --> v_x  --> frozen velocity network --> torques
                                            --> turn PID on an angle --> w_z

The speed loop drives the distance ahead of the car, ``x``, to zero (a goal behind is reached backing up). The turn loop
turns the car toward the goal, or away from it when the goal is behind (it backs up to it), and near the goal turns it
to the goal heading: the angle is ``(1 - s) bearing + s heading`` with ``s = 1 - tanh(distance / NEAR_M)``, so the
switch is smooth (no derivative kick). Both outputs are clipped to the ranges the velocity network was trained on.

It is the PID of ``BalanceCar-Position-Gains-v0`` (``mdp/actions/position_gain_action.py``), whose network writes its
six gains; the tuned gains below are its zero action and the "PID" of ``tools/compare_layers.py --layer position``.
"""

from __future__ import annotations

import math

import torch

from Balance_Car_RL.car.pid_control.pid import PID

GOAL_SPEED_KP = 1.0
"""Speed loop P gain [(m/s)/m]: 0.4 m/s (the top speed) from 0.4 m away."""
GOAL_TURN_KP = 2.0
"""Turn loop P gain [(rad/s)/rad]: 2 rad/s (the top turn rate) from 1 rad off."""
GOAL_INT_LIMIT = 0.5
"""Limit of the integral of each error [m s, rad s]."""
NEAR_M = 0.1
"""[m] Within about this distance the turn loop follows the goal heading instead of the bearing."""


def goal_errors(goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Errors of the two loops for goals ``[x, y, heading]`` (N, 3) in the car's frame: distance ahead [m] and angle
    [rad] (see the module docstring)."""
    x, y, heading = goal[:, 0], goal[:, 1], goal[:, 2]
    bearing = torch.atan2(y, x)
    behind = bearing.abs() > math.pi / 2
    bearing = torch.where(behind, torch.atan2(-y, -x), bearing)  # the goal seen from the back of the car
    near = 1.0 - torch.tanh(torch.hypot(x, y) / NEAR_M)
    return x, (1.0 - near) * bearing + near * heading


class GoToGoalPID:
    """``step(goal) -> (v_x, w_z)``, each (N,), with the tuned gains or the gains set on ``speed`` / ``turn``."""

    def __init__(self, dt: float, speed_max: float, turn_max: float):
        self.dt = dt
        self.speed = PID(kp=GOAL_SPEED_KP, out_limit=speed_max, int_limit=GOAL_INT_LIMIT)
        self.turn = PID(kp=GOAL_TURN_KP, out_limit=turn_max, int_limit=GOAL_INT_LIMIT)

    def reset(self, env_ids=None) -> None:
        self.speed.reset(env_ids)
        self.turn.reset(env_ids)

    def step(self, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ahead, angle = goal_errors(goal)
        return self.speed.update(ahead, self.dt), self.turn.update(angle, self.dt)
