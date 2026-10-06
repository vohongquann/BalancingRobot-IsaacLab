# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cascaded PID controller (no simulator needed)."""

import torch

from Balance_Car_RL.car.pid_control import CascadePID


def test_pid_recovers_from_tilt_and_kick(closed_loop):
    for z0 in ([0.25, 0, 0], [-0.25, 0, 0], [0, 0, 7.5]):
        z = closed_loop(CascadePID(), z0)
        assert abs(z[0]) < 1e-2 and abs(z[2]) < 0.5, z0


def test_pid_leans_into_the_fall():
    """A forward lean must give a forward (positive) wheel torque on both wheels."""
    action = CascadePID().act(torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert action.shape == (1, 2) and torch.all(action > 0.0) and torch.all(action <= 1.0)


def test_pid_reset_clears_the_integral():
    pid = CascadePID(ki=1.0)
    obs = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    pid.act(obs)
    assert torch.all(pid.pitch.integral > 0.0)
    pid.reset(torch.tensor([0]))
    assert pid.pitch.integral[0] == 0.0 and pid.pitch.integral[1] > 0.0
