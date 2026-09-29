# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LQR controller and the linearized plant (no simulator needed)."""

import numpy as np
import torch

from Balance_Car_RL.car.lqr_control import LQRController, design_lqr


def test_open_loop_has_one_unstable_pole(plant):
    a, _, _ = plant
    eig = np.linalg.eigvals(a)
    assert np.sum(eig.real > 1e-6) == 1
    assert 5.0 < eig.real.max() < 20.0  # docs/03-dynamics.md: about 9.2 rad/s


def test_lqr_stabilizes_the_linear_model(plant):
    _, ad, bd = plant
    assert np.max(np.abs(np.linalg.eigvals(ad - np.outer(bd, design_lqr())))) < 1.0


def test_lqr_recovers_from_tilt_and_kick(closed_loop):
    for z0 in ([0.25, 0, 0], [-0.25, 0, 0], [0, 0, 7.5]):
        z = closed_loop(LQRController(), z0)
        assert abs(z[0]) < 1e-2 and abs(z[2]) < 0.5, z0


def test_lqr_leans_into_the_fall():
    """A forward lean must give a forward (positive) wheel torque on both wheels."""
    action = LQRController().act(torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert action.shape == (1, 2) and torch.all(action > 0.0) and torch.all(action <= 1.0)
