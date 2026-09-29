# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures: the discretized linear plant and a closed loop that goes through a controller's obs interface."""

import numpy as np
import pytest
import torch
from scipy.signal import cont2discrete

from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM
from Balance_Car_RL.car.lqr_control import PlantParams, linear_model


@pytest.fixture(scope="session")
def plant():
    """``(a, ad, bd)``: continuous ``A``, and the 50 Hz ZOH ``A_d``, ``B_d`` (shape 3) of the linear plant."""
    a, b = linear_model(PlantParams.from_urdf())
    ad, bd, *_ = cont2discrete((a, b, np.eye(3), np.zeros((3, 1))), 0.02, method="zoh")
    return a, ad, bd[:, 0]


@pytest.fixture(scope="session")
def closed_loop(plant):
    """``run(controller, z0, steps)``: the linear plant driven (with saturation) through ``controller.act(obs)``."""
    _, ad, bd = plant

    def run(controller, z0, steps=250):
        z = np.array(z0, float)
        for _ in range(steps):
            theta, theta_dot, psi_dot = z
            joint_speed = psi_dot - theta_dot  # what the wheel encoders report (relative to the body)
            obs = torch.tensor([[theta, theta_dot, joint_speed, joint_speed, 0.0, 0.0]], dtype=torch.float32)
            action = controller.act(obs)[0, 0].item()
            z = ad @ z + bd * (2.0 * WHEEL_STALL_TORQUE_NM * action)
        return z

    return run
