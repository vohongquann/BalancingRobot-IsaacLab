# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Linearized wheeled inverted pendulum built from the same numbers as the simulation (URDF + motor constants).

Derivation: ``docs/03-dynamics.md``.

Coordinates: wheel angle ``psi`` (absolute, positive rolls forward) and body pitch ``theta`` (positive leans forward).
The joint velocity Isaac Lab reports is relative to the body, ``q_dot_joint = psi_dot - theta_dot``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from Balance_Car_RL.assets import BALANCE_CAR_RL_ASSETS_DIR
from Balance_Car_RL.car.car_cfg import WHEEL_ARMATURE, WHEEL_RADIUS_M

GRAVITY = 9.81


@dataclass(frozen=True)
class PlantParams:
    """Physical parameters of the balancing robot (SI units)."""

    body_mass: float
    """Mass of the body without wheels [kg]."""
    com_height: float
    """Height of the body centre of mass above the axle [m]."""
    body_inertia: float
    """Body inertia about the pitch axis through its own COM [kg m^2]."""
    wheel_mass: float
    """Mass of both wheels [kg]."""
    wheel_inertia: float
    """Spin inertia of both wheels [kg m^2]."""
    rotor_inertia: float
    """Rotor inertia of both motors reflected to the wheels [kg m^2]."""
    wheel_radius: float

    @classmethod
    def from_urdf(cls, path=None) -> PlantParams:
        root = ET.parse(path or BALANCE_CAR_RL_ASSETS_DIR / "balboa" / "balboa.urdf").getroot()
        links = {link.get("name"): link for link in root.iter("link")}

        def inertial(name):
            inertial = links[name].find("inertial")
            com = np.array(inertial.find("origin").get("xyz").split(), float)
            return float(inertial.find("mass").get("value")), com, inertial.find("inertia").attrib

        m_b, com, inertia = inertial("base_link")
        m_l, _, wheel_inertia = inertial("left_wheel")
        m_r, _, _ = inertial("right_wheel")
        return cls(
            body_mass=m_b,
            com_height=float(com[2]),
            body_inertia=float(inertia["iyy"]),
            wheel_mass=m_l + m_r,
            wheel_inertia=2.0 * float(wheel_inertia["izz"]),
            rotor_inertia=2.0 * WHEEL_ARMATURE,
            wheel_radius=WHEEL_RADIUS_M,
        )


def linear_model(p: PlantParams) -> tuple[np.ndarray, np.ndarray]:
    """Continuous model ``z_dot = A z + B u`` with ``z = [theta, theta_dot, psi_dot]`` and ``u`` the total wheel torque.

    The wheel angle itself does not appear (it is a cyclic coordinate), so wheel *speed* is regulated, not position.
    ``u`` is the sum of both wheel torques [N m]; the motor pushes the wheel forward and the body backward.
    """
    m, L, r = p.body_mass, p.com_height, p.wheel_radius
    mass = np.array(
        [
            [(m + p.wheel_mass) * r**2 + p.wheel_inertia + p.rotor_inertia, m * r * L - p.rotor_inertia],
            [m * r * L - p.rotor_inertia, m * L**2 + p.body_inertia + p.rotor_inertia],
        ]
    )
    minv = np.linalg.inv(mass)
    # q_ddot = minv (B u + [0, m g L theta]),  q = [psi, theta],  B = [1, -1]
    b = minv @ np.array([1.0, -1.0])
    g = minv @ np.array([0.0, m * GRAVITY * L])
    a = np.array(
        [
            [0.0, 1.0, 0.0],
            [g[1], 0.0, 0.0],
            [g[0], 0.0, 0.0],
        ]
    )
    return a, np.array([[0.0], [b[1]], [b[0]]])
