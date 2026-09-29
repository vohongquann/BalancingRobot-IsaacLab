# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Consistency checks for the Balboa URDF and the constants of ``car_cfg`` (no simulator needed)."""

import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np

from Balance_Car_RL.assets import BALANCE_CAR_RL_ASSETS_DIR
from Balance_Car_RL.car import car_cfg

URDF = BALANCE_CAR_RL_ASSETS_DIR / "balboa" / "balboa.urdf"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _robot() -> ET.Element:
    return ET.parse(URDF).getroot()


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def test_meshes_exist():
    for mesh in _robot().iter("mesh"):
        assert (URDF.parent / mesh.get("filename")).is_file(), mesh.get("filename")


def test_wheel_joints_drive_the_same_direction():
    """Both wheel axes must map to +y of the car, so equal joint velocities mean straight driving."""
    from scipy.spatial.transform import Rotation as R

    axes = []
    for name in ("left_wheel_joint", "right_wheel_joint"):
        joint = next(j for j in _robot().iter("joint") if j.get("name") == name)
        rot = R.from_euler("xyz", _floats(joint.find("origin").get("rpy"))).as_matrix()
        axes.append(rot @ _floats(joint.find("axis").get("xyz")))
    assert np.allclose(axes[0], [0, 1, 0], atol=1e-4)
    assert np.allclose(axes[1], [0, 1, 0], atol=1e-4)


def test_wheel_geometry_matches_constants():
    cylinder = next(_robot().iter("cylinder"))
    assert abs(float(cylinder.get("radius")) - car_cfg.WHEEL_RADIUS_M) < 1e-4
    ys = [_floats(j.find("origin").get("xyz"))[1] for j in _robot().iter("joint")]
    assert np.isclose(ys[0], -ys[1]) and abs(ys[0]) > 0.04  # symmetric track


def test_base_com_is_above_the_axle():
    """The URDF is posed at the inverted-pendulum equilibrium: COM straight above the axle, not too tall."""
    base = next(link for link in _robot().iter("link") if link.get("name") == "base_link")
    com = _floats(base.find("inertial/origin").get("xyz"))
    assert abs(com[0]) < 1e-3 and abs(com[1]) < 1e-3
    assert 0.01 < com[2] < 0.10


def test_inertias_are_physical():
    for link in _robot().iter("link"):
        inertial = link.find("inertial")
        assert float(inertial.find("mass").get("value")) > 0.0
        i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        tensor = np.array(
            [[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]]
        )
        eig = np.linalg.eigvalsh(tensor)
        assert eig.min() > 0.0
        assert eig.max() <= eig.sum() / 2 * (1 + 1e-6)  # triangle inequality of rigid-body inertia


def test_urdf_masses_match_car_cfg():
    """The URDF is generated from these constants: rebuild it (tools/build_balboa_urdf.py) when this fails."""
    links = {link.get("name"): link for link in _robot().iter("link")}
    mass = {n: float(link.find("inertial/mass").get("value")) for n, link in links.items()}
    assert np.isclose(mass["base_link"], car_cfg.BASE_MASS_KG)
    assert np.isclose(mass["left_wheel"], car_cfg.WHEEL_MASS_KG) and np.isclose(
        mass["right_wheel"], car_cfg.WHEEL_MASS_KG
    )


def test_motor_constants():
    assert np.isclose(car_cfg.TOTAL_RATIO, 148.3, atol=0.1)  # Pololu gear ratio chart: 49:17 with 50:1
    assert 0.15 < car_cfg.WHEEL_STALL_TORQUE_NM < 0.3
    assert 15.0 < car_cfg.WHEEL_NO_LOAD_SPEED_RAD_S < 35.0


def test_ros_torque_scale_matches_training():
    text = (ROOT / "ros/src/car_bridge/car_bridge/policy.py").read_text()
    scale = float(re.search(r"^TORQUE_SCALE_NM = ([0-9.]+)", text, re.MULTILINE).group(1))
    assert np.isclose(scale, car_cfg.WHEEL_STALL_TORQUE_NM, atol=1e-4)
