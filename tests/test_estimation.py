# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IMU to pitch: the torch filter of the simulation, the numpy filter of the ROS node and the constants they share."""

import numpy as np
import torch
from car_bridge.estimator import R_CAR_FROM_IMU, TAU_S, GravityEstimator

from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.estimation import (
    compensate_acceleration,
    gravity_step,
    imu_to_pitch,
    pitch_from_gravity,
)

G = 9.80665
DT = 0.02


def gravity_from_accel(accel: torch.Tensor) -> torch.Tensor:
    """Gravity direction from the accelerometer alone (valid at rest): the reference the filter is compared with."""
    return -accel / accel.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def test_ros_and_sim_share_the_constants():
    assert np.allclose(R_CAR_FROM_IMU, np.array(car_cfg.IMU_R_CAR_FROM_IMU), atol=1e-8)
    assert TAU_S == car_cfg.GRAVITY_FILTER_TAU_S


def test_mount_is_a_rotation():
    r = np.array(car_cfg.IMU_R_CAR_FROM_IMU)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(r) - 1.0) < 1e-6


def test_imu_quaternion_matches_the_rotation_matrix():
    """The simulated IMU sensor is posed with ``IMU_QUAT_XYZW`` (``rl_control/car_env_cfg.py``); the filter rotates with
    ``IMU_R_CAR_FROM_IMU`` (this file and ``estimator.py``). ``tools/build_balboa_urdf.py`` writes both from the same
    rotation, but nothing previously checked they still agree -- a regenerated ``imu_mount.json`` with a different
    quaternion convention, or one of the two edited by hand, would otherwise train and deploy on two different
    mountings without either test noticing."""
    from scipy.spatial.transform import Rotation

    matrix_from_quat = Rotation.from_quat(car_cfg.IMU_QUAT_XYZW).as_matrix()
    assert np.allclose(matrix_from_quat, np.array(car_cfg.IMU_R_CAR_FROM_IMU), atol=1e-6)


def test_imu_noise_is_the_datasheet_noise_density_times_sqrt_bandwidth():
    assert np.isclose(car_cfg.IMU_ACCEL_NOISE_STD, 90e-6 * G * np.sqrt(52.0))
    assert np.isclose(car_cfg.IMU_GYRO_NOISE_STD, np.radians(7e-3) * np.sqrt(52.0))


def test_torch_and_numpy_filters_agree():
    """Both filters, with the encoder-based acceleration compensation, on the same random sequence.

    The torch side calls ``imu_to_pitch`` -- the exact function ``ImuPitchAndRate`` (mdp/observations.py) calls -- not
    a hand-derived restatement of it, so a sign or transpose error in the shared gravity/pitch conversion would fail
    here instead of being silently duplicated into the test.
    """
    rng = np.random.default_rng(0)
    r = torch.tensor(R_CAR_FROM_IMU, dtype=torch.float64)
    estimator = GravityEstimator(dt=DT)
    alpha = TAU_S / (TAU_S + DT)
    alpha_a = car_cfg.ACCEL_COMP_TAU_S / (car_cfg.ACCEL_COMP_TAU_S + DT)
    g, speed_prev, a_lp = None, None, 0.0
    for step in range(200):
        gyro = rng.normal(0.0, 0.5, 3)
        accel = np.array([0.3, -0.2, 9.8]) + rng.normal(0.0, 0.5, 3)
        wheel = rng.normal(0.0, 3.0)
        pitch_np, rate_np = estimator.update(gyro, accel, wheel)
        # the same axle-speed low-pass in torch (this bit is simple enough that ImuPitchAndRate does it inline too)
        w = torch.tensor(gyro[None])
        speed = car_cfg.WHEEL_RADIUS_M * (wheel + (w @ r.T)[0, 1].item())
        raw = 0.0 if speed_prev is None else (speed - speed_prev) / DT
        a_lp, speed_prev = alpha_a * a_lp + (1.0 - alpha_a) * raw, speed
        a_imu = torch.tensor(accel[None])
        if g is None:
            g = gravity_from_accel(compensate_acceleration(a_imu, torch.tensor([a_lp]), r))
            rate_t = (w @ r.T)[0, 1].item()
            pitch_t = pitch_from_gravity(g @ r.T).item()
        else:
            g, pitch_t_, rate_t_ = imu_to_pitch(g, w, a_imu, torch.tensor([a_lp]), r, DT, alpha)
            pitch_t, rate_t = pitch_t_.item(), rate_t_.item()
        assert abs(pitch_np - pitch_t) < 1e-9, step
        assert abs(rate_np - rate_t) < 1e-9, step


def test_pitch_sign_and_value_at_rest():
    for pitch in (0.2, -0.2):
        f_car = torch.tensor([[-G * np.sin(pitch), 0.0, G * np.cos(pitch)]], dtype=torch.float64)
        f_imu = f_car @ torch.tensor(R_CAR_FROM_IMU, dtype=torch.float64)  # v_imu = R^T v_car (row vectors)
        g_imu = gravity_from_accel(f_imu)
        got = pitch_from_gravity(g_imu @ torch.tensor(R_CAR_FROM_IMU, dtype=torch.float64).T).item()
        assert abs(got - pitch) < 1e-9


def test_filter_without_a_sample_only_uses_the_gyro():
    g = torch.tensor([[0.0, 0.0, -1.0]])
    out = gravity_step(g, torch.zeros(1, 3), torch.zeros(1, 3), DT, 0.98)
    assert torch.allclose(out, g)


def test_acceleration_compensation_matches_between_torch_and_numpy():
    from car_bridge.estimator import ACCEL_COMP_TAU_S, WHEEL_RADIUS_M

    assert ACCEL_COMP_TAU_S == car_cfg.ACCEL_COMP_TAU_S and WHEEL_RADIUS_M == car_cfg.WHEEL_RADIUS_M
    r = np.array(car_cfg.IMU_R_CAR_FROM_IMU)
    accel, a = np.array([0.3, -0.2, 9.8]), 0.7
    got = compensate_acceleration(torch.tensor(accel)[None], torch.tensor([a]), torch.tensor(r))[0].numpy()
    assert np.allclose(got, accel - a * r[0])


def test_driving_is_not_taken_for_a_tilt():
    """Forward acceleration a at the true pitch 0.05: the estimate is biased without the encoders, not with them."""
    dt, g, pitch, a = 0.02, 9.80665, 0.05, 0.5
    f_car = np.array([a - g * np.sin(pitch), 0.0, g * np.cos(pitch)])  # specific force: acceleration minus gravity
    accel = R_CAR_FROM_IMU.T @ f_car
    plain, compensated = GravityEstimator(dt=dt), GravityEstimator(dt=dt)
    # the car speeds up at a: the wheels turn faster by a * dt / r every step (the pitch rate is zero)
    plain.update(np.zeros(3), R_CAR_FROM_IMU.T @ np.array([-g * np.sin(pitch), 0.0, g * np.cos(pitch)]))
    compensated.update(np.zeros(3), accel, 0.0)
    for k in range(1, 250):
        pitch_plain, _ = plain.update(np.zeros(3), accel)
        pitch_comp, _ = compensated.update(np.zeros(3), accel, a * k * dt / car_cfg.WHEEL_RADIUS_M)
    assert abs(pitch_comp - pitch) < 0.01
    assert abs(pitch_plain - pitch) > 0.03  # the uncorrected estimate sees the acceleration as a backward lean
