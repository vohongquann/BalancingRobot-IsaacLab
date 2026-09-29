import numpy as np

from car_bridge.estimator import GravityEstimator, R_CAR_FROM_IMU

G = 9.80665


def _accel_at_rest(pitch):
    """What a resting accelerometer reads in the IMU frame when the car leans forward by ``pitch``."""
    f_car = np.array([-G * np.sin(pitch), 0.0, G * np.cos(pitch)])  # specific force = -gravity
    return R_CAR_FROM_IMU.T @ f_car


def test_mount_is_a_rotation():
    assert np.allclose(R_CAR_FROM_IMU @ R_CAR_FROM_IMU.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R_CAR_FROM_IMU) - 1.0) < 1e-6


def test_resting_car_gives_its_pitch():
    for pitch in (0.0, 0.17, -0.25):
        estimator = GravityEstimator(dt=0.02)
        got, rate = estimator.update(np.zeros(3), _accel_at_rest(pitch))
        assert abs(got - pitch) < 1e-6 and abs(rate) < 1e-9


def test_forward_lean_is_positive():
    got, _ = GravityEstimator(dt=0.02).update(np.zeros(3), _accel_at_rest(0.2))
    assert got > 0.19


def test_gyro_carries_the_estimate_through_a_fast_tilt():
    """The accelerometer is held at its old value: a rotation of 0.2 rad in 0.2 s must show up through the gyro."""
    estimator = GravityEstimator(dt=0.02)
    estimator.update(np.zeros(3), _accel_at_rest(0.0))
    gyro = R_CAR_FROM_IMU.T @ np.array([0.0, 1.0, 0.0])  # 1 rad/s about the car y axis
    for _ in range(10):
        pitch, rate = estimator.update(gyro, _accel_at_rest(0.0))
    assert 0.17 < pitch < 0.2 and abs(rate - 1.0) < 1e-9


def test_accelerometer_removes_gyro_drift():
    estimator = GravityEstimator(dt=0.02)
    estimator.update(np.zeros(3), _accel_at_rest(0.1))
    bias = R_CAR_FROM_IMU.T @ np.array([0.0, 0.01, 0.0])  # 0.01 rad/s bias
    for _ in range(1000):  # 20 s
        pitch, _ = estimator.update(bias, _accel_at_rest(0.1))
    assert abs(pitch - 0.1) < 0.02  # settles at about bias * tau = 0.01 rad off, does not grow like 0.2 rad


def test_no_sample_no_estimate():
    estimator = GravityEstimator(dt=0.02)
    assert estimator.update(np.zeros(3), np.zeros(3))[0] == 0.0
