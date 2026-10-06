"""Gravity direction from a raw IMU (accelerometer + gyroscope) with a complementary filter.

``guide/02_sensing.md``. The same equations run in the simulation (this module, torch) and on the robot
(``ros/src/car_bridge/car_bridge/estimator.py``, numpy); ``tests/test_estimation.py`` checks that they agree.
"""

import torch


def gravity_step(g: torch.Tensor, gyro: torch.Tensor, accel: torch.Tensor, dt: float, alpha: float) -> torch.Tensor:
    """One filter step on the gravity direction in the IMU frame.

    Args:
        g: Current estimate, unit vectors, shape (N, 3). Gravity seen from the body: (0, 0, -1) when the IMU is flat.
        gyro: Angular velocity [rad/s] in the IMU frame, shape (N, 3).
        accel: Accelerometer output (specific force, any unit) in the IMU frame, shape (N, 3).
        dt: Time since the previous step [s].
        alpha: Weight of the gyro-propagated estimate, close to 1.

    Returns:
        The new unit vectors, shape (N, 3). Where the accelerometer reads zero (no sample yet) only the gyro is used.
    """
    g_pred = g - torch.cross(gyro, g, dim=-1) * dt  # gravity rotates opposite to the body: g_dot = -w x g
    norm = accel.norm(dim=-1, keepdim=True)
    g_meas = -accel / norm.clamp_min(1e-6)  # a resting accelerometer reads -gravity
    g_new = torch.where(norm > 1e-3, alpha * g_pred + (1.0 - alpha) * g_meas, g_pred)
    return g_new / g_new.norm(dim=-1, keepdim=True)


def compensate_acceleration(accel: torch.Tensor, a_axle: torch.Tensor, r_car_from_imu: torch.Tensor) -> torch.Tensor:
    """Remove the forward acceleration of the axle from the accelerometer output.

    A resting accelerometer reads minus gravity, but while the car accelerates it also reads that acceleration, which
    would be taken for a tilt (0.5 m/s^2 looks like 0.05 rad). The axle acceleration is known from the encoders, so
    it is subtracted: ``f_car = f_car - a_axle e_x``, in the IMU frame ``f_imu - a_axle R[0]``.

    Args:
        accel: Accelerometer output in the IMU frame, shape (N, 3).
        a_axle: Forward acceleration of the axle [same unit as accel], shape (N,).
        r_car_from_imu: Rotation with ``v_car = R v_imu``, shape (3, 3).
    """
    return accel - a_axle.unsqueeze(-1) * r_car_from_imu[0]


def pitch_from_gravity(g_car: torch.Tensor) -> torch.Tensor:
    """Forward lean [rad] from the gravity direction in the car frame (positive leans forward, +x)."""
    return torch.atan2(g_car[..., 0], -g_car[..., 2])


def imu_to_pitch(
    g: torch.Tensor,
    gyro_imu: torch.Tensor,
    accel_imu: torch.Tensor,
    a_axle: torch.Tensor,
    r_car_from_imu: torch.Tensor,
    dt: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One complementary-filter tick: compensate the axle acceleration, filter, then read pitch in the car frame.

    The single function both ``ImuPitchAndRate`` (``mdp/observations.py``, torch, the simulation) and
    ``tests/test_estimation.py`` call, so a test comparing it against the ROS ``GravityEstimator`` (numpy) exercises
    the exact production code, not a hand-copied restatement of it.

    Args:
        g: Current gravity estimate, unit vectors in the IMU frame, shape (N, 3).
        gyro_imu: Gyroscope reading, IMU frame, shape (N, 3).
        accel_imu: Accelerometer reading (not yet compensated), IMU frame, shape (N, 3).
        a_axle: Forward acceleration of the axle, shape (N,).
        r_car_from_imu: Rotation with ``v_car = R v_imu``, shape (3, 3).
        dt: Time since the previous tick [s].
        alpha: Weight of the gyro-propagated estimate, close to 1.

    Returns:
        ``(new_g, pitch, pitch_rate)``: the updated gravity estimate (IMU frame) and the pitch [rad] and pitch rate
        [rad/s] in the car frame.
    """
    accel = compensate_acceleration(accel_imu, a_axle, r_car_from_imu)
    g_new = gravity_step(g, gyro_imu, accel, dt, alpha)
    pitch = pitch_from_gravity(g_new @ r_car_from_imu.T)
    pitch_rate = (gyro_imu @ r_car_from_imu.T)[..., 1]
    return g_new, pitch, pitch_rate
