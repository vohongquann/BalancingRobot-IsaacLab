"""
Pitch and pitch rate from a raw IMU (accelerometer + gyroscope) with a complementary filter.

The same equations as ``Balance_Car_RL/car/estimation.py`` (torch, used in the simulation);
``tests/test_estimation.py`` checks that the two agree. ``docs/02-sensing.md``.
"""

import numpy as np

# IMU frame (control board axes) in the car frame, v_car = R @ v_imu.
# Copy of ``IMU_R_CAR_FROM_IMU`` in ``Balance_Car_RL/car/car_cfg.py`` (a test checks it).
R_CAR_FROM_IMU = np.array(
    [
        [0.0, 0.172939702, -0.984932414],
        [-1.0, 0.0, 0.0],
        [0.0, 0.984932414, 0.172939702],
    ]
)

TAU_S = 1.0
"""Filter time constant [s], equal to ``GRAVITY_FILTER_TAU_S`` in ``car_cfg.py``."""

ACCEL_COMP_TAU_S = 0.05
"""Time constant [s] of the low-pass on the axle acceleration, equal to ``ACCEL_COMP_TAU_S`` in ``car_cfg.py``."""

WHEEL_RADIUS_M = 0.040
"""Equal to ``WHEEL_RADIUS_M`` in ``car_cfg.py``."""


class GravityEstimator:
    """Gravity direction in the IMU frame, and from it the pitch and the pitch rate of the car."""

    def __init__(self, dt: float, rotation=R_CAR_FROM_IMU, tau: float = TAU_S):
        self._alpha = tau / (tau + dt)
        self._alpha_a = ACCEL_COMP_TAU_S / (ACCEL_COMP_TAU_S + dt)
        self._dt = dt
        self._r = np.asarray(rotation, dtype=float)
        self._g = None
        self._speed_prev = None
        self._a_lp = 0.0

    def reset(self) -> None:
        """Forget the estimate; the next sample starts it again from the accelerometer (the robot must be resting)."""
        self._g = None
        self._speed_prev = None
        self._a_lp = 0.0

    def update(self, gyro, accel, wheel_speed=0.0):
        """One step; returns ``(pitch, pitch_rate)``.

        ``gyro`` [rad/s] and ``accel`` [m/s^2] are 3-vectors in the IMU frame; ``wheel_speed`` [rad/s] is the mean
        wheel joint speed from the encoders (relative to the body). The axle acceleration derived from it is removed from
        the accelerometer, so that driving is not taken for a tilt (``docs/02-sensing.md``).
        """
        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)
        speed = WHEEL_RADIUS_M * (wheel_speed + (self._r @ gyro)[1])
        raw = 0.0 if self._speed_prev is None else (speed - self._speed_prev) / self._dt
        self._a_lp = self._alpha_a * self._a_lp + (1.0 - self._alpha_a) * raw
        self._speed_prev = speed
        accel = accel - self._a_lp * self._r[0]
        norm = np.linalg.norm(accel)
        if self._g is None:
            if norm < 1e-3:
                return 0.0, float((self._r @ gyro)[1])
            self._g = -accel / norm
        else:
            g_pred = self._g - np.cross(gyro, self._g) * self._dt  # g_dot = -w x g
            if norm > 1e-3:
                g_pred = self._alpha * g_pred + (1.0 - self._alpha) * (-accel / norm)
            self._g = g_pred / np.linalg.norm(g_pred)
        g_car = self._r @ self._g
        return float(np.arctan2(g_car[0], -g_car[2])), float((self._r @ gyro)[1])
