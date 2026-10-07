"""
From the output of a gain network to PID gains, the numpy copy of ``Balance_Car_RL/car/mdp/gains.py``.

Action ``a`` in [-1, 1], one value per gain, order [kp, ki, kd]. The tuned PID is the zero action::

    gain of the tuned PID not 0:   gain = nominal * 3^a
    gain of the tuned PID is 0:    gain = maximum * max(a, 0)

``tests/test_ros_gains.py`` checks every constant against the simulation. ``guide/07_gains.md``.
"""

import numpy as np

FACTOR = 3.0
KI_PER_KP = 0.3
KD_PER_KP = 0.05


class GainLayer:
    """The tuned gains of one layer and the map from a network output to gains."""

    def __init__(self, kp: float, ki: float, kd: float, int_limit: float):
        self.nominal = np.array([kp, ki, kd])
        self.maximum = np.array([0.0, KI_PER_KP * kp, KD_PER_KP * kp])
        self.int_limit = int_limit

    def gains(self, action) -> np.ndarray:
        """Network output (3,) -> [kp, ki, kd]."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return np.where(self.nominal > 0.0, self.nominal * FACTOR**a, self.maximum * np.maximum(a, 0.0))


PITCH = GainLayer(kp=1.6, ki=0.05, kd=0.05, int_limit=0.2)
"""Pitch loop: kp [N m/rad], ki [N m/(rad s)], kd on the pitch rate [N m s/rad]."""

SPEED = GainLayer(kp=0.015, ki=0.0, kd=0.0, int_limit=5.0)
"""Speed loop on the wheel speed error [rad/s]: kp [rad/(rad/s)]; ki and kd are off."""
