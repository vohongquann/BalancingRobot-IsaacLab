"""
Plain PID for one robot, the numpy copy of ``Balance_Car_RL/car/pid_control/pid.py``.

``tests/test_ros_gains.py`` checks that the two give the same output for the same errors.
"""


class PID:
    """out = kp * e + ki * integral(e dt) + kd * de/dt, integral and output limited (None: no limit)."""

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, out_limit=None, int_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit, self.int_limit = out_limit, int_limit
        self.integral = 0.0
        self.prev_error = None

    def reset(self) -> None:
        """Clear the integral and the derivative memory."""
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        """One step with the error ``setpoint - measurement`` and the time ``dt`` [s] since the last call."""
        if self.prev_error is None:
            self.prev_error = error  # first call: no derivative kick
        self.integral += error * dt
        if self.int_limit is not None:
            self.integral = max(-self.int_limit, min(self.int_limit, self.integral))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.out_limit is not None:
            out = max(-self.out_limit, min(self.out_limit, out))
        return out
