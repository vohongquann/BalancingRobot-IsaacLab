"""
The gain networks on the robot: a network writes PID gains, the PID computes the command.

The numpy copy of ``BalanceCar-Pitch-Gains-v0`` and ``BalanceCar-Velocity-Gains-v0`` (``Balance_Car_RL/car/mdp/
actions/gain_action.py``); ``tests/test_ros_gains.py`` checks that the two give the same torques. Once per control
step (50 Hz), after the estimator::

    speed layer (optional)   speed target -> speed PID with the gains of its network -> pitch target
    pitch layer              pitch target -> pitch PID with the gains of its network -> wheel torques

A layer without a network runs the tuned PID (the zero action of the network). ``guide/07_gains.md``.
"""

from car_bridge import gains
from car_bridge.estimator import WHEEL_RADIUS_M
from car_bridge.pid import PID
from car_bridge.policy import OnnxActor, TORQUE_SCALE_NM
import numpy as np

DT_S = 0.02
"""Control period the networks were trained at (50 Hz); the PID gains only mean something at this period."""

PITCH_TARGET_MAX_RAD = 0.08
"""Largest pitch target [rad], the range the pitch layer was trained on (``pitch_env_cfg.PITCH_TARGET_MAX_RAD``)."""

SPEED_TARGET_MAX_M_S = 0.4
"""Largest speed target [m/s] it was trained on (``velocity_env_cfg.SPEED_MAX_M_S``)."""

SPEED_GUARD_M_S = 0.5
"""Speed above which a pitch target that speeds the car up further is zeroed (``pitch_env_cfg.SPEED_GUARD_M_S``)."""

NO_ACTION = np.zeros(3, dtype=np.float32)


def apply_speed_guard(target: float, speed: float, guard: float = SPEED_GUARD_M_S) -> float:
    """Zero a pitch target that would speed up a car already faster than ``guard`` [m/s]."""
    if abs(speed) > guard and np.sign(target) == np.sign(speed):
        return 0.0
    return target


class PitchGainController:
    """Pitch target -> wheel torques [N m] with the gains of a network (``actor`` None: the tuned PID).

    Observation of the network, 8 values: ``[pitch, pitch rate, left wheel speed, right wheel speed, last action x3,
    pitch target]``, the order of ``PitchPolicyCfg`` in ``rl_control/pitch_env_cfg.py``.
    """

    def __init__(self, actor=None, dt: float = DT_S):
        self._actor = actor
        self._dt = dt
        self._pid = PID(int_limit=gains.PITCH.int_limit)
        self._last_action = NO_ACTION.copy()
        self.gains = gains.PITCH.nominal.copy()  # kp, ki, kd of the last step

    def reset(self) -> None:
        self._pid.reset()
        self._last_action = NO_ACTION.copy()

    def step(self, pitch: float, pitch_rate: float, wheel_vel, target: float) -> np.ndarray:
        """One control step; returns the torque [left, right] in N m."""
        action = NO_ACTION.copy()
        if self._actor is not None:
            obs = [pitch, pitch_rate, wheel_vel[0], wheel_vel[1], *self._last_action, target]
            action = np.clip(self._actor(obs), -1.0, 1.0)
        self._last_action = action
        self.gains = gains.PITCH.gains(action)
        self._pid.kp, self._pid.ki, kd = self.gains
        total = self._pid.update(pitch - target, self._dt) + kd * pitch_rate
        wheel = float(np.clip(total / (2.0 * TORQUE_SCALE_NM), -1.0, 1.0)) * TORQUE_SCALE_NM
        return np.array([wheel, wheel])


class SpeedGainController:
    """Speed target -> pitch target [rad] with the gains of a network (``actor`` None: the tuned PID).

    Observation of the network, 7 values: ``[pitch, pitch rate, speed, speed target, last action x3]``, the order of
    ``VelocityPolicyCfg`` in ``rl_control/velocity_env_cfg.py``. The speed is the estimate of the robot, from the
    encoders and the pitch rate.
    """

    def __init__(self, actor=None, dt: float = DT_S):
        self._actor = actor
        self._dt = dt
        self._pid = PID(out_limit=PITCH_TARGET_MAX_RAD, int_limit=gains.SPEED.int_limit)
        self._last_action = NO_ACTION.copy()
        self.gains = gains.SPEED.nominal.copy()

    def reset(self) -> None:
        self._pid.reset()
        self._last_action = NO_ACTION.copy()

    def step(self, pitch: float, pitch_rate: float, speed: float, speed_target: float) -> float:
        """One control step; returns the pitch target [rad], zeroed by the speed guard."""
        action = NO_ACTION.copy()
        if self._actor is not None:
            obs = [pitch, pitch_rate, speed, speed_target, *self._last_action]
            action = np.clip(self._actor(obs), -1.0, 1.0)
        self._last_action = action
        self.gains = gains.SPEED.gains(action)
        self._pid.kp, self._pid.ki, self._pid.kd = self.gains
        target = self._pid.update((speed_target - speed) / WHEEL_RADIUS_M, self._dt)  # the loop works on rad/s
        return apply_speed_guard(target, speed)


class GainCascade:
    """The two layers on top of each other. ``speed_layer`` False: the pitch target is the command itself."""

    def __init__(
        self, pitch_gains_path: str = '', speed_gains_path: str = '', speed_layer: bool = False, dt: float = DT_S
    ):
        self.pitch = PitchGainController(OnnxActor(pitch_gains_path) if pitch_gains_path else None, dt)
        self.speed = None
        if speed_layer or speed_gains_path:
            self.speed = SpeedGainController(OnnxActor(speed_gains_path) if speed_gains_path else None, dt)

    def reset(self) -> None:
        self.pitch.reset()
        if self.speed is not None:
            self.speed.reset()

    def act(
        self, pitch: float, pitch_rate: float, wheel_vel, pitch_cmd: float = 0.0, speed_cmd: float = 0.0
    ) -> np.ndarray:
        """Torque [left, right] in N m for one control step.

        ``wheel_vel`` are the encoder speeds [rad/s], relative to the body. The commands are limited to the range the
        networks were trained on: ``pitch_cmd`` [rad] is used without a speed layer, ``speed_cmd`` [m/s] with one.
        """
        if self.speed is None:
            target = float(np.clip(pitch_cmd, -PITCH_TARGET_MAX_RAD, PITCH_TARGET_MAX_RAD))
        else:
            speed = WHEEL_RADIUS_M * (0.5 * (wheel_vel[0] + wheel_vel[1]) + pitch_rate)
            speed_target = float(np.clip(speed_cmd, -SPEED_TARGET_MAX_M_S, SPEED_TARGET_MAX_M_S))
            target = self.speed.step(pitch, pitch_rate, speed, speed_target)
        return self.pitch.step(pitch, pitch_rate, wheel_vel, target)
