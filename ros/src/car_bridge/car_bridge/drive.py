"""
The RL drive cascade on the robot: velocity network, position network (or go-to-goal PID with network gains) over it.

The numpy copy of ``BalanceCar-Velocity-v0``, ``BalanceCar-Position-v0`` and ``BalanceCar-Position-Gains-v0``
(``Balance_Car_RL/car/mdp/actions/{wheel_action,velocity_action,position_gain_action}.py``,
``pid_control/go_to_goal.py``); ``tests/test_ros_drive.py`` checks that the two give the same torques. Once per control
step (50 Hz), after the estimator::

    velocity      (v_x, w_z)           -> velocity network -> [common, turn] -> wheel torques
    position      goal (x, y, heading) -> position network -> (v_x, w_z) -> velocity
    position_gains goal                -> go-to-goal PID with the gains of its network -> (v_x, w_z) -> velocity
    lqr_velocity  (v_x, w_z)           -> LQR at the wheel speed v_x / r, PI on the gyro turn rate -> wheel torques
    lqr_position  goal                 -> go-to-goal PID (tuned gains) -> (v_x, w_z) -> lqr_velocity

The two ``lqr_`` controllers have no network: the numpy copy of ``pid_control/drive.py`` with ``LQRController``.

The goal is in the odometry frame (where the car started, x forward); :class:`Odometry` integrates the speed estimate
and the gyro's turn rate into the car's pose, which the simulation reads from the simulator instead.
"""

import math

from car_bridge.estimator import WHEEL_RADIUS_M
from car_bridge.gains import GainLayer
from car_bridge.pid import PID
from car_bridge.policy import OnnxActor, TORQUE_SCALE_NM
import numpy as np

DT_S = 0.02
"""Control period the networks were trained at (50 Hz)."""

SPEED_MAX_M_S = 0.4
"""Forward speed [m/s] of a velocity action of 1 (``velocity_env_cfg.SPEED_MAX_M_S``)."""

YAW_RATE_MAX_RAD_S = 2.0
"""Turn rate [rad/s] of a velocity action of 1 (``velocity_env_cfg.YAW_RATE_MAX_RAD_S``)."""

TURN_SHARE = 0.2
"""Share of the wheel torque range of the turning action (``velocity_env_cfg.TURN_SHARE``)."""

GOAL_OBS_MAX_M = 2.0
"""The goal is seen shortened to this distance [m] (``position_env_cfg.GOAL_OBS_MAX_M``)."""

GOAL_SPEED_KP, GOAL_TURN_KP, GOAL_INT_LIMIT, NEAR_M = 1.0, 2.0, 0.5, 0.1
"""Go-to-goal PID: speed and turn P gains, integral limit, near-goal distance (``pid_control/go_to_goal.py``)."""

LQR_GAIN = (-0.44600805367738416, -0.04572749998995848, -0.009428096500378624)
"""``K`` of the LQR ``u = -K [pitch, pitch rate, wheel speed error]``, u the total wheel torque [N m]: the default
``lqr_control.design_lqr()`` (tests/test_ros_drive.py checks it)."""

TURN_KP, TURN_KI = 0.3, 2.0
"""Turn-rate PI of the classical drive [turn action/(rad/s)], [turn action/rad] (``pid_control/drive.py``)."""

GOAL_LAYERS = (
    GainLayer(GOAL_SPEED_KP, 0.0, 0.0, int_limit=GOAL_INT_LIMIT),
    GainLayer(GOAL_TURN_KP, 0.0, 0.0, int_limit=GOAL_INT_LIMIT),
)
"""Speed and turn loop of the go-to-goal PID as gain layers (``position_gain_action.POSITION_GAIN_LAYERS``)."""


def mix_wheels(action) -> np.ndarray:
    """``[common, turn]`` in [-1, 1] -> ``[left, right]`` wheel actions in [-1, 1] (``wheel_action.mix_wheels``)."""
    common, turn = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return np.clip([common - TURN_SHARE * turn, common + TURN_SHARE * turn], -1.0, 1.0)


def shortened(goal) -> np.ndarray:
    """Goal ``[x, y, heading]`` with ``(x, y)`` shortened to ``GOAL_OBS_MAX_M`` (``navigation.pose_command_2d``)."""
    x, y, heading = goal
    scale = min(1.0, GOAL_OBS_MAX_M / max(math.hypot(x, y), 1e-6))
    return np.array([x * scale, y * scale, heading])


def goal_errors(goal) -> tuple[float, float]:
    """Errors of the go-to-goal PID (``go_to_goal.goal_errors``): distance ahead [m] and angle [rad]."""
    x, y, heading = goal
    bearing = math.atan2(y, x)
    if abs(bearing) > math.pi / 2:  # the goal seen from the back of the car: it backs up to it
        bearing = math.atan2(-y, -x)
    near = 1.0 - math.tanh(math.hypot(x, y) / NEAR_M)
    return x, (1.0 - near) * bearing + near * heading


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Odometry:
    """Pose ``(x, y, heading)`` of the car in the frame it started in, from the speed estimate and the gyro."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.heading = 0.0, 0.0, 0.0

    def update(self, speed: float, turn_rate: float, dt: float = DT_S) -> None:
        mid = self.heading + 0.5 * turn_rate * dt
        self.x += speed * math.cos(mid) * dt
        self.y += speed * math.sin(mid) * dt
        self.heading = wrap(self.heading + turn_rate * dt)

    def goal_in_car_frame(self, goal) -> np.ndarray:
        """Goal ``(x, y, heading)`` of the odometry frame -> ``[x, y, heading]`` in the car's frame."""
        dx, dy = goal[0] - self.x, goal[1] - self.y
        c, s = math.cos(self.heading), math.sin(self.heading)
        return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(goal[2] - self.heading)])


class VelocityController:
    """``(v_x, w_z)`` -> wheel torques [N m] through the velocity network. Observation, 8 values: ``[pitch, pitch rate,
    speed, turn rate, v_x, w_z, last action x2]`` (``VelocityPolicyCfg``)."""

    def __init__(self, actor):
        self._actor = actor
        self._last_action = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self._last_action[:] = 0.0

    def step(self, pitch, pitch_rate, speed, turn_rate, command) -> np.ndarray:
        v = float(np.clip(command[0], -SPEED_MAX_M_S, SPEED_MAX_M_S))
        w = float(np.clip(command[1], -YAW_RATE_MAX_RAD_S, YAW_RATE_MAX_RAD_S))
        obs = [pitch, pitch_rate, speed, turn_rate, v, w, *self._last_action]
        self._last_action = np.clip(self._actor(obs), -1.0, 1.0).astype(np.float32)  # clip_actions = 1 in training
        return mix_wheels(self._last_action) * TORQUE_SCALE_NM


class PositionController:
    """Goal in the car's frame -> ``(v_x, w_z)`` through the position network. Observation, 9 values: ``[pitch, pitch
    rate, speed, turn rate, goal x, y, heading, last action x2]`` (``PositionPolicyCfg``)."""

    def __init__(self, actor):
        self._actor = actor
        self._last_action = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self._last_action[:] = 0.0

    def step(self, pitch, pitch_rate, speed, turn_rate, goal) -> np.ndarray:
        obs = [pitch, pitch_rate, speed, turn_rate, *shortened(goal), *self._last_action]
        self._last_action = np.clip(self._actor(obs), -1.0, 1.0).astype(np.float32)
        return self._last_action * np.array([SPEED_MAX_M_S, YAW_RATE_MAX_RAD_S])


class PositionGainController:
    """Goal in the car's frame -> ``(v_x, w_z)``: the go-to-goal PID with the six gains of a network (``actor`` None:
    the tuned PID). Observation, 13 values: ``[pitch, pitch rate, speed, turn rate, goal x, y, heading, last 6
    actions]``."""

    def __init__(self, actor=None, dt: float = DT_S):
        self._actor, self._dt = actor, dt
        self._speed = PID(out_limit=SPEED_MAX_M_S, int_limit=GOAL_INT_LIMIT)
        self._turn = PID(out_limit=YAW_RATE_MAX_RAD_S, int_limit=GOAL_INT_LIMIT)
        self._last_action = np.zeros(6, dtype=np.float32)

    def reset(self) -> None:
        self._speed.reset()
        self._turn.reset()
        self._last_action[:] = 0.0

    def step(self, pitch, pitch_rate, speed, turn_rate, goal) -> np.ndarray:
        goal = shortened(goal)
        action = np.zeros(6, dtype=np.float32)
        if self._actor is not None:
            obs = [pitch, pitch_rate, speed, turn_rate, *goal, *self._last_action]
            action = np.clip(self._actor(obs), -1.0, 1.0).astype(np.float32)
        self._last_action = action
        self._speed.kp, self._speed.ki, self._speed.kd = GOAL_LAYERS[0].gains(action[:3])
        self._turn.kp, self._turn.ki, self._turn.kd = GOAL_LAYERS[1].gains(action[3:])
        ahead, angle = goal_errors(goal)
        return np.array([self._speed.update(ahead, self._dt), self._turn.update(angle, self._dt)])


class LqrVelocityController:
    """``(v_x, w_z)`` -> wheel torques [N m] without a network: LQR balance at the wheel speed ``v_x / r`` (driving
    upright at a constant speed is an equilibrium too) and a PI on the turn rate; ``DriveVelocity`` with
    ``LQRController`` in ``pid_control/drive.py``."""

    def __init__(self, gain=LQR_GAIN, dt: float = DT_S):
        self._gain, self._dt = np.asarray(gain, dtype=float), dt
        self._turn = PID(kp=TURN_KP, ki=TURN_KI, out_limit=1.0, int_limit=1.0 / TURN_KI)

    def reset(self) -> None:
        self._turn.reset()

    def step(self, pitch, pitch_rate, speed, turn_rate, command) -> np.ndarray:
        v = float(np.clip(command[0], -SPEED_MAX_M_S, SPEED_MAX_M_S))
        w = float(np.clip(command[1], -YAW_RATE_MAX_RAD_S, YAW_RATE_MAX_RAD_S))
        state = np.array([pitch, pitch_rate, (speed - v) / WHEEL_RADIUS_M])
        common = float(np.clip(-(self._gain @ state) / (2.0 * TORQUE_SCALE_NM), -1.0, 1.0))
        turn = self._turn.update(w - turn_rate, self._dt)
        return mix_wheels([common, turn]) * TORQUE_SCALE_NM


class DriveCascade:
    """The whole cascade of one controller: ``velocity``, ``position``, ``position_gains``, ``lqr_velocity`` or
    ``lqr_position``. ``act`` takes the estimator's pitch and pitch rate, the encoder speeds and the gyro's turn rate,
    and the command: ``(v_x, w_z)`` for ``velocity`` and ``lqr_velocity``, the goal ``(x, y, heading)`` in the odometry
    frame otherwise (None: hold the pose where it started)."""

    def __init__(self, controller: str, velocity_path: str = '', position_path: str = '', lqr_gain=LQR_GAIN):
        self.controller = controller
        if controller.startswith('lqr_'):
            self.velocity = LqrVelocityController(lqr_gain)
        else:
            self.velocity = VelocityController(OnnxActor(velocity_path))
        self.position = None
        if controller == 'position':
            self.position = PositionController(OnnxActor(position_path))
        elif controller in ('position_gains', 'lqr_position'):
            path = position_path if controller == 'position_gains' else ''
            self.position = PositionGainController(OnnxActor(path) if path else None)
        self.odometry = Odometry()
        self.command = np.zeros(2)  # the (v_x, w_z) of the last step

    def reset(self) -> None:
        self.velocity.reset()
        if self.position is not None:
            self.position.reset()

    def act(self, pitch, pitch_rate, wheel_vel, turn_rate, command) -> np.ndarray:
        speed = WHEEL_RADIUS_M * (0.5 * (wheel_vel[0] + wheel_vel[1]) + pitch_rate)
        self.odometry.update(speed, turn_rate)
        if self.position is None:
            self.command = np.asarray(command, dtype=float)
        else:
            goal = self.odometry.goal_in_car_frame(command if command is not None else (0.0, 0.0, 0.0))
            self.command = self.position.step(pitch, pitch_rate, speed, turn_rate, goal)
        return self.velocity.step(pitch, pitch_rate, speed, turn_rate, self.command)
