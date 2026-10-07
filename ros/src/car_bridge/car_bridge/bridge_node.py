"""
ROS 2 node that runs the trained balance policy on live IMU and wheel-encoder topics.

Subscribes:  imu (sensor_msgs/Imu: raw angular_velocity and linear_acceleration in the IMU frame, no orientation
             needed), joint_states (sensor_msgs/JointState); and the command of the controller: cmd_pitch / cmd_speed
             (std_msgs/Float64, ``gains``), cmd_vel (geometry_msgs/Twist: linear.x [m/s], angular.z [rad/s],
             ``velocity``), goal_pose (geometry_msgs/PoseStamped in the odom frame, ``position``, ``position_gains``)
Publishes:   wheel_torque_cmd (std_msgs/Float64MultiArray, [left, right] in N m); with a drive controller also odom
             (nav_msgs/Odometry: the pose from the encoders and the gyro, frame where the node started)

Controllers (parameter ``controller``):

* ``upright`` (default): the ONNX policy of the upright task (``policy_path``) writes the torques.
* ``gains``: networks write PID gains and a PID computes the torques (``gain_cascade.py``, ``guide/07_gains.md``).
  ``pitch_gains_path`` is the pitch layer (empty: the tuned PID), ``speed_gains_path`` the speed layer on top of it
  (empty: no speed layer, the pitch is commanded on ``cmd_pitch``; ``speed_layer:=true`` without a path runs the tuned
  speed PID).
* ``velocity``: the velocity network (``velocity_path``) follows cmd_vel (``drive.py``).
* ``position``: the position network (``position_path``) over the velocity network drives to goal_pose.
* ``position_gains``: the go-to-goal PID with the gains of a network (``position_path``; empty: the tuned PID) over the
  velocity network drives to goal_pose.
* ``lqr_velocity``, ``lqr_position``: the same without networks: LQR balance and a PI on the turn rate follow cmd_vel,
  the tuned go-to-goal PID over them drives to goal_pose (``lqr_gain``: the 3 LQR gains, default the designed ones).

Speed and turn commands go back to zero when none arrives for ``COMMAND_TIMEOUT_S``; a goal is held until the next one
(none yet: stay where the node started).

Pitch comes from the raw IMU through a complementary filter (estimator.py), as in the simulation. Start the node with
the robot resting: the filter starts from the first accelerometer sample.

``step_on``: ``timer`` (default) runs the controller at ``rate_hz`` on the latest samples, as on the robot; ``sensors``
runs it once per pair of imu and joint_states messages with the same stamp, for a simulator that waits for the torque
before it steps (``tools/ros_sim_demo.py``). The same node runs the real car and the simulated one.
"""

import math
import time

from car_bridge.drive import DriveCascade, LQR_GAIN
from car_bridge.estimator import GravityEstimator
from car_bridge.gain_cascade import GainCascade
from car_bridge.policy import BalancePolicy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64, Float64MultiArray

DRIVE_CONTROLLERS = ('velocity', 'position', 'position_gains', 'lqr_velocity', 'lqr_position')
VELOCITY_CONTROLLERS = ('velocity', 'lqr_velocity')  # follow cmd_vel; the others drive to goal_pose

LEFT_JOINT = 'left_wheel_joint'
RIGHT_JOINT = 'right_wheel_joint'

TRAINED_RATE_HZ = 50.0
"""The control rate the policy and the estimator's filter time constants were trained at (car_cfg.py, PHYSICS_HZ /
DECIMATION). ``rate_hz`` changes the estimator's dt but not what the policy learned; it must match."""

COMMAND_TIMEOUT_S = 1.0
"""A pitch or speed command older than this is taken as zero (a lost teleop must stop the car, not keep it moving)."""

STALE_PERIODS = 3
"""Sensor messages older than this many control periods stop torque publishing (a disconnected topic must not drive
the motors from a frozen reading)."""


class BridgeNode(Node):
    def __init__(self):
        super().__init__('bridge_node')
        self.declare_parameter('controller', 'upright')
        self.declare_parameter('policy_path', '')
        self.declare_parameter('pitch_gains_path', '')
        self.declare_parameter('speed_gains_path', '')
        self.declare_parameter('speed_layer', False)
        self.declare_parameter('velocity_path', '')
        self.declare_parameter('position_path', '')
        self.declare_parameter('lqr_gain', list(LQR_GAIN))
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('step_on', 'timer')

        self._controller = self.get_parameter('controller').value
        self._policy = None
        self._cascade = None
        self._drive = None
        if self._controller == 'upright':
            policy_path = self.get_parameter('policy_path').value
            if not policy_path:
                raise RuntimeError('parameter policy_path (exported ONNX policy) is required')
            self._policy = BalancePolicy(policy_path)
        elif self._controller == 'gains':
            self._cascade = GainCascade(
                self.get_parameter('pitch_gains_path').value,
                self.get_parameter('speed_gains_path').value,
                self.get_parameter('speed_layer').value,
            )
        elif self._controller in DRIVE_CONTROLLERS:
            velocity_path = self.get_parameter('velocity_path').value
            position_path = self.get_parameter('position_path').value
            lqr = self._controller.startswith('lqr_')
            if not lqr and (not velocity_path or (self._controller == 'position' and not position_path)):
                raise RuntimeError('parameter velocity_path (and position_path for position) is required')
            lqr_gain = self.get_parameter('lqr_gain').value
            if len(lqr_gain) != 3:
                raise RuntimeError(f'parameter lqr_gain must have 3 values, not {len(lqr_gain)}')
            self._drive = DriveCascade(self._controller, velocity_path, position_path, lqr_gain)
        else:
            raise RuntimeError(
                f'parameter controller must be one of upright, gains, {", ".join(DRIVE_CONTROLLERS)}, '
                f'not {self._controller!r}'
            )

        rate_hz = self.get_parameter('rate_hz').value
        if abs(rate_hz - TRAINED_RATE_HZ) > 1e-6:
            self.get_logger().warn(
                f'rate_hz={rate_hz} does not match the {TRAINED_RATE_HZ} Hz the policy and the filter time constants '
                'were trained at: the estimator and the policy will see a different world than in simulation.'
            )
        self._stale_s = STALE_PERIODS / rate_hz
        self._estimator = GravityEstimator(dt=1.0 / rate_hz)
        self._gyro = np.zeros(3)
        self._accel = np.zeros(3)
        self._wheel_vel = (0.0, 0.0)
        self._imu_stamp = None
        self._wheel_stamp = None
        self._pitch_cmd = 0.0
        self._speed_cmd = 0.0
        self._pitch_cmd_stamp = None
        self._speed_cmd_stamp = None
        self._vel_cmd = (0.0, 0.0)
        self._vel_cmd_stamp = None
        self._goal = None
        self._imu_msg_stamp = None
        self._wheel_msg_stamp = None
        self._stepped_stamp = None

        self.create_subscription(Imu, 'imu', self._on_imu, 10)
        self.create_subscription(JointState, 'joint_states', self._on_joints, 10)
        if self._cascade is not None:
            self.create_subscription(Float64, 'cmd_pitch', self._on_cmd_pitch, 10)
            self.create_subscription(Float64, 'cmd_speed', self._on_cmd_speed, 10)
        if self._controller in VELOCITY_CONTROLLERS:
            self.create_subscription(Twist, 'cmd_vel', self._on_cmd_vel, 10)
        elif self._drive is not None:
            self.create_subscription(PoseStamped, 'goal_pose', self._on_goal, 10)
        self._pub = self.create_publisher(Float64MultiArray, 'wheel_torque_cmd', 10)
        self._odom_pub = self.create_publisher(Odometry, 'odom', 10) if self._drive is not None else None
        self._step_on = self.get_parameter('step_on').value
        if self._step_on == 'timer':
            self.create_timer(1.0 / rate_hz, self._on_timer)
        elif self._step_on != 'sensors':
            raise RuntimeError(f"parameter step_on must be 'timer' or 'sensors', not {self._step_on!r}")

    def _on_imu(self, msg: Imu) -> None:
        w, a = msg.angular_velocity, msg.linear_acceleration
        self._gyro = np.array([w.x, w.y, w.z])
        self._accel = np.array([a.x, a.y, a.z])
        self._imu_stamp = time.monotonic()
        self._imu_msg_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._step_on_sensors()

    def _on_joints(self, msg: JointState) -> None:
        vel = dict(zip(msg.name, msg.velocity))
        if LEFT_JOINT in vel and RIGHT_JOINT in vel:
            self._wheel_vel = (vel[LEFT_JOINT], vel[RIGHT_JOINT])
            self._wheel_stamp = time.monotonic()
            self._wheel_msg_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            self._step_on_sensors()

    def _step_on_sensors(self) -> None:
        """``step_on:=sensors``: one control step per pair of samples with the same stamp."""
        stamp = self._imu_msg_stamp
        if self._step_on == 'sensors' and stamp == self._wheel_msg_stamp and stamp != self._stepped_stamp:
            self._stepped_stamp = stamp
            self._control(time.monotonic())

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._vel_cmd = (msg.linear.x, msg.angular.z)
        self._vel_cmd_stamp = time.monotonic()

    def _on_goal(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        heading = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._goal = (msg.pose.position.x, msg.pose.position.y, heading)

    def _on_cmd_pitch(self, msg: Float64) -> None:
        self._pitch_cmd = msg.data
        self._pitch_cmd_stamp = time.monotonic()

    def _on_cmd_speed(self, msg: Float64) -> None:
        self._speed_cmd = msg.data
        self._speed_cmd_stamp = time.monotonic()

    def _command(self, value: float, stamp, now: float) -> float:
        return value if stamp is not None and now - stamp <= COMMAND_TIMEOUT_S else 0.0

    def _on_timer(self) -> None:
        if self._imu_stamp is None or self._wheel_stamp is None:
            return
        now = time.monotonic()
        if now - self._imu_stamp > self._stale_s or now - self._wheel_stamp > self._stale_s:
            self.get_logger().warn(
                'imu or joint_states has gone stale: not publishing a torque command',
                throttle_duration_sec=1.0,
            )
            for controller in (self._cascade, self._drive):
                if controller is not None:
                    controller.reset()  # no integral may wind up while the sensors are away
            return
        self._control(now)

    def _control(self, now: float) -> None:
        """One control step, like the simulation: estimator, then the controller."""
        pitch, pitch_rate = self._estimator.update(self._gyro, self._accel, 0.5 * sum(self._wheel_vel))
        if self._policy is not None:
            torque = self._policy.act(pitch, pitch_rate, self._wheel_vel)
        elif self._cascade is not None:
            torque = self._cascade.act(
                pitch,
                pitch_rate,
                self._wheel_vel,
                self._command(self._pitch_cmd, self._pitch_cmd_stamp, now),
                self._command(self._speed_cmd, self._speed_cmd_stamp, now),
            )
        else:
            turn_rate = float((self._estimator.rotation @ self._gyro)[2])  # the gyro's z in the car frame
            fresh = self._vel_cmd_stamp is not None and now - self._vel_cmd_stamp <= COMMAND_TIMEOUT_S
            command = self._goal
            if self._controller in VELOCITY_CONTROLLERS:
                command = self._vel_cmd if fresh else (0.0, 0.0)
            torque = self._drive.act(pitch, pitch_rate, self._wheel_vel, turn_rate, command)
            self._publish_odom()
        self._pub.publish(Float64MultiArray(data=[float(torque[0]), float(torque[1])]))

    def _publish_odom(self) -> None:
        odom, pose = Odometry(), self._drive.odometry
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id, odom.child_frame_id = 'odom', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = pose.x, pose.y
        odom.pose.pose.orientation.z = math.sin(0.5 * pose.heading)
        odom.pose.pose.orientation.w = math.cos(0.5 * pose.heading)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = (float(v) for v in self._drive.command)
        self._odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
