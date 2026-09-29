"""
ROS 2 node that runs the trained balance policy on live IMU and wheel-encoder topics.

Subscribes:  imu (sensor_msgs/Imu: raw angular_velocity and linear_acceleration in the IMU frame, no orientation
             needed), joint_states (sensor_msgs/JointState)
Publishes:   wheel_torque_cmd (std_msgs/Float64MultiArray, [left, right] in N m)

Pitch comes from the raw IMU through a complementary filter (estimator.py), as in the simulation. Start the node with
the robot resting: the filter starts from the first accelerometer sample.

The same node works against Isaac Sim (through the isaacsim.ros2.bridge extension) and against
the real car, as long as both publish these topics.
"""

import time

from car_bridge.estimator import GravityEstimator
from car_bridge.policy import BalancePolicy
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

LEFT_JOINT = 'left_wheel_joint'
RIGHT_JOINT = 'right_wheel_joint'

TRAINED_RATE_HZ = 50.0
"""The control rate the policy and the estimator's filter time constants were trained at (car_cfg.py, PHYSICS_HZ /
DECIMATION). ``rate_hz`` changes the estimator's dt but not what the policy learned; it must match."""

STALE_PERIODS = 3
"""Sensor messages older than this many control periods stop torque publishing (a disconnected topic must not drive
the motors from a frozen reading)."""


class BridgeNode(Node):

    def __init__(self):
        super().__init__('bridge_node')
        self.declare_parameter('policy_path', '')
        self.declare_parameter('rate_hz', 50.0)

        policy_path = self.get_parameter('policy_path').value
        if not policy_path:
            raise RuntimeError('parameter policy_path (exported ONNX policy) is required')
        self._policy = BalancePolicy(policy_path)

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

        self.create_subscription(Imu, 'imu', self._on_imu, 10)
        self.create_subscription(JointState, 'joint_states', self._on_joints, 10)
        self._pub = self.create_publisher(Float64MultiArray, 'wheel_torque_cmd', 10)
        self.create_timer(1.0 / rate_hz, self._on_timer)

    def _on_imu(self, msg: Imu) -> None:
        w, a = msg.angular_velocity, msg.linear_acceleration
        self._gyro = np.array([w.x, w.y, w.z])
        self._accel = np.array([a.x, a.y, a.z])
        self._imu_stamp = time.monotonic()

    def _on_joints(self, msg: JointState) -> None:
        vel = dict(zip(msg.name, msg.velocity))
        if LEFT_JOINT in vel and RIGHT_JOINT in vel:
            self._wheel_vel = (vel[LEFT_JOINT], vel[RIGHT_JOINT])
            self._wheel_stamp = time.monotonic()

    def _on_timer(self) -> None:
        if self._imu_stamp is None or self._wheel_stamp is None:
            return
        now = time.monotonic()
        if now - self._imu_stamp > self._stale_s or now - self._wheel_stamp > self._stale_s:
            self.get_logger().warn(
                'imu or joint_states has gone stale: not publishing a torque command',
                throttle_duration_sec=1.0,
            )
            return
        pitch, pitch_rate = self._estimator.update(self._gyro, self._accel, 0.5 * sum(self._wheel_vel))  # once per control step, like the simulation
        torque = self._policy.act(pitch, pitch_rate, self._wheel_vel)
        self._pub.publish(Float64MultiArray(data=[float(torque[0]), float(torque[1])]))


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
