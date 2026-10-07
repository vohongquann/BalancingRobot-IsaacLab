"""The simulated car as a ROS 2 robot: the controller is the ROS node of ``ros/src/car_bridge`` (``bridge_node``), in
its own process, exactly as on the real car. Used by ``record_demo.py --ros``.

Every policy step the simulation publishes what the robot's drivers publish (``joint_states``: wheel speeds; ``imu``:
raw gyro and accelerometer in the IMU frame, with the noise and offset of the LSM6DS33 numbers of ``car_cfg.py``), and
the command of the controller (``cmd_vel``, ``goal_pose`` in the odom frame = where the car started, or ``cmd_speed``);
it then waits for the node's ``wheel_torque_cmd`` [N m] and applies it. The node runs with ``step_on:=sensors``: one
control step per pair of sensor messages, so the two stay in lock step however slowly the simulation renders.

Needs ROS 2 in the environment (``source /opt/ros/jazzy/setup.bash``) and onnxruntime in this Python.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.envs import mdp as isaac_mdp

from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.mdp.actions.frozen_policy import FROZEN_DIR

REPO = Path(__file__).resolve().parents[4]
ROS_PACKAGE = REPO / "ros" / "src" / "car_bridge"
JOINTS = ("left_wheel_joint", "right_wheel_joint")
WAIT_S = 30.0
"""Longest wait [s] for the node: its first answer comes after it has started and found the topics."""


def wheel_torque_actions_cfg():
    """Action of the simulated car driven over ROS: Isaac Lab's ``JointEffortAction``, one torque per wheel (left,
    right), an action of 1 being the stall torque, as the node's torque [N m] / stall torque."""
    return isaac_mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=list(JOINTS), preserve_order=True, scale=car_cfg.WHEEL_STALL_TORQUE_NM
    )


def node_parameters(cascade: str) -> dict:
    """``bridge_node`` parameters that run ``cascade`` (``record_demo.py --cascade``) on its frozen ONNX networks."""
    onnx = {stage: str(FROZEN_DIR / stage / "policy.onnx") for stage in ("velocity", "position", "position_gains")}
    if cascade == "gains":
        return {
            "controller": "gains",
            "pitch_gains_path": str(FROZEN_DIR / "pitch_gains" / "policy.onnx"),
            "speed_gains_path": str(FROZEN_DIR / "velocity_gains" / "policy.onnx"),
        }
    if cascade in ("lqr", "lqr_position"):  # no network: the node's numpy LQR drive
        return {"controller": cascade if cascade == "lqr_position" else "lqr_velocity"}
    controller = {"rl": "velocity", "position": "position", "position_gains": "position_gains"}[cascade]
    params = {"controller": controller, "velocity_path": onnx["velocity"]}
    if controller != "velocity":
        params["position_path"] = onnx[controller]
    return params


class RosLink:
    """``step(command) -> action``: publishes the sensors and the command, returns the node's torques as the action of
    :func:`wheel_torque_actions_cfg`. ``command``: ``(v_x, w_z)`` (rl, lqr), ``(x, y, heading)`` relative to the
    start (position), or a speed [m/s] (gains)."""

    def __init__(self, env, cascade: str):
        import rclpy
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, JointState
        from std_msgs.msg import Float64, Float64MultiArray

        self._msgs = {"Imu": Imu, "JointState": JointState, "Twist": Twist, "PoseStamped": PoseStamped}
        self._msgs["Float64"] = Float64
        self.env, self.cascade = env, cascade
        self._robot, self._imu = env.scene["robot"], env.scene["imu"]
        self._joint_ids = self._robot.find_joints(list(JOINTS), preserve_order=True, as_proxy=True)[0].torch
        self._r_car_from_imu = torch.tensor(car_cfg.IMU_R_CAR_FROM_IMU, device=env.device)
        rand = lambda limit: (2.0 * torch.rand(3, device=env.device) - 1.0) * limit  # noqa: E731
        self._accel_bias, self._gyro_bias = rand(car_cfg.IMU_ACCEL_BIAS_MAX), rand(car_cfg.IMU_GYRO_BIAS_MAX)

        rclpy.init()
        self._rclpy = rclpy
        self.node = rclpy.create_node("sim_car")
        self._pub_imu = self.node.create_publisher(Imu, "imu", 10)
        self._pub_joints = self.node.create_publisher(JointState, "joint_states", 10)
        topic, kind = {"rl": ("cmd_vel", Twist), "lqr": ("cmd_vel", Twist), "gains": ("cmd_speed", Float64)}.get(
            cascade, ("goal_pose", PoseStamped)
        )
        self._pub_command = self.node.create_publisher(kind, topic, 10)
        self.torque, self.command = None, (0.0, 0.0)
        self.node.create_subscription(Float64MultiArray, "wheel_torque_cmd", self._on_torque, 10)
        self.node.create_subscription(Odometry, "odom", self._on_odom, 10)

        params = {**node_parameters(cascade), "step_on": "sensors"}
        args = [f"{k}:={v}" for k, v in params.items()]
        env_vars = {**os.environ, "PYTHONPATH": f"{ROS_PACKAGE}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        cmd = [sys.executable, "-m", "car_bridge.bridge_node", "--ros-args"]
        for arg in args:
            cmd += ["-p", arg]
        print("ROS node:", " ".join(cmd), flush=True)
        self._process = subprocess.Popen(cmd, env=env_vars)
        deadline = time.time() + WAIT_S
        while self.node.count_subscribers("imu") == 0 or self.node.count_publishers("wheel_torque_cmd") == 0:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() > deadline or self._process.poll() is not None:
                raise RuntimeError("the ROS node did not come up (see its output above)")
        time.sleep(0.5)  # the node's own subscriptions finish matching
        self._step = 0

    def _on_torque(self, msg) -> None:
        self.torque = np.array(msg.data, dtype=np.float32)

    def _on_odom(self, msg) -> None:
        self.command = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _stamp(self, header) -> None:
        nanoseconds = int(round(self._step * self.env.step_dt * 1e9))
        header.stamp.sec, header.stamp.nanosec = nanoseconds // 10**9, nanoseconds % 10**9

    def _publish_command(self, command) -> None:
        if self.cascade in ("rl", "lqr"):
            msg = self._msgs["Twist"]()
            msg.linear.x, msg.angular.z = float(command[0]), float(command[1])
        elif self.cascade == "gains":
            msg = self._msgs["Float64"](data=float(command))
        else:
            msg = self._msgs["PoseStamped"]()
            msg.header.frame_id = "odom"
            msg.pose.position.x, msg.pose.position.y = float(command[0]), float(command[1])
            msg.pose.orientation.z, msg.pose.orientation.w = math.sin(command[2] / 2), math.cos(command[2] / 2)
        self._pub_command.publish(msg)

    def _publish_sensors(self) -> None:
        n = self._imu.data.lin_acc_b.torch[0].shape
        accel = (
            self._imu.data.lin_acc_b.torch[0]
            + self._accel_bias
            + torch.randn(n, device=self.env.device) * (car_cfg.IMU_ACCEL_NOISE_STD)
        )
        gyro = (
            self._imu.data.ang_vel_b.torch[0]
            + self._gyro_bias
            + torch.randn(n, device=self.env.device) * (car_cfg.IMU_GYRO_NOISE_STD)
        )
        joints = self._msgs["JointState"]()
        self._stamp(joints.header)
        joints.name = list(JOINTS)
        joints.velocity = [float(v) for v in self._robot.data.joint_vel.torch[0, self._joint_ids]]
        imu = self._msgs["Imu"]()
        self._stamp(imu.header)
        imu.header.frame_id = "imu_link"
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(v) for v in gyro)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (float(v) for v in accel)
        self._pub_joints.publish(joints)
        self._pub_imu.publish(imu)

    def step(self, command) -> torch.Tensor:
        self._publish_command(command)
        self.torque = None
        self._publish_sensors()
        deadline = time.time() + (WAIT_S if self._step == 0 else 5.0)
        while self.torque is None:
            self._rclpy.spin_once(self.node, timeout_sec=0.01)
            if time.time() > deadline:
                raise RuntimeError(f"no wheel_torque_cmd from the ROS node at step {self._step}")
        self._step += 1
        return torch.tensor(self.torque / car_cfg.WHEEL_STALL_TORQUE_NM, device=self.env.device).unsqueeze(0)

    def close(self) -> None:
        self._process.send_signal(signal.SIGINT)  # the node's own Ctrl-C shutdown
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self.node.destroy_node()
        self._rclpy.try_shutdown()
