"""The drive cascade of the ROS node (numpy, ``car_bridge/drive.py``) against the simulation (torch): constants, wheel
mixing, the goal the networks see, the go-to-goal PID with network gains, odometry, and the exported ONNX against the
frozen TorchScript. Needs ``PYTHONPATH=ros/src/car_bridge``; no simulator, no ROS."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from car_bridge import drive
from car_bridge.drive import (
    DriveCascade,
    LqrVelocityController,
    Odometry,
    PositionGainController,
    VelocityController,
)
from car_bridge.policy import TORQUE_SCALE_NM, OnnxActor

from Balance_Car_RL.car.mdp.actions.frozen_policy import frozen_policy_path
from Balance_Car_RL.car.mdp.actions.position_gain_action import POSITION_GAIN_LAYERS, PositionGainLayer
from Balance_Car_RL.car.mdp.actions.velocity_action import velocity_observation
from Balance_Car_RL.car.mdp.actions.wheel_action import mix_wheels
from Balance_Car_RL.car.lqr_control import LQRController, design_lqr
from Balance_Car_RL.car.mdp.navigation import pose_command_2d
from Balance_Car_RL.car.pid_control import DrivePosition, DriveVelocity, go_to_goal
from Balance_Car_RL.car.pid_control import drive as sim_drive
from Balance_Car_RL.car.rl_control import position_env_cfg, velocity_env_cfg

rng = np.random.default_rng(0)


class Recorder:
    """A stand-in for an ONNX actor: returns the given actions in turn and keeps the observations it was fed."""

    def __init__(self, actions):
        self.actions, self.observations = list(actions), []

    def __call__(self, obs):
        self.observations.append(np.array(obs, dtype=np.float32))
        return np.asarray(self.actions[len(self.observations) - 1], dtype=np.float32)


def test_ros_and_sim_share_the_constants():
    assert drive.SPEED_MAX_M_S == velocity_env_cfg.SPEED_MAX_M_S
    assert drive.YAW_RATE_MAX_RAD_S == velocity_env_cfg.YAW_RATE_MAX_RAD_S
    assert drive.TURN_SHARE == velocity_env_cfg.TURN_SHARE
    assert drive.GOAL_OBS_MAX_M == position_env_cfg.GOAL_OBS_MAX_M
    assert (drive.GOAL_SPEED_KP, drive.GOAL_TURN_KP) == (go_to_goal.GOAL_SPEED_KP, go_to_goal.GOAL_TURN_KP)
    assert (drive.GOAL_INT_LIMIT, drive.NEAR_M) == (go_to_goal.GOAL_INT_LIMIT, go_to_goal.NEAR_M)
    for ros_layer, name in zip(drive.GOAL_LAYERS, ("speed", "turn"), strict=True):
        sim_layer = POSITION_GAIN_LAYERS[name]
        assert np.allclose(ros_layer.nominal, sim_layer.nominal.numpy()) and ros_layer.int_limit == sim_layer.int_limit


def test_wheel_mix_matches_the_simulation():
    actions = rng.uniform(-1.5, 1.5, size=(50, 2))
    want = mix_wheels(torch.tensor(actions), drive.TURN_SHARE).numpy()
    assert np.allclose([drive.mix_wheels(a) for a in actions], want)


def test_shortened_goal_and_goal_errors_match_the_simulation():
    goals = np.column_stack([rng.uniform(-4, 4, (60, 2)), rng.uniform(-math.pi, math.pi, 60)])
    command = torch.tensor(np.insert(goals, 2, 0.0, axis=1), dtype=torch.float32)  # Isaac Lab's [x, y, z, heading]
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda name: command))
    sim = pose_command_2d(env, max_distance=drive.GOAL_OBS_MAX_M).numpy()
    assert np.allclose([drive.shortened(g) for g in goals], sim, atol=1e-5)
    ahead, angle = go_to_goal.goal_errors(torch.tensor(sim))
    ros = np.array([drive.goal_errors(g) for g in sim])
    assert np.allclose(ros[:, 0], ahead.numpy(), atol=1e-5) and np.allclose(ros[:, 1], angle.numpy(), atol=1e-5)


@pytest.mark.parametrize("with_network", [False, True])
def test_position_gain_controller_gives_the_command_of_the_simulation_layer(with_network):
    steps = 40
    actions = rng.uniform(-1.2, 1.2, size=(steps, 6)) if with_network else np.zeros((steps, 6))
    goals = np.column_stack([rng.uniform(-1, 1, (steps, 2)), rng.uniform(-3, 3, steps)])
    ros = PositionGainController(Recorder(actions) if with_network else None)
    sim = PositionGainLayer(drive.DT_S, drive.SPEED_MAX_M_S, drive.YAW_RATE_MAX_RAD_S)
    for action, goal in zip(actions, goals, strict=True):
        want, _ = sim.step(torch.tensor(np.clip(action, -1, 1), dtype=torch.float32)[None], torch.tensor(goal)[None])
        got = ros.step(0.0, 0.0, 0.0, 0.0, goal)
        assert np.allclose(got, want[0].numpy(), atol=1e-5)


def test_velocity_controller_feeds_the_training_observation_and_mixes_the_wheels():
    actor = Recorder([[0.4, -1.7], [0.0, 0.0]])
    controller = VelocityController(actor)
    torque = controller.step(0.1, 0.2, 0.15, 0.7, (0.9, -5.0))  # the command is clipped to the trained range
    want = velocity_observation(
        torch.tensor([[0.1, 0.2]]),
        torch.tensor([[0.15]]),
        torch.tensor([[0.7]]),
        torch.tensor([[0.4, -2.0]]),
        torch.zeros(1, 2),
    )
    assert np.allclose(actor.observations[0], want[0].numpy())
    assert np.allclose(torque, drive.mix_wheels([0.4, -1.0]) * TORQUE_SCALE_NM)
    controller.step(0.1, 0.2, 0.15, 0.7, (0.0, 0.0))
    assert actor.observations[1][6:].tolist() == [pytest.approx(0.4), -1.0]  # the clipped action is fed back


def test_odometry_integrates_a_circle_and_turns_the_goal_into_the_car_frame():
    odometry = Odometry()
    for _ in range(int(round(math.pi / 0.02))):  # half a circle of radius 0.2 m at 0.2 m/s, 1 rad/s
        odometry.update(0.2, 1.0)
    assert (odometry.x, odometry.y) == (pytest.approx(0.0, abs=1e-3), pytest.approx(0.4, abs=1e-3))
    assert abs(abs(odometry.heading) - math.pi) < 0.01  # 157 steps of 0.02 s: 3.14 s
    # the car faces -x at (0, 0.4): the start is 0.4 m to its left, facing back
    x, y, heading = odometry.goal_in_car_frame((0.0, 0.0, 0.0))
    assert (x, y) == (pytest.approx(0.0, abs=1e-3), pytest.approx(0.4, abs=1e-3)) and abs(abs(heading) - math.pi) < 0.01


ONNX = {stage: frozen_policy_path(stage).with_name("policy.onnx") for stage in ("velocity", "position")}


@pytest.mark.skipif(not all(p.is_file() for p in ONNX.values()), reason="drive networks not frozen yet")
@pytest.mark.parametrize("stage", list(ONNX))
def test_exported_onnx_matches_the_frozen_torchscript(stage):
    onnx_actor = OnnxActor(str(ONNX[stage]))
    frozen = torch.jit.load(str(frozen_policy_path(stage)), map_location="cpu").eval()
    size = {"velocity": 8, "position": 9}[stage]
    for obs in rng.normal(0.0, 0.5, size=(20, size)).astype(np.float32):
        with torch.no_grad():
            want = frozen(torch.from_numpy(obs)[None])[0].numpy()
        assert np.allclose(onnx_actor(obs), want, atol=1e-4)


@pytest.mark.skipif(not all(p.is_file() for p in ONNX.values()), reason="drive networks not frozen yet")
def test_position_cascade_at_its_goal_asks_for_little():
    """At the goal, upright and still, the first command is small (open loop: without the car turning in response,
    the network's own last action would feed back, so only one step is meaningful here)."""
    cascade = DriveCascade("position", str(ONNX["velocity"]), str(ONNX["position"]))
    torque = cascade.act(0.0, 0.0, (0.0, 0.0), 0.0, None)
    assert abs(cascade.command[0]) < 0.02 and abs(cascade.command[1]) < 0.15
    assert np.all(np.abs(torque) < 0.05 * TORQUE_SCALE_NM)


def test_lqr_drive_shares_the_gains_of_the_simulation():
    assert np.allclose(drive.LQR_GAIN, design_lqr())
    assert (drive.TURN_KP, drive.TURN_KI) == (sim_drive.TURN_KP, sim_drive.TURN_KI)


def sim_torque(action) -> np.ndarray:
    """``[common, turn]`` of the simulation (N, 2) -> wheel torques [N m] of the first row."""
    return mix_wheels(action, drive.TURN_SHARE)[0].numpy() * TORQUE_SCALE_NM


def test_lqr_velocity_controller_gives_the_torques_of_the_simulation():
    steps = 60
    ros = LqrVelocityController()
    sim = DriveVelocity(LQRController(), drive.DT_S)
    states = rng.normal(0.0, [0.1, 0.5, 0.2, 0.8], size=(steps, 4))
    commands = np.column_stack([rng.uniform(-0.4, 0.4, steps), rng.uniform(-2.0, 2.0, steps)])
    for (pitch, rate, speed, turn_rate), command in zip(states, commands, strict=True):
        want = sim.step(
            *(torch.tensor([v], dtype=torch.float32) for v in (pitch, rate, speed, turn_rate)),
            torch.tensor(command[None], dtype=torch.float32),
        )
        assert np.allclose(ros.step(pitch, rate, speed, turn_rate, command), sim_torque(want), atol=1e-5)


def test_lqr_position_cascade_gives_the_torques_of_the_simulation():
    """``lqr_position`` in the car frame (the odometry stays at the start: no speed, no turn rate)."""
    cascade = DriveCascade("lqr_position")
    sim = DrivePosition(LQRController(), drive.DT_S, drive.SPEED_MAX_M_S, drive.YAW_RATE_MAX_RAD_S)
    pitches = rng.normal(0.0, 0.05, size=(40, 2))
    goal = (0.8, -0.3, 1.0)
    for pitch, rate in pitches:
        got = cascade.act(pitch, rate, (-rate, -rate), 0.0, goal)  # wheels turn back with the body: no speed
        want = sim.step(
            *(torch.tensor([v], dtype=torch.float32) for v in (pitch, rate, 0.0, 0.0)),
            torch.tensor([goal], dtype=torch.float32),
        )
        assert np.allclose(got, sim_torque(want), atol=1e-5)
        assert np.allclose(cascade.command, sim.command[0].numpy(), atol=1e-5)
