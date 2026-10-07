# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The gain networks of the ROS node (numpy) against the ones of the simulation (torch): constants, PID, torques, the
observation the networks are fed, and the exported ONNX against the frozen TorchScript. Needs
``PYTHONPATH=ros/src/car_bridge``; no simulator, no ROS."""

import numpy as np
import pytest
import torch
from car_bridge import gain_cascade, gains
from car_bridge.estimator import WHEEL_RADIUS_M
from car_bridge.gain_cascade import GainCascade, PitchGainController, SpeedGainController
from car_bridge.pid import PID as NumpyPID
from car_bridge.policy import TORQUE_SCALE_NM

from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.mdp import gains as sim_gains
from Balance_Car_RL.car.mdp.actions.frozen_policy import frozen_policy_path
from Balance_Car_RL.car.mdp.actions.gain_action import PitchGainLayer, SpeedGainLayer, _pitch_gains_observation
from Balance_Car_RL.car.mdp.actions.pitch_action import apply_speed_guard
from Balance_Car_RL.car.pid_control import PID as TorchPID
from Balance_Car_RL.car.rl_control.pitch_env_cfg import PITCH_TARGET_MAX_RAD, SPEED_GUARD_M_S
from Balance_Car_RL.car.rl_control.velocity_env_cfg import SPEED_MAX_M_S

DT = 0.02
rng = np.random.default_rng(0)


class Recorder:
    """A stand-in for an ONNX actor: returns the given actions in turn and keeps the observations it was fed."""

    def __init__(self, actions):
        self.actions, self.observations = list(actions), []

    def __call__(self, obs):
        self.observations.append(np.array(obs, dtype=np.float32))
        return np.asarray(self.actions[len(self.observations) - 1], dtype=np.float32)


def test_ros_and_sim_share_the_constants():
    assert abs(TORQUE_SCALE_NM - car_cfg.WHEEL_STALL_TORQUE_NM) < 1e-4
    assert WHEEL_RADIUS_M == car_cfg.WHEEL_RADIUS_M
    assert gains.FACTOR == sim_gains.FACTOR
    assert gains.KI_PER_KP == sim_gains.KI_PER_KP and gains.KD_PER_KP == sim_gains.KD_PER_KP
    for name, ros_layer in (("pitch", gains.PITCH), ("speed", gains.SPEED)):
        sim_layer = sim_gains.GAIN_LAYERS[name]
        assert np.allclose(ros_layer.nominal, sim_layer.nominal.numpy()), name
        assert np.allclose(ros_layer.maximum, sim_layer.maximum.numpy()), name
        assert ros_layer.int_limit == sim_layer.int_limit, name
    assert gain_cascade.DT_S == DT
    assert gain_cascade.PITCH_TARGET_MAX_RAD == PITCH_TARGET_MAX_RAD
    assert gain_cascade.SPEED_TARGET_MAX_M_S == SPEED_MAX_M_S
    assert gain_cascade.SPEED_GUARD_M_S == SPEED_GUARD_M_S


def test_gain_map_matches_the_simulation_for_any_action():
    actions = rng.uniform(-1.5, 1.5, size=(50, 3)).astype(np.float32)  # also outside [-1, 1]: both clip
    for name, ros_layer in (("pitch", gains.PITCH), ("speed", gains.SPEED)):
        sim = torch.stack(sim_gains.GAIN_LAYERS[name].gains(torch.from_numpy(actions)), dim=-1).numpy()
        ros = np.stack([ros_layer.gains(a) for a in actions])
        assert np.allclose(ros, sim, rtol=1e-5, atol=1e-8), name


def test_pid_matches_the_simulation_pid():
    errors = rng.normal(0.0, 0.3, size=200)
    ros = NumpyPID(kp=1.6, ki=0.5, kd=0.05, out_limit=1.0, int_limit=0.2)
    sim = TorchPID(kp=1.6, ki=0.5, kd=0.05, out_limit=1.0, int_limit=0.2)
    for e in errors:
        assert ros.update(float(e), DT) == pytest.approx(float(sim.update(torch.tensor([e]), DT)[0]), abs=1e-5)
    ros.reset()
    assert ros.integral == 0.0 and ros.prev_error is None


@pytest.mark.parametrize("with_network", [False, True])
def test_pitch_controller_gives_the_torques_of_the_simulation_layer(with_network):
    steps = 60
    actions = rng.uniform(-1.0, 1.0, size=(steps, 3)) if with_network else np.zeros((steps, 3))
    ros = PitchGainController(Recorder(actions) if with_network else None)
    sim = PitchGainLayer(DT)
    for k in range(steps):
        imu, wheel, target = rng.normal(0, 0.15, 2), rng.normal(0, 3.0, 2), rng.uniform(-0.08, 0.08)
        want = (
            sim.step(
                torch.tensor([actions[k]], dtype=torch.float32),
                torch.tensor([imu], dtype=torch.float32),
                torch.tensor([[target]], dtype=torch.float32),
            )[0]
            * car_cfg.WHEEL_STALL_TORQUE_NM
        )
        got = ros.step(imu[0], imu[1], wheel, target)
        assert np.allclose(got, want.numpy(), atol=1e-6) and got[0] == got[1]
    assert np.all(np.abs(got) <= TORQUE_SCALE_NM)


def test_pitch_controller_feeds_the_network_the_training_observation():
    actions = [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]
    actor = Recorder(actions)
    ros = PitchGainController(actor)
    ros.step(0.05, -0.1, (1.0, 2.0), 0.03)
    ros.step(0.06, -0.2, (1.5, 2.5), -0.02)
    first = _pitch_gains_observation(
        torch.tensor([[0.05, -0.1]]), torch.tensor([[1.0, 2.0]]), torch.zeros(1, 3), torch.tensor([[0.03]])
    )
    second = _pitch_gains_observation(
        torch.tensor([[0.06, -0.2]]), torch.tensor([[1.5, 2.5]]), torch.tensor([actions[0]]), torch.tensor([[-0.02]])
    )  # last gains fed back
    assert np.allclose(actor.observations[0], first[0].numpy()) and actor.observations[0].shape == (8,)
    assert np.allclose(actor.observations[1], second[0].numpy())


def test_pitch_controller_clips_the_action_and_feeds_the_clipped_one_back():
    actor = Recorder([[5.0, -5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ros = PitchGainController(actor)
    ros.step(0.0, 0.0, (0.0, 0.0), 0.0)
    ros.step(0.0, 0.0, (0.0, 0.0), 0.0)
    assert actor.observations[1][4:7].tolist() == [1.0, -1.0, 0.0]
    ros.reset()
    ros.step(0.0, 0.0, (0.0, 0.0), 0.0)  # a reset forgets the history: the next observation starts from zero again
    assert actor.observations[2][4:7].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("with_network", [False, True])
def test_speed_controller_gives_the_pitch_target_of_the_simulation_layer(with_network):
    steps = 80
    actions = rng.uniform(-1.0, 1.0, size=(steps, 3)) if with_network else np.zeros((steps, 3))
    ros = SpeedGainController(Recorder(actions) if with_network else None)
    sim = SpeedGainLayer(DT, PITCH_TARGET_MAX_RAD)
    for k in range(steps):
        speed, target = rng.normal(0, 0.4), rng.uniform(-0.4, 0.4)
        want = sim.step(
            torch.tensor([actions[k]], dtype=torch.float32),
            torch.tensor([[speed]], dtype=torch.float32),
            torch.tensor([[target]], dtype=torch.float32),
        )
        want = apply_speed_guard(want, torch.tensor([[speed]], dtype=torch.float32), SPEED_GUARD_M_S)
        assert ros.step(0.0, 0.0, speed, target) == pytest.approx(float(want[0, 0]), abs=1e-6)


def test_speed_controller_observation_and_guard():
    actor = Recorder([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    ros = SpeedGainController(actor)
    ros.step(0.04, -0.3, 0.1, 0.25)
    ros.step(0.05, -0.2, 0.2, 0.30)
    assert actor.observations[0].tolist() == pytest.approx([0.04, -0.3, 0.1, 0.25, 0.0, 0.0, 0.0])
    assert actor.observations[1].tolist() == pytest.approx([0.05, -0.2, 0.2, 0.30, 0.0, 0.0, 0.0])
    assert actor.observations[1].shape == (7,)
    # faster than the guard and still asked to go faster: no lean forward (a lean back to slow down still passes)
    assert SpeedGainController().step(0.0, 0.0, 0.6, 0.4) < 0.0  # too fast for the target: it leans back, not guarded
    assert gain_cascade.apply_speed_guard(0.05, 0.6) == 0.0
    assert gain_cascade.apply_speed_guard(-0.05, 0.6) == -0.05


def test_cascade_limits_the_commands_to_the_trained_range():
    cascade = GainCascade(speed_layer=False)
    big = cascade.act(0.0, 0.0, (0.0, 0.0), pitch_cmd=1.0)
    limit = GainCascade(speed_layer=False).act(0.0, 0.0, (0.0, 0.0), pitch_cmd=PITCH_TARGET_MAX_RAD)
    assert np.allclose(big, limit)
    cascade = GainCascade(speed_layer=True)
    big = cascade.act(0.0, 0.0, (0.0, 0.0), speed_cmd=5.0)
    limit = GainCascade(speed_layer=True).act(0.0, 0.0, (0.0, 0.0), speed_cmd=SPEED_MAX_M_S)
    assert np.allclose(big, limit)


def test_cascade_speed_estimate_is_the_encoders_plus_the_pitch_rate():
    actor = Recorder([[0.0, 0.0, 0.0]])
    cascade = GainCascade(speed_layer=True)
    cascade.speed = SpeedGainController(actor)
    cascade.act(0.0, 0.5, (3.0, 5.0), speed_cmd=0.1)
    assert actor.observations[0][2] == pytest.approx(WHEEL_RADIUS_M * (4.0 + 0.5))


def test_the_tuned_cascade_keeps_the_linear_plant_upright_and_slows_it_down(plant):
    """Without any network: the tuned pitch and speed PIDs of the node on the linear model of the robot."""
    _, ad, bd = plant
    for z0 in ([0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 6.0]):
        cascade, z = GainCascade(speed_layer=True), np.array(z0, float)
        for _ in range(500):
            theta, theta_dot, psi_dot = z
            joint = psi_dot - theta_dot  # what the encoders report (relative to the body)
            torque = cascade.act(theta, theta_dot, (joint, joint))
            z = ad @ z + bd * (torque[0] + torque[1])
        assert abs(z[0]) < 0.01 and abs(z[2]) < 0.5, z0


ONNX = {stage: frozen_policy_path(stage).with_name("policy.onnx") for stage in ("pitch_gains", "velocity_gains")}


@pytest.mark.skipif(not all(p.is_file() for p in ONNX.values()), reason="gain networks not frozen yet")
@pytest.mark.parametrize("stage", ["pitch_gains", "velocity_gains"])
def test_exported_onnx_matches_the_frozen_torchscript(stage):
    """The file the robot loads and the network the simulation froze give the same action."""
    from car_bridge.policy import OnnxActor

    onnx_actor = OnnxActor(str(ONNX[stage]))
    frozen = torch.jit.load(str(frozen_policy_path(stage)), map_location="cpu").eval()
    obs_dim = 8 if stage == "pitch_gains" else 7
    for _ in range(20):
        obs = rng.normal(0.0, 0.3, obs_dim).astype(np.float32)
        with torch.no_grad():
            want = frozen(torch.from_numpy(obs)[None])[0].numpy()
        assert np.allclose(onnx_actor(obs), want, atol=1e-4)


@pytest.mark.skipif(not all(p.is_file() for p in ONNX.values()), reason="gain networks not frozen yet")
def test_frozen_gain_cascade_keeps_the_linear_plant_upright(plant):
    _, ad, bd = plant
    for z0 in ([0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]):
        cascade = GainCascade(str(ONNX["pitch_gains"]), str(ONNX["velocity_gains"]))
        z = np.array(z0, float)
        for _ in range(500):
            theta, theta_dot, psi_dot = z
            joint = psi_dot - theta_dot
            torque = cascade.act(theta, theta_dot, (joint, joint))
            assert np.all(np.isfinite(torque)) and np.all(np.abs(torque) <= TORQUE_SCALE_NM + 1e-9)
            z = ad @ z + bd * (torque[0] + torque[1])
        assert abs(z[0]) < 0.05, z0
