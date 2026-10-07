"""MDP terms that need no simulator: they are built without Isaac Lab's managers, on stub environments."""

from types import SimpleNamespace

import pytest
import torch

from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.mdp.actions.velocity_action import FrozenVelocityAction
from Balance_Car_RL.car.mdp.commands import ScalarCommand
from Balance_Car_RL.car.mdp.observations import axle_speed
from Balance_Car_RL.car.mdp.rewards import yaw_rate_l2

N = 4
STALL = car_cfg.WHEEL_STALL_TORQUE_NM


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _robot(joint_vel, lin_vel=None, ang_vel=None):
    data = _ns(
        joint_vel=_ns(torch=torch.as_tensor(joint_vel, dtype=torch.float32)),
        root_lin_vel_b=_ns(torch=torch.as_tensor(lin_vel if lin_vel is not None else torch.zeros(N, 3))),
        root_ang_vel_b=_ns(torch=torch.as_tensor(ang_vel if ang_vel is not None else torch.zeros(N, 3))),
    )
    return _ns(data=data)


def _env(joint_vel, pitch_rate=0.0, yaw_rate=0.0, **robot_kwargs):
    """Stub environment: a robot, the IMU term (only its ``cache`` and ``yaw_rate`` are read) and the manager
    attributes."""
    robot = _robot(joint_vel, **robot_kwargs)
    cache = torch.zeros(N, 2)
    cache[:, 0], cache[:, 1] = 0.1, pitch_rate
    car_imu = _ns(cache=cache, yaw_rate=torch.full((N, 1), yaw_rate))
    return _ns(scene={"robot": robot}, car_imu=car_imu, num_envs=N, device="cpu")


# ── axle speed ──────────────────────────────────────────────────────────────────────────


def test_axle_speed_adds_the_pitch_rate_to_the_encoders():
    """The encoders are relative to the body: v = r (mean wheel speed + pitch rate)."""
    env = _env(torch.tensor([[10.0, 14.0]] * N), pitch_rate=1.0)
    assert torch.allclose(axle_speed(env, slice(None)), torch.full((N,), car_cfg.WHEEL_RADIUS_M * 13.0))


def test_axle_speed_needs_the_imu_term():
    env = _env(torch.zeros(N, 2))
    del env.car_imu
    try:
        axle_speed(env, slice(None))
    except RuntimeError as err:
        assert "ImuPitchAndRate" in str(err)
    else:
        raise AssertionError("no error without the IMU term")


# ── reward ──────────────────────────────────────────────────────────────────────────────


def test_yaw_rate_penalty_only_counts_the_turn():
    ang_vel = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, -0.5], [0.0, 0.0, 0.0]])
    env = _env(torch.zeros(N, 2))
    env.scene["robot"].data.root_ang_vel_b = _ns(torch=ang_vel)
    assert torch.allclose(yaw_rate_l2(env), torch.tensor([0.0, 9.0, 0.25, 0.0]))


# ── command ─────────────────────────────────────────────────────────────────────────────


def _command(speed, drawn, guard=0.5):
    term = ScalarCommand.__new__(ScalarCommand)
    term.cfg = _ns(low=-1.0, high=1.0, zero_probability=0.0, speed_guard=guard)
    term._env = _env(torch.zeros(N, 2))
    term._robot = _robot(torch.zeros(N, 2), lin_vel=torch.tensor([[v, 0.0, 0.0] for v in speed]))
    term._drawn = torch.tensor(drawn).unsqueeze(-1)
    term._command = term._drawn.clone()
    return term


def test_speed_guard_flips_a_command_that_would_speed_the_car_up():
    term = _command(speed=[0.8, 0.8, -0.8, 0.3], drawn=[0.05, -0.05, 0.05, 0.05])
    term._update_command()
    # fast forward asked forward: flipped; asked backward: kept; fast backward asked forward: kept; slow: kept
    assert term.command[:, 0].tolist() == pytest.approx([-0.05, -0.05, 0.05, 0.05])
    term._robot.data.root_lin_vel_b = _ns(torch=torch.zeros(N, 3))
    term._update_command()  # slow again: the drawn command is back (the guard never overwrites it)
    assert term.command[:, 0].tolist() == pytest.approx([0.05, -0.05, 0.05, 0.05])


def test_no_guard_leaves_the_command_alone():
    term = _command(speed=[2.0] * N, drawn=[0.05] * N, guard=None)
    term._update_command()
    assert term.command[:, 0].tolist() == pytest.approx([0.05] * N)


def test_resampled_command_stays_in_range_and_can_be_zero():
    term = _command(speed=[0.0] * N, drawn=[0.0] * N)
    term.cfg = _ns(low=-0.08, high=0.08, zero_probability=0.5, speed_guard=None)
    term._env.num_envs = 4000
    term._drawn, term._command = torch.zeros(4000, 1), torch.zeros(4000, 1)
    term._resample_command(torch.arange(4000))
    value = term._drawn[:, 0]
    assert value.abs().max() <= 0.08 and 0.4 < (value == 0).float().mean() < 0.6 and value.std() > 0.02


# ── frozen velocity action (position task) ──────────────────────────────────────────────


class _Recorder:
    """Stands in for the frozen velocity policy: remembers its observation, answers a fixed action."""

    def __init__(self, answer):
        self.answer, self.seen = torch.tensor(answer), None

    def __call__(self, obs):
        self.seen = obs.clone()
        return self.answer.expand(obs.shape[0], 2).clone()


def _action(joint_vel, answer=(0.2, -0.3), pitch_rate=0.0, yaw_rate=0.0):
    term = FrozenVelocityAction.__new__(FrozenVelocityAction)
    term.cfg = _ns(torque_scale=STALL, turn_share=0.5)
    term._env = _env(torch.as_tensor(joint_vel, dtype=torch.float32), pitch_rate=pitch_rate, yaw_rate=yaw_rate)
    term._asset = term._env.scene["robot"]
    term._joint_ids = slice(None)
    term._scale = torch.tensor([0.4, 2.0])
    term.policy = _Recorder(answer)
    term._raw_actions, term._command = torch.zeros(N, 2), torch.zeros(N, 2)
    term._policy_last_action, term._torque = torch.zeros(N, 2), torch.zeros(N, 2)
    return term


def test_velocity_action_scales_clamps_and_builds_the_training_observation():
    term = _action(torch.tensor([[10.0, 12.0]] * N), yaw_rate=0.7)
    term.process_actions(torch.tensor([[0.5, 0.5], [-1.0, 0.0], [3.0, -3.0], [0.0, 0.0]]))
    expected = torch.tensor([[0.2, 1.0], [-0.4, 0.0], [0.4, -2.0], [0.0, 0.0]])  # [-1, 1] x (0.4 m/s, 2 rad/s)
    assert torch.allclose(term.processed_actions, expected)
    # [pitch, pitch rate, speed, gyro turn rate, v, w, last action x2]: the order the velocity task was trained on
    speed = car_cfg.WHEEL_RADIUS_M * 11.0
    assert term.policy.seen[0].tolist() == pytest.approx([0.1, 0.0, speed, 0.7, 0.2, 1.0, 0.0, 0.0])
    # common 0.2, turn -0.3 x 0.5: left 0.35, right 0.05
    assert torch.allclose(term._torque, torch.tensor([[0.35, 0.05]]).expand(N, 2) * STALL)


def test_velocity_action_feeds_back_the_clipped_action_like_training():
    term = _action(torch.zeros(N, 2), answer=(1.7, -2.5))
    term.process_actions(torch.zeros(N, 2))
    assert term._policy_last_action[0].tolist() == [1.0, -1.0]  # clip_actions = 1.0 in training
    assert torch.allclose(term._torque, torch.tensor([[1.0, 0.5]]).expand(N, 2) * STALL)  # 1 +- 0.5, clipped
    term.process_actions(torch.zeros(N, 2))
    assert term.policy.seen[0, 6:8].tolist() == [1.0, -1.0]
