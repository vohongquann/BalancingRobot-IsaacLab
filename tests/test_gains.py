# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The gain tasks: the map from network output to PID gains, the two gain layers, and the task definitions (no
simulator)."""

import pytest
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import Balance_Car_RL.tasks  # noqa: F401
from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.mdp.actions.frozen_policy import STAGE_IO, contract_fingerprint
from Balance_Car_RL.car.mdp.actions.gain_action import PitchGainLayer, SpeedGainLayer, _pitch_gains_observation
from Balance_Car_RL.car.mdp.gains import FACTOR, GAIN_LAYERS, KD_PER_KP, KI_PER_KP
from Balance_Car_RL.car.pid_control import (
    PITCH_KD,
    PITCH_KI,
    PITCH_KP,
    SPEED_KV,
    CascadePID,
)
from Balance_Car_RL.car.rl_control.pitch_env_cfg import PITCH_TARGET_MAX_RAD, SPEED_GUARD_M_S

DT = 0.02


def test_zero_action_is_the_tuned_pid():
    pitch = GAIN_LAYERS["pitch"].gains(torch.zeros(2, 3))
    assert [float(g[0]) for g in pitch] == pytest.approx([PITCH_KP, PITCH_KI, PITCH_KD])
    speed = GAIN_LAYERS["speed"].gains(torch.zeros(2, 3))
    assert [float(g[0]) for g in speed] == pytest.approx([SPEED_KV, 0.0, 0.0])


def test_gains_stay_within_a_factor_of_the_tuned_ones_and_off_terms_stay_off():
    lo, hi = GAIN_LAYERS["pitch"].gains(-torch.ones(1, 3)), GAIN_LAYERS["pitch"].gains(torch.ones(1, 3))
    for nominal, low, high in zip((PITCH_KP, PITCH_KI, PITCH_KD), lo, hi, strict=True):
        assert float(low) == pytest.approx(nominal / FACTOR) and float(high) == pytest.approx(nominal * FACTOR)
    # a term the tuned PID does not have is off for a <= 0 and grows to its maximum at a = 1
    off = GAIN_LAYERS["speed"].gains(-torch.ones(1, 3))
    on = GAIN_LAYERS["speed"].gains(torch.ones(1, 3))
    assert float(off[1]) == 0.0 and float(off[2]) == 0.0
    assert float(on[1]) == pytest.approx(KI_PER_KP * SPEED_KV) and float(on[2]) == pytest.approx(KD_PER_KP * SPEED_KV)
    assert float(GAIN_LAYERS["speed"].gains(5.0 * torch.ones(1, 3))[0]) == pytest.approx(SPEED_KV * FACTOR)  # clipped


def test_pitch_layer_with_the_zero_action_is_the_pitch_loop_of_the_cascade_pid():
    """Same torque as ``CascadePID`` with its speed loop off (pitch target 0) over several steps, integral included:
    the layer is that loop with its gains read from an action instead of constants."""
    imu = torch.tensor([[0.12, -0.3], [-0.05, 0.1]])
    pid, layer = CascadePID(kv=0.0), PitchGainLayer(DT)
    for _ in range(5):
        want = pid.act(torch.cat([imu, torch.zeros(2, 4)], dim=1))
        got = layer.step(torch.zeros(2, 3), imu, torch.zeros(2, 1))
        assert torch.allclose(got, want, atol=1e-6)
    assert got.shape == (2, 2) and torch.all(got[:, 0] == got[:, 1])


def test_pitch_layer_leans_into_the_fall_and_follows_the_gains():
    imu, target = torch.tensor([[0.2, 0.0]]), torch.zeros(1, 1)
    soft = PitchGainLayer(DT).step(-torch.ones(1, 3), imu, target)
    stiff = PitchGainLayer(DT).step(torch.ones(1, 3), imu, target)
    assert float(soft[0, 0]) > 0.0  # a forward lean drives the wheels forward
    assert float(stiff[0, 0]) > float(soft[0, 0])  # a higher kp pushes harder
    assert PitchGainLayer(DT).step(torch.zeros(1, 3), imu, torch.full((1, 1), 0.2))[0, 0] == pytest.approx(
        0.0, abs=0.05
    )


def test_pitch_layer_reset_clears_the_integral_of_the_rows_it_is_given():
    layer = PitchGainLayer(DT)
    layer.step(torch.zeros(2, 3), torch.tensor([[0.1, 0.0], [0.1, 0.0]]), torch.zeros(2, 1))
    assert torch.all(layer.pid.integral > 0.0)
    layer.reset(torch.tensor([0]))
    assert layer.pid.integral[0] == 0.0 and layer.pid.integral[1] > 0.0


def test_speed_layer_leans_toward_the_target_and_is_limited_to_the_pitch_range():
    layer = SpeedGainLayer(DT, PITCH_TARGET_MAX_RAD)
    still = torch.zeros(1, 1)
    forward = layer.step(torch.zeros(1, 3), still, torch.full((1, 1), 0.2))
    assert float(forward) == pytest.approx(SPEED_KV * 0.2 / car_cfg.WHEEL_RADIUS_M)  # kv times the error in rad/s
    backward = layer.step(torch.zeros(1, 3), torch.full((1, 1), 0.2), still)
    assert float(backward) < 0.0  # too fast: lean back
    big = layer.step(torch.ones(1, 3), still, torch.full((1, 1), 100.0))
    assert float(big) == pytest.approx(PITCH_TARGET_MAX_RAD)  # the pitch layer below is never asked for more


def test_gain_layers_run_on_many_environments_at_once():
    n = 5
    assert PitchGainLayer(DT).step(torch.rand(n, 3), torch.rand(n, 2), torch.rand(n, 1)).shape == (n, 2)
    assert SpeedGainLayer(DT, 0.08).step(torch.rand(n, 3), torch.rand(n, 1), torch.rand(n, 1)).shape == (n, 1)


def test_pitch_gains_observation_order_is_the_contract_of_the_frozen_gain_network():
    imu, wheel_vel, gains, target = (
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[5.0, 6.0, 7.0]]),
        torch.tensor([[8.0]]),
    )
    obs = _pitch_gains_observation(imu, wheel_vel, gains, target)
    assert obs.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    assert obs.shape[1] == STAGE_IO["pitch_gains"][0]


def _terms(cfg):
    group = cfg.observations.policy
    return [k for k in group.__dict__ if not k.startswith("_") and hasattr(getattr(group, k), "func")]


def test_gain_tasks_are_their_stage_with_another_action():
    pitch = load_cfg_from_registry("BalanceCar-Pitch-Gains-v0", "env_cfg_entry_point")
    velocity = load_cfg_from_registry("BalanceCar-Velocity-Gains-v0", "env_cfg_entry_point")
    assert _terms(pitch) == ["imu", "wheel_vel", "last_action", "pitch_target"]  # the order of _pitch_gains_observation
    assert _terms(velocity) == ["imu", "speed", "speed_target", "last_action"]
    assert pitch.commands.target.speed_guard == SPEED_GUARD_M_S
    action = velocity.actions.speed_gains
    assert action.pitch_scale == PITCH_TARGET_MAX_RAD and action.speed_guard == SPEED_GUARD_M_S
    assert action.torque_scale == car_cfg.WHEEL_STALL_TORQUE_NM and action.below_stage == "pitch_gains"
    assert pitch.actions.pitch_gains.torque_scale == car_cfg.WHEEL_STALL_TORQUE_NM


def test_gain_tasks_have_their_own_log_folders_and_a_bounded_gain_noise():
    for task, name in (
        ("BalanceCar-Pitch-Gains-v0", "balance_car_pitch_gains"),
        ("BalanceCar-Velocity-Gains-v0", "balance_car_velocity_gains"),
    ):
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        assert agent.experiment_name == name
        assert agent.actor.distribution_cfg.std_range[1] <= 0.3 and agent.clip_actions == 1.0


def test_frozen_pitch_gains_have_a_contract_and_the_velocity_gains_do_not():
    assert len(contract_fingerprint("pitch_gains")) == 64
    assert contract_fingerprint("velocity_gains") is None
