"""Terms of the velocity and position tasks (``mdp/locomotion.py``, ``mdp/navigation.py``) and the classical
go-to-goal controller (``tools/path_following.py``): no simulator."""

import math
from types import SimpleNamespace

import pytest
import torch

from Balance_Car_RL.car.mdp.locomotion import velocity_command
from Balance_Car_RL.car.mdp.navigation import heading_error_near_goal, pose_command_2d
from Balance_Car_RL.car.pid_control.go_to_goal import GoToGoalPID, goal_errors
from Balance_Car_RL.car.tools.path_following import path_goal, path_reference


def _env(command):
    manager = SimpleNamespace(get_command=lambda name: torch.as_tensor(command, dtype=torch.float32))
    return SimpleNamespace(command_manager=manager)


def test_velocity_command_drops_the_sideways_speed():
    assert velocity_command(_env([[0.3, 0.0, -1.0]]))[0].tolist() == pytest.approx([0.3, -1.0])


def test_pose_command_is_shortened_to_the_largest_distance():
    env = _env([[3.0, 4.0, 0.0, 0.5], [0.3, 0.4, 0.0, -1.0]])
    obs = pose_command_2d(env, max_distance=2.0)
    assert torch.allclose(obs, torch.tensor([[1.2, 1.6, 0.5], [0.3, 0.4, -1.0]]))


def test_heading_counts_near_the_goal_only():
    env = _env([[0.0, 0.0, 0.0, 1.0], [3.0, 4.0, 0.0, 1.0]])
    # the heading counts at the goal, hardly 5 m away
    near, far = heading_error_near_goal(env, std=0.2).tolist()
    assert near == pytest.approx(1.0) and far < 1e-6


def test_go_to_goal_turns_to_the_goal_and_backs_up_to_one_behind():
    goal = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.5], [-1.0, 0.5, 0.0]])
    ahead, angle = goal_errors(goal)
    assert ahead.tolist() == pytest.approx([1.0, 0.0, -1.0, 0.0, -1.0])  # the speed loop drives x to 0
    # ahead: straight on; left: turn left; behind: no turn, back up; on the goal: its heading; behind-left: the tail
    # turns toward it (turn right)
    assert angle[:4].tolist() == pytest.approx([0.0, math.pi / 2 * math.tanh(10.0), 0.0, 0.5], abs=1e-4)
    assert angle[4] < 0
    speed, turn = GoToGoalPID(0.02, speed_max=0.4, turn_max=2.0).step(goal)
    assert speed.tolist() == pytest.approx([0.4, 0.0, -0.4, 0.0, -0.4])  # clipped to the velocity task's range
    assert turn.abs().max() <= 2.0 and turn[3] == pytest.approx(1.0)


def test_path_goal_leads_the_reference_and_faces_along_the_path():
    x, y, heading = path_goal(5.0)
    ahead = path_reference(5.6)
    assert (x, y) == pytest.approx(tuple(ahead[0]))
    assert heading == pytest.approx(math.atan2(ahead[1][1], ahead[1][0]))


def test_wheel_mix_turns_with_a_share_of_the_range():
    from Balance_Car_RL.car.mdp.actions.wheel_action import mix_wheels

    wheels = mix_wheels(torch.tensor([[0.5, 1.0], [0.0, -1.0], [1.0, 1.0], [2.0, 0.0]]), turn_share=0.2)
    # right wheel more for a positive turn (turns left); clipped to [-1, 1]
    assert torch.allclose(wheels, torch.tensor([[0.3, 0.7], [0.2, -0.2], [0.8, 1.0], [1.0, 1.0]]))


def test_position_gain_layer_with_zero_action_is_the_tuned_go_to_goal_pid():
    from Balance_Car_RL.car.mdp.actions.position_gain_action import PositionGainLayer

    goal = torch.tensor([[0.5, 0.2, 0.0], [-0.3, -0.1, 1.0], [0.0, 0.0, -0.5]])
    layer, pid = PositionGainLayer(0.02, 0.4, 2.0), GoToGoalPID(0.02, speed_max=0.4, turn_max=2.0)
    for _ in range(3):  # a few steps, so the integral and the derivative take part too
        command, gains = layer.step(torch.zeros(3, 6), goal)
        speed, turn = pid.step(goal)
    assert torch.allclose(command, torch.stack([speed, turn], dim=1))
    assert torch.allclose(gains[0], torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]))  # kp, ki, kd of speed, then turn


def test_position_gain_task_is_the_position_task_with_another_action():
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import Balance_Car_RL.tasks  # noqa: F401

    gains = load_cfg_from_registry("BalanceCar-Position-Gains-v0", "env_cfg_entry_point")
    rl = load_cfg_from_registry("BalanceCar-Position-v0", "env_cfg_entry_point")
    assert gains.rewards.to_dict() == rl.rewards.to_dict() and gains.commands.to_dict() == rl.commands.to_dict()
    action = gains.actions.position_gains
    for field in ("stage", "speed_scale", "yaw_rate_scale", "torque_scale", "turn_share"):
        assert getattr(action, field) == getattr(rl.actions.velocity_command, field), field
    agent = load_cfg_from_registry("BalanceCar-Position-Gains-v0", "rsl_rl_cfg_entry_point")
    assert agent.algorithm.symmetry_cfg is None and agent.experiment_name == "balance_car_position_gains"
