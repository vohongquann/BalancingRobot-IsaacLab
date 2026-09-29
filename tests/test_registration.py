# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import Balance_Car_RL.tasks  # noqa: F401

TASK = "BalanceCar-Upright-v0"


def test_task_registered():
    """The balance car task is registered with environment and agent entry points."""
    spec = gym.spec(TASK)
    assert "env_cfg_entry_point" in spec.kwargs
    assert "rsl_rl_cfg_entry_point" in spec.kwargs


def test_configs_instantiate():
    """Both configs can be built without launching the simulator."""
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    assert env_cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert env_cfg.actions.wheel_torque.joint_names == [".*_wheel_joint"]
    assert agent_cfg.experiment_name == "balance_car_upright"
