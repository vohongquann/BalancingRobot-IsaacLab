# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import Balance_Car_RL.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    assert all(
        not str(spec.kwargs.get("env_cfg_entry_point", "")).startswith("Balance_Car_RL.tasks")
        for spec in gym.registry.values()
    )

