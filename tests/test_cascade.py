# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The cascade: task registration, the contract between stage 1 and stage 2, and the frozen policy (no simulator)."""

import importlib.util
import json
import pathlib
import sys

import pytest
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import Balance_Car_RL.tasks  # noqa: F401
from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.mdp.actions import FROZEN_DIR, _stage1_observation, contract_fingerprint, frozen_policy_path
from Balance_Car_RL.car.rl_control.pitch_env_cfg import PITCH_TARGET_MAX_RAD, SPEED_GUARD_M_S

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cascade_script():
    spec = importlib.util.spec_from_file_location("train_cascade", ROOT / "scripts/train_cascade.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


def _terms(cfg):
    return [
        name
        for name in cfg.observations.policy.__dict__
        if not name.startswith("_")
        and name not in {"concatenate_terms", "enable_corruption", "history_length", "flatten_history_dim"}
        and hasattr(getattr(cfg.observations.policy, name), "func")
    ]


def test_stage_table_matches_the_registry(cascade_script):
    for stage in cascade_script.STAGES.values():
        env_cfg = load_cfg_from_registry(stage.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(stage.task, "rsl_rl_cfg_entry_point")
        assert env_cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
        assert agent_cfg.experiment_name == stage.experiment


def test_stages_are_ordered_by_dependency(cascade_script):
    done = set()
    for name in cascade_script.DEFAULT_ORDER:
        assert set(cascade_script.STAGES[name].needs) <= done, name
        done.add(name)


def test_stage_1_observation_order_is_the_contract_of_the_frozen_policy():
    cfg = load_cfg_from_registry("BalanceCar-Pitch-v0", "env_cfg_entry_point")
    assert _terms(cfg) == ["imu", "wheel_vel", "last_action", "pitch_target"]  # mdp/actions.py builds this vector


def test_stage_2_uses_the_range_and_scale_of_stage_1():
    cfg = load_cfg_from_registry("BalanceCar-Velocity-v0", "env_cfg_entry_point")
    action = cfg.actions.pitch_target
    assert action.stage == "pitch"
    assert action.pitch_scale == PITCH_TARGET_MAX_RAD
    assert action.torque_scale == car_cfg.WHEEL_STALL_TORQUE_NM
    assert action.speed_guard == SPEED_GUARD_M_S  # the same guard stage 1's own command was trained with
    stage_1 = load_cfg_from_registry("BalanceCar-Pitch-v0", "env_cfg_entry_point")
    assert stage_1.commands.target.high == PITCH_TARGET_MAX_RAD and stage_1.commands.target.low == -PITCH_TARGET_MAX_RAD
    assert stage_1.actions.wheel_torque.scale == car_cfg.WHEEL_STALL_TORQUE_NM
    assert stage_1.commands.target.speed_guard == SPEED_GUARD_M_S


def test_frozen_action_rebuilds_the_exact_training_observation_order():
    """The concatenation ``FrozenPitchAction.process_actions`` builds must put each value where stage 1 (whose term
    order is checked above) put it -- this exercises the actual glue function without needing a simulator to
    construct a real ``FrozenPitchAction``."""
    imu, wheel_vel, last_action, target = (
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
        torch.tensor([[7.0]]),
    )
    obs = _stage1_observation(imu, wheel_vel, last_action, target)
    assert obs.shape == (1, 7)
    assert obs.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]


def test_contract_fingerprint_has_no_entry_for_a_stage_nothing_depends_on():
    assert contract_fingerprint("velocity") is None  # nothing is frozen on top of it (yet)
    assert isinstance(contract_fingerprint("pitch"), str) and len(contract_fingerprint("pitch")) == 64


@pytest.mark.skipif(not frozen_policy_path("pitch").is_file(), reason="stage 'pitch' not trained yet")
def test_frozen_pitch_meta_records_the_contract_it_was_frozen_with():
    """Not a hard requirement (older ``meta.json`` files predate this field, see ``FrozenPitchAction.__init__``), but
    once present it must be the real fingerprint of the files that were current when this test's own frozen policy
    was produced, not a placeholder -- otherwise the staleness warning could never fire."""
    meta = json.loads(frozen_policy_path("pitch").with_name("meta.json").read_text())
    if "contract_sha256" in meta:
        assert meta["contract_sha256"] == contract_fingerprint("pitch"), (
            "the frozen 'pitch' policy's recorded contract hash no longer matches the current source: it is stale "
            "and should be retrained (`python scripts/train_cascade.py --stages pitch velocity`)"
        )


def test_command_ranges_are_reachable():
    cfg = load_cfg_from_registry("BalanceCar-Velocity-v0", "env_cfg_entry_point")
    no_load_speed = car_cfg.WHEEL_NO_LOAD_SPEED_RAD_S * car_cfg.WHEEL_RADIUS_M
    assert cfg.commands.target.high < no_load_speed  # the wheels can reach the fastest command


def test_frozen_paths_are_inside_the_package():
    assert frozen_policy_path("pitch") == FROZEN_DIR / "pitch" / "policy.pt"
    assert FROZEN_DIR.parent.name == "rl_control"


@pytest.mark.skipif(not frozen_policy_path("pitch").is_file(), reason="stage 'pitch' not trained yet")
def test_frozen_pitch_policy_matches_its_contract():
    path = frozen_policy_path("pitch")
    policy = torch.jit.load(str(path), map_location="cpu").eval()
    out = policy(torch.zeros(3, 7))
    assert out.shape == (3, 2) and torch.isfinite(out).all()
    leaning = policy(torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert leaning[0, 0] > 0.0 and leaning[0, 1] > 0.0  # a forward lean is answered with forward torque on both wheels
    meta = json.loads(path.with_name("meta.json").read_text())
    assert meta["stage"] == "pitch" and meta["evaluation"]["pass"] is True
    assert path.with_name("policy.onnx").is_file()
