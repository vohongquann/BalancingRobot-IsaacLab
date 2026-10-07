# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The RL cascade: task registration, the contract between the velocity and position stages, and the frozen policies
(no simulator)."""

import importlib.util
import json
import pathlib
import re
import sys

import pytest
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import Balance_Car_RL.tasks  # noqa: F401
from Balance_Car_RL.assets import BALANCE_CAR_RL_ASSETS_DIR
from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.mdp.actions.frozen_policy import (
    FROZEN_DIR,
    STAGE_IO,
    _normalized,
    contract_fingerprint,
    file_sha256,
    frozen_policy_path,
    load_frozen,
)
from Balance_Car_RL.car.mdp.actions.velocity_action import velocity_observation
from Balance_Car_RL.car.rl_control.velocity_env_cfg import SPEED_MAX_M_S, YAW_RATE_MAX_RAD_S

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
    for name in cascade_script.ALL_ORDER:
        assert set(cascade_script.STAGES[name].needs) <= done, name
        done.add(name)


def test_every_task_starts_its_observation_with_the_imu():
    """The shared ``PolicyCfg`` (``car_env_cfg.py``) puts ``imu`` first; each task appends its terms in order."""
    expected = {
        "BalanceCar-Upright-v0": ["imu", "wheel_vel", "last_action"],
        # the frozen velocity policy's order, rebuilt by mdp/actions/velocity_action.py
        "BalanceCar-Velocity-v0": ["imu", "speed", "yaw_rate", "velocity_command", "last_action"],
        "BalanceCar-Position-v0": ["imu", "speed", "yaw_rate", "pose_command", "last_action"],
    }
    for task, terms in expected.items():
        assert _terms(load_cfg_from_registry(task, "env_cfg_entry_point")) == terms, task


def test_position_stage_uses_the_ranges_and_scale_of_the_velocity_stage():
    cfg = load_cfg_from_registry("BalanceCar-Position-v0", "env_cfg_entry_point")
    action = cfg.actions.velocity_command
    assert action.stage == "velocity"
    assert (action.speed_scale, action.yaw_rate_scale) == (SPEED_MAX_M_S, YAW_RATE_MAX_RAD_S)
    assert action.torque_scale == car_cfg.WHEEL_STALL_TORQUE_NM
    velocity = load_cfg_from_registry("BalanceCar-Velocity-v0", "env_cfg_entry_point")
    ranges = velocity.commands.base_velocity.ranges
    assert ranges.lin_vel_x == (-SPEED_MAX_M_S, SPEED_MAX_M_S) and ranges.lin_vel_y == (0.0, 0.0)
    assert ranges.ang_vel_z == (-YAW_RATE_MAX_RAD_S, YAW_RATE_MAX_RAD_S)
    mix = velocity.actions.wheel_torque
    assert mix.scale == car_cfg.WHEEL_STALL_TORQUE_NM and mix.turn_share == action.turn_share
    assert mix.joint_names == action.joint_names == ["left_wheel_joint", "right_wheel_joint"]
    # the robot the frozen policy was trained on, randomization included
    assert type(cfg.events) is type(velocity.events)


def test_turn_penalty_is_off_where_the_turn_is_commanded():
    for task in ("BalanceCar-Velocity-v0", "BalanceCar-Position-v0"):
        assert load_cfg_from_registry(task, "env_cfg_entry_point").rewards.yaw_rate is None, task
    assert load_cfg_from_registry("BalanceCar-Upright-v0", "env_cfg_entry_point").rewards.yaw_rate is not None


def test_frozen_velocity_action_rebuilds_the_velocity_observation_order():
    imu, speed, yaw_rate, command, last_action = (
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0]]),
        torch.tensor([[4.0]]),
        torch.tensor([[5.0, 6.0]]),
        torch.tensor([[7.0, 8.0]]),
    )
    obs = velocity_observation(imu, speed, yaw_rate, command, last_action)
    assert obs.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    assert obs.shape[1] == STAGE_IO["velocity"][0]


def test_contract_fingerprint_has_no_entry_for_a_stage_nothing_depends_on():
    assert contract_fingerprint("position") is None  # nothing is frozen on top of it
    for stage in ("velocity", "pitch_gains"):
        assert isinstance(contract_fingerprint(stage), str) and len(contract_fingerprint(stage)) == 64


def test_fingerprint_ignores_comments_and_docstrings_but_not_code(tmp_path):
    source = tmp_path / "terms.py"
    source.write_text('"""Module doc."""\nWEIGHT = 2.0  # a comment\nWEIGHT_DOC = 1\n"""doc of the constant"""\n')
    before = _normalized(source)
    source.write_text('"""Another module doc."""\n# only a comment\nWEIGHT = 2.0\nWEIGHT_DOC = 1\n')
    assert _normalized(source) == before
    source.write_text('"""Module doc."""\nWEIGHT = 2.5\nWEIGHT_DOC = 1\n')
    assert _normalized(source) != before


def test_fingerprint_covers_what_the_policy_depends_on():
    from Balance_Car_RL.car.mdp.actions.frozen_policy import _CONTRACT_FILES

    expected = {"car_cfg.py", "balboa.urdf", "imu_mount.json", "body_collision.stl", "rewards.py", "car_ppo_cfg.py"}
    for stage in ("velocity", "pitch_gains"):
        assert expected <= {path.name for path in _CONTRACT_FILES[stage]}, stage
        assert all(path.is_file() for path in _CONTRACT_FILES[stage]), stage
    assert {"locomotion.py", "velocity_env_cfg.py", "velocity_ppo_cfg.py", "symmetry.py"} <= {
        path.name for path in _CONTRACT_FILES["velocity"]
    }


@pytest.mark.parametrize("stage", ["velocity", "pitch_gains"])
def test_frozen_meta_records_the_contract_it_was_frozen_with(stage):
    """The recorded fingerprint must be present (a policy frozen without one can never be called stale) and equal to
    the current one, otherwise the policy is stale and should be retrained."""
    if not frozen_policy_path(stage).is_file():
        pytest.skip(f"stage '{stage}' not trained yet")
    meta = json.loads(frozen_policy_path(stage).with_name("meta.json").read_text())
    assert "contract_sha256" in meta, "frozen without a contract fingerprint: retrain with scripts/train_cascade.py"
    assert meta["contract_sha256"] == contract_fingerprint(stage), (
        f"the frozen '{stage}' policy's recorded contract hash no longer matches the current source: it is stale "
        "and should be retrained (`python scripts/train_cascade.py`)"
    )


@pytest.mark.parametrize(("upper", "lower"), [("position", "velocity"), ("velocity_gains", "pitch_gains")])
def test_frozen_stage_was_trained_on_the_frozen_stage_below(upper, lower):
    if not (frozen_policy_path(upper).is_file() and frozen_policy_path(lower).is_file()):
        pytest.skip(f"stages '{lower}' and '{upper}' not both trained yet")
    meta = json.loads(frozen_policy_path(upper).with_name("meta.json").read_text())
    assert meta["needs_sha256"][lower] == file_sha256(frozen_policy_path(lower)), (
        f"the '{lower}' policy was retrained after '{upper}': retrain {upper} on it (--stages {lower} {upper})"
    )


def test_retraining_a_stage_marks_the_stages_trained_on_it_as_stale(cascade_script, tmp_path, monkeypatch):
    monkeypatch.setattr(cascade_script, "FROZEN", tmp_path)
    (tmp_path / "velocity").mkdir()
    (tmp_path / "position").mkdir()
    (tmp_path / "velocity" / "policy.pt").write_bytes(b"first velocity policy")
    (tmp_path / "position" / "policy.pt").write_bytes(b"position policy")
    meta = {"needs_sha256": {"velocity": file_sha256(tmp_path / "velocity" / "policy.pt")}}
    (tmp_path / "position" / "meta.json").write_text(json.dumps(meta))
    assert cascade_script.stale_dependents("velocity") == []
    (tmp_path / "velocity" / "policy.pt").write_bytes(b"retrained velocity policy")
    assert cascade_script.stale_dependents("velocity") == ["position"]
    assert cascade_script.stale_dependents("position") == []


def test_command_ranges_are_reachable():
    urdf = (BALANCE_CAR_RL_ASSETS_DIR / "balboa" / "balboa.urdf").read_text()
    y = [float(v) for v in re.findall(r'_wheel_joint" type="continuous">\s*<origin xyz="0 (-?[\d.]+) 0"', urdf)]
    assert len(y) == 2
    track = abs(y[0] - y[1])
    no_load_speed = car_cfg.WHEEL_NO_LOAD_SPEED_RAD_S * car_cfg.WHEEL_RADIUS_M
    # the outer wheel of the fastest turn at the fastest speed is still below the no-load speed
    assert no_load_speed > SPEED_MAX_M_S + YAW_RATE_MAX_RAD_S * track / 2


def test_frozen_paths_are_inside_the_package():
    assert frozen_policy_path("velocity") == FROZEN_DIR / "velocity" / "policy.pt"
    assert FROZEN_DIR.parent.name == "rl_control"


@pytest.mark.skipif(not frozen_policy_path("velocity").is_file(), reason="stage 'velocity' not trained yet")
def test_frozen_velocity_policy_matches_its_contract():
    path = frozen_policy_path("velocity")
    policy = load_frozen("velocity", "cpu")
    out = policy(torch.zeros(3, 8))
    assert out.shape == (3, 2) and torch.isfinite(out).all()
    leaning = policy(torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert leaning[0, 0] > 0.0  # a forward lean is answered with forward common torque
    meta = json.loads(path.with_name("meta.json").read_text())
    assert meta["stage"] == "velocity" and meta["evaluation"]["pass"] is True
    assert path.with_name("policy.onnx").is_file()


class _Actor(torch.nn.Module):
    """The layout rsl_rl exports (``mlp.0``, ``mlp.2``, ``mlp.4``), without a trained network."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.mlp(x)


def _write_policy(root, stage, obs_dim, action_dim):
    (root / stage).mkdir(parents=True)
    torch.jit.script(_Actor(obs_dim, action_dim)).save(str(root / stage / "policy.pt"))


@pytest.mark.parametrize("stage", list(STAGE_IO))
def test_load_frozen_accepts_the_size_of_the_stage_and_freezes_it(tmp_path, stage):
    _write_policy(tmp_path, stage, *STAGE_IO[stage])
    policy = load_frozen(stage, "cpu", tmp_path)
    assert policy(torch.zeros(1, STAGE_IO[stage][0])).shape == (1, STAGE_IO[stage][1])
    assert not any(p.requires_grad for p in policy.parameters())


def test_load_frozen_rejects_another_size_and_a_missing_file(tmp_path):
    _write_policy(tmp_path, "velocity", 6, 2)  # the upright layout, not the 8 inputs of the velocity stage
    with pytest.raises(ValueError, match="obs/action size 6/2"):
        load_frozen("velocity", "cpu", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_cascade.py --stages position"):
        load_frozen("position", "cpu", tmp_path)


def test_load_frozen_warns_when_no_contract_was_recorded(tmp_path, capsys):
    _write_policy(tmp_path, "velocity", *STAGE_IO["velocity"])
    load_frozen("velocity", "cpu", tmp_path)
    assert "no contract recorded" in capsys.readouterr().out
    meta = tmp_path / "velocity" / "meta.json"
    meta.write_text(json.dumps({"contract_sha256": contract_fingerprint("velocity")}))
    load_frozen("velocity", "cpu", tmp_path)
    assert capsys.readouterr().out == ""
    meta.write_text(json.dumps({"contract_sha256": "0" * 64}))
    load_frozen("velocity", "cpu", tmp_path)
    assert "its observation/action contract changed" in capsys.readouterr().out
