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
from Balance_Car_RL.car.mdp.actions.frozen_policy import (
    FROZEN_DIR,
    STAGE_IO,
    _normalized,
    contract_fingerprint,
    file_sha256,
    frozen_policy_path,
    load_frozen,
)
from Balance_Car_RL.car.mdp.actions.pitch_action import _stage1_observation
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
    # mdp/actions/pitch_action.py builds this vector
    assert _terms(cfg) == ["imu", "wheel_vel", "last_action", "pitch_target"]


def test_every_task_starts_its_observation_with_the_imu():
    """The shared ``PolicyCfg`` (``car_env_cfg.py``) puts ``imu`` first; each task appends its terms in order."""
    expected = {
        "BalanceCar-Upright-v0": ["imu", "wheel_vel", "last_action"],
        "BalanceCar-Velocity-v0": ["imu", "speed", "speed_target", "last_action"],  # the frozen velocity policy's order
    }
    for task, terms in expected.items():
        assert _terms(load_cfg_from_registry(task, "env_cfg_entry_point")) == terms, task


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

    names = {path.name for path in _CONTRACT_FILES["pitch"]}
    expected = {"car_cfg.py", "balboa.urdf", "imu_mount.json", "body_collision.stl", "rewards.py", "car_ppo_cfg.py"}
    assert expected <= names
    assert all(path.is_file() for path in _CONTRACT_FILES["pitch"])


@pytest.mark.skipif(not frozen_policy_path("pitch").is_file(), reason="stage 'pitch' not trained yet")
def test_frozen_pitch_meta_records_the_contract_it_was_frozen_with():
    """The recorded fingerprint must be present (a policy frozen without one can never be called stale) and equal to
    the current one, otherwise the policy is stale and should be retrained."""
    meta = json.loads(frozen_policy_path("pitch").with_name("meta.json").read_text())
    assert "contract_sha256" in meta, "frozen without a contract fingerprint: retrain with scripts/train_cascade.py"
    assert meta["contract_sha256"] == contract_fingerprint("pitch"), (
        "the frozen 'pitch' policy's recorded contract hash no longer matches the current source: it is stale "
        "and should be retrained (`python scripts/train_cascade.py --stages pitch velocity`)"
    )


@pytest.mark.skipif(
    not (frozen_policy_path("pitch").is_file() and frozen_policy_path("velocity").is_file()),
    reason="stages 'pitch' and 'velocity' not both trained yet",
)
def test_frozen_velocity_was_trained_on_the_frozen_pitch():
    meta = json.loads(frozen_policy_path("velocity").with_name("meta.json").read_text())
    assert meta["needs_sha256"]["pitch"] == file_sha256(frozen_policy_path("pitch")), (
        "the 'pitch' policy was retrained after 'velocity': retrain velocity on it (--stages pitch velocity)"
    )


def test_retraining_a_stage_marks_the_stages_trained_on_it_as_stale(cascade_script, tmp_path, monkeypatch):
    monkeypatch.setattr(cascade_script, "FROZEN", tmp_path)
    (tmp_path / "pitch").mkdir()
    (tmp_path / "velocity").mkdir()
    (tmp_path / "pitch" / "policy.pt").write_bytes(b"first pitch policy")
    (tmp_path / "velocity" / "policy.pt").write_bytes(b"velocity policy")
    meta = {"needs_sha256": {"pitch": file_sha256(tmp_path / "pitch" / "policy.pt")}}
    (tmp_path / "velocity" / "meta.json").write_text(json.dumps(meta))
    assert cascade_script.stale_dependents("pitch") == []
    (tmp_path / "pitch" / "policy.pt").write_bytes(b"retrained pitch policy")
    assert cascade_script.stale_dependents("pitch") == ["velocity"]
    assert cascade_script.stale_dependents("velocity") == []  # nothing is trained on velocity


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
    policy = load_frozen("pitch", "cpu")
    out = policy(torch.zeros(3, 7))
    assert out.shape == (3, 2) and torch.isfinite(out).all()
    leaning = policy(torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert leaning[0, 0] > 0.0 and leaning[0, 1] > 0.0  # a forward lean is answered with forward torque on both wheels
    meta = json.loads(path.with_name("meta.json").read_text())
    assert meta["stage"] == "pitch" and meta["evaluation"]["pass"] is True
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
    _write_policy(tmp_path, "pitch", 6, 2)  # the upright layout, not the 7 inputs of the pitch stage
    with pytest.raises(ValueError, match="obs/action size 6/2"):
        load_frozen("pitch", "cpu", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_cascade.py --stages velocity"):
        load_frozen("velocity", "cpu", tmp_path)


def test_load_frozen_warns_when_no_contract_was_recorded(tmp_path, capsys):
    _write_policy(tmp_path, "pitch", *STAGE_IO["pitch"])
    load_frozen("pitch", "cpu", tmp_path)
    assert "no contract recorded" in capsys.readouterr().out
    (tmp_path / "pitch" / "meta.json").write_text(json.dumps({"contract_sha256": contract_fingerprint("pitch")}))
    load_frozen("pitch", "cpu", tmp_path)
    assert capsys.readouterr().out == ""
    (tmp_path / "pitch" / "meta.json").write_text(json.dumps({"contract_sha256": "0" * 64}))
    load_frozen("pitch", "cpu", tmp_path)
    assert "its observation/action contract changed" in capsys.readouterr().out
