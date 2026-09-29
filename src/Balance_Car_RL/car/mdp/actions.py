# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term of the outer stage: the action is a pitch target, a frozen inner policy turns it into wheel torques."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_CAR_DIR = pathlib.Path(__file__).resolve().parents[1]

FROZEN_DIR = _CAR_DIR / "rl_control" / "frozen"
"""Where the frozen policies of the finished stages live: ``<stage>/policy.pt`` (TorchScript), ``policy.onnx`` and
``meta.json``."""

_CONTRACT_FILES: dict[str, tuple[pathlib.Path, ...]] = {
    "pitch": (
        _CAR_DIR / "car_cfg.py",
        _CAR_DIR / "estimation.py",
        _CAR_DIR / "mdp" / "observations.py",
        _CAR_DIR / "mdp" / "commands.py",
        _CAR_DIR / "rl_control" / "common.py",
        _CAR_DIR / "rl_control" / "pitch_env_cfg.py",
    ),
}
"""Source files that define each stage's observation/action contract, for :func:`contract_fingerprint`."""


def frozen_policy_path(stage: str) -> pathlib.Path:
    """Path of the frozen TorchScript policy of a finished stage."""
    return FROZEN_DIR / stage / "policy.pt"


def contract_fingerprint(stage: str) -> str | None:
    """SHA-256 over the source files that define ``stage``'s observation/action contract, or ``None`` if ``stage``
    has no entry in :data:`_CONTRACT_FILES` (nothing currently depends on a frozen policy of that stage).

    Stored in ``frozen/<stage>/meta.json`` when a stage is frozen (``scripts/train_cascade.py``), and compared
    against the current files by :class:`FrozenPitchAction` and by ``train_cascade.py`` before a later stage trains
    on a frozen one. Neither of those files changing, nor a stage's checkpoint changing without its config, is
    detected by the checksum alone; they only catch source edits to the files listed in :data:`_CONTRACT_FILES`.
    """
    if stage not in _CONTRACT_FILES:
        return None
    digest = hashlib.sha256()
    for path in _CONTRACT_FILES[stage]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stage1_observation(
    imu: torch.Tensor, wheel_vel: torch.Tensor, last_action: torch.Tensor, pitch_target: torch.Tensor
) -> torch.Tensor:
    """Exactly the observation vector stage 1 (``rl_control/pitch_env_cfg.py``) was trained on: ``[imu, wheel_vel,
    last_action, pitch_target]``. A free function so ``tests/test_cascade.py`` can check the order without a
    simulator: :class:`FrozenPitchAction` cannot be constructed outside one (it needs a live ``env.scene``)."""
    return torch.cat([imu, wheel_vel, last_action, pitch_target], dim=1)


class FrozenPitchAction(ActionTerm):
    """Pitch target in, wheel torque out, through a frozen stage-1 policy.

    The action ``a`` (shape (num_envs, 1), in [-1, 1]) is scaled to a pitch target ``a * pitch_scale`` [rad]. The frozen
    policy of stage ``pitch`` sees exactly what it saw in training, ``[pitch, pitch rate, left wheel speed, right wheel
    speed, its previous action x2, pitch target]``, and returns the wheel actions, which are scaled by the motor stall
    torque like in stage 1. Its weights are never updated; only the outer policy learns.

    The IMU estimate is the one the environment's observation already computed this step (``env.car_imu``), so both
    stages read the same sensor.
    """

    cfg: FrozenPitchActionCfg

    def __init__(self, cfg: FrozenPitchActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        path = frozen_policy_path(cfg.stage)
        if not path.is_file():
            raise FileNotFoundError(
                f"No frozen policy for stage '{cfg.stage}' at {path}. Train and freeze it first: "
                f"`python scripts/train_cascade.py --stages {cfg.stage}`."
            )
        self._inner = torch.jit.load(str(path), map_location=self.device).eval()
        for p in self._inner.parameters():
            p.requires_grad_(False)
        joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True, as_proxy=True)
        self._joint_ids = joint_ids.torch
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        self._target = torch.zeros(self.num_envs, 1, device=self.device)
        self._inner_last = torch.zeros(self.num_envs, 2, device=self.device)
        self._torque = torch.zeros(self.num_envs, 2, device=self.device)
        self.meta = json.loads(path.with_name("meta.json").read_text()) if path.with_name("meta.json").is_file() else {}
        expected = self.meta.get("contract_sha256")
        if expected and expected != contract_fingerprint(cfg.stage):
            print(
                f"[WARN] Frozen policy '{cfg.stage}' at {path} looks stale: its observation/action contract changed "
                f"since it was frozen. Retrain and refreeze: `python scripts/train_cascade.py --stages {cfg.stage}`."
            )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._target

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._target[ids] = 0.0
        self._inner_last[ids] = 0.0
        self._torque[ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        target = torch.clamp(actions, -1.0, 1.0) * self.cfg.pitch_scale
        if self.cfg.speed_guard is not None:
            # stage 1 never trained on a same-direction target above this speed (mdp/commands.py, speed_guard):
            # zero it instead, rather than feed the frozen policy an input it has never seen
            speed = self._asset.data.root_lin_vel_b.torch[:, :1]
            out_of_distribution = (speed.abs() > self.cfg.speed_guard) & (torch.sign(target) == torch.sign(speed))
            target = torch.where(out_of_distribution, torch.zeros_like(target), target)
        self._target[:] = target
        imu = self._env.car_imu.cache  # [pitch, pitch rate] from this step's observation
        wheel_vel = self._asset.data.joint_vel.torch[:, self._joint_ids]
        inner_obs = _stage1_observation(imu, wheel_vel, self._inner_last, self._target)
        with torch.inference_mode():
            inner_action = self._inner(inner_obs)
        self._inner_last[:] = inner_action
        self._torque[:] = inner_action * self.cfg.torque_scale

    def apply_actions(self) -> None:
        self._asset.set_joint_effort_target_index(target=self._torque, joint_ids=self._joint_ids)


@configclass
class FrozenPitchActionCfg(ActionTermCfg):
    class_type: type = FrozenPitchAction
    joint_names: list[str] = [".*_wheel_joint"]
    stage: str = "pitch"
    """Name of the frozen stage whose policy is used (``rl_control/frozen/<stage>/policy.pt``)."""
    pitch_scale: float = 0.08
    """Pitch target [rad] for an action of 1. Must equal the frozen stage's own ``PITCH_TARGET_MAX_RAD``; there is no
    automatic link between the two, only the check in ``tests/test_cascade.py``. The env config that uses this term
    always passes it explicitly (``velocity_env_cfg.py``) -- this default is only a fallback, kept equal to the
    current stage-1 range so an omission fails a test rather than silently training out of range."""
    torque_scale: float = 0.20917
    """Torque [N m] for an inner action of 1: the wheel stall torque (``car_cfg.WHEEL_STALL_TORQUE_NM``), which the
    env config passes explicitly; see ``pitch_scale`` above for why this default is kept in sync anyway."""
    speed_guard: float | None = None
    """Forward speed [m/s] above which a target that would speed the car up further is zeroed instead of applied:
    the same guard the frozen stage's own command was trained with (``mdp/commands.py::ScalarCommand``), so the
    outer stage cannot ask it for a lean it has never seen. ``None``: no guard."""
