"""A finished stage that no longer learns: the TorchScript actor that ``play`` exports, frozen by
``scripts/train_cascade.py`` into ``rl_control/frozen/<stage>/`` (``policy.pt``, ``policy.onnx``, ``meta.json``).

Pure torch: no Isaac needed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

import torch

from Balance_Car_RL.assets import BALANCE_CAR_RL_ASSETS_DIR

_CAR_DIR = pathlib.Path(__file__).resolve().parents[2]
_DATA_DIR = BALANCE_CAR_RL_ASSETS_DIR / "balboa"

FROZEN_DIR = _CAR_DIR / "rl_control" / "frozen"
"""Where the frozen policies of the finished stages live: ``<stage>/policy.pt`` (TorchScript), ``policy.onnx`` and
``meta.json``."""

STAGE_IO = {
    "velocity": (8, 2),
    "position": (9, 2),
    "pitch_gains": (8, 3),
    "velocity_gains": (7, 3),
    "position_gains": (13, 6),
}
"""Observation and action size of the policy of each stage (``rl_control/<stage>_env_cfg.py``; the gain stages are
``rl_control/gains_env_cfg.py``, their action is the three PID gains)."""

_CONTRACT_FILES: dict[str, tuple[pathlib.Path, ...]] = {
    "velocity": (
        # the robot: constants, model, mounting of the IMU
        _CAR_DIR / "car_cfg.py",
        _DATA_DIR / "balboa.urdf",
        _DATA_DIR / "imu_mount.json",
        _DATA_DIR / "meshes" / "body_collision.stl",
        # what the policy sees
        _CAR_DIR / "estimation.py",
        _CAR_DIR / "mdp" / "observations.py",
        _CAR_DIR / "mdp" / "locomotion.py",
        # what its output means
        _CAR_DIR / "mdp" / "actions" / "wheel_action.py",
        # what it is trained on and for
        _CAR_DIR / "mdp" / "rewards.py",
        _CAR_DIR / "rl_control" / "car_env_cfg.py",
        _CAR_DIR / "rl_control" / "velocity_env_cfg.py",
        _CAR_DIR / "rl_control" / "agents" / "car_ppo_cfg.py",
        _CAR_DIR / "rl_control" / "agents" / "velocity_ppo_cfg.py",
        _CAR_DIR / "rl_control" / "agents" / "symmetry.py",
    ),
    "pitch_gains": (
        # the robot and what the policy sees: those of the pitch stage
        _CAR_DIR / "car_cfg.py",
        _DATA_DIR / "balboa.urdf",
        _DATA_DIR / "imu_mount.json",
        _DATA_DIR / "meshes" / "body_collision.stl",
        _CAR_DIR / "estimation.py",
        _CAR_DIR / "mdp" / "observations.py",
        _CAR_DIR / "mdp" / "commands.py",
        _CAR_DIR / "mdp" / "rewards.py",
        _CAR_DIR / "rl_control" / "car_env_cfg.py",
        _CAR_DIR / "rl_control" / "pitch_env_cfg.py",
        # what the gains mean: the PID they go into, and how
        _CAR_DIR / "pid_control" / "pid.py",
        _CAR_DIR / "pid_control" / "cascade_pid.py",
        _CAR_DIR / "mdp" / "gains.py",
        _CAR_DIR / "mdp" / "actions" / "gain_action.py",
        # what it is trained on and for
        _CAR_DIR / "rl_control" / "gains_env_cfg.py",
        _CAR_DIR / "rl_control" / "agents" / "car_ppo_cfg.py",
        _CAR_DIR / "rl_control" / "agents" / "gains_ppo_cfg.py",
    ),
}
"""Source files that define each stage's observation/action contract and training, for :func:`contract_fingerprint`.
Only the stages another stage is trained on need an entry."""


def _normalized(path: pathlib.Path) -> bytes:
    """What of a file counts for the fingerprint: Python files as their syntax tree without docstrings and comments
    (so editing a comment does not make a policy look stale), everything else byte for byte."""
    if path.suffix != ".py":
        return path.read_bytes()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):  # a string statement is a docstring, also the ones under an assignment
            node.body = [
                s
                for s in body
                if not (
                    isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)
                )
            ] or [ast.Pass()]
    return ast.dump(tree).encode()


def frozen_policy_path(stage: str, frozen_dir: str | pathlib.Path = FROZEN_DIR) -> pathlib.Path:
    """Path of the frozen TorchScript policy of a finished stage."""
    return pathlib.Path(frozen_dir) / stage / "policy.pt"


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_fingerprint(stage: str) -> str | None:
    """SHA-256 over the files that define ``stage``'s observation/action contract and training, or ``None`` if ``stage``
    has no entry in :data:`_CONTRACT_FILES` (nothing is trained on a frozen policy of that stage).

    Stored in ``frozen/<stage>/meta.json`` when a stage is frozen (``scripts/train_cascade.py``), and compared
    against the current files by :func:`load_frozen` and by ``train_cascade.py`` before a later stage trains on a
    frozen one. A stage trained on another stage records the hash of that stage's ``policy.pt`` too
    (``needs_sha256``), so retraining the lower stage is noticed in the upper one.
    """
    if stage not in _CONTRACT_FILES:
        return None
    digest = hashlib.sha256()
    for path in _CONTRACT_FILES[stage]:
        digest.update(path.name.encode() + b"\0" + _normalized(path))
    return digest.hexdigest()


def load_frozen(stage: str, device, frozen_dir: str | pathlib.Path = FROZEN_DIR) -> torch.jit.ScriptModule:
    """The frozen network of ``stage`` (observation -> action, normaliser inside), without gradients; checks that it
    still matches the stage definition of :data:`STAGE_IO` and warns when its contract changed since it was frozen."""
    path = frozen_policy_path(stage, frozen_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"No frozen policy for stage '{stage}' at {path}. Train and freeze it first: "
            f"`python scripts/train_cascade.py --stages {stage}`."
        )
    policy = torch.jit.load(str(path), map_location=device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    weights = [v for k, v in policy.state_dict().items() if k.startswith("mlp.") and k.endswith(".weight")]
    obs_dim, action_dim = weights[0].shape[1], weights[-1].shape[0]
    if (obs_dim, action_dim) != STAGE_IO[stage]:
        raise ValueError(
            f"Frozen '{stage}' policy ({path}) has obs/action size {obs_dim}/{action_dim}, but the stage now has "
            f"{STAGE_IO[stage][0]}/{STAGE_IO[stage][1]}: retrain and refreeze it (`python scripts/train_cascade.py "
            f"--stages {stage}`)."
        )
    meta_path = path.with_name("meta.json")
    expected = json.loads(meta_path.read_text()).get("contract_sha256") if meta_path.is_file() else None
    if contract_fingerprint(stage) and expected != contract_fingerprint(stage):
        reason = "its observation/action contract changed since it was frozen" if expected else "no contract recorded"
        print(
            f"[WARN] Frozen policy '{stage}' at {path} looks stale ({reason}). Retrain and refreeze: "
            f"`python scripts/train_cascade.py --stages {stage}`."
        )
    return policy
