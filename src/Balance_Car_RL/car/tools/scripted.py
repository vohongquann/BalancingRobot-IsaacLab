"""A task env driven by a fixed program instead of the random commands of training: what ``compare_layers.py``,
``record_demo.py`` and ``scripts/evaluate.py`` share. Import it after the app is launched (it imports Isaac Lab).

    scripted_cfg(cfg, length_s)   nominal robot, standing still at the start, no random pushes, no ends
    Command(env).set(obs, value)  writes a command into the command term and into the policy observation
    with_drive_action(cfg)        a velocity or position task driven by [common, turn] torques (classical drive)
    ClassicalDrive(env, balance)  LQR or PID in place of the velocity or position network: obs -> [common, turn]
"""

from __future__ import annotations

import numpy as np
import torch

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.pid_control import DrivePosition, DriveVelocity
from Balance_Car_RL.car.rl_control.position_env_cfg import GOAL_OBS_MAX_M
from Balance_Car_RL.car.rl_control.velocity_env_cfg import SPEED_MAX_M_S, YAW_RATE_MAX_RAD_S, VelocityActionsCfg


def program_value(program, t: float):
    """Value of a step program ``[(start time [s], value), ...]`` at time ``t``: held between the keys."""
    return next(v for start, v in reversed(program) if t >= start)


def scripted_cfg(cfg, length_s: float, start_pitch: float = 0.0):
    """``cfg`` for a scripted run of ``length_s`` seconds: starts still and facing +x (tilted by ``start_pitch``),
    nominal robot (no startup randomization), no random pushes, never ends, no command markers."""
    cfg.episode_length_s = length_s + 5.0
    cfg.events.reset_base.params["pose_range"] = {"pitch": (start_pitch, start_pitch)}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_wheels.params["velocity_range"] = (0.0, 0.0)
    cfg.events.push = None
    for name in ("wheel_friction", "body_mass", "body_com"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    for term in getattr(getattr(cfg, "commands", None), "__dict__", {}).values():
        if hasattr(term, "debug_vis"):
            term.debug_vis = False
    cfg.terminations.time_out = None
    cfg.terminations.fallen = None
    return cfg


def term_slice(env, name: str) -> slice:
    """Columns of the observation term ``name`` in the policy observation."""
    names = env.observation_manager.active_terms["policy"]
    dims = [int(np.prod(d)) for d in env.observation_manager.group_obs_term_dim["policy"]]
    start = sum(dims[: names.index(name)])
    return slice(start, start + dims[names.index(name)])


class Command:
    """Writes the program into the command term of an environment and into the policy observation, so the controller
    sees exactly the program (the random resampling of training is switched off)."""

    def __init__(self, env):
        self.env = env
        terms = env.command_manager.active_terms
        self.kind = "target" if "target" in terms else "base_velocity" if "base_velocity" in terms else "pose_command"
        self.term = env.command_manager.get_term(self.kind)
        self.term._resample_command = lambda env_ids: None
        if self.kind != "pose_command":
            self.term._update_command = lambda: None  # (the pose command keeps it: goal in the car's frame)
        names = env.observation_manager.active_terms["policy"]
        obs_term = {"base_velocity": "velocity_command", "pose_command": "pose_command"}.get(self.kind)
        if obs_term is None:
            obs_term = "pitch_target" if "pitch_target" in names else "speed_target"
        self.columns = term_slice(env, obs_term)
        self.start_xy = env.scene["robot"].data.root_pos_w.torch[:, :2].clone()

    def set(self, obs: torch.Tensor, value) -> None:
        """``value``: pitch or speed (gain tasks); ``(v_x, w_z)`` (velocity task); ``(x, y, heading)`` relative to the
        start (position task)."""
        if self.kind == "target":
            self.term._command[:] = value
            self.term._drawn[:] = value
            obs[:, self.columns] = value
        elif self.kind == "base_velocity":
            self.term.vel_command_b[:] = torch.tensor([value[0], 0.0, value[1]], device=self.env.device)
            obs[:, self.columns] = self.term.vel_command_b[:, [0, 2]]
        else:
            x, y, heading = value
            self.term.pos_command_w[:, :2] = self.start_xy + torch.tensor([x, y], device=self.env.device)
            self.term.heading_command_w[:] = heading
            self.term._update_command()
            obs[:, self.columns] = mdp.pose_command_2d(self.env, "pose_command", GOAL_OBS_MAX_M)


def with_drive_action(cfg):
    """``cfg`` (velocity or position task) whose action is the ``[common, turn]`` torque of the velocity task, for
    :class:`ClassicalDrive`: the position task's own action is the frozen velocity network."""
    cfg.actions = VelocityActionsCfg()
    return cfg


class ClassicalDrive:
    """``controller(obs) -> [common, turn]``: the classical counterpart of the velocity network (velocity task) or of
    the position network over it (position task), ``pid_control/drive.py``, on the policy observation of ``env``.
    ``balance`` is an :class:`~Balance_Car_RL.car.lqr_control.LQRController` or a
    :class:`~Balance_Car_RL.car.pid_control.CascadePID`. The env must take :func:`with_drive_action`."""

    def __init__(self, env, balance):
        position = "pose_command" in env.observation_manager.active_terms["policy"]
        target = "pose_command" if position else "velocity_command"
        if position:
            self.drive = DrivePosition(balance, env.step_dt, SPEED_MAX_M_S, YAW_RATE_MAX_RAD_S)
        else:
            self.drive = DriveVelocity(balance, env.step_dt)
        self.columns = [term_slice(env, name) for name in ("imu", "speed", "yaw_rate", target)]

    def reset(self, env_ids=None) -> None:
        self.drive.reset(env_ids)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        imu, speed, turn_rate, target = (obs[:, c] for c in self.columns)
        return self.drive.step(imu[:, 0], imu[:, 1], speed[:, 0], turn_rate[:, 0], target)

    @property
    def command(self) -> torch.Tensor | None:
        """The ``(v_x, w_z)`` the go-to-goal PID asked for on the last step (position task; None before it)."""
        return getattr(self.drive, "command", None)
