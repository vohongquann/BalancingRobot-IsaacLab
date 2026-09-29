# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar command terms: a target pitch [rad] for the inner stage, a target forward speed [m/s] for the outer stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv


class ScalarCommand(CommandTerm):
    """One number per environment, drawn uniformly in ``[low, high]`` and held for a random time.

    With probability ``zero_probability`` the command is exactly zero, so the policy also practises staying still.
    Shape of the command: (num_envs, 1).

    With ``speed_guard`` set, the sign of the command is flipped whenever the car is faster than the guard and the
    command would speed it up further. A lean cannot be held forever (the wheels have a top speed), and the pitch
    commands of the outer stage never ask for that, so training does not ask for it either.
    """

    cfg: ScalarCommandCfg

    def __init__(self, cfg: ScalarCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        self._drawn = torch.zeros(self.num_envs, 1, device=self.device)
        self._robot = env.scene["robot"]

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        n = len(env_ids)
        value = torch.empty(n, device=self.device).uniform_(self.cfg.low, self.cfg.high)
        value = torch.where(
            torch.rand(n, device=self.device) < self.cfg.zero_probability, torch.zeros_like(value), value
        )
        self._drawn[env_ids, 0] = value
        self._command[env_ids, 0] = value

    def _update_command(self) -> None:
        guard = self.cfg.speed_guard
        if guard is None:
            return
        speed = self._robot.data.root_lin_vel_b.torch[:, :1]
        speeds_up = (speed.abs() > guard) & (torch.sign(self._drawn) == torch.sign(speed))
        self._command[:] = torch.where(speeds_up, -self._drawn, self._drawn)

    def _update_metrics(self) -> None:
        """No metrics are logged for this term."""


@configclass
class ScalarCommandCfg(CommandTermCfg):
    class_type: type = ScalarCommand
    low: float = -1.0
    """Lower bound of the command."""
    high: float = 1.0
    """Upper bound of the command."""
    zero_probability: float = 0.2
    """Share of the resamples that give exactly zero."""
    speed_guard: float | None = None
    """Forward speed [m/s] above which a command that speeds the car up is flipped. None: no guard."""
