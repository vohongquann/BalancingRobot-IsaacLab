"""Action of ``BalanceCar-Position-Gains-v0``: the network writes the six gains of the go-to-goal PID
(``pid_control/go_to_goal.py``), the PID writes the (v_x, w_z) command of the frozen velocity network.

    gains (6) -> go-to-goal PID on the goal (x, y, heading) -> (v_x, w_z) -> frozen velocity policy -> wheel torques

Action order: ``[kp, ki, kd]`` of the speed loop, then of the turn loop; each maps to a gain like the other gain tasks
(``mdp/gains.py``: ``nominal * 3^a``, or ``maximum * max(a, 0)`` for a gain that is 0 when tuned). The zero action is
the tuned go-to-goal PID. Kept in its own file (and the task in ``rl_control/position_gains_env_cfg.py``) because
``gains.py`` and ``gains_env_cfg.py`` are in the contract of the frozen pitch gain stage.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

from Balance_Car_RL.car.mdp.actions.velocity_action import FrozenVelocityAction, FrozenVelocityActionCfg
from Balance_Car_RL.car.mdp.gains import GainLayer
from Balance_Car_RL.car.mdp.navigation import pose_command_2d
from Balance_Car_RL.car.pid_control.go_to_goal import GOAL_INT_LIMIT, GOAL_SPEED_KP, GOAL_TURN_KP, GoToGoalPID

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

POSITION_GAIN_LAYERS = {
    "speed": GainLayer("goal_speed", GOAL_SPEED_KP, 0.0, 0.0, int_limit=GOAL_INT_LIMIT),
    "turn": GainLayer("goal_turn", GOAL_TURN_KP, 0.0, 0.0, int_limit=GOAL_INT_LIMIT),
}
"""The two loops of the go-to-goal PID, as gain layers (tuned gains, ranges)."""


class PositionGainLayer:
    """Go-to-goal PID with the gains of a network: goal ``[x, y, heading]`` (N, 3) and gain action (N, 6) -> (v_x, w_z)
    (N, 2) and the gains (N, 6). Pure torch."""

    def __init__(self, dt: float, speed_max: float, turn_max: float):
        self.pid = GoToGoalPID(dt, speed_max, turn_max)

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def step(self, gain_action: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        speed_gains = POSITION_GAIN_LAYERS["speed"].gains(gain_action[:, :3])
        turn_gains = POSITION_GAIN_LAYERS["turn"].gains(gain_action[:, 3:])
        self.pid.speed.kp, self.pid.speed.ki, self.pid.speed.kd = speed_gains
        self.pid.turn.kp, self.pid.turn.ki, self.pid.turn.kd = turn_gains
        speed, turn = self.pid.step(goal)
        return torch.stack([speed, turn], dim=1), torch.stack([*speed_gains, *turn_gains], dim=1)


class PositionGainAction(FrozenVelocityAction):
    """Six gains in, wheel torques out: the go-to-goal PID with the network's gains writes the velocity command, the
    frozen velocity policy turns it into torques (:class:`FrozenVelocityAction`)."""

    cfg: PositionGainActionCfg

    def __init__(self, cfg: PositionGainActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.layer = PositionGainLayer(env.step_dt, cfg.speed_scale, cfg.yaw_rate_scale)
        self._gain_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._gains = torch.zeros(self.num_envs, 6, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._gain_actions

    @property
    def gains(self) -> torch.Tensor:
        """The six gains now (N, 6); ``processed_actions`` stays the velocity command, as in the position task."""
        return self._gains

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.layer.reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._gain_actions[env_ids] = 0.0
        self._gains[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._gain_actions[:] = actions.clamp(-1.0, 1.0)
        goal = pose_command_2d(self._env, self.cfg.command_name, self.cfg.goal_max_distance)
        command, self._gains[:] = self.layer.step(self._gain_actions, goal)
        super().process_actions(command / self._scale)  # back to [-1, 1]: the velocity action scales it again


@configclass
class PositionGainActionCfg(FrozenVelocityActionCfg):
    class_type: type = PositionGainAction
    command_name: str = "pose_command"
    goal_max_distance: float = MISSING
    """The goal is seen shortened to this distance [m], like the observation of the position task."""
