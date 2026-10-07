"""Position gain task (``BalanceCar-Position-Gains-v0``): the network writes the six gains of the go-to-goal PID
(``pid_control/go_to_goal.py``), which writes the (v_x, w_z) command of the frozen velocity network.

Everything else is the position task (``position_env_cfg.py``: goal command, observation terms, rewards, resets), so the
three controllers of the position layer, PID (zero action), RL gains (this task) and RL (``BalanceCar-Position-v0``),
drive the same frozen velocity network on the same task. The action history in the observation is that of the six
gains: 13 values. In its own file because ``gains_env_cfg.py`` is in the contract of the frozen pitch gain stage.
"""

from isaaclab.utils import configclass

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.rl_control.position_env_cfg import FROZEN_VELOCITY, GOAL_OBS_MAX_M, PositionEnvCfg


@configclass
class PositionGainsActionsCfg:
    """The go-to-goal PID with the gains of the network, over the frozen velocity network."""

    position_gains = mdp.PositionGainActionCfg(**FROZEN_VELOCITY, goal_max_distance=GOAL_OBS_MAX_M)


@configclass
class PositionGainsEnvCfg(PositionEnvCfg):
    """Go-to-goal PID gains. Observation 13: IMU pitch and rate, speed, turn rate, goal x, y, heading, 6 last gains."""

    actions: PositionGainsActionsCfg = PositionGainsActionsCfg()
