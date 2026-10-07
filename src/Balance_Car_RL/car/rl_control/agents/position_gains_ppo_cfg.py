"""PPO settings of the position gain task (``BalanceCar-Position-Gains-v0``): the other gain tasks' small gain noise
(``gains_ppo_cfg.gains_actor``), the discount of the position task (``position_ppo_cfg.py``), no mirror symmetry (gains
do not change sign in a mirror)."""

from isaaclab.utils import configclass

from Balance_Car_RL.car.rl_control.agents.car_ppo_cfg import CarPPORunnerCfg, experiment_name
from Balance_Car_RL.car.rl_control.agents.gains_ppo_cfg import gains_actor


@configclass
class PositionGainsPPORunnerCfg(CarPPORunnerCfg):
    max_iterations = 400
    experiment_name = experiment_name("position") + "_gains"
    actor = gains_actor()

    def __post_init__(self):
        self.algorithm.gamma = 0.995
