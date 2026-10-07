"""PPO settings of the position task (``BalanceCar-Position-v0``, RL cascade stage 2): those of the velocity task
(``velocity_ppo_cfg.py``) with the position task's mirror symmetry and discount.

From Isaac Lab's navigation task (``NavigationEnvPPORunnerCfg``): 128 x 128. Its policy runs at 5 Hz; this one at 50 Hz,
so the discount is 0.995 instead of 0.99 to look about as far ahead in seconds (4 s: a goal 1.4 m away at 0.4 m/s).
"""

from isaaclab.utils import configclass

from Balance_Car_RL.car.rl_control.agents.car_ppo_cfg import experiment_name
from Balance_Car_RL.car.rl_control.agents.symmetry import position_symmetry, symmetry_cfg
from Balance_Car_RL.car.rl_control.agents.velocity_ppo_cfg import VelocityPPORunnerCfg


@configclass
class PositionPPORunnerCfg(VelocityPPORunnerCfg):
    experiment_name = experiment_name("position")

    def __post_init__(self):
        self.algorithm.gamma = 0.995
        self.algorithm.symmetry_cfg = symmetry_cfg(position_symmetry)
