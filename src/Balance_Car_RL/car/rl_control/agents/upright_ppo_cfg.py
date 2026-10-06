"""PPO settings of the upright task (``BalanceCar-Upright-v0``); the rest is in ``car_ppo_cfg.py``."""

from isaaclab.utils import configclass

from Balance_Car_RL.car.rl_control.agents.car_ppo_cfg import CarPPORunnerCfg, experiment_name


@configclass
class UprightPPORunnerCfg(CarPPORunnerCfg):
    experiment_name = experiment_name("upright")
