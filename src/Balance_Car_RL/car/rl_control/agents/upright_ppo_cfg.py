"""PPO settings of the upright task (``BalanceCar-Upright-v0``); the rest is in ``car_ppo_cfg.py``."""

from isaaclab.utils import configclass

from Balance_Car_RL.car.rl_control.agents.car_ppo_cfg import CarPPORunnerCfg, experiment_name, make_actor


@configclass
class UprightPPORunnerCfg(CarPPORunnerCfg):
    """Less action noise than the shared default, as for the RL cascade (``velocity_ppo_cfg.py``): with the std at 0.5
    the run of 2026-10-07 spun at 1.6 rad/s RMS with a mean torque of 0.13 N m."""

    experiment_name = experiment_name("upright")
    actor = make_actor(init_std=0.3, std_range=(0.02, 0.3))
