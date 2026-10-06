"""PPO settings of the velocity task (``BalanceCar-Velocity-v0``, cascade stage 2); the rest is in
``car_ppo_cfg.py``."""

from isaaclab.utils import configclass

from Balance_Car_RL.car.rl_control.agents.car_ppo_cfg import CarPPORunnerCfg, experiment_name, make_actor


@configclass
class VelocityPPORunnerCfg(CarPPORunnerCfg):
    """The frozen pitch policy does the balancing, but the task needs more iterations."""

    max_iterations = 500
    experiment_name = experiment_name("velocity")
    actor = make_actor(init_std=0.5)
