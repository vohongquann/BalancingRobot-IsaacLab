"""PPO settings of the pitch task (``BalanceCar-Pitch-v0``, cascade stage 1); the rest is in ``car_ppo_cfg.py``."""

from isaaclab.utils import configclass

from Balance_Car_RL.car.rl_control.agents.car_ppo_cfg import CarPPORunnerCfg, experiment_name


@configclass
class PitchPPORunnerCfg(CarPPORunnerCfg):
    """Same network and settings as the upright task, own experiment folder."""

    experiment_name = experiment_name("pitch")
