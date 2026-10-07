"""PPO settings of the gain tasks (``BalanceCar-<Stage>-Gains-v0``): the shared ones of ``car_ppo_cfg.py`` with the
action noise of a gain schedule. The gains act through a PID, so the noise of a gain (the std is in the exponent of
3^a) stays small: 0.3 at most. Each task has its own log folder ``balance_car_<stage>_gains``."""

from isaaclab.utils import configclass

from Balance_Car_RL.car.rl_control.agents.car_ppo_cfg import CarPPORunnerCfg, experiment_name, make_actor


def gains_actor():
    return make_actor(init_std=0.2, std_range=(0.02, 0.3))


@configclass
class PitchGainsPPORunnerCfg(CarPPORunnerCfg):
    experiment_name = experiment_name("pitch") + "_gains"
    actor = gains_actor()


@configclass
class VelocityGainsPPORunnerCfg(CarPPORunnerCfg):
    """The speed layer needs more iterations than the pitch layer."""

    max_iterations = 500
    experiment_name = experiment_name("velocity") + "_gains"
    actor = gains_actor()
