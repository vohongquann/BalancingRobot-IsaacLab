"""PPO settings of the velocity task (``BalanceCar-Velocity-v0``, RL cascade stage 1); the rest is in
``car_ppo_cfg.py``.

From Isaac Lab's flat-terrain locomotion (``AnymalDFlatPPORunnerWithSymmetryCfg``): wider network, symmetry
augmentation. Kept from this project: observation normalization (the frozen network carries it along) and the bounded
action noise (with an std of 0.5 two independent wheel torques made the cars chatter and spin).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from Balance_Car_RL.car.rl_control.agents.car_ppo_cfg import BoundedGaussianCfg, CarPPORunnerCfg, experiment_name
from Balance_Car_RL.car.rl_control.agents.symmetry import symmetry_cfg, velocity_symmetry

HIDDEN = [128, 128]
"""Actor and critic layers: balancing, speed and turning in one network (Isaac Lab flat locomotion: 128 x 3)."""


@configclass
class VelocityPPORunnerCfg(CarPPORunnerCfg):
    max_iterations = 600
    experiment_name = experiment_name("velocity")
    actor = RslRlMLPModelCfg(
        hidden_dims=HIDDEN,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BoundedGaussianCfg(init_std=0.3, std_range=(0.02, 0.3)),
    )
    critic = RslRlMLPModelCfg(hidden_dims=HIDDEN, activation="elu", obs_normalization=True)

    def __post_init__(self):
        self.algorithm.symmetry_cfg = symmetry_cfg(velocity_symmetry)
