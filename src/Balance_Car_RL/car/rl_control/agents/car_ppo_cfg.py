"""PPO (rsl_rl) settings shared by the three tasks; each task file only changes what differs.

Actor and critic 64 x 64, observation normalisation inside the actor so the frozen network carries it along. The
experiment name is ``balance_car_<task>`` (``scripts/train_cascade.py`` and ``scripts/evaluate.py`` look for runs
there).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def experiment_name(task: str) -> str:
    return f"balance_car_{task}"


@configclass
class BoundedGaussianCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian action noise whose std is clamped (rsl_rl ``GaussianDistribution.std_range``). Copied from Drone_RL:
    there, without a bound the std of the first rate run grew from 0.2 to 8.5: the actions sat at the clip limits and
    the motors only switched."""

    std_range: tuple[float, float] = (0.02, 0.5)


def make_actor(init_std: float, std_range: tuple[float, float] = (0.02, 0.5)) -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BoundedGaussianCfg(init_std=init_std, std_range=std_range),
    )


@configclass
class CarPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    clip_actions = 1.0  # like Drone_RL: the env, and so ``last_action``, only ever see actions in [-1, 1]
    experiment_name = experiment_name("upright")
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = make_actor(init_std=0.5)
    critic = RslRlMLPModelCfg(hidden_dims=[64, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
