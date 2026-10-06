"""Upright task (``BalanceCar-Upright-v0``): keep the car upright for 10 s while random pushes hit it.

IMU pitch and rate, wheel speeds -> one torque per wheel. No command. Everything else is ``car_env_cfg.py``;
observation, action and reward are described in ``guide/04_training.md``.
"""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.rl_control.car_env_cfg import CarEnvCfg, ObservationsCfg, PolicyCfg, RewardsCfg


@configclass
class UprightPolicyCfg(PolicyCfg):
    """IMU pitch and rate, wheel speeds, last action (6)."""

    wheel_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
    last_action = ObsTerm(func=isaac_mdp.last_action)


@configclass
class UprightRewardsCfg(RewardsCfg):
    upright = RewTerm(
        func=mdp.upright_exp,
        weight=2.0,
        params={"std": 0.3},
    )
    wheel_vel = RewTerm(
        func=mdp.wheel_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )


@configclass
class UprightEnvCfg(CarEnvCfg):
    observations: ObservationsCfg = ObservationsCfg(policy=UprightPolicyCfg())
    rewards: UprightRewardsCfg = UprightRewardsCfg()
