"""Velocity task (``BalanceCar-Velocity-v0``), stage 2 of the cascade: track a commanded forward speed.

IMU pitch and rate, speed, speed target -> pitch target. The frozen stage-1 policy (``frozen/pitch/policy.pt``) turns
the pitch target into wheel torques (``mdp.FrozenPitchAction``); only this policy learns. Everything else is
``car_env_cfg.py``; observation, action and reward are described in ``guide/04_training.md``.
"""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM
from Balance_Car_RL.car.rl_control.car_env_cfg import CarEnvCfg, ObservationsCfg, PolicyCfg, RewardsCfg
from Balance_Car_RL.car.rl_control.pitch_env_cfg import PITCH_TARGET_MAX_RAD, SPEED_GUARD_M_S

SPEED_MAX_M_S = 0.4
"""Largest commanded speed [m/s]; the wheels can reach 0.94 m/s (no-load speed 23.6 rad/s x 0.04 m)."""


@configclass
class VelocityCommandsCfg:
    target = mdp.ScalarCommandCfg(
        resampling_time_range=(3.0, 6.0),
        low=-SPEED_MAX_M_S,
        high=SPEED_MAX_M_S,
        zero_probability=0.2,
    )


@configclass
class VelocityActionsCfg:
    """One number in [-1, 1]: the pitch target as a share of the range the frozen stage was trained on."""

    pitch_target = mdp.FrozenPitchActionCfg(
        asset_name="robot",
        stage="pitch",
        pitch_scale=PITCH_TARGET_MAX_RAD,
        torque_scale=WHEEL_STALL_TORQUE_NM,
        speed_guard=SPEED_GUARD_M_S,
    )


@configclass
class VelocityPolicyCfg(PolicyCfg):
    """IMU pitch and rate, forward speed from the encoders, speed target, last action (5)."""

    speed = ObsTerm(
        func=mdp.wheel_speed_estimate,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )
    speed_target = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "target"})
    last_action = ObsTerm(func=isaac_mdp.last_action)


@configclass
class VelocityRewardsCfg(RewardsCfg):
    speed = RewTerm(
        func=mdp.speed_error_exp,
        weight=2.0,
        params={"command_name": "target", "std": 0.2},
    )
    speed_error = RewTerm(
        func=mdp.speed_error_l1,
        weight=-1.0,
        params={"command_name": "target"},
    )
    upright = RewTerm(
        func=mdp.upright_exp,
        weight=0.5,
        params={"std": 0.3},
    )
    output_change = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=-0.05,
    )


@configclass
class VelocityEnvCfg(CarEnvCfg):
    commands: VelocityCommandsCfg = VelocityCommandsCfg()
    actions: VelocityActionsCfg = VelocityActionsCfg()
    observations: ObservationsCfg = ObservationsCfg(policy=VelocityPolicyCfg())
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()
