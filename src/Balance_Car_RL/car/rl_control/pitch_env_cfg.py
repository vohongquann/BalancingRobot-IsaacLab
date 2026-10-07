"""Pitch layer of the gain cascade (``BalanceCar-Pitch-Gains-v0``, ``gains_env_cfg.py``): hold a commanded lean while
random pushes hit the car.

IMU pitch and rate, wheel speeds, pitch target -> wheel torque. A leaning car accelerates, so a pitch target is a way
to ask for acceleration; the speed layer above (``gains_env_cfg.py``) uses this. A lean cannot be held for long: the
wheels reach 0.94 m/s at no load, and at that speed the motors have no torque left. The target is therefore small,
changes every 0.5 to 1.5 s and is flipped when the car is already fast (``mdp/commands.py``), like the output of a speed
controller. Everything else is ``car_env_cfg.py``; details: ``guide/04_training.md``.
"""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.rl_control.car_env_cfg import CarEnvCfg, ObservationsCfg, PolicyCfg, RewardsCfg

SPEED_GUARD_M_S = 0.5
"""Forward speed above which the pitch command is flipped so it slows the car down (see ``mdp/commands.py``)."""

PITCH_TARGET_MAX_RAD = 0.08
"""Largest commanded lean [rad]: about 0.8 m/s^2 of acceleration, and the range the stage above is allowed to use."""


@configclass
class PitchCommandsCfg:
    target = mdp.ScalarCommandCfg(
        resampling_time_range=(0.5, 1.5),
        low=-PITCH_TARGET_MAX_RAD,
        high=PITCH_TARGET_MAX_RAD,
        zero_probability=0.2,
        speed_guard=SPEED_GUARD_M_S,
    )


@configclass
class PitchPolicyCfg(PolicyCfg):
    """IMU pitch and rate, wheel speeds, last action, pitch target (7).

    The order is fixed: the stage above rebuilds this vector for the frozen policy (``mdp/actions/pitch_action.py``).
    """

    wheel_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
    last_action = ObsTerm(func=isaac_mdp.last_action)
    pitch_target = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "target"})


@configclass
class PitchRewardsCfg(RewardsCfg):
    pitch = RewTerm(
        func=mdp.pitch_error_exp,
        weight=2.0,
        params={"command_name": "target", "std": 0.05},
    )
    wheel_vel = RewTerm(
        func=isaac_mdp.joint_vel_l2,  # squared wheel speed: the car should not drive away while balancing
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )


@configclass
class PitchEnvCfg(CarEnvCfg):
    commands: PitchCommandsCfg = PitchCommandsCfg()
    observations: ObservationsCfg = ObservationsCfg(policy=PitchPolicyCfg())
    rewards: PitchRewardsCfg = PitchRewardsCfg()
