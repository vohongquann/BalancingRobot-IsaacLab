# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BalanceCar-Pitch-v0, stage 1 of the cascade: hold a commanded lean while random pushes hit the car.

A leaning car accelerates, so a pitch target is a way to ask for acceleration; the outer stage (``velocity_env_cfg.py``)
uses this. A lean cannot be held for long: the wheels reach 0.94 m/s at no load, and at that speed the motors have no
torque left. The target is therefore small, changes every 0.5 to 1.5 s and is flipped when the car is already fast
(``mdp/commands.py``), like the output of a speed controller. Details: ``docs/04-training.md``.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM

from .common import BalanceCarEventCfg, BalanceCarSceneCfg, BalanceCarTerminationsCfg, configure_car_sim

SPEED_GUARD_M_S = 0.5
"""Forward speed above which the pitch command is flipped so it slows the car down (see ``mdp/commands.py``)."""

PITCH_TARGET_MAX_RAD = 0.08
"""Largest commanded lean [rad]: about 0.8 m/s^2 of acceleration, and the range the outer stage is allowed to use."""


@configclass
class CommandsCfg:
    target = mdp.ScalarCommandCfg(
        resampling_time_range=(0.5, 1.5),
        low=-PITCH_TARGET_MAX_RAD,
        high=PITCH_TARGET_MAX_RAD,
        zero_probability=0.2,
        speed_guard=SPEED_GUARD_M_S,
    )


@configclass
class ActionsCfg:
    """One normalized torque per wheel; scale (the motor stall torque) converts it to N m."""

    wheel_torque = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*_wheel_joint"], scale=WHEEL_STALL_TORQUE_NM
    )


@configclass
class ObservationsCfg:
    """The stage-1 policy sees IMU pitch and rate, wheel speeds, its last action and the pitch target (7 values).

    The order is fixed: the outer stage rebuilds this vector for the frozen policy (``mdp/actions.py``).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        imu = ObsTerm(func=mdp.ImuPitchAndRate, params={"sensor_cfg": SceneEntityCfg("imu")})  # [pitch, pitch rate]
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        pitch_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "target"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)
    track_pitch = RewTerm(func=mdp.pitch_track_exp, weight=2.0, params={"command_name": "target", "std": 0.05})
    pitch_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    wheel_vel = RewTerm(
        func=mdp.wheel_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class BalanceCarPitchEnvCfg(ManagerBasedRLEnvCfg):
    """Hold a commanded pitch by driving the wheels."""

    scene: BalanceCarSceneCfg = BalanceCarSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: BalanceCarEventCfg = BalanceCarEventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: BalanceCarTerminationsCfg = BalanceCarTerminationsCfg()

    def __post_init__(self) -> None:
        configure_car_sim(self)
