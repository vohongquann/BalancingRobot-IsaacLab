# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BalanceCar-Upright-v0: keep the two-wheeled car upright for 10 s while random pushes hit it.

Observation, action and reward are described in ``docs/04-training.md``.
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


@configclass
class ActionsCfg:
    """The policy outputs a normalized torque for each wheel; scale (the motor stall torque) converts it to N m."""

    wheel_torque = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*_wheel_joint"], scale=WHEEL_STALL_TORQUE_NM
    )


@configclass
class ObservationsCfg:
    """Everything below is measurable on the real car: a raw IMU (filtered to pitch) and the wheel encoders."""

    @configclass
    class PolicyCfg(ObsGroup):
        imu = ObsTerm(func=mdp.ImuPitchAndRate, params={"sensor_cfg": SceneEntityCfg("imu")})  # [pitch, pitch rate]
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)
    upright = RewTerm(func=mdp.upright_exp, weight=2.0, params={"std": 0.3})
    pitch_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    wheel_vel = RewTerm(
        func=mdp.wheel_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class BalanceCarEnvCfg(ManagerBasedRLEnvCfg):
    """Keep a two-wheeled car upright by driving its wheels."""

    scene: BalanceCarSceneCfg = BalanceCarSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: BalanceCarEventCfg = BalanceCarEventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: BalanceCarTerminationsCfg = BalanceCarTerminationsCfg()

    def __post_init__(self) -> None:
        configure_car_sim(self)
