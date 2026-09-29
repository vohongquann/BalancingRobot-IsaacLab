# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BalanceCar-Velocity-v0, stage 2 of the cascade: track a commanded forward speed.

The policy outputs a pitch target; the frozen stage-1 policy (``rl_control/frozen/pitch/policy.pt``) turns it into wheel
torques. Only the outer policy learns. Observation, action and reward are described in ``docs/04-training.md``.
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
from .pitch_env_cfg import PITCH_TARGET_MAX_RAD, SPEED_GUARD_M_S

SPEED_MAX_M_S = 0.4
"""Largest commanded speed [m/s]; the wheels can reach 0.94 m/s (no-load speed 23.6 rad/s x 0.04 m)."""


@configclass
class CommandsCfg:
    target = mdp.ScalarCommandCfg(
        resampling_time_range=(3.0, 6.0), low=-SPEED_MAX_M_S, high=SPEED_MAX_M_S, zero_probability=0.2
    )


@configclass
class ActionsCfg:
    """One number in [-1, 1]: the pitch target as a share of the range the frozen stage was trained on."""

    pitch_target = mdp.FrozenPitchActionCfg(
        asset_name="robot",
        stage="pitch",
        pitch_scale=PITCH_TARGET_MAX_RAD,
        torque_scale=WHEEL_STALL_TORQUE_NM,
        speed_guard=SPEED_GUARD_M_S,
    )


@configclass
class ObservationsCfg:
    """IMU pitch and rate, forward speed from the encoders, the speed command and the last action (5 values)."""

    @configclass
    class PolicyCfg(ObsGroup):
        imu = ObsTerm(func=mdp.ImuPitchAndRate, params={"sensor_cfg": SceneEntityCfg("imu")})  # [pitch, pitch rate]
        speed = ObsTerm(
            func=mdp.wheel_speed_estimate, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])}
        )
        speed_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "target"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)
    track_speed = RewTerm(func=mdp.speed_track_exp, weight=2.0, params={"command_name": "target", "std": 0.2})
    speed_error = RewTerm(func=mdp.speed_error_l1, weight=-1.0, params={"command_name": "target"})
    upright = RewTerm(func=mdp.upright_exp, weight=0.5, params={"std": 0.3})
    pitch_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)


@configclass
class BalanceCarVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Track a forward speed by commanding a pitch to the frozen balancing policy."""

    scene: BalanceCarSceneCfg = BalanceCarSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: BalanceCarEventCfg = BalanceCarEventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: BalanceCarTerminationsCfg = BalanceCarTerminationsCfg()

    def __post_init__(self) -> None:
        configure_car_sim(self)
