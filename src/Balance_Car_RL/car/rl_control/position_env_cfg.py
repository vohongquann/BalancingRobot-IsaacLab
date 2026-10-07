"""Position task (``BalanceCar-Position-v0``), stage 2 of the RL cascade: drive to a goal on the floor, face a heading.

Isaac Lab's navigation task (``isaaclab_tasks/contrib/navigation/config/anymal_c/navigation_env_cfg.py``) for the car:
the policy writes the (v_x, w_z) command of the frozen velocity policy (``frozen/velocity/policy.pt``,
``mdp.FrozenVelocityAction``); only this policy learns.

    command      goal (x, y, heading) drawn around the car, +-1 m, any heading, 10 % "stay here" (``mdp/navigation.py``)
    observation  IMU pitch and rate, speed, turn rate, goal in the car's frame (x, y, heading), last action (9)
    action       (v_x, w_z) in [-1, 1] x the ranges the velocity task was trained on
    reward       Isaac Lab's 1 - tanh(distance / std) with a coarse and a fine std, the heading error near the goal,
                 upright, command change, alive / fall

Unlike Isaac Lab (5 Hz over a 50 Hz policy) both layers run at 50 Hz: the IMU filter of the car runs once per step and
is shared by the two observations. Everything else is ``car_env_cfg.py``; details: ``guide/04_training.md``.
"""

import math

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.navigation.mdp as nav_mdp

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM
from Balance_Car_RL.car.rl_control.car_env_cfg import CarEnvCfg, ObservationsCfg, PolicyCfg, RewardsCfg
from Balance_Car_RL.car.rl_control.velocity_env_cfg import (
    SPEED_MAX_M_S,
    TURN_SHARE,
    YAW_RATE_MAX_RAD_S,
    VelocityEventCfg,
)

GOAL_RANGE_M = 1.0
"""A new goal is up to this far from the car along x and along y [m]: 1.4 m at most, 3.5 s at the top speed."""

GOAL_OBS_MAX_M = 2.0
"""The observed goal is shortened to this distance [m] (``mdp.pose_command_2d``)."""


@configclass
class PositionCommandsCfg:
    pose_command = mdp.CarPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(5.0, 8.0),
        hold_probability=0.1,
        debug_vis=True,
        position_success_threshold=0.05,
        ranges=mdp.CarPose2dCommandCfg.Ranges(
            pos_x=(-GOAL_RANGE_M, GOAL_RANGE_M), pos_y=(-GOAL_RANGE_M, GOAL_RANGE_M), heading=(-math.pi, math.pi)
        ),
    )

    def __post_init__(self):
        self.pose_command.goal_pose_visualizer_cfg.markers["arrow"].scale = (0.05, 0.05, 0.2)


FROZEN_VELOCITY = {
    "asset_name": "robot",
    "stage": "velocity",
    "speed_scale": SPEED_MAX_M_S,
    "yaw_rate_scale": YAW_RATE_MAX_RAD_S,
    "torque_scale": WHEEL_STALL_TORQUE_NM,
    "turn_share": TURN_SHARE,
}
"""The frozen velocity network under the position layer, as the velocity task was trained (also the position gain
task's)."""


@configclass
class PositionActionsCfg:
    """Two numbers in [-1, 1]: the velocity command as a share of the ranges the frozen velocity task was trained on."""

    velocity_command = mdp.FrozenVelocityActionCfg(**FROZEN_VELOCITY)


@configclass
class PositionPolicyCfg(PolicyCfg):
    """IMU pitch and rate, forward speed, turn rate, goal (x, y, heading) in the car's frame, last action (v, w): 9."""

    speed = ObsTerm(func=mdp.wheel_speed_estimate, params={"asset_cfg": mdp.WHEELS})
    yaw_rate = ObsTerm(func=mdp.gyro_yaw_rate)
    pose_command = ObsTerm(
        func=mdp.pose_command_2d, params={"command_name": "pose_command", "max_distance": GOAL_OBS_MAX_M}
    )
    last_action = ObsTerm(func=isaac_mdp.last_action)


@configclass
class PositionRewardsCfg(RewardsCfg):
    position_coarse = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={"command_name": "pose_command", "std": 1.0},
    )
    position_fine = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={"command_name": "pose_command", "std": 0.1},
    )
    heading = RewTerm(
        func=mdp.heading_error_near_goal,
        weight=-0.5,
        params={"command_name": "pose_command", "std": 0.2},
    )
    upright = RewTerm(
        func=mdp.upright_exp,
        weight=0.5,
        params={"std": 0.3},
    )
    output_change = RewTerm(
        func=isaac_mdp.action_rate_l2,  # at -0.05 the turn command chattered +-1-2 rad/s at the goal
        weight=-0.2,
    )


@configclass
class PositionEnvCfg(CarEnvCfg):
    commands: PositionCommandsCfg = PositionCommandsCfg()
    actions: PositionActionsCfg = PositionActionsCfg()
    observations: ObservationsCfg = ObservationsCfg(policy=PositionPolicyCfg())
    events: VelocityEventCfg = VelocityEventCfg()  # the robot the frozen velocity policy was trained on
    rewards: PositionRewardsCfg = PositionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.yaw_rate = None  # turning toward the goal is the job
